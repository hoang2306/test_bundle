import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import (
    TransformerEncoder,
    to_tensor,
    laplace_transform,
    np_edge_dropout
)
from collections import OrderedDict

eps = 1e-9 # avoid zero division

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)

def recon_loss_function(recon_x, x):
    negLogLike = torch.sum(F.log_softmax(recon_x, 1) * x, -1) / x.sum(dim=-1)
    negLogLike = -torch.mean(negLogLike)
    return negLogLike

infonce_criterion = nn.CrossEntropyLoss()

def cl_loss_function(a, b, temp=0.2):
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    logits = torch.mm(a, b.T)
    logits /= temp
    labels = torch.arange(a.shape[0]).to(a.device)
    return infonce_criterion(logits, labels)

def disentangle_loss(zs, margin=0.5):
    # zs: list of [N, d] (each is a modality-specific embedding)
    # Encourage orthogonality between different modality embeddings
    loss = 0
    n = len(zs)
    for i in range(n):
        for j in range(i+1, n):
            # cosine similarity between modalities
            sim = F.cosine_similarity(zs[i], zs[j], dim=-1)
            # want sim to be close to zero (orthogonal)
            loss += torch.mean(torch.clamp(sim.abs() - margin, min=0))
    return loss / (n * (n-1) / 2 + eps)

class DisentangledHierarchicalEncoder(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(DisentangledHierarchicalEncoder, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.attention_components = self.conf["attention"]

        self.content_feature, self.text_feature, self.cf_feature = features

        items_in_train = self.bi_graph_train.sum(axis=0, dtype=bool)
        self.warm_indices = torch.LongTensor(np.argwhere(items_in_train)[:, 1]).to(device)
        self.cold_indices = torch.LongTensor(np.argwhere(~items_in_train)[:, 1]).to(device)

        # MM >>>
        self.content_feature = F.normalize(self.content_feature, dim=-1)
        self.text_feature = F.normalize(self.text_feature, dim=-1)

        def dense(feature):
            module = nn.Sequential(OrderedDict([
                ('w1', nn.Linear(feature.shape[1], feature.shape[1])),
                ('act1', nn.ReLU()),
                ('w2', nn.Linear(feature.shape[1], 256)),
                ('act2', nn.ReLU()),
                ('w3', nn.Linear(256, 64)),
            ]))
            for m in module:
                init(m)
            return module

        # Disentangled encoders for each modality
        self.c_encoder = dense(self.content_feature)
        self.t_encoder = dense(self.text_feature)
        self.cf_encoder = nn.Linear(self.embedding_size, self.embedding_size)
        init(self.cf_encoder)

        # Learnable item id embedding (structure)
        self.item_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_item, self.embedding_size)
        )
        init(self.item_embeddings)

        # For fusion
        self.fusion_proj = nn.Linear(self.embedding_size * 4, self.embedding_size)
        init(self.fusion_proj)

        # For adaptive fusion weights
        self.modal_weight = nn.Parameter(torch.Tensor([0.25, 0.25, 0.25, 0.25]))
        self.softmax = nn.Softmax(dim=0)

        # For self-attention fusion (optional)
        self.w_q = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

        # bi LightGCN
        self.bundle_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_bundle, self.embedding_size)
        )
        self.num_layers = 1

    def selfAttention(self, features):
        # features: [bs, #modality, d]
        if "layernorm" in self.attention_components:
            features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        if "w_v" in self.attention_components:
            v = self.w_v(features)
        else:
            v = features
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        features = attn @ v  # [bs, #modality, d]
        y = features.mean(dim=-2)  # [bs, d]
        return y

    def get_modality_embeddings(self, indices=None):
        # Return disentangled embeddings for each modality
        c_emb = self.c_encoder(self.content_feature)
        t_emb = self.t_encoder(self.text_feature)
        cf_emb = self.cf_encoder(self.cf_feature)
        id_emb = self.item_embeddings

        if indices is not None:
            c_emb = c_emb[indices]
            t_emb = t_emb[indices]
            cf_emb = cf_emb[indices]
            id_emb = id_emb[indices]
        return c_emb, t_emb, cf_emb, id_emb

    def forward_all(self):
        # Get all disentangled embeddings
        c_emb, t_emb, cf_emb, id_emb = self.get_modality_embeddings()
        # Stack for fusion: [N, 4, d]
        features = torch.stack([c_emb, t_emb, cf_emb, id_emb], dim=1)
        # Adaptive fusion
        modal_weight = self.softmax(self.modal_weight)
        fused = (modal_weight[0] * c_emb +
                 modal_weight[1] * t_emb +
                 modal_weight[2] * cf_emb +
                 modal_weight[3] * id_emb)
        # Optionally, use self-attention fusion
        fused = self.selfAttention(F.normalize(features, dim=-1))
        return fused, [c_emb, t_emb, cf_emb, id_emb]

    def forward(self, seq_modify, all=False):
        if all is True:
            return self.forward_all()[0]

        modify_mask = (seq_modify == self.num_item)
        seq_modify.masked_fill_(modify_mask, 0)

        c_emb, t_emb, cf_emb, id_emb = self.get_modality_embeddings()
        c_emb = c_emb[seq_modify]  # [bs, n_token, d]
        t_emb = t_emb[seq_modify]
        cf_emb = cf_emb[seq_modify]
        id_emb = id_emb[seq_modify]

        features = torch.stack([c_emb, t_emb, cf_emb, id_emb], dim=-2)  # [bs, n_token, 4, d]
        bs, n_token, N_modal, d = features.shape

        # Adaptive fusion
        modal_weight = self.softmax(self.modal_weight)
        fused = (modal_weight[0] * c_emb +
                 modal_weight[1] * t_emb +
                 modal_weight[2] * cf_emb +
                 modal_weight[3] * id_emb)
        # Optionally, use self-attention fusion
        fused = self.selfAttention(F.normalize(features.view(-1, N_modal, d), dim=-1))
        fused = fused.view(bs, n_token, d)
        return fused

    def generate_two_subs(self, dropout_ratio=0):
        c_emb, t_emb, cf_emb, id_emb = self.get_modality_embeddings()
        features = torch.stack([c_emb, t_emb, cf_emb, id_emb], dim=1)  # [N, 4, d]
        size = features.shape[:2]  # (N, 4)

        def random_mask():
            random_tensor = torch.rand(size).to(features.device)
            mask_bool = random_tensor < dropout_ratio
            masked_feat = features.masked_fill(mask_bool.unsqueeze(-1), 0)
            final_feature = self.selfAttention(F.normalize(masked_feat, dim=-1))
            return final_feature

        return random_mask(), random_mask()

class CLHE(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.item_augmentation = self.conf["item_augment"]

        self.encoder = DisentangledHierarchicalEncoder(conf, raw_graph, features)
        self.decoder = DisentangledHierarchicalEncoder(conf, raw_graph, features)

        self.bundle_encode = TransformerEncoder(
            conf={
                "n_layer": conf["trans_layer"],
                "dim": 64,
                "num_token": 100,
                "device": self.device,
            }, 
            data={
                "sp_graph": self.bi_graph_seen
            }
        )

        self.cl_temp = conf['cl_temp']
        self.cl_alpha = conf['cl_alpha']

        self.bundle_cl_temp = conf['bundle_cl_temp']
        self.bundle_cl_alpha = conf['bundle_cl_alpha']
        self.cl_projector = nn.Linear(self.embedding_size, self.embedding_size)
        init(self.cl_projector)
        if self.item_augmentation in ["FD", "MD"]:
            self.dropout_rate = conf["dropout_rate"]
            self.dropout = nn.Dropout(p=self.dropout_rate)
        elif self.item_augmentation in ["FN"]:
            self.noise_weight = conf['noise_weight']

        # bundle id 
        self.bundle_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_bundle, self.embedding_size)
        )

        # disentangle loss weight
        self.disentangle_alpha = conf.get('disentangle_alpha', 1.0)
        self.disentangle_margin = conf.get('disentangle_margin', 0.5)

    def forward(self, batch):
        idx, full, seq_full, modify, seq_modify = batch  # x: [bs, #items]
        mask = seq_full == self.num_item
        # Get disentangled embeddings and fused
        feat_bundle_view, bundle_modalities = self.encoder.forward(seq_modify)  # [N, d], [c, t, cf, id]
            

        # bundle feature construction >>>
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        if self.conf.get('use_bundle_id', False):
            bundle_feature = bundle_feature + F.normalize(self.bundle_embeddings[idx])

        feat_retrival_view, retrival_modalities = self.decoder.forward_all()

        # compute loss >>>
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)
        loss = recon_loss_function(logits, full)  # main_loss

        # Disentangle loss for encoder and decoder
        disentangle_loss_enc = disentangle_loss(bundle_modalities, margin=self.disentangle_margin)
        disentangle_loss_dec = disentangle_loss(retrival_modalities, margin=self.disentangle_margin)
        disentangle_loss_total = self.disentangle_alpha * (disentangle_loss_enc + disentangle_loss_dec)

        # # item-level contrastive learning >>>
        items_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()
        item_loss = torch.tensor(0).to(self.device)
        if self.cl_alpha > 0:
            if self.item_augmentation == "FD":
                item_features, _ = self.encoder.forward_all()
                item_features = item_features[items_in_batch]
                sub1 = self.cl_projector(self.dropout(item_features))
                sub2 = self.cl_projector(self.dropout(item_features))
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), 
                    sub2.view(-1, self.embedding_size), 
                    self.cl_temp
                )
            elif self.item_augmentation == "NA":
                item_features, _ = self.encoder.forward_all()
                item_features = item_features[items_in_batch]
                item_loss = self.cl_alpha * cl_loss_function(
                    item_features.view(-1, self.embedding_size), 
                    item_features.view(-1, self.embedding_size), 
                    self.cl_temp
                )
            elif self.item_augmentation == "FN":
                item_features, _ = self.encoder.forward_all()
                item_features = item_features[items_in_batch]
                sub1 = self.cl_projector(
                    self.noise_weight * torch.randn_like(item_features) + item_features
                )
                sub2 = self.cl_projector(
                    self.noise_weight * torch.randn_like(item_features) + item_features
                )
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), 
                    sub2.view(-1, self.embedding_size), 
                    self.cl_temp
                )
            elif self.item_augmentation == "MD":
                sub1, sub2 = self.encoder.generate_two_subs(self.dropout_rate)
                sub1 = sub1[items_in_batch]
                sub2 = sub2[items_in_batch]
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), 
                    sub2.view(-1, self.embedding_size), 
                    self.cl_temp
                )
        # # item-level contrastive learning <<<

        # bundle-level contrastive learning >>>
        bundle_loss = torch.tensor(0).to(self.device)
        if self.bundle_cl_alpha > 0:
            feat_bundle_view2 = self.encoder(seq_modify)  # [bs, n_token, d]
            bundle_feature2 = self.bundle_encode(feat_bundle_view2, mask=mask)
            bundle_loss = self.bundle_cl_alpha * cl_loss_function(
                bundle_feature.view(-1, self.embedding_size), 
                bundle_feature2.view(-1, self.embedding_size), 
                self.bundle_cl_temp
            )
        # bundle-level contrastive learning <<<

        return {
            'loss': loss + item_loss + bundle_loss + disentangle_loss_total,
            'item_loss': item_loss.detach(),
            'bundle_loss': bundle_loss.detach(),
            'disentangle_loss': disentangle_loss_total.detach()
        }

    def evaluate(self, _, batch):
        idx, x, seq_x = batch
        mask = seq_x == self.num_item
        feat_bundle_view, _ = self.encoder.forward_all()
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)
        feat_retrival_view, _ = self.decoder.forward_all()
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)
        return logits

    def propagate(self, test=False):
        return None
