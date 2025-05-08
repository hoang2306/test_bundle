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


class DiffusionEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        
        # Linear noise schedule
        self.register_buffer('beta', torch.linspace(beta_start, beta_end, num_steps))
        self.register_buffer('alpha', 1 - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        
        # Denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 4),
            nn.LayerNorm(embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Initialize weights
        for m in self.denoise_net.modules():
            if isinstance(m, nn.Linear):
                init(m)
    
    def forward_diffusion(self, x_0, t):
        # Add noise to embeddings
        noise = torch.randn_like(x_0)
        alpha_t = self.alpha_bar[t].view(-1, 1).to(x_0.device)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        return x_t, noise
    
    def reverse_diffusion(self, x_t, t):
        # Denoise embeddings
        t_emb = t.float() / self.num_steps
        t_emb = t_emb.view(-1, 1).repeat(1, self.embedding_dim)
        x_input = torch.cat([x_t, t_emb], dim=-1)
        noise_pred = self.denoise_net(x_input)
        return noise_pred
    
    def sample(self, x_0, num_steps=None):
        if num_steps is None:
            num_steps = self.num_steps
        
        x_t = x_0
        for t in range(num_steps-1, -1, -1):
            t_batch = torch.full((x_t.shape[0],), t, device=x_t.device)
            noise_pred = self.reverse_diffusion(x_t, t_batch)
            
            alpha_t = self.alpha_bar[t].to(x_t.device)
            alpha_t_prev = self.alpha_bar[t-1].to(x_t.device) if t > 0 else torch.tensor(1.0, device=x_t.device)
            
            beta_t = 1 - alpha_t
            beta_t_prev = 1 - alpha_t_prev
            
            # Update step
            mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_t)) * noise_pred)
            if t > 0:
                noise = torch.randn_like(x_t)
                var = beta_t_prev
                x_t = mean + torch.sqrt(var) * noise
            else:
                x_t = mean
        
        return x_t


class HierachicalEncoder(nn.Module):
    def __init__(self, conf, raw_graph, features):
        super(HierachicalEncoder, self).__init__()
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

        # encoders for media feature
        self.c_encoder = dense(self.content_feature)
        self.t_encoder = dense(self.text_feature)

        self.multimodal_feature_dim = self.embedding_size
        # MM <<<

        # BI >>>
        self.item_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_item, self.embedding_size)
        )
        init(self.item_embeddings)
        self.multimodal_feature_dim += self.embedding_size
        # BI <<<

        # UI >>>
        self.cf_transformation = nn.Linear(self.embedding_size, self.embedding_size)
        init(self.cf_transformation)
        items_in_cf = self.ui_graph.sum(axis=0, dtype=bool)
        self.warm_indices_cf = torch.LongTensor(np.argwhere(items_in_cf)[:, 1]).to(device)
        self.cold_indices_cf = torch.LongTensor(np.argwhere(~items_in_cf)[:, 1]).to(device)
        self.multimodal_feature_dim += self.embedding_size
        # UI <<<

        # Multimodal Fusion:
        self.w_q = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        init(self.w_q)
        self.w_k = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        init(self.w_k)
        self.w_v = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_size, elementwise_affine=False)

        # adaptive weight modality fusion 
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

        # bi LightGCN
        self.bundle_embeddings = nn.Parameter(
            torch.FloatTensor(self.num_bundle, self.embedding_size)
        )
        self.num_layers = 1

        # Add diffusion embedding enhancement
        self.diffusion_enhancer = DiffusionEmbedding(
            embedding_dim=self.embedding_size,
            num_steps=conf.get('diffusion_steps', 1000),
            beta_start=conf.get('beta_start', 1e-4),
            beta_end=conf.get('beta_end', 0.02)
        )

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
        # [bs, #modality, #modality]
        attn = q.mul(self.embedding_size ** -0.5) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        features = attn @ v  # [bs, #modality, d]
        # average pooling
        y = features.mean(dim=-2)  # [bs, d]

        return y

    def get_propation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device 
        propagation_graph = sp.bmat(
            [[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]]
        )

        return to_tensor(
            laplace_transform(propagation_graph)
        ).to(device)
        

    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat([A_feature, B_feature], 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(
            all_features, 
            split_size_or_sections=(A_feature.shape[0], B_feature.shape[0]),
            dim=0
        )

        return A_feature, B_feature

    def forward_all(self):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        modal_weight = self.softmax(self.modal_weight)
        mm_feature_full = modal_weight[0] * c_feature + modal_weight[1] * t_feature

        features = [mm_feature_full]
        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        features.append(cf_feature_full)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]

        # multimodal fusion >>>
        final_feature = self.selfAttention(F.normalize(features, dim=-1))
        
        # Apply diffusion enhancement
        if self.training:
            # During training, apply forward diffusion and denoising
            t = torch.randint(0, self.diffusion_enhancer.num_steps, (final_feature.shape[0],), device=final_feature.device)
            noisy_feature, noise = self.diffusion_enhancer.forward_diffusion(final_feature, t)
            noise_pred = self.diffusion_enhancer.reverse_diffusion(noisy_feature, t)
            final_feature = final_feature + noise_pred
        else:
            # During inference, sample from diffusion process
            final_feature = self.diffusion_enhancer.sample(final_feature)
        
        return final_feature

    def forward(self, seq_modify, all=False):
        if all is True:
            return self.forward_all()

        modify_mask = (seq_modify == self.num_item)
        seq_modify.masked_fill_(modify_mask, 0)

        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        modal_weight = self.softmax(self.modal_weight)
        mm_feature_full = modal_weight[0] * c_feature + modal_weight[1] * t_feature
        mm_feature = mm_feature_full[seq_modify]  # [bs, n_token, d]

        features = [mm_feature]
        bi_feature_full = self.item_embeddings
        bi_feature = bi_feature_full[seq_modify]
        features.append(bi_feature)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        cf_feature = cf_feature_full[seq_modify]
        features.append(cf_feature)

        features = torch.stack(features, dim=-2)  # [bs, n_token, #modality, d]
        bs, n_token, N_modal, d = features.shape

        # multimodal fusion >>>
        final_feature = self.selfAttention(
            F.normalize(features.view(-1, N_modal, d), dim=-1)
        )
        final_feature = final_feature.view(bs, n_token, d)
        
        # Apply diffusion enhancement
        if self.training:
            # During training, apply forward diffusion and denoising
            t = torch.randint(0, self.diffusion_enhancer.num_steps, (final_feature.shape[0],), device=final_feature.device)
            noisy_feature, noise = self.diffusion_enhancer.forward_diffusion(final_feature, t)
            noise_pred = self.diffusion_enhancer.reverse_diffusion(noisy_feature, t)
            final_feature = final_feature + noise_pred
        else:
            # During inference, sample from diffusion process
            final_feature = self.diffusion_enhancer.sample(final_feature)
        
        return final_feature

    def generate_two_subs(self, dropout_ratio=0):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        # early-fusion
        modal_weight = self.softmax(self.modal_weight)
        mm_feature_full = modal_weight[0] * c_feature + modal_weight[1] * t_feature
        features = [mm_feature_full]

        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        features.append(cf_feature_full)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]
        size = features.shape[:2]  # (bs, #modality)

        def random_mask():
            random_tensor = torch.rand(size).to(features.device)
            mask_bool = random_tensor < dropout_ratio  # the remainders are true
            masked_feat = features.masked_fill(mask_bool.unsqueeze(-1), 0)

            # multimodal fusion >>>
            final_feature = self.selfAttention(
                F.normalize(masked_feat, dim=-1))
            # multimodal fusion <<<
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

        self.encoder = HierachicalEncoder(conf, raw_graph, features)
        # decoder has the similar structure of the encoder
        self.decoder = HierachicalEncoder(conf, raw_graph, features)

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

        # Add diffusion-specific parameters
        self.diffusion_loss_weight = conf.get('diffusion_loss_weight', 0.1)

    def forward(self, batch):
        idx, full, seq_full, modify, seq_modify = batch  # x: [bs, #items]
        mask = seq_full == self.num_item
        feat_bundle_view = self.encoder(seq_full)  # [bs, n_token, d]

        # bundle feature construction >>>
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)
        # print(f'bundle feature shape: {bundle_feature.shape}')

        if self.conf['use_bundle_id']:
            bundle_feature = bundle_feature + F.normalize(self.bundle_embeddings[idx])
        # print(f'bundle feature shape: {bundle_feature.shape}')

        feat_retrival_view = self.decoder(batch, all=True)

        # compute loss >>>
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)
        loss = recon_loss_function(logits, full)  # main_loss

        # # item-level contrastive learning >>>
        items_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()
        item_loss = torch.tensor(0).to(self.device)
        if self.cl_alpha > 0:
            if self.item_augmentation == "FD":
                item_features = self.encoder(batch, all=True)[items_in_batch]
                sub1 = self.cl_projector(self.dropout(item_features))
                sub2 = self.cl_projector(self.dropout(item_features))
                item_loss = self.cl_alpha * cl_loss_function(
                    sub1.view(-1, self.embedding_size), 
                    sub2.view(-1, self.embedding_size), 
                    self.cl_temp
                )
            elif self.item_augmentation == "NA":
                item_features = self.encoder(batch, all=True)[items_in_batch]
                item_loss = self.cl_alpha * cl_loss_function(
                    item_features.view(-1, self.embedding_size), 
                    item_features.view(-1, self.embedding_size), 
                    self.cl_temp
                )
            elif self.item_augmentation == "FN":
                item_features = self.encoder(batch, all=True)[items_in_batch]
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

        # Add diffusion loss if in training mode
        if self.training:
            diffusion_loss = F.mse_loss(
                self.encoder.diffusion_enhancer.reverse_diffusion(feat_bundle_view, torch.zeros(feat_bundle_view.shape[0], device=feat_bundle_view.device)),
                feat_bundle_view
            )
            loss = loss + self.diffusion_loss_weight * diffusion_loss

        return {
            'loss': loss + item_loss + bundle_loss,
            'item_loss': item_loss.detach(),
            'bundle_loss': bundle_loss.detach()
        }

    def evaluate(self, _, batch):
        idx, x, seq_x = batch
        mask = seq_x == self.num_item
        feat_bundle_view = self.encoder(seq_x)

        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        feat_retrival_view = self.decoder(
            (idx, x, seq_x, None, None), 
            all=True
        )

        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)

        return logits

    def propagate(self, test=False):
        return None
