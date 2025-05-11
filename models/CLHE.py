import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import (
    TransformerEncoder, 
    to_tensor
)
from collections import OrderedDict
import scipy.sparse as sp

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

def convert_csrmatrix_to_sparsetensor(csr_matrix):
    coo = csr_matrix.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.int64)
    values = torch.tensor(coo.data, dtype=torch.float32)
    shape = coo.shape
    # print(f'shape convert tensor: {shape}')
    # print(f'indices covert tensor: {indices}')
    sparse_tensor = torch.sparse_coo_tensor(
        indices, 
        values, 
        torch.Size(shape)
    )
    return sparse_tensor

def get_hyper_deg(incidence_matrix):
    '''
    # incidence_matrix = [num_nodes, num_hyperedges]
    hyper_deg = np.array(incidence_matrix.sum(axis=axis)).squeeze()
    hyper_deg[hyper_deg == 0.] = 1
    hyper_deg = sp.diags(1.0 / hyper_deg)
    '''

    # H  = [num_node, num_edge]
    # DV = [num_node, num_node]
    # DV * H = [num_node, num_edge]

    # HT = [num_edge, num_node]
    # DE = [num_edge, num_edge]
    # DE * HT = [num_edge, num_node]

    # hyper_deg = incidence_matrix.sum(1)
    # inv_hyper_deg = hyper_deg.power(-1)
    # inv_hyper_deg_diag = sp.diags(inv_hyper_deg.toarray()[0])

    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv

class hyper_graph_conv_layer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, item_emb, bi_graph_seen, ib_graph_seen):
        # print(f'device item_emb: {item_emb.device}') # cuda
        # print(f'device bi_graph: {bi_graph_seen.device}') # cuda
        message_item_agg = torch.sparse.mm(bi_graph_seen, item_emb) # [n_bundles, dim]
        propagate_item_emb = torch.sparse.mm(ib_graph_seen, message_item_agg) # [n_items, dim]

        return propagate_item_emb

class hyper_graph_conv_net(nn.Module):
    def __init__(self, num_layer, device, bi_graph_seen):
        super().__init__()
        self.num_layer = num_layer
        self.device = device
        self.hyper_graph_layer = hyper_graph_conv_layer()
        # self.bi_graph_seen_sparse_tensor = convert_csrmatrix_to_sparsetensor(bi_graph_seen).to(device)
        # bi graph
        self.deg_bi = get_hyper_deg(bi_graph_seen)
        self.bi_graph = self.deg_bi * bi_graph_seen
        self.bi_graph_seen_sparse_tensor = convert_csrmatrix_to_sparsetensor(self.bi_graph).to(device)

        # ib graph
        ib_graph_seen = bi_graph_seen.T 
        self.deg_ib = get_hyper_deg(ib_graph_seen)
        self.ib_graph = self.deg_ib * ib_graph_seen
        self.ib_graph_seen_sparse_tensor = convert_csrmatrix_to_sparsetensor(self.ib_graph).to(device)

    def forward(self, item_emb):
        features = [item_emb]
        for layer in range(self.num_layer):
            item_emb = self.hyper_graph_layer(
                item_emb, 
                self.bi_graph_seen_sparse_tensor, 
                self.ib_graph_seen_sparse_tensor
            )
            features.append(item_emb)
        features = F.normalize(torch.mean(torch.stack(features, dim=0), dim=0))
        return features


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
        # print(f'bi_graph_seen: {self.bi_graph_seen}')
        # self.bi_graph_seen_sparse_tensor = convert_csrmatrix_to_sparsetensor(self.bi_graph_seen)
        # print(f'bi sparse tensor: {self.bi_graph_seen_sparse_tensor}')

        self.attention_components = self.conf["attention"]

        self.content_feature, self.text_feature, self.cf_feature = features

        items_in_train = self.bi_graph_train.sum(axis=0, dtype=bool)
        self.warm_indices = torch.LongTensor(np.argwhere(items_in_train)[:, 1]).to(device)
        self.cold_indices = torch.LongTensor(np.argwhere(~items_in_train)[:, 1]).to(device)

        # MM >>>
        self.content_feature = F.normalize(self.content_feature, dim=-1)
        self.text_feature = F.normalize(self.text_feature, dim=-1)
        # build sim graph 
        self.mm_adj_weight = 0.5
        self.knn_k = 10
        self.alpha_sim_graph= 0.1
        self.num_layer_modal_graph = 2
        self.item_emb_modal = nn.Parameter(
            torch.FloatTensor(self.num_item, self.embedding_size)
        )
        
        if conf['use_modal_sim_graph']:
            print('starting build sim graph of image')
            indices, image_adj = self.get_knn_adj_mat(self.content_feature)
            print(f'starting build sim graph of text')
            indices, text_adj = self.get_knn_adj_mat(self.text_feature)
            self.mm_adj = self.mm_adj_weight*image_adj + (1-self.mm_adj_weight)*text_adj
            # self.mm_adj  = torch.cat([image_adj, text_adj], dim=1)
            # self.mm_adj = self.mm_adj.cpu() # move to cpu to reduce GPU resource
            # self.mm_adj = torch.sparse.mm(self.mm_adj, self.mm_adj.T)

            print(f'shape of mm_adj: {self.mm_adj.shape}')
            del text_adj 
            del image_adj
        else:
            print('not use modality sim graph')

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

        # hypergraph net
        self.item_hyper_emb = nn.Parameter(torch.FloatTensor(self.num_item, self.embedding_size))
        self.hyper_graph_conv_net = hyper_graph_conv_net(
            num_layer=1, 
            device=self.device, 
            bi_graph_seen=self.bi_graph_seen
        )

        self.get_bundle_agg_graph_ori(self.bi_graph_seen)


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

    def get_bundle_agg_graph_ori(self, graph):
        bi_graph = graph
        device = self.device
        bundle_size = bi_graph.sum(axis=1) + 1e-8 # calculate size for each bundle 
        # print(f"bundle size: {bundle_size.shape}")
        # print(f"diag bundle: {sp.diags(1/bundle_size.A.ravel()).shape}")
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph # sp.diags(1/bundle_size.A.ravel()): D^-1 
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device) 

    # def get_knn_adj_mat(self, mm_embeddings):
    #     context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
    #     sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    #     _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
    #     adj_size = sim.size()
    #     del sim
    #     # construct sparse adj
    #     indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
    #     indices0 = torch.unsqueeze(indices0, 1)
    #     indices0 = indices0.expand(-1, self.knn_k)
    #     indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
    #     # norm
    #     return indices, self.compute_normalized_laplacian(indices, adj_size)

    def get_knn_adj_mat(self, mm_embeddings, batch_size=1024):
        with torch.no_grad():  # tránh giữ lại graph nếu không cần backward
            device = self.device
            N = mm_embeddings.size(0)
            context_norm = mm_embeddings / mm_embeddings.norm(p=2, dim=-1, keepdim=True)

            knn_indices = torch.empty((N, self.knn_k), dtype=torch.long, device=device)

            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                batch = context_norm[i:end_i]  # (B, D)
                sim_batch = torch.matmul(batch, context_norm.transpose(0, 1))  # (B, N)
                _, topk = torch.topk(sim_batch, self.knn_k, dim=-1)
                knn_indices[i:end_i] = topk

            adj_size = (N, N)

            indices0 = torch.arange(N, device=device).unsqueeze(1).expand(-1, self.knn_k)
            indices = torch.stack((indices0.flatten(), knn_indices.flatten()), dim=0)

            return indices, self.compute_normalized_laplacian(indices, adj_size)


    def compute_normalized_laplacian(self, indices, adj_size):
        # adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def forward_all(self):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        modal_weight = self.softmax(self.modal_weight)
        # mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        mm_feature_full = modal_weight[0] * F.normalize(c_feature) + modal_weight[1] * F.normalize(t_feature)
        
        features = []
        features.append(mm_feature_full)
        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        features.append(cf_feature_full)

        if self.conf['use_modal_sim_graph']:
            h = self.item_emb_modal
            for i in range(self.num_layer_modal_graph):
                h = torch.sparse.mm(self.mm_adj, h)
            features.append(h)

        # hypergraph net 
        # if self.conf['use_hyper_graph']:
        #     item_hyper_emb = self.hyper_graph_conv_net(
        #         self.item_hyper_emb
        #     )
        #     features.append(item_hyper_emb)

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]

        # multimodal fusion >>>
        final_feature = self.selfAttention(F.normalize(features, dim=-1))
        # print(
        #     f'shape of final feature in forward_all: {final_feature.shape}'
        # ) # [48676, 64] ~ [n_items, dim]

        # graph propagation with mm_adj graph
        # here: i use 1 layer for graph 
        # if self.conf['use_modal_sim_graph']:
        #     h = self.item_emb_modal
        #     for i in range(1):
        #         h = torch.sparse.mm(self.mm_adj, h)
        #     final_feature = final_feature + self.alpha_sim_graph * F.normalize(h)

        item_hyper_emb = self.hyper_graph_conv_net(
            self.item_hyper_emb
        )
        # final_feature = final_feature + item_hyper_emb

        # multimodal fusion <<<

        return final_feature, item_hyper_emb

    def forward(self, seq_modify, all=False):
        if all is True:
            return self.forward_all()

        # modify_mask = seq_modify == self.num_item
        # seq_modify.masked_fill_(modify_mask, 0)

        # c_feature = self.c_encoder(self.content_feature)
        # t_feature = self.t_encoder(self.text_feature)

        # modal_weight = self.softmax(self.modal_weight)
        # # mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        # mm_feature_full = modal_weight[0]*F.normalize(c_feature) + modal_weight[1]*F.normalize(t_feature)
        # mm_feature = mm_feature_full[seq_modify]  # [bs, n_token, d]

        # features = [mm_feature]
        # bi_feature_full = self.item_embeddings
        # bi_feature = bi_feature_full[seq_modify]
        # features.append(bi_feature)

        # cf_feature_full = self.cf_transformation(self.cf_feature)
        # cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        # cf_feature = cf_feature_full[seq_modify]
        # features.append(cf_feature)

        # features = torch.stack(features, dim=-2)  # [bs, n_token, #modality, d]
        # bs, n_token, N_modal, d = features.shape

        # # multimodal fusion >>>
        # final_feature = self.selfAttention(
        #     F.normalize(features.view(-1, N_modal, d), dim=-1))

        # # print(f'shape of final feature: {final_feature.shape}') # [1280, 64]

        # final_feature = final_feature.view(bs, n_token, d)
        # # multimodal fusion <<<

        modify_mask = seq_modify == self.num_item
        seq_modify.masked_fill_(modify_mask, 0)

        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        modal_weight = self.softmax(self.modal_weight)
        # mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        mm_feature_full = modal_weight[0]*F.normalize(c_feature) + modal_weight[1]*F.normalize(t_feature)
        # mm_feature = mm_feature_full[seq_modify]  # [bs, n_token, d]

        features = []
        features.append(mm_feature_full)
        bi_feature_full = self.item_embeddings
        # bi_feature = bi_feature_full[seq_modify]
        features.append(bi_feature_full)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        # cf_feature = cf_feature_full[seq_modify]
        features.append(cf_feature_full)

        if self.conf['use_modal_sim_graph']:
            h = self.item_emb_modal
            for i in range(self.num_layer_modal_graph):
                h = torch.sparse.mm(self.mm_adj, h)
            features.append(h)

        # if self.conf['use_hyper_graph']:
        #     item_hyper_emb = self.hyper_graph_conv_net(
        #         self.item_hyper_emb
        #     )
        #     features.append(item_hyper_emb)
        

        features = torch.stack(features, dim=-2)  # [n_items, n_modal, dim]
        # print(f'shape of features in forward: {features.shape}')

        # multimodal fusion >>>
        final_feature = self.selfAttention(F.normalize(features, dim=-1)) # [n_items, dim]
        # print(f'shape of final feature in forward: {final_feature.shape}')

        # graph propagation
        # if self.conf['use_modal_sim_graph']:
        #     h = self.item_emb_modal
        #     for i in range(1):
        #         h = torch.sparse.mm(self.mm_adj, h)
        #     final_feature = final_feature + self.alpha_sim_graph * F.normalize(h)

        item_hyper_emb = self.hyper_graph_conv_net(
            self.item_hyper_emb
        )
        # final_feature = final_feature + item_hyper_emb
        bundle_hyper_emb = self.bundle_agg_graph_ori @ item_hyper_emb

        final_feature = final_feature[seq_modify] # [bs, n_token, d]
        # print(f'shape of final feature in forward: {final_feature.shape}')

        
        bs, n_token, d = final_feature.shape
        # final_feature = self.selfAttention(F.normalize(features.view(-1, N_modal, d), dim=-1))

        # print(f'shape of final feature: {final_feature.shape}') # [1280, 64]

        final_feature = final_feature.view(bs, n_token, d)
        # multimodal fusion <<<

        return final_feature, bundle_hyper_emb

    def generate_two_subs(self, dropout_ratio=0):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        # early-fusion
        modal_weight = self.softmax(self.modal_weight)
        # mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        mm_feature_full = modal_weight[0]*F.normalize(c_feature) + modal_weight[1]*F.normalize(t_feature)

        features = []
        features.append(mm_feature_full)
        features.append(self.item_embeddings)

        cf_feature_full = self.cf_transformation(self.cf_feature)
        cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        features.append(cf_feature_full)

        # if self.conf['use_modal_sim_graph']:
        #     h = self.item_emb_modal
        #     for i in range(self.num_layer_modal_graph):
        #         h = torch.sparse.mm(self.mm_adj, h)
        #     features.append(h)
        
        # if self.conf['use_hyper_graph']:
        #     item_hyper_emb = self.hyper_graph_conv_net(
        #         self.item_hyper_emb
        #     )
        #     features.append(item_hyper_emb)

        

        features = torch.stack(features, dim=-2)  # [bs, #modality, d]
        size = features.shape[:2]  # (bs, #modality)

        def random_mask():
            random_tensor = torch.rand(size).to(features.device)
            mask_bool = random_tensor < dropout_ratio  # the remainders are true
            masked_feat = features.masked_fill(mask_bool.unsqueeze(-1), 0)

            # multimodal fusion >>>
            final_feature = self.selfAttention(
                F.normalize(masked_feat, dim=-1))
            # if self.conf['use_modal_sim_graph']:
            #     h = self.item_emb_modal
            #     for i in range(1):
            #         h = torch.sparse.mm(self.mm_adj, h)
            #     final_feature = final_feature + self.alpha_sim_graph * F.normalize(h)

            item_hyper_emb = self.hyper_graph_conv_net(
                self.item_hyper_emb
            )
            final_feature = final_feature + item_hyper_emb
            
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

    def forward(self, batch):
        idx, full, seq_full, modify, seq_modify = batch  # x: [bs, #items]
        mask = seq_full == self.num_item
        feat_bundle_view = self.encoder(seq_full)  # [bs, n_token, d]

        # bundle feature construction >>>
        bundle_feature, bundle_hyper_emb = self.bundle_encode(feat_bundle_view, mask=mask)

        feat_retrival_view, item_hyper_emb = self.decoder(batch, all=True)

        # compute loss >>>
        bundle_feature = bundle_feature + bundle_hyper_emb[idx]
        feat_retrival_view = feat_retrival_view + item_hyper_emb
        logits = bundle_feature @ feat_retrival_view.transpose(0, 1)
        loss = recon_loss_function(logits, full)  # main_loss

        # # item-level contrastive learning >>>
        # items_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()
        # item_loss = torch.tensor(0).to(self.device)
        # if self.cl_alpha > 0:
        #     if self.item_augmentation == "FD":
        #         item_features = self.encoder(batch, all=True)[items_in_batch]
        #         sub1 = self.cl_projector(self.dropout(item_features))
        #         sub2 = self.cl_projector(self.dropout(item_features))
        #         item_loss = self.cl_alpha * cl_loss_function(
        #             sub1.view(-1, self.embedding_size), 
        #             sub2.view(-1, self.embedding_size), 
        #             self.cl_temp
        #         )
        #     elif self.item_augmentation == "NA":
        #         item_features = self.encoder(batch, all=True)[items_in_batch]
        #         item_loss = self.cl_alpha * cl_loss_function(
        #             item_features.view(-1, self.embedding_size), 
        #             item_features.view(-1, self.embedding_size), 
        #             self.cl_temp
        #         )
        #     elif self.item_augmentation == "FN":
        #         item_features = self.encoder(batch, all=True)[items_in_batch]
        #         sub1 = self.cl_projector(
        #             self.noise_weight * torch.randn_like(item_features) + item_features
        #         )
        #         sub2 = self.cl_projector(
        #             self.noise_weight * torch.randn_like(item_features) + item_features
        #         )
        #         item_loss = self.cl_alpha * cl_loss_function(
        #             sub1.view(-1, self.embedding_size), 
        #             sub2.view(-1, self.embedding_size), 
        #             self.cl_temp
        #         )
        #     elif self.item_augmentation == "MD":
        #         sub1, sub2 = self.encoder.generate_two_subs(self.dropout_rate)
        #         sub1 = sub1[items_in_batch]
        #         sub2 = sub2[items_in_batch]
        #         item_loss = self.cl_alpha * cl_loss_function(
        #             sub1.view(-1, self.embedding_size), 
        #             sub2.view(-1, self.embedding_size), 
        #             self.cl_temp
        #         )
        # # # item-level contrastive learning <<<

        # # bundle-level contrastive learning >>>
        # bundle_loss = torch.tensor(0).to(self.device)
        # if self.bundle_cl_alpha > 0:
        #     feat_bundle_view2 = self.encoder(seq_modify)  # [bs, n_token, d]
        #     bundle_feature2 = self.bundle_encode(feat_bundle_view2, mask=mask)
        #     bundle_loss = self.bundle_cl_alpha * cl_loss_function(
        #         bundle_feature.view(-1, self.embedding_size), 
        #         bundle_feature2.view(-1, self.embedding_size), 
        #         self.bundle_cl_temp
        #     )
        # bundle-level contrastive learning <<<

        combine_loss = {
            'loss': loss,
            'item_loss': loss,
            'bundle_loss': loss
        }

        return combine_loss

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
