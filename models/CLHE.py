import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import scipy.sparse as sp
import os 
from utility import slash
from models.pwc import PWC


from models.utils import (
    TransformerEncoder, 
    to_tensor, 
    convert_csrmatrix_to_sparsetensor,
    get_hyper_deg,
    init, 
    hyper_graph_conv_net, 
    hyper_graph_conv_layer,
    slash
)
from models.gat import (
    Amatrix, 
    AsymMatrix
)

from models.diffusion_process import (
    DiffusionProcess, 
    SDNet
)

from models.moe import MixtureOfExperts
from models.cross_attention import CrossAttentionFusion

eps = 1e-9 # avoid zero division

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

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(2*dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        x = torch.cat([x1,x2], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x 

class MLP_pwc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(3*dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x 

class HierachicalEncoder(nn.Module):
    def __init__(self, conf, raw_graph, features, cate):
        super(HierachicalEncoder, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.num_user = self.conf["num_users"]
        self.num_bundle = self.conf["num_bundles"]
        self.num_item = self.conf["num_items"]
        self.embedding_size = 64
        self.ui_graph, self.bi_graph_train, self.bi_graph_seen = raw_graph
        self.cate = cate 
        # print(f'bi_graph_seen: {self.bi_graph_seen}')
        # self.bi_graph_seen_sparse_tensor = convert_csrmatrix_to_sparsetensor(self.bi_graph_seen)
        # print(f'bi sparse tensor: {self.bi_graph_seen_sparse_tensor}')

        self.attention_components = self.conf["attention"]

        self.content_feature, self.text_feature, self.cf_feature = features
        # pog 
        # content feature dim: 768
        # text feature dim: 768 

        items_in_train = self.bi_graph_train.sum(axis=0, dtype=bool)
        self.warm_indices = torch.LongTensor(np.argwhere(items_in_train)[:, 1]).to(device)
        self.cold_indices = torch.LongTensor(np.argwhere(~items_in_train)[:, 1]).to(device)

        # cate embedding
        # self.cate_emb = nn.Parameter(
        #     torch.FloatTensor(len(self.cate), self.embedding_size)
        # )
        # init(self.cate_emb)

        # MM >>>
        self.content_feature = F.normalize(self.content_feature, dim=-1)
        self.text_feature = F.normalize(self.text_feature, dim=-1)
        # build sim graph 
        self.mm_adj_weight = 0.5
        self.knn_k = conf['knn_k']
        self.alpha_sim_graph= 0.1
        self.num_layer_modal_graph = 2
        
        self.num_layer_gat = conf["num_layer_gat"]
        if conf['use_modal_sim_graph']:
            print('use modal sim graph')
            self.item_emb_modal = nn.Parameter(
                torch.FloatTensor(self.num_item, self.embedding_size)
            )
            init(self.item_emb_modal)
            print('starting build sim graph of image')
            indices, image_adj = self.get_knn_adj_mat(self.content_feature)
            print(f'starting build sim graph of text')
            indices, text_adj = self.get_knn_adj_mat(self.text_feature)
            self.mm_adj = self.mm_adj_weight*image_adj + (1-self.mm_adj_weight)*text_adj
            # self.mm_adj  = torch.cat([image_adj, text_adj], dim=1)
            # self.mm_adj = self.mm_adj.cpu() # move to cpu to reduce GPU resource
            # self.mm_adj = torch.sparse.mm(self.mm_adj, self.mm_adj.T)

            _, cross_image_text_adj = self.get_cross_modal_knn_adj_mat(
                mm_embeddings_1=self.content_feature,
                mm_embeddings_2=self.text_feature,
                batch_size=1024
            )
            _, cross_text_image_adj = self.get_cross_modal_knn_adj_mat(
                mm_embeddings_1=self.text_feature,
                mm_embeddings_2=self.content_feature,
                batch_size=1024
            )  
            self.cross_mm_adj_weight = 0
            self.cross_mm_adj = self.cross_mm_adj_weight * cross_image_text_adj + (1-self.cross_mm_adj_weight) * cross_text_image_adj
            

            print(f'shape of mm_adj: {self.mm_adj.shape}')
            del text_adj 
            del image_adj
            print(f'shape of cross_mm_adj: {self.cross_mm_adj.shape}')
            del cross_image_text_adj
            del cross_text_image_adj

            # print(f'mm adj type: {type(self.mm_adj)}') # tensor
            # print(f'cross mm adj type: {type(self.cross_mm_adj)}') # tensor
            # save mm_adj and cross_mm_adj
            mm_adj_save_path = f'./datasets/{conf["dataset"]}/mm_adj.pt'
            if not os.path.exists(mm_adj_save_path):
                torch.save(self.mm_adj, mm_adj_save_path)
                print(f'saved mm_adj to {mm_adj_save_path}')
            else:
                print(f'mm_adj already exists at {mm_adj_save_path}')
            cross_mm_adj_save_path = f'./datasets/{conf["dataset"]}/cross_mm_adj.pt'
            if not os.path.exists(cross_mm_adj_save_path):
                torch.save(self.cross_mm_adj, cross_mm_adj_save_path)
                print(f'saved cross_mm_adj to {cross_mm_adj_save_path}')
            else:
                print(f'cross_mm_adj already exists at {cross_mm_adj_save_path}')

            self.ii_modal_sim_gat = Amatrix(
                in_dim=64,
                out_dim=64,
                n_layer=self.num_layer_gat,
                dropout=0.1,
                heads=2, 
                concat=False,
                self_loop=False,
                extra_layer=True,
                type_gnn=conf['type_gnn']
            )
            self.cross_modal_sim_gnn = Amatrix(
                in_dim=64,
                out_dim=64,
                n_layer=self.num_layer_gat,
                dropout=0.1,
                heads=2, 
                concat=False,
                self_loop=False,
                extra_layer=True,
                type_gnn=conf['type_gnn']
            )

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

        self.get_bundle_agg_graph_ori(self.bi_graph_seen)

        # adaptive weight modality fusion 
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

        self.layer_times = 2 
        self.dropout_rate = 0.2 
        self.BeFA_v = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * self.layer_times),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Tanh(),
            nn.Linear(self.embedding_size * self.layer_times, self.embedding_size * self.layer_times),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embedding_size * self.layer_times, self.embedding_size),
            nn.Sigmoid()
        )

        self.BeFA_t = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size * self.layer_times),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Tanh(),
            nn.Linear(self.embedding_size * self.layer_times, self.embedding_size * self.layer_times),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embedding_size * self.layer_times, self.embedding_size),
            nn.Sigmoid()
        )


        # hypergraph net
        self.item_hyper_emb = nn.Parameter(torch.FloatTensor(self.num_item, self.embedding_size))
        init(self.item_hyper_emb)
        self.hyper_graph_conv_net = hyper_graph_conv_net(
            num_layer=conf['num_layer_hypergraph'], 
            device=self.device, 
            bi_graph_seen=self.bi_graph_seen
        )

        slash()
        # asymmetric gat 
        print(f'use iui conv: {conf["use_iui_conv"]}')
        if self.conf['use_iui_conv']:
            print(f'USE IUI CONV')
            self.iui_edge_index = torch.tensor(
                np.load(
                    f"./datasets/{conf['dataset']}/n_neigh_iui_{conf['gnn_knn']}.npy", 
                    allow_pickle=True
                )
            ).to(self.device)

            self.ibi_edge_index = torch.tensor(
                np.load(
                    f"./ii_data/{conf['dataset']}/n_neigh_ibi_2.npy" if self.conf['dataset'] == 'pog' else f"./datasets/{conf['dataset']}/n_neigh_ibi_2.npy", 
                    allow_pickle=True
                )
            ).to(self.device)

            self.iui_gat_conv = Amatrix(
                in_dim=64,
                out_dim=64,
                n_layer=self.num_layer_gat,
                dropout=0.1,
                heads=2, 
                concat=False,
                self_loop=False,
                extra_layer=True,
                type_gnn=conf['type_gnn']
            )

            self.light_gcn = Amatrix(
                in_dim=64,
                out_dim=64,
                n_layer=self.num_layer_gat,
                dropout=0.1,
                heads=2, 
                concat=False,
                self_loop=False, # lightgcn no need self loop
                extra_layer=True,
                type_gnn='light_gcn'
            )

            self.ibi_gat_conv = Amatrix(
                in_dim=64,
                out_dim=64,
                n_layer=self.num_layer_gat,
                dropout=0.1,
                heads=2, 
                concat=False,
                self_loop=False,
                extra_layer=True,
                type_gnn=conf['type_gnn']
            )

        self.item_gat_emb = nn.Parameter(torch.FloatTensor(self.num_item, self.embedding_size))
        init(self.item_gat_emb)
        

        # diffusion
        if self.conf['use_diffusion']:
            print('use diffusion model')
            self.diff_process = DiffusionProcess(
                noise_schedule=conf['noise_schedule'],
                noise_scale=conf['noise_scale'],
                noise_min=conf['noise_min'],
                noise_max=conf['noise_max'],
                steps=conf['steps'],
                device=self.device
            ).to(self.device)
            self.SDNet = SDNet(
                in_dims=[64,64],
                out_dims=[64,64],
                emb_size=16,
                time_type='cat',
                norm=True
            ).to(self.device)

        # mlp for fusion 
        self.mlp = MLP(dim=64)

        # mixture of experts
        self.moe_layer = MixtureOfExperts(
            text_dim=64,
            image_dim=64,
            hidden_dim=128,
            output_dim=64,
            num_experts=2
        )
        # print(self.moe_layer)

        # pwc 
        self.pwc_fusion = PWC(
            self.num_item,
            self.embedding_size,
            self.embedding_size // 4,
            self.device,
            base=0.9,
            w1=0.9 
        )
        self.mlp_pwc = MLP_pwc(dim=self.embedding_size)

        # cross attention fusion
        self.cross_attention = CrossAttentionFusion(embedding_dim=self.embedding_size)

        # self.attention
        self.attn_image = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.1, batch_first=True)
        self.attn_text = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.1, batch_first=True)
        self.cross_attn_image = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.1, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(embed_dim=64, num_heads=1, dropout=0.1, batch_first=True)

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
        eps = 1e-8
        bundle_size = bi_graph.sum(axis=1) + eps # calculate size for each bundle 
        # print(f"bundle size: {bundle_size.shape}")
        # print(f"diag bundle: {sp.diags(1/bundle_size.A.ravel()).shape}")
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph # sp.diags(1/bundle_size.A.ravel()): D^-1 
        # print(f'graph: {graph}')
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device) 

    def get_knn_adj_mat(self, mm_embeddings, batch_size=1024):
        with torch.no_grad():  
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

    def get_cross_modal_knn_adj_mat(self, mm_embeddings_1, mm_embeddings_2, batch_size=1024):
        print(f'calculating cross modal sim graph')
        with torch.no_grad():  
            device = self.device
            assert mm_embeddings_1.shape[1] == mm_embeddings_2.shape[1] # equal dim
            N = mm_embeddings_1.size(0)
            context_norm_1 = mm_embeddings_1 / mm_embeddings_1.norm(p=2, dim=-1, keepdim=True)
            context_norm_2 = mm_embeddings_2 / mm_embeddings_2.norm(p=2, dim=-1, keepdim=True)

            knn_indices = torch.empty((N, self.knn_k), dtype=torch.long, device=device)

            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                batch = context_norm_1[i:end_i]  # (B, D)
                sim_batch = torch.matmul(batch, context_norm_2.transpose(0, 1))  # (B, N)
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

    def forward_all(self, test=False):
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        # c_feature_attn = c_feature.unsqueeze(1)
        # t_feature_attn = t_feature.unsqueeze(1)
        # c_feature, _ = self.attn_image(c_feature_attn, c_feature_attn, c_feature_attn)
        # t_feature, _ = self.attn_text(t_feature_attn, t_feature_attn, t_feature_attn)
        # c_feature = c_feature.squeeze(1)
        # t_feature = t_feature.squeeze(1)
        #
        # c_feature_t, _ = self.cross_attn_image(query=c_feature, key=t_feature, value=t_feature)
        # t_feature_t, _ = self.cross_attn_text(query=t_feature, key=c_feature, value=c_feature)
        # c_feature = c_feature_t.squeeze(1)
        # t_feature = t_feature_t.squeeze(1)

        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        
        # mm_feature_full = torch.abs(
        #     (torch.mul(mm_feature_full, mm_feature_full) + torch.mul(self.item_embeddings, self.item_embeddings))/2 + 1e-8
        # ).sqrt()

        mm_moe = self.moe_layer(
            F.normalize(t_feature),
            F.normalize(c_feature)
        )
        
        features = []
        # features.append(mm_feature_full)
        # features.append(mm_moe)
        bi_feature_full = self.item_embeddings
        bi_feature_full_graph, _ = self.light_gcn(bi_feature_full, self.iui_edge_index, return_attention_weights=True)
        bi_feature_full = bi_feature_full + bi_feature_full_graph
        features.append(bi_feature_full)

        # cf_feature_full = self.cf_transformation(self.cf_feature)
        # cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        # features.append(cf_feature_full)

        if self.conf['use_modal_sim_graph']:
            # h = self.item_emb_modal
            # for i in range(self.num_layer_modal_graph):
            #     h = torch.sparse.mm(self.mm_adj, h)
            # features.append(h)

            item_emb_modal, _ = self.ii_modal_sim_gat(
                self.item_emb_modal,
                self.mm_adj.coalesce(),
                return_attention_weights=True
            )
            cross_modal_item_emb, _ = self.cross_modal_sim_gnn(
                self.item_emb_modal,
                self.cross_mm_adj.coalesce(),
                return_attention_weights=True
            )
            # features.append(cross_modal_item_emb)
            # print(f'type of cross_modal_item_emb forward_all: {type(cross_modal_item_emb)}')
            # features.append(item_emb_modal)
        # features.append((mm_feature_full + cross_modal_item_emb)/2)
        features.append(mm_feature_full)

        # hypergraph net 
        # if self.conf['use_hyper_graph']:
        #     item_hyper_emb = self.hyper_graph_conv_net(
        #         self.item_hyper_emb
        #     )
        #     features.append(item_hyper_emb)

        if not self.conf['use_pwc_fusion']:
            features = torch.stack(features, dim=-2)  # [bs, #modality, d]
            final_feature = self.selfAttention(F.normalize(features, dim=-1))
        else:
            final_feature = self.pwc_fusion(
                a=features[0],
                b=features[1],
                c=features[2]
            )
            final_feature = self.mlp_pwc(final_feature)

        # enhancing final_feature 
        final_feature_enhanced, _ = self.light_gcn(final_feature, self.iui_edge_index, return_attention_weights=True)
        final_feature = self.conf['final_feature_alpha']*final_feature + (1-self.conf['final_feature_alpha'])*final_feature_enhanced # residual connection

        # final_feature = final_feature + cate_emb
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

        # hyper graph
        item_hyper_emb = self.hyper_graph_conv_net(
            self.item_hyper_emb
        )

        # gat asymmetric
        if self.conf['use_iui_conv']:
            item_gat_emb, _ = self.iui_gat_conv(
                self.item_gat_emb,
                self.iui_edge_index,
                return_attention_weights=True
            )
            item_b_gat_emb, _ = self.ibi_gat_conv(
                self.item_gat_emb,
                self.ibi_edge_index,
                return_attention_weights=True
            )
        # item_gat_emb = item_gat_emb + item_b_gat_emb
        # diffusion with final_feature
        elbo = 0
        if self.conf['use_diffusion']:
            if not test:
                item_diff = self.diff_process.caculate_losses(
                    self.SDNet,
                    item_gat_emb,
                    self.conf['reweight']
                )   
                elbo = item_diff['loss'].mean()
                # final_feature = final_feature + item_diff['pred_xstart']
                item_gat_emb = item_gat_emb + item_diff['pred_xstart']
            else: 
                # test
                item_diff = self.diff_process.p_sample(
                    self.SDNet, 
                    item_gat_emb, 
                    self.conf['sampling_steps'],
                    self.conf['sampling_noise']
                )
                item_gat_emb = item_gat_emb + item_diff
            # final_feature = final_feature + item_diff_pred

        # multimodal fusion <<<

        # graph fusion
        graph_f = [item_gat_emb, item_emb_modal]
        graph_f = torch.stack(graph_f, dim=-2)
        graph_f = self.cross_attention(graph_f)
        graph_f = graph_f.mean(dim=-2)

        return final_feature, item_gat_emb, item_emb_modal, cross_modal_item_emb , elbo, graph_f

    def forward(self, seq_modify, all=False, test=False):
        if all is True:
            return self.forward_all(test=test)

        modify_mask = seq_modify == self.num_item
        seq_modify.masked_fill_(modify_mask, 0)
        c_feature = self.c_encoder(self.content_feature)
        t_feature = self.t_encoder(self.text_feature)

        # c_feature_attn = c_feature.unsqueeze(1)
        # t_feature_attn = t_feature.unsqueeze(1)
        # c_feature, _ = self.attn_image(c_feature_attn, c_feature_attn, c_feature_attn)
        # t_feature, _ = self.attn_text(t_feature_attn, t_feature_attn, t_feature_attn)
        # c_feature = c_feature.squeeze(1)
        # t_feature = t_feature.squeeze(1)
        #
        # c_feature_t, _ = self.cross_attn_image(query=c_feature, key=t_feature, value=t_feature)
        # t_feature_t, _ = self.cross_attn_text(query=t_feature, key=c_feature, value=c_feature)
        # c_feature = c_feature_t.squeeze(1)
        # t_feature = t_feature_t.squeeze(1)

        mm_feature_full = F.normalize(c_feature) + F.normalize(t_feature)
        
        # mm_feature_full = torch.abs(
        #     (torch.mul(mm_feature_full, mm_feature_full) + torch.mul(self.item_embeddings, self.item_embeddings))/2 + 1e-8
        # ).sqrt()

        mm_moe = self.moe_layer(
            F.normalize(t_feature),
            F.normalize(c_feature)
        )

        features = []
        # features.append(mm_feature_full)
        # features.append(mm_moe)
        bi_feature_full = self.item_embeddings
        bi_feature_full_graph, _ = self.light_gcn(bi_feature_full, self.iui_edge_index, return_attention_weights=True)
        bi_feature_full = bi_feature_full + bi_feature_full_graph
        features.append(bi_feature_full)
        # features.append(self.item_embeddings)

        # cf_feature_full = self.cf_transformation(self.cf_feature)
        # cf_feature_full[self.cold_indices_cf] = mm_feature_full[self.cold_indices_cf]
        # features.append(cf_feature_full)

        if self.conf['use_modal_sim_graph']:
            # h = self.item_emb_modal
            # for i in range(self.num_layer_modal_graph):
            #     h = torch.sparse.mm(self.mm_adj, h)
            # features.append(h)

            item_emb_modal, _ = self.ii_modal_sim_gat(
                self.item_emb_modal,
                self.mm_adj.coalesce(),
                return_attention_weights=True
            )
            cross_modal_item_emb, _ = self.cross_modal_sim_gnn(
                self.item_emb_modal,
                self.cross_mm_adj.coalesce(),
                return_attention_weights=True
            )

        features.append(mm_feature_full)

        # features.append((mm_feature_full + cross_modal_item_emb)/2)
            # features.append(cross_modal_item_emb)
            # print(f'type of cross_modal_item_emb forward: {type(cross_modal_item_emb)}')

        # if self.conf['use_hyper_graph']:
        #     item_hyper_emb = self.hyper_graph_conv_net(
        #         self.item_hyper_emb
        #     )
        #     features.append(item_hyper_emb)
        
        if not self.conf['use_pwc_fusion']:
            features = torch.stack(features, dim=-2)  # [n_items, n_modal, dim]
            final_feature = self.selfAttention(F.normalize(features, dim=-1)) # [n_items, dim]
        else:
            final_feature = self.pwc_fusion(
                a=features[0],
                b=features[1],
                c=features[2]
            )
            final_feature = self.mlp_pwc(final_feature)
        # print(f'pwc feature in forward: {final_feature.shape}') 

        # final_feature_apha = 0.8 -> recall@20: 0.0388 ndcg@20: 0.02420
        final_feature_enhanced, _ = self.light_gcn(final_feature, self.iui_edge_index, return_attention_weights=True)
        final_feature = self.conf['final_feature_alpha']*final_feature + (1-self.conf['final_feature_alpha'])*final_feature_enhanced

        # final_feature = final_feature + cate_emb
        # graph propagation
        # if self.conf['use_modal_sim_graph']:
        #     h = self.item_emb_modal
        #     for i in range(1):
        #         h = torch.sparse.mm(self.mm_adj, h)
        #     final_feature = final_feature + self.alpha_sim_graph * F.normalize(h)

        # hyper graph 
        item_hyper_emb = self.hyper_graph_conv_net(
            self.item_hyper_emb
        )
        # final_feature = final_feature + item_hyper_emb
        bundle_hyper_emb = self.bundle_agg_graph_ori @ item_hyper_emb

        # gat asymmetric
        if self.conf['use_iui_conv']:
            item_gat_emb, _ = self.iui_gat_conv(
                self.item_gat_emb,
                self.iui_edge_index,
                return_attention_weights=True
            )
            item_b_gat_emb, _ = self.ibi_gat_conv(
                self.item_gat_emb,
                self.ibi_edge_index,
                return_attention_weights=True
            )
        # item_gat_emb = item_gat_emb + item_b_gat_emb
        # diffusion 
        # item_gat_emb = (item_gat_emb + item_emb_modal) / 2 
        # item_gat_emb = item_emb_modal + cross_modal_item_emb
        # item_gat_emb = item_gat_emb
        # item_gat_emb = self.mlp(item_gat_emb, item_emb_modal)

        elbo = 0
        if self.conf['use_diffusion']:
            if not test:
                item_diff = self.diff_process.caculate_losses(
                    self.SDNet,
                    item_gat_emb,
                    self.conf['reweight']
                )   
                elbo = item_diff['loss'].mean()
                item_gat_emb = item_gat_emb + item_diff['pred_xstart']
            else: 
                # test
                item_diff = self.diff_process.p_sample(
                    self.SDNet, 
                    item_gat_emb, 
                    self.conf['sampling_steps'],
                    self.conf['sampling_noise']
                )
                item_gat_emb = item_gat_emb + item_diff

        # graph fusion
        graph_f = [item_gat_emb, item_emb_modal]
        graph_f = torch.stack(graph_f, dim=-2)
        graph_f = self.cross_attention(graph_f)
        graph_f = graph_f.mean(dim=-2)

        bundle_gat_emb = self.bundle_agg_graph_ori @ item_gat_emb 
        bundle_modal_emb = self.bundle_agg_graph_ori @ item_emb_modal
        bundle_cross_emb = self.bundle_agg_graph_ori @ cross_modal_item_emb
        bundle_f_emb = self.bundle_agg_graph_ori @ graph_f

        final_feature = final_feature[seq_modify] # [bs, n_token, d]
        # print(f'shape of final feature in forward: {final_feature.shape}')
        
        bs, n_token, d = final_feature.shape
        # final_feature = self.selfAttention(F.normalize(features.view(-1, N_modal, d), dim=-1))

        # print(f'shape of final feature: {final_feature.shape}') # [1280, 64]

        final_feature = final_feature.view(bs, n_token, d)
        # multimodal fusion <<<

        # graph fusion

        return final_feature, bundle_gat_emb, bundle_modal_emb, bundle_cross_emb, elbo, bundle_f_emb

class CLHE(nn.Module):
    def __init__(self, conf, raw_graph, features, cate):
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
        self.cate = cate 

        self.encoder = HierachicalEncoder(conf, raw_graph, features, cate)
        # decoder has the similar structure of the encoder
        self.decoder = HierachicalEncoder(conf, raw_graph, features, cate)

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

        self.get_bundle_agg_graph_ori(self.bi_graph_seen)

        self.print_model_using()

    def print_model_using(self):
        print(f'use contrastive loss: {self.conf["use_cl"]}')

    def get_bundle_agg_graph_ori(self, graph):
        bi_graph = graph
        device = self.device
        bundle_size = bi_graph.sum(axis=1) + 1e-8 # calculate size for each bundle 
        # print(f"bundle size: {bundle_size.shape}")
        # print(f"diag bundle: {sp.diags(1/bundle_size.A.ravel()).shape}")
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph # sp.diags(1/bundle_size.A.ravel()): D^-1 
        # print(f'graph: {graph}')
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device) 

    def forward(self, batch):
        idx, full, seq_full, modify, seq_modify = batch  # x: [bs, #items]
        mask = seq_full == self.num_item
        feat_bundle_view, bundle_gat_emb, bundle_modal_emb, bundle_cross_emb, _, bundle_f = self.encoder(seq_full)  # [bs, n_token, d]

        # bundle feature construction >>>
        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        feat_retrival_view, item_gat_emb, item_modal_emb, cross_modal_item_emb, _, item_f = self.decoder(batch, all=True)

        # option 1 
        bundle_feature = bundle_feature + bundle_gat_emb[idx] + bundle_modal_emb[idx] 
        feat_retrival_view = feat_retrival_view + item_gat_emb + item_modal_emb 
        # bundle_feature = bundle_feature + bundle_f[idx]
        # feat_retrival_view = feat_retrival_view + item_f
        main_score = bundle_feature @ feat_retrival_view.transpose(0, 1) 

        modal_bundle_feature = bundle_modal_emb[idx] + bundle_cross_emb[idx]
        modal_item_feature = item_modal_emb + cross_modal_item_emb
        modal_score = modal_bundle_feature @ modal_item_feature.transpose(0, 1)

        items_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()
        # best 0.1 
        item_loss = 0.1 * cl_loss_function(
            item_gat_emb[items_in_batch], item_modal_emb[items_in_batch], 0.2
        )
        bundle_loss = 0.1 * cl_loss_function(
            bundle_gat_emb[idx], bundle_modal_emb[idx], 0.2
        )

        # option 2 
        # bundle_feature = bundle_feature + bundle_cross_emb[idx]
        # feat_retrival_view = feat_retrival_view + cross_modal_item_emb
        # main_score = bundle_feature @ feat_retrival_view.transpose(0, 1)
        # modal_bundle_feature = bundle_modal_emb[idx] + bundle_gat_emb[idx] 
        # modal_item_feature = item_modal_emb + item_gat_emb
        # modal_score = modal_bundle_feature @ modal_item_feature.transpose(0, 1)

        logits = 0
        if self.conf['use_cl']:
            logits = main_score + modal_score
        else:
            logits = main_score
        
        # main loss 
        loss = recon_loss_function(logits, full)  

        # contrastive loss: only calculate with item in batch to avoid out of memory
        item_contras_loss = 0
        bundle_contras_loss = 0

        if self.conf['use_cl']:
            item_in_batch = torch.argwhere(full.sum(dim=0)).squeeze()
            item_contras_loss = 0.1 * cl_loss_function(
                feat_retrival_view[item_in_batch].view(-1, self.embedding_size),
                modal_item_feature[item_in_batch].view(-1, self.embedding_size),
            )

            bundle_contras_loss = 0.1 * cl_loss_function(
                bundle_feature, 
                modal_bundle_feature
            )

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
            'loss': loss + item_loss + bundle_loss,
            # 'loss': loss,
            'item_loss': loss,
            'bundle_loss': loss
        }

        return combine_loss

    def evaluate(self, _, batch):
        idx, x, seq_x = batch
        mask = seq_x == self.num_item
        feat_bundle_view, bundle_gat_emb, bundle_modal_emb, bundle_cross_emb, _, bundle_f = self.encoder(seq_x, test=True)

        bundle_feature = self.bundle_encode(feat_bundle_view, mask=mask)

        feat_retrival_view, item_gat_emb, item_modal_emb, cross_modal_item_emb, _, item_f = self.decoder(
            (idx, x, seq_x, None, None), 
            all=True,
            test=True 
        )

        # option 1 
        bundle_feature = bundle_feature + bundle_gat_emb[idx] + bundle_modal_emb[idx]
        feat_retrival_view = feat_retrival_view + item_gat_emb + item_modal_emb
        # bundle_feature = bundle_feature + bundle_f[idx]
        # feat_retrival_view = feat_retrival_view + item_f
        main_score = bundle_feature @ feat_retrival_view.transpose(0, 1)

        modal_bundle_feature = bundle_modal_emb[idx] + bundle_cross_emb[idx] 
        item_modal_feature = item_modal_emb + cross_modal_item_emb
        modal_score = modal_bundle_feature @ item_modal_feature.transpose(0, 1)

        # option 2 
        # bundle_feature = bundle_feature + bundle_cross_emb[idx]
        # feat_retrival_view = feat_retrival_view + cross_modal_item_emb
        # main_score = bundle_feature @ feat_retrival_view.transpose(0, 1)
        # modal_bundle_feature = bundle_modal_emb[idx] + bundle_gat_emb[idx] 
        # item_modal_feature = item_modal_emb + item_gat_emb
        # modal_score = modal_bundle_feature @ item_modal_feature.transpose(0, 1)

        logits = 0
        if self.conf['use_cl']:
            logits = main_score + modal_score
        else:
            logits = main_score

        return logits

    def propagate(self, test=False):
        return None
