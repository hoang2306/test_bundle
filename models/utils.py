from typing import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp 

eps = 1e-9 # avoid zero division


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(
        torch.LongTensor(indices), 
        torch.FloatTensor(values), 
        torch.Size(graph.shape)
    )
    return graph
    
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

def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_uniform_(m)


class SublayerConnection(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = {
            "dim": 64,
            "residual": False,
            "layernorm": False,
            "dropout": 0,
            "device": None
        }

        for i in conf:
            self.conf[i] = conf[i]
        self.device = self.conf['device']
        self.ln = nn.LayerNorm(
            self.conf["dim"], 
            eps=eps, 
            elementwise_affine=False
        ).to(self.device)
        self.dropout = nn.Dropout(self.conf['dropout'])

    def forward(self, x, sublayer):
        if self.conf['layernorm']:
            x = self.ln(x)
        y = self.dropout(sublayer(x))
        y += x if self.conf["residual"] else 0
        return y


class SelfAttention(nn.Module):
    def __init__(self, conf, data={}):
        super().__init__()
        self.conf = {
            "dim": 64,
            "n_head": 1,
            "device": None,
            "w_v": False,
            "layernorm": False,
            "ffn": False,
            "residual": False,
            "dropout_ratio": 0,
        }

        for i in conf:
            self.conf[i] = conf[i]
        self.device = self.conf["device"]

        self.w_q = nn.Linear(self.conf['dim'], self.conf['dim'], bias=False).to(self.device)
        self.w_k = nn.Linear(self.conf['dim'], self.conf['dim'], bias=False).to(self.device)

        if self.conf["w_v"]:
            self.w_v = nn.Linear(self.conf['dim'], self.conf['dim'], bias=False).to(self.device)

        self.ln = nn.LayerNorm(self.conf["dim"], eps=eps, elementwise_affine=False).to(self.device)

        self.dropout = nn.Dropout(self.conf['dropout_ratio'])

        if self.conf["ffn"]:
            self.ffn = nn.Sequential(
                OrderedDict([
                ('w1', nn.Linear(self.conf["dim"], self.conf["dim"]*2)),
                ('act1', nn.ReLU()),
                ('w2', nn.Linear(self.conf["dim"]*2,  self.conf["dim"])),
            ])
            ).to(self.device)

    def multiHeadAttention(self, x, mask, dims):
        bs, heads, n_token, d = dims

        if self.conf['layernorm']:
            x = self.ln(x)

        q = self.w_q(x)
        k = self.w_k(x)

        if self.conf["w_v"]:
            v = self.w_v(x)
        else:
            v = x

        q = q.view(bs, n_token, heads, d//heads).transpose(2, 1)
        k = k.view(bs, n_token, heads, d//heads).transpose(2, 1)
        A = q.mul(d ** -0.5) @ k.transpose(-2, -1)

        att = A.masked_fill(mask.view(bs, 1, 1, n_token), -1e9)
        att = att.softmax(dim=-1)

        v = v.view(bs, n_token, heads, d//heads).transpose(2, 1)
        y = att @ v  # [bs, N_bundle, heads, n_token, d']

        y = y.transpose(2, 1).contiguous().view(bs, n_token, d)

        y += x if self.conf["residual"] else 0

        if self.conf["ffn"]:
            x = y
            if self.conf['layernorm']:
                x = self.ln(x)
            y = self.ffn(x)

            y += x if self.conf["residual"] else 0

        return y

    def forward(self, x, mask, dims):
        bs, heads, n_token, d = dims
        assert tuple(x.shape) == (bs, n_token, d)
        assert tuple(mask.shape) == (bs, 1, n_token)
        x = self.dropout(self.multiHeadAttention(x, mask, dims))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, conf, data):
        super(TransformerEncoder, self).__init__()
        self.conf = {
            "n_layer": 3,
            "dim": 64,
            "num_sequence": None,
            "num_token": 200,
            "n_head": 1,
            "dropout_ratio": 0,
            "data_path": None,
            "dataset": None,
            "layernorm": True,
            "ffn": False,
            "w_v": False,
            "residual": False,
            "device": None,
            "tag": ""
        }

        self.data = {
            "sp_graph": None,
        }
        for i in conf:
            self.conf[i] = conf[i]
        self.device = self.conf["device"]
        for i in data:
            self.data[i] = data[i]

        self.sp_graph = self.data["sp_graph"]
        self.num_sequence = self.conf["num_sequence"]

        self.attn_encode = [
            SelfAttention(conf=self.conf) for _ in range(self.conf["n_layer"])
        ]

        self.dropout = nn.Dropout(p=self.conf["dropout_ratio"])

    def forward(self, x, mask):
        # x: [bs, n_token, d]
        # mask: [bs, n_token]
        bs = x.shape[0]

        len = (~mask).sum(dim=-1).unsqueeze(-1)  # [bs, 1]
        mask = mask.unsqueeze(-2)  # [bs, 1, n_token]
        n_token = mask.shape[-1]

        dims = (
            bs,
            self.conf["n_head"],
            n_token,
            self.conf["dim"]
        )

        for layer in range(self.conf['n_layer']):
            x = self.attn_encode[layer](
                x,
                mask=mask,
                dims=dims
            )

        x = x.masked_fill(mask.view(bs, n_token, 1), 0)
        y = x.sum(-2) / (len + eps)  # mean-pooling, [bs, n_token, d]

        return y.squeeze()  # [bs, d]
