import torch
import os
import numpy as np
import time
from tqdm import tqdm
import pickle as pkl
import argparse 


def compute_normalized_laplacian(indices, adj_size):
    # adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
    adj = torch.sparse_coo_tensor(
        indices, torch.ones_like(indices[0]), adj_size)
    row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
    r_inv_sqrt = torch.pow(row_sum, -0.5)
    rows_inv_sqrt = r_inv_sqrt[indices[0]]
    cols_inv_sqrt = r_inv_sqrt[indices[1]]
    values = rows_inv_sqrt * cols_inv_sqrt
    return torch.sparse_coo_tensor(indices, values, adj_size)


def get_cross_modal_knn_adj_mat(mm_embeddings_1, mm_embeddings_2, batch_size=1024):
    with torch.no_grad():
        # equal dim
        assert mm_embeddings_1.shape[1] == mm_embeddings_2.shape[1]
        N = mm_embeddings_1.size(0)
        context_norm_1 = mm_embeddings_1 / \
            mm_embeddings_1.norm(p=2, dim=-1, keepdim=True)
        context_norm_2 = mm_embeddings_2 / \
            mm_embeddings_2.norm(p=2, dim=-1, keepdim=True)

        knn_indices = torch.empty((N, knn_k), dtype=torch.long, device=device)

        # trick ~ math.ceil(N / batch_size)
        total_loop = (N + batch_size - 1) // batch_size

        # print(f'total loop: {total_loop}')
        pbar = tqdm(range(0, N, batch_size), total=total_loop,
                    desc=f'calculating {data_name} cross modal knn adj mat')
        for i in pbar:
            end_i = min(i + batch_size, N)
            batch = context_norm_1[i:end_i]  # (B, D)
            sim_batch = torch.matmul(
                batch, context_norm_2.transpose(0, 1))  # (B, N)
            _, topk = torch.topk(sim_batch, knn_k, dim=-1)
            knn_indices[i:end_i] = topk

        adj_size = (N, N)

        indices0 = torch.arange(
            N, device=device).unsqueeze(1).expand(-1, knn_k)
        indices = torch.stack(
            (indices0.flatten(), knn_indices.flatten()), dim=0)

        return indices, compute_normalized_laplacian(indices, adj_size)

# parser
parser = argparse.ArgumentParser(description='generate cross-modal adjacency matrix.')
parser.add_argument('--device', type=str, default='cpu', help='select device for generate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size for knn search')
parser.add_argument('--knn_k', type=int, default=10, help='number of nearest neighbors for knn search')
parser.add_argument('--data_name', type=str, default='pog', help='name of the dataset')
args = parser.parse_args()

device = args.device
knn_k = args.knn_k
batch_size = args.batch_size
data_name = args.data_name
dataset_root = os.path.join('datasets', data_name)
# dataset_root = os.path.join('BundleConstruction', data_name)
print(f'dataset root: {dataset_root}')

content_feature_path = os.path.join(dataset_root, 'content_feature.pt')
content_feature = torch.load(
    content_feature_path, weights_only=True, map_location=device)
description_feature_path = os.path.join(dataset_root, 'description_feature.pt')
description_feature = torch.load(
    description_feature_path, weights_only=True, map_location=device)

print(f'content_feature shape: {content_feature.shape}')
print(f'description_feature shape: {description_feature.shape}')

content_indices, content_mm_adj = get_cross_modal_knn_adj_mat(
    content_feature, content_feature, batch_size=batch_size)
print(f'content_mm_adj shape: {content_mm_adj.shape}')

description_indices, description_mm_adj = get_cross_modal_knn_adj_mat(
    description_feature, description_feature, batch_size=batch_size)
print(f'description_mm_adj shape: {description_mm_adj.shape}')

# save file
content_mm_adj_path = os.path.join(dataset_root, 'content_mm_adj.pt')
description_mm_adj_path = os.path.join(dataset_root, 'description_mm_adj.pt')
torch.save(content_mm_adj, content_mm_adj_path)
torch.save(description_mm_adj, description_mm_adj_path)
print(f'content_mm_adj saved to {content_mm_adj_path}')
print(f'description_mm_adj saved to {description_mm_adj_path}')

# save pkl 
mm_adj_dict = {
    'content_mm_adj': content_mm_adj,
    'description_mm_adj': description_mm_adj,
    'content_indices': content_indices,
    'description_indices': description_indices
}

with open(os.path.join(dataset_root, 'mm_adj.pkl'), 'wb') as f:
    pkl.dump(mm_adj_dict, f)
print(f'mm_adj.pkl saved to {os.path.join(dataset_root, "mm_adj.pkl")}')