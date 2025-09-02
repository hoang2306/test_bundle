import torch
import torch.nn.functional as F 
import pickle as pkl 
import numpy as np 

def load_pkl(path):
    with open(path, 'rb') as f:
        return pkl.load(f)

def cal_statistic(cate_map):
    n_item = len(cate_map)
    n_cate = len(set(cate_map.values()))
    return n_item, n_cate

cate_map = load_pkl('item_id_2_cate.pkl')

all_item_id = np.arange(len(cate_map))
n_item, n_cate = cal_statistic(cate_map)
print(f'n item: {n_item}')
print(f'n cate: {n_cate}')
print(f'element need to store: {n_item * n_cate}')

x_ = [cate_map[item_id] for item_id in all_item_id]

cate_one_hot = F.one_hot(torch.tensor(x_), num_classes=n_cate).float() 
print(cate_one_hot)
print((cate_one_hot.sum(dim=-1) > 1).sum(dim=0))

# rows = torch.arange(len(x_))
# cols = torch.tensor(x_)
# values = torch.ones(len(x_))

# cate_one_hot_sparse = torch.sparse_coo_tensor(
#     indices=torch.stack([rows, cols]),
#     values=values,
#     size=(len(all_item_id), int(cols.max())+1)
# )

# print(10673 * 192)



