import json
import numpy as np 
from scipy.sparse import coo_matrix 
from scipy import sparse as sp 
import torch
import os 
from tqdm import tqdm 
import argparse 

def slash(s=''):
    print('-'*30 + s + '-'*30)

def np_to_sparse(data, shape):
    val = np.ones(len(data), dtype=np.float32)
    sp_mat = sp.coo_matrix(
        (val, (data[:,0], data[:,1])), shape=shape
    )
    return sp_mat

def load_stats(path):
    with open(path, 'r') as f:
        data = json.load(f)
    n_item, n_bundle, n_user = data["#I"], data['#B'], data['#U']
    return n_item, n_bundle, n_user

def load_txt(path):
    pair_data = []
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split(', ')
            data = [int(i) for i in data]
            bundle_id = data[0]
            for item_id in data[1:]:
                pair_data.append([bundle_id, item_id])
                
    return np.array(pair_data)

class gen_matrix:
    def __init__(
        self,
        iui_npz_name,
        n_neigh_iui_name,
        ui, 
        threshold,
        fill_thresh_hold,
        output_dir
    ):
        self.iui_npz_name = iui_npz_name
        self.ui = ui
        self.threshold = threshold
        self.n_neigh_iui_name = n_neigh_iui_name
        self.fill_thresh_hold = fill_thresh_hold
        self.output_name = f"{self.n_neigh_iui_name}_{self.fill_thresh_hold}.npy"
        # self.output_dir = '/kaggle/working/'
        self.output_dir = output_dir
        self.output = os.path.join(self.output_dir, self.output_name)

    def load_sp_mat(self, name):
        return sp.load_npz(name)
    
    def gen_ii_asym(self, mat, threshold=0):
        # bi -> ii: bi.T @ bi
        ii_co = mat.T @ mat 
        mask = ii_co > threshold
        return ii_co.multiply(mask)
    
    def process(self): 
        iui = self.gen_ii_asym(self.ui) 
        # save file 
        sp.save_npz(self.iui_npz_name, iui)
        print(f'save file {self.iui_npz_name}')

        ibi = self.load_sp_mat(self.iui_npz_name)
        print(f'load file {self.iui_npz_name}')

        ibi_filter = ibi >= self.fill_thresh_hold

        # mask all diagonal elements
        diag_filter_i = sp.coo_matrix(
                (np.ones(n_items), ([i for i in range(0, n_items)], [i for i in range(0, n_items)])),
                shape=ibi.shape).tocsr()
        fil_ibi = ibi.multiply(ibi_filter)
        diag_filter_ibi = fil_ibi.multiply(diag_filter_i)

        neighbor_ibi = fil_ibi - diag_filter_ibi.tocsc()
        n_ibi = neighbor_ibi.tocoo()
        ibi_edge_index = torch.tensor([list(n_ibi.row), list(n_ibi.col)], dtype=torch.int64)
        # print(ibi_edge_index.shape)
        np.save(self.output, ibi_edge_index)
        print(f'output final shape: {ibi_edge_index.shape}')
        print(f'save output final file: {self.output}')
        slash()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='pog', help="select dataset")
parser.add_argument("--mode", type=str, default='iui', help="select type of dataset")
args = parser.parse_args()
print(args)

data_root_path = os.path.join('datasets', args.dataset)
print(f'data path: {data_root_path}')
n_items, n_bundles, n_users = load_stats(os.path.join(data_root_path, 'count.json'))
print(f'n_items: {n_items}, n_bundles: {n_bundles}, n_users: {n_users}')

# mode = 'iui' # ibi or iui
mode = args.mode
print(f'mode: {mode}')
data_path = 'ui_full.txt' if mode=='iui' else 'bi_train.txt'
print(f'data path: {data_path}')
ui_inter = load_txt(path=os.path.join(data_root_path,data_path))
shape_data = (n_users,n_items) if mode=='iui' else (n_bundles, n_items)
print(f'shape data: {shape_data}')
ui = np_to_sparse(data=ui_inter, shape=shape_data).tocsr()
# set threshold 
# threshold_list = [1,2,3,4,5]
threshold_list = np.arange(1,11)
print(f'threshold list: {threshold_list}')
slash()

output_dir = data_root_path
for threshold in threshold_list:
    iui_npz_name = os.path.join(output_dir, f"{mode}_{threshold}.npz")
    n_neigh_iui_name = f"n_neigh_{mode}" 
    gen_matrix_obj = gen_matrix(
        iui_npz_name=iui_npz_name,
        n_neigh_iui_name=n_neigh_iui_name,
        ui=ui, 
        threshold=0,
        fill_thresh_hold=threshold,
        output_dir=output_dir
    )
    gen_matrix_obj.process()