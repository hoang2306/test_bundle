import torch 
import os 
import numpy as np 
from utility import slash 
import argparse


parser = argparse.ArgumentParser(description='generate cross-modal adjacency matrix.')
parser.add_argument('--device', type=str, default='cpu', help='select device for generate')
parser.add_argument('--data_name', type=str, default='pog', help='name of the dataset')
args = parser.parse_args()

device = args.device
data_name = args.data_name # the code test on pog dataset, so default is pog
dataset_root = os.path.join('datasets', data_name)
# dataset_root = os.path.join('BundleConstruction', data_name)
print(f'dataset root: {dataset_root}')
print(f'use device: {device}')
slash()

# check content and check description feature mm adj 
content_sim_path = os.path.join(dataset_root, 'content_sim.pt')
des_sim_path = os.path.join(dataset_root, 'des_sim.pt')
content_sim = torch.load(content_sim_path, weights_only=True, map_location=device)
des_sim = torch.load(des_sim_path, weights_only=True, map_location=device)
print(f'content_sim shape: {content_sim.shape}')
print(f'des_sim shape: {des_sim.shape}')
slash()

content_mm_adj_path = os.path.join(dataset_root, 'content_mm_adj.pt')
des_mm_adj_path = os.path.join(dataset_root, 'description_mm_adj.pt')
content_mm_adj = torch.load(content_mm_adj_path, weights_only=True, map_location=device)
des_mm_adj = torch.load(des_mm_adj_path, weights_only=True, map_location=device)
print(f'content_mm_adj shape: {content_mm_adj.shape}')
print(f'des_mm_adj shape: {des_mm_adj.shape}')
slash()

# check 
top_content_sample = content_sim[0].numpy()
print(f'top_content_sample: {top_content_sample}')
slash()
content_mm_adj_idx = content_mm_adj[0].coalesce().indices().squeeze().cpu().numpy()
print(f'content_mm_adj_idx: {content_mm_adj_idx}')

for idx in content_mm_adj_idx:
    if idx not in top_content_sample:
        print(f'content_mm_adj_idx {idx} not in top_content_sample')
print(f'check done for content mm adj idx')

top_des_sim_sample = des_sim[0].numpy()
print(f'top content sim sample: {top_des_sim_sample}')
slash()
des_mm_adj_idx = des_mm_adj[0].coalesce().indices().squeeze().cpu().numpy()
print(f'des_mm_adj_idx: {des_mm_adj_idx}')

for idx in des_mm_adj_idx:
    if idx not in top_des_sim_sample:
        print(f'des_mm_adj_idx {idx} not in top_des_sim_sample')

print(f'check done for des mm adj idx')