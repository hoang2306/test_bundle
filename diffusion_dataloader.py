import os 
import torch
import numpy as np 
from scipy.sparse import coo_matrix
import torch.utils.data as data
from utility import slash
from tqdm import tqdm 
import argparse 
from models import diffusion_mm 


def adj2coo(mat, data_shape):
    coo_mat = coo_matrix(
        (np.ones(mat.shape[1]), (mat[0], mat[1])), 
        shape=(data_shape[0], data_shape[1])
    )
    return coo_mat


class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index].to_dense()
        return item, index

    def __len__(self):
        return len(self.data)


class diffusion_dataloader:
    def __init__(self, conf, batch_size):
        print('diffusion dataloader')

        self.batch_size = batch_size
        self.conf = conf

        self.neigh_threshold = 1
        n_item = 48676
        # iui_edge_path = os.path.join('data_test', conf['dataset'], f'n_neigh_iui_{self.neigh_threshold}.npy')
        iui_edge_path = os.path.join('datasets', conf['dataset'], f'n_neigh_iui_{self.neigh_threshold}.npy')
        print(f'iui_edge_path: {iui_edge_path}')
        
        self.iui_edge_index = np.load(iui_edge_path)
        self.iui_edge_index_tensor = torch.from_numpy(self.iui_edge_index).long().t().contiguous()
        print(f'loaded {iui_edge_path}')

        coo_iui = adj2coo(self.iui_edge_index, data_shape=(n_item, n_item))
        indices = np.vstack((coo_iui.row, coo_iui.col))
        
        iui_tensor_sparse = torch.sparse_coo_tensor(
            indices=torch.tensor(indices),
            values=torch.tensor(coo_iui.data),
            size=coo_iui.shape,
            dtype=torch.float32  
        )
        
        print(iui_tensor_sparse.shape)
        print(iui_tensor_sparse.coalesce().indices())  # Removed [3] which is invalid
        self.diffusion_data = DiffusionData(data=iui_tensor_sparse)
        self.diffusion_loader = torch.utils.data.DataLoader(
            self.diffusion_data, 
            batch_size=batch_size,
            shuffle=True
        )


# argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pog', help='dataset name')
parser.add_argument('--device', type=str, default='cpu', help='choose device')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='num of epochs')
args = parser.parse_args()

# train diffusion 
print(f'import model ok')


# test train 
conf = {
    'dataset': args.dataset
}
diff_data = diffusion_dataloader(
    conf=conf, 
    batch_size=args.batch_size
)
diff_data_loader = diff_data.diffusion_loader

# create dummy embedding
item_emb = torch.randn(48676, 64) 
image_emb = torch.rand(48676, 64)
text_emb = torch.rand(48676, 64)
print(f'item emb shape: {item_emb.shape}')

device = args.device
# model
out_dims = [1000, 48676]
in_dims = out_dims[::-1]
denoise_model_text = diffusion_mm.Denoise_cuda(
    in_dims=in_dims, 
    out_dims=out_dims, 
    emb_size=64, 
    norm=True, 
    dropout=0.5
).to(device)
print(f'init denoise model text')
denoise_model_image = diffusion_mm.Denoise_cuda(
    in_dims=in_dims, 
    out_dims=out_dims, 
    emb_size=64, 
    norm=True, 
    dropout=0.5
).to(device)
print(f'init denoise model image')

diffusion_model = diffusion_mm.GaussianDiffusion_cuda(
			noise_scale=0.1, 
			noise_min=0.0001, 
			noise_max=0.02, 
			steps=5
		).to(device)
print(f'init diffusion model')

denoise_opt_image = torch.optim.Adam(denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

for ep in range(args.epochs):
    total_loss = 0
    pbar = tqdm(enumerate(diff_data_loader), total=len(diff_data_loader))
    for i, batch in pbar:
        batch_item, batch_index = batch
        # print(f'batch {i}, x shape: {x.shape}, index: {index}')
        # print(f'x: {x}')

        batch_item = batch_item.to(device)
        batch_index = batch_index.to(device)

        diff_loss_image, gc_loss_image = diffusion_model.training_losses(
            model=denoise_model_image, 
            x_start=batch_item, 
            itmEmbeds=item_emb.to(device), 
            batch_index=batch_index, 
            model_feats=image_emb.to(device)
        )

        # print(f'diff_loss_image: {diff_loss_image}')
        # print(f'gc_loss_image: {gc_loss_image}')

        # loss_image = diff_loss_image.mean() + gc_loss_image.mean()
        loss_image = gc_loss_image.mean()
        loss_image.backward()
        denoise_opt_image.step()
        # print(f'loss image: {loss_image}')
        total_loss += loss_image.item()
    total_loss = np.mean(total_loss)
    print(f'epoch {ep}, total loss: {total_loss}')

  