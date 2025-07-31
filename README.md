# test_bundle

### generate multimodal adj matrix 
use gen_mm_adj.py script to generate multimodal adj matrix for gnn 
```
python gen_mm_adj.py --dataset pog --device cpu
```
or 
```
sh gen_mm_adj.sh
```
hyperparams support: data_name (str), knn_k (int), device (str), batch_size (int)