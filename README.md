# test_bundle

### generate multimodal adj matrix 
use `gen_mm_adj.py` script to generate multimodal adj matrix for gnn 
```
python gen_mm_adj.py --dataset pog --device cpu
```
or 
```
sh gen_mm_adj.sh
```
hyperparams support: data_name (str), knn_k (int), device (str), batch_size (int)

note that: in `gen_mm_adj.py` script, we calculate mm adj graph by using function from multi-modal recommendation SOTA and calculate by hand (we calculate by 2 ways) for validate purpose

to validate if generated mm adj is correct? use `validate_gen_mm_adj.py`
```
python validate_gen_mm_adj.py --dataset_name pog --device cpu
```
hyperparams support: dataset_nme (str), device (str)

if the output has form like: "check done for des mm adj idx --- ok", then the generated graph is calculated correctly

### how to run model
to run model for train, pls using the script (example for pog dataset):
```python
!python3 -u train.py \
    -g 0 \
    --dataset="pog" \
    --model="CLHE" \
    --type_gnn="anti_symmetric" \
    --use_wandb \
    --num_layer_gat=2 \
    --epoch=200 \
    --knn_k=10 \
    --num_layer_hypergraph=1 \
    --use_iui_conv\
    --use_modal_sim_graph \
    --use_hyper_graph \
    --use_cl \
    --num_workers=4 \
    --lr=1e-4 \
    --item_augment="FN" \
    --bundle_augment="ID" \
    --bundle_ratio=0.5 \
    --bundle_cl_temp=0.01 \
    --bundle_cl_alpha=0.5 \
    --cl_temp=0.5 \
    --cl_alpha=2 \
    --steps=120 \
    --noise_scale=0.1
```
