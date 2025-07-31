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


