import os
import time 
import yaml
import json
from tqdm import tqdm
from datetime import datetime
from parse import get_cmd
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import torch.optim as optim
from utility import (
    Datasets,
    setup_seed
)
from metrics import (
    init_best_metrics,
    log_metrics,
    get_metrics,
    get_recall,
    get_ndcg,
    write_log
)
import models
import wandb 


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]
    conf = conf[dataset_name]
    for p in paras:
        conf[p] = paras[p]

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    setup_seed(conf["seed"])

    dataset = Datasets(conf)
    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    lr = paras['lr'] if "lr" in paras else conf['lrs'][0]
    l2_reg = paras['reg'] if "reg" in paras else conf['l2_regs'][0]
    embedding_size = paras['embedding_size'] if "embedding_size" in paras else conf['embedding_sizes'][0]
    num_layers = paras['num_layers'] if "num_layers" in paras else conf['num_layerss'][0]

    log_path = "./log/%s/%s" % (conf["dataset"], conf["model"])
    run_path = "./runs/%s/%s" % (conf["dataset"], conf["model"])
    checkpoint_model_path = "./checkpoints/%s/%s/model" % (
        conf["dataset"], conf["model"])
    checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (
        conf["dataset"], conf["model"])
    if not os.path.isdir(run_path):
        os.makedirs(run_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)
    if not os.path.isdir(checkpoint_conf_path):
        os.makedirs(checkpoint_conf_path)

    conf["l2_reg"] = l2_reg
    conf["embedding_size"] = embedding_size

    settings = []
    if conf["info"] != "":
        settings += [conf["info"]]

    settings += [
        "Epoch%d" % (conf['epochs']), str(conf["batch_size_train"]), str(lr), str(l2_reg), str(embedding_size)
    ]

    conf["num_layers"] = num_layers

    setting = "_".join(settings)
    log_path = log_path + "/" + setting
    run_path = run_path + "/" + setting
    checkpoint_model_path = checkpoint_model_path + "/" + setting
    checkpoint_conf_path = checkpoint_conf_path + "/" + setting
    run = SummaryWriter(run_path)

    try:
        model = getattr(models, conf['model'])(
            conf, dataset.graphs, dataset.features
        ).to(device)
    except:
        raise ValueError("Unimplemented model %s" % (conf["model"]))
    
    # load trained weight 
    try:
        model.load_state_dict(torch.load(
            checkpoint_model_path, 
            weights_only=True, 
            map_location=device
        ))
        print(f'load saved model successfully')
    except:
        raise ValueError('checkpoint model path is not exactly')

    with open(log_path, "a") as log:
        log.write(f"{conf}\n")
        print(conf)

    metrics = {}
    start_time = time.time()
    metrics["val"] = test(model, dataset.val_loader, conf)
    time_val = time.time() - start_time
    metrics["test"] = test(model, dataset.test_loader, conf)
    time_test = time.time() - time_val - start_time

    print(f'val time: {time_val}')
    print(f'test time: {time_test}')
    for topk in conf["topk"]:
        write_log(
            run=run,
            log_path=log_path,
            topk=topk,
            batch_anchor=0,
            metrics=metrics
        )

@torch.no_grad()
def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate()
    pbar = tqdm(dataloader, total=len(dataloader))
    for index, b_i_input, seq_b_i_input, b_i_gt in pbar:
        pred_i = model.evaluate(
            rs, (index.to(device), b_i_input.to(device), seq_b_i_input.to(device))
        )
        pred_i = pred_i - 1e8 * b_i_input.to(device)  # mask
        tmp_metrics = get_metrics(
            tmp_metrics, b_i_gt.to(device), pred_i, conf["topk"]
        )

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


if __name__ == "__main__":
    main()
