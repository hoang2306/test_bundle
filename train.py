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
    setup_seed, 
    slash
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
# wandb.login()

def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads if g is not None])

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
            conf, dataset.graphs, dataset.features, dataset.cate
        ).to(device)
    except:
        raise ValueError("Unimplemented model %s" % (conf["model"]))
    
    # calculate total params:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable_params}")
    print(f'total params: {total_params}')
    log = open(log_path, "a")
    log.write(f"total params: {total_params}\n")
    log.write(f"trainable params: {trainable_params}\n")
    log.close()

    with open(log_path, "a") as log:
        log.write(f"{conf}\n")
        print(conf)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr,
        weight_decay=conf["l2_reg"]
    )
    batch_cnt = len(dataset.train_loader)
    test_interval_bs = int(batch_cnt * conf["test_interval"])

    best_metrics, best_perform = init_best_metrics(conf)
    best_epoch = 0
    setup_seed(conf["seed"])
    num_epoch = conf['epochs'] if conf['epoch'] == -1 else conf["epoch"]
    
    # store information when training
    total_loss_history = [] 
    train_time_list = [] 

    # wandb
    if conf['use_wandb']:
        run_name = f"{conf['model']}-{conf['dataset']}-{datetime.now().strftime('%H%M%S')}"
        run_wandb = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="hoangggp-uet-vnu",
            # Set the wandb project where this run will be logged.
            project="bundle_construction_test",
            # Track hyperparameters and run metadata.
            config=conf,
            name=run_name
        )


    print(f'num of epoch: {num_epoch}')
    for epoch in range(num_epoch):
        start_train_time = time.time()
        total_test_time = [] 
        epoch_anchor = epoch * batch_cnt
        model.train(True)
        pbar = tqdm(
            enumerate(dataset.train_loader),
            total=len(dataset.train_loader)
        )
        avg_losses = {}
        grad_encoder_history = []
        grad_decoder_history = []
        grad_gat_encoder_history = []
        grad_gat_decoder_history = []
        sim_grad_history = []
        sim_grad_gat_history = []
        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            batch_anchor = epoch_anchor + batch_i

            losses = model(batch)

            losses['loss'].backward(retain_graph=True)

            # clip grad to prevent exploding gradient 
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            # analysis gradient flow 
            loss_analysis = model(batch)['loss']
            params_encoder, params_decoder = list(model.encoder.parameters()), list(model.decoder.parameters())
            # grad encoder-decoder
            # g_encoder = torch.autograd.grad(loss_analysis, params_encoder, retain_graph=True, allow_unused=True)
            # g_decoder = torch.autograd.grad(loss_analysis, params_decoder, retain_graph=True, allow_unused=True)
            # f_g_encoder = flat(g_encoder)
            # f_g_decoder = flat(g_decoder)
            # cos_sim = torch.dot(f_g_decoder, f_g_encoder) / (f_g_encoder.norm() * f_g_decoder.norm() + 1e-8)
            # grad_encoder_history.append(f_g_encoder)
            # grad_decoder_history.append(f_g_decoder)
            # sim_grad_history.append(cos_sim)    
            
            # grad gat 
            g_gat_encoder = torch.autograd.grad(loss_analysis, params_gat_encoder, retain_graph=True, allow_unused=True)
            g_gat_decoder = torch.autograd.grad(loss_analysis, params_gat_decoder, retain_graph=True, allow_unused=True)
            params_gat_encoder, params_gat_decoder = list(model.encoder.ii_modal_sim_gat.parameters()), list(model.decoder.ii_modal_sim_gat.parameters())
            f_g_gat_encoder = flat(g_gat_encoder)
            f_g_gat_decoder = flat(g_gat_decoder)
            cos_sim_gat = torch.dot(f_g_gat_decoder, f_g_gat_encoder) / (f_g_gat_encoder.norm() * f_g_gat_decoder.norm() + 1e-8)
            grad_gat_encoder_history.append(f_g_gat_encoder)
            grad_gat_decoder_history.append(f_g_gat_decoder)
            sim_grad_gat_history.append(cos_sim_gat)

            for l in losses:
                if l not in avg_losses:
                    avg_losses[l] = [losses[l].detach().cpu().item()]
                else:
                    avg_losses[l].append(losses[l].detach().cpu().item())

            loss_str = ", ".join([
                "%s: %.5f" % (l, losses[l].detach()) for l in losses
            ])
            pbar.set_description("epoch: %d, %s" % (epoch, loss_str))

            start_test_time = time.time()
            if (batch_anchor+1) % test_interval_bs == 0:
                metrics = {}
                metrics["val"], bundle_val_list, item_val_list, score_val_list = test(model, dataset.val_loader, conf)
                time_val = time.time() - start_test_time
                metrics["test"], bundle_test_list, item_test_list, score_test_list = test(model, dataset.test_loader, conf)
                time_test = time.time() - time_val - start_test_time
                
                best_metrics, best_perform, best_epoch, is_better = log_metrics(
                    conf, model, metrics, run, log_path, checkpoint_model_path, 
                    checkpoint_conf_path, epoch, batch_anchor, 
                    best_metrics, best_perform, best_epoch, 
                    bundle_test_list, item_test_list, score_test_list
                )

                # print(metrics["test"])
                if conf['use_wandb']:
                    for type_data in ['test', 'val']:
                        for type_metric in ['recall', 'ndcg']:
                            for topk in [5, 10, 20, 40, 80]:
                                run_wandb.log({
                                    f'{type_data}_{type_metric}@{topk}': metrics[type_data][type_metric][topk],
                                    f'best_{type_data}_{type_metric}@{topk}': best_metrics[type_data][type_metric][topk]
                                })

                # print(f'best metrics: {best_metrics}')


            test_time = time.time() - start_test_time
            total_test_time.append(test_time)
        train_time = time.time() - start_train_time - sum(total_test_time)
        train_time_list.append(train_time)

        total_loss_history.append(
            np.mean(avg_losses['loss'])
        )

        run_wandb.log({
            'total_loss': total_loss_history[-1]
        })

        # avg_grad_sim = torch.mean(torch.stack(sim_grad_history))
        # print(f'avg grad sim: {avg_grad_sim.item():.4f}')
        avg_grad_gat_sim = torch.mean(torch.stack(sim_grad_gat_history))
        print(f'avg grad gat sim: {avg_grad_gat_sim.item():.4f}')

        for l in avg_losses:
            run.add_scalar(l, np.mean(avg_losses[l]), epoch)
        avg_losses = {}
    
    # save information 
    # np.save(f"{log_path}/total_history_loss.npy", np.array(total_loss_history))
    # np.save(f"{log_path}/train_time_list.npy", np.array(train_time_list))

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
    
    # output predict list 
    bundle_list = []
    item_list = []
    score_list = []

    pbar = tqdm(dataloader, total=len(dataloader))
    for index, b_i_input, seq_b_i_input, b_i_gt in pbar:
        bundle_list.append(index)
        pred_i = model.evaluate(
            rs, (index.to(device), b_i_input.to(device), seq_b_i_input.to(device))
        )
        pred_i = pred_i - 1e8 * b_i_input.to(device)  # mask
        tmp_metrics = get_metrics(
            tmp_metrics, b_i_gt.to(device), pred_i, conf["topk"]
        )
        score, predict_list = torch.topk(pred_i, k=100)
        item_list.append(predict_list) 
        score_list.append(score)

    # convert to tensor
    bundle_list = torch.cat(bundle_list)
    item_list = torch.cat(item_list)
    score_list = torch.cat(score_list)

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics, bundle_list, item_list, score_list


if __name__ == "__main__":
    main()
