import argparse

def get_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_saved_model", action="store_true", help="test by using saved weight of model")

    parser.add_argument("-g", "--gpu", default="0",
                        type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="spotify",
                        type=str, help="which dataset to use")
    parser.add_argument("-m", "--model", default="",
                        type=str, help="which model to use")
    parser.add_argument("-i", "--info", default="", type=str,
                        help="any auxilary info that will be appended to the log file name")
    parser.add_argument("-l", "--lrs", nargs='+', default=[1e-3], type=float, help="Learning rate")
    parser.add_argument("-r", "--regs", nargs='+', default=[1e-5], type=float, help="weight decay")

    parser.add_argument("--item_augment", default="NA", type=str,
                        help="NA (No Augmentation), FD (Factor-wise Dropout), FN (Factor-wise Noise), MD (Modality-wise Noise)")
    parser.add_argument("--bundle_ratio", default=0.5, type=float,
                        help="the ratio of reserved items in a bundle, [0, 0.25, 0,5, 0.75, 1, 1.25, 1.5, 1.75, 2]")
    parser.add_argument("--bundle_augment", default="ID",
                        type=str, help="ID (Item Dropout), IR (Item Replacement)")
    parser.add_argument("--dropout_rate", default=0.2,
                        type=float, help="item-level dropout")
    parser.add_argument("--noise_weight", default=0.02,
                        type=float, help="item-level noise")
    parser.add_argument("--cl_temp", default=0.2, type=float,
                        help="tau for item-level contrastive learning")
    parser.add_argument("--cl_alpha", default=0, type=float,
                        help="alpha for item-level contrastive learning")
    parser.add_argument("--bundle_cl_temp", default=0.2, type=float,
                        help="tau for bundle-level contrastive learning")
    parser.add_argument("--bundle_cl_alpha", default=0.1, type=float,
                        help="alpha for bundle-level contrastive learning")
    parser.add_argument("--attention", default='', type=str,
                        help="wether to use layernorm or w_v")
    parser.add_argument("--trans_layer", default=1, type=int,
                        help="the number of layers for layernorm")
    parser.add_argument("--num_token", default=200, type=int,
                        help="the number of tokens (items in the bundle)")
    
    parser.add_argument("--seed", default=2023, type=int, help="")
    parser.add_argument("--epoch", default=-1, type=int, help="")
    parser.add_argument("--num_workers", default=4, type=int, help="num workers for dataloader")
    # parser.add_argument("--batch_size_train", default=256, type=int, help="batch size for training")
    # parser.add_argument("--batch_size_test", default=256, type=int, help="batch size for testing")

    # use wandb 
    # parser.add_argument("--wandb", default=0, type=int, help="")
    parser.add_argument("--use_wandb", action="store_true", help="flag for using wandb")
    parser.add_argument("--run_name_wandb", default="", type=str, help="run name for wandb")

    # graph 
    parser.add_argument("--use_modal_sim_graph", action="store_true", help="Enable modal similarity graph")
    parser.add_argument("--use_hyper_graph", action="store_true", help="Enable modal similarity graph")
    parser.add_argument("--num_layer_hypergraph", default=1, type=int, help="number of hyper graph layer")
    parser.add_argument("--num_layer_gats", nargs='+', default=[1], type=int, help="number of layer of gnn")
    parser.add_argument("--knn_ks", nargs='+', default=[10], type=int, help="k-knn for semantic graph")
    # argparse support prefix matching, so we can 
    # type type_gnn instead of type_gnns (very useful)
    parser.add_argument(
        "--type_gnns", 
        nargs='+', 
        default=["anti_symmetric", "light_gcn", "gat"], 
        type=str, 
        help="type of gnn for graph"
    )
    parser.add_argument("--k_gats", nargs='+', default=[5], type=int, help="k for gats")

    # iui graph 
    parser.add_argument("--use_iui_conv", action="store_true", help="enable iui graph_conv")

    # diffusion
    parser.add_argument("--use_diffusion", action="store_true", help="Enable modal similarity graph")
    parser.add_argument('--steps', type=int, default=20, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.01, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

    # diffusion for item-item graph
    parser.add_argument("--use_diff_graph", action="store_true", help="use diffusion item-item graph")

    # setting for ablation 
    
    # contrastive loss mode
    parser.add_argument("--use_cl", action="store_true", help="contrastive loss mode")

    # other 
    parser.add_argument("--use_pwc_fusion", action="store_true", help="use pwc fusion")
    parser.add_argument("--write_bi_predict_list", action="store_true", help="write bundle-item predict list")

    args = parser.parse_args()
    return args