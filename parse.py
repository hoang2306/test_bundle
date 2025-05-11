import argparse

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0",
                        type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="spotify",
                        type=str, help="which dataset to use")
    parser.add_argument("-m", "--model", default="",
                        type=str, help="which model to use")
    parser.add_argument("-i", "--info", default="", type=str,
                        help="any auxilary info that will be appended to the log file name")
    parser.add_argument("-l", "--lr", default=1e-3,
                        type=float, help="Learning rate")
    parser.add_argument("-r", "--reg", default=1e-5,
                        type=float, help="weight decay")

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
    parser.add_argument("--wandb", default=0, type=int, help="")

    parser.add_argument("--num_workers", default=4, type=int, help="num workers for dataloader")

    # parser.add_argument("--use_modal_sim_graph", default=False, type=bool, help="")
    parser.add_argument("--use_modal_sim_graph", action="store_true", help="Enable modal similarity graph")
    parser.add_argument("--use_hyper_graph", action="store_true", help="Enable modal similarity graph")

    args = parser.parse_args()
    return args