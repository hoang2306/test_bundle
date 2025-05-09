import argparse

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2023, type=int, help="")
    parser.add_argument("--epoch", default=-1, type=int, help="")
    parser.add_argument("--wandb", default=0, type=int, help="")

    parser.add_argument("--num_workers", default=4, type=int, help="num workers for dataloader")

    # parser.add_argument("--use_modal_sim_graph", default=False, type=bool, help="")
    parser.add_argument("--use_modal_sim_graph", action="store_true", help="Enable modal similarity graph")

    args = parser.parse_args()
    return args

conf = get_cmd().__dict__
print(conf)

if conf['use_modal_sim_graph']:
    print('use modal graph')
else:
    print('no using')