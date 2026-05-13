import argparse
import os
from utils.param_aug import ParamAugment


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_online_clients', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)
    parser.add_argument('--batch_size_local_training', type=int, default=64)
    parser.add_argument('--batch_real', type=int, default=100)
    parser.add_argument('--repeat_num', type=int, default=1)
    parser.add_argument('--rounds_of_hp', type=int, default=30)
    parser.add_argument('--length_of_hp', type=int, default=5)
    parser.add_argument('--hp_info_t', type=float, default=0.05, help='tem for local training')
    parser.add_argument('--lr_hp', type=float, default=0.1, help='learning rate for hp')
    parser.add_argument('--meta_length', type=int, default=50)
    parser.add_argument('--batch_size_test', type=int, default=100)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=536)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--use_margin', type=bool, default=False)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--dis_metric', type=str, default='matching', help='gradient matching')
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--method', type=str, default='DSA', help='data aug')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate')
    
    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)
    args = parser.parse_args()
    args.dsa_param = ParamAugment()
    args.dsa = True if args.method == 'DSA' else False
    
    return args