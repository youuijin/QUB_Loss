import random, torch
import numpy as np

from Trainers import *

def set_seed(seed=706):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_attack_hyperparam(train_attack, parser):
    if train_attack == 'Free':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--m', type=int, default=8)
    elif train_attack == 'FGSM_RS':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--alpha', type=float, default=10.)
    elif train_attack == 'FGSM_GA':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--lamb', type=float, default=0.2)
    elif train_attack == 'FGSM_CKPT':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--alpha', type=float, default=10.)
        parser.add_argument('--c', type=int, default=3)
    elif train_attack == 'FGSM_PGI':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--alpha', type=float, default=8.)
        parser.add_argument('--epoch_reset', type=int, default=40)
        parser.add_argument('--momentum_decay', type=float, default=0.3)
        parser.add_argument('--lamb', type=float, default=10.)
    elif train_attack == 'FGSM_UAP':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--uap_eps', type=float, default=10.)
        parser.add_argument('--lamb', type=float, default=10.)
        parser.add_argument('--uap_num', type=int, default=50)
    elif train_attack == 'PGD_Linf':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--alpha', type=float, default=2.)
        parser.add_argument('--iter', type=int, default=10)
        parser.add_argument('--restart', type=int, default=1)
    elif train_attack == 'GAT':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--alpha', type=float, default=4.)
        parser.add_argument('--lamb', type=float, default=10.)
        parser.add_argument('--reg_mul', type=float, default=4.)
    elif train_attack == 'NuAT':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--alpha', type=float, default=4.)
        parser.add_argument('--nuc_reg', type=float, default=2.5)
    elif train_attack == 'TRADES':
        parser.add_argument('--eps', type=float, default=8.)
        parser.add_argument('--alpha', type=float, default=2.)
        parser.add_argument('--iter', type=int, default=10)
        parser.add_argument('--beta', type=float, default=6.)

    return parser

def set_trainer(train_attack, args):
    if train_attack == 'None':
        trainer = Natural_Trainer(args)
    elif train_attack == 'Free':
        trainer = Free_Trainer(args)
    elif train_attack == 'FGSM_RS':
        trainer = FGSM_RS_Trainer(args)
    elif train_attack == 'FGSM_GA':
        trainer = FGSM_GA_Trainer(args)
    elif train_attack == 'FGSM_CKPT':
        trainer = FGSM_CKPT_Trainer(args)
    elif train_attack == 'FGSM_PGI':
        trainer = FGSM_PGI_Trainer(args)
    elif train_attack == 'FGSM_UAP':
        trainer = FGSM_UAP_Trainer(args)
    elif train_attack == 'PGD_Linf':
        trainer = PGD_Linf_Trainer(args)
    elif train_attack == 'GAT':
        trainer = GAT_Trainer(args)
    elif train_attack == 'NuAT':
        trainer = NuAT_Trainer(args)
    elif train_attack == 'TRADES':
        trainer = TRADES_Trainer(args)

    return trainer