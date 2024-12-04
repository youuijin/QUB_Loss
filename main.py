import argparse, csv
from utils.main_utils import set_seed, set_attack_hyperparam, set_trainer

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 PGD_Linf Training')

    # environment options
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)

    # path options
    parser.add_argument('--log_dir', type=str, default='./logs', help='path to log by Summarywriter')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='path to save checkpoint models, Error when the folder not defined')
    parser.add_argument('--csv_name', type=str, default='./csvs/results')

    # model options
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'preresnet18', 'wrn_28_10', 'wrn_34_10'])

    # dataset options
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny_imagenet'])

    # train options
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--valid_epoch', type=int, default=5, help='validation interval')
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cyclic', 'none'])
    parser.add_argument('--decay_epochs', type=str, default='70,85', help='if you choose multistep scheduler, put epochs to decay split by (,)')
    parser.add_argument('--up_and_down_epochs', type=str, default='15,15', help='if you choose cyclic scheduler, put number of epochs to go up and down split by (,)')
    parser.add_argument('--base_lr', type=float, default=0.0, help='learning rate')

    # Adversarial Training options
    parser.add_argument('--train_attack', type=str, default='None', 
                        choices=['None', 'Free', 'FGSM_RS', 'FGSM_GA', 'FGSM_CKPT', 'FGSM_PGI', 'FGSM_UAP', 'PGD_Linf', 'GAT', 'NuAT', 'TRADES'])
    parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'QUB'])
    parser.add_argument('--QUB_reg', type=float, default=0., help='if you want to use regularizer, set positive value')
    parser.add_argument('--QUB_func', type=str, default='linear', choices=['linear', 'const', 'acc'], help='if you want to use regularizer, set positive value')

    # 기본 파싱
    args, _ = parser.parse_known_args()

    # train_attack Hyperparameter
    parser = set_attack_hyperparam(args.train_attack, parser)

    # validation options
    parser.add_argument('--valid_eps', type=float, default=8.)

    args = parser.parse_args()
    
    return args

# OOP structure
if __name__ == '__main__':
    args = get_args()

    set_seed(seed=args.seed)
    trainer = set_trainer(args.train_attack, args)

    # train
    print("Start Training", trainer.log_name)
    best_acc, best_adv_acc, last_acc, last_adv_acc = trainer.train()

    print(f'Finish Training\nlog name: {trainer.log_name}\nbest acc:{best_acc}%  best adv acc:{best_adv_acc}% ')
    file_name = f'{args.csv_name}.csv'
    with open(file_name, 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([trainer.log_name, best_acc, best_adv_acc, last_acc, last_adv_acc])
