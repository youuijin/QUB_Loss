import torch
import csv, os

import argparse

import torchattacks
from utils.train_utils import *

def check_is_tested(model_path, tested_model_paths):
    if model_path in tested_model_paths:
        return True
    else:
        return False
    
def accuracy(model, inputs, targets):
    outputs = model(inputs)
    pred = outputs.data.max(1)[1]
    correct = pred.eq(targets.data).sum().item()
    return correct

def test(args):
    device = f'cuda:{args.device}'
    set_seed(seed=args.random_seed) # fix test seed

    # make test list
    dir_name = args.dir_name
    saved_dir = f'{args.saved_dir}/{args.dataset}/{args.model}/env1/seed{args.seed}'
    csv_name = f'{args.dataset}_{args.model}_seed{args.seed}'
    tested_model_paths = []

    f = open(f'{dir_name}/{csv_name}.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    next(rdr)
    for line in rdr:
        tested_model_paths.append(line[0].strip())
    f.close() 

    if args.model_path != "":
        if check_is_tested(args.model_path+".pt", tested_model_paths):
            raise ValueError("This model is already tested")
        model_paths = [f'{args.model_path}.pt']
    else: 
        exist_model_paths = os.listdir(saved_dir)
        model_paths = []
        
        for exist_model_path in exist_model_paths:
            # 학습한 것 중, test하지 않은 모델 불러오기
            if not check_is_tested(exist_model_path, tested_model_paths):
                model_paths.append(exist_model_path)

    norm_mean, norm_std = (0, 0, 0), (1, 1, 1)

    _, test_loader, n_way, imgsz = set_dataloader(args.dataset, args.batch_size, norm_mean, norm_std)

    # Model Loading
    print('==> Building model..')
    model = set_model(model_name=args.model, n_class=n_way)
    model = model.to(device)
    model.eval()

    # Test
    for model_path in model_paths:
        if 'best' not in model_path:
            continue # only test best model
        
        print(model_path)
        model.load_state_dict(torch.load(f'{saved_dir}/{model_path}', map_location=device))
        # attack.set_normalization_used(mean=list(preprocessing['mean']), std=list(preprocessing['std']))
        attacks = [
            torchattacks.PGD(model, eps=args.eps/255, alpha=2/255, steps=10, random_start=True), # PGD10
            torchattacks.PGD(model, eps=args.eps/255, alpha=2/255, steps=20, random_start=True), # PGD20
            torchattacks.MultiAttack([torchattacks.PGD(model, eps=args.eps/255, alpha=2/255, steps=50, random_start=True)]*10), # PGD50-10
            torchattacks.AutoAttack(model, norm='Linf', eps=args.eps/255, version='standard', n_classes=n_way, seed=args.random_seed, verbose=False) # Auto Attack
        ]
        attack_names = ['PGD10', 'PGD20', 'PGD-50-10', 'AA']
        attack_success = np.zeros((len(attacks), 2))
        for i, attack in enumerate(attacks):
            print(f"Attack Method: {attack_names[i]}")
            natural_acc = []
            robust_acc = []
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                nat_acc = accuracy(model, images, labels)
                natural_acc.append(nat_acc)
                # check the meaning of each term at: https://foolbox.jonasrauber.de/guide/getting-started.html#multiple-epsilons
                adv_images = attack(images, labels)
                adv_images.to(device)
                rob_acc = accuracy(model, adv_images, labels)
                robust_acc.append(rob_acc)
            attack_success[i][0] = 100. * sum(natural_acc)/len(test_loader.dataset)
            attack_success[i][1] = 100. * sum(robust_acc)/len(test_loader.dataset)
            print(f"natural accuracy: {attack_success[i][0]}, robust accuracy: {attack_success[i][1]}")
        # print()

        with open(f'{dir_name}/{csv_name}.csv', 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([f'{model_path}', args.eps, attack_success[0][0]] + [attack_success[i][1] for i in range(4)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Whitebox Test using Auto Attack')
    # model options
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--model', choices=['resnet18', 'resnet34', 'preresnet18', 'wrn_28_10', 'wrn_34_10'], default='resnet18')
    parser.add_argument('--dir_name', default='csvs/test_final')

    # dataset options
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'tiny_imagenet'], default='cifar10')

    # test options
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--saved_dir', type=str, default="./saved_models_final")

    # attack options
    parser.add_argument('--eps', type=float, default=8.0)

    args = parser.parse_args()

    test(args)