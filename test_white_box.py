import torch
import csv, os
import pandas as pd
from tqdm import tqdm

import argparse

import torchattacks
from utils.train_utils import *

# def check_is_tested(model_path, tested_model_paths):
#     # if model_path in tested_model_paths:
#     #     return True
#     # else:
#     #     return False
    
    
def accuracy(model, inputs, targets):
    outputs = model(inputs)
    pred = outputs.data.max(1)[1]
    correct = pred.eq(targets.data).sum().item()
    return correct

def test(args):
    device = f'cuda:{args.device}'
    set_seed(seed=args.random_seed) # fix test seed

    df = pd.read_csv('./csvs/train_status.csv')
    filtered_df = df[(df['dataset'] == args.dataset) & (df['seed'].astype(str) == args.seed) & (df['model']==args.model)]

    test_paths = []
    for _, row in filtered_df.iterrows():
        if pd.notnull(row['save_dir']) and pd.isnull(row['SA']):
            # print(row['save_dir'], row['model_name'])
            test_paths += [[row['save_dir'], row['model_name']]]

    norm_mean, norm_std = (0, 0, 0), (1, 1, 1)

    _, test_loader, n_way, imgsz = set_dataloader(args.dataset, args.batch_size, norm_mean, norm_std)

    # Model Loading
    print('==> Building model..')
    model = set_model(model_name=args.model, n_class=n_way)
    model = model.to(device)
    model.eval()

    # Test
    for idx, (dir_name, model_name) in enumerate(test_paths):
        print(f'Test [{idx+1}/{len(test_paths)}]:', f'{dir_name}/{model_name}')
        # EPS. to args.eps
        model.load_state_dict(torch.load(f'{dir_name}/{model_name}', map_location=device))
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
            if i<3:
                continue
            natural_acc = []
            robust_acc = []
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                nat_acc = accuracy(model, images, labels)
                natural_acc.append(nat_acc)
                # check the meaning of each term at: https://foolbox.jonasrauber.de/guide/getting-started.html#multiple-epsilons
                adv_images = attack(images, labels)
                adv_images.to(device)
                rob_acc = accuracy(model, adv_images, labels)
                robust_acc.append(rob_acc)
                print(np.array(rob_acc).mean())
            attack_success[i][0] = 100. * sum(natural_acc)/len(test_loader.dataset)
            attack_success[i][1] = 100. * sum(robust_acc)/len(test_loader.dataset)
            print(f"natural accuracy: {attack_success[i][0]}, robust accuracy: {attack_success[i][1]}")
        # print()

        with open(f'./csvs/test_last/{args.dataset}_{args.model}_seed{args.seed}.csv', 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([model_name, args.eps, attack_success[0][0]] + [attack_success[i][1] for i in range(4)])

        # df.loc[(df['save_dir'] == dir_name) & (df['model_name'] == model_name), 'SA'] = attack_success[0][0]
        # for att, acc in zip(attack_names, attack_success[:][1]):
        #     df.loc[(df['save_dir'] == dir_name) & (df['model_name'] == model_name), att] = acc

    # 수정된 데이터프레임 저장
    # df.to_csv('./csvs/train_status_temp.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Whitebox Test using Auto Attack')
    # model options
    parser.add_argument('--seed', type=str, default='none')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--model', choices=['resnet18', 'resnet34', 'preresnet18', 'wrn_28_10', 'wrn_34_10'], default='none')
    # parser.add_argument('--dir_name', default='csvs/test_final')

    # dataset options
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'tiny_imagenet'], default='none')

    # test options
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    # attack options
    parser.add_argument('--eps', type=float, default=8.0)

    args = parser.parse_args()

    test(args)