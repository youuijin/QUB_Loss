import torch
import torch.optim as optim
import torch.nn.functional as F
import csv, os

import argparse

import torchattacks
from utils.train_utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test using torchattacks')
# model options
parser.add_argument('--random_seed', type=int, default=706)
parser.add_argument('--model', choices=['resnet18', 'resnet34', 'preresnet18', 'wrn_28_10', 'wrn_34_10'], default='resnet18')

# dataset options
parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--normalize', choices=['none', 'twice', 'imagenet', 'cifar'], default='none')

# test options
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--saved_dir', type=str, default="./saved_models")
parser.add_argument('--csv_name', type=str, default="./test.csv")

# attack options
parser.add_argument('--eps', type=float, default=8.)

args = parser.parse_args()

device = f'cuda:{args.device}'
set_seed()

# make test list
tested_model_paths = []
f = open(args.csv_name, 'r', encoding='utf-8')
rdr = csv.reader(f)
next(rdr)
for line in rdr:
    tested_model_paths.append(line[0].strip())
f.close() 

def check_is_tested(model_path):
    if model_path in tested_model_paths:
        return True
    else:
        return False
    
if args.model_path != "":
    if check_is_tested(args.model_path+".pt"):
        raise ValueError("This model is already tested")
    model_paths = [f'{args.model_path}.pt']
else: 
    exist_model_paths = os.listdir(f'{args.saved_dir}/')
    model_paths = []
    
    for exist_model_path in exist_model_paths:
        if not check_is_tested(exist_model_path):
            model_paths.append(exist_model_path)

if args.normalize == "imagenet":
    norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
elif args.normalize == "cifar":
    norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
elif args.normalize == "twice":
    norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
else: 
    norm_mean, norm_std = (0, 0, 0), (1, 1, 1)

_, test_loader, n_way = set_dataloader(args.dataset, args.batch_size, norm_mean, norm_std)

def accuracy(model, inputs, labels):
    net_out = model(inputs)
    pred = net_out.data.max(1)[1]
    correct = pred.eq(labels.data).sum()
    return correct.item()

# Model Loading
print('==> Building model..')
model = set_model(model_name=args.model, n_class=n_way)
model = model.to(device)
model.eval()

# Test
for model_path in model_paths:
    model.load_state_dict(torch.load(f'{args.saved_dir}/{model_path}', map_location=device))
    # attack.set_normalization_used(mean=list(preprocessing['mean']), std=list(preprocessing['std']))
    attacks = [
        torchattacks.PGD(model, eps=args.eps/255, alpha=2/255, steps=10, random_start=True), # PGD10
        torchattacks.PGD(model, eps=args.eps/255, alpha=2/255, steps=20, random_start=True), # PGD20
        torchattacks.MultiAttack([torchattacks.PGD(model, eps=args.eps/255, alpha=2/255, steps=50, random_start=True)]*10), # PGD50-10
        torchattacks.AutoAttack(model, norm='Linf', eps=args.eps/255, version='standard', n_classes=n_way, seed=args.random_seed, verbose=False) # Auto Attack
    ]
    attack_success = np.zeros((len(attacks), 2))
    for i, attack in enumerate(attacks):
        print(f"Attack Method: {attack}")
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

with open(f'{args.csv_name}', 'a', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    wr.writerow([f'{args.model_path}', attack_success[0][0]] + [attack_success[i][1] for i in range(4)])
