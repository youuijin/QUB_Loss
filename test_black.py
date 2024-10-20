import torch
import torch.optim as optim
import torch.nn.functional as F
import csv, os

import argparse

import torchattacks
from utils.train_utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Blackbox Attack Test using torchattacks')
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
parser.add_argument('--csv_name', type=str, default="./test_black.csv")

parser.add_argument('--env', type=int, default=0)

# attack options
parser.add_argument('--eps', type=float, default=8.0)

args = parser.parse_args()

device = f'cuda:{args.device}'
set_seed()

# make test list
if "seed" in args.saved_dir:
    seed_name = args.saved_dir.split('/')[-1]
    csv_name = f'./csvs/test/{args.dataset}_{args.model}_black_{seed_name}.csv'
else:
    seed_name = None
    csv_name = f'./csvs/test/{args.dataset}_{args.model}_black.csv'
tested_model_paths = []

f = open(csv_name, 'r', encoding='utf-8')
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

norm_mean, norm_std = (0, 0, 0), (1, 1, 1)

_, test_loader, n_way, imgsz = set_dataloader(args.dataset, args.batch_size, norm_mean, norm_std)

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
    print(model_path)
    if args.model not in model_path:
        continue
    if f"eps{args.eps}" not in model_path:
        if "no_AT" not in model_path:
            continue
    model.load_state_dict(torch.load(f'{args.saved_dir}/{model_path}', map_location=device))

    attack_success = np.zeros((5, 2))

    # RN18 natural model PGD50
    RN18_natural_model = set_model(model_name='resnet18', n_class=n_way)
    RN18_natural_model.load_state_dict(torch.load(f'./test_model/resnet18_{args.dataset}_natural.pt', map_location=device))
    RN18_natural_model.to(device)
    RN18_natural_model.eval()

    attack = torchattacks.PGD(RN18_natural_model, eps=args.eps/255, alpha=2/255, steps=50, random_start=True)

    print("Attack Method: RN18 natural model PGD50")
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
    attack_success[0][0] = 100. * sum(natural_acc)/len(test_loader.dataset)
    attack_success[0][1] = 100. * sum(robust_acc)/len(test_loader.dataset)
    print(f"natural accuracy: {attack_success[0][0]}, robust accuracy: {attack_success[0][1]}\n")

    del RN18_natural_model

    # RN18 PGD model PGD50
    RN18_PGD10_model = set_model(model_name='resnet18', n_class=n_way)
    RN18_PGD10_model.load_state_dict(torch.load(f'./test_model/resnet18_{args.dataset}_PGD10.pt', map_location=device))
    RN18_PGD10_model.to(device)
    RN18_PGD10_model.eval()

    attack = torchattacks.PGD(RN18_PGD10_model, eps=args.eps/255, alpha=2/255, steps=50, random_start=True)

    print("Attack Method: RN18 PGD10 model PGD50")
    natural_acc = []
    robust_acc = []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        nat_acc = accuracy(model, images, labels)
        natural_acc.append(nat_acc)
        adv_images = attack(images, labels)
        adv_images.to(device)
        rob_acc = accuracy(model, adv_images, labels)
        robust_acc.append(rob_acc)
    attack_success[1][0] = 100. * sum(natural_acc)/len(test_loader.dataset)
    attack_success[1][1] = 100. * sum(robust_acc)/len(test_loader.dataset)
    print(f"natural accuracy: {attack_success[1][0]}, robust accuracy: {attack_success[1][1]}\n")

    del RN18_PGD10_model

    # WRN34_10 natural model PGD50
    WRN34_10_natural_model = set_model(model_name='wrn_34_10', n_class=n_way)
    WRN34_10_natural_model.load_state_dict(torch.load(f'./test_model/wrn_34_10_{args.dataset}_natural.pt', map_location=device))
    WRN34_10_natural_model.to(device)
    WRN34_10_natural_model.eval()

    attack = torchattacks.PGD(WRN34_10_natural_model, eps=args.eps/255, alpha=2/255, steps=50, random_start=True)

    print("Attack Method: WRN34_10 natural model PGD50")
    natural_acc = []
    robust_acc = []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        nat_acc = accuracy(model, images, labels)
        natural_acc.append(nat_acc)
        adv_images = attack(images, labels)
        adv_images.to(device)
        rob_acc = accuracy(model, adv_images, labels)
        robust_acc.append(rob_acc)
    attack_success[2][0] = 100. * sum(natural_acc)/len(test_loader.dataset)
    attack_success[2][1] = 100. * sum(robust_acc)/len(test_loader.dataset)
    print(f"natural accuracy: {attack_success[2][0]}, robust accuracy: {attack_success[2][1]}\n")

    del WRN34_10_natural_model

    # WRN34_10 PGD model PGD50
    WRN34_10_PGD10_model = set_model(model_name='wrn_34_10', n_class=n_way)
    WRN34_10_PGD10_model.load_state_dict(torch.load(f'./test_model/wrn_34_10_{args.dataset}_PGD10.pt', map_location=device))
    WRN34_10_PGD10_model.to(device)
    WRN34_10_PGD10_model.eval()

    attack = torchattacks.PGD(WRN34_10_PGD10_model, eps=args.eps/255, alpha=2/255, steps=50, random_start=True)

    print("Attack Method: WRN34_10 PGD10 model PGD50")
    natural_acc = []
    robust_acc = []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        nat_acc = accuracy(model, images, labels)
        natural_acc.append(nat_acc)
        adv_images = attack(images, labels)
        adv_images.to(device)
        rob_acc = accuracy(model, adv_images, labels)
        robust_acc.append(rob_acc)
    attack_success[3][0] = 100. * sum(natural_acc)/len(test_loader.dataset)
    attack_success[3][1] = 100. * sum(robust_acc)/len(test_loader.dataset)
    print(f"natural accuracy: {attack_success[3][0]}, robust accuracy: {attack_success[3][1]}\n")

    del WRN34_10_PGD10_model
    
    # Square Attack
    attack = torchattacks.PGD(model, eps=args.eps/255)

    print("Attack Method: Square Attack")
    natural_acc = []
    robust_acc = []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        nat_acc = accuracy(model, images, labels)
        natural_acc.append(nat_acc)
        adv_images = attack(images, labels)
        adv_images.to(device)
        rob_acc = accuracy(model, adv_images, labels)
        robust_acc.append(rob_acc)
    attack_success[4][0] = 100. * sum(natural_acc)/len(test_loader.dataset)
    attack_success[4][1] = 100. * sum(robust_acc)/len(test_loader.dataset)
    print(f"natural accuracy: {attack_success[4][0]}, robust accuracy: {attack_success[4][1]}\n")

    with open(f'{csv_name}', 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([f'{model_path}', args.eps, f'env{args.env}', attack_success[0][0]] + [attack_success[i][1] for i in range(5)])