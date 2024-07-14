'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.nn.functional as F

import csv

import argparse
from datetime import datetime, timedelta
import time

from utils.train_utils import *
from attack.pgd_attack import PGDAttack
from attack.trades_attack import TRADESAttack

from torch.utils.tensorboard import SummaryWriter

import torch.autograd as autograd
autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 TRADES Training')
# model options
parser.add_argument('--model', choices=['resnet18', 'resnet34', 'preresnet18', 'wrn_28_10', 'wrn_34_10'], default='resnet18')

# dataset options
parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--normalize', choices=['none', 'twice', 'imagenet'], default='none')

# train options
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--sche', default='multistep', type=str, help='learning rate scheduler')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--device', type=int, default=0)

# attack options
parser.add_argument('--beta', type=float, default=6.)
parser.add_argument('--num_step', type=int, default=10)
parser.add_argument('--eps', type=float, default=8.)
parser.add_argument('--alpha', type=float, default=2.) # step size

args = parser.parse_args()

device = f'cuda:{args.device}'
best_acc, best_adv_acc = 0, 0  # best test accuracy

set_seed()
method = 'TRADES'
cur = datetime.now().strftime('%m-%d_%H-%M')
log_name = f'{method}(eps{args.eps}_iter{args.num_step})_beta{args.beta}_epoch{args.epoch}_{args.normalize}_{cur}'

# Summary Writer
writer = SummaryWriter(f'logs/{args.dataset}/{args.model}/{log_name}')

# Data
print('==> Preparing data..')

if args.normalize == "imagenet":
    norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
elif args.normalize == "twice":
    norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
else: 
    norm_mean, norm_std = (0, 0, 0), (1, 1, 1)

train_loader, test_loader, n_way = set_dataloader(args.dataset, args.batch_size, norm_mean, norm_std)

# Model
print('==> Building model..')
model = set_model(model_name=args.model, n_class=n_way)
model = model.to(device)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.8)], gamma=0.1)

# Loss function
criterion_kl = nn.KLDivLoss(size_average=False)

# Train Attack & Test Attack
attack = TRADESAttack(model, eps=args.eps, alpha=args.alpha, iter=args.num_step, mean=norm_mean, std=norm_std, device=device)
test_attack = PGDAttack(model, eps=8., alpha=2., iter=10, mean=norm_mean, std=norm_std, device=device)

# Train 1 epoch
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        adv_inputs = attack.perturb(inputs, targets)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss_natural = F.cross_entropy(outputs, targets)
        loss_robust = (1.0 / inputs.shape[0]) * criterion_kl(F.log_softmax(model(adv_inputs), dim=1), F.softmax(model(inputs), dim=1)+1e-10)
        loss = loss_natural + args.beta * loss_robust

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    writer.add_scalar('train/acc', 100.*correct/total, epoch)
    writer.add_scalar('train/loss', round(train_loss/total, 4), epoch)
    print('train acc:', 100.*correct/total, 'train_loss:', round(train_loss/total, 4))

def test(epoch):
    global best_acc
    global best_adv_acc
    global best_epoch
    model.eval()
    correct, adv_correct = 0, 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            adv_inputs = test_attack.perturb(inputs, targets)
            adv_outputs = model(adv_inputs)
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()

            total += targets.size(0)
        print('test acc:', 100.*correct/total, 'test adv acc:', 100.*adv_correct/total)
        writer.add_scalar('test/SA', 100.*correct/total, epoch)
        writer.add_scalar('test/RA', 100.*adv_correct/total, epoch)

    # Save checkpoint.
    adv_acc = 100.*adv_correct/total
    if adv_acc > best_adv_acc:
        torch.save(model.state_dict(), f'./saved_models/{args.model}_{log_name}.pt')
        best_adv_acc = adv_acc
        best_acc = 100.*correct/total
        best_epoch = epoch

print('start training..')


train_time = timedelta()
train_start = datetime.now()
for epoch in range(args.epoch):
    start = datetime.now()
    train(epoch)
    train_time += datetime.now() - start
    if epoch%10 == 0:
        test(epoch)
    if args.sche == 'multistep':
        scheduler.step()
tot_time = datetime.now() - train_start


print('======================================')
print(f'best acc:{best_acc}%  best adv acc:{best_adv_acc}%  in epoch {best_epoch}')
with open(f'./{args.dataset}.csv', 'a', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    wr.writerow([f'{args.model}_{log_name}', args.model, method, best_acc, best_adv_acc, str(train_time).split(".")[0], str(tot_time).split(".")[0],])
