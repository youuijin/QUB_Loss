'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.nn.functional as F

import csv

import argparse
from datetime import datetime, timedelta
from torch.autograd import Variable

from utils.train_utils import *
from attack.pgd_attack import PGDAttack

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 PGD_Linf Training')

# env options
parser.add_argument('--env', type=int, default=0)

# model options
parser.add_argument('--model', choices=['resnet18', 'resnet34', 'preresnet18', 'wrn_28_10', 'wrn_34_10'], default='resnet18')

# dataset options
parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--normalize', choices=['none', 'twice', 'imagenet'], default='none')
parser.add_argument('--sche', choices=['multistep', 'cyclic'], default='cyclic')

# train options
parser.add_argument('--loss', choices=['CE', 'QUB'], default='CE')
parser.add_argument('--log_upper', default=False, action='store_true')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--device', type=int, default=0)

# attack options
parser.add_argument('--m', type=int, default=8)
parser.add_argument('--eps', type=float, default=8.)

# test options
parser.add_argument('--test_eps', type=float, default=8.)

args = parser.parse_args()

device = f'cuda:{args.device}'
best_acc, best_adv_acc = 0, 0  # best test accuracy

set_seed()
method = 'Free_AT'
cur = datetime.now().strftime('%m-%d_%H-%M')
# log_name = f'{method}(eps{args.eps}_m{args.m})_epoch{args.epoch}_lr{args.lr}_{args.normalize}_{cur}'
log_name = f'{args.loss}_{method}(eps{args.eps})_lr{args.lr}_{cur}'

# Summary Writer
writer = SummaryWriter(f'logs/{args.dataset}/{args.model}/env{args.env}/{log_name}')
if args.log_upper:
    upper_writer = SummaryWriter(f'upper_logs/{args.dataset}/{args.model}/env{args.env}/{log_name}/upper')
    real_writer = SummaryWriter(f'upper_logs/{args.dataset}/{args.model}/env{args.env}/{log_name}/real')


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
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)
# lr_steps = args.epoch * len(train_loader)
# if args.sche == 'multistep':
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_steps*0.5), int(lr_steps*0.75)], gamma=0.1)
# elif args.sche == 'cyclic':
#     lr_min = 0.0
#     scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_min, max_lr=args.lr,
#         step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_steps = args.epoch * len(train_loader)
if args.env == 1:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_steps*100/200), int(lr_steps*150/200)], gamma=0.1)
elif args.env == 2:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_steps*100/110), int(lr_steps*105/110)], gamma=0.1)
elif args.env == 3:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0, max_lr=0.1,
        step_size_up=lr_steps/2, step_size_down=lr_steps/2)
else: 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.8)], gamma=0.1)

def _label_smoothing(label, factor):
    one_hot = np.eye(10)[label.to(device).data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))
    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

# Train Attack & Test Attack
test_attack = PGDAttack(model, eps=args.test_eps, alpha=2., iter=10, mean=norm_mean, std=norm_std, device=device)

norm_mean = torch.tensor(norm_mean).to(device).view(1, 3, 1, 1)
norm_std = torch.tensor(norm_std).to(device).view(1, 3, 1, 1)
upper_limit = ((1 - norm_mean) / norm_std)
lower_limit = ((0 - norm_mean) / norm_std)
eps = args.eps/255./norm_std

global_noise = torch.zeros(args.batch_size, 3, 32, 32)
global_noise = global_noise.to(device)

# Train 1 epoch
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    real_adv_loss = 0
    global delta
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        for _ in range(args.m):
            optimizer.zero_grad()

            delta = Variable(global_noise[0:inputs.size(0)], requires_grad=True).to(device)
            advx = torch.clamp(inputs + delta, lower_limit, upper_limit)

            if args.loss == 'CE':
                outputs = model(advx)
                loss = F.cross_entropy(outputs, targets)  

            elif args.loss == 'QUB':
                outputs = model(inputs)
                softmax = F.softmax(outputs, dim=1)
                y_onehot = F.one_hot(targets, num_classes = softmax.shape[1])
                loss_natural = F.cross_entropy(outputs, targets, reduction='none')
                
                adv_outputs = model(advx)
                adv_norm = torch.norm(adv_outputs-outputs, dim=1)

                upper_loss = loss_natural + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + 0.5/2.0*torch.pow(adv_norm, 2)
                loss = upper_loss.mean()

            optimizer.zero_grad()
            loss.backward()

            grad = delta.grad
            global_noise[0:inputs.size(0)] += (eps * torch.sign(grad)).data
            global_noise.clamp_(-eps, eps)

            optimizer.step()
            scheduler.step() # if stepwise update

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if args.log_upper:
                real_adv_loss += F.cross_entropy(outputs, targets).item()
        
    writer.add_scalar('train/acc', 100.*correct/total, epoch)
    writer.add_scalar('train/loss', round(train_loss/total, 4), epoch)
    if args.log_upper:
        upper_writer.add_scalar(f'train/{log_name}', round(train_loss/total, 4), epoch)
        real_writer.add_scalar(f'train/{log_name}', round(real_adv_loss/total, 4), epoch)
    # print('train acc:', 100.*correct/total, 'train_loss:', round(train_loss/total, 4))

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
        # print('test acc:', 100.*correct/total, 'test adv acc:', 100.*adv_correct/total)
        writer.add_scalar('test/SA', 100.*correct/total, epoch)
        writer.add_scalar('test/RA', 100.*adv_correct/total, epoch)

    # Save checkpoint.
    adv_acc = 100.*adv_correct/total
    if adv_acc > best_adv_acc:
        torch.save(model.state_dict(), f'./env_models/env{args.env}/{args.model}_{log_name}.pt')
        best_adv_acc = adv_acc
        best_acc = 100.*correct/total
        best_epoch = epoch

print('start training..')

train_time = timedelta()
train_start = datetime.now()
for epoch in range(int(args.epoch/args.m)):
    start = datetime.now()
    train(epoch)
    train_time += datetime.now() - start
    test(epoch)
tot_time = datetime.now() - train_start

print('======================================')
print(f'best acc:{best_acc}%  best adv acc:{best_adv_acc}%  in epoch {best_epoch}')
if args.env>0:
    file_name = f'./csvs/env{args.env}/{args.model}.csv'
else:
    file_name = f'./{args.dataset}.csv'
with open(file_name, 'a', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    wr.writerow([f'{args.model}_{log_name}', args.model, method, best_acc, best_adv_acc, str(train_time).split(".")[0], str(tot_time).split(".")[0],])
