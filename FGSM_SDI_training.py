'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import csv

import argparse
from datetime import datetime, timedelta
import time

from utils.train_utils import *
from attack.pgd_attack import PGDAttack

from utils.model.FGSM_SDI_attacker import One_Layer_Attacker

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 FGSM-SDI Training')

# env options
parser.add_argument('--env', type=int, default=0)
parser.add_argument('--seed', type=int, default=706)

# model options
parser.add_argument('--model', choices=['resnet18', 'resnet34', 'preresnet18', 'wrn_28_10', 'wrn_34_10'], default='resnet18')

# dataset options
parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'tiny_imagenet'], default='cifar10')
parser.add_argument('--normalize', choices=['none', 'twice', 'imagenet', 'cifar'], default='none')

# train options
parser.add_argument('--loss', choices=['CE', 'QUB'], default='CE')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=110)
parser.add_argument('--device', type=int, default=0)

# attack options
parser.add_argument('--eps', type=float, default=8.)
parser.add_argument('--lr_att', type=float, default=0.001)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--factor', default=0.5, type=float)
parser.add_argument('--K', type=float, default=0.5)

# test options
parser.add_argument('--test_eps', type=float, default=8.)

# Logger options
parser.add_argument('--log_upper', default=False, action='store_true')
parser.add_argument('--log_K', default=False, action='store_true')
parser.add_argument('--param_grad_norm', default=False, action='store_true')
parser.add_argument('--input_grad_norm', default=False, action='store_true')

args = parser.parse_args()

device = f'cuda:{args.device}'
best_acc, best_adv_acc = 0, 0  # best test accuracy
last_acc, last_adv_acc = 0, 0

set_seed(seed=args.seed)
method = 'FGSM_SDI'
cur = datetime.now().strftime('%m-%d_%H-%M')
# log_name = f'{args.loss}_{method}(eps{args.eps}_k{args.k})_epoch{args.epoch}_{args.normalize}_{args.factor}_{cur}'
log_name = f'{method}(eps{args.eps}_k{args.k})_{args.loss}_lr{args.lr}_att{args.lr_att}_{cur}'
if args.loss == 'QUB':
    log_name = f'{method}(eps{args.eps}_k{args.k})_{args.loss}(K{args.K})_lr{args.lr}_att{args.lr_att}_{cur}'

# Summary Writer
if not args.input_grad_norm:
    writer = SummaryWriter(f'logs/{args.dataset}/{args.model}/env{args.env}/seed{args.seed}/{log_name}')
else:
    writer = SummaryWriter(f'grad_norm_logs/{args.dataset}/{args.model}/env{args.env}/{log_name}')
if args.loss=='QUB' and args.log_upper:
    upper_writer = SummaryWriter(f'upper_logs/{args.dataset}/{args.model}/env{args.env}/{log_name}/upper')
    real_writer = SummaryWriter(f'upper_logs/{args.dataset}/{args.model}/env{args.env}/{log_name}/real')

# Data
print('==> Preparing data..')

if args.normalize == "imagenet":
    norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
elif args.normalize == "cifar":
    norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
elif args.normalize == "twice":
    norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
else: 
    norm_mean, norm_std = (0, 0, 0), (1, 1, 1)

train_loader, test_loader, n_way, imgsz = set_dataloader(args.dataset, args.batch_size, norm_mean, norm_std)

# Model
print('==> Building model..')
model = set_model(model_name=args.model, n_class=n_way)
model = model.to(device)

# Train Attack & Test Attack
attacker = One_Layer_Attacker(eps=args.eps/255., input_channel=6).to(device)
test_attack = PGDAttack(model, eps=args.test_eps, alpha=args.test_eps/4., iter=10, mean=norm_mean, std=norm_std, device=device)

# Optimizer
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 105], gamma=0.1)

# optimizer_att = torch.optim.SGD(attacker.parameters(), lr=args.lr_att, momentum=0.9,
#                               weight_decay=5e-4)
# # optimizer_att = optimizer_att = torch.optim.Adam(attacker.parameters(), lr=args.lr_att, weight_decay=5e-4)
# attacker_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_att,
#                                                       milestones=[100, 105],
#                                                       gamma=0.1)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_att = torch.optim.SGD(attacker.parameters(), lr=args.lr_att, momentum=0.9, weight_decay=5e-4)
lr_steps = args.epoch * len(train_loader)
if args.env == 1:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_steps*100/200), int(lr_steps*150/200)], gamma=0.1)
    attacker_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_att, milestones=[int(lr_steps*100/200), int(lr_steps*150/200)], gamma=0.1)
elif args.env == 2:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_steps*100/110), int(lr_steps*105/110)], gamma=0.1)
    attacker_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_att, milestones=[int(lr_steps*100/110), int(lr_steps*105/110)], gamma=0.1)
elif args.env == 3:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0, max_lr=args.lr, step_size_up=lr_steps/2, step_size_down=lr_steps/2)
    attacker_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_att, base_lr=0.0, max_lr=args.lr_att, step_size_up=lr_steps/2, step_size_down=lr_steps/2)
elif args.env == 4:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_steps*70/100), int(lr_steps*85/100)], gamma=0.1)
    attacker_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_att, milestones=[int(lr_steps*70/100), int(lr_steps*85/100)], gamma=0.1)
else: 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.8)], gamma=0.1)
    attacker_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_att,
                                                          milestones=[100, 105],
                                                          gamma=0.1)

mean = torch.tensor(norm_mean).to(device).view(1, 3, 1, 1)
std = torch.tensor(norm_std).to(device).view(1, 3, 1, 1)
upper_limit = ((1 - mean) / std)
lower_limit = ((0 - mean) / std)
eps = args.eps/255./std
# alpha = args.alpha/255./std

tot_K, max_K = 0, 0

def adv_FGSM_loss(grad,
                x_natural,
                y,
                for_attacker = 0):
    global tot_K, max_K
    if for_attacker == 0:
        model.train()
        attacker.eval()
    else:
        attacker.train()

    advinput = torch.cat([x_natural, 1.0 * (torch.sign(grad))], 1).detach()

    # generate adversarial example
    perturbation = attacker(advinput)

    x_adv = x_natural + perturbation
    x_adv = torch.clamp(x_adv, lower_limit, upper_limit)

    x_adv.requires_grad_()

    with torch.enable_grad():
        loss_adv = F.cross_entropy(model(x_adv), y)
        grad_adv = torch.autograd.grad(loss_adv, [x_adv])[0]
        perturbation_1 = torch.clamp(eps * torch.sign(grad_adv), -eps, eps)

    perturbation_total = perturbation + perturbation_1
    perturbation_total = torch.clamp(perturbation_total, -eps, eps)

    x_adv_final = x_natural + perturbation_total
    x_adv_final = torch.clamp(x_adv_final, lower_limit, upper_limit)

    adv_outputs = model(x_adv_final)
    # default : Label Smoothing Loss -> Change to CE Loss
    # loss_robust = LabelSmoothLoss(output, label_smoothing.float())

    grad_norm_list = []

    if args.loss == 'CE':
        if args.input_grad_norm:
            outputs = model(x_natural)
            softmax = F.softmax(outputs, dim=1)
            y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        loss_robust = F.cross_entropy(adv_outputs, y)
    elif args.loss == 'QUB':
        outputs = model(x_natural)
        softmax = F.softmax(outputs, dim=1)
        if args.log_K:
            K_values = calc_K(softmax)
            tot_K += K_values.sum().item()
            if K_values.max().item()>max_K:
                max_K = K_values.max().item()
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        loss_natural = F.cross_entropy(outputs, y, reduction='none')
        
        # adv_outputs = model(advx)
        adv_norm = torch.norm(adv_outputs-outputs, dim=1)

        upper_loss = loss_natural + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + args.K/2.0*torch.pow(adv_norm, 2)
        if args.K<0:
                upper_loss = loss_natural + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + K_values/2.0*torch.pow(adv_norm, 2)

        loss_robust = upper_loss.mean()

    loss = loss_robust
    if args.input_grad_norm:
        grad_norm_list = [outputs, softmax, y_onehot]
    return loss, adv_outputs, grad_norm_list

# Train 1 epoch
def train(epoch):
    train_loss = 0
    correct = 0
    total = 0
    real_adv_loss = 0
    tot_grad_norm = 0
    global tot_K, max_K
    tot_K, max_K = 0, 0
    grad_norm_list, cos_sim = [0, 0, 0], []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad_()
        with torch.enable_grad():
            loss_natural = F.cross_entropy(model(inputs), targets)
        grad = torch.autograd.grad(loss_natural, [inputs])[0]

        # for _ in range(args.att_iter):
        if batch_idx % args.k==0:
            optimizer_att.zero_grad()

            attacker.zero_grad()
            model.zero_grad()
            loss_adv, _, _ = adv_FGSM_loss(grad=grad,
                                        x_natural=inputs,
                                        y=targets,
                                        for_attacker=1)

            loss_adv=-loss_adv

            loss_adv.backward()
            optimizer_att.step()

        optimizer.zero_grad()
        model.zero_grad()
        attacker.zero_grad()

        # calculate robust loss
        loss, adv_outputs, others = adv_FGSM_loss(grad=grad,                                
                                    x_natural=inputs,
                                    y=targets,
                                    for_attacker=0)
        loss.backward()
        if args.param_grad_norm:
            grad_norm = get_grad_norm(model.parameters(), norm_type=2)
            tot_grad_norm += grad_norm.item()
        optimizer.step()
        
        train_loss += loss.item()*targets.size(0)
        _, predicted = adv_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.loss=='QUB' and args.log_upper:
            real_adv_loss += F.cross_entropy(adv_outputs, targets).item()

        if args.input_grad_norm:
            outputs, softmax, y_onehot = others
            grad_norm_list[0] += input_loss_norm(model, inputs, targets).sum().item()
            grad_norm_list[1] += input_logit_norm(model, inputs, targets).sum().item()
            grad_norm_list[2] += logit_loss_norm(model, inputs, targets).sum().item()
            cos_sim += F.cosine_similarity((adv_outputs-outputs), (softmax-y_onehot), dim=1).tolist()

        scheduler.step()
        attacker_scheduler.step()

    writer.add_scalar('train/acc', 100.*correct/total, epoch)
    writer.add_scalar('train/loss', round(train_loss/total, 4), epoch)
    if args.loss=='QUB' and args.log_upper:
        upper_writer.add_scalar(f'train/{log_name}', round(train_loss/total, 4), epoch)
        real_writer.add_scalar(f'train/{log_name}', round(real_adv_loss/total, 4), epoch)
    if args.loss=='QUB' and args.log_K:
        writer.add_scalar('train/Mean_K', round(tot_K/total, 4), epoch)
        writer.add_scalar('train/Max_K', max_K, epoch)
    if args.param_grad_norm:
        writer.add_scalar('train/grad_norm', tot_grad_norm, epoch)
    if args.input_grad_norm:
        writer.add_scalar('grad_norm/input_loss', round(grad_norm_list[0]/total, 4), epoch)
        writer.add_scalar('grad_norm/input_logit', grad_norm_list[1]/total, epoch)
        writer.add_scalar('grad_norm/logit_loss', round(grad_norm_list[2]/total, 4), epoch)
        writer.add_scalar('cos_sim/mean', np.mean(cos_sim), epoch)
        writer.add_scalar('cos_sim/std', np.std(cos_sim), epoch)
        writer.add_scalar('cos_sim/min', np.min(cos_sim), epoch)
        writer.add_scalar('cos_sim/max', np.max(cos_sim), epoch)
    # print('train acc:', 100.*correct/total, 'train_loss:', round(train_loss/total, 4))

def test(epoch):
    global best_acc, best_adv_acc, best_epoch
    global last_acc, last_adv_acc
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
        if not args.input_grad_norm:
            torch.save(model.state_dict(), f'./env_models/env{args.env}/{args.dataset}/seed{args.seed}/{args.model}_{log_name}_best.pt')
        best_adv_acc = adv_acc
        best_acc = 100.*correct/total
        best_epoch = epoch
    last_acc = 100.*correct/total
    last_adv_acc = 100.*adv_correct/total
    torch.save(model.state_dict(), f'./env_models/env{args.env}/{args.dataset}/seed{args.seed}/{args.model}_{log_name}_last.pt')

print('start training..')

train_time = timedelta()
train_start = datetime.now()
for epoch in range(args.epoch):
    start = datetime.now()
    train(epoch)
    train_time += datetime.now() - start
    test(epoch)
tot_time = datetime.now() - train_start

print('======================================')
print(f'best acc:{best_acc}%  best adv acc:{best_adv_acc}%  in epoch {best_epoch}')
if not args.input_grad_norm:
    if args.env>0:
        file_name = f'./csvs/env{args.env}/{args.dataset}/{args.model}_seed{args.seed}.csv'
    else:
        file_name = f'./{args.dataset}.csv'
    with open(file_name, 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([f'{args.model}_{log_name}', args.model, method, best_acc, best_adv_acc, best_epoch, last_acc, last_adv_acc, str(train_time).split(".")[0], str(tot_time).split(".")[0],])
