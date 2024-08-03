'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import csv

import argparse
from datetime import datetime, timedelta

from utils.train_utils import *
from attack.pgd_attack import PGDAttack

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 FGSM-PGI Training')

# env options
parser.add_argument('--env', type=int, default=0)


# model options
parser.add_argument('--model', choices=['resnet18', 'resnet34', 'preresnet18', 'wrn_28_10', 'wrn_34_10'], default='resnet18')

# dataset options
parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10')
parser.add_argument('--normalize', choices=['none', 'twice', 'imagenet', 'cifar'], default='none')

# train options
parser.add_argument('--loss', choices=['CE', 'QUB'], default='CE')
parser.add_argument('--log_upper', default=False, action='store_true')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--sche', default='multistep', choices=['multistep', 'cyclic'], help='learning rate')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=110)
parser.add_argument('--device', type=int, default=0)

# attack options
parser.add_argument('--eps', type=float, default=8.)
parser.add_argument('--alpha', type=float, default=8.)
parser.add_argument('--epoch_reset', default=40, type=int)
parser.add_argument('--momentum_decay', type=float, default=0.3)
parser.add_argument('--lamb', default=10, type=float)
parser.add_argument('--delta_init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
parser.add_argument('--factor', default=0.6, type=float, help='Label Smoothing')

# QUB options
parser.add_argument('--wo_regularizer', action='store_true', default=False, help='if true, dont use MSE Loss')

# test options
parser.add_argument('--test_eps', type=float, default=8.)

args = parser.parse_args()

device = f'cuda:{args.device}'
best_acc, best_adv_acc = 0, 0  # best test accuracy
best_epoch = -1

set_seed(seed=0)
method = 'FGSM_PGI'
cur = datetime.now().strftime('%m-%d_%H-%M')
# log_name = f'{args.loss}_{method}(eps{args.eps}_mom{args.momentum_decay}_lamb{args.lamb}_{args.delta_init})_epoch{args.epoch}_{args.normalize}_{args.sche}_{args.factor}_{cur}'
log_name = f'{args.loss}_{method}(eps{args.eps})_lr{args.lr}_{cur}'

# Summary Writer
writer = SummaryWriter(f'logs/{args.dataset}/{args.model}/env{args.env}/{log_name}')
if args.log_upper:
    upper_writer = SummaryWriter(f'upper_logs/{args.dataset}/{args.model}/env{args.env}/{log_name}/upper')
    real_writer = SummaryWriter(f'upper_logs/{args.dataset}/{args.model}/env{args.env}/{log_name}/real')

def _label_smoothing(label, factor):
    one_hot = np.eye(10)[label.to(device).data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))

    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

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

# set dataset
train_data, test_data, n_way = set_dataset(args.dataset, norm_mean, norm_std)
train_loader = DataLoader(train_data, batch_size=50000, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

cifar_x, cifar_y = None, None

for i, (X, y) in enumerate(train_loader):
    cifar_x, cifar_y = X.to(device), y.to(device)

# Model
print('==> Building model..')
model = set_model(model_name=args.model, n_class=n_way)
model = model.to(device)

# Train Attack & Test Attack
test_attack = PGDAttack(model, eps=args.test_eps, alpha=2., iter=10, mean=norm_mean, std=norm_std, device=device)

num_of_example = 50000
batch_size = args.batch_size

iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)

# Optimizer
# lr_steps = args.epoch * iter_num
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps * 100/110, lr_steps * 105 / 110], gamma=0.1)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_steps = args.epoch * iter_num
if args.env == 1:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_steps*100/200), int(lr_steps*150/200)], gamma=0.1)
elif args.env == 2:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_steps*100/110), int(lr_steps*105/110)], gamma=0.1)
elif args.env == 3:
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0, max_lr=0.1,
        step_size_up=lr_steps/2, step_size_down=lr_steps/2)
    
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps * 100/110, lr_steps * 105 / 110], gamma=0.1)

mean = torch.tensor(norm_mean).to(device).view(1, 3, 1, 1)
std = torch.tensor(norm_std).to(device).view(1, 3, 1, 1)
upper_limit = ((1 - mean) / std)
lower_limit = ((0 - mean) / std)
eps = args.eps/255./std
alpha = args.alpha/255./std

print('start training..')

train_time = timedelta()
train_start = datetime.now()

def atta_aug(input_tensor, rst):
    batch_size = input_tensor.shape[0]
    x = torch.zeros(batch_size)
    y = torch.zeros(batch_size)
    flip = [False] * batch_size

    for i in range(batch_size):
        flip_t = bool(random.getrandbits(1))
        x_t = random.randint(0, 8)
        y_t = random.randint(0, 8)

        rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
        if flip_t:
            rst[i] = torch.flip(rst[i], [2])
        flip[i] = flip_t
        x[i] = x_t
        y[i] = y_t

    return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}

for epoch in range(args.epoch):
    model.train()
    start = datetime.now()
    # a_delta, a_mom = train(epoch, cifar_x, cifar_y, a_delta, a_mom)
    train_loss = 0
    correct = 0
    total = 0
    real_adv_loss = 0

    # global cifar_x, cifar_y, all_delta, all_momentum

    batch_size = args.batch_size
    cur_order = np.random.permutation(num_of_example)
    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
    batch_idx = -batch_size

    if epoch % args.epoch_reset == 0:
        temp = torch.rand(50000,3,32,32)
        if args.delta_init != 'previous':
            all_delta = torch.zeros_like(temp).to(device)
            all_momentum=torch.zeros_like(temp).to(device)
        if args.delta_init == 'random':
            for j in range(len(eps.squeeze())):
                all_delta[:, j, :, :].uniform_(-eps[0][j][0][0].item(), eps[0][j][0][0].item())
            #all_delta.data = clamp(all_delta, lower_limit - cifar_x, upper_limit - cifar_x)
            #all_delta.requires_grad = True
            all_delta.data = torch.clamp(alpha * torch.sign(all_delta), -eps, eps)
            # print(all_data.data)
            #all_delta.data[:cifar_x.size(0)] = clamp(all_delta[:cifar_x.size(0)], lower_limit - cifar_x, upper_limit - cifar_x)
    
    idx = torch.randperm(cifar_x.shape[0])
    
    cifar_x = cifar_x[idx, :,:,:].view(cifar_x.size())
    cifar_y = cifar_y[idx].view(cifar_y.size())
    all_delta=all_delta[idx, :, :, :].view(all_delta.size())
    all_momentum=all_momentum[idx, :, :, :].view(all_delta.size())
    
    for i in range(iter_num):
        batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
        X=cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
        y= cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
        delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
        next_delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()

        momentum=all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
        X=X.to(device)
        y=y.to(device)
        batch_size = X.shape[0]

        # ## add Att
        rst = torch.zeros(batch_size, 3, 32, 32).to(device)
        X, transform_info = atta_aug(X, rst)

        label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.factor)).to(device)).float()

        delta.requires_grad = True
        ori_output = model(X + delta[:X.size(0)])

        # default : Label Smoothing Loss -> Change to CE Loss
        # ori_loss = LabelSmoothLoss(ori_output, label_smoothing.float())
        ori_loss = F.cross_entropy(ori_output, y)


        decay = args.momentum_decay
        # with amp.scale_loss(loss, opt) as scaled_loss:
        ori_loss.backward(retain_graph=True)
        x_grad = delta.grad.detach()
        # y_grad = delta_y.grad.detach()
        grad_norm = torch.norm(x_grad, p=1)
        momentum = x_grad/grad_norm+momentum * decay

        next_delta.data = torch.clamp(delta + alpha * torch.sign(momentum), -eps, eps)
        next_delta.data[:X.size(0)] = torch.clamp(next_delta[:X.size(0)], lower_limit - X, upper_limit - X)

        delta.data = torch.clamp(delta + alpha * torch.sign(x_grad), -eps, eps)
        delta.data[:X.size(0)] = torch.clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)

        delta = delta.detach()

        output = model(X + delta[:X.size(0)])

        loss_fn = torch.nn.MSELoss(reduction='mean')
        if args.loss == 'CE':
            # default : Label Smoothing Loss -> Change to CE Loss
            loss = F.cross_entropy(output, y) + args.lamb*loss_fn(output.float(), ori_output.float())
            # loss = LabelSmoothLoss(output, (label_smoothing).float())+args.lamb*loss_fn(output.float(), ori_output.float())
        elif args.loss == 'QUB':
            clean_outputs = model(X)
            softmax = F.softmax(clean_outputs, dim=1)
            y_onehot = F.one_hot(y, num_classes = softmax.shape[1])

            # adv_inputs = attack.perturb(inputs, targets)
            # adv_outputs = model(adv_inputs)
            adv_norm = torch.norm(clean_outputs-output, dim=1)

            loss = F.cross_entropy(X, y, reduction='none')

            upper_loss = loss + torch.sum((adv_outputs-clean_outputs)*(softmax-y_onehot), dim=1) + 0.5/2.0*torch.pow(adv_norm, 2)
            loss = upper_loss.mean()

            if not args.wo_regularizer:
                loss += args.lamb * loss_fn(output.float(), ori_output.float())

        optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        loss.backward()

        optimizer.step()
        train_loss += loss.item() * y.size(0)
        correct += (output.max(1)[1] == y).sum().item()
        total += y.size(0)

        if args.log_upper:
            real_adv_loss += F.cross_entropy(adv_outputs, targets).item()

        scheduler.step()

        all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = momentum

        all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]=next_delta


    writer.add_scalar('train/acc', 100.*correct/total, epoch)
    writer.add_scalar('train/loss', round(train_loss/total, 4), epoch)
    if args.log_upper:
        upper_writer.add_scalar(f'train/{log_name}', round(train_loss/total, 4), epoch)
        real_writer.add_scalar(f'train/{log_name}', round(real_adv_loss/total, 4), epoch)

    # print(f'Epoch {epoch}: acc {100.*correct/total} \t loss {round(train_loss/total, 4)}')

    train_time += datetime.now() - start
    
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

    # print('test:', 'SA:', 100.*correct/total, '\tRA:',  100.*adv_correct/total)

    # Save checkpoint.
    adv_acc = 100.*adv_correct/total
    if adv_acc > best_adv_acc:
        torch.save(model.state_dict(), f'./env_models/env{args.env}/{args.model}_{log_name}.pt')
        best_adv_acc = adv_acc
        best_acc = 100.*correct/total
        best_epoch = epoch
    
    
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
