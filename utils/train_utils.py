import random
import  torch
import  numpy as np
import copy

from utils.model.resnet import *
from utils.model.wrn import *

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder

from torch.utils.data import DataLoader, random_split, Subset

from Trainers import *

def set_seed(seed=706):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_model(model_name, n_class):
    if model_name=="resnet18":
        return ResNet18(n_class)
    elif model_name=="preresnet18":
        return PreActResNet18(n_class)
    elif model_name=="resnet34":
        return ResNet34(n_class)
    elif model_name=='wrn_28_10':
        return WideResNet_28_10(n_class, dropout=0.3)
    elif model_name=='wrn_34_10':
        return WideResNet_34_10(n_class, dropout=0.3)
    else:
        raise ValueError('Undefined Model Architecture')

def set_dataloader(dataset, batch_size, norm_mean, norm_std):
    if dataset in ['cifar10', 'cifar100']:
        imgsz = 32
    else:
        imgsz = 64

    ### transform ###
    transform = transforms.Compose([transforms.RandomCrop(imgsz, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])
    transform_test = transforms.Compose([transforms.Resize(imgsz),
                                         transforms.ToTensor(),
                                         transforms.Normalize(norm_mean, norm_std)])

    if dataset == 'cifar10':
        train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        data_num = 10
    elif dataset == 'cifar100':
        train_data = CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        data_num = 100
    elif dataset == 'svhn':
        train_data = SVHN(root='./data', split='train', download=True, transform=transform)
        test_data = SVHN(root='./data', split='test', download=True, transform=transform_test)
        data_num = 10
    elif dataset == 'tiny_imagenet':
        tot_dataset = ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        train_size, test_size = int(len(tot_dataset)*0.8), len(tot_dataset) - int(len(tot_dataset)*0.8)

        train_data, test_data = random_split(tot_dataset, [train_size, test_size])
        test_data = Subset(
            ImageFolder('./data/tiny-imagenet-200/train', transform=transform_test),
            test_data.indices
        )
        data_num = 200

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader, data_num, imgsz

def set_dataset(dataset, norm_mean, norm_std):
    if dataset in ['cifar10', 'cifar100']:
        imgsz = 32
    else:
        imgsz = 64
    ### dataset ###
    transform = transforms.Compose([transforms.Pad(padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    if dataset == 'cifar10':
        train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        data_num = 10
    elif dataset == 'cifar100':
        train_data = CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        data_num = 100
    elif dataset == 'svhn':
        train_data = SVHN(root='./data', split='train', download=True, transform=transform)
        test_data = SVHN(root='./data', split='test', download=True, transform=transform)
        data_num = 10
    elif dataset == 'tiny_imagenet':
        tot_dataset = ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
        train_size, test_size = int(len(tot_dataset)*0.8), len(tot_dataset) - int(len(tot_dataset)*0.8)

        train_data, test_data = random_split(tot_dataset, [train_size, test_size])
        test_data = Subset(
            ImageFolder('./data/tiny-imagenet-200/train', transform=transform_test),
            test_data.indices
        )
        data_num = 200


    return train_data, test_data, data_num, imgsz

@torch.no_grad()
def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def calc_K(softmax_vector):
    # A : [bs, 10] (softmax)
    batch_size, n = softmax_vector.size()
    H = torch.zeros(batch_size, n, n, device=softmax_vector.device)
    
    # Compute the diagonal elements
    diag_elements = softmax_vector * (1 - softmax_vector)
    diag_indices = torch.arange(n, device=softmax_vector.device)
    H[:, diag_indices, diag_indices] = diag_elements
    
    # Compute the off-diagonal elements
    for i in range(n):
        for j in range(n):
            if i != j:
                H[:, i, j] = -softmax_vector[:, i] * softmax_vector[:, j]

    # print(H.shape)

    K_values = torch.linalg.matrix_norm(H, ord=2, dim=(-2, -1), keepdim=False)
    # print(K_values.shape)
    return K_values.detach()

def input_logit_norm(model, input, target):
    copy_model = copy.deepcopy(model)
    # \nabla_x h(x)
    input = input.clone().detach()
    input.requires_grad_(True)
    logit = copy_model(input)
    logit = logit.sum()

    logit.backward()
    input_gradient = input.grad
    
    input_gradient = input_gradient.reshape(input.shape[0], -1)
    gradient_norms = input_gradient.norm(dim=1)

    del copy_model
    return gradient_norms

def logit_loss_norm(model, input, target):
    copy_model = copy.deepcopy(model)
    # \nabla_h L(h(x))
    output = copy_model(input)
    softmax = F.softmax(output, dim=1)
    y_onehot = F.one_hot(target, num_classes = softmax.shape[1])

    logit_gradient = softmax - y_onehot
    gradient_norms = logit_gradient.norm(dim=1)

    del copy_model
    return gradient_norms

def input_loss_norm(model, input, target):
    copy_model = copy.deepcopy(model)
    # \nabla_x L(h(x))
    input = input.clone().detach()
    input.requires_grad_(True)
    logit = copy_model(input)
    loss = F.cross_entropy(logit, target, reduction='sum')

    loss.backward()

    input_gradient = input.grad

    input_gradient = input_gradient.reshape(input.shape[0], -1)
    gradient_norms = input_gradient.norm(dim=-1)

    del copy_model
    return gradient_norms

def set_K(mode, K, tot_epoch, cur_epoch, acc, l_min, l_max, acc_func):
    if mode == 'none':
        return K
    elif mode == 'linear':
        K = cur_epoch/tot_epoch * (l_max - l_min) + l_min
        return K
    elif mode == 'acc':
        if acc_func == 'frac':
            K = K/(1-acc)
        elif acc_func == 'line':
            K = (l_max-K)*acc + K
        elif acc_func == 'quad':
            K = -1*(l_max-K)*(1-acc)*(1-acc) + l_max
        return K
    else:
        raise ValueError("set K with mode none, linear or acc")
    
def calc_reg(reg_func, reg, tot_epoch, now_epoch):
    if reg_func == 'const':
        now_reg = reg
    elif reg_func == 'linear_increase':
        now_reg = reg * (now_epoch) / tot_epoch
    return now_reg

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
