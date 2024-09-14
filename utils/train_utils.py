import random
import  torch
import  numpy as np

from utils.model.resnet import *
from utils.model.wrn import *

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder

from torch.utils.data import DataLoader, random_split, Subset

# from attack.AttackLoss import *


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

    return train_loader, test_loader, data_num

def set_dataset(dataset, norm_mean, norm_std):
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


    return train_data, test_data, data_num

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
    # \nabla_x h(x)
    input = input.clone().detach()
    input.requires_grad_(True)
    logit = model(input)
    logit = logit.sum()

    logit.backward()
    input_gradient = input.grad
    
    input_gradient = input_gradient.reshape(input.shape[0], -1)
    gradient_norms = input_gradient.norm(dim=1)
    return gradient_norms

def logit_loss_norm(model, input, target):
    # \nabla_h L(h(x))
    output = model(input)
    softmax = F.softmax(output, dim=1)
    y_onehot = F.one_hot(target, num_classes = softmax.shape[1])

    logit_gradient = softmax - y_onehot
    gradient_norms = logit_gradient.norm(dim=1)

    return gradient_norms

def input_loss_norm(model, input, target):
    # \nabla_x L(h(x))
    input = input.clone().detach()
    input.requires_grad_(True)
    logit = model(input)
    loss = F.cross_entropy(logit, target)

    loss.backward()

    input_gradient = input.grad

    input_gradient = input_gradient.reshape(input.shape[0], -1)
    gradient_norms = input_gradient.norm(dim=1)
    return gradient_norms