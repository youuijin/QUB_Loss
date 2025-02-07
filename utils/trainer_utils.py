from utils.model.resnet import *
from utils.model.wrn import *

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder

from torch.utils.data import DataLoader, random_split, Subset

import torch

# from Trainers import *

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
        train_data = CIFAR10(root='./data', train=True, download=False, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        data_num = 10
    elif dataset == 'cifar100':
        train_data = CIFAR100(root='./data', train=True, download=False, transform=transform)
        test_data = CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        data_num = 100
    elif dataset == 'svhn':
        train_data = SVHN(root='./data', split='train', download=False, transform=transform)
        test_data = SVHN(root='./data', split='test', download=False, transform=transform_test)
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
        train_data = CIFAR10(root='./data', train=True, download=False, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        data_num = 10
    elif dataset == 'cifar100':
        train_data = CIFAR100(root='./data', train=True, download=False, transform=transform)
        test_data = CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        data_num = 100
    elif dataset == 'svhn':
        train_data = SVHN(root='./data', split='train', download=False, transform=transform)
        test_data = SVHN(root='./data', split='test', download=False, transform=transform)
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

# import torch

def get_gradient_norm(model, norm_type='L2'):
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            if norm_type == 'L2':
                param_norm = p.grad.detach().data.norm(2)  # L2 norm (Euclidean norm)
                total_norm += param_norm.item() ** 2
            elif norm_type == 'L1':
                param_norm = p.grad.detach().data.abs().sum()  # L1 norm (sum of absolute values)
                total_norm += param_norm.item()
            elif norm_type == 'Linf':
                param_norm = p.grad.detach().data.abs().max()  # L∞ norm (max of absolute values)
                total_norm = max(total_norm, param_norm.item())
            else:
                raise ValueError("Invalid norm type. Choose from 'L2', 'L1', or 'Linf'.")
    
    if norm_type == 'L2':
        total_norm = total_norm ** 0.5  # sqrt to get final L2 norm
    
    return total_norm

def get_logit_norm(logit, loss):
    total_norm = torch.norm(logit.grad, p=2)
    # total_norm = (-1*y+logit)
    # total_norm = total_norm ** 0.5  # sqrt to get final L2 norm
    return total_norm
