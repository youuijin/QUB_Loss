import random
import  torch
import  numpy as np

from utils.model.resnet import *
from utils.model.wrn import *

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from torch.utils.data import DataLoader

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
    ### dataset ###
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
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