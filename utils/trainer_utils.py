from utils.model.resnet import *
from utils.model.wrn import *

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder

from torch.utils.data import DataLoader, random_split, Subset

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
