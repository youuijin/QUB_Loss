import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = F.relu(self.bn1(out))
    #     out = F.avg_pool2d(out, 8)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)

    #     return out
    
    def forward(self, x, feature_layer=0):
        out = self.conv1(x)
        out = self.layer1(out)
        if feature_layer == 1:
            feature = F.avg_pool2d(out, 32)
            feature = feature.view(feature.size(0), -1)
            return feature
        
        out = self.layer2(out)
        if feature_layer == 2:
            feature = F.avg_pool2d(out, 16)
            feature = feature.view(feature.size(0), -1)
            return feature
        
        out = self.layer3(out)
        if feature_layer == 3:
            feature = F.avg_pool2d(out, 8)
            feature = feature.view(feature.size(0), -1)
            return feature
        
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def WideResNet_28_10(n_class, dropout=0.0):
    return WideResNet(28, 10, dropout_rate=dropout, num_classes=n_class)

def WideResNet_34_10(n_class, dropout=0.0):
    return WideResNet(34, 10, dropout_rate=dropout, num_classes=n_class)