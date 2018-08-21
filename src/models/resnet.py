'''
Credit: github/kuangliu - https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
Modified by James Lucas.

The code here can be used to create basic ResNets and Wide Resnets.


ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from models.base import ClassificationModel
import torchvision.models as tvmodels

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, block_config, num_classes=10):
        super(ResNet, self).__init__()

        num_blocks = block_config['num_blocks']
        channels = block_config['num_channels']
        assert len(channels) == len(num_blocks)

        self.in_planes = channels[0]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])


        self.layers = [self._make_layer(block, channels[0], num_blocks[0], stride=1)]

        for i in range(1,len(channels)):
            self.layers.append(self._make_layer(block, channels[i], num_blocks[i], stride=2))
        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AvgPool2d(block_config['pool_size'])
        self.linear = nn.Linear(channels[-1]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

from models import register_model

#### Imagenet Models ####
@register_model('resnet18')
def ResNet18(config):
    block_config = {
        "num_blocks": [2,2,2,2],
        "num_channels": [64, 128, 256, 512],
        "pool_size": 4
    }
    return ClassificationModel(ResNet(BasicBlock, block_config, config['data']['class_count']))

@register_model('resnet34')
def ResNet34(config):
    block_config = {
        "num_blocks": [3,4,6,3],
        "num_channels": [64, 128, 256, 512],
        "pool_size": 4
    }
    return ClassificationModel(ResNet(BasicBlock, block_config, config['data']['class_count']))

@register_model('resnet50')
def ResNet50(config):
    block_config = {
        "num_blocks": [3,4,6,3],
        "num_channels": [64, 128, 256, 512],
        "pool_size": 4
    }
    return ClassificationModel(ResNet(Bottleneck, block_config, config['data']['class_count']))

@register_model('resnet101')
def ResNet101(config):
    block_config = {
        "num_blocks": [3,4,23,3],
        "num_channels": [64, 128, 256, 512],
        "pool_size": 4
    }
    return ClassificationModel(ResNet(Bottleneck, block_config, config['data']['class_count']))

@register_model('resnet152')
def ResNet152(config):
    block_config = {
        "num_blocks": [3,8,36,3],
        "num_channels": [64, 128, 256, 512],
        "pool_size": 4
    }
    return ClassificationModel(ResNet(Bottleneck, block_config, config['data']['class_count']))


#### CIFAR MODELS ####
@register_model('resnet20')
def CifarResNet20(config):
    block_config = {
        "num_blocks": [3,3,3],
        "num_channels": [16, 32, 64],
        "pool_size": 8
    }
    return ClassificationModel(ResNet(BasicBlock, block_config, config['data']['class_count']))

@register_model('resnet32')
def CifarResNet32(config):
    block_config = {
        "num_blocks": [5,5,5],
        "num_channels": [16, 32, 64],
        "pool_size": 8
    }
    return ClassificationModel(ResNet(BasicBlock, block_config, config['data']['class_count']))

@register_model('resnet44')
def CifarResNet44(config):
    block_config = {
        "num_blocks": [7,7,7],
        "num_channels": [16, 32, 64],
        "pool_size": 8
    }
    return ClassificationModel(ResNet(Bottleneck, block_config, config['data']['class_count']))

@register_model('resnet56')
def CifarResNet56(config):
    block_config = {
        "num_blocks": [9,9,9],
        "num_channels": [16, 32, 64],
        "pool_size": 8
    }
    return ClassificationModel(ResNet(Bottleneck, block_config, config['data']['class_count']))

##### Torchvision imagenet models ####
@register_model('imagenet-resnet18')
def ImageNetResNet18(config):
    return ClassificationModel(nn.DataParallel(tvmodels.resnet18(False)))

@register_model('imagenet-resnet34')
def ImageNetResNet34(config):
    return ClassificationModel(nn.DataParallel(tvmodels.resnet34(False)))

@register_model('imagenet-resnet50')
def ImageNetResNet50(config):
    return ClassificationModel(nn.DataParallel(tvmodels.resnet50(False)))

@register_model('imagenet-resnet101')
def ImageNetResNet101(config):
    return ClassificationModel(nn.DataParallel(tvmodels.resnet101(False)))

@register_model('imagenet-resnet152')
def ImageNetResNet152(config):
    return ClassificationModel(nn.DataParallel(tvmodels.resnet152(False)))
