'''
CIFAR-10 Classification
'''

import torch
import torch.nn as nn

from models.base import ClassificationModel

class FCNet(torch.nn.Module):
    def __init__(self, layers, input_dim, activation='relu'):
        super(FCNet, self).__init__()

        self.layer_sizes = layers.copy()
        self.input_dim = input_dim

        self.layer_sizes.insert(0, self.input_dim)

        if activation == 'relu':
            act_func = nn.ReLU
        elif activation == 'sigmoid':
            act_func = nn.Sigmoid
        else:
            raise Exception('Unexpected activation function. ReLU or Sigmoid supported.')


        layers = [nn.Linear(self.layer_sizes[0], self.layer_sizes[1])]

        for i in range(2, len(self.layer_sizes)):
            layers.append(act_func())
            layers.append(nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i]))


        self.model = nn.Sequential(*layers)

    def __len__(self):
        return len(self.model)

    def __getitem__(self, idx):
        self.model[idx]

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class AlexNet(nn.Module):

    def __init__(self, class_count):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, class_count)

    def forward(self, x):
        x = x .view(-1, 3, 32, 32)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


from models import register_model

@register_model('alexnet_cifar')
def load_alexnet_classification(config):
    model = AlexNet(config['data']['class_count'])
    return ClassificationModel(model)