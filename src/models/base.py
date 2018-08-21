from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchnet as tnt

from torch.autograd import Variable

def batch_mean_mse(recon, inputs):
    return torch.sum(torch.mean((recon - inputs) ** 2, 0))

class ExperimentModel(nn.Module):

    def __init__(self, model):
        super(ExperimentModel, self).__init__()
        self.model = model
        self.init_meters()

    def forward(self, x):
        return self.model(x)

    def loss(self, sample, test=False):
        raise NotImplementedError

    def input_size(self):
        return self.model.input_size

    def init_meters(self):
        self.meters = OrderedDict([('loss', tnt.meter.AverageValueMeter())])

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].data.item())

class ClassificationModel(ExperimentModel):

    def init_meters(self):
        super(ClassificationModel, self).init_meters()
        self.meters['acc'] = tnt.meter.ClassErrorMeter(accuracy=True)

    def loss(self, sample, test=False):
        inputs = sample[0]
        targets = sample[1]
        o = self.model.forward(inputs)
        return F.cross_entropy(o, targets), {'logits': o}

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].data.item())
        self.meters['acc'].add(state['output']['logits'].data, state['sample'][1])

class AutoencoderModel(ExperimentModel):

    def init_meters(self):
        super(AutoencoderModel, self).init_meters()
        self.meters['mse'] = tnt.meter.AverageValueMeter()

    def loss(self, sample, test=False):
        inputs = sample[0]
        reconstruction = self.model.forward(inputs)

        mse = batch_mean_mse(reconstruction, inputs.view(-1, self.input_size()))
        loss = F.binary_cross_entropy(reconstruction, inputs.view(-1, self.input_size()))
        return loss, {'reconstruction': reconstruction, 'mse': mse}

    def add_to_meters(self, state):
        self.meters['loss'].add(state['loss'].data.item())
        self.meters['mse'].add(state['output']['mse'].data.item())
