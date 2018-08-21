import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import torchnet as tnt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image


from aggmo import AggMo

import datetime
import os, errno

def get_optimizer(config, params):
    optim_cfg = config['optim']['optimizer']
    optim_name = optim_cfg['name']
    lr = optim_cfg['lr']

    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr, momentum=optim_cfg['momentum'], weight_decay=config['optim']['wdecay'])
    elif optim_name == 'nesterov':
        optimizer = torch.optim.SGD(params, lr, momentum=optim_cfg['momentum'], nesterov=True,
                                    weight_decay=config['optim']['wdecay'])
    elif optim_name == 'aggmo':
        optimizer = AggMo(params, lr, betas=optim_cfg['betas'], weight_decay=config['optim']['wdecay'])
    elif optim_name == 'aggmo_exp':
        optimizer = AggMo.from_exp_form(params, lr, a=optim_cfg['a'], k=optim_cfg['K'], weight_decay=config['optim']['wdecay'])
    elif optim_name =='adam':
        optimizer = torch.optim.Adam(params, lr, betas=(optim_cfg['beta1'], optim_cfg['beta2']), weight_decay=config['optim']['wdecay'])
    else:
        raise Exception('Unknown optimizer')
    return optimizer

def get_scheduler(config, optimizer):
    lr_schedule_conf = config['optim']['lr_schedule']
    if lr_schedule_conf['name'] == 'exp':
        return lr_scheduler.ExponentialLR(optimizer, lr_schedule_conf['lr_decay'], lr_schedule_conf['last_epoch'])
    elif lr_schedule_conf['name'] == 'step':
        return lr_scheduler.MultiStepLR(optimizer, lr_schedule_conf['milestones'], lr_schedule_conf['lr_decay'])

def save_model(model, save_path):
    try:
        os.makedirs(os.path.dirname(save_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    torch.save(model.state_dict(), save_path)

def save_imgs(tensor, fname, save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    save_image(tensor, os.path.join(save_dir, fname))

def get_data_transforms(config):
    train_transform = None
    test_transform = None

    transform_cfg = config['data']['transform']
    if transform_cfg:
        name = transform_cfg['name']
        normalize = transforms.Normalize(mean=transform_cfg['norm_mean'],
                                        std=transform_cfg['norm_std'])
        if name == 'cifar':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

        elif name == 'imagenet':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        train_transform = transforms.ToTensor()
    if test_transform is None:
        test_transform = transforms.ToTensor()
    return train_transform, test_transform

def load_data(config):
    data_name = config['data']['name'].lower()
    path = os.path.join(config['data']['root'], data_name)

    train_transform, test_transform = get_data_transforms(config)

    if data_name == 'mnist':
        train_data = datasets.MNIST(path, download=True, transform=train_transform)
        val_data = datasets.MNIST(path, download=True, transform=test_transform)
        test_data = datasets.MNIST(path, train=False, download=True, transform=test_transform)
    elif data_name == 'cifar10':
        train_data = datasets.CIFAR10(path, download=True, transform=train_transform)
        val_data = datasets.CIFAR10(path, download=True, transform=test_transform)
        test_data = datasets.CIFAR10(path, train=False, download=True, transform=test_transform)
    elif data_name == 'cifar100':
        train_data = datasets.CIFAR100(path, download=True, transform=train_transform)
        val_data = datasets.CIFAR100(path, download=True, transform=test_transform)
        test_data = datasets.CIFAR100(path, train=False, download=True, transform=test_transform)
    elif data_name == 'fashion-mnist':
        train_data = datasets.FashionMNIST(path, download=True, transform=train_transform)
        val_data = datasets.FashionMNIST(path, download=True, transform=test_transform)
        test_data = datasets.FashionMNIST(path, train=False, download=True, transform=test_transform)
    elif data_name == 'imagenet-torchvision':
        train_data = datasets.ImageFolder(os.path.join(path, 'train'), transform=train_transform)
        val_data = datasets.ImageFolder(os.path.join(path, 'valid'), transform=test_transform)
        # Currently not loaded
        test_data = None
    else:
        raise NotImplementedError('Data name %s not supported' % data_name)

    # Manually readjust train/val size for memory saving
    if data_name != 'imagenet-torchvision':
        data_size = len(train_data)
        train_size = int(data_size * config['data']['train_size'])

        train_data.train_data = train_data.train_data[:train_size]
        train_data.train_labels = train_data.train_labels[:train_size]

        if config['data']['train_size'] != 1:
            val_data.train_data = val_data.train_data[train_size:]
            val_data.train_labels = val_data.train_labels[train_size:]
        else:
            val_data = None

    batch_size = config['optim']['batch_size']
    num_workers = config['data']['num_workers']
    loaders = {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'validation': DataLoader(val_data, batch_size=batch_size, num_workers=num_workers),
        'test': DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    }

    return loaders