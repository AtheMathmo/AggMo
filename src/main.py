import torch

from logger import Logger
from config import process_config
from engine import Engine
from models import get_model

from utils import * 

from functools import partial

import numpy as np
from tqdm import tqdm

import datetime
import os, errno

def get_experiment_name(config):
    now = datetime.datetime.now()

    base_exp_name = config['exp_name']
    task_name = config['task']
    data_name = config['data']['name']
    optim_name = config['optim']['optimizer']['name']

    exp_name = "{}_{}_{}_{}_{}".format(base_exp_name, task_name, data_name, optim_name, now.strftime("%Y%m%d_%H-%M-%S-%f"))
    return exp_name

def train(model, loaders, config):
    exp_dir = os.path.join(config['output_root'], get_experiment_name(config))
    log_dir = os.path.join(exp_dir, 'logs')
    model_dir = os.path.join(exp_dir, 'checkpoints')
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    img_dir = os.path.join(exp_dir, 'imgs')

    model.cuda()

    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)

    logger = Logger(log_dir)
    logger.log_config(config)

    engine = Engine()

    if config['optim']['criterion']['minmax'] == 'min':
        best_val = np.inf
    else:
        best_val = -np.inf

    def log_meters(prefix, logger, state):
        if 'epoch' in state:
            epoch = state['epoch']
        else:
            epoch = 0
        for tag, meter in state['model'].meters.items():
            file_id = '{}_{}'.format(prefix, tag)
            logger.log(file_id, epoch, meter.value()[0])

    def save_best_model(state, best_val):
        criterion = config['optim']['criterion']
        new_best = False
        for tag, meter in state['model'].meters.items():
            if tag == criterion['tag']:
                new_val = meter.value()[0]
                if criterion['minmax'] == 'min':
                    if new_val < best_val:
                        best_val = new_val
                        new_best = True
                else:
                    if new_val > best_val:
                        best_val = new_val
                        new_best = True
                break
        if new_best:
            print('Saving new best model')
            save_model(state['model'], best_model_path)
        return best_val, new_best


    def on_sample(state):
        if config['cuda']:
            state['sample'] = [x.cuda() for x in state['sample']]

    def on_forward(state):
        state['model'].add_to_meters(state)

    def on_start(state):
        state['loader'] = state['iterator']

    def on_start_epoch(state):
        state['model'].reset_meters()
        state['iterator'] = tqdm(state['loader'], desc='Epoch {}'.format(state['epoch']))

    def on_end_epoch(hook_state, state):
        scheduler.step()
        print("Training loss: {:.4f}".format(state['model'].meters['loss'].value()[0]))
        log_meters('train', logger, state)

        if ('reconstruction' in state['output']) and (state['epoch'] % 20 == 0):
            recon = state['output']['reconstruction'].data.view(-1, 28, 28).unsqueeze(1)
            save_imgs(recon, 'reconstruction_{}.jpg'.format(state['epoch']), img_dir)

        if state['epoch'] % 20 == 0:
            save_path = os.path.join(model_dir, "model_{}.pt".format(state['epoch']))
            save_model(model, save_path)

        # do validation at the end of each epoch
        if config['data']['validation']:
            state['model'].reset_meters()
            engine.test(model, loaders['validation'])
            print("Val loss: {:.4f}".format(state['model'].meters['loss'].value()[0]))
            log_meters('val', logger, state)
            hook_state['best_val'], new_best = save_best_model(state, hook_state['best_val'])
            if new_best:
                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1
                if hook_state['wait'] > config['optim']['patience']:
                    state['stop'] = True

        if state['epoch'] == (config['optim']['epochs'] - config['optim']['finetune']['epochs']):
            print('Momentum fine tuning for last stage')
            if config['optim']['finetune']['warm']:
                finetune_mom = config['optim']['finetune']['final_mom']
                for group in state['optimizer'].param_groups:
                    mom = [0.0, finetune_mom] if config['optim']['optimizer'].lower() == 'aggmo' else finetune_mom
                    group['momentum'] = mom
            else:
                #TODO: The cold-start finetuning does not change the momentum
                state['optimizer'] = get_optimizer(config, state['model'].parameters())
            for tag, meter in state['model'].meters.items():
                file_id = 'pre_finetune_{}'.format(tag)
                logger.log(file_id, state['epoch'], meter.value()[0])

    engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = partial(on_end_epoch, {'best_val': best_val, 'wait': 0})
    engine.train(model, loaders['train'], maxepoch=config['optim']['epochs'], optimizer=optimizer)

    model.reset_meters()
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    if loaders['test'] is not None:
        log_meters('test', logger, engine.test(model, loaders['test']))
    return model

if __name__ == '__main__':
    config = process_config()
    model_init = get_model(config)
    loaders = load_data(config)
    model = train(model_init, loaders, config)
