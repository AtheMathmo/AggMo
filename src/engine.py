'''
Based on code from https://github.com/pytorch/tnt/blob/master/torchnet/engine/engine.py

Edited by Jake Snell

(Minor tweaks by James Lucas)

'''
class Engine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, model, iterator, maxepoch, optimizer):
        state = {
            'model': model,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'epoch': 0,
            't': 0,
            'train': True,
            'stop': False
        }
        model.train()
        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch'] and not state['stop']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['model'].loss(state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    loss.backward()
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    # state['output'] = None
                    # state['loss'] = None
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, model, iterator):
        state = {
            'model': model,
            'iterator': iterator,
            't': 0,
            'train': False,
        }

        model.eval()
        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['model'].loss(state['sample'], test=True)
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                # to free memory in save_for_backward
                # state['output'] = None
                # state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_end', state)
        return state
