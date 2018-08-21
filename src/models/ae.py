import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import AutoencoderModel
from models.nnet import FCNet

class AutoEncoder(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.input_size = self.encoder.input_dim

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return F.sigmoid(x)

class FCEncoder(FCNet):

    def __init__(self, config):
        super(FCEncoder, self).__init__(config['model']['layers'].copy(),
                                        config['data']['input_dim'],
                                        config['model']['activation'])

from models import register_model

@register_model('ce_fc_ae')
def load_ce_fc_ae(config):
    encoder = FCEncoder(config)

    layer_rev = config['model']['layers'][::-1]
    decoder_in_dim = layer_rev.pop(0)
    layer_rev.append(config['data']['input_dim'])
    decoder = FCNet(layer_rev, decoder_in_dim, config['model']['activation'])

    autoencoder = AutoEncoder(encoder, decoder)
    return AutoencoderModel(autoencoder)


