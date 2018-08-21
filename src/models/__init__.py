MODEL_REGISTRY = {}

def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator

def get_model(config):
    model_name = config['model']['name']
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](config)
    else:
        raise ValueError("Unknown model {:s}".format(model_name))

import models.ae
import models.nnet
import models.resnet
