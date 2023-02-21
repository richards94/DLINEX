import losses
import torch

def build_loss(name, **kwargs):
    if hasattr(torch.nn, name):
        criterion = getattr(torch.nn, name)(**kwargs)
    else:
        criterion = losses.__dict__[name](**kwargs)
    return criterion
