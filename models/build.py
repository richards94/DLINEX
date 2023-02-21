import models


def build_model(name, **kwargs):
    return models.__dict__[name](**kwargs)
