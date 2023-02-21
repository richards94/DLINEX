from datasets import transforms


def build_transform(type):
    return transforms.__dict__[type]()