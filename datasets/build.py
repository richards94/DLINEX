import datasets
from datasets.transforms import build_transform
import torchvision

def build_dataset(type, ds_name, root, **kwargs):

    transform = build_transform(type)

    if hasattr(torchvision.datasets, ds_name):
        ds = getattr(torchvision.datasets, ds_name)(root=root, transform=transform, **kwargs)
    else:
        ds = datasets.__dict__[ds_name](root=root, transform=transform, **kwargs)
    return ds