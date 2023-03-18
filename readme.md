# DLINEX Loss Fuction
This repository contains the implementation of DLINEX Loss Fuction in image classification task.

## Usage
All experiments are implemented on a Linux workstation with an RTX 3090 and Intel(R) Xeon(R) CPU E5-2650.
- python 3.6+
- pytorch 1.7.0+
- CUDA 11.0+

## train
```
python linex_train_main.py --model resnet18 --dataset cifar10 --workdir /a/b/c/ --root /dataset
```

## test
```
python linex_test_main.py --model resnet18 --dataset cifar10 --root /dataset --checkpoint /output/checkpoint.pth
```
