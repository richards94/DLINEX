## train
```
python linex_train_main.py --model resnet18 --dataset cifar10 --workdir /a/b/c/ --root /dataset
```

## test
```
python linex_test_main.py --model resnet18 --dataset cifar10 --root /dataset --checkpoint /output/checkpoint.pth
```