import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision import datasets
from datasets import build_dataset
from models import build_model
from losses import build_loss
from utils import util

import argparse
import numpy as np
import time
import os
import random


parser = argparse.ArgumentParser(description='CNN_train')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152, 121, 169, 201, 161], type=int, metavar='N', default=None,
                    help='resnet depth (default: resnet18)')
parser.add_argument('--model', default='resnet18', help='model setting')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')  # 0.0001
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')  # ??0.9
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')  # ??30
parser.add_argument('--lr-decay', default=0.005, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')  # ??0.1
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true', help='use pre-trained model')
parser.add_argument('-a', '--a', default='1', type=str, metavar='LINEX',
                    help='parameter in Linex loss function')
parser.add_argument('--dataset', default='cifar100', help='dataset setting') # dataset
parser.add_argument('--workdir', default='./output', help='dataset root')
parser.add_argument('--root', default='./data', help='dataset root') # dataset
parser.add_argument('--pos_class', default=19, type=int, help='dataset setting') # dataset
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling') # dataset
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()


def setup_seed(seed=5):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train(train_loader, model, criterion, optimizer, device):
    train_size = len(train_loader)

    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(device), targets.cuda(device)
        targets[targets == 0] = -1
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs[:, 1]
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print("\tIter [%d/%d] Loss: %.7f" % (i + 1, train_size, loss.item()))


def eval(val, model, criterion, device):
    n = 0
    total, correct, total_loss = 0, 0, 0
    TP, FP, TN, FN = 0, 0, 0, 0

    for i, (inputs, targets) in enumerate(val):
        with torch.no_grad():
            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)
            targets[targets == 0] = -1
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            predicted[predicted == 0] = -1
            outputs = outputs[:, 1]
            loss = criterion(outputs, targets)
            total_loss += loss
            n += 1
            total += targets.size(0)
            len = targets.size(0)
            for j in range(len):
                if (predicted[j].cpu() == 1) and (targets[j].cpu() == 1):
                    TP += 1
                elif (predicted[j].cpu() == 1) and (targets[j].cpu() == -1):
                    FP += 1
                elif (predicted[j].cpu() == -1) and (targets[j].cpu() == 1):
                    FN = FN+1
                else:
                    TN = TN+1

    avg_val_acc = 100*(TP+TN)/total
    sensitivity = 100*TP/(TP+FN)
    specificity = 100*TN/(TN+FP)
    avg_val_gmeans = pow(sensitivity*specificity, 1/2)
    avg_val_F1 = 100*(2*TP)/(2*TP+FN+FP)
    avg_loss = total_loss / n
    
    return avg_val_acc, avg_val_gmeans, avg_val_F1, avg_loss, sensitivity, specificity


def main():
    setup_seed(5)
    device = get_device()
    print(f'DEVICE: {device}')
    
    print('Loading data')
    train_dataset = build_dataset(type=args.datatype+"_train", ds_name=args.dataset, root=args.root, positive_class=args.pos_class, rand_number=args.rand_number, train=True, download=False)
    val_dataset = build_dataset(type=args.datatype+"_test", ds_name=args.dataset, root=args.root, positive_class=args.pos_class, rand_number=args.rand_number, train=False, download=False)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    classes = train_dataset.get_cls_num_list()
    model = build_model(name=args.model, pretrained=args.pretrained, num_classes=len(classes))
    model.to(device)
    print('Using ' + args.model)
    print(model)
    print('Running on ' + str(device))

    criterion = build_loss(name='LinexLoss',a=args.a)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.lr_decay)
    n_epochs = args.epochs

    best_gmeans,best_acc,best_F,best_loss = 0, 0, 0, 0
    best_sensitivity, best_specificity = 0, 0
    start_epoch = 0
    for epoch in range(start_epoch, n_epochs):
        print('\n-----------------------------------')
        print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
        start = time.time()

        model.train()
        train(train_loader, model, criterion, optimizer, device)
        print('Time taken: %.5f sec.' % (time.time() - start))

        print('\nEvaluation:')
        model.eval()

        avg_val_acc, avg_val_gmeans, \
        avg_val_F1, avg_loss, \
        sensitivity, specificity = eval(val_loader, model, criterion, device)

        if avg_val_gmeans > best_gmeans:
            print('\tSaving checkpoint - Gmeans: %.5f' % avg_val_gmeans)
            best_gmeans = avg_val_gmeans
            best_acc = avg_val_acc
            best_F = avg_val_F1
            best_loss = avg_loss
            best_sensitivity = sensitivity
            best_specificity = specificity
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'gemans': avg_val_gmeans,
                'acc': avg_val_acc,
                'f1': avg_val_F1,
                'loss':avg_loss,
                'sensitivity':sensitivity,
                'specificity':specificity,
                'best_gmeans': best_gmeans,
                'optimizer': optimizer.state_dict(),
            }, args.workdir)
            
        
        print(f'current gemans: {avg_val_gmeans:.5f},     '
        f'current accuracy: {avg_val_acc:.5f},     '
        f'current F1: {avg_val_F1:.5f},     '
        f'sensitivity: {sensitivity:.5f},     '
        f'specificity: {specificity:.5f},     '
        )
        print(f'best gemans: {best_gmeans:.5f},     '
        f'accuracy in best epoch: {best_acc:.5f},     '
        f'F1 in best epoch: {best_F:.5f},     '
        f'sensitivity in best epoch: {best_sensitivity:.5f},     '
        f'specificity in best epoch: {best_specificity:.5f},     '
        )

if __name__ == '__main__':
    main()