import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
import numpy as np
import time
import os


from datasets import build_dataset
from models import build_model
from losses import build_loss
from utils import util

from sklearn.metrics import classification_report


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
parser.add_argument('--root', default='./data', help='dataset root') # dataset
parser.add_argument('--checkpoint', default='./output')
parser.add_argument('--pos_class', default=19, type=int, help='dataset setting') # dataset
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling') # dataset
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()


def load_checkpoint(model, args):
    global best_gmeans, start_epoch
    print('\n==> Loading checkpoint..')
    checkpoint = torch.load(args.checkpoint)
    best_gmeans = checkpoint['best_gmeans']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def test(model, data_loader, device, is_test=True):
    if is_test:
        load_checkpoint(model, args)

    n = 0
    predict, predict_score = [], []
    true = []
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)
            targets[targets == 0] = -1
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            predict_score.append(outputs)
            n += 1
            _, predicted = torch.max(outputs.data, 1)
            predicted[predicted == 0] = -1
            predict.append(predicted)
            true.append(targets)

    predict = torch.cat((predict), 0)
    predict = np.array(predict.cpu().numpy())
    predict_score = torch.cat((predict_score), 0)
    predict_score = np.array(predict_score.cpu().numpy())
    true = torch.cat((true), 0)
    true = np.array(true.cpu().numpy())
    # print(classification_report(true, predict, digits=5))

    return predict,predict_score,true

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('Loading data')
    test_datasets = build_dataset(type=args.datatype+"_test", ds_name=args.dataset, root=args.root, positive_class=args.pos_class, rand_number=args.rand_number, train=False, download=False)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, shuffle=True, num_workers=4)

    classes = test_datasets.get_cls_num_list()
    model = build_model(name=args.model, pretrained=args.pretrained, num_classes=len(classes))
    model.to(device)
    cudnn.benchmark = True
    print('Using ' + args.model)
    print('Running on ' + str(device))

    model.eval()
    predict,predict_score,true = test(model, test_loader, device)
    test_len=len(predict)
    TP, FP, TN, FN = 0, 0, 0, 0
    for j in range(test_len):
        if (predict[j]==1) and (true[j]==1):
            TP+=1
        elif (predict[j]==1) and (true[j]==-1):
            FP+=1
        elif (predict[j] ==-1) and (true[j]==1):
            FN=FN+1
        else:
            TN=TN+1
    test_acc=100*(TP+TN)/test_len
    sensitivity=100*TP/(TP+FN)
    specificity=100*TN/(TN+FP)
    test_gmeans=pow(sensitivity*specificity,1/2)
    test_F1=100*(2*TP)/(2*TP+FN+FP)

    print('\t test_gmeans: %.5f' % test_gmeans)
    print('\t tst_acc: %.5f' % test_acc)
    print('\t test_F1: %.5f' % test_F1)
    print(f'gemans: {test_gmeans:.5f},     '
        f'accuracy: {test_acc:.5f},     '
        f'F1: {test_F1:.5f},     '
        f'sensitivity: {sensitivity:.5f},     '
        f'specificity: {specificity:.5f},     '
        )

if __name__ == '__main__':
    main()