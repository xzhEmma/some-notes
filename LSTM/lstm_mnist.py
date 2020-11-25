# _*_ coding: utf-8 _*_
"""
Created on Tues Nov 17 9:21 2020

@author: emma
"""

import sys
sys.path.append('..')

import torch
import datetime
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader
from lstm import my_lstm

from torchvision import transforms as tfs
from torchvision.datasets import MNIST

#define the data
data_tf = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5],[0.5])
])
train_set = MNIST('./MNIST',train=True,transform=data_tf,download=True)
test_set = MNIST('./MNIST',train=True,transform=data_tf,download=True)

train_data = DataLoader(train_set, 64, True, num_workers=4)
test_data = DataLoader(test_set, 128, False , num_workers=4)

net = my_lstm()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(net.parameters(), 1e-1)

def get_acc(output, label):
    total = output.shape[0]
    # debug print output.shape
    _,pred_label = output.max(1)
    #debug
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

def train(net, train_data, valid_data, num_epochs, optiizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            if torch.cuda.is_available():
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)
            #forward
            output = net(im)
            loss = criterion(output,label)
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss +=loss.item()
            train_acc +=get_acc(output,label)

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im,label in valid_data:
                if torch.cuda.is_available():
                    im = Variable(im.cuda())
                    label = Variable(label.cuda())
                else:
                    im = Variable(im)
                    label = Variable(label)
                output = net(im)
                loss = criterion(output,label)
                valid_loss += loss.item()
                valid_acc += get_acc(output,label)
                epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss:%f,Valid Acc:%f,"
                         %(epoch, train_loss /len(train_data),train_acc / len(train_data),
                           valid_loss / len(valid_data),
                           valid_acc / len(valid_data)))
        else:
           epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f"
                             % (epoch, train_loss / len(train_data), train_acc / len(train_data)))
        print(epoch_str)


train(net,train_data,test_data,10,optimizer,criterion)
