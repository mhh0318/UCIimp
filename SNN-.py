#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/14 0:14
@author: merci
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import os
import numpy as np
import time

import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class simple_CNN(nn.Module):
    def __init__(self, n_dims, classes, units, layers):
        super(simple_CNN, self).__init__()
        self.hidden = nn.Linear(units, units)
        self.output = nn.Linear(units, classes)
        self.input = nn.Linear(n_dims, units)
        self.SELU = nn.SELU()
        self.layers = layers
        self.bn = nn.BatchNorm1d(units)
        self.RELU = nn.ReLU()
        self.ln = nn.LayerNorm(units)

    def forward(self, x):

        ## SNN
        x = self.SELU(self.input(x))
        for _ in range(self.layers):
            x = self.SELU(self.hidden(x))
        x = self.SELU(self.output(x))


        ### BN
        #x = self.RELU(self.bn(self.input(x)))
        #for _ in range(self.layers):
        #    x = self.RELU(self.bn(self.hidden(x)))
        #x = self.RELU(self.output(x))

        ### LN
        #x = self.RELU(self.ln(self.input(x)))
        #for _ in range(self.layers):
        #    x = self.RELU(self.ln(self.hidden(x)))
        #x = self.RELU(self.output(x))

        return x


class Highway(nn.Module):
    def __init__(self, size, n_class, num_layers):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = torch.nn.functional.relu

        self.fc = nn.Linear(size,n_class)

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        x = self.fc(x)

        return x


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        # if i % 1000 == 0:
        # print('Train Epoch :{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        # epoch+1, i*len(data), len(train_loader.dataset), 100.*i/len(train_loader), loss.item()))


def test(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.long().to(device)
        output = model(data)
        test_loss += torch.nn.functional.cross_entropy(output, labels).item()
        pred = torch.max(output, 1)[1]
        correct += pred.eq(labels.data.view_as(pred)).sum()
        predictions.extend(pred.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


class UCI(data.Dataset):
    def __init__(self, data_name, folds, train=True, val=False):
        self.train = train
        root_path = '/home/hu/eRVFL/UCIdata'
        dataX = np.loadtxt('{0}/{1}/{1}_py.dat'.format(root_path, data_name), delimiter=',')
        dataY = np.loadtxt('{}/{}/labels_py.dat'.format(root_path, data_name), delimiter=',')
        Validation = np.loadtxt('{}/{}/validation_folds_py.dat'.format(root_path, data_name), delimiter=',')
        Folds_index = np.loadtxt('{}/{}/folds_py.dat'.format(root_path, data_name), delimiter=',')
        if val == False:
            if train == True:
                idx = np.where(Folds_index[:, folds] == 0)[0]
            else:
                idx = np.where(Folds_index[:, folds] == 1)[0]
        else:
            if train == True:
                idx = np.where(Validation[:, folds] == 0)[0]
            else:
                idx = np.where(Validation[:, folds] == 1)[0]
        X = dataX[idx, :]
        Y = dataY[idx]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(Y)
        # self.y_data = torch.max(self.y_data, 1)[1]
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def main():
    train_loader = data.DataLoader(dataset=UCI('iris', folds=0, train=True, val=False), batch_size=16,
                                   shuffle=False)
    test_loader = data.DataLoader(dataset=UCI('iris', folds=0, train=False, val=False), batch_size=16,
                                  shuffle=True)
    r = [0.01, 0.1, 1]
    u = [1024, 512, 256]
    #u = [0]
    l = [2, 3, 4, 8, 16, 32]
    minLoss = 1000
    for rate in r:
        for U in u:
            for L in l:
                a = time.time()
                n_class = np.unique(train_loader.dataset.y_data).size
                dims = train_loader.dataset.x_data.shape[1]
                model = simple_CNN(dims, n_class, U, L).to(device)
                #model = Highway(dims,n_class,L).to(device)
                optimizer = optim.SGD(model.parameters(), lr=rate)
                for epoch in range(100):
                    train(train_loader, model, optimizer, epoch)
                    loss = test(test_loader, model)
                b = time.time()
                print(
                    'r{}\tu{}\tl{}\n#########################################\nNO CONVERGE Training time for 1 option:{:.2f}s\n#########################################'.format(rate,U,L,b - a))

    val_test_loader = data.DataLoader(dataset=UCI('iris', folds=0, train=False, val=True), batch_size=16,
                                      shuffle=True)
    a = time.time()
    test(val_test_loader, model)
    b = time.time()
    print('Test time :{:.2f}s'.format(b - a))


if __name__ == '__main__':
    rate = [0.01, 0.1, 1]
    start = time.time()
    main()
    end = time.time()
    print("#####################################################\ntotal time:{}".format(end - start))
