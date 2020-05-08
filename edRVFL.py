import numpy as np
import cupy as cp
import scipy.io as sio
from rescale import *
from option import option as op
from MRVFL import *

# load dataset
# dataX

mat = sio.loadmat('dataX.mat')
dataX = mat['dataX']
dataX = cp.array(dataX)

# dataY
mat = sio.loadmat('dataY.mat')
dataY = mat['dataY']
dataY = cp.array(dataY)

# test_indx
mat = sio.loadmat('test_indx.mat')
test_indx = mat['test_indx']
test_indx = cp.array(test_indx)
test_indx = test_indx.flatten() - 1

# train_indx
mat = sio.loadmat('train_indx.mat')
train_indx = mat['train_indx']
train_indx = cp.array(train_indx)
train_indx = train_indx.flatten() - 1

U_dataY = cp.unique(dataY)
nclass = U_dataY.size
dataY_temp = cp.zeros((dataY.size, nclass))

# 0-1 coding for the target
for i in range(nclass):
    for j in range(dataY.size):  # remove this loop
        if dataY[j] == U_dataY[i]:
            dataY_temp[j, i] = 1

dataX = rescale(dataX)

trainX = dataX[train_indx, :]
trainY = dataY_temp[train_indx, :]
testX = dataX[test_indx, :]
testY = dataY_temp[test_indx, :]

# default values, you need to tune them for best results
option = op(100, 10, pow(2, -2), pow(2, 0.5), 0, mode='replace')
option.N = 100
option.L = 4
option.C = 2 ** 6
option.scale = 1
option.seed = 0

[model, train_acc, test_acc, training_time, testing_time] = MRVFL(trainX, trainY, testX, testY, option)
print(train_acc)
print(test_acc)
