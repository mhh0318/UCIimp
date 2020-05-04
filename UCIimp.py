#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/3 15:35
@author: merci
"""
import cupy as cp
import numpy as np
from rescale import *
from option import option as op
from MRVFL import *
import pickle

cp.cuda.Device(5).use()
root_path = '/home/hu/eRVFL/UCIdata'
data_name = 'arrhythmia'

# load dataset
# dataX
datax = np.loadtxt('{0}/{1}/{1}_py.dat'.format(root_path, data_name), delimiter=',')
dataX = cp.asarray(datax)
# dataY
datay = np.loadtxt('{}/{}/labels_py.dat'.format(root_path, data_name), delimiter=',')
dataY = cp.asarray(datay)

# Validation Index
Validation = np.loadtxt('{}/{}/validation_folds_py.dat'.format(root_path, data_name), delimiter=',')
validation = cp.asarray(Validation)

# Folds Index
Folds_index = np.loadtxt('{}/{}/folds_py.dat'.format(root_path, data_name), delimiter=',')
folds_index = cp.asarray(Folds_index)

types = cp.unique(dataY)
n_types = types.size
n_CV = folds_index.shape[1]
# One hot coding for the target
dataY_tmp = cp.zeros((dataY.size, n_types))
for i in range(n_types):
    for j in range(dataY.size):  # remove this loop
        if dataY[j] == types[i]:
            dataY_tmp[j, i] = 1

option = op(256, 32, 2**-6, 0.5, 0)
N_range = [256,512,1024];
option.L = 32;
option.scale = 0.5;
C_range = range(-6,12,2);

Models_tmp = []
Models = []
dataX = rescale(dataX)

train_acc_result = cp.zeros((n_CV,1))
test_acc_result = cp.zeros((n_CV,1))
train_time_result = cp.zeros((n_CV,1))
test_time_result = cp.zeros((n_CV,1))

MAX_acc = 0
option_best = op(256, 32, 2**-6, 0.5, 0)
for i in range(n_CV):

    MAX_acc = 0
    train_idx = cp.where(validation[:,i]==0)[0]
    test_idx = cp.where(validation[:,i]==1)[0]
    trainX = dataX[train_idx, :]
    trainY = dataY_tmp[train_idx, :]
    testX = dataX[test_idx, :]
    testY = dataY_tmp[test_idx, :]

    for n in N_range:
        option.N = n
        for j in C_range:
            option.C = j
            for k in range(2,option.L):
                train_idx_val = cp.where(folds_index[:, i] == 0)[0]
                test_idx_val = cp.where(validation[:, i] == 1)[0]
                trainX_val = dataX[train_idx_val, :]
                trainY_val = dataY_tmp[train_idx_val, :]
                testX_val = dataX[test_idx_val, :]
                testY_val = dataY_tmp[test_idx_val, :]
                [model_tmp, train_acc_temp, test_acc_temp, training_time_temp, testing_time_temp] = MRVFL(trainX_val, trainY_val, testX_val, testY_val, option)
                if test_acc_temp > MAX_acc:
                    MAX_acc = test_acc_temp
                    option_best.acc_test = test_acc_temp
                    option_best.acc_train = train_acc_temp
                    option_best.C = option.C
                    option_best.N = option.N
                    option_best.L = k
                    option_best.scale = option.scale
                    option_best.nCV = i
                    print('Temp Best Option:{}'.format(option_best.__dict__))
                    del model_tmp
                    cp._default_memory_pool.free_all_blocks()
    [model_RVFL, train_acc, test_acc, train_time, test_time] = MRVFL(trainX,trainY,testX,testY,option_best)
    Models.append(model_RVFL)
    train_acc_result[i] = train_acc
    test_acc_result[i] = test_acc
    train_time_result[i] = train_time
    test_time_result[i] = test_time
    option_best = op(256, 32, 2**-6, 0.5, 0)
    del model_RVFL
    cp._default_memory_pool.free_all_blocks()


mean_train_acc = train_acc.mean()
mean_test_acc = test_acc.mean()
print('Train accuracy:{}\nTest accuracy:{}'.format(train_acc, test_acc))
print('Mean train accuracy:{}\nMean test accuracy:{}'.format(mean_train_acc, mean_test_acc))
save_result = open('Model_{}'.format(data_name),'wb')
pickle.dump(Models, save_result)
save_result.close()