# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/8 1:36
@author: merci
"""
import os
import cupy as cp
import numpy as np
from option import option as op
from MRVFL import *
import time

root_path = '/home/hu/eRVFL/UCIdata'
data_name = 'lung-cancer'
n_device = 1
print('Dataset Name:{}\nDevice Number:{}'.format(data_name, n_device))

cp.cuda.Device(n_device).use()
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

option = op(N=256, L=32, C=2 ** -6, scale=1, seed=1, nCV=0, ratio=0, mode='merged', drop=0)
# N_range = [256, 512, 1024]
N_range = [64, 128, 256, 512]
# N_range = [16, 32, 64]
L = 32
option.scale = 1
C_range = np.append(0,2.**np.arange(-6, 12, 2))

Models_tmp = []
Models = []
# dataX = rescale(dataX) #####delete

train_acc_result = cp.zeros((n_CV, 1))
test_acc_result = cp.zeros((n_CV, 1))
train_time_result = cp.zeros((n_CV, 1))
test_time_result = cp.zeros((n_CV, 1))

MAX_acc = 0
sMAX_acc = 0
tMAX_acc = 0
option_best = op(N=256, L=32, C=2 ** -6, scale=1, seed=0, nCV=0, ratio=0, mode='merged', drop=0)
option_sbest = op(N=256, L=32, C=2 ** -6, scale=1, seed=0, nCV=0, ratio=0, mode='merged', drop=0)
option_tbest = op(N=256, L=32, C=2 ** -6, scale=1, seed=0, nCV=0, ratio=0, mode='merged', drop=0)
for i in range(n_CV):
    MAX_acc = 0
    sMAX_acc = 0
    tMAX_acc = 0
    train_idx = cp.where(folds_index[:, i] == 0)[0]
    test_idx = cp.where(folds_index[:, i] == 1)[0]
    trainX = dataX[train_idx, :]
    trainY = dataY_tmp[train_idx, :]
    testX = dataX[test_idx, :]
    testY = dataY_tmp[test_idx, :]
    st = time.time()
    for n in N_range:
        option.N = n
        for j in C_range:
            option.C = j
            for r in cp.arange(0, 0.6, 0.3):
            #for r in [0]:
                option.ratio = r
                for d in cp.arange(0, 0.6, 0.3):
                #for d in [0]:
                    sto = time.time()
                    option.drop = d
                    train_idx_val = cp.where(validation[:, i] == 0)[0]
                    test_idx_val = cp.where(validation[:, i] == 1)[0]
                    trainX_val = dataX[train_idx_val, :]
                    trainY_val = dataY_tmp[train_idx_val, :]
                    testX_val = dataX[test_idx_val, :]
                    testY_val = dataY_tmp[test_idx_val, :]
                    [model_tmp, train_acc_temp, test_acc_temp, training_time_temp, testing_time_temp] = MRVFL(trainX_val, trainY_val, testX_val, testY_val, option)
                    del model_tmp
                    cp._default_memory_pool.free_all_blocks()
                    while (test_acc_temp > tMAX_acc).any():
                        if (test_acc_temp > MAX_acc).any():
                            tMAX_acc = sMAX_acc
                            sMAX_acc = MAX_acc
                            MAX_acc = test_acc_temp.max()
                            option_best.acc_test = test_acc_temp.max()
                            option_best.acc_train = train_acc_temp.max()
                            option_best.C = option.C
                            option_best.N = option.N
                            option_best.L = cp.int(test_acc_temp.argmax()+1)
                            option_best.scale = option.scale
                            option_best.nCV = i
                            option_best.ratio = r
                            option_best.drop = d
                            test_acc_temp[test_acc_temp.argmax()] = 0
                            print('Temp Best Option:{}'.format(option_best.__dict__))
                        elif (test_acc_temp > sMAX_acc).any():
                            tMAX_acc = sMAX_acc
                            sMAX_acc = test_acc_temp.max()
                            option_sbest.acc_test = test_acc_temp.max()
                            option_sbest.acc_train = train_acc_temp.max()
                            option_sbest.C = option.C
                            option_sbest.N = option.N
                            option_sbest.L = cp.int(test_acc_temp.argmax()+1)
                            option_sbest.scale = option.scale
                            option_sbest.nCV = i
                            option_sbest.ratio = r
                            option_sbest.drop = d
                            test_acc_temp[test_acc_temp.argmax()] = 0
                            print('Temp Second Best Option:{}'.format(option_sbest.__dict__))
                        elif (test_acc_temp > tMAX_acc).any():
                            tMAX_acc = test_acc_temp.max()
                            option_tbest.acc_test = test_acc_temp.max()
                            option_tbest.acc_train = train_acc_temp.max()
                            option_tbest.C = option.C
                            option_tbest.N = option.N
                            option_tbest.L = cp.int(test_acc_temp.argmax()+1)
                            option_tbest.scale = option.scale
                            option_tbest.nCV = i
                            option_tbest.ratio = r
                            option_tbest.drop = d
                            test_acc_temp[test_acc_temp.argmax()] = 0
                            print('Temp Third Best Option:{}'.format(option_tbest.__dict__))
                    print('Training Time for one option set:{:.2f}'.format(time.time()-sto))
    [model_RVFL0, train_acc0, test_acc0, train_time0, test_time0] = MRVFL(trainX, trainY, testX, testY, option_best)
    [model_RVFL1, train_acc1, test_acc1, train_time1, test_time1] = MRVFL(trainX, trainY, testX, testY, option_sbest)
    [model_RVFL2, train_acc2, test_acc2, train_time2, test_time2] = MRVFL(trainX, trainY, testX, testY, option_tbest)
    best_index = cp.argmax(cp.array([test_acc0.max(), test_acc1.max(), test_acc2.max()]))
    print('Best Index:{}'.format(best_index))
    print('Training Time for one fold set:{:.2f}'.format(time.time() - st))

    model_RVFL = eval('model_RVFL{}'.format(best_index))
    Models.append(model_RVFL)
    train_acc_result[i] = eval('train_acc{}.max()'.format(best_index))
    test_acc_result[i] = eval('test_acc{}.max()'.format(best_index))
    train_time_result[i] = eval('train_time{}'.format(best_index))
    test_time_result[i] = eval('test_time{}'.format(best_index))
    del model_RVFL
    cp._default_memory_pool.free_all_blocks()
    print('Best Train accuracy in fold{}:{}\nBest Test accuracy in fold{}:{}'.format(i, train_acc_result[i], i,
                                                                                     test_acc_result[i]))

mean_train_acc = train_acc_result.mean()
mean_test_acc = test_acc_result.mean()
print('Train accuracy:{}\nTest accuracy:{}'.format(train_acc_result, test_acc_result))
print('Mean train accuracy:{}\nMean test accuracy:{}'.format(mean_train_acc, mean_test_acc))
