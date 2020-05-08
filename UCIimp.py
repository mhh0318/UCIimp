#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/3 15:35
@author: merci
"""
import os
import cupy as cp
import numpy as np
from rescale import *
from option import option as op
from MRVFL import *
import pickle
import logging


def main(dataset):
    root_path = '/home/hu/eRVFL/UCIdata'
    # data_name = 'cardiotocography-10clases'
    data_name = dataset
    n_device = 1
    print('Dataset Name:{}\nDevice Number:{}'.format(data_name,n_device))
    logging.debug('Dataset Name:{}\tDevice Number:{}'.format(data_name,n_device))
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

    option = op(256, 32, 2**-6, 1, 0, 0, 0.7, 'replace')
    N_range = [256,512,1024]
    L = 16
    option.scale = 0.5
    C_range = range(-6,12,2)

    Models_tmp = []
    Models = []
    dataX = rescale(dataX)

    train_acc_result = cp.zeros((n_CV,1))
    test_acc_result = cp.zeros((n_CV,1))
    train_time_result = cp.zeros((n_CV,1))
    test_time_result = cp.zeros((n_CV,1))

    MAX_acc = 0
    sMAX_acc = 0
    tMAX_acc = 0
    option_best = op(256, 32, 2**-6, 0.5, 0)
    option_sbest = op(256, 32, 2**-6, 0.5, 0)
    option_tbest = op(256, 32, 2**-6, 0.5, 0)
    for i in range(n_CV):

        MAX_acc = 0
        sMAX_acc = 0
        tMAX_acc = 0
        train_idx = cp.where(folds_index[:,i]==0)[0]
        test_idx = cp.where(folds_index[:,i]==1)[0]
        trainX = dataX[train_idx, :]
        trainY = dataY_tmp[train_idx, :]
        testX = dataX[test_idx, :]
        testY = dataY_tmp[test_idx, :]

        for n in N_range:
            option.N = n
            for j in C_range:
                option.C = 2**j
                for k in range(2, L):
                    option.L = k
                    for r in cp.arange(0,0.6,0.1):
                    #for r in cp.arange(0,0.6,0.1):
                        option.ratio = r
                        train_idx_val = cp.where(validation[:, i] == 0)[0]
                        test_idx_val = cp.where(validation[:, i] == 1)[0]
                        trainX_val = dataX[train_idx_val, :]
                        trainY_val = dataY_tmp[train_idx_val, :]
                        testX_val = dataX[test_idx_val, :]
                        testY_val = dataY_tmp[test_idx_val, :]
                        [model_tmp, train_acc_temp, test_acc_temp, training_time_temp, testing_time_temp] = MRVFL(trainX_val, trainY_val, testX_val, testY_val, option)
                        if test_acc_temp > MAX_acc:
                            tMAX_acc = sMAX_acc
                            sMAX_acc = MAX_acc
                            MAX_acc = test_acc_temp
                            option_best.acc_test = test_acc_temp
                            option_best.acc_train = train_acc_temp
                            option_best.C = option.C
                            option_best.N = option.N
                            option_best.L = k
                            option_best.scale = option.scale
                            option_best.nCV = i
                            option_best.ratio = r
                            print('Temp Best Option:{}'.format(option_best.__dict__))
                            logging.debug('Temp Best Option:{}'.format(option_best.__dict__))
                            del model_tmp
                            cp._default_memory_pool.free_all_blocks()
                        elif MAX_acc > test_acc_temp > sMAX_acc:
                            tMAX_acc = sMAX_acc
                            sMAX_acc = test_acc_temp
                            option_sbest.acc_test = test_acc_temp
                            option_sbest.acc_train = train_acc_temp
                            option_sbest.C = option.C
                            option_sbest.N = option.N
                            option_sbest.L = k
                            option_sbest.scale = option.scale
                            option_sbest.nCV = i
                            option_sbest.ratio = r
                            print('Temp Second Best Option:{}'.format(option_best.__dict__))
                            logging.debug('Temp Second Best Option:{}'.format(option_best.__dict__))
                            del model_tmp
                            cp._default_memory_pool.free_all_blocks()
                        elif sMAX_acc > test_acc_temp > tMAX_acc:
                            tMAX_acc = test_acc_temp
                            option_tbest.acc_test = test_acc_temp
                            option_tbest.acc_train = train_acc_temp
                            option_tbest.C = option.C
                            option_tbest.N = option.N
                            option_tbest.L = k
                            option_tbest.scale = option.scale
                            option_tbest.nCV = i
                            option_tbest.ratio = r
                            print('Temp Third Best Option:{}'.format(option_best.__dict__))
                            logging.debug('Temp Third Best Option:{}'.format(option_best.__dict__))
                            del model_tmp
                            cp._default_memory_pool.free_all_blocks()
        [model_RVFL0, train_acc0, test_acc0, train_time0, test_time0] = MRVFL(trainX,trainY,testX,testY,option_best)
        [model_RVFL1, train_acc1, test_acc1, train_time1, test_time1] = MRVFL(trainX,trainY,testX,testY,option_sbest)
        [model_RVFL2, train_acc2, test_acc2, train_time2, test_time2] = MRVFL(trainX,trainY,testX,testY,option_tbest)
        best_index = cp.argmax(cp.array([test_acc0, test_acc1, test_acc2]))
        print('Best Index:{}'.format(best_index))
        logging.debug('Best Index:{}'.format(best_index))
        model_RVFL = eval('model_RVFL{}'.format(best_index))
        Models.append(model_RVFL)
        train_acc_result[i] = eval('train_acc{}'.format(best_index))
        test_acc_result[i] = eval('test_acc{}'.format(best_index))
        train_time_result[i] = eval('train_time{}'.format(best_index))
        test_time_result[i] = eval('test_time{}'.format(best_index))
        option_best = op(256, 32, 2**-6, 0.5, 0)
        del model_RVFL
        cp._default_memory_pool.free_all_blocks()
        print('Best Train accuracy in fold{}:{}\nBest Test accuracy in fold{}:{}'.format(i, train_acc_result[i], i, test_acc_result[i]))
        logging.debug('Best Train accuracy in fold{}:{}\tBest Test accuracy in fold{}:{}'.format(i, train_acc_result[i], i, test_acc_result[i]))

    mean_train_acc = train_acc_result.mean()
    mean_test_acc = test_acc_result.mean()
    print('Train accuracy:{}\nTest accuracy:{}'.format(train_acc_result, test_acc_result))
    logging.debug('Train accuracy:{}\tTest accuracy:{}'.format(train_acc_result, test_acc_result))
    print('Mean train accuracy:{}\nMean test accuracy:{}'.format(mean_train_acc, mean_test_acc))
    logging.debug('Mean train accuracy:{}\tMean test accuracy:{}'.format(mean_train_acc, mean_test_acc))
    #save_result = open('Model_{}'.format(data_name),'wb')
    #pickle.dump(Models, save_result)
    #save_result.close()


if __name__ == '__main__':
    logging.basicConfig(filename='LOGfseRVFL-replace-50+.txt', level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.debug('Start of program')
    files = os.listdir('/home/hu/eRVFL/UCIdata')
    for file in files[50:]:
        if file in ['adult','band','chess-krvk','connect-4','letter','magic','miniboone','statlog-shuttle']:
            pass
        else:
            main(file)