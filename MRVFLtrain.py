import cupy as cp
import numpy as np
import time
from function import *
from l2_weights import *
from majorityVoting import *
from model import model as mod
from inf_fs import *
import joblib
from joblib import Parallel,delayed



def MRVFLtrain(trainX, trainY, option):
    fs_mode = 'RIDGE'
    rand_seed = np.random.RandomState(2)

    [n_sample, n_dims] = trainX.shape
    N = option.N
    L = option.L
    C = option.C
    s = option.scale
    mode = option.mode
    ratio = option.ratio
    drop = option.drop

    TrainingAccuracy = cp.zeros(L)

    if mode == 'merged':
        drop_amount = cp.int(cp.floor(drop*N))
        selected_amount = cp.int(cp.floor(ratio*N))
        bi = []


    A = []
    beta = []
    weights = []
    biases = []
    mu = []
    sigma = []
    sfi = []
    fs = []

    A_input = trainX

    time_start = time.time()

    for i in range(L):

        if i == 0:
            w = s * 2 * cp.asarray(rand_seed.rand(n_dims, N)) - 1

        elif mode == 'merged':
            ######################### SETTING
            # w = s * 2 * cp.asarray(rand_seed.rand(n_dims - drop_amount + N, N)) - 1
            w = s * 2 * cp.asarray(rand_seed.rand(n_dims + selected_amount - drop_amount + N, N)) - 1
            # w = s * 2 * cp.asarray(rand_seed.rand(n_dims + selected_amount*i - drop_amount + N, N)) - 1


        b = s * cp.asarray(rand_seed.rand(1, N))
        weights.append(w)
        biases.append(b)

        A_ = cp.matmul(A_input, w)   # A_ should be 100 at any loop
        # layer normalization
        A_mean = cp.mean(A_, axis=0)
        A_std = cp.std(A_, axis=0)
        A_ = (A_ - A_mean) / A_std
        mu.append(A_mean)
        sigma.append(A_std)

        A_ = A_ + cp.repeat(b, n_sample, 0)
        A_ = selu(A_)
        if i == 0:
            A_tmp = cp.concatenate([trainX, A_, cp.ones((n_sample, 1))], axis=1)
        else:
            A_tmp = cp.concatenate([trainX, sf, A_, cp.ones((n_sample, 1))], axis=1)
        beta_ = l2_weights(A_tmp, trainY, C, n_sample)

        if fs_mode == 'LASSO':
            significance = cp.linalg.norm(beta_, ord=1, axis=1)
            ranked_index = cp.argsort(significance[n_dims:-1])
        if fs_mode == 'RIDGE':
            significance = cp.linalg.norm(beta_, ord=2, axis=1)
            ranked_index = cp.argsort(significance[n_dims:-1])
        elif fs_mode == 'MI':
            at = cp.asnumpy(A_tmp[:,n_dims:-1])
            ty = cp.asnumpy(cp.asarray([cp.argmax(i) for i in trainY]))
            mis = mi(at, ty)
            # with joblib.parallel_backend('loky'):
                # mis = Parallel(10)(delayed(mi)(at[:, i].reshape(-1, 1), ty) for i in range(N))
            # ranked_index = cp.argsort(cp.asarray(mis).ravel())
            ranked_index = cp.argsort(mis)
        elif fs_mode == 'INF':
            at = cp.asnumpy(A_tmp[:, n_dims:-1])
            rank, score = inf_fs(at)
            ranked_index = rank

        A.append(A_tmp)
        beta.append(beta_)

        selected_index = ranked_index[:selected_amount]  # chosen features, used in the next layers

        sfi.append(selected_index)
        left_amount = N - drop_amount
        left_index = ranked_index[:left_amount]
        A_except_trainX = A_tmp[:, n_dims: -1]
        A_selected = A_except_trainX[:, selected_index]
        fs.append(A_selected)
        A_ = A_except_trainX[:,left_index]

        ################### SETTING
        sf = A_selected
        # sf = cp.concatenate(fs, axis=1)

        ################### SETTING
        A_input = cp.concatenate([trainX, sf, A_], axis=1)
        # A_input = cp.concatenate([trainX,  A_], axis=1)


        bi.append(left_index)

        pred_result = cp.zeros((n_sample, i+1))
        for j in range(i+1):
            Ai = A[j]
            beta_temp = beta[j]
            predict_score = cp.matmul(Ai, beta_temp)
            predict_index = cp.argmax(predict_score, axis=1).ravel()
            # indx=indx.reshape(n_sample,1)
            pred_result[:, j] = predict_index
        TrainingAccuracy_temp = majorityVoting(trainY,pred_result)
        TrainingAccuracy[i] = TrainingAccuracy_temp

    '''    
    ## Calculate the training accuracy
    pred_result = cp.zeros((n_sample, L))
    for i in range(L):
        Ai = A[i]
        beta_temp = beta[i]
        predict_score = cp.matmul(Ai, beta_temp)
        predict_index = cp.argmax(predict_score, axis=1).ravel()
        # indx=indx.reshape(n_sample,1)
        pred_result[:, i] = predict_index
        
    TrainingAccuracy = majorityVoting(trainY, pred_result)
    '''

    time_end = time.time()
    Training_time = time_end - time_start



    model = mod(L, weights, biases, beta, mu, sigma, sfi, bi)

    return model, TrainingAccuracy, Training_time