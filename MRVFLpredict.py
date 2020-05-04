import cupy as cp
import time
import numpy.matlib
from function import *
from majorityVoting import *


def MRVFLpredict(testX, testY, model):
    [n_sample, n_dims] = testX.shape

    w = model.w
    b = model.b
    beta = model.beta
    mu = model.mu
    sigma = model.sigma
    L = model.L
    sfi = model.sfi

    A = []
    A_input = testX
    selected_features = []

    time_start = time.time()

    for i in range(L):
        A_ = cp.matmul(A_input, w[i])
        A_ = (A_ - mu[i]) / sigma[i]
        A_ = A_ + cp.repeat(b[i], n_sample, 0)
        A_ = relu(A_)
        A_tmp = cp.concatenate([A_input, A_, cp.ones((n_sample, 1))], axis=1)
        sf_tmp = A_[:, sfi[i]]

        selected_features.append(sf_tmp)
        A.append(A_tmp)
        sfs = cp.concatenate(selected_features, axis=1)
        A_input = cp.concatenate([testX, A_, sfs], axis=1)
        # print('layer:{}'.format(i+1))

    pred_idx = cp.array([n_sample, L])
    for i in range(L):
        A_temp = A[i]
        beta_temp = beta[i]
        trainY_temp = cp.matmul(A_temp, beta_temp)
        indx = cp.argmax(trainY_temp, axis=1)
        indx = indx.reshape(n_sample, 1)
        if i == 0:
            pred_idx = indx
        else:
            pred_idx = cp.concatenate([pred_idx, indx], axis=1)

    TestingAccuracy = majorityVoting(testY, pred_idx)

    time_end = time.time()

    Testing_time = time_end - time_start

    return TestingAccuracy, Testing_time
