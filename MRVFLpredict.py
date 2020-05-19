import cupy as cp
import time
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
    bi = model.bi
    TestingAccuracy= cp.zeros(L)

    A = []
    A_input = testX
    fs = []

    time_start = time.time()

    for i in range(L):
        A_ = cp.matmul(A_input, w[i])
        A_ = (A_ - mu[i]) / sigma[i]
        A_ = A_ + cp.repeat(b[i], n_sample, 0)
        A_ = selu(A_)        # Replace Relu to selu
        if i==0:
            A_tmp = cp.concatenate([testX, A_, cp.ones((n_sample, 1))], axis=1)
        else:
            A_tmp = cp.concatenate([testX, A_, sf_tmp, cp.ones((n_sample, 1))], axis=1)

        A.append(A_tmp)
        A_except_testX = A_tmp[:, n_dims: -1]
        A_ = A_except_testX[:,bi[i]]
        A_select = A_except_testX[:, sfi[i]]
        fs.append(A_select)

        sf_tmp = A_select
        #sf_tmp = cp.concatenate(fs, axis=1)

        ############ SETTINNG
        A_input = cp.concatenate([testX, sf_tmp, A_], axis=1)
        # A_input = cp.concatenate([testX, A_], axis=1)

        pred_result = cp.zeros((n_sample, i+1))
        for j in range(i+1):
            Ai = A[j]
            beta_temp = beta[j]
            predict_score = cp.matmul(Ai, beta_temp)
            predict_index = cp.argmax(predict_score, axis=1).ravel()
            # indx=indx.reshape(n_sample,1)
            pred_result[:, j] = predict_index
        TestingAccuracy_temp = majorityVoting(testY, pred_result)
        TestingAccuracy[i] = TestingAccuracy_temp
    '''
    pred_result = cp.zeros((n_sample, L))
    for i in range(L):
        Ai = A[i]
        beta_temp = beta[i]
        predict_score = cp.matmul(Ai, beta_temp)
        predict_index = cp.argmax(predict_score, axis=1).ravel()
        # indx=indx.reshape(n_sample,1)
        pred_result[:, i] = predict_index

    TestingAccuracy = majorityVoting(testY, predict_idx)
    '''
    time_end = time.time()

    Testing_time = time_end - time_start

    return TestingAccuracy, Testing_time
