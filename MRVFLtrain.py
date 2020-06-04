import cupy as cp
import time
from function import *
from l2_weights import *
from majorityVoting import *
from model import model as mod


def MRVFLtrain(trainX, trainY, option):
    # mode = 'append'
    rand_seed = cp.random.RandomState(option.seed)

    [n_sample, n_dims] = trainX.shape
    N = option.N
    L = option.L
    C = option.C
    s = option.scale
    mode = option.mode
    ratio = option.ratio

    if mode == 'append':
        selected_amount = 10
    elif mode == 'replace':
        ratio = 0.2
        selected_amount = cp.int(cp.floor(ratio*N))
        tfi = []
        tfs = []
        bi = []
    elif mode == 'updated':
        selected_amount = cp.int(cp.floor(ratio*N))
        # selected_amount = 20

    A = []
    beta = []
    weights = []
    biases = []
    mu = []
    sigma = []
    selected_features = []
    sfi = []

    A_input = trainX

    time_start = time.time()

    for i in range(L):

        if i == 0:
            w = s * 2 * rand_seed.rand(n_dims, N) - 1

        elif mode == 'append':
            w = s * 2 * rand_seed.rand(n_dims + selected_amount * i + N, N) - 1

        elif mode == 'replace':
            w = s * 2 * rand_seed.rand(n_dims + N, N) - 1

        elif mode == 'updated':
            w = s * 2 * rand_seed.rand(n_dims + selected_amount + N, N) - 1

        b = s * rand_seed.rand(1, N)
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
        #A_tmp = cp.concatenate([A_input, A_, cp.ones((n_sample, 1))], axis=1)  # Double the input, append index should be (n_dims + selected_amount * i +N)
        if i == 0 or mode == 'replace':
            A_tmp = cp.concatenate([trainX, A_, cp.ones((n_sample, 1))], axis=1)
        else:
            A_tmp = cp.concatenate([trainX, sf, A_, cp.ones((n_sample, 1))], axis=1)
        beta_ = l2_weights(A_tmp, trainY, C, n_sample)
        significance = cp.linalg.norm(beta_, ord=1, axis=1)
        ranked_index = cp.argsort(significance)
        if i == 0:
            ranked_index = ranked_index[ranked_index - n_dims >= 0] - (n_dims + selected_amount)

        elif mode == 'append':
            ranked_index = ranked_index[ranked_index - (n_dims + selected_amount * i) >= 0] - (n_dims + selected_amount * i)

        elif mode == 'updated':
            ranked_index = ranked_index[ranked_index - (n_dims + selected_amount) >= 0] - (n_dims + selected_amount)

        elif mode == 'replace':
            ranked_index = ranked_index[ranked_index - n_dims >= 0] - n_dims

        A.append(A_tmp)
        beta.append(beta_)

        selected_index = ranked_index[:selected_amount]  # chosen features, used in the next layers
        A_selected = A_[:, selected_index]
        sfi.append(selected_index)
        selected_features.append(A_selected)

        if mode == 'append':
            sf = cp.concatenate(selected_features, axis=1)
            A_input = cp.concatenate([trainX, sf, A_], axis=1)
        elif mode == 'updated':
            sf = A_selected
            A_input = cp.concatenate([sf, trainX, A_], axis=1)
        elif mode == 'replace':
            if i == 0:
                A_input = cp.concatenate([trainX, A_], axis=1)
                top_features = A_selected
                tfi.append(selected_index)
                tfs.append(top_features)
            else:
                bottom_index = ranked_index[-selected_amount:]
                A_[:,bottom_index] = top_features
                A_tmp_replaced = cp.concatenate([trainX, A_, cp.ones((n_sample, 1))], axis=1)
                beta_replaced = l2_weights(A_tmp_replaced, trainY, C, n_sample)
                magnitude = cp.linalg.norm(beta_replaced, ord=1, axis=1)
                ranked_replaced_index = cp.argsort(magnitude)
                ranked_replaced_index = ranked_replaced_index[ranked_replaced_index - n_dims >= 0] - n_dims
                top_replaced_index = ranked_replaced_index[:selected_amount]
                top_features = A_[:, top_replaced_index]
                tfs.append(top_features)
                tfi.append(top_replaced_index)
                bi.append(bottom_index)
                A_input = cp.concatenate([trainX, A_], axis=1)
        #print('layer{}'.format(i+1))

    time_end = time.time()
    Training_time = time_end - time_start

    ## Calculate the training accuracy
    pred_result = cp.random.rand(n_sample, L)
    for i in range(L):
        Ai = A[i]
        beta_temp = beta[i]
        predict_score = cp.matmul(Ai, beta_temp)
        predict_index = cp.argmax(predict_score, axis=1).ravel()
        # indx=indx.reshape(n_sample,1)
        pred_result[:, i] = predict_index

    TrainingAccuracy = majorityVoting(trainY, pred_result)
    if mode == 'replace':
        model = mod(L, weights, biases, beta, mu, sigma, sfi, mode, tfi, bi)
    else:
        model = mod(L, weights, biases, beta, mu, sigma, sfi, mode)

    return model, TrainingAccuracy, Training_time
