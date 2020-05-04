import cupy as cp
import time
from function import *
from l2_weights import *
from majorityVoting import *
from model import model as mod
from sklearn.linear_model import ridge_regression as L2Regularization


def MRVFLtrain(trainX, trainY, option):
    selected_amount = 10

    rand_seed = cp.random.RandomState(option.seed)

    [n_sample, n_dims] = trainX.shape
    N = option.N
    L = option.L
    C = option.C
    s = option.scale

    A = []
    beta = []
    weights = []
    biases = []
    mu = []
    sigma = []
    ProbScores = []
    selected_features = []
    sfi = []

    A_input = trainX

    time_start = time.time()

    for i in range(L):

        if i == 0:
            w = s * 2 * rand_seed.rand(n_dims, N) - 1

        else:
            w = s * 2 * rand_seed.rand(n_dims + selected_amount * i + N, N) - 1

        b = s * rand_seed.rand(1, N)
        weights.append(w)
        biases.append(b)

        A_ = cp.matmul(A_input, w)
        # layer normalization
        A_mean = cp.mean(A_, axis=0)
        A_std = cp.std(A_, axis=0)
        A_ = (A_ - A_mean) / A_std
        mu.append(A_mean)
        sigma.append(A_std)

        A_ = A_ + cp.repeat(b, n_sample, 0)
        A_ = relu(A_)
        A_tmp = cp.concatenate([A_input, A_, cp.ones((n_sample, 1))], axis=1)
        # beta_ = L2Regularization(A_tmp,trainY,1/C).T
        beta_ = l2_weights(A_tmp, trainY, C, n_sample)
        significance = cp.linalg.norm(beta_, ord=1, axis=1)
        top_index = cp.argsort(significance)
        if i == 0:
            top_index = top_index[top_index - (n_dims + selected_amount) >= 0] - (n_dims + selected_amount)
        else:
            top_index = top_index[top_index - (n_dims + selected_amount * i + N) >= 0] - (
                    n_dims + selected_amount * i + N)
        # Replace

        # Updating Append List

        # Append
        selected_index = top_index[:selected_amount]
        A_selected = A_[:, selected_index]
        sfi.append(selected_index)
        selected_features.append(A_selected)

        A.append(A_tmp)
        beta.append(beta_)

        # print('Layer:{}'.format(i+1))
        sf = cp.concatenate(selected_features, axis=1)
        A_input = cp.concatenate([trainX, A_, sf], axis=1)

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

    model = mod(L, weights, biases, beta, mu, sigma, sfi)

    return model, TrainingAccuracy, Training_time
