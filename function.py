import cupy as cp

def relu(x):
    return cp.maximum(0, x)

def softmax(x):
    num=cp.exp(x)
    dem=cp.sum(num, axis=1).reshape(-1, 1)
    dem=cp.repeat(dem, x.shape[1], 0)
    return num/dem

def sigmoid(x):
    x = 1 / (1 + cp.exp(-x))
    return x

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*cp.where(x >= 0, x, alpha*(cp.exp(x)-1))