import cupy as cp
from scipy import stats
 

def majorityVoting(Y,pred_idx):

    Nsample= Y.shape[0]
    Ind_corrClass=cp.argmax(Y, axis=1)
    indx=cp.zeros(Nsample)
    for i in range (Nsample):
        Y=pred_idx[i,:]
        npy = cp.asnumpy(Y)
        indx[i]=stats.mode(npy)[0][0]
        
    acc=cp.mean(indx == Ind_corrClass)

    return acc
        