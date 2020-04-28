import numpy as np
import time
import numpy.matlib
from function import *
from majorityVoting import *

def MRVFLpredict(testX,testY,model):


    [Nsample,Nfea]= testX.shape


    w=model.w
    b=model.b
    beta=model.beta
    mu=model.mu
    sigma=model.sigma
    L= model.L


    A=[]
    A_input=testX

    time_start=time.time()

    for i in range(L):
        A1=np.matmul(A_input,w[i])
        A1 = (A1-mu[i])/sigma[i]
        A1 = A1+numpy.matlib.repmat(b[i],Nsample,1)
        A1=relu(A1)
        A1_temp1 = np.concatenate([A_input,A1,np.ones((Nsample,1))],axis=1)


        A.append(A1_temp1)

        #clear A1 A1_temp1 A1_temp2 beta1
        A_input = np.concatenate([testX,A1],axis=1)


    ## Calculate the training accuracy
    pred_idx=np.array([Nsample,L])
    for i in range(L):
        A_temp=A[i]
        beta_temp=beta[i]
        trainY_temp=np.matmul(A_temp,beta_temp)
        indx=np.argmax(trainY_temp,axis=1)
        indx=indx.reshape(Nsample,1)
        if i==0:
            pred_idx=indx
        else:
            pred_idx=np.concatenate([pred_idx,indx],axis=1)


    TestingAccuracy = majorityVoting(testY,pred_idx)



    time_end=time.time()

    Testing_time = time_end-time_start

    return TestingAccuracy,Testing_time


