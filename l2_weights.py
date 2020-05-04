import cupy as cp

def l2_weights(X, target, ridge_parameter, n_sample):
    '''
    eig_values,T= cp.linalg.eig(cp.matmul(X.T,X))
    Z = cp.dot(X,T)
    eig_diag = cp.diag(eig_values)
    alpha_ols = cp.linalg.multi_dot([T,cp.linalg.inv(cp.dot(Z.T,Z)),Z.T,target])
    error = target-cp.dot(Z,alpha_ols)
    # sigma = cp.dot(error.T,error)/(n_sample-10)
    sigma = (cp.dot(target.T,target)-cp.linalg.multi_dot([alpha_ols.T,Z.T,target]))/(n_sample-11)
    alpha_ob =cp.expand_dims(alpha_ols,1)
    K = cp.linalg.det(sigma) / cp.matmul(alpha_ob,cp.swapaxes(alpha_ob,1,2)).ravel()
    # alpha = cp.linalg.multi_dot([T, cp.linalg.inv(eig_diag + cp.diag(K)), Z.T, target])
    alpha = cp.linalg.multi_dot([T, (cp.eye(X.shape[1])-cp.dot(K,cp.linalg.inv(eig_diag+K))),alpha_ols])
    '''
    if X.shape[1]<n_sample:
        beta = cp.matmul(cp.matmul(cp.linalg.inv(cp.eye(X.shape[1]) / ridge_parameter + cp.matmul(X.T, X)), X.T), target)
    else:
        beta = cp.matmul(X.T, cp.matmul(cp.linalg.inv(cp.eye(X.shape[0]) / ridge_parameter + cp.matmul(X, X.T)), target))

    return beta