

import numpy as np
import numpy.linalg
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import chi
from collections import namedtuple


from sklearn.linear_model import Ridge

Params = namedtuple('Params', ['W_rff', 'b_rff', 'U', 'U_b', 'W_FE_F', 'b_FE_F', 'W_pred', 'b_pred', 'q_1'])

class RFF_Projection:
    
    def __init__(self, kernel_width, seed, nRFF, n_feat):
        rbf_sampler = RBFSampler(gamma=kernel_width, random_state=seed, n_components=nRFF)
        rbf_sampler.fit(np.zeros((1, n_feat)))
      
        self.W = rbf_sampler.random_weights_
        self.b = rbf_sampler.random_offset_
        self.nRFF = nRFF
        
    def project(self, x):
        return np.cos((x.T.dot(self.W) + self.b).T)*np.sqrt(2.)/np.sqrt(self.nRFF)

# calculates the kahtri-rao product of two matrices
def khatriRaoProduct(X,Y):
    XY = np.zeros((X.shape[0]*Y.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        XY[:,i] = np.kron(X[:,i], Y[:,i].T).reshape(X.shape[0]*Y.shape[0])
    return XY

# perform ridge regression to X from Y with ridge regression parameter lr
def ridgeRegressionBiased(X, Y, lr):
    
    ridge = Ridge(fit_intercept=True, alpha=lr, random_state=0, normalize=True, tol=1e-20)
    ridge.fit(Y.T,X.T)
    W = ridge.coef_
    b = ridge.intercept_.reshape((-1,1))
    
    return W, b

# split data into P, F, FS, Obs
def featurize(data, k):
    
    nFeat = data.shape[0]
    nData = data.shape[1]
    
    stacked = np.zeros((nFeat*(2*k+1),nData-2*k))
    for i in range(2*k+1):
        stacked[nFeat*i:nFeat*(i+1),:] = data[:, i:nData-(2*k-i)]
    
    P = stacked[:nFeat*k, :]
    F = stacked[nFeat*k:2*nFeat*k, :]
    Obs = stacked[k*nFeat:(k+1)*nFeat, :]
    FS = stacked[(k+1)*nFeat:, :]
    
    return (Obs, P, F, FS)
    
    
def svd_projection(x, y, nSvd, whiten=False):
    
    # Calculate Covariance Matrices
    mu_x = np.mean(x,axis=1).reshape((-1,1))
    mu_y = np.mean(y,axis=1).reshape((-1,1))
    if whiten:
        x = x - mu_x
        y = y - mu_y
    C = x.dot(y.T)/x.shape[1]
    
    # Calculate matrix of singular vectors
    U, S, V = sp.sparse.linalg.svds(C, nSvd)
    
    # whiten projection
    for i in range(S.size):
        if S[i] > 0:
            S[i] = 1/np.sqrt(S[i])
            
    W = U.dot(np.diag(S))
    b = -W.T.dot(x).mean()
    
    return W, b 

def two_stage_regression(raw_data, 
                         data,
                         kernel_width_Obs, kernel_width_P, kernel_width_F, 
                         seed, 
                         nRFF_Obs, nRFF_P, nRFF_F,
                         dim_Obs, dim_P, dim_F, 
                         reg_rate, 
                         obs_window):
    
    
    n_feat = data.shape[0]
    n_data = data.shape[1]
    
    # generate features of history/future
    print("featurizing")
    Obs, P, F, FS = featurize(data, obs_window)
    
    # split raw data 
    raw_F = raw_data[:,obs_window:n_data-obs_window]
    raw_FS = raw_data[:,obs_window+1:n_data-obs_window+1]
    raw_P = raw_data[:,:n_data-obs_window-1]

    # project into RBF space
    print("project into rff space")
    
    Obs_rff = Obs
    P_rff = P
    F_rff = F
    FS_rff = FS
    
    Obs_Proj = RFF_Projection(kernel_width_Obs, seed*1, nRFF_Obs, Obs_rff.shape[0])
    Obs_rff = Obs_Proj.project(Obs_rff)
    
    P_Proj = RFF_Projection(kernel_width_P, seed*2, nRFF_P, P_rff.shape[0])
    P_rff = P_Proj.project(P_rff)
    
    F_Proj = RFF_Projection(kernel_width_F, seed*3, nRFF_F, F_rff.shape[0])
    F_rff = F_Proj.project(F_rff)
    FS_rff = F_Proj.project(FS_rff)
    
    # project the data onto top few singular vectors
    print('project onto svd')
    U_Obs, U_Obs_b = svd_projection(Obs_rff, P_rff, dim_Obs)
    Obs_U = U_Obs.T.dot(Obs_rff)
    
    U_P, U_P_b = svd_projection(P_rff, F_rff, dim_P, whiten=True)
    P_U = U_P.T.dot(P_rff) #todo
    
    U_F, U_F_b = svd_projection(F_rff, P_rff, dim_F, whiten=True)
    F_U = U_F.T.dot(F_rff) #todo
    FS_U = U_F.T.dot(FS_rff) # todo
    
    # calculate extended future from shifted future and observation
    print('extended future')
    FE_U = khatriRaoProduct(FS_U, Obs_U)
    
    # stage 1 regression
    print('stage 1')
    W_F_P_bias, b_F_P_bias = ridgeRegressionBiased(F_U, P_U, reg_rate)
    W_FE_P_bias, b_FE_P_bias = ridgeRegressionBiased(FE_U, P_U, reg_rate)
    
    # apply stage 1 regression to data to generate input for stage2 regression
    print('apply stage 1')
    E_F_bias = W_F_P_bias.dot(P_U) + b_F_P_bias
    E_FE_F_bias = W_FE_P_bias.dot(P_U) + b_FE_P_bias
    
    # stage 2 regression
    print('stage 2')
    W_FE_F, b_FE_F = ridgeRegressionBiased(E_FE_F_bias, E_F_bias, reg_rate)
    
    # calculate initial state
    print('apply stage 2')    
    q_1 = np.mean(E_F_bias,axis=1).reshape((-1,1))

    # perform filtering using learned model    
    s = np.zeros((F_U.shape[0],F_U.shape[1]+1))
    s[:,0] = q_1.reshape((dim_P))
    for i in range(F_U.shape[1]):
        W = W_FE_F.dot(s[:,i]) + b_FE_F.reshape(-1)
        W = W.reshape((dim_F, dim_P))
        s[:,i+1] = W.dot(Obs_U[:,i])
        s[:,i+1] = s[:,i+1]/np.linalg.norm(s[:,i+1])
    s = s[:,1:]
    
    # regress from state to predictions
    logreg = LogisticRegression()
    F_raw_augmented = raw_F.flatten()
    unq_labels = set(F_raw_augmented.tolist())
    idx = -1
    for i in range(n_feat):
        if i not in unq_labels:
            F_raw_augmented[idx] = i
            idx -= 1
            
    y = raw_FS.reshape((raw_FS.size))
    y[0:49] = np.arange(49)

    logreg.fit(s.T,y)
    W_pred = logreg.coef_
    b_pred = logreg.intercept_.reshape((-1,1))
    
    tsr_params = Params(Obs_Proj.W,
                       Obs_Proj.b,
                       U_Obs,
                       False,
                       W_FE_F,
                       b_FE_F,
                       W_pred,
                       b_pred,
                       q_1)

    return tsr_params