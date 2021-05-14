#%% Imports
import algorithms
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

#%% Load data
X = np.load('optimal-agent/cartpole-states.npy').T
U = np.load('optimal-agent/cartpole-actions.npy').reshape(1,-1)

#%% X_tilde
X_tilde = np.append(X, U, axis=0)[:, :1000]
d = X_tilde.shape[0]
m = X_tilde.shape[1]
Y_tilde = np.append(np.roll(X_tilde,-1)[:, :-1], np.zeros((d,1)), axis=1)

#%% RBF Sampler
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X_tilde)
psi = lambda x: X_features.T @ x.reshape(-1,1)

def getPsiMatrix(psi, X):
    matrix = np.empty(())
    for :
        matrix psi(x)
    return 

Psi_X_tilde = getPsiMatrix(psi, X_tilde)
Psi_Y_tilde = getPsiMatrix(psi, Y_tilde)

#%% Koopman
# || Y - X B ||
# || Psi_Y_tilde - K Psi_X_tilde ||
K = algorithms.rrr(Psi_Y_tilde.T, Psi_X_tilde.T)

#%%
B = algorithms.SINDy(Psi_X_tilde.T, X_tilde.T, X_features.shape[0])

#%%
