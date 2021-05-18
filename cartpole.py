#%% Imports
import gym
import helpers
import observables
import torch
import domain
import kernels
import numpy as np
import scipy as sp
import numba as nb
import algorithms
from continuous_cartpole import CartPoleEnv

#%% Data
X = np.load('optimal-agent/cartpole-states.npy').T
U = np.load('optimal-agent/cartpole-actions.npy').astype(np.float64).reshape(1,-1)

state_dim = X.shape[0]

#%% Training data
percent_training = 0.8
train_ind = int(np.around(X.shape[1]*percent_training))
X_train = X[:,:train_ind]
U_train = U[:,:train_ind]

#%% Median trick
num_pairs = 1000
pairwise_distances = []
for _ in range(num_pairs):
    i, j = np.random.choice(np.arange(X.shape[1]), 2)
    x_i = X[:,i]
    x_j = X[:,j]
    pairwise_distances.append(np.linalg.norm(x_i - x_j))
pairwise_distances = np.array(pairwise_distances)
gamma = np.quantile(pairwise_distances, 0.9)

#%% RBF sampler kernel
# from sklearn.kernel_approximation import RBFSampler
# rbf_feature = RBFSampler(gamma=gamma, random_state=1)
# X_features = rbf_feature.fit_transform(X_tilde)
# def psi(x):
#     return X_features.T @ x

#%% Nystroem
from sklearn.kernel_approximation import Nystroem
feature_map_nystroem = Nystroem(gamma=gamma, random_state=1, n_components=state_dim)
data_transformed = feature_map_nystroem.fit_transform(X)
def psi(x):
    return data_transformed @ x

#%% Psi matrices
def getPsiMatrix(psi, X):
    k = psi(X[:,0].reshape(-1,1)).shape[0]
    m = X.shape[1]
    matrix = np.empty((k,m))
    for col in range(m):
        matrix[:, col] = psi(X[:, col].reshape(-1,1))[:, 0]
    return matrix

Psi_X = getPsiMatrix(psi, X)

#%% Koopman (use generator?)
# || Y - X B           ||
# || U - K Psi_X       ||
# || U.T - Psi_X.T K.T ||
K = algorithms.rrr(Psi_X.T, U.T).T

#%% To go from psi(x_tilde) -> x_tilde
# B = algorithms.rrr(Psi_X_tilde.T, X_tilde_train.T)

#%% Imitation
@nb.njit(fastmath=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_action(x):
    action = (K @ psi(x))[0]
    return int(np.around(sigmoid(action)))

#%% Run in environment
env = CartPoleEnv()
episodes = 100
rewards = []
for episode in range(episodes):
    # reset environment and variables
    episode_reward = 0
    current_state = env.reset()
    action = env.action_space.sample()
    done = False

    while done == False:
        # env.render()

        action = get_action(current_state)
        current_state, reward, done, _ = env.step(action)
        episode_reward += reward

    rewards.append(episode_reward)

env.close()

print("Average reward:", np.mean(rewards))

#%%