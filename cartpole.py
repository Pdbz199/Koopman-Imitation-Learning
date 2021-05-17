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
U = np.load('optimal-agent/cartpole-actions.npy').reshape(1,-1)

#%% Training data
percent_training = 0.8
train_ind = int(np.around(X.shape[1]*percent_training))
X_train = X[:,:train_ind]
U_train = U[:,:train_ind]

#%% Data reformulation
X_tilde = np.append(X, U, axis=0)
state_dim = X_tilde.shape[0]
X_tilde_train = np.append(X_train, U_train, axis=0)
d = X_tilde_train.shape[0]
m = X_tilde_train.shape[1]
Y_tilde_train = np.append(np.roll(X_tilde_train,-1)[:, :-1], np.zeros((d,1)), axis=1)

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
data_transformed = feature_map_nystroem.fit_transform(X_tilde)
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

Psi_X_tilde = getPsiMatrix(psi, X_tilde_train)
Psi_Y_tilde = getPsiMatrix(psi, Y_tilde_train)

#%% Koopman (use generator?)
# || Y             - X B               ||
# || Psi_Y_tilde   - K Psi_X_tilde     ||
# || Psi_Y_tilde.T - Psi_X_tilde.T K.T ||
K = algorithms.rrr(Psi_X_tilde.T, Psi_Y_tilde.T).T

#%% To go from psi(x_tilde) -> x_tilde
B = algorithms.rrr(Psi_X_tilde.T, X_tilde_train.T)

#%% Imitation
def sample_action(x, u):
    x_tilde = np.append(x, u).reshape(-1,1)
    psi_x_tilde = psi(x_tilde)
    psi_x_tilde_prime = K @ psi_x_tilde
    x_tilde_prime = B.T @ psi_x_tilde_prime
    # x_prime = x_tilde_prime[:state_dim-1]
    action = x_tilde_prime[state_dim-1]
    return action

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
        current_state, reward, done, _ = env.step(action)
        action = sample_action(current_state, action)
        episode_reward += reward

    rewards.append(episode_reward)

env.close()

print("Average reward:", np.mean(rewards))

#%%