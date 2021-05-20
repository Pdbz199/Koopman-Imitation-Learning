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
import matplotlib.pyplot as plt
from continuous_cartpole import CartPoleEnv

env = CartPoleEnv()
# env.reset()

#%% Data
X = np.load('optimal-agent/cartpole-states.npy').T
U = np.load('optimal-agent/cartpole-actions.npy').astype(np.float64).reshape(1,-1)

state_dim = X.shape[0]

#%% Training data
percent_training = 0.8
train_ind = int(np.around(X.shape[1]*percent_training))

X_train = X[:,:train_ind]
U_train = U[:,:train_ind]

X_validation = X[:,train_ind:]
U_validation = U[:,train_ind:]

#%% Median trick
# from http://alex.smola.org/teaching/kernelcourse/day_2.pdf
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
from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=gamma, random_state=1, n_components=400)
X_features = rbf_feature.fit_transform(X)
def psi(x):
    return X_features.T @ x

#%% Nystroem
# from sklearn.kernel_approximation import Nystroem
# feature_map_nystroem = Nystroem(gamma=gamma, random_state=1, n_components=state_dim)
# data_transformed = feature_map_nystroem.fit_transform(X)
# def psi(x):
#     return data_transformed @ x

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

#%% Imitation
@nb.njit(fastmath=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_action(x):
    action = (K @ psi(x))[0]
    return int(np.around(sigmoid(action)))

#%% Error testing
# print("Training error:")
# norms = []
# for state_index in range(X_train.shape[1]):
#     true_action = U_train[:,state_index]
#     predicted_action = np.array([get_action(X_train[:,state_index])])
#     l2_norm = helpers.l2_norm(true_action, predicted_action)
#     norms.append(l2_norm)
# norms = np.array(norms)
# print("Mean difference:", np.mean(norms))
# accuracy = (norms.shape[0]-np.count_nonzero(norms)) / norms.shape[0]
# print("Percent accuracy:", accuracy * 100)

# print("\nValidation error:")
# norms = []
# for state_index in range(X_validation.shape[1]):
#     true_action = U_validation[:,state_index]
#     predicted_action = np.array([get_action(X_validation[:,state_index])])
#     l2_norm = helpers.l2_norm(true_action, predicted_action)
#     norms.append(l2_norm)
# norms = np.array(norms)
# print("Mean difference:", np.mean(norms))
# accuracy = (norms.shape[0]-np.count_nonzero(norms)) / norms.shape[0]
# print("Percent accuracy:", accuracy * 100)

# # Both Nystroem and RBF Sampler give the same results:
# # Training error:
# # Mean difference: 0.1479375
# # Percent accuracy: 85.20625
# # Validation error:
# # Mean difference: 0.143
# # Percent accuracy: 85.7

# #%% Decompose Koopman
# u, s, vh = helpers.SVD(K)
# sv_functions = lambda l, x: np.inner(vh[l], psi(x))

# #%%
# plt.imshow(env.render(vh[0], 'rgb_array'))
# #%%
# plt.imshow(env.render(vh[1], 'rgb_array'))
# #%%
# plt.imshow(env.render(vh[2], 'rgb_array'))
# #%%
# plt.imshow(env.render(vh[3], 'rgb_array'))

# #%%
# # Xs = np.append(X_train[0,:100].reshape(1,-1), np.array([X_train[1:,0] for i in range(100)]).T, axis=0).T
# Xs = X[:,:100].T
# sv_function_0_outputs = np.array([sv_functions(0, x) for x in Xs])
# sv_function_1_outputs = np.array([sv_functions(1, x) for x in Xs])
# sv_function_2_outputs = np.array([sv_functions(2, x) for x in Xs])
# sv_function_3_outputs = np.array([sv_functions(3, x) for x in Xs])

# #%%
# plt.plot(sv_function_0_outputs, marker='.', linestyle='')
# plt.show()
# #%%
# plt.plot(sv_function_1_outputs, marker='.', linestyle='')
# plt.show()
# #%%
# plt.plot(sv_function_2_outputs, marker='.', linestyle='')
# plt.show()
# #%%
# plt.plot(sv_function_3_outputs, marker='.', linestyle='')
# plt.show()

#%% Run in environment
episodes = 2
rewards = []
for episode in range(episodes):
    # reset environment and variables
    episode_reward = 0
    current_state = env.reset()
    action = env.action_space.sample()
    done = False

    while done == False:
        env.render()

        action = get_action(current_state)
        current_state, reward, done, _ = env.step(action)
        episode_reward += reward

    rewards.append(episode_reward)

env.close()

print("Average reward:", np.mean(rewards))
# Average reward per episode: 3771.22

#%%