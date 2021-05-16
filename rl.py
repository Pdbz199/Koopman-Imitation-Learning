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
from pytorch_mppi import mppi
from cartpole_reward import cartpoleCost
from continuous_cartpole import CartPoleEnv

#%% State constants
N_SAMPLES = 300
TIMESTEPS = 10
noise_sigma = torch.tensor(1.0, dtype=torch.double)
action_bounds = [0.0, 1.0]
lambda_ = 1.0

#%% Data
X = np.load('optimal-agent/cartpole-states.npy').T
U = np.load('optimal-agent/cartpole-actions.npy').reshape(1,-1)

#%% Data reformulation
X_tilde = np.append(X, U, axis=0)[:, :1000]
d = X_tilde.shape[0]
m = X_tilde.shape[1]
Y_tilde = np.append(np.roll(X_tilde,-1)[:, :-1], np.zeros((d,1)), axis=1)

#%% RBF sampler kernel
from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=0.7, random_state=1)
X_features = rbf_feature.fit_transform(X_tilde)
k = X_features.shape[1]
def psi(x):
    return X_features.T @ x

#%% Psi matrices
def getPsiMatrix(psi, X):
    matrix = np.empty((k,m))
    for col in range(m):
        matrix[:, col] = psi(X[:, col].reshape(-1,1))[:, 0]
    return matrix

Psi_X_tilde = getPsiMatrix(psi, X_tilde)
Psi_Y_tilde = getPsiMatrix(psi, Y_tilde)

#%% Koopman (use generator?)
# || Y             - X B               ||
# || Psi_Y_tilde   - K Psi_X_tilde     ||
# || Psi_Y_tilde.T - Psi_X_tilde.T K.T ||
# K = algorithms.rrr(Psi_X_tilde.T, Psi_Y_tilde.T).T
K = algorithms.SINDy(Psi_X_tilde.T, Psi_Y_tilde.T, Psi_Y_tilde.shape[0]).T

#%% To go from psi(x_tilde) -> x_tilde
B = algorithms.SINDy(Psi_X_tilde.T, X_tilde.T, X_tilde.shape[0])

#%% Dynamics
def dynamics(x, u):
    x_tilde = np.append(x, u, axis=1).T
    psi_x_tilde = psi(x_tilde)
    psi_x_tilde_prime = K @ psi_x_tilde
    x_tilde_prime = B.T @ psi_x_tilde_prime
    x_prime = x_tilde_prime[:4]
    return torch.from_numpy(x_prime)

#%% Create controller with chosen parameters
ctrl = mppi.MPPI(dynamics, cartpoleCost, d, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                        lambda_=lambda_,
                        u_min=torch.tensor(action_bounds[0], dtype=torch.double),
                        u_max=torch.tensor(action_bounds[1], dtype=torch.double))

#%% Run in environment
env = CartPoleEnv()

episodes = 100

rewards = []
for episode in range(episodes):
    # reset environment and variables
    episode_reward = 0
    current_state = env.reset()
    done = False

    while done == False:
        action = ctrl.command(current_state).cpu().numpy()
        current_state, reward, done, _ = env.step(action[0])
        episode_reward += reward

    rewards.append(episode_reward)

env.close()

print("Average reward:", np.mean(rewards))

#%%