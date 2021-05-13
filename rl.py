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
from pytorch_mppi import mppi
from algorithms import rrr, SINDy
from cartpole_reward import cartpoleCost

#%% State constants
N_SAMPLES = 300
TIMESTEPS = 8
noise_sigma = torch.tensor(1.0, dtype=torch.double)
action_bounds = [0.0, 1.0]
lambda_ = 1.0

#%% Data
X = np.load('optimal-agent/cartpole-states.npy').T
U = np.load('optimal-agent/cartpole-actions.npy').reshape(1,-1)

if np.linalg.matrix_rank(X) == X.shape[0]:
    print("X is full-rank")
if np.linalg.matrix_rank(U) == U.shape[0]:
    print("U is full-rank")

#%% Data reformulation
# psi = observables.monomials(2)
sigma = np.sqrt(0.3)
kernel = kernels.gaussianKernelGeneralized(sigma)

def psi(x):
    result = []
    for x_tilde in X_tilde.T:
        result.append(kernel(x, x_tilde))
    return np.array(result)

X_tilde = np.append(X, U, axis=0)[:, :1000]
d = X_tilde.shape[0]
m = X_tilde.shape[1]
Y_tilde = np.append(np.roll(X_tilde,-1)[:, :-1], np.zeros((d,1)), axis=1)

epsilon = 0
G_0 = kernels.gramian(X_tilde, kernel)
G_1 = kernels.gramian2(X_tilde, Y_tilde, kernel).T

# Psi_X_tilde = psi(X_tilde)
# k = Psi_X_tilde.shape[0]
# nablaPsi = psi.diff(X_tilde)
# nabla2Psi = psi.ddiff(X_tilde)
# dPsi_X_tilde = helpers.dPsiMatrix(X_tilde, nablaPsi, nabla2Psi, k, m)

#%% Model
# L = gedmd(Psi_X_tilde.T, dPsi_X_tilde.T)
# K = np.zeros((k,k))
# K[:L.shape[0],:L.shape[1]] = sp.linalg.expm(L)
# dimensions reduce to not work with K @ psi_x
# L = rrr(Psi_X_tilde.T, dPsi_X_tilde.T)
L = np.linalg.pinv(G_0 + epsilon * np.identity(m), rcond=1e-15) @ G_1
K = sp.linalg.expm(L)

#%% To go from psi(x_tilde) -> x_tilde
# Y - X B
# X_tilde - B.T Psi_X_tilde
# X_tilde.T - Psi_X_tilde.T B
#SINDy(Theta=Psi_X_T, dXdt=dPsi_X_T, lamb=0.05, n=d)
# B = SINDy(Psi_X_tilde.T, X_tilde.T, X_tilde.shape[0])
# Y - X B
# Y.T - B.T X.T
# X_tilde.T - B.T G_0.T
B = SINDy(G_0.T, X_tilde.T, d)
# B.T @ G_0[:, 0].reshape(-1,1) = X_tilde[:,0].reshape(-1,1)

#%% Dynamics
#? how do I change this function for kernels?
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
env = gym.make('CartPole-v0')

episodes = 100

rewards = []
for episode in range(episodes):
    # reset environment and variables
    episode_reward = 0
    current_state = env.reset()
    done = False

    while done == False:
        action = ctrl.command(current_state).cpu().numpy()
        current_state, reward, done, _ = env.step(int(np.around(action[0])))
        episode_reward += reward

    rewards.append(episode_reward)

env.close()

print("Average reward:", np.mean(rewards))