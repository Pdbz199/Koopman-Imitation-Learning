#%% Imports
import gym
import helpers
import observables
import torch
import numpy as np
import scipy as sp
import numba as nb
from pytorch_mppi import mppi
from algorithms import rrr, SINDy
from cartpole_reward import cartpoleCost

#%% State constants
N_SAMPLES = 10
TIMESTEPS = 5
state_dimension = 4
noise_sigma = torch.tensor(1.0, dtype=torch.double)
action_bounds = [0.0, 1.0]
lambda_ = 1.0

#%% Data
X = np.load('optimal-agent/cartpole-states.npy')
U = np.load('optimal-agent/cartpole-actions.npy').reshape(-1,1)
m = X.shape[0]

#? is X full rank?

#%% Data reformulation
psi = observables.monomials(2)
X_tilde = np.append(X, U, axis=1).T
Psi_X_tilde = psi(X_tilde)
k = Psi_X_tilde.shape[0]
nablaPsi = psi.diff(X_tilde)
nabla2Psi = psi.ddiff(X_tilde)
dPsi_X_tilde = helpers.dPsiMatrix(X_tilde, nablaPsi, nabla2Psi, k, m)

#%% Model
# L = gedmd(Psi_X_tilde.T, dPsi_X_tilde.T)
# K = np.zeros((k,k))
# K[:L.shape[0],:L.shape[1]] = sp.linalg.expm(L)
# dimensions reduce to not work with K @ psi_x
L = rrr(Psi_X_tilde.T, dPsi_X_tilde.T)
K = sp.linalg.expm(L)

#%% To go from psi(x) -> x
# Y - X B
# X_tilde - B.T Psi_X_tilde
# X_tilde.T - Psi_X_tilde.T B
#SINDy(Theta=Psi_X_T, dXdt=dPsi_X_T, lamb=0.05, n=d)
B = SINDy(Psi_X_tilde.T, X_tilde.T, X_tilde.shape[0])

#%% Dynamics
# needs to take in x and return x'
# currently takes in x and returns psi(x)'
def dynamics(x, u):
    x_tilde = np.append(x, u, axis=1).T
    psi_x_tilde = psi(x_tilde)
    psi_x_tilde_prime = K @ psi_x_tilde
    x_tilde_prime = B.T @ psi_x_tilde_prime
    return torch.from_numpy(x_tilde_prime)

#%% Create controller with chosen parameters
ctrl = mppi.MPPI(dynamics, cartpoleCost, state_dimension, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                        lambda_=lambda_,
                        u_min=torch.tensor(action_bounds[0], dtype=torch.double),
                        u_max=torch.tensor(action_bounds[1], dtype=torch.double))

#%% Run in environment
env = gym.make('CartPole-v0')
observation = env.reset()
episode_rewards = []
for i in range(100):
    action = ctrl.command(observation) #psi(observation.reshape(-1,1))
    observation, reward, done, _ = env.step(action.cpu().numpy())
    episode_rewards.append(reward)

print("Average episode reward:", np.mean(episode_rewards))