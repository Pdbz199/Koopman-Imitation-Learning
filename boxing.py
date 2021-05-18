#%%
import gym
import d4rl_atari
import algorithms
import numpy as np

#%%
env = gym.make('boxing-expert-v0') # -v{0, 1, 2, 3, 4} for datasets with the other random seeds

#%% interaction with its environment through dopamine-style Atari wrapper
observation = env.reset() # observation.shape == (84, 84)
observation, reward, terminal, info = env.step(env.action_space.sample())

#%% dataset will be automatically downloaded into ~/.d4rl/datasets/[GAME]/[INDEX]/[EPOCH]
dataset = env.get_dataset()
# dataset['observations'] # observation data in (1000000, 1, 84, 84)
# dataset['actions'] # action data in (1000000,)
# dataset['rewards'] # reward data in (1000000,)
# dataset['terminals'] # terminal flags in (1000000,)

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

#%%
print("Run in environment")

episodes = 100
total_reward = 0
for episode in range(episodes):
    if (episode+1) % 25 == 0: print(episode+1)
    observation = env.reset()
    done = False

    while not done:
        action = dataset['actions'][episode]
        observation, reward, done, _ = env.step(action)
        total_reward += reward

env.close()
print("Average episode reward:", total_reward/episodes)

#%%