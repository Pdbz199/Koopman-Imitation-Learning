#%%
import math
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

#%%
states = dataset['observations']
actions = dataset['actions']

percent_training = 0.05
training_ind = int(np.around(states.shape[0]*percent_training))
states_training = states[:training_ind]
states_training = states_training.reshape(states_training.shape[0], int(math.pow(states_training.shape[2], 2)))

states_training = states_training.T
actions_training = actions[:training_ind].astype(np.float64).reshape(1,-1)

state_dim = states_training.shape[0]

#%% Median trick
num_pairs = 1000
pairwise_distances = []
for _ in range(num_pairs):
    i, j = np.random.choice(np.arange(states.shape[0]), 2)
    x_i = states[i]
    x_j = states[j]
    pairwise_distances.append(np.linalg.norm(x_i - x_j))
pairwise_distances = np.array(pairwise_distances)
gamma = np.quantile(pairwise_distances, 0.9)

#%% RBF sampler kernel
from sklearn.kernel_approximation import RBFSampler
rbf_feature = RBFSampler(gamma=gamma, random_state=1, n_components=1000)
state_features = rbf_feature.fit_transform(states_training)
#%%
def psi(state):
    return state_features.T @ state

#%% Psi matrices
def getPsiMatrix(psi, X):
    k = psi(X[:,0].reshape(-1,1)).shape[0]
    m = X.shape[1]
    matrix = np.empty((k,m))
    for col in range(m):
        matrix[:, col] = psi(X[:,col].reshape(-1,1))[:, 0]
    return matrix

Psi_X = getPsiMatrix(psi, states_training)

#%% Koopman (use generator?)
# || Y         - X B         ||
# || actions   - K Psi_X     ||
# || actions.T - Psi_X.T K.T ||
# K = algorithms.rrr(Psi_X.T, actions_training.T).T
K = algorithms.SINDy(Psi_X.T, actions_training.T, lamb=0.0005).T

#%%
def get_action(state):
    return int(np.around((K @ psi(state))[0,0]))

#%%
print("Run in environment")

episodes = 5
total_reward = 0
for episode in range(episodes):
    if (episode+1) % 25 == 0: print(episode+1)
    observation = env.reset()
    state = observation.reshape(-1,1)
    done = False

    while not done:
        env.render()

        action = get_action(state)
        if action >= 18: action = 17
        observation, reward, done, _ = env.step(action)
        state = observation.reshape(-1,1)
        total_reward += reward

env.close()
print("Average episode reward:", total_reward/episodes)

#%%