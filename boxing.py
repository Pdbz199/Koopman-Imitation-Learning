#%%
import math
import gym
import d4rl_atari
import helpers
import algorithms
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

#%%
env = gym.make('boxing-expert-v0') # -v{0, 1, 2, 3, 4} for datasets with the other random seeds

#%% interaction with its environment through dopamine-style Atari wrapper
observation = env.reset() # observation.shape == (84, 84)
observation, reward, terminal, info = env.step(env.action_space.sample())

#%% dataset will be automatically downloaded into ~/.d4rl/datasets/[GAME]/[INDEX]/[EPOCH]
dataset = env.get_dataset()

#%%
states = dataset['observations']
actions = dataset['actions']

screen_dims = states.shape[-2:]

percent_training = 0.05
training_ind = int(np.around(states.shape[0]*percent_training))

states_training = states[:training_ind]
states_training = states_training.reshape(states_training.shape[0], int(math.pow(states_training.shape[2], 2)))
states_training = states_training.T
actions_training = actions[:training_ind].astype(np.float64).reshape(1,-1)

states_validation = states[training_ind:]
states_validation = states_validation.reshape(states_validation.shape[0], int(math.pow(states_validation.shape[2], 2)))
states_validation = states_validation.T
actions_validation = actions[training_ind:].astype(np.float64).reshape(1,-1)

state_dim = states_training.shape[0]

#%% Median trick
print("Median trick:")
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
print("RBF Sampler:")
from sklearn.kernel_approximation import RBFSampler
component_dim = int(np.around(8000))
rbf_feature = RBFSampler(gamma=gamma, random_state=1, n_components=component_dim)
state_features = rbf_feature.fit_transform(states_training)
# state_features = np.load('boxing_arrays/rbf_state_features.npy')
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
# Psi_X = np.load('boxing_arrays/psi_x.npy')

#%% Koopman (use generator?)
# || Y         - X B         ||
# || actions   - K Psi_X     ||
# || actions.T - Psi_X.T K.T ||
print("Learn Koopman operator:")
K = algorithms.rrr(Psi_X.T, actions_training.T).T
# K = np.load('boxing_arrays/koopman_operator.npy')

#%% find B s.t. B.T psi(x) -> x
# || Y        - X B         ||
# || states   - psi_x B     ||
# || states.T - B.T psi_x.T ||
B = algorithms.rrr(Psi_X.T, states_training.astype(np.float64).T)
# B = np.load('boxing_arrays/psi_x_to_x_operator.npy')

#%%
def get_action(state):
    return int(np.around(( K @ psi(state) )[0]))

#%% Error testing
print("Training error:")
norms = []
for state_index in range(states_training.shape[1]):
    true_action = actions_training[:,state_index]
    predicted_action = np.array([get_action(states_training[:,state_index])])
    l2_norm = helpers.l2_norm(true_action, predicted_action)
    norms.append(l2_norm)
norms = np.array(norms)
print("Mean difference:", np.mean(norms))
accuracy = (norms.shape[0]-np.count_nonzero(norms)) / norms.shape[0]
print("Percent accuracy:", accuracy * 100)

print("\nValidation error:")
norms = []
for state_index in range(int(np.around(states_validation.shape[1] / 10))):
    true_action = actions_validation[:,state_index]
    predicted_action = np.array([get_action(states_validation[:,state_index])])
    l2_norm = helpers.l2_norm(true_action, predicted_action)
    norms.append(l2_norm)
norms = np.array(norms)
print("Mean difference:", np.mean(norms))
accuracy = (norms.shape[0]-np.count_nonzero(norms)) / norms.shape[0]
print("Percent accuracy:", accuracy * 100)

# Training error:
# Mean difference: 22.2551
# Percent accuracy: 6.554
# Validation error:
# Mean difference: 32.497368421052634
# Percent accuracy: 5.707368421052632

#%%
_, __, vh = helpers.SVD(K)
important_states = B.T @ vh.T
for i in range(10):
    plt.imshow(important_states[:,i].reshape(screen_dims))
    plt.show()

#%%
print("Run in environment:")

episodes = 1
total_reward = 0
for episode in range(episodes):
    if (episode+1) % 25 == 0: print(episode+1)
    observation = env.reset()
    state = observation.reshape(-1,1)
    done = False

    while not done:
        # env.render()

        action = get_action(state)
        if action < 0: action = 0
        elif action > 17: action = 17
        observation, reward, done, _ = env.step(action)
        state = observation.reshape(-1,1)
        total_reward += reward

env.close()
print("Average episode reward:", total_reward/episodes)

#%%