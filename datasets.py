#%%
import gym
import d4rl_atari

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