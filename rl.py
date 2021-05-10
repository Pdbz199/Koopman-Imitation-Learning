import gym
from pytorch_mppi import mppi
from algorithms import gedmd

# create controller with chosen parameters
ctrl = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_, device=d, 
                         u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=d),
                         u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))

# assuming you have a gym-like env
obs = env.reset()
for i in range(100):
    action = ctrl.command(obs)
    obs, reward, done, _ = env.step(action.cpu().numpy())