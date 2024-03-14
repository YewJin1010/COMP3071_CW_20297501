from ppo import train_ppo
from a2c import train_a2c

import torch
import numpy as np
import gym
from gym.envs import box2d

train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

SEED = 1234

train_env.seed(SEED)
test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_ppo(train_env, test_env)
train_a2c(train_env, test_env)