# Import libraries
import torch
import numpy as np
import gym
from gym.envs import box2d

# Import agents
from agents.ppo import train_ppo
from agents.a2c import train_a2c
from agents.dqn import train_dqn

# Import environment
from envs.lunar_lander import LunarLander

train_env = LunarLander()
test_env = LunarLander()


# Enter gravity -12 - 0
LunarLander.gravity = -12

#train_env = gym.make('LunarLander-v2')
#test_env = gym.make('LunarLander-v2')

SEED = 1234

train_env.seed(SEED)
test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)


print("Select agent to train: ")
print("1. PPO")
print("2. A2C")
print("3. DQN")

agent = int(input("Enter the number of the agent: "))
if agent == 1:
    train_ppo(train_env, test_env)
elif agent == 2:
    train_a2c(train_env, test_env)
elif agent == 3:
    train_dqn(train_env)
else:
    print("Invalid input")
