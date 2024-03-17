# Import libraries
import torch
import numpy as np
import gym
from gym.envs import box2d

# Import agents
from agents.ppo import train_ppo
from agents.a2c import train_a2c
from agents.dqn import train_dqn

# Import parameters
from parameters.params import initialise_lunar_lander

print("Select environment to train: ")
print("1. LunarLander")
print("2. CartPole")

env = int(input("Enter the number of the environment: "))
if env == 1: 
    train_env, test_env = initialise_lunar_lander()
    
elif env == 2:
    train_env = gym.make('CartPole-v0')
    test_env = gym.make('CartPole-v0')
else:
    print("Invalid input")

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
