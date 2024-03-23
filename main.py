# Import libraries
import torch
import numpy as np
import gym
from gym.envs import box2d
import matplotlib.pyplot as plt

# Import agents
from agents.ppo import train_ppo
from agents.a2c import train_a2c
from agents.dqn import train_dqn


def plot_results(train_rewards, test_rewards, reward_treshold, env, agent):

    plt.figure(figsize=(12,8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(reward_treshold, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f"results/{env}/{agent}_{env}.png")

print("Select environment to train: ")
print("1. LunarLander")
print("2. CartPole")

env = int(input("Enter the number of the environment: "))
if env == 1: 
    env = "LunarLander"
    train_env = gym.make(
        "LunarLander-v2",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    test_env = gym.make(
        "LunarLander-v2",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )
    
    
elif env == 2:
    env = "CartPole"
    train_env = gym.make('CartPole-v0')
    test_env = gym.make('CartPole-v0')
else:
    print("Invalid input")
    exit()

print("Select agent to train: ")
print("1. PPO")
print("2. A2C")
print("3. DQN")
print("4. All")

agent = int(input("Enter the number of the agent: "))
if agent == 1:
    agent = "PPO"
    train_rewards, test_rewards, reward_treshold = train_ppo(train_env, test_env)
elif agent == 2:
    agent = "A2C"
    train_rewards, test_rewards, reward_treshold = train_a2c(train_env, test_env)
elif agent == 3:
    agent = "DQN"
    train_rewards, test_rewards, reward_treshold = train_dqn(train_env, test_env)
elif agent == 4:
    agent = ["PPO", "A2C", "DQN"]
    for a in agent:
        if a == "PPO":
            train_rewards, test_rewards, reward_treshold = train_ppo(train_env, test_env)
            plot_results(train_rewards, test_rewards, reward_treshold, env, a)
        elif a == "A2C":
            train_rewards, test_rewards, reward_treshold = train_a2c(train_env, test_env)
            plot_results(train_rewards, test_rewards, reward_treshold, env, a)
        elif a == "DQN":
            train_rewards, test_rewards, reward_treshold = train_dqn(train_env, test_env)
            plot_results(train_rewards, test_rewards, reward_treshold, env, a)
else:
    print("Invalid input")
    exit()

