# Import libraries
import torch
import numpy as np
import gym
from gym.envs import box2d
import matplotlib.pyplot as plt
import os
import datetime
import csv

# Import agents
from agents.ppo import train_ppo
from agents.a2c_2 import train_a2c
from agents.dqn import train_dqn
from agents.a2c_ppo import train_a2c_ppo
from agents.a2c_buffer import train_a2c_buffer

def plot_results(train_rewards, test_rewards, reward_threshold, env, agent, params, now):
    plt.figure(figsize=(12,8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    # create a directory to save the results
    save_path = f"results/{env}/{agent}"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"results/{env}/{agent}/{agent}_{env}_{params}_{now}.png")

def write_results(episodes, train_rewards, test_rewards, reward_threshold, env, agent, params, now):  
    # create a directory to save the results
    save_path = f"results/{env}"
    os.makedirs(save_path, exist_ok=True)
    # write results to a file
    with open(f"results/{env}/{agent}/{agent}_{env}_{params}_{now}.txt", "w") as f:
        f.write(f"Environment: {env}\n")
        f.write(f"Agent: {agent}\n")
        f.write(f"Parameters: {params}\n")
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Train rewards: {train_rewards}\n")
        f.write(f"Test rewards: {test_rewards}\n")
        f.write(f"Reward threshold: {reward_threshold}\n")

print("Select environment to train: ")
print("1. LunarLander")
print("2. CartPole")

env = int(input("Enter the number of the environment: "))
if env == 1: 
    """
    gravity = float(input("Enter the gravity value (-10 to -1): "))
    wind_power = float(input("Enter the wind power value (0 to 20): "))
    turbulence_power = float(input("Enter the turbulence power value (0 to 2): "))
   """
    
    gravity = -10
    wind_power = 0
    turbulence_power = 0
    
    env = "LunarLander"
    train_env = gym.make(
        "LunarLander-v2",
        continuous=False,
        gravity=gravity,
        enable_wind=wind_power != 0,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        #render_mode="human",
    )

    test_env = gym.make(
        "LunarLander-v2",
        continuous=False,
        gravity=gravity,
        enable_wind=wind_power != 0,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        #render_mode="human",
    )

    SEED = 1234

    train_env.seed(SEED)
    test_env.seed(SEED+1)
         
elif env == 2:
    env = "CartPole"
    train_env = gym.make(
        'CartPole-v0',
        #render_mode="human"                  
    )

    test_env = gym.make(
        'CartPole-v0',
        #render_mode="human"
    )

else:
    print("Invalid input")
    exit()

print("Select agent to train: ")
print("1. PPO")
print("2. A2C")
print("3. DQN")
print("4. A2CPPO") 
print("5. A2CBuffer")
print("6. All")

agent = int(input("Enter the number of the agent: "))
if agent == 1:
    agent = "PPO"
    train_rewards, test_rewards, reward_threshold, episode = train_ppo(train_env, test_env)
elif agent == 2:
    agent = "A2C"
    train_rewards, test_rewards, reward_threshold, episode = train_a2c(train_env, test_env)
elif agent == 3:
    agent = "DQN"
    train_rewards, test_rewards, reward_threshold, episode = train_dqn(train_env, test_env)
elif agent == 4:
    agent = "A2CPPO"
    train_rewards, test_rewards, reward_threshold, episode = train_a2c_ppo(train_env, test_env)
elif agent == 5:
    agent = "A2CBuffer"
    train_rewards, test_rewards, reward_threshold, episode = train_a2c_buffer(train_env, test_env)
elif agent == 6:
    agent = ["PPO", "A2C", "DQN", "A2CPPO", "A2CBuffer"]
    now = datetime.datetime.now()
    now = now.strftime("%d-%m-%Y_%H-%M-%S")
    for a in agent:
        
        if a == "PPO":
            train_rewards, test_rewards, reward_threshold, episode = train_ppo(train_env, test_env)
            plot_results(train_rewards, test_rewards, reward_threshold, env, a, now)
            write_results(episode, train_rewards, test_rewards, reward_threshold, env, a, now)

        elif a == "A2C":
            train_rewards, test_rewards, reward_threshold, episode = train_a2c(train_env, test_env)
            plot_results(train_rewards, test_rewards, reward_threshold, env, a, now)
            write_results(episode, train_rewards, test_rewards, reward_threshold, env, a, now)

        elif a == "DQN":
            train_rewards, test_rewards, reward_threshold, episode = train_dqn(train_env, test_env)
            plot_results(train_rewards, test_rewards, reward_threshold, env, a, now)
            write_results(episode, train_rewards, test_rewards, reward_threshold, env, a, now)

        elif a == "A2CPPO":
            train_rewards, test_rewards, reward_threshold, episode = train_a2c_ppo(train_env, test_env)
            plot_results(train_rewards, test_rewards, reward_threshold, env, a, now)
            write_results(episode, train_rewards, test_rewards, reward_threshold, env, a, now)

        elif a == "A2CBuffer":
            train_rewards, test_rewards, reward_threshold, episode = train_a2c_buffer(train_env, test_env)
            plot_results(train_rewards, test_rewards, reward_threshold, env, a, now)
            write_results(episode, train_rewards, test_rewards, reward_threshold, env, a, now)
else:
    print("Invalid input")
    exit()

if env == "LunarLander":
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    params = f"gravity_{gravity}_wind_{wind_power}_turbulence_{turbulence_power}"
    plot_results(train_rewards, test_rewards, reward_threshold, "LunarLander", agent, params, now)
    write_results(episode, train_rewards, test_rewards, reward_threshold, "LunarLander", agent, params, now)

else: 
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    params = ""
    plot_results(train_rewards, test_rewards, reward_threshold, "CartPole", agent, params, now)
    write_results(episode, train_rewards, test_rewards, reward_threshold, "CartPole", agent, params, now)
