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
from agents.a2c import train_a2c
from agents.dqn import train_dqn
from agents.a2c_ppo import train_a2c_ppo
from agents.a2c_buffer import train_a2c_buffer

def plot_results(train_rewards, test_rewards, reward_threshold, env, agent, now):
    """Plot training and testing rewards."""
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    # create a directory to save the results
    save_path = f"results/{env}/{agent}/plots"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"results/{env}/{agent}/plots/{agent}_{env}_{now}.png")

def write_results(episodes, train_rewards, test_rewards, reward_threshold, env, agent, now):  
    """Write results to a file."""
    # create a directory to save the results
    save_path = f"results/{env}/{agent}/logs"
    os.makedirs(save_path, exist_ok=True)
    # write results to a file
    with open(f"results/{env}/{agent}/logs/{agent}_{env}_{now}.txt", "w") as f:
        f.write(f"Environment: {env}\n")
        f.write(f"Agent: {agent}\n")
        f.write(f"Solved in {episodes} Episodes\n")
        f.write(f"Reward threshold: {reward_threshold}\n")
        f.write("Episode\tTrain Reward\tTest Reward\n")
        for i in range(len(train_rewards)):
            f.write(f"{i+1}\t{train_rewards[i]}\t{test_rewards[i]}\n")

def select_env():
    print("Select environment to train:")
    print("1. LunarLander")
    print("2. CartPole")
    
    while True:
        try:
            env_selection = int(input("Enter the number of the environment: "))
            if env_selection == 1:
                return "LunarLander-v2"
            elif env_selection == 2:
                return "CartPole-v0"
            else:
                print("Invalid input. Please enter either 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def create_env(env_name):
    train_env = gym.make(env_name)
    test_env = gym.make(env_name)
    
    seed = 1234
    train_env.seed(seed)
    test_env.seed(seed + 1)
    
    return train_env, test_env

def select_agent():
    """Select the agent to train."""
    print("Select agent to train:")
    print("1. A2C")
    print("2. PPO")
    print("3. DQN")
    print("4. A2CPPO") 
    print("5. A2CBuffer")
    
    while True:
        try:
            agent_selection = int(input("Enter the number of the agent: "))
            if agent_selection in range(1, 7):
                return agent_selection
            else:
                print("Invalid input. Please enter a number between 1 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    env_name = select_env()
    train_env, test_env = create_env(env_name)
    
    agent_selection = select_agent()
    agents = {
        1: ("A2C", train_a2c),
        2: ("PPO", train_ppo),
        3: ("DQN", train_dqn),
        4: ("A2CPPO", train_a2c_ppo),
        5: ("A2CBuffer", train_a2c_buffer),
    }
    
    agent_name, agent_function = agents[agent_selection]

    train_rewards, test_rewards, reward_threshold, episode = agent_function(train_env, test_env)
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    plot_results(train_rewards, test_rewards, reward_threshold, env_name, agent_name, now)
    write_results(episode, train_rewards, test_rewards, reward_threshold, env_name, agent_name, now)
