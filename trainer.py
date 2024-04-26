import torch
import numpy as np
import gym
from gym.envs import box2d
import matplotlib.pyplot as plt
import os
import datetime
import csv
import pandas as pd

# Import agents
from agents.ppo import train_ppo
from agents.a2c import train_a2c
from agents.dqn import train_dqn
from agents.a2c_su import train_a2c_su
from agents.a2c_mlp import train_a2c_mlp

def plot_results(train_rewards, test_rewards, reward_threshold, now, plot_save_path):
    """Plot training and testing rewards."""
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()

    # Save the plot
    plt.savefig(f"{plot_save_path}/plot_{now}.png")

def write_results(episodes, train_rewards, test_rewards, mean_train_rewards_list, mean_test_rewards_list, now, duration, log_save_path):  
    """Write results to a file."""

    episodes = list(range(1, episodes + 1))
  
    df = pd.DataFrame({
        "Episode": episodes,
        "Train Reward": train_rewards,
        "Test Reward": test_rewards,
        "Mean Train Reward": mean_train_rewards_list,
        "Mean Test Reward": mean_test_rewards_list,
        "Duration": duration
    })

    # Save the DataFrame to a CSV file
    df.to_csv(f"{log_save_path}/log_{now}.csv", index=False)

def select_experiment():
    print("Select experiment to run:")
    print("1. Experiment with standard Lunar Lander environment")
    print("2. Experiment with random gravity")
    print("3. Experiment with random wind and turbulence")

    while True:
        try:
            experiment_selection = int(input("Enter the number of the experiment: "))
            if experiment_selection in [1, 2, 3]:
                return experiment_selection
            else:
                print("Invalid input. Please enter either 1, 2 or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

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

def create_env(env_name, params):

    print("params: ", params)

    if env_name == "CartPole-v0":
        experiment = "cartpole_experiment"
        parameter = "standard"
        return gym.make(env_name), gym.make(env_name), experiment, parameter

    elif env_name == "LunarLander-v2":

        if "standard" in params: 
            experiment = "standard_lunarlander_experiment"
            parameter = "standard"
          
        elif "max_gravity" in params:
            # Modify the gravity
            maximum_gravity = params["max_gravity"]
            experiment = "gravity_experiment"
            parameter = f"Gravity = {maximum_gravity}"

        elif "max_wind_power" in params or "max_turbulence_power" in params:
            # Modify the wind and turbulence
            maximum_wind_power = params.get("max_wind_power", 0)
            maximum_turbulence_power = params.get("max_turbulence_power", 0)
            experiment = "wind_and_turbulence_experiment"
            parameter = f"Wind power = {maximum_wind_power}, Turbulence power = {maximum_turbulence_power}"

        train_env = gym.make(env_name)
        test_env = gym.make(env_name)

        seed = 1234
        train_env.seed(seed)
        test_env.seed(seed + 1)
        
        return train_env, test_env, experiment, parameter

def select_agent():
    """Select the agent to train."""
    print("Select agent to train:")
    print("1. PPO")
    print("2. A2C")
    print("3. DQN")
    print("4. A2C_SU")
    print("5. A2C_MLP")
    
    while True:
        try:
            agent_selection = int(input("Enter the number of the agent: "))
            if agent_selection in range(1, 7):
                return agent_selection
            else:
                print("Invalid input. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

"""
Experiment Trainer
"""
env_name = select_env()

# Define the parameter combinations for experiments
experiment_parameters = [
    {"standard": None},  # Standard Lunar Lander environment
    {"max_gravity": -1},  # High gravity 
    {"max_wind_power": 20, "max_turbulence_power": 2},  # High wind 
]

max_episodes = int(input("Enter the maximum number of episodes to run (2000+ recommended): "))
if max_episodes < 2000:
    print("Warning: Training for less than 2000 episodes may not result in optimal performance.")

# Number of experiments to run
num_experiments = int(input("Enter the number of experiments to run: "))
 
agents = {
        1: ("PPO", train_ppo),
        2: ("A2C", train_a2c),
        3: ("DQN", train_dqn),
        4: ("A2C_MLP", train_a2c_mlp),
        5: ("A2C_SU", train_a2c_su)
    }

mean_train_rewards_list = []
mean_test_rewards_list = []

for agent_id, (agent_name, agent_function) in agents.items():
    print(f"Running experiments for {agent_name}")
    if env_name == "LunarLander-v2":
        for params in experiment_parameters:
            # Modify the environment based on the current parameter combination
            train_env, test_env, experiment, parameter = create_env(env_name, params)
            
            for i in range(num_experiments):
                print(f"Running {experiment}: {i+1}/{num_experiments} for {agent_name} with parameters: {parameter}")
                train_rewards, test_rewards, reward_threshold, episode, duration, mean_train_rewards_list, mean_test_rewards_list = agent_function(train_env, test_env, max_episodes, parameter)
                now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                plot_save_path = f"results/{agent_name}/{experiment}/plots"
                os.makedirs(plot_save_path, exist_ok=True)

                log_save_path = f"results/{agent_name}/{experiment}/logs"
                os.makedirs(log_save_path, exist_ok=True)

                plot_results(train_rewards, test_rewards, reward_threshold, now, plot_save_path)
                write_results(episode, train_rewards, test_rewards, mean_train_rewards_list, mean_test_rewards_list, now, duration, log_save_path)

                print(f"Experiment {i+1}/{num_experiments} completed for {agent_name} with parameters: {parameter}")
            
    else: 
        train_env, test_env, experiment, parameter = create_env(env_name, {})
        for i in range(num_experiments):
            print(f"Running {experiment}: {i+1}/{num_experiments} for {agent_name}")
            train_rewards, test_rewards, reward_threshold, episode, duration, mean_train_rewards_list, mean_test_rewards_list = agent_function(train_env, test_env, max_episodes, parameter)
            now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            plot_save_path = f"results/{agent_name}/{experiment}/plots"
            os.makedirs(plot_save_path, exist_ok=True)

            log_save_path = f"results/{agent_name}/{experiment}/logs"
            os.makedirs(log_save_path, exist_ok=True)

            plot_results(train_rewards, test_rewards, reward_threshold, now, plot_save_path)
            write_results(episode, train_rewards, test_rewards, mean_train_rewards_list, mean_test_rewards_list, now, duration, log_save_path)

            print(f"Experiment {i+1}/{num_experiments} completed for {agent_name}")