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
from agents.a2c_target import train_a2c_target
from agents.a2c_su import train_a2c_su

def plot_results(train_rewards, test_rewards, reward_threshold, env, agent, experiment, parameter, now):
    """Plot training and testing rewards."""
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    if parameter == "None":
        parameter = "standard"
    # create a directory to save the results
    save_path = f"results/{env}/{agent}/{experiment}/{parameter}/plots"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{agent}_{env}_{now}.png")

def write_results(episodes, train_rewards, test_rewards, reward_threshold, env, agent, experiment, parameter, now, duration):  
    """Write results to a file."""
    if parameter == "None":
        parameter = "standard"
    # create a directory to save the results
    save_path = f"results/{env}/{agent}/{experiment}/{parameter}/logs"
    os.makedirs(save_path, exist_ok=True)
    # write results to a file
    with open(f"{save_path}/{agent}_{env}_{now}.txt", "w") as f:
        f.write(f"Environment: {env}\n")
        f.write(f"Agent: {agent}\n")
        f.write(f"Experiment: {experiment}\n")
        if parameter:
            f.write(f"Parameter: {parameter}\n")
        f.write(f"Time Taken: {duration} seconds\n")
        f.write(f"Episodes: {episodes} episodes\n")
        f.write(f"Reward threshold: {reward_threshold}\n")
        f.write("Episode\tTrain Reward\tTest Reward\n")
        for i in range(len(train_rewards)):
            f.write(f"{i+1}\t{train_rewards[i]}\t{test_rewards[i]}\n")

def select_experiment():
    print("Select experiment to run:")
    print("1. Experiment with standard Lunar Lander environment")
    print("2. Experiment with gravity modification")
    print("3. Experiment with wind and turbulence modification")

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

def create_env(env_name, params=None):
    enable_wind = False
    experiment = None
    parameter = None

    if env_name == "CartPole-v0":
        experiment = "standard CartPole experiment"
        parameter = "standard"
        return gym.make(env_name), gym.make(env_name), experiment, parameter
    
    elif env_name == "LunarLander-v2":
      
        # Automatically select experiment based on provided parameters
        if "gravity" in params:
            experiment_selection = 2
        elif "wind_power" in params or "turbulence_power" in params:
            experiment_selection = 3
        else:
            experiment_selection = 1

        if experiment_selection == 1:
            experiment = "Random Gravity LunarLander experiment"
            parameter = "random gravity"
            train_env = gym.make(env_name)
            test_env = gym.make(env_name)

        elif experiment_selection == 2:
            experiment = "gravity experiment"
            gravity = params.get("gravity")  
            train_env = gym.make(env_name, gravity=gravity)
            test_env = gym.make(env_name, gravity=gravity)
            parameter = f"Gravity = {gravity}"

        elif experiment_selection == 3:
            experiment = "wind_and_turbulence experiment"
            wind_power = params.get("wind_power")
            turbulence_power = params.get("turbulence_power")
            if wind_power > 0 or turbulence_power > 0:
                enable_wind = True
            train_env = gym.make(env_name, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
            test_env = gym.make(env_name, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
            parameter = f"Wind power = {wind_power}, Turbulence power = {turbulence_power}"
        
        if params is None:
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
    print("4. A2C_Target") 
    print("5. A2C_SU")
    
    while True:
        try:
            agent_selection = int(input("Enter the number of the agent: "))
            if agent_selection in range(1, 6):
                return agent_selection
            else:
                print("Invalid input. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

env_name = select_env()

# Define the parameter combinations for experiments
experiment_parameters = [
    {"standard": None},  # Standard Lunar Lander environment
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
        4: ("A2C_Target", train_a2c_target),
        5: ("A2C_SU", train_a2c_su),
    }

for agent_id, (agent_name, agent_function) in agents.items():
    print(f"Running experiments for {agent_name}")
    for params in experiment_parameters:
        # Modify the environment based on the current parameter combination
        train_env, test_env, experiment, parameter = create_env(env_name, params)
        
        for i in range(num_experiments):
            print(f"Running {experiment}: {i+1}/{num_experiments} for {agent_name} with parameters: {parameter}")
            train_rewards, test_rewards, reward_threshold, episode, duration = agent_function(train_env, test_env, max_episodes)
            now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            plot_results(train_rewards, test_rewards, reward_threshold, env_name, agent_name, experiment, parameter, now)
            write_results(episode, train_rewards, test_rewards, reward_threshold, env_name, agent_name, experiment, parameter, now, duration)
            print(f"Experiment {i+1}/{num_experiments} completed for {agent_name} with parameters: {parameter}")