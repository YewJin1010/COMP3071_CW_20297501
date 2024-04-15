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
    # create a directory to save the results
    save_path = f"results/{env}/{agent}/{experiment}/plots"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{agent}_{env}_{now}.png")

def write_results(episodes, train_rewards, test_rewards, reward_threshold, env, agent, experiment, parameter, now, duration, max_episodes):  
    """Write results to a file."""
    # create a directory to save the results
    save_path = f"results/{env}/{agent}/{experiment}/logs"
    os.makedirs(save_path, exist_ok=True)
    # write results to a file
    with open(f"{save_path}/{agent}_{env}_{now}.txt", "w") as f:
        f.write(f"Environment: {env}\n")
        f.write(f"Agent: {agent}\n")
        f.write(f"Experiment: {experiment}\n")
        f.write(f"Parameter: {parameter}\n")
        f.write(f"Time Taken: {duration} seconds\n")
        f.write(f"Episodes: {episodes} episodes\n")
        f.write(f"Max Episodes: {max_episodes} episodes\n")
        f.write(f"Reward threshold: {reward_threshold}\n")
        f.write("Episode\tTrain Reward\tTest Reward\n")
        for i in range(len(train_rewards)):
            f.write(f"{i+1}\t{train_rewards[i]}\t{test_rewards[i]}\n")

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

def create_env(env_name):

    if env_name == "CartPole-v0":
        experiment = "cartpole_experiment"
        parameter = "standard"
        return gym.make(env_name), gym.make(env_name), experiment, parameter

    elif env_name == "LunarLander-v2":
        experiment_selection = select_experiment()

        if experiment_selection == 1: 
            experiment = "standard_experiment"
            parameter = "standard"
          
        elif experiment_selection == 2:
            # Modify the gravity
            while True:
                maximum_gravity = float(input("Enter maximum gravity (-10 to -1): "))
                if -10 <= maximum_gravity <= -1:
                    break
                else:
                    print("Gravity must be within the range -10 to -1. Please enter a valid value.")
            experiment = "gravity_experiment"
            parameter = f"Gravity = {maximum_gravity}"

        elif experiment_selection == 3:
            # Modify the wind and turbulence
            while True:
                maximum_wind_power = float(input("Enter maximum wind power (0 to 20): "))
                maximum_turbulence_power = float(input("Enter maximum turbulence power (0 to 2): "))
                if 0 <= maximum_wind_power <= 20 or 0 <= maximum_turbulence_power <= 2:
                    break
                else:
                    print("Wind power must be within the range 0 to 20 and Turbulence power must be within the range 0 to 2. Please enter a valid value.")
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

if __name__ == "__main__":
    env_name = select_env()
    train_env, test_env, experiment, parameter = create_env(env_name)

    agent_selection = select_agent()
    agents = {
        1: ("PPO", train_ppo),
        2: ("A2C", train_a2c),
        3: ("DQN", train_dqn),
        4: ("A2C_Target", train_a2c_target),
        5: ("A2C_SU", train_a2c_su),
    }
    
    agent_name, agent_function = agents[agent_selection]
    
    max_episodes = int(input("Enter the maximum number of episodes to run (2000+ recommended): "))
    if max_episodes < 2000:
        print("Warning: Training for less than 2000 episodes may not result in optimal performance.")

    num_experiments = int(input("Enter the number of experiments to run: "))

    for i in range(num_experiments):
        print(f"Experiment {i+1}")
        train_rewards, test_rewards, reward_threshold, episode, duration = agent_function(train_env, test_env, max_episodes, parameter)
        now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        plot_results(train_rewards, test_rewards, reward_threshold, env_name, agent_name, experiment, parameter, now)
        write_results(episode, train_rewards, test_rewards, reward_threshold, env_name, agent_name, experiment, parameter, now, duration, max_episodes)