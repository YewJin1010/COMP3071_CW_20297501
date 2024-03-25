import torch
import numpy as np
import gym
from gym.envs import box2d
import matplotlib.pyplot as plt
import os
import datetime

# Import agents
from agents.ppo import train_ppo
from agents.a2c import train_a2c
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

# Define parameter ranges
gravity_values = [-10.0, -5.0, 0.0]  # values for gravity
wind_power_values = [0.0, 15.0, 20.0]  # values for wind power
turbulence_power_values = [0.0, 1.0, 2.0]  # values for turbulence power

agents = ["PPO", "A2C", "DQN"]

for agent in agents:
    for gravity in gravity_values:
        for wind_power in wind_power_values:
            for turbulence_power in turbulence_power_values:
                # Create environments with current parameter values
                train_env = gym.make(
                    "LunarLander-v2",
                    continuous=False,
                    gravity=gravity,
                    enable_wind=wind_power != 0,
                    wind_power=wind_power,
                    turbulence_power=turbulence_power,
                )

                test_env = gym.make(
                    "LunarLander-v2",
                    continuous=False,
                    gravity=gravity,
                    enable_wind=wind_power != 0,
                    wind_power=wind_power,
                    turbulence_power=turbulence_power,
                )

                # Train current agent
                if agent == "PPO":
                    train_rewards, test_rewards, reward_threshold, episode = train_ppo(train_env, test_env)
                elif agent == "A2C":
                    train_rewards, test_rewards, reward_threshold, episode = train_a2c(train_env, test_env)
                elif agent == "DQN":
                    train_rewards, test_rewards, reward_threshold, episode = train_dqn(train_env, test_env)
                else:
                    print("Unknown agent:", agent)
                    continue

                # Save results
                now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                params = f"gravity_{gravity}_wind_{wind_power}_turbulence_{turbulence_power}"
                plot_results(train_rewards, test_rewards, reward_threshold, "LunarLander", agent, params, now)
                write_results(episode, train_rewards, test_rewards, reward_threshold, "LunarLander", agent, params, now)