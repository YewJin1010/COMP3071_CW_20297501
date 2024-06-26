import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym
import time
import random

EPS_START = 1.0             # Starting epsilon for epsilon-greedy strategy
EPS_END = 0.01              # Minimum epsilon
EPS_DECAY = 0.995           # Epsilon decay rate

# Multi-Layer Perceptron (MLP) network
class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) network."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    # Forward pass through the network
    def forward(self, x):
        x = self.net(x)
        return x

# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        """
        :param actor: nn.Module: Actor network
        :param critic: nn.Module: Critic network
        """ 
        super().__init__()
        
        # Actor and Critic networks
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        # Forward pass through the actor and critic networks
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred

# Initialize the weights of the network
def init_weights(m):
    """
    Initialize the weights of the neural network."""
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

# Train the agent using Proximal Policy Optimization (PPO)
def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip, eps):
    """
    param env: gym.Env: OpenAI Gym environment
    param policy: nn.Module: Actor-Critic network
    param optimizer: torch.optim: Optimizer
    param discount_factor: float: Discount factor for future rewards
    param ppo_steps: int: Number of steps to optimize the policy
    param ppo_clip: float: Clipping parameter for the policy loss
    param eps: float: Epsilon for epsilon-greedy strategy
    """
        
    policy.train()
        
    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        if isinstance(state, tuple):
            state, _ = state

        state = torch.FloatTensor(state).unsqueeze(0)

        states.append(state)        
        action_pred, value_pred = policy(state)
        action_prob = F.softmax(action_pred, dim = -1)

        p = random.random()
        if p <= eps:
            action = env.action_space.sample()
        else:
            action = torch.argmax(action_prob, dim=-1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()        
        log_prob_action = dist.log_prob(action)
        state, reward, done, _= env.step(action.item())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward
    
    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)
    
    # L2 Regularization
    l2_reg = 0.0
    l2_lambda = 0.1
    for param in policy.parameters():
        l2_reg += torch.norm(param)
    policy_loss += l2_lambda * l2_reg
    
    return policy_loss, value_loss, episode_reward

def calculate_returns(rewards, discount_factor, normalize = True):
    """
    Calculate the discounted returns."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values, normalize = True):
    """Calculate the advantages."""
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    """Update the policy using Proximal Policy Optimization (PPO)."""
    total_policy_loss = 0 
    total_value_loss = 0

    states = states.detach()
    actions = actions.detach()
    log_prob_actions = log_prob_actions.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    
    for _ in range(ppo_steps):
                
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
        value_loss = F.smooth_l1_loss(returns, value_pred).mean()
    
        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def evaluate(env, policy):
    """
    Evaluate the policy on the environment."""
    policy.eval()
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        if isinstance(state, tuple):
            state, _ = state
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
        
            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)
                
        action = torch.argmax(action_prob, dim = -1)
                
        state, reward, done, _= env.step(action.item())

        episode_reward += reward
    return episode_reward

def randomise_gravity(train_env, test_env, parameters):
    # Extract gravity value from parameters
    max_gravity = float(parameters.split('=')[1].strip())
    min_gravity = -10

    new_gravity = np.random.uniform(low=min_gravity, high=max_gravity)
    train_env.env.gravity = new_gravity
    test_env.env.gravity = new_gravity

def randomise_wind(train_env, test_env, parameters):
    # Extract wind and turbulence values from parameters
    parts = parameters.split(',')
    max_wind_power = float(parts[0].split('=')[1].strip())
    max_turburlence_power = float(parts[1].split('=')[1].strip())
 
    min_wind_power = 1
    min_turburlence_power = 0.1

    wind_power = np.random.uniform(low=min_wind_power, high=max_wind_power)
    turburlence_power = np.random.uniform(low=min_turburlence_power, high=max_turburlence_power)

    if wind_power > 0 or turburlence_power > 0:
        train_env.env.enable_wind = True
        test_env.env.enable_wind = True
    
    train_env.env.wind_power = wind_power
    test_env.env.wind_power = wind_power

    train_env.env.turbulence_power = turburlence_power
    test_env.env.turbulence_power = turburlence_power


def train_ppo(train_env, test_env, max_episodes, parameters):
    MAX_EPISODES = max_episodes # Maximum number of episodes to run
    DISCOUNT_FACTOR = 0.99 # Discount factor for future rewards
    N_TRIALS = 100 # Number of trials to average rewards over
    PRINT_EVERY = 100 # How often to print the progress
    PPO_STEPS = 5 # Number of steps to optimize the policy
    PPO_CLIP = 0.2 # Clipping parameter for the policy loss
    LEARNING_RATE = 5e-4 # Learning rate for the optimizer
    consecutive_episodes = 0 # Number of consecutive episodes that have reached the reward threshold
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander

    # Initialize the environment
    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    # Initialize the agent
    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weights)

    # Initialize the optimizer
    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    train_rewards = []
    test_rewards = []
    mean_train_rewards_list = []
    mean_test_rewards_list = []

    start_time = time.time()

    eps = EPS_START
    # Train the agent
    for episode in range(1, MAX_EPISODES+1):
        if 'Gravity' in parameters:
            randomise_gravity(train_env, test_env, parameters)
        if 'Wind' in parameters:
            randomise_wind(train_env, test_env, parameters)
        
        policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP, eps)
        
        test_reward = evaluate(test_env, policy)
        
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        eps = max(EPS_END, eps * EPS_DECAY)  # Decay epsilon

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        
        mean_train_rewards_list.append(mean_train_rewards)
        mean_test_rewards_list.append(mean_test_rewards)

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

        if test_env.unwrapped.spec.id == 'CartPole-v0':
            if mean_test_rewards >= REWARD_THRESHOLD_CARTPOLE:
                consecutive_episodes += 1
                if consecutive_episodes >= 100:

                    end_time = time.time()
                    duration = end_time - start_time

                    print(f'Reached reward threshold in {episode} episodes for CartPole')
                    return train_rewards, test_rewards, REWARD_THRESHOLD_CARTPOLE, episode, duration, mean_train_rewards_list, mean_test_rewards_list
            else:
                consecutive_episodes = 0
        elif test_env.unwrapped.spec.id == 'LunarLander-v2':
            if mean_test_rewards >= REWARD_THRESHOLD_LUNAR_LANDER:

                end_time = time.time()
                duration = end_time - start_time

                print(f'Reached reward threshold in {episode} episodes for Lunar Lander')
                return train_rewards, test_rewards, REWARD_THRESHOLD_LUNAR_LANDER, episode, duration, mean_train_rewards_list, mean_test_rewards_list

    end_time = time.time()
    duration = end_time - start_time

    print("Did not reach reward threshold")
    return train_rewards, test_rewards, None, episode, duration, mean_train_rewards_list, mean_test_rewards_list