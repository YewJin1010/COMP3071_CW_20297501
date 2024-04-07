import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import numpy as np
import gym
from collections import deque
import random

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
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

    def forward(self, x):
        x = self.net(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):

        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def train(env, policy, optimizer, discount_factor, replay_buffer, batch_size):
    EPSILON = 1.0
    
    policy.train()

    states = []
    actions = []
    log_prob_actions = []
    rewards = []
    values = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done: 
        if isinstance(state, tuple):
            state, _ = state

        state = torch.FloatTensor(state).unsqueeze(0)
        states.append(state)
        action_pred, value_pred = policy(state)

        action_prob = F.softmax(action_pred, dim=-1)

        p = random.random()
        if p <= EPSILON:
            action = env.action_space.sample()
        else:
            action = torch.argmax(action_prob, dim=-1)

        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.item())
        done = torch.FloatTensor([done]).unsqueeze(0)

        # Store experience in replay buffer
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)

        state = next_state

    # Convert experience to Tensors
    states = torch.cat(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.cat(next_states)
    dones = torch.cat(dones)

    # Estimate state values using the critic network
    values = policy.critic(states).squeeze(-1)

    # Calculate returns using Bellman equation with consideration for done states
    returns = rewards + (1 - dones) * discount_factor * policy.critic(next_states).squeeze(-1)

    # Detach gradients from returns to avoid propagating errors back to critic
    returns = returns.detach()

    # Calculate action probabilities and log probabilities
    action_pred, _ = policy(states)
    action_prob = F.softmax(action_pred, dim=-1)
    log_prob_actions = torch.log(action_prob + 1e-10)

    returns = calculate_returns(rewards, discount_factor)
    # Detach gradients from advantages and returns for policy and value updates
    advantages = calculate_advantages(returns, policy.critic(states).squeeze(-1))

    # Select log probabilities corresponding to the chosen actions
    selected_log_prob_actions = log_prob_actions.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Calculate advantages (difference between actual and expected returns)
    advantages = returns - values

    # Policy Loss (uses advantages and log probabilities)
    policy_loss = - (advantages * selected_log_prob_actions).mean()

    # Value Loss (uses MSE between returns and critic's value estimates)
    value_loss = F.smooth_l1_loss(returns, values).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()

def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def evaluate(env, policy):
    """
    Evaluates the policy on the environment by playing episodes.
    """
    policy.eval()

    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)

        state, reward, done, _ = env.step(action.item())
        episode_reward += reward

    return episode_reward


def train_a2c_buffer(train_env, test_env):
    """
    Trains the A2C agent with experience replay in multiple environments.
    """
    MAX_EPISODES = 2000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 100
    PRINT_EVERY = 10
    LEARNING_RATE = 0.001
    REPLAY_BUFFER_CAPACITY = 10000
    BATCH_SIZE = 32
    consecutive_episodes = 0 # Number of consecutive episodes that have reached the reward threshold
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    # Create actor and critic networks
    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    # Initialize weights
    policy = ActorCritic(actor, critic)
    policy.apply(init_weights)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    train_rewards = []
    test_rewards = []

    for episode in range(1, MAX_EPISODES + 1):
        # Interact with environment and fill replay buffer
        train_reward = evaluate(train_env, policy)  # Using evaluate instead of interact
        train_rewards.append(train_reward)

        # Train the policy using samples from replay buffer
        if len(replay_buffer) > BATCH_SIZE:
            policy_loss, value_loss = train(train_env, policy, optimizer, DISCOUNT_FACTOR, replay_buffer, BATCH_SIZE)

        # Evaluate the policy on the test environment
        test_reward = evaluate(test_env, policy)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

        if test_env.unwrapped.spec.id == 'CartPole-v0':
            if mean_test_rewards >= REWARD_THRESHOLD_CARTPOLE:
                consecutive_episodes += 1
                if consecutive_episodes >= 100:
                    print(f'Reached reward threshold in {episode} episodes for CartPole')
                    return train_rewards, test_rewards, REWARD_THRESHOLD_CARTPOLE, episode
            else:
                consecutive_episodes = 0
        elif test_env.unwrapped.spec.id == 'LunarLander-v2':
            if mean_test_rewards >= REWARD_THRESHOLD_LUNAR_LANDER:
                print(f'Reached reward threshold in {episode} episodes for Lunar Lander')
                return train_rewards, test_rewards, REWARD_THRESHOLD_LUNAR_LANDER, episode

    print("Did not reach reward threshold")
    return train_rewards, test_rewards, None, episode


train_env = gym.make('CartPole-v0')
test_env = gym.make('CartPole-v0')

train_a2c_buffer(train_env, test_env)