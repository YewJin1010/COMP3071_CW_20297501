import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import numpy as np
import gym
import time
import random

EPS_START = 1.0             # Starting epsilon for epsilon-greedy strategy
EPS_END = 0.01              # Minimum epsilon
EPS_DECAY = 0.995           # Epsilon decay rate

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array([np.squeeze(s) for s in states])
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array([np.squeeze(s) for s in next_states])
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def train(env, policy, optimizer, discount_factor, replay_buffer, batch_size):

    policy.train()
    
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample_batch(batch_size)
    log_prob_actions = []
    values = []
    rewards = []
    actions = []
    episode_reward = 0

    for state, action, reward, next_state, done in zip(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.tensor(action).unsqueeze(0)
        action_pred = policy.actor(state)
        value_pred = policy.critic(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        actions.append(action)
        log_prob_action = dist.log_prob(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward
        replay_buffer.add(state.numpy(), action.item(), reward, next_state.numpy(), done)

    states = torch.FloatTensor(batch_states)
    actions = torch.LongTensor(batch_actions)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(batch_rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, optimizer)

    # L2 Regularization
    l2_reg = 0.0
    l2_lambda = 0.1
    for param in policy.parameters():
        l2_reg += torch.norm(param)
    policy_loss += l2_lambda * l2_reg

    return policy_loss, value_loss, episode_reward

def calculate_returns(batch_rewards, discount_factor):
    returns = []
    R = 0
    for r in reversed(batch_rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    return torch.tensor(returns)

def calculate_advantages(returns, values):
    advantages = returns - values
    return advantages

def update_policy(advantages, log_prob_actions, returns, values, optimizer):
    policy_loss = - (advantages * log_prob_actions).sum()
    value_loss = F.smooth_l1_loss(returns, values).sum()
    optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    value_loss.backward(retain_graph=True)
    optimizer.step()
    return policy_loss.item(), value_loss.item()

def evaluate(env, policy):
    policy.eval()
    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1)
        state, reward, done, _ = env.step(action.item())
        episode_reward += reward
    return episode_reward

def train_a2c_with_replay_buffer(train_env, test_env):
    MAX_EPISODES = 2000
    WARMUP_EPISODES = 100
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 100
    PRINT_EVERY = 10
    LEARNING_RATE = 5e-4
    REWARD_THRESHOLD_CARTPOLE = 195
    REWARD_THRESHOLD_LUNAR_LANDER = 200

    BUFFER_SIZE = int(1e5)
    BATCH_SIZE = 64

    state_dim = train_env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = train_env.action_space.n

    actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE)
    actor_critic.apply(init_weights)
                           
    replay_buffer = ReplayBuffer(capacity=BUFFER_SIZE)

    train_rewards = []
    test_rewards = []

    start_time = time.time()

    eps = EPS_START
    for _ in range(WARMUP_EPISODES):
        state = train_env.reset()
        done = False
        while not done:
            if isinstance(state, tuple):
                state, _ = state
            action = train_env.action_space.sample()
            next_state, reward, done, _ = train_env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

    for episode in range(1, MAX_EPISODES+1):
        policy_loss, value_loss, train_reward = train(train_env, actor_critic, optimizer, DISCOUNT_FACTOR, replay_buffer, BATCH_SIZE)
        test_reward = evaluate(test_env, actor_critic)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        eps = max(EPS_END, eps * EPS_DECAY)  # Decay epsilon

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

train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

train_rewards, test_rewards, reward_threshold, episodes = train_a2c_with_replay_buffer(train_env, test_env)
