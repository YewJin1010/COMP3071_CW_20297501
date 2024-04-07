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

    def push(self, state, action, log_prob_action, value, reward):
        self.buffer.append((state, action, log_prob_action, value, reward))

    def sample(self, batch_size):
        states, actions, log_prob_actions, values, rewards = zip(*random.sample(self.buffer, batch_size))
        return states, actions, log_prob_actions, values, rewards

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
    policy.train()

    states, actions, log_prob_actions, values, rewards = replay_buffer.sample(batch_size)
    states = torch.FloatTensor(states)
    actions = torch.tensor(actions, dtype=torch.int64)
    log_prob_actions = torch.stack(log_prob_actions)
    values = torch.stack(values)
    rewards = torch.FloatTensor(rewards)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, optimizer)

    return policy_loss, value_loss, rewards.sum().item()  # Return the episode reward

def calculate_returns(rewards, discount_factor, normalize=True):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + discount_factor * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Adding a small value to avoid division by zero
    return returns

def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Adding a small value to avoid division by zero
    return advantages

def update_policy(advantages, log_prob_actions, returns, values, optimizer):
    advantages = advantages.detach()
    returns = returns.detach()

    policy_loss = - (advantages * log_prob_actions).sum()
    value_loss = F.smooth_l1_loss(returns, values).sum()

    optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    value_loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()

def evaluate(env, policy):
    policy.eval()

    rewards = 0
    done = False
    state = env.reset()

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_pred, _ = policy(state_tensor)
        action_prob = F.softmax(action_pred, dim=-1)
        action = torch.argmax(action_prob, dim=-1).item()
        state, reward, done, _ = env.step(action)
        rewards += reward

    return rewards

def train_a2c_buffer(train_env, test_env):
    MAX_EPISODES = 2000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 100
    PRINT_EVERY = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    REPLAY_BUFFER_CAPACITY = 10000
    consecutive_episodes = 0 # Number of consecutive episodes that have reached the reward threshold
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weights)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    train_rewards = []
    test_rewards = []

    for episode in range(1, MAX_EPISODES + 1):
        state = train_env.reset()
        episode_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_pred, value_pred = policy(state_tensor)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample().item()
            log_prob_action = dist.log_prob(torch.tensor(action))
            next_state, reward, done, _ = train_env.step(action)
            replay_buffer.push(state, action, log_prob_action, value_pred, reward)
            state = next_state
            episode_reward += reward

            if done:
                break

        train_rewards.append(episode_reward)

        if len(replay_buffer.buffer) >= BATCH_SIZE:
            policy_loss, value_loss, _ = train(train_env, policy, optimizer, DISCOUNT_FACTOR, replay_buffer, BATCH_SIZE)
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
