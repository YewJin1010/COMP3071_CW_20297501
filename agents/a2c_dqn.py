import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import numpy as np
import gym
import random
import time

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
  def __init__(self, actor, critic, target_critic):
    super().__init__()
    self.actor = actor
    self.critic = critic
    self.target_critic = target_critic

    # Update target network periodically through soft update
    self.tau = 0.01  # Target network update parameter

  def forward(self, state):
    action_pred = self.actor(state)
    value_pred = self.critic(state)
    return action_pred, value_pred

  def update_target(self):
    # Polyak averaging for soft update of target network
    for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.xavier_normal_(m.weight)
    m.bias.data.fill_(0)

def train(env, policy, optimizer, discount_factor):
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

    state, reward, done, _ = env.step(action.item())

    actions.append(action)
    log_prob_actions.append(log_prob_action)
    rewards.append(reward)
    episode_reward += reward

  states = torch.cat(states)
  actions = torch.cat(actions)
  log_prob_actions = torch.cat(log_prob_actions)

  returns = calculate_returns(rewards, discount_factor)
  # Detach gradients from advantages and returns for policy and value updates
  advantages = calculate_advantages(returns, policy.critic(states).squeeze(-1))

  # Decay epsilon after each episode
  EPSILON *= 0.999
  
  policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, policy.critic, optimizer, states, policy)

  # Update target network using soft update
  policy.update_target()
  
  return policy_loss, value_loss, episode_reward

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

def update_policy(advantages, log_prob_actions, returns, critic, optimizer, states, policy):
    """
    Updates the actor and critic networks using advantages, log probabilities, 
    returns, and the critic network for value estimation.
    """
    advantages = advantages.detach()
    returns = returns.detach()

    # Policy Loss (uses advantages and log probabilities)
    policy_loss = - (advantages * log_prob_actions).sum()

    # Value Loss (uses MSE between returns and critic's value estimates)
    target_values = policy.target_critic(states).squeeze(-1)
    value_loss = F.smooth_l1_loss(returns, target_values).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


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


def train_a2c_dqn(train_env, test_env):
    """
    Trains the A2C agent with DQN-like target network in multiple environments.
    """
    MAX_EPISODES = 2000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 100
    PRINT_EVERY = 10
    LEARNING_RATE = 0.001
    consecutive_episodes = 0 # Number of consecutive episodes that have reached the reward threshold
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    # Create actor, critic, and target critic networks
    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
    target_critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)  # Target critic with same architecture

    # Initialize weights
    policy = ActorCritic(actor, critic, target_critic)
    policy.apply(init_weights)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    train_rewards = []
    test_rewards = []

    start_time = time.time()

    for episode in range(1, MAX_EPISODES + 1):
        policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR)
        test_reward = evaluate(test_env, policy)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

        if test_env.unwrapped.spec.id == 'CartPole-v0':
            if mean_test_rewards >= REWARD_THRESHOLD_CARTPOLE:
                consecutive_episodes += 1
                if consecutive_episodes >= 100:

                    end_time = time.time()
                    duration = end_time - start_time

                    print(f'Reached reward threshold in {episode} episodes for CartPole')
                    return train_rewards, test_rewards, REWARD_THRESHOLD_CARTPOLE, episode, duration
            else:
                consecutive_episodes = 0
        elif test_env.unwrapped.spec.id == 'LunarLander-v2':
            if mean_test_rewards >= REWARD_THRESHOLD_LUNAR_LANDER:

                end_time = time.time()
                duration = end_time - start_time

                print(f'Reached reward threshold in {episode} episodes for Lunar Lander')
                return train_rewards, test_rewards, REWARD_THRESHOLD_LUNAR_LANDER, episode, duration

    end_time = time.time()
    duration = end_time - start_time

    print("Did not reach reward threshold")
    return train_rewards, test_rewards, None, episode, duration

"""
train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

train_rewards, test_rewards, reward_threshold, episode = train_a2c_dqn(train_env, test_env)
"""