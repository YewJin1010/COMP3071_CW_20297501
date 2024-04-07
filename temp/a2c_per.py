import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import numpy as np
import gym

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
    self.tau = 0.01

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

def train(env, policy, optimizer, discount_factor, beta):
  """
  Trains the Actor-Critic with PER using a prioritized replay buffer.

  Args:
      env: The gym environment.
      policy: The Actor-Critic model.
      optimizer: The optimizer for training the model.
      discount_factor: Discount factor for future rewards.
      beta: PER beta parameter for priority importance sampling.
  """
  policy.train()

  states, actions, rewards, next_states, dones, priorities = [], [], [], [], [], []
  episode_reward = 0

  state = env.reset()
  while True:
    if isinstance(state, tuple):
      state, _ = state

    state = torch.FloatTensor(state).unsqueeze(0)
    states.append(state)

    action_pred, value_pred = policy(state)
    action_prob = F.softmax(action_pred, dim=-1)
    dist = distributions.Categorical(action_prob)

    action = dist.sample()
    log_prob_action = dist.log_prob(action)

    next_state, reward, done, _ = env.step(action.item())

    next_states.append(torch.FloatTensor(next_state).unsqueeze(0))
    actions.append(action)
    rewards.append(reward)
    dones.append(done)
    episode_reward += reward

    # Calculate TD-error for prioritized experience replay
    td_error = rewards[-1] + discount_factor * policy.target_critic(next_states[-1]).squeeze(-1) - policy.critic(states[-1]).squeeze(-1)
    priorities.append(np.abs(td_error.item())**beta)

    if done:
      break

  states = torch.cat(states)
  actions = torch.cat(actions)
  log_prob_actions = torch.cat([log_prob.unsqueeze(0) for log_prob in log_prob_action])
  rewards = torch.tensor(rewards, dtype=torch.float32)
  next_states = torch.cat(next_states)
  dones = torch.tensor(dones, dtype=torch.float32)
  priorities = torch.tensor(priorities, dtype=torch.float32)

 # Sample transitions with prioritized experience replay
  probabilities = priorities / priorities.sum()
  indices = torch.multinomial(probabilities, len(states), replacement=False)

  # Sample importance sampling weights for loss calculation
  importance_sampling_weights = (torch.numel(probabilities) * probabilities[indices]) ** (-beta)
  importance_sampling_weights /= importance_sampling_weights.sum()

  state_batch = states[indices]
  action_batch = actions[indices]
  reward_batch = rewards[indices]
  next_state_batch = next_states[indices]
  done_batch = dones[indices]

  # Actor Loss
  action_pred, _ = policy(state_batch)
  dist = distributions.Categorical(F.softmax(action_pred, dim=-1))
  log_prob_action_batch = dist.log_prob(action_batch)

  advantage = reward_batch + discount_factor * policy.target_critic(next_state_batch).squeeze(-1) - policy.critic(state_batch).squeeze(-1)
  actor_loss = -(log_prob_action_batch * advantage * importance_sampling_weights).mean()

  # Critic Loss
  target_value = reward_batch + discount_factor * (1 - done_batch) * policy.target_critic(next_state_batch).squeeze(-1)
  critic_loss = F.mse_loss(policy.critic(state_batch).squeeze(-1), target_value)

  # Update Networks
  optimizer.zero_grad()
  actor_loss.backward(retain_graph=True)
  critic_loss.backward()
  optimizer.step()

  # Update target network
  policy.update_target()

  # Reset experience buffer for next episode
  states, actions, rewards, next_states, dones, priorities = [], [], [], [], [], []

  return episode_reward

# Example Usage
env = gym.make("CartPole-v1")
actor = MLP(env.observation_space.shape[0], 64, env.action_space.n)
critic = MLP(env.observation_space.shape[0], 64, 1)
target_critic = MLP(env.observation_space.shape[0], 64, 1)

policy = ActorCritic(actor, critic, target_critic)
policy.apply(init_weights)

optimizer = optim.Adam(policy.parameters(), lr=1e-3)
discount_factor = 0.99
beta = 0.5  # PER beta parameter

for episode in range(1000):
  episode_reward = train(env, policy, optimizer, discount_factor, beta)
  print(f"Episode: {episode+1}, Reward: {episode_reward}")

env.close()
