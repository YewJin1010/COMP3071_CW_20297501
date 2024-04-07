import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import gym

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

def update_policy(policy, buffer, optimizer, gamma, ppo_steps, ppo_clip):
    batch, indices, weights = buffer.sample(len(buffer.buffer))
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    policy.train()

    for _ in range(ppo_steps):
        optimizer.zero_grad()
        action_pred, value_pred = policy(states)

        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)

        log_prob_actions = dist.log_prob(actions)
        entropy = dist.entropy()

        next_value_pred = policy.critic(next_states)

        target_values = rewards + gamma * (1 - dones) * next_value_pred.detach()
        advantages = target_values - value_pred.squeeze(-1)  # Ensure value_pred has same shape as target_values

        policy_loss = -torch.min(log_prob_actions * advantages.detach(), log_prob_actions * advantages.detach().clamp(1.0 - ppo_clip, 1.0 + ppo_clip)).mean()
        value_loss = F.smooth_l1_loss(value_pred.squeeze(-1), target_values)  # Squeeze value_pred to match target_values shape

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

        loss.backward()
        optimizer.step()

    priorities = (advantages.abs() + 1e-6).detach().numpy()
    buffer.update_priorities(indices, priorities)


def train_a2c_with_per(env, policy, optimizer, buffer, gamma, ppo_steps, ppo_clip, max_episodes):
    episode_rewards = []

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim=-1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample().item()
            next_state, reward, done, _ = env.step(action)
            buffer.add(state.squeeze(0).numpy(), action, reward, next_state, done)
            total_reward += reward
            state = next_state

        update_policy(policy, buffer, optimizer, gamma, ppo_steps, ppo_clip)
        episode_rewards.append(total_reward)

    return episode_rewards

# Define hyperparameters
env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
hidden_dim = 128
output_dim = env.action_space.n
lr = 0.001
gamma = 0.99
ppo_steps = 5
ppo_clip = 0.2
max_episodes = 1000
buffer_capacity = 10000

# Initialize actor-critic model and optimizer
actor = MLP(input_dim, hidden_dim, output_dim)
critic = MLP(input_dim, hidden_dim, 1)
policy = ActorCritic(actor, critic)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# Initialize Prioritized Experience Replay buffer
buffer = PrioritizedReplayBuffer(buffer_capacity)

# Train A2C with Prioritized Experience Replay
episode_rewards = train_a2c_with_per(env, policy, optimizer, buffer, gamma, ppo_steps, ppo_clip, max_episodes)

# Plot the rewards
import matplotlib.pyplot as plt
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('A2C with Prioritized Experience Replay')
plt.show()
