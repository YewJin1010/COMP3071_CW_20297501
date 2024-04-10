import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

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
        return self.actor(state), self.critic(state)

def train_a2c(env_name, max_episodes=1000, gamma=0.99, lr=0.001):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0] 
    actor_critic = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

    for episode in range(max_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []

        while True:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_mean, value = actor_critic(state)
            dist = torch.distributions.Normal(action_mean, torch.exp(torch.tensor(0.1)))  # Add exploration noise
            action = torch.tanh(dist.sample())
            action = torch.clamp(action, -1.0, 1.0)

            next_state, reward, done, _ = env.step(action.numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

            if done:
                break

        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        actor_loss = 0
        critic_loss = 0
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()

            actor_loss += -log_prob * advantage
            critic_loss += F.smooth_l1_loss(value, torch.tensor([R]))

        optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        optimizer.step()

        if episode % 10 == 0:
            test_reward = evaluate(actor_critic, env_name)
            print(f"Episode {episode}, Test Reward: {test_reward}")

    env.close()

def evaluate(actor_critic, env_name, max_episodes=10):
    env = gym.make(env_name)
    total_reward = 0
    for _ in range(max_episodes):
        state = env.reset()
        done = False
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            action, _ = actor_critic(state)
            state, reward, done, _ = env.step(action.detach().numpy())
            total_reward += reward
    env.close()
    return total_reward / max_episodes



train_a2c("LunarLander-v2")