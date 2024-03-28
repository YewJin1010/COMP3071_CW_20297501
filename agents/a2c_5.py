import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

def evaluate(env, model, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state = torch.FloatTensor(state)
            action_probs, _ = model(state)
            action = torch.argmax(action_probs).item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def train_a2c(env_name, max_episodes=1000, lr=0.001, gamma=0.99, test_episodes=10, reward_threshold=200):
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCritic(input_dim, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_rewards = []
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []

        while True:
            state = torch.FloatTensor(state)
            action_probs, value = model(state)
            dist = torch.distributions.Categorical(F.softmax(action_probs, dim=-1))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state, reward, done, _ = env.step(action.item())
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            if done:
                break
            state = next_state

        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_loss = []
        value_loss = []
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        mean_reward = np.mean(episode_rewards[-100:])
        all_rewards.append(mean_reward)

        print(f"Episode: {episode+1}, Mean Reward (Last 100 episodes): {mean_reward}")

        # Evaluate the model
        test_reward = evaluate(env, model, episodes=test_episodes)
        print(f"Test Reward: {test_reward}")

        if test_reward >= reward_threshold:
            print(f"Environment solved with test reward {test_reward} in {episode+1} episodes!")
            break

    return all_rewards

env_name = "LunarLander-v2"
rewards = train_a2c(env_name)
