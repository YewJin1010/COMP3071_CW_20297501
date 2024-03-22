import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQN():
    def __init__(self, input_dim, output_dim, lr=0.0005, gamma=0.99, batch_size=64, memory_size=10000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size

        # Q-Networks
        self.q_network = QNetwork(input_dim, output_dim)
        self.target_network = QNetwork(input_dim, output_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay Memory
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(dim=1)[0].unsqueeze(1)
        target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (~dones.unsqueeze(1))

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(env, agent, num_episodes=2000, max_steps=1000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, reward_threshold=200):
    train_rewards = []
    test_rewards = []

    epsilon = epsilon_start
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            if done:
                break

        train_rewards.append(episode_reward)
        test_reward = evaluate(env, agent, n_episodes=10)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-100:])
        mean_test_rewards = np.mean(test_rewards[-100:])
        if episode % 10 == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

        if mean_test_rewards >= reward_threshold:
            print(f'Reached reward threshold in {episode} episodes')
            return train_rewards, test_rewards, reward_threshold

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    print("Did not reach reward threshold")
    return train_rewards, test_rewards, reward_threshold

def evaluate(env, agent, n_episodes=10):
    total_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0)  # Greedy action selection
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = DQN(input_dim, output_dim)

    train_rewards, test_rewards, reward_threshold = train_dqn(env, agent)
