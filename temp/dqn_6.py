import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()

        # Define the layers
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        ])

        # Initialize weights (optional)
        self._initialize_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

class DQNAgent:
    def __init__(self, input_dim, hidden_dim ,output_dim, lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = MLP(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))
    
    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.output_dim)
        if isinstance(state, tuple):
            state = state[0]
        q_values = self.model(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Flatten the states
        states = torch.FloatTensor([s for state in states for s in state.flatten()])
        next_states = torch.FloatTensor([s for state in next_states for s in state.flatten()])
        
        # Compute Q-values for current and next states
        q_values = self.model(states)
        next_q_values = self.model(next_states).detach()

        targets = torch.tensor(rewards) + self.gamma * torch.max(next_q_values, dim=1).values
        q_values[range(batch_size), actions] = targets

        # Update the network
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q_values, self.model(states))
        loss.backward()
        self.optimizer.step()

def train(env, agent):
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
    
    return episode_reward

def evaluate(env, agent):
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        action = agent.act(state)
        state, reward, done, _, info = env.step(action)
        episode_reward += reward
    
    return episode_reward

def train_dqn(train_env, test_env):

    MAX_EPISODES = 2000
    BATCH_SIZE = 64
    N_TRIALS = 25
    REWARD_THRESHOLD = 200
    PRINT_EVERY = 10
    
    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    train_rewards = []
    test_rewards = []

    agent = DQNAgent(INPUT_DIM, HIDDEN_DIM ,OUTPUT_DIM)
    
    for episode in range(1, MAX_EPISODES):

        train_reward = train(train_env, agent)
        test_reward = evaluate(test_env, agent)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        agent.replay(BATCH_SIZE)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

        if mean_test_rewards >= REWARD_THRESHOLD:

            print(f'Reached reward threshold in {episode} episodes')
            return train_rewards, test_rewards, REWARD_THRESHOLD
     
    print("Did not reach reward threshold")
    return train_rewards, test_rewards, REWARD_THRESHOLD


