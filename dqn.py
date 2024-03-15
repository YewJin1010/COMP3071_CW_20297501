import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class MLP(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.1):
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
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_dim)
        q_values = self.model(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

    def replay(self, BATCH_SIZE):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.FloatTensor(next_state))).item()
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(env, agent):
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
    
    return episode_reward

# Function to train the DQN agent
def train_dqn(env):
    MAX_EPISODES = 1000
    BATCH_SIZE = 64
    N_TRIALS = 25
    REWARD_THRESHOLD = 200
    PRINT_EVERY = 10
    
    INPUT_DIM = env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n

    train_rewards = []
    test_rewards = []

    agent = DQNAgent(INPUT_DIM, HIDDEN_DIM ,OUTPUT_DIM)
    scores = deque(maxlen=N_TRIALS)

    for episode in range(1, MAX_EPISODES):

        train_reward = train(env, agent)
        train_rewards.append(train_reward)
       
        agent.replay(BATCH_SIZE)
        mean_score = np.mean(train_rewards[-N_TRIALS:])
        if episode % PRINT_EVERY == 0:
            print(f"Episode {episode}/{MAX_EPISODES}, Mean Score: {mean_score}")
        if mean_score >= REWARD_THRESHOLD:
            print(f"Environment solved in {episode} episodes!")
            break
    env.close()

# Train the DQN agent for the LunarLander-v2 environment
env = gym.make('LunarLander-v2', render_mode='human')
train_env = gym.make('LunarLander-v2', render_mode='human')
test_env = gym.make('LunarLander-v2', render_mode='human')

SEED = 1234

train_env.seed(SEED)
test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_dqn(env, train_env, test_env)
