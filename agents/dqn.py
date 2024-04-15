import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import gym
import time

# Hyperparameters 
BUFFER_SIZE = int(1e5)      # Replay buffer size
BATCH_SIZE = 64             # Minibatch size
GAMMA = 0.99                # Discount factor
TAU = 1e-3                  # Target network update factor
LR = 5e-4                   # Learning rate
UPDATE_EVERY = 4            # Update target network every UPDATE_EVERY steps
EPS_START = 1.0             # Starting epsilon for epsilon-greedy strategy
EPS_END = 0.01              # Minimum epsilon
EPS_DECAY = 0.995           # Epsilon decay rate

class QNetwork(nn.Module):
    """Deep Q-Network (DQN) for estimating action values."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Forward pass of the neural network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory and learn from it."""
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Select an action using epsilon-greedy policy."""
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Get max predicted Q values for next states from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def evaluate(env, agent):
    """Evaluate agent's performance on a single episode."""
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

def randomise_gravity(train_env, test_env, parameters):
    # Extract gravity value from parameters
    max_gravity = float(parameters.split('=')[1].strip())
    min_gravity = -10

    new_gravity = np.random.uniform(low=min_gravity, high=max_gravity)
    train_env.env.gravity = new_gravity
    test_env.env.gravity = new_gravity

def randomise_wind(train_env, test_env, parameters):
    # Extract wind and turbulence values from parameters
    parts = parameters.split(',')
    max_wind_power = float(parts[0].split('=')[1].strip())
    max_turburlence_power = float(parts[1].split('=')[1].strip())
 
    min_wind_power = 1
    min_turburlence_power = 0.1

    wind_power = np.random.uniform(low=min_wind_power, high=max_wind_power)
    turburlence_power = np.random.uniform(low=min_turburlence_power, high=max_turburlence_power)

    if wind_power > 0 or turburlence_power > 0:
        train_env.env.enable_wind = True
        test_env.env.enable_wind = True
    
    train_env.env.wind_power = wind_power
    test_env.env.wind_power = wind_power

    train_env.env.turbulence_power = turburlence_power
    test_env.env.turbulence_power = turburlence_power


def train_dqn(train_env, test_env, max_episodes, parameters):
    """Deep Q-Learning algorithm."""
    N_TRIALS = 100              # Number of consecutive trials needed to solve environment
    PRINT_EVERY = 100           # How often to print progress
    REWARD_THRESHOLD_CARTPOLE = 195  # Mean reward needed to solve CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200  # Mean reward needed to solve LunarLander

    state_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n
    agent = Agent(state_size, action_size, seed=0)
    train_rewards = []
    test_rewards = []
    consecutive_episodes = 0
    start_time = time.time()

    eps = EPS_START  # Initialize epsilon
    for episode in range(1, max_episodes + 1):
        if 'Gravity' in parameters:
            randomise_gravity(train_env, test_env, parameters)
        if 'Wind' in parameters:
            randomise_wind(train_env, test_env, parameters)

        state = train_env.reset()
        episode_reward = 0
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, _ = train_env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
            
        train_rewards.append(episode_reward)
        eps = max(EPS_END, eps * EPS_DECAY)  # Decay epsilon
        test_reward = evaluate(test_env, agent)
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

