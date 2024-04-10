import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import gym
import time

# Hyperparameters (may need to be tuned for optimal performance)
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # Target network update factor
LEARNING_RATE = 0.001    # Learning rate
UPDATE_EVERY = 4        # Update target network every UPDATE_EVERY steps
EPSILON_DECAY = 0.995    # Epsilon decay rate for epsilon-greedy exploration
EPSILON_MIN = 0.01       # Minimum epsilon value

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer(object):
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent(object):
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.epsilon = 1.0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn from replay memory if enough samples are available
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences[0], experiences[1], experiences[2], experiences[3], experiences[4])

    def act(self, state, eps=0.):
        """Act using an epsilon-greedy strategy."""
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > self.epsilon:
            return np.argmax(q_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, states, actions, rewards, next_states, dones):
        """Update value parameters using a batch of experiences."""
        # Get max predicted Q values from target model (for next states)
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss (MSE)
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network (soft update)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # Decay epsilon for exploration
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)

    def soft_update(self, local_model, target_model, tau):
        """Soft update target model parameters towards local model parameters."""
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

def add_noise_to_observation(observation, noise_stddev):
    noise = torch.randn_like(observation) * noise_stddev
    return observation + noise

def train_dqn(train_env, test_env, max_episodes, noise_stddev):
    """Train DQN agent on the Lunar Lander environment."""
    MAX_EPISODES = max_episodes  # Maximum number of training episodes
    N_TRIALS = 100        # Number of episodes to consider for mean reward
    PRINT_EVERY = 10      # Print frequency
    consecutive_episodes = 0 # Number of consecutive episodes that have reached the reward threshold
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander

    # Initialize agent based on environment
    agent = Agent(train_env.observation_space.shape[0], train_env.action_space.n, 0)

    # Lists to store rewards
    train_rewards = []
    test_rewards = []

    start_time = time.time()

    for episode in range(1, MAX_EPISODES + 1):
        state = train_env.reset()
        episode_reward = 0
        for t in range(train_env._max_episode_steps):
            if noise_stddev > 0.0:
                state = add_noise_to_observation(state, noise_stddev)

            action = agent.act(state)
            next_state, reward, done, _ = train_env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                break

        train_rewards.append(episode_reward)
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

"""
train_env = gym.make("CartPole-v0")
test_env = gym.make("CartPole-v0")

train_dqn(train_env, test_env)

# cartpole state_size 4 action_size 2
# lunar lander state_size 8 action_size 4
"""