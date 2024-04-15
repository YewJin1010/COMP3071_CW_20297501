import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

# Environment
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate_actor = 0.001
learning_rate_critic = 0.005
gamma = 0.99
num_episodes = 2000
max_steps_per_episode = 2000
N_TRIALS = 100
PRINT_EVERY = 10
REWARD_THRESHOLD_LUNAR_LANDER = 200  # Set your desired reward threshold here

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_a2c(train_env, test_env, max_episodes, parameters):
    actor = Actor(state_size, action_size)
    critic = Critic(state_size)

    # Optimizers
    optimizer_actor = optim.Adam(actor.parameters(), lr=learning_rate_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=learning_rate_critic)

    train_rewards = []
    test_rewards = []
    start_time = time.time()

    # Epsilon parameters
    EPSILON_MAX = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995  # Decay factor for epsilon

    # A2C Training
    for episode in range(max_episodes):
        state = train_env.reset()
        episode_reward = 0
        epsilon = max(EPSILON_MIN, EPSILON_MAX * (EPSILON_DECAY ** episode))  # Decay epsilon

        for step in range(max_steps_per_episode):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Calculate action probabilities
            action_probs = actor(state_tensor)

            # Choose action
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)  # Random action with probability epsilon
            else:
                action = torch.multinomial(action_probs, 1).item()  # Choose action according to probabilities

            # Take action
            next_state, reward, done, _ = train_env.step(action)
            episode_reward += reward

            # Calculate advantages
            next_state_value = critic(torch.FloatTensor(next_state).unsqueeze(0))
            td_target = reward + gamma * next_state_value * (1 - done)
            td_error = td_target - critic(state_tensor)

            # Actor loss
            log_action_prob = torch.log(action_probs[0, action])
            actor_loss = -log_action_prob * td_error

            # Critic loss
            critic_loss = 0.5 * torch.square(td_error)

            # Update networks
            optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            if done:
                break

            state = next_state

        train_rewards.append(episode_reward)

        if episode % PRINT_EVERY == 0:
            mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
            #print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Epsilon: {epsilon:.3f} |')

        # Evaluation on test environment
        if episode % PRINT_EVERY == 0:
            test_reward = 0
            for _ in range(N_TRIALS):
                state = test_env.reset()
                for _ in range(max_steps_per_episode):
                    action = torch.argmax(actor(torch.FloatTensor(state).unsqueeze(0))).item()
                    state, reward, done, _ = test_env.step(action)
                    test_reward += reward
                    if done:
                        break
            test_rewards.append(test_reward / N_TRIALS)
            
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

            if mean_test_rewards >= REWARD_THRESHOLD_LUNAR_LANDER:
                end_time = time.time()
                duration = end_time - start_time
                print(f'Reached reward threshold in {episode} episodes for Lunar Lander')
                break

    env.close()

# Run for 5 times
for i in range(5):
    train_a2c(env, env, num_episodes, None)
