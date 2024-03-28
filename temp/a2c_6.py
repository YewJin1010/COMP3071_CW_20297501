import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

class ActorCritic(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_values = self.critic(x)
        return action_probs, state_values

def evaluate(env, model):
    total_reward = 0
    state = env.reset()
    done = False

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = model(state)
        action = torch.argmax(action_probs, dim=-1)
        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward
        state = next_state

    return total_reward

def train_a2c(env, model, optimizer, max_episodes, print_every, reward_threshold):
    train_rewards = []
    test_rewards = []
    n_trials = 25
    print_episode = 10

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = model(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            # Compute TD error
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_state_value = model(next_state)
            td_target = reward + 0.99 * next_state_value * (1 - done)
            td_error = td_target - state_value

            # Compute losses
            actor_loss = -dist.log_prob(action) * td_error.detach()
            critic_loss = td_error.pow(2)

            # Update network weights
            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        train_rewards.append(episode_reward)
        mean_train_rewards = np.mean(train_rewards[-n_trials:])
        test_reward = evaluate(env, model)
        test_rewards.append(test_reward)
        mean_test_rewards = np.mean(test_rewards[-n_trials:])

        if episode % print_episode == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

        if mean_test_rewards >= reward_threshold:
            print(f'Reached reward threshold in {episode} episodes')
            return train_rewards, test_rewards, reward_threshold, episode

    print("Did not reach reward threshold")
    return train_rewards, test_rewards, reward_threshold, episode

# Environment
env = gym.make('LunarLander-v2')
input_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# Model
model = ActorCritic(input_dim, num_actions)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
train_rewards, test_rewards, reward_threshold, episode = train_a2c(env, model, optimizer, max_episodes=2000, print_every=10, reward_threshold=200)
