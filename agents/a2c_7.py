import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

class A2C(nn.Module):  # Modified to inherit from nn.Module
    def __init__(self, input_size, hidden_size, output_size, lr_actor=0.001, lr_critic=0.001):
        super(A2C, self).__init__()  # Call the parent class constructor
        self.actor_critic = ActorCritic(input_size, hidden_size, output_size)
        self.optimizer_actor = optim.Adam(self.actor_critic.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=lr_critic)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs, _ = self.actor_critic(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, env, optimizer, gamma=0.99):
        self.train()

        states = []
        actions = []
        log_prob_actions = []
        values = []
        rewards = []
        done = False
        episode_reward = 0

        state = env.reset()

        while not done:
            if isinstance(state, tuple):
                state, _ = state

            state = torch.FloatTensor(state).unsqueeze(0)
            states.append(state)
            action_pred, value_pred = self.actor_critic(state)
            action_prob = nn.functional.softmax(action_pred, dim=-1)
            dist = Categorical(action_prob)

            action = dist.sample()
            log_prob_action = dist.log_prob(action)

            state, reward, done, _ = env.step(action.item())

            actions.append(action)
            log_prob_actions.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)
            episode_reward += reward

        states = torch.cat(states)
        actions = torch.cat(actions)
        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values).squeeze(-1)

        returns = calculate_returns(rewards, gamma)
        advantages = calculate_advantages(returns, values)

        policy_loss, value_loss = update_policy(self, states, actions, advantages, log_prob_actions, returns, values,
                                                optimizer)

        return policy_loss, value_loss, episode_reward

    def evaluate(self, env):
        self.eval()

        done = False
        episode_reward = 0

        state = env.reset()

        while not done:

            if isinstance(state, tuple):
                state, _ = state
            state = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():

                action_pred, _ = self.actor_critic(state)

                action_prob = nn.functional.softmax(action_pred, dim=-1)

            action = torch.argmax(action_prob, dim=-1)

            state, reward, done, _ = env.step(action.item())

            episode_reward += reward

        return episode_reward

def calculate_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * gamma
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def update_policy(policy, states, actions, advantages, log_prob_actions, returns, values, optimizer):
    advantages = advantages.detach()
    returns = returns.detach()

    action_pred, value_pred = policy.actor_critic(states)
    value_pred = value_pred.squeeze(-1)
    action_prob = nn.functional.softmax(action_pred, dim=-1)
    dist = Categorical(action_prob)

    new_log_prob_actions = dist.log_prob(actions)

    policy_ratio = torch.exp(new_log_prob_actions - log_prob_actions)
    clipped_policy_ratio = torch.clamp(policy_ratio, min=1.0 - 0.2, max=1.0 + 0.2)

    policy_loss_1 = policy_ratio * advantages
    policy_loss_2 = clipped_policy_ratio * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    value_pred_expanded = value_pred.expand_as(returns)
    value_loss = nn.functional.smooth_l1_loss(returns, value_pred_expanded).mean()

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()

def train_a2c(train_env, test_env):
    MAX_EPISODES = 2000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 100
    PRINT_EVERY = 10
    LEARNING_RATE = 0.001

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    actor = ActorCritic(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    policy = A2C(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    train_rewards = []
    test_rewards = []

    consecutive_episodes = 0
    REWARD_THRESHOLD_CARTPOLE = 195

    for episode in range(1, MAX_EPISODES + 1):
        policy_loss, value_loss, train_reward = policy.update(train_env, optimizer, DISCOUNT_FACTOR)
        test_reward = policy.evaluate(test_env)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

        if test_env.unwrapped.spec.id == 'CartPole-v0':
            if mean_train_rewards >= REWARD_THRESHOLD_CARTPOLE:
                consecutive_episodes += 1
                if consecutive_episodes >= 100:
                    print(f'Reached reward threshold in {episode} episodes')


train_env = gym.make('CartPole-v0')
test_env = gym.make('CartPole-v0')

train_a2c(train_env, test_env)
