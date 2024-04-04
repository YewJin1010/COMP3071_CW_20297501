import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import numpy as np
import gym
import random 

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
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
    
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

def epsilon_greedy_action(action_probabilities, epsilon):
    if random.random() < epsilon:
        # Explore: Choose a random action
        action = random.randrange(len(action_probabilities))
    else:
        # Exploit: Choose the action with the highest probability
        action = torch.argmax(action_probabilities).item()
    return action

def train(env, policy, optimizer, discount_factor, learning_rate, epsilon):
    policy.train()
        
    # Initialize state means and stds (for online calculation)
    state_means = np.zeros(env.observation_space.shape[0])  # Adjust shape if needed
    state_stds = np.ones(env.observation_space.shape[0])  # Avoid zero division initially

    states = [] 
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()
    decay_factor = 0.99  # Decay factor for epsilon
    min_epsilon = 0.01

    while not done:

        if isinstance(state, tuple):
            state, _ = state

        state_means = (1 - learning_rate) * state_means + learning_rate * state  # Learning rate of LEARNING_RATE
        state_stds = (1 - learning_rate) * state_stds + learning_rate * (state - state_means) ** 2

        state = torch.FloatTensor((state - state_means) / state_stds).unsqueeze(0)

        states.append(state)        
        action_pred, value_pred = policy(state)
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        #action = dist.sample()        

        # Epsilon
        action = epsilon_greedy_action(action_prob, epsilon)
        action = dist.sample()        
        log_prob_action = dist.log_prob(action)
        state, reward, done, _= env.step(action.item())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        episode_reward += reward

        # Update epsilon for the next time step
        epsilon *= decay_factor
        epsilon = max(min_epsilon, epsilon)

    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    # Compute policy and value losses using a clipped surrogate objective
    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer)

    return policy_loss, value_loss, episode_reward

def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer):
    
    total_policy_loss = 0 
    total_value_loss = 0

    policy.train()

    action_pred, value_pred = policy(states)
    action_prob = F.softmax(action_pred, dim=-1)
    dist = distributions.Categorical(action_prob)
    
    new_log_prob_actions = dist.log_prob(actions)

    policy_loss = -(advantages * (new_log_prob_actions - log_prob_actions.detach())).mean()
    value_loss = F.smooth_l1_loss(returns, value_pred.squeeze(-1)).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    value_loss.backward()

    optimizer.step()

    total_policy_loss += policy_loss.item()
    total_value_loss += value_loss.item()

    return total_policy_loss, total_value_loss

def evaluate(env, policy):
    
    policy.eval()
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        if isinstance(state, tuple):
            state, _ = state
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
        
            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim=-1)
                
        action = torch.argmax(action_prob, dim=-1)
                
        state, reward, done, _= env.step(action.item())

        episode_reward += reward
    return episode_reward

def train_a2c_ep(train_env, test_env):
    MAX_EPISODES = 2000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 100
    PRINT_EVERY = 10
    LEARNING_RATE = 0.005
    consecutive_episodes = 0 # Number of consecutive episodes that have reached the reward threshold
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    epsilon = 0.1

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weights)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    train_rewards = []
    test_rewards = []

    for episode in range(1, MAX_EPISODES + 1):
        policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, LEARNING_RATE, epsilon)
        test_reward = evaluate(test_env, policy)
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')

        if test_env.unwrapped.spec.id == 'CartPole-v0':
            if mean_test_rewards >= REWARD_THRESHOLD_CARTPOLE:
                consecutive_episodes += 1
                if consecutive_episodes >= 100:
                    print(f'Reached reward threshold in {episode} episodes for CartPole')
                    return train_rewards, test_rewards, REWARD_THRESHOLD_CARTPOLE, episode
            else:
                consecutive_episodes = 0
        elif test_env.unwrapped.spec.id == 'LunarLander-v2':
            if mean_test_rewards >= REWARD_THRESHOLD_LUNAR_LANDER:
                print(f'Reached reward threshold in {episode} episodes for Lunar Lander')
                return train_rewards, test_rewards, REWARD_THRESHOLD_LUNAR_LANDER, episode

    print("Did not reach reward threshold")
    return train_rewards, test_rewards, None, episode

"""
train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

train_rewards, test_rewards, _, _ = train_a2c(train_env, test_env)
"""