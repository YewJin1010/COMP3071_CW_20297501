import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym

EPSILON_CLIP = 0.2 # PPO clip parameter

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
        # if x is not tensor, convert it to tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
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

def train(env, policy, optimizer, discount_factor):
    
    policy.train()
    
    log_prob_actions_old = []  # Store old log probabilities
    log_prob_actions_new = []  # Store new log probabilities
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        if isinstance(state, tuple):
            state, _ = state

        state = torch.FloatTensor(state).unsqueeze(0)
        action_pred = policy.actor(state)
        value_pred = policy.critic(state)

        action_prob = F.softmax(action_pred, dim=-1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()
        
        log_prob_action_old = dist.log_prob(action)
        log_prob_actions_old.append(log_prob_action_old)

        state, reward, done, _ = env.step(action.item())
        
        # Calculate new log probability of action
        action_prob = F.softmax(policy.actor(state), dim=-1)
        log_prob_action_new = distributions.Categorical(action_prob).log_prob(action)
        log_prob_actions_new.append(log_prob_action_new)

        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward
    
    log_prob_actions_old = torch.cat(log_prob_actions_old)
    log_prob_actions_new = torch.cat(log_prob_actions_new)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    
    policy_loss, value_loss = update_policy(advantages, log_prob_actions_old, log_prob_actions_new, returns, values, optimizer)

    return policy_loss, value_loss, episode_reward

def calculate_returns(rewards, discount_factor, normalize = True):
    
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)   
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values, normalize = True):

    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()
    return advantages

def update_policy(advantages, log_prob_actions_old, log_prob_actions_new, returns, values, optimizer):
    advantages = advantages.detach()
    returns = returns.detach()

    # Calculate ratio of new policy probability to old policy probability
    ratio = torch.exp(log_prob_actions_new - log_prob_actions_old)

    # Calculate surrogate objective
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1.0 - EPSILON_CLIP, 1.0 + EPSILON_CLIP) * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()

    # Compute value loss
    value_loss = F.smooth_l1_loss(returns, values).mean()

    # Total loss
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()

def evaluate(env, policy):
    
    policy.eval()
    
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        
        if isinstance(state, tuple):
            state, _ = state
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
        
            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)
                
        action = torch.argmax(action_prob, dim = -1)
                
        state, reward, done, _ = env.step(action.item())

        episode_reward += reward
        
    return episode_reward

def train_a2c_ppo(train_env, test_env): 
    
    MAX_EPISODES = 2000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 200
    PRINT_EVERY = 10

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = test_env.action_space.n

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weights)

    LEARNING_RATE = 0.0005

    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    train_rewards = []
    test_rewards = []

    for episode in range(1, MAX_EPISODES+1):
        
        policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR)
        
        test_reward = evaluate(test_env, policy)
        
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        
        if episode % PRINT_EVERY == 0:
        
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')
        
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            return train_rewards, test_rewards, REWARD_THRESHOLD
     
    print("Did not reach reward threshold")
    return train_rewards, test_rewards, REWARD_THRESHOLD

train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

train_rewards, test_rewards, reward_threshold = train_a2c_ppo(train_env, test_env)