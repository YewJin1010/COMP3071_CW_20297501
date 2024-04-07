import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import numpy as np
import gym

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        if len(self.memory) < batch_size:
            #print("len(self.memory): ", len(self.memory))
            #print("batch_size: ", batch_size)
            raise ValueError("Sample larger than population")
        batch = random.sample(self.memory, batch_size)
       
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array([s.numpy().squeeze() if isinstance(s, torch.Tensor) else s for s in states])
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones
    
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

def train(env, policy, optimizer, discount_factor, replay_buffer, batch_size):
    policy.train()
    
    # Sample a mini-batch from the replay buffer
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample_batch(batch_size)
    
    log_prob_actions = []
    values = []
    rewards = []
    actions = []
    states = []
    episode_reward = 0

    state = env.reset()

    for state, action, reward, next_state, done in zip(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        
        state = torch.FloatTensor(state).unsqueeze(0)
        states.append(state)
        action = torch.tensor(action).unsqueeze(0)  # Convert action to tensor
        action_pred = policy.actor(state)
        value_pred = policy.critic(state)

        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)

        actions.append(action)
        log_prob_action = dist.log_prob(action)
        next_state, reward, done, _ = env.step(action.item())

        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward

        # Add transition to replay buffer
        replay_buffer.add(state, action.item(), reward, next_state, done)

    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)
    
    policy_loss, value_loss = update_policy(advantages, log_prob_actions, returns, values, optimizer)

    return policy_loss, value_loss, episode_reward

def calculate_returns(batch_rewards, discount_factor, normalize = True):
    returns = []
    R = 0
    for r in reversed(batch_rewards):
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

def update_policy(advantages, log_prob_actions, returns, values, optimizer):
    
    advantages = advantages.detach()
    returns = returns.detach()
        
    policy_loss = - (advantages * log_prob_actions).sum()
    
    value_loss = F.smooth_l1_loss(returns, values).sum()
        
    optimizer.zero_grad()
    
    policy_loss.backward()
    value_loss.backward()
    
    optimizer.step()
    
    return policy_loss.item(), value_loss.item()

def evaluate(env, policy):
    
    policy.eval()
    
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
        
            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)
                
        action = torch.argmax(action_prob, dim = -1)
                
        state, reward, done, _ = env.step(action.item())

        episode_reward += reward
    
    return episode_reward

def train_a2c_buffer(train_env, test_env): 
    MAX_EPISODES = 2000
    WARMUP_EPISODES = 100
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 100
    PRINT_EVERY = 10
    LEARNING_RATE = 0.001
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = test_env.action_space.n

    BUFFER_SIZE = int(1e5)
    BATCH_SIZE = 64

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weights)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    replay_buffer = ReplayBuffer(capacity=BUFFER_SIZE)

    train_rewards = []
    test_rewards = []

    # Replay buffer warm-up phase
    for _ in range(WARMUP_EPISODES):
        state = train_env.reset()
        done = False
        while not done:
            if isinstance(state, tuple):
                state, _ = state
            action = train_env.action_space.sample()
            next_state, reward, done, _ = train_env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
    
    for episode in range(1, MAX_EPISODES+1):
        
        policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, replay_buffer, BATCH_SIZE)
        
        test_reward = evaluate(test_env, policy)
        
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        #print("train rewards: ", train_rewards)
        #print("test rewards: ", test_rewards)
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

train_env = gym.make('CartPole-v0')
test_env = gym.make('CartPole-v0')

train_rewards, test_rewards, reward_threshold, episodes = train_a2c_buffer(train_env, test_env)