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
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.transitions = namedtuple("Transitions", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.transitions(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        transitionss = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in transitionss if e is not None])).float().to()
        actions = torch.from_numpy(np.vstack([e.action for e in transitionss if e is not None])).long().to()
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitionss if e is not None])).float().to()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in transitionss if e is not None])).float().to()
        dones = torch.from_numpy(np.vstack([e.done for e in transitionss if e is not None]).astype(np.uint8)).float().to()

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

def train(env, policy, optimizer, discount_factor, replay_buffer):
    policy.train()

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
        action_pred = policy.actor(state)
        value_pred = policy.critic(state)

        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)

        action = dist.sample()

        log_prob_action = dist.log_prob(action)
        state, reward, done, _ = env.step(action.item())

        replay_buffer.add(state, action.item(), reward, state, done)

        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(policy, log_prob_actions, advantages, returns, optimizer)

    return policy_loss.item(), value_loss.item(), episode_reward
    
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

def train_a2c_buffer(train_env, test_env): 
    MAX_EPISODES = 2000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 200
    PRINT_EVERY = 10
    LEARNING_RATE = 0.0001

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = test_env.action_space.n

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weights)

    BUFFER_SIZE = 10000  # replay buffer size
    BATCH_SIZE = 128  # minibatch size
    BUFFER_SEED = 0 
    replay_buffer = ReplayBuffer(OUTPUT_DIM, BUFFER_SIZE, BATCH_SIZE, BUFFER_SEED)

    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    train_rewards = []
    test_rewards = []

    for episode in range(1, MAX_EPISODES+1):
        
        policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, replay_buffer)
        
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