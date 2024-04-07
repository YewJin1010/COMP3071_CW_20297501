import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import random
from collections import deque, namedtuple
import numpy as np
import gym

class ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = deque(maxlen=capacity)
    self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

  def add(self, state, action, reward, next_state, done):
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)

  def sample_batch(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = np.array([s.numpy().squeeze() if isinstance(s, torch.Tensor) else s for s in states])
    actions = np.array([a.numpy().squeeze() if isinstance(a, torch.Tensor) else a for a in actions])
    rewards = np.array(rewards)
    next_states = np.array([np.squeeze(s) for s in next_states])
    dones = np.array(dones)

    return states, actions, rewards, next_states, dones
  
  def __len__(self):
    return len(self.memory) 
  
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
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample_batch(batch_size)

    # Convert batch data to tensors for efficient processing
    batch_states = torch.FloatTensor(batch_states)
    batch_actions = torch.LongTensor(batch_actions)  # Assuming discrete actions
    batch_rewards = torch.FloatTensor(batch_rewards)
    batch_dones = torch.FloatTensor(batch_dones)

    # Leverage the policy network for batch-wise predictions
    batch_action_pred, batch_value_pred = policy(batch_states)

    # Calculate loss based on batch data
    batch_action_prob = F.softmax(batch_action_pred, dim=-1)
    dist = distributions.Categorical(batch_action_prob)
    batch_log_prob_actions = dist.log_prob(batch_actions)

    batch_returns = calculate_returns(batch_rewards, discount_factor)
    batch_advantages = calculate_advantages(batch_returns, batch_value_pred)

    policy_loss, value_loss = update_policy(batch_advantages, batch_log_prob_actions, batch_returns, batch_value_pred, optimizer)
    return policy_loss, value_loss


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
    return returns

def calculate_advantages(returns, values, normalize=True):
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
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 100
    PRINT_EVERY = 10
    LEARNING_RATE = 0.001
    consecutive_episodes = 0 # Number of consecutive episodes that have reached the reward threshold
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander
    
    BUFFER_SIZE = int(1e5)
    BATCH_SIZE = 64
    WARMUP_EPISODES = 100

    replay_buffer = ReplayBuffer(capacity=BUFFER_SIZE)

    INPUT_DIM = train_env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = train_env.action_space.n

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)
    policy.apply(init_weights)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

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

    for episode in range(1, MAX_EPISODES + 1):
     
        if len(replay_buffer) >= BATCH_SIZE or episode == 1:  # Train only if enough samples in buffer
            policy_loss, value_loss = train(train_env, policy, optimizer, DISCOUNT_FACTOR, replay_buffer, BATCH_SIZE)

        # Get the episode reward
        train_reward = evaluate(train_env, policy)
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


train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

train_rewards, test_rewards, _, _ = train_a2c_buffer(train_env, test_env)
