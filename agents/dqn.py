import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import gym

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_unit=64, fc2_unit=64):
        super(QNetwork, self).__init__() 
        self.seed = T.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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

        states = T.from_numpy(np.vstack([e.state for e in transitionss if e is not None])).float().to()
        actions = T.from_numpy(np.vstack([e.action for e in transitionss if e is not None])).long().to()
        rewards = T.from_numpy(np.vstack([e.reward for e in transitionss if e is not None])).float().to()
        next_states = T.from_numpy(np.vstack([e.next_state for e in transitionss if e is not None])).float().to()
        dones = T.from_numpy(np.vstack([e.done for e in transitionss if e is not None]).astype(np.uint8)).float().to()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 0.0005  # learning rate
UPDATE_EVERY = 4  # how often to update the network

class Agent:
    def __init__(self, state_size, action_size, seed):
        random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY stepsilon)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save transitions in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time stepsilon.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                transitionss = self.memory.sample()
                self.learn(transitionss, GAMMA)

    def act(self, state, epsilon=0.):
        state = T.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with T.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, transitionss, gamma):
        states, actions, rewards, next_states, dones = transitionss

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def evaluate(env, agent):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward
    
def train_dqn(env):
    MAX_EPISODES = 2000
    N_TRIALS = 25
    REWARD_THRESHOLD = 200
    PRINT_EVERY = 10
    LEARNING_RATE = 0.0005
    max_timesteps = 1000 # maximum number of timesteps per episode

    agent = Agent(state_size=8, action_size=4, seed=0)

    train_rewards = []  # list containing rewards from each episode
    test_rewards = []  # list containing rewards from each test episode
    epsilon = 1.0  # initialize epsilon
    epsilon_decay = 0.995  # epsilon decay
    epsilon_min = 0.01  # minimum epsilon
    
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        train_rewards.append(episode_reward)  # save train reward
        epsilon = max(epsilon_min, epsilon_decay * epsilon)  # decrease epsilon

        test_reward = evaluate(env, agent)
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

# initialize environment
env = gym.make('LunarLander-v2')
env.seed(0)

# run the training session
rewards = train_dqn(env)