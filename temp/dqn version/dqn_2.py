import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import gym

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LEARNING_RATE = 0.001 # learning rate
UPDATE_EVERY = 4  # how often to update the network

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = T.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.transitions = namedtuple("Transitions", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = self.transitions(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        transitions = random.sample(self.memory, k=self.batch_size)

        states = T.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float()
        actions = T.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long()
        rewards = T.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float()
        next_states = T.from_numpy(np.vstack([e.next_state for e in transitions if e is not None])).float()
        dones = T.from_numpy(np.vstack([e.done for e in transitions if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory.""" 
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size, seed, l2_reg=0.1): 
        random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        
        # Add L2 regularization to the optimizer
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=LEARNING_RATE, weight_decay=l2_reg)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY stepsilon)
        self.timestep = 0

    def step(self, state, action, reward, next_state, done):
        # Save transitions in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time step.
        self.timestep = (self.timestep + 1) % UPDATE_EVERY
        if self.timestep == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                transitions = self.memory.sample()
                self.learn(transitions, GAMMA)

    def act(self, state, epsilon=0.0):
        state = T.from_numpy(state).float().unsqueeze(0)
        self.qnetwork.eval()
        with T.no_grad():

            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, transitions, gamma):
        states, actions, rewards, next_states, dones = transitions

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork, self.qnetwork_target, TAU)

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
    
def train_dqn(train_env, test_env):
    MAX_EPISODES = 2000 # Maximum number of episodes to run
    N_TRIALS = 100
    PRINT_EVERY = 10
    max_timesteps = 1000 # maximum number of timesteps per episode
    consecutive_episodes = 0 # Number of consecutive episodes that have reached the reward threshold
    REWARD_THRESHOLD_CARTPOLE = 195 # Reward threshold for CartPole
    REWARD_THRESHOLD_LUNAR_LANDER = 200 # Reward threshold for Lunar Lander

    # Initialize the agent based on the environment
    if train_env.unwrapped.spec.id == "LunarLander-v2":
        agent = Agent(state_size=8, action_size=4, seed=0)
    elif train_env.unwrapped.spec.id == "CartPole-v0":
        agent = Agent(state_size=4, action_size=2, seed=0)
    else:
        raise ValueError("Unsupported environment")

    train_rewards = []  # list containing rewards from each episode
    test_rewards = []  # list containing rewards from each test episode
    epsilon = 1.0  # initialize epsilon
    epsilon_decay = 0.995  # epsilon decay
    epsilon_min = 0.01  # minimum epsilon
    
    for episode in range(1, MAX_EPISODES + 1):
        state = train_env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = train_env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
        train_rewards.append(episode_reward)  # save train reward
        epsilon = max(epsilon_min, epsilon_decay * epsilon)  # decrease epsilon

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