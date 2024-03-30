import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

import gym

LR = .01  # Learning rate
SEED = None  # Random seed for reproducibility
MAX_EPISODES = 2000  # Max number of episodes
REWARD_THRESHOLD = 195
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
MAX_EPISODE_DURATION = 300 
SOLVED_THRESHOLD = 100
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return np.random.choice(self.buffer, batch_size)

    
class A2C_Buffer(nn.Module):

    def __init__(self, env, hidden_size=128, gamma=.99):
        """
        Assumes fixed continuous observation space
        and fixed discrete action space (for now)

        :param env: target gym environment
        :param gamma: the discount factor parameter for expected reward function :float
        """
        super().__init__()

        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.in_size = len(env.observation_space.sample().flatten())
        self.out_size = self.env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size)
        ).double()

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).double()

    def train_env_episode(self, render=False, replay_buffer=None):
        """
        Runs one episode and collects critic values, expected return,
        :return: A tensor with total/expected reward, critic eval, and action information
        """
        rewards = []
        critic_vals = []
        action_lp_vals = []
        
        # Run episode and save information
        observation = self.env.reset()
        done = False
        time = 0 # Episode duration timer
        while not done and time < MAX_EPISODE_DURATION:
            if render:
                self.env.render()

            # Check if the observation is a tuple
            if isinstance(observation, tuple):
                observation = observation[0]
                
            observation = torch.from_numpy(observation).double()

            # Get action from actor
            action_logits = self.actor(observation)

            action = Categorical(logits=action_logits).sample()

            # Get action probability
            action_log_prob = action_logits[action]

            # Get value from critic
            pred = torch.squeeze(self.critic(observation).view(-1))

            # Write prediction and action/probabilities to arrays
            action_lp_vals.append(action_log_prob)
            critic_vals.append(pred)

            # Send action to environment and get rewards, next state
            next_observation, reward, done, info = self.env.step(action.item())
            rewards.append(torch.tensor(reward).double())

            # Store transition in replay buffer
            if replay_buffer is not None:
                replay_buffer.push(observation, action, reward, next_observation, done)

            observation = next_observation

        total_reward = sum(rewards)
        # Convert reward array to expected return and standardize
        for t_i in range(len(rewards)):

            for t in range(t_i + 1, len(rewards)):
                rewards[t_i] += rewards[t] * (self.gamma ** (t_i - t))

        # Convert output arrays to tensors using torch.stack
        def f(inp):
            return torch.stack(tuple(inp), 0)

        # Standardize rewards
        rewards = f(rewards)
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + .000000000001)

        return rewards, f(critic_vals), f(action_lp_vals), total_reward, time

    
    @staticmethod
    def compute_loss(action_p_vals, G, V, critic_loss=nn.SmoothL1Loss()):
        """
        Actor Advantage Loss, where advantage = G - V
        Critic Loss, using mean squared error
        :param critic_loss: loss function for critic   :Pytorch loss module
        :param action_p_vals: Action Log Probabilities  :Tensor
        :param G: Actual Expected Returns   :Tensor
        :param V: Predicted Expected Returns    :Tensor
        :return: Actor loss tensor, Critic loss tensor  :Tensor
        """
        assert len(action_p_vals) == len(G) == len(V)
        advantage = G - V.detach()
        return -(torch.sum(action_p_vals * advantage)), critic_loss(G, V)


def train_a2c_buffer(env):

    agent = A2C_Buffer(env)

    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

    # Init optimizers
    actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
    critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

    r = []  # Array containing total rewards
    avg_r = 0  # Value storing average reward over last 100 episodes
    avg_rewards_list = []
    successful_episode = 0
    for i in range(MAX_EPISODES):
        print("EPISODE: ", i)
        critic_optim.zero_grad()
        actor_optim.zero_grad()

        rewards, critic_vals, action_lp_vals, total_reward, time = agent.train_env_episode(render=False, replay_buffer=replay_buffer)
        r.append(total_reward)

        l_actor, l_critic = agent.compute_loss(action_p_vals=action_lp_vals, G=rewards, V=critic_vals)

        l_actor.backward()
        l_critic.backward()

        actor_optim.step()
        critic_optim.step()

        # Check average reward every 100 episodes, print, and end script if solved
        if len(r) >= 100:  # check average every 100 episodes
            print("EPISODE NUMBER: ", i)

            prev_episodes = r[len(r) - 100:]
            avg_r = sum(prev_episodes) / len(prev_episodes)
            avg_r_float = avg_r.item()
            avg_rewards_list.append(avg_r_float)


            if avg_r_float > REWARD_THRESHOLD:
         
                successful_episode += 1

                print("Successful Episode: ", successful_episode)

                if successful_episode > SOLVED_THRESHOLD:
                    print(f"Solved with average reward {avg_r_float} at {i} episodes")
                    env.close()
                    return avg_rewards_list, i

    return avg_rewards_list, i

env = gym.make('CartPole-v0')
train_a2c_buffer(env)