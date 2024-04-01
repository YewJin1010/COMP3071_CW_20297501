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
class A2C:
    def __init__(self, input_size, hidden_size, output_size, lr_actor=0.001, lr_critic=0.001):
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

    def update(self, states, actions, rewards, next_states, dones, gamma=0.99):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)

        _, critic_values = self.actor_critic(states_tensor)
        _, next_critic_values = self.actor_critic(next_states_tensor)

        # Convert rewards and dones to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        td_targets = rewards_tensor + gamma * next_critic_values * (1 - dones_tensor)
        advantages = td_targets - critic_values

        # Actor loss
        log_probs = []
        for action in actions:
            _, log_prob = self.select_action(states[action])
            log_probs.append(log_prob)
        actor_loss = -torch.stack(log_probs).mean()

        # Critic loss
        critic_loss = nn.MSELoss()(critic_values, td_targets.detach())

        # Update networks
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        return actor_loss.item(), critic_loss.item()

def train_a2c(train_env, test_env):
    input_size = train_env.observation_space.shape[0]
    hidden_size = 128
    output_size = train_env.action_space.n
    a2c_agent = A2C(input_size, hidden_size, output_size)

    max_episodes = 2000
    max_steps = 1000
    gamma = 0.99
    N_TRIALS = 100
    REWARD_THRESHOLD = 195  # Same as the CartPole environment

    train_rewards = []
    test_rewards = []

    for episode in range(max_episodes):
        state = train_env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        total_reward = 0

        for step in range(max_steps):
            action, log_prob = a2c_agent.select_action(state)
            next_state, reward, done, _ = train_env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            total_reward += reward

            if done:
                break

        actor_loss, critic_loss = a2c_agent.update(states, actions, rewards, next_states, dones, gamma)

        train_rewards.append(total_reward)


        if episode % N_TRIALS == 0:
            avg_train_reward = sum(train_rewards[-N_TRIALS:]) / N_TRIALS
            print(f"Average training reward over last {N_TRIALS} episodes: {avg_train_reward}")

            if avg_train_reward >= REWARD_THRESHOLD:
                print(f"Lunar Lander environment solved in {episode} episodes.")
                break

        if episode % 10 == 0:
            test_reward = 0
            state = test_env.reset()
            for _ in range(max_steps):
                action, _ = a2c_agent.select_action(state)
                state, reward, done, _ = test_env.step(action)
                test_reward += reward
                if done:
                    break

            test_rewards.append(test_reward)
            print(f"Episode: {episode}, Test reward: {test_reward}, Actor loss: {actor_loss}, Critic loss: {critic_loss}")

    return train_rewards, test_rewards

train_env = gym.make("LunarLander-v2")
test_env = gym.make("LunarLander-v2")

train_rewards, test_rewards = train_a2c(train_env, test_env)
