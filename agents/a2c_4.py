import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import torch.nn.functional as F

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

# A2C Agent
class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor_critic.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.actor_critic.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.actor_critic(state)
        action_probs = F.softmax(action_probs, dim=1)  # Apply softmax to convert logits to probabilities
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()


    def update(self, states, actions, rewards, next_states, dones):
        # Convert rewards to a Python list
        rewards = [rewards]

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Compute TD targets
        _, critic_values = self.actor_critic(next_states)
        td_targets = rewards + self.gamma * critic_values * (1 - dones)

        # Compute advantage estimates
        _, critic_values = self.actor_critic(states)
        advantages = td_targets - critic_values.detach()

        # Actor loss
        action_probs, _ = self.actor_critic(states)
        action_probs = action_probs.view(-1, self.num_actions)  # Ensure correct shape
        action_dist = torch.distributions.Categorical(F.softmax(action_probs, dim=1))
        log_probs = action_dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        critic_loss = nn.MSELoss()(critic_values.squeeze(), td_targets.detach())

        # Update networks
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

# Train A2C Agent
def train_a2c(agent, env, max_episodes=1000):
    scores = []
    score_window = deque(maxlen=100)
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        scores.append(total_reward)
        score_window.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Score: {np.mean(score_window)}")
        if np.mean(score_window) >= 200:
            print(f"Environment solved in {episode} episodes!")

            break
    return scores


# Run expeirment 5 times
for i in range(5):
    print(f"Experiment {i+1}")
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    agent = A2CAgent(state_dim, action_dim, hidden_dim)
    scores = train_a2c(agent, env)

    # Record results
    with open(f"results_a2c_{i+1}.txt", "w") as f:
        f.write("Episode,Score\n")
        for i, score in enumerate(scores):
            f.write(f"{i+1},{score}\n")
