import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.functional import mse_loss

# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
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

# Initialize the environment
env = gym.make('LunarLander-v2')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Initialize the Actor-Critic network
model = ActorCritic(input_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
MAX_EPISODES = 1000
REWARD_THRESHOLD = 200  # Desired average reward to solve the environment
WINDOW_SIZE = 100  # Window size for computing average reward
reward_window = []  # Window for storing recent rewards

# Training loop
for episode in range(MAX_EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs, value = model(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        _, next_value = model(next_state_tensor)

        # Compute advantage
        advantage = reward + 0.99 * next_value - value

        # Actor loss
        actor_loss = -dist.log_prob(action) * advantage.detach()

        # Critic loss
        critic_loss = mse_loss(value, reward + 0.99 * next_value)

        # Total loss
        loss = actor_loss + critic_loss

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    # Store the total reward of the episode
    reward_window.append(total_reward)

    # Remove oldest reward if window size exceeded
    if len(reward_window) > WINDOW_SIZE:
        reward_window.pop(0)

    # Compute average reward over the window
    avg_reward = np.mean(reward_window)

    print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")

    # Check if the environment is solved
    if avg_reward >= REWARD_THRESHOLD:
        print(f"Environment solved in {episode+1} episodes!")
        break

# Save the trained model
print("Model saved as lunar_lander_model.pth")
