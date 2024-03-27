import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.functional import mse_loss, softmax

# Actor-Critic network with gradient clipping
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

def train(env, model, optimizer, discount_factor=0.99, entropy_coef=0.001, grad_clip_norm=0.5):
    model.train()
    log_prob_actions, values, rewards = [], [], []
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs, value = model(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        episode_reward += reward

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        _, next_value = model(next_state_tensor)


        returns = torch.tensor(returns, dtype=torch.float32).detach()
        values_tensor = torch.tensor(values, dtype=torch.float32)  # Convert values list to tensor
        advantages = returns - values_tensor  

        # Actor loss
        actor_loss = (-dist.log_prob(action) * advantages.detach()).mean() 

        # Critic loss
        critic_loss = mse_loss(value, returns).mean() 

        # Total loss with entropy bonus
        entropy = -(action_probs * torch.log(action_probs)).sum()
        loss = actor_loss + critic_loss - entropy_coef * entropy

        # Update model with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        state = next_state

        log_prob_actions.append(dist.log_prob(action))
        values.append(value)
        rewards.append(reward)

    return actor_loss.item(), critic_loss.item(), episode_reward

def evaluate(env, policy):
    policy.eval()
    done = False
    episode_reward = 0

    state = env.reset()

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_prob, _ = policy(state_tensor)
            action_prob = softmax(action_prob, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
        state, reward, done, _ = env.step(action.item())
        episode_reward += reward

    return episode_reward

def train_a2c(train_env, test_env): 
   # Training parameters
    MAX_EPISODES = 2000
    REWARD_THRESHOLD = 200  # Desired average reward to solve the environment
    N_TRIALS = 25
    PRINT_EVERY = 10
    LEARNING_RATE = 0.0001

    input_size = train_env.observation_space.shape[0]
    output_size = train_env.action_space.n

    # Initialize the Actor-Critic network
    model = ActorCritic(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)

    train_rewards = []
    test_rewards = []

    # Training loop
    for episode in range(1, MAX_EPISODES+1):

        actor_loss, critic_loss, train_reward = train(train_env, model, optimizer)

        test_reward = evaluate(test_env, model)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:

            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f} | Mean Test Rewards: {mean_test_rewards:7.1f} |')
        
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            return train_rewards, test_rewards, REWARD_THRESHOLD, episode
     
    print("Did not reach reward threshold")
    return train_rewards, test_rewards, REWARD_THRESHOLD, episode

train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

train_a2c(train_env, test_env)
