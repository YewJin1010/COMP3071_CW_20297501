import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import gym
MAX_EPISODE_DURATION = 300 
SOLVED_THRESHOLD = 100
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
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

def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip):
        
    policy.train()
        
    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    violations = []
    time = 0

    state = env.reset()

    while not done and time < MAX_EPISODE_DURATION:

        if isinstance(state, tuple):
            print("State 1: ", state)
            state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0)
        else:
            print("State 2: ", state)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        #append state here, not after we get the next state from env.step()
        states.append(state)
        
        action_pred, value_pred = policy(state)
                
        action_prob = F.softmax(action_pred, dim = -1)
                
        dist = distributions.Categorical(action_prob)
        
        action = dist.sample()
        
        log_prob_action = dist.log_prob(action)
        
        state, reward, done, truncated, _ = env.step(action.item())

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        
        violation = _.get('constraint_costs', [0])
        violations.append(violation)

        episode_reward += reward
        time += 1
    
    states = torch.cat(states)
    actions = torch.cat(actions)    
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    violations_flat = [item for sublist in violations for item in sublist]
    total_violations = sum(violations_flat)
    
    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, episode_reward, total_violations, time

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

def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    
    total_policy_loss = 0 
    total_value_loss = 0
    
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
        
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
    
        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def train_ppo(env):
    SEED = None
    env.seed(SEED)
    np.random.seed(SEED)

    INPUT_DIM = env.observation_space.shape[0]
    HIDDEN_DIM = 128
    OUTPUT_DIM = env.action_space.n

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)

    policy.apply(init_weights)

    LEARNING_RATE = 0.01

    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    MAX_EPISODES = 1000
    DISCOUNT_FACTOR = 0.99
    REWARD_THRESHOLD = 195
    PPO_STEPS = 5
    PPO_CLIP = 0.2

    train_rewards = []
    violation_list = []
    avg_rewards_list = []
    avg_violations_list = []
    duraiton_list = []
    avg_durations = []
    avg_durations_list = []
    successful_episode = 0
    for episode in range(1, MAX_EPISODES+1):
        #env.render()
        
        policy_loss, value_loss, train_reward, total_violations, time = train(env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)
        print("POLICY LOSS: ", policy_loss)
        print("VALUE_LOSS: ", value_loss)

        train_rewards.append(train_reward)
        print("TRAIN REWARDS: ", train_rewards)
        print("LEN TRAIN REWARDS: ", len(train_rewards))

        violation_list.append(total_violations)
        duraiton_list.append(time)

        print("Episode", episode, "duration: ", time)


        if len(train_rewards) >= 100:
            print("EPISODE NUMBER: ", episode)

            prev_episodes = train_rewards[len(train_rewards) - 100:]
            avg_reward = sum(prev_episodes) / len(prev_episodes)
            avg_rewards_list.append(avg_reward)

            prev_violations = violation_list[len(violation_list) - 100:]
            avg_violations = sum(prev_violations) / len(prev_violations)
            avg_violations_int = int(avg_violations)
            avg_violations_list.append(avg_violations_int)

            prev_durations = duraiton_list[len(duraiton_list) - 100:]
            avg_durations = sum(prev_durations) / len(prev_durations)
            avg_durations_list.append(avg_durations)

            if avg_reward > REWARD_THRESHOLD:
                print("Average number of violations: ", avg_violations_int)
                print("Average duration: ", avg_durations)
                
                successful_episode += 1

                print("Successful Episode: ", successful_episode)

                if successful_episode > SOLVED_THRESHOLD:
                    print(f"Solved with average reward {avg_reward} at {episode} episodes")
                    env.close()
                    return avg_rewards_list, avg_violations_list, episode, avg_durations_list

    return avg_rewards_list, avg_violations_list, episode, avg_durations_list

env = gym.make('CartPole-v1')
avg_rewards_list, avg_violations_list, episode, avg_durations_list = train_ppo(env)