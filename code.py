import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import warnings

# Suppress deprecated numpy bool warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
MAX_EPISODES = 500
MAX_STEPS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Q-network
class DQN(nn.Module):
    def _init_(self, obs_dim, act_dim):
        super(DQN, self)._init_()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

# Experience Replay Buffer
class ReplayBuffer:
    def _init_(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*transitions))

    def _len_(self):
        return len(self.buffer)

# Epsilon-greedy action selection
def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.argmax().item()

# Train DQN
def train():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = DQN(obs_dim, act_dim).to(device)
    target_net = DQN(obs_dim, act_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    episode_rewards = []

        for episode in range(1, MAX_EPISODES + 1):
            state, _ = env.reset()
            total_reward = 0

        for _ in range(MAX_STEPS):
            action = select_action(state, policy_net, epsilon, act_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Penalize if episode ends
            reward = reward if not done else -10

            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

                # Compute current Q values
                q_values = policy_net(states).gather(1, actions)

                # Compute target Q values
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q_values = rewards + GAMMA * next_q_values * (~dones)

                loss = nn.MSELoss()(q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()

if _name_ == "_main_":
    train()
