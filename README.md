# finsearch_code-
This includes the coding part, involving solution to the inverted pendulum problem using DQN.

Inverted Pendulum Problem using DQN
Tanish Palsapure (24b3946)
Arnav Pandit (24b3946)
Imports and Setup


import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

In order to create the CartPole environment, the script first imports the necessary libraries, torch, and gym.for neural network construction and training, and torch.optimise for optimisation. Whereas random allows for randomisation in sampling and exploration, numpy manages numerical computations. Python's collections module provides a deque that is used to implement the experience replay buffer. Lastly, to prevent cluttering the output, a deprecated numpy warning is suppressed using warnings.

Hyperparameters


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

To regulate training, a number of hyperparameters are defined. The significance of future benefits is determined by the GAMMA (discount factor). LR stands for the optimizer's learning rate. Training batch and replay buffer sizes are managed by BATCH_SIZE and MEMORY_SIZE. DECAY, EPSILON_START, and EPSILON_END Set up the greedy-epsilon exploration approach. MAX_EPISODES and MAX_STEPS specify the overall training duration and steps per episode, whereas TARGET_UPDATE_FREQ determines the frequency of target network updates.

Device parameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Checks if GPU is available (torch.cuda.is_available()), uses GPU if true, otherwise CPU.

Q-Network definition
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

The neural network design for estimating Q-values is defined by the DQN class. With three fully connected layers—an input layer with 128 neurones, a hidden layer with 64 neurones, and an output layer that generates Q-values for every action—it is derived from torch.nn.Module. Non-linearity is introduced by ReLU activations, and the forward technique establishes the network's data flow.

Experience Replay Buffer


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

A fixed-size memory for storing previous transitions (state, action, reward, next_state, done) is implemented by the ReplayBuffer class. When full, it automatically discards the oldest experiences via a deque. The sample retrieves a batch at random for training, whereas the push technique adds a transition. The current number of stored transitions is returned by the __len__ method.

Epsilon-Greedy Action Selection


def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state)
        return q_values.argmax().item()

The epsilon-greedy strategy is used by the select_action function to balance exploration with exploitation. The agent choose a random action with probability epsilon in order to investigate novel possibilities. If not, the action with the greatest projected Q-value is chosen using the policy network. The state is processed without gradient tracking after being transformed into a PyTorch tensor.

Training Function


def train():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

First, we use gym.make('CartPole-v1') to construct the CartPole environment. env.observation_space.shape[0] yields the dimension of the observation space, obs_dim, while env.action_space.n yields the size of the action space, act_dim. These specify the agent's state inputs and potential actions.

Initialize Networks and Optimizer


    policy_net = DQN(obs_dim, act_dim).to(device)
    target_net = DQN(obs_dim, act_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())


    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)


    epsilon = EPSILON_START
    episode_rewards = []

While the target_net uses target_net.load_state_dict to occasionally copy weights from policy_net, the policy_net serves as the primary Q-network for learning and provides steady Q-value targets. The network weights are updated using an Adam optimiser, and previous training experiences are stored in a replay_buffer. The initial rate of exploration in the environment is controlled by the epsilon parameter.

Main training loop


    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0

A variable is initialised to monitor the overall reward accrued during the episode, and the environment is reset to its initial condition at the beginning of each episode.
Step loop


        for _ in range(MAX_STEPS):
            action = select_action(state, policy_net, epsilon, act_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


            reward = reward if not done else -10


            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

The epsilon-greedy approach is used to select an action at each step, and the environment returns the next state, reward, and done flag. A penalty of -10 is imposed if the program concludes. The reward is accrued, the current state is modified, and the experience is saved in the replay buffer.

Train policy network


            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)


                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.BoolTensor(dones).unsqueeze(1).to(device)


                q_values = policy_net(states).gather(1, actions)


                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q_values = rewards + GAMMA * next_q_values * (~dones)


                loss = nn.MSELoss()(q_values, expected_q_values)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

After being randomly selected from the replay buffer, a batch of experiences is transformed into PyTorch tensors. While the target Q-values are estimated using the target network, the Q-values for the activities that were taken are calculated. The network weights are updated by backpropagation after the loss is computed as the mean squared error between the target and forecast Q-values.

Target Network Update & Logging


        episode_rewards.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)


        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())


        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")


    env.close()

To ensure stability, the target network is updated with each TARGET_UPDATE_FREQ episode. Epsilon decays after each episode to progressively move from exploration to exploitation, and the average reward is printed every ten episodes to track progress.


Main Function


if _name_ == "_main_":
    train()

Guarantees that train() only executes when the script is invoked directly.


