import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from collections import deque
from model import DQN


class DQNAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        replay_buffer_size: int = 1000,
    ):
        self.gamma = gamma
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def act(self, state, epsilon):
        """
        Epsilon-greedy action selection
        """

        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()

    def learn(self, states, actions, rewards, next_states, dones):
        """
        One DQN training step
        """

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Q(s,a)
        current_q = self.policy_net(states)
        current_q = current_q.gather(1, actions).squeeze(1)

        # r + gamma * max Q'(s')
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q = next_q.max(1)[0]

            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class ReplayBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

    def max_size(self):
        return self.size
