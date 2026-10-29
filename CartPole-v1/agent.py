import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN


class DQNAgent:

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.loss_fn = nn.MSELoss()

    def act(self, state):
        """
        Epsilon-greedy action selection
        """

        if torch.random() < self.epsilon:
            return torch.randint(0, self.action_dim - 1).item()

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()

    def learn(self, states, actions, rewards, next_states, dones):
        """
        One DQN training step
        """

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q(s,a)
        current_q = self.policy_net(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # r + gamma * max Q'(s')
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q = next_q.max(dim=1)[0]

            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
