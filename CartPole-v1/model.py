import torch
import torch.nn as nn


class DQN(nn.Module):
    """Generates a basic DQN network with two hidden layers"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        """Completes one forward pass with inputs x"""
        return self.network(x)
