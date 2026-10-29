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

    def act(self, state, epsilon=0.0):
        """Selects one action, either randomly or based on the highest value found from a forward pass with the state as input"""

        # Randomly select action if a random value is < epsilon
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.network[-1].out_features, (1,)).item()

        # Return the highest value action after finding values from one forward pass with the state
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.network(state)
            return q_values.argmax(dim=1).item()
