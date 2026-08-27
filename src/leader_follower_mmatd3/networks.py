from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    """Decentralized actor: 64 -> 650 -> 500 -> 2 with tanh output."""

    def __init__(self, state_dim: int = 64, action_dim: int = 2, hidden_sizes=(650, 500)) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class QNetwork(nn.Module):
    """One centralized Q-function with separate state/action branches before fusion."""

    def __init__(self, joint_state_dim: int = 128, joint_action_dim: int = 4, hidden_sizes=(650, 500)) -> None:
        super().__init__()
        h1, h2 = hidden_sizes
        self.state_fc1 = nn.Linear(joint_state_dim, h1)
        self.state_fc2 = nn.Linear(h1, h2)
        self.action_fc = nn.Linear(joint_action_dim, h2, bias=False)
        self.output = nn.Linear(h2, 1)
        self.activation = nn.ReLU()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_features = self.activation(self.state_fc1(state))
        fused = self.state_fc2(state_features) + self.action_fc(action)
        return self.output(self.activation(fused))


class TwinCritic(nn.Module):
    """Two centralized Q-functions used by MATD3 and M-MATD3."""

    def __init__(self, joint_state_dim: int = 128, joint_action_dim: int = 4, hidden_sizes=(650, 500)) -> None:
        super().__init__()
        self.q1 = QNetwork(joint_state_dim, joint_action_dim, hidden_sizes)
        self.q2 = QNetwork(joint_state_dim, joint_action_dim, hidden_sizes)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.q1(state, action), self.q2(state, action)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(state, action)


class SingleCritic(nn.Module):
    """One centralized Q-function for the MADDPG baseline."""

    def __init__(self, joint_state_dim: int = 128, joint_action_dim: int = 4, hidden_sizes=(650, 500)) -> None:
        super().__init__()
        self.q = QNetwork(joint_state_dim, joint_action_dim, hidden_sizes)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(state, action)
