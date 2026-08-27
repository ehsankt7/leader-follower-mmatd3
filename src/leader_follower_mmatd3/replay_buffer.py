from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_states: torch.Tensor


class ReplayBuffer:
    """Preallocated replay buffer for fixed-size continuous-control transitions."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, n_agents), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.size = 0
        self.position = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, reward, done, next_state) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.next_states[i] = next_state
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self.size < batch_size:
            raise ValueError(f"Replay buffer has {self.size} items; {batch_size} requested.")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            states=torch.as_tensor(self.states[idx], device=device),
            actions=torch.as_tensor(self.actions[idx], device=device),
            rewards=torch.as_tensor(self.rewards[idx], device=device),
            dones=torch.as_tensor(self.dones[idx], device=device),
            next_states=torch.as_tensor(self.next_states[idx], device=device),
        )
