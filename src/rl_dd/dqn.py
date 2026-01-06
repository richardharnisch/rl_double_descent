from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        idx = self.rng.integers(0, self.size, size=batch_size)
        obs = torch.from_numpy(self.obs[idx]).to(device)
        actions = torch.from_numpy(self.actions[idx]).to(device)
        rewards = torch.from_numpy(self.rewards[idx]).to(device)
        next_obs = torch.from_numpy(self.next_obs[idx]).to(device)
        dones = torch.from_numpy(self.dones[idx]).to(device)
        return ReplayBatch(obs, actions, rewards, next_obs, dones)
