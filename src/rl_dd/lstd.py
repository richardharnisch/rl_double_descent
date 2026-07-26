from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OnlineLSTDQ:
    """Incremental random-feature LSTD-Q estimator.

    The sufficient statistics are updated after every environment transition;
    transitions are not retained as a frozen training dataset. For terminal
    transitions the Bellman target is just the observed reward.
    """

    input_dim: int
    action_dim: int
    feature_dim: int
    gamma: float = 0.0
    ridge: float = 1e-3
    seed: int = 0
    solve_interval: int = 1

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.action_dim < 2 or self.feature_dim <= 0:
            raise ValueError("LSTD dimensions must be positive and action_dim >= 2.")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative.")
        if self.solve_interval <= 0:
            raise ValueError("solve_interval must be positive.")
        rng = np.random.default_rng(self.seed)
        base_dim = self.input_dim + self.action_dim + 1
        # Generate one feature vector per row before transposing. This makes
        # widths nested for a fixed seed: a wider estimator contains exactly
        # the same random features as every narrower estimator.
        self._projection = rng.normal(
            0.0,
            1.0 / np.sqrt(base_dim),
            size=(self.feature_dim, base_dim),
        ).T
        self._a = np.eye(self.feature_dim, dtype=np.float64) * self.ridge
        self._b = np.zeros(self.feature_dim, dtype=np.float64)
        self._theta = np.zeros(self.feature_dim, dtype=np.float64)
        self._updates = 0

    def features(self, observation: np.ndarray, action: int) -> np.ndarray:
        if not 0 <= int(action) < self.action_dim:
            raise ValueError("action is outside the action space.")
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if obs.size != self.input_dim:
            raise ValueError("observation dimension does not match the estimator.")
        action_one_hot = np.zeros(self.action_dim, dtype=np.float64)
        action_one_hot[int(action)] = 1.0
        base = np.concatenate((obs, action_one_hot, np.ones(1)))
        return np.tanh(base @ self._projection)

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        return np.array(
            [self.features(observation, action) @ self._theta for action in range(self.action_dim)]
        )

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: Optional[np.ndarray],
        terminated: bool,
    ) -> None:
        phi = self.features(observation, action)
        if terminated or next_observation is None:
            next_phi = np.zeros_like(phi)
        else:
            next_q = self.q_values(next_observation)
            next_action = int(np.argmax(next_q))
            next_phi = self.features(next_observation, next_action)
        self._a += np.outer(phi, phi - self.gamma * next_phi)
        self._b += phi * float(reward)
        self._updates += 1
        if self._updates % self.solve_interval:
            return
        self.solve()

    def solve(self) -> None:
        system = self._a + self.ridge * np.eye(self.feature_dim)
        try:
            self._theta = np.linalg.solve(system, self._b)
        except np.linalg.LinAlgError:
            self._theta = np.linalg.lstsq(system, self._b, rcond=None)[0]

    @property
    def effective_parameter_count(self) -> int:
        return int(self.feature_dim)
