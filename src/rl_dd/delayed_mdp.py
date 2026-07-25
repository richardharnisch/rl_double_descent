from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class DelayedContextMDPConfig:
    context_dim: int = 4
    action_dim: int = 4
    teacher_hidden: int = 2
    teacher_seed: int = 3
    reward_noise_std: float = 0.1
    max_steps: int = 2


class DelayedContextMDPEnv(gym.Env):
    """Two-step online MDP with action-dependent delayed rewards."""

    metadata = {"render_modes": ["rgb_array"]}
    is_delayed_mdp = True

    def __init__(self, config: DelayedContextMDPConfig, seed: Optional[int] = None):
        super().__init__()
        if config.context_dim <= 0 or config.action_dim < 2:
            raise ValueError("Invalid delayed-MDP dimensions.")
        if config.max_steps != 2:
            raise ValueError("DelayedContextMDPEnv requires max_steps=2.")
        if config.reward_noise_std < 0.0:
            raise ValueError("reward_noise_std must be non-negative.")
        self.config = config
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(config.context_dim + config.action_dim + 1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(config.action_dim)
        self._transition_rng = np.random.default_rng(seed)
        teacher_rng = np.random.default_rng(config.teacher_seed)
        self._teacher_first = teacher_rng.normal(
            0.0, 1.0, size=(config.teacher_hidden, config.context_dim)
        )
        self._teacher_second = teacher_rng.normal(
            0.0, 1.0, size=(config.action_dim, config.teacher_hidden)
        )
        self._context = np.zeros(config.context_dim, dtype=np.float32)
        self._first_action = 0
        self._target_action = 0
        self._branch_values = np.zeros(config.action_dim, dtype=np.float32)
        self._steps = 0

    def _context_from_seed(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        context = rng.normal(0.0, 1.0, size=self.config.context_dim)
        context /= max(float(np.linalg.norm(context)), 1e-8)
        return context.astype(np.float32)

    def _teacher_logits(self, context: np.ndarray) -> np.ndarray:
        hidden = np.tanh(self._teacher_first @ context)
        return self._teacher_second @ hidden

    def _observation(self) -> np.ndarray:
        first_action = np.zeros(self.config.action_dim, dtype=np.float32)
        if self._steps:
            first_action[self._first_action] = 1.0
        return np.concatenate(
            (self._context, first_action, np.array([float(self._steps)], dtype=np.float32))
        ).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is None:
            raise ValueError("A seed is required for a new delayed-MDP context.")
        self._context = self._context_from_seed(int(seed))
        logits = self._teacher_logits(self._context)
        self._target_action = int(np.argmax(logits))
        scaled = logits - float(np.max(logits))
        probabilities = np.exp(scaled) / np.exp(scaled).sum()
        self._branch_values = np.clip(
            0.05 + 0.95 * self.config.action_dim * probabilities,
            0.05,
            1.0,
        ).astype(np.float32)
        self._first_action = 0
        self._steps = 0
        return self._observation(), {"optimal_action": int(np.argmax(self._branch_values))}

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self._steps >= self.config.max_steps:
            raise RuntimeError("step() called after the delayed-MDP episode ended.")
        if self._steps == 0:
            self._first_action = action
            self._steps = 1
            return (
                self._observation(),
                0.0,
                False,
                False,
                {"optimal_action": int(np.argmax(self._branch_values))},
            )

        self._steps = 2
        correct_second = action == self._target_action
        reward = float(self._branch_values[self._first_action]) if correct_second else 0.0
        if self.config.reward_noise_std > 0.0:
            reward += float(self._transition_rng.normal(0.0, self.config.reward_noise_std))
        return (
            self._observation(),
            reward,
            True,
            False,
            {
                "correct": float(correct_second),
                "optimal_action": self._target_action,
            },
        )

    def render(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = 120
        frame[24:40, 8 + 12 * self._first_action : 20 + 12 * self._first_action, 1] = 220
        return frame
