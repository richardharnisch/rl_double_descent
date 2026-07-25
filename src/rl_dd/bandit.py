from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ContextualBanditConfig:
    """Configuration for a live, finite-context contextual bandit.

    A seed identifies a context, but rewards are sampled from a persistent
    transition RNG on every episode. The context identity is therefore fixed
    for train/test evaluation while reward observations remain online samples;
    no observation/reward table is stored or replayed by this environment.
    """

    context_dim: int = 16
    action_dim: int = 4
    teacher_hidden: int = 12
    teacher_seed: int = 17
    reward_noise_std: float = 0.1
    max_steps: int = 1


class ContextualBanditEnv(gym.Env):
    """One-step contextual bandit with a nonlinear, deterministic task signal."""

    metadata = {"render_modes": ["rgb_array"]}
    is_contextual_bandit = True

    def __init__(self, config: ContextualBanditConfig, seed: Optional[int] = None):
        super().__init__()
        if config.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if config.action_dim < 2:
            raise ValueError("action_dim must be at least 2.")
        if config.teacher_hidden <= 0:
            raise ValueError("teacher_hidden must be positive.")
        if config.reward_noise_std < 0.0:
            raise ValueError("reward_noise_std must be non-negative.")
        if config.max_steps != 1:
            raise ValueError("ContextualBanditEnv requires max_steps=1.")

        self.config = config
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(config.context_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(config.action_dim)
        self._transition_rng = np.random.default_rng(seed)
        teacher_rng = np.random.default_rng(config.teacher_seed)
        self._teacher_first = teacher_rng.normal(
            0.0, 1.0, size=(config.teacher_hidden, config.context_dim)
        ).astype(np.float32)
        self._teacher_second = teacher_rng.normal(
            0.0, 1.0, size=(config.action_dim, config.teacher_hidden)
        ).astype(np.float32)
        self._context = np.zeros(config.context_dim, dtype=np.float32)
        self._optimal_action = 0
        self._steps = 0

    def _context_from_seed(self, seed: int) -> np.ndarray:
        context_rng = np.random.default_rng(int(seed))
        context = context_rng.normal(0.0, 1.0, size=self.config.context_dim)
        context /= max(float(np.linalg.norm(context)), 1e-8)
        return context.astype(np.float32)

    def _teacher_action(self, context: np.ndarray) -> int:
        hidden = np.tanh(self._teacher_first @ context)
        logits = self._teacher_second @ hidden
        return int(np.argmax(logits))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        keep_context = bool(options and options.get("keep_context", False))
        if not keep_context:
            if seed is None:
                raise ValueError("A seed is required for a new bandit context.")
            self._context = self._context_from_seed(int(seed))
            self._optimal_action = self._teacher_action(self._context)
        self._steps = 0
        return self._context.copy(), {"optimal_action": self._optimal_action}

    def step(self, action: int):
        if self._steps >= self.config.max_steps:
            raise RuntimeError("step() called after the bandit episode ended.")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self._steps += 1
        reward = 1.0 if action == self._optimal_action else -1.0
        if self.config.reward_noise_std > 0.0:
            reward += float(
                self._transition_rng.normal(0.0, self.config.reward_noise_std)
            )
        return (
            self._context.copy(),
            reward,
            True,
            False,
            {
                "optimal_action": self._optimal_action,
                "correct": float(action == self._optimal_action),
            },
        )

    def render(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        intensity = int(64 + 160 * (self._optimal_action / max(1, self.action_space.n - 1)))
        frame[:, :, 0] = intensity
        frame[24:40, 8 + 12 * self._optimal_action : 20 + 12 * self._optimal_action, 1] = 220
        return frame
