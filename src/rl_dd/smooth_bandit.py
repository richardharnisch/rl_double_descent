from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class SmoothContextualBanditConfig:
    """Configuration for a smooth live contextual-bandit teacher."""

    context_dim: int = 1
    action_dim: int = 2
    teacher_seed: int = 0
    reward_noise_std: float = 0.2
    reward_distance_scale: float = 0.5
    frequency: float = 2.0
    max_steps: int = 1


class SmoothContextualBanditEnv(gym.Env):
    """One-step bandit with a smooth latent target and online payoffs.

    The context is sampled at reset and the payoff is sampled after the
    selected action. The environment never stores a training observation or
    reward table; repeated contexts are only used during explicit evaluation.
    """

    metadata = {"render_modes": ["rgb_array"]}
    is_contextual_bandit = True

    def __init__(
        self,
        config: SmoothContextualBanditConfig,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if config.context_dim != 1 or config.action_dim != 2:
            raise ValueError("SmoothContextualBanditEnv requires context_dim=1 and action_dim=2.")
        if config.reward_noise_std < 0.0:
            raise ValueError("reward_noise_std must be non-negative.")
        if config.reward_distance_scale <= 0.0:
            raise ValueError("reward_distance_scale must be positive.")
        if config.frequency <= 0.0:
            raise ValueError("frequency must be positive.")
        if config.max_steps != 1:
            raise ValueError("SmoothContextualBanditEnv requires max_steps=1.")

        self.config = config
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self._transition_rng = np.random.default_rng(seed)
        teacher_rng = np.random.default_rng(config.teacher_seed)
        self._phase = float(teacher_rng.uniform(0.0, 2.0 * np.pi))
        self._context = np.zeros(1, dtype=np.float32)
        self._target = 0.5
        self._optimal_action = 0
        self._steps = 0

    def _context_from_seed(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        return rng.uniform(-1.0, 1.0, size=1).astype(np.float32)

    def _target_from_context(self, context: np.ndarray) -> float:
        x = float(context[0])
        oscillation = np.sin(self.config.frequency * np.pi * x + self._phase)
        return float(np.clip(0.5 + 0.49 * oscillation, 0.0, 1.0))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        keep_context = bool(options and options.get("keep_context", False))
        if not keep_context:
            if seed is None:
                raise ValueError("A seed is required for a new bandit context.")
            self._context = self._context_from_seed(int(seed))
            self._target = self._target_from_context(self._context)
            self._optimal_action = int(np.rint(self._target))
        self._steps = 0
        return self._context.copy(), {
            "optimal_action": self._optimal_action,
            "target": self._target,
        }

    def step(self, action: int):
        if self._steps >= self.config.max_steps:
            raise RuntimeError("step() called after the bandit episode ended.")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self._steps += 1
        distance = (action - self._target) / self.config.reward_distance_scale
        reward = 1.0 / (1.0 + distance**2)
        if self.config.reward_noise_std > 0.0:
            reward += float(self._transition_rng.normal(0.0, self.config.reward_noise_std))
        return (
            self._context.copy(),
            float(reward),
            True,
            False,
            {
                "optimal_action": self._optimal_action,
                "target": self._target,
                "correct": float(action == self._optimal_action),
            },
        )

    def render(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = 120
        frame[24:40, 20 + 24 * self._optimal_action : 44 + 24 * self._optimal_action, 1] = 220
        return frame
