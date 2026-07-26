from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ContinuousContextualBanditConfig:
    """Configuration for an online noisy-payoff contextual bandit.

    A context seed determines only the latent target. The payoff is sampled
    after the action from the environment's persistent transition RNG, so the
    learner receives online reward observations rather than a stored label
    table.
    """

    context_dim: int = 4
    action_dim: int = 5
    teacher_hidden: int = 2
    teacher_seed: int = 0
    reward_noise_std: float = 0.2
    reward_distance_scale: float = 1.0
    max_steps: int = 1


class ContinuousContextualBanditEnv(gym.Env):
    """One-step bandit with a continuous latent target and discrete actions."""

    metadata = {"render_modes": ["rgb_array"]}
    is_contextual_bandit = True

    def __init__(
        self,
        config: ContinuousContextualBanditConfig,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if config.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if config.action_dim < 2:
            raise ValueError("action_dim must be at least 2.")
        if config.teacher_hidden <= 0:
            raise ValueError("teacher_hidden must be positive.")
        if config.reward_noise_std < 0.0:
            raise ValueError("reward_noise_std must be non-negative.")
        if config.reward_distance_scale <= 0.0:
            raise ValueError("reward_distance_scale must be positive.")
        if config.max_steps != 1:
            raise ValueError("ContinuousContextualBanditEnv requires max_steps=1.")

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
        )
        self._teacher_second = teacher_rng.normal(
            0.0, 1.0, size=config.teacher_hidden
        )
        self._context = np.zeros(config.context_dim, dtype=np.float32)
        self._target = 0.0
        self._optimal_action = 0
        self._steps = 0

    def _context_from_seed(self, seed: int) -> np.ndarray:
        context_rng = np.random.default_rng(int(seed))
        context = context_rng.normal(0.0, 1.0, size=self.config.context_dim)
        context /= max(float(np.linalg.norm(context)), 1e-8)
        return context.astype(np.float32)

    def _target_from_context(self, context: np.ndarray) -> float:
        hidden = np.tanh(self._teacher_first @ context)
        normalized = (np.tanh(self._teacher_second @ hidden) + 1.0) / 2.0
        return float(normalized * (self.config.action_dim - 1))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        keep_context = bool(options and options.get("keep_context", False))
        if not keep_context:
            if seed is None:
                raise ValueError("A seed is required for a new bandit context.")
            self._context = self._context_from_seed(int(seed))
            self._target = self._target_from_context(self._context)
            self._optimal_action = int(
                np.clip(np.rint(self._target), 0, self.config.action_dim - 1)
            )
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
            reward += float(
                self._transition_rng.normal(0.0, self.config.reward_noise_std)
            )
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
        start = 8 + 12 * self._optimal_action
        frame[24:40, start : start + 12, 1] = 220
        return frame
