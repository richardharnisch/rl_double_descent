from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class GridWorldConfig:
    grid_size: int = 8
    obstacle_prob: float = 0.2
    max_steps: int = 64
    max_gen_attempts: int = 200
    frame_stack: int = 2


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: GridWorldConfig, seed: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.grid_size * config.grid_size * 4 * config.frame_stack,),
            dtype=np.float32,
        )
        self._walls: Optional[np.ndarray] = None
        self._agent_pos: Tuple[int, int] = (0, 0)
        self._goal_pos: Tuple[int, int] = (config.grid_size - 1, config.grid_size - 1)
        self._steps = 0
        self._cell_size = 16
        self._obs_stack: deque[np.ndarray] = deque(maxlen=config.frame_stack)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._walls = self._generate_grid()
        elif self._walls is None:
            self._walls = self._generate_grid()
        self._agent_pos = (0, 0)
        self._steps = 0
        frame = self._get_obs()
        self._obs_stack.clear()
        for _ in range(self.config.frame_stack):
            self._obs_stack.append(frame)
        return self._get_stacked_obs(), {}

    def step(self, action: int):
        self._steps += 1
        row, col = self._agent_pos
        prev_dist = np.linalg.norm(np.array([row, col]) - np.array(self._goal_pos))
        if action == 0:
            candidate = (row - 1, col)
        elif action == 1:
            candidate = (row, col + 1)
        elif action == 2:
            candidate = (row + 1, col)
        else:
            candidate = (row, col - 1)

        if self._is_valid(candidate):
            self._agent_pos = candidate

        terminated = self._agent_pos == self._goal_pos
        truncated = self._steps >= self.config.max_steps
        new_row, new_col = self._agent_pos
        new_dist = np.linalg.norm(
            np.array([new_row, new_col]) - np.array(self._goal_pos)
        )
        bonus = (prev_dist - new_dist) / 100.0
        reward = 1.0 if terminated else (-0.01 + bonus)
        frame = self._get_obs()
        if self.config.frame_stack > 0:
            self._obs_stack.append(frame)
        return self._get_stacked_obs(), reward, terminated, truncated, {}

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        row, col = pos
        if row < 0 or row >= self.config.grid_size:
            return False
        if col < 0 or col >= self.config.grid_size:
            return False
        if self._walls is None:
            return True
        return not self._walls[row, col]

    def _get_obs(self) -> np.ndarray:
        grid = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.int64)
        if self._walls is not None:
            grid[self._walls] = 1
        goal_r, goal_c = self._goal_pos
        grid[goal_r, goal_c] = 3
        agent_r, agent_c = self._agent_pos
        grid[agent_r, agent_c] = 2
        one_hot = np.eye(4, dtype=np.float32)[grid.reshape(-1)]
        return one_hot.reshape(-1)

    def _get_stacked_obs(self) -> np.ndarray:
        if not self._obs_stack:
            frame = self._get_obs()
            return np.tile(frame, self.config.frame_stack)
        if len(self._obs_stack) < self.config.frame_stack:
            first = self._obs_stack[0]
            while len(self._obs_stack) < self.config.frame_stack:
                self._obs_stack.append(first)
        return np.concatenate(list(self._obs_stack), axis=0)

    def _generate_grid(self) -> np.ndarray:
        size = self.config.grid_size
        for _ in range(self.config.max_gen_attempts):
            walls = self.rng.random((size, size)) < self.config.obstacle_prob
            walls[0, 0] = False
            walls[size - 1, size - 1] = False
            if self._has_path(walls):
                return walls
        return np.zeros((size, size), dtype=bool)

    def render(self):
        size = self.config.grid_size
        image = np.ones((size, size, 3), dtype=np.uint8) * 255
        if self._walls is not None:
            image[self._walls] = np.array([30, 30, 30], dtype=np.uint8)
        goal_r, goal_c = self._goal_pos
        image[goal_r, goal_c] = np.array([50, 180, 50], dtype=np.uint8)
        agent_r, agent_c = self._agent_pos
        image[agent_r, agent_c] = np.array([50, 80, 220], dtype=np.uint8)
        scale = self._cell_size
        return np.kron(image, np.ones((scale, scale, 1), dtype=np.uint8))

    def _has_path(self, walls: np.ndarray) -> bool:
        size = self.config.grid_size
        start = (0, 0)
        goal = (size - 1, size - 1)
        queue = deque([start])
        visited = {start}
        while queue:
            row, col = queue.popleft()
            if (row, col) == goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= size or nc < 0 or nc >= size:
                    continue
                if walls[nr, nc]:
                    continue
                nxt = (nr, nc)
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return False
