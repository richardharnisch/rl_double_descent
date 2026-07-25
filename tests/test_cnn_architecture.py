from __future__ import annotations

import unittest

import torch

from rl_dd.env import GridWorldConfig, GridWorldEnv
from rl_dd.train import build_network, count_parameters
from rl_dd.trpo import build_policy, build_value


class CNNArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.env = GridWorldEnv(
            GridWorldConfig(grid_size=16, frame_stack=2, max_steps=256),
            seed=0,
        )
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.action_dim = int(self.env.action_space.n)

    def test_cnn_q_network_accepts_flattened_16x16_stacked_observation(self) -> None:
        obs, _ = self.env.reset(seed=0)
        model = build_network(
            self.obs_dim,
            self.action_dim,
            [8, 8],
            self.device,
            arch="cnn",
            grid_size=16,
            frame_stack=2,
        )

        with torch.no_grad():
            q_values = model(torch.from_numpy(obs).float().unsqueeze(0))

        self.assertEqual(tuple(q_values.shape), (1, self.action_dim))

    def test_cnn_policy_and_value_accept_flattened_16x16_stacked_observation(
        self,
    ) -> None:
        obs, _ = self.env.reset(seed=0)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        policy = build_policy(
            self.obs_dim,
            self.action_dim,
            [8, 8],
            self.device,
            arch="cnn",
            grid_size=16,
            frame_stack=2,
        )
        value = build_value(
            self.obs_dim,
            [8, 8],
            self.device,
            arch="cnn",
            grid_size=16,
            frame_stack=2,
        )

        with torch.no_grad():
            logits = policy(obs_t)
            value_pred = value(obs_t)

        self.assertEqual(tuple(logits.shape), (1, self.action_dim))
        self.assertEqual(tuple(value_pred.shape), (1,))

    def test_cnn_parameter_count_increases_with_width(self) -> None:
        small = build_policy(
            self.obs_dim,
            self.action_dim,
            [4, 4, 4],
            self.device,
            arch="cnn",
            grid_size=16,
            frame_stack=2,
        )
        large = build_policy(
            self.obs_dim,
            self.action_dim,
            [8, 8, 8],
            self.device,
            arch="cnn",
            grid_size=16,
            frame_stack=2,
        )

        self.assertGreater(count_parameters(large), count_parameters(small))

    def test_sticky_action_is_reproducible_and_reset_clears_history(self) -> None:
        config = GridWorldConfig(
            grid_size=4,
            obstacle_prob=0.0,
            max_steps=8,
            frame_stack=1,
            start_corner=0,
            goal_corner=2,
            sticky_action_prob=1.0,
        )
        env = GridWorldEnv(config, seed=7)
        env.reset(seed=7)
        env.step(1)
        _, _, _, _, _ = env.step(2)
        self.assertEqual(env._agent_pos, (0, 2))

        env.reset(seed=7)
        _, _, _, _, _ = env.step(2)
        self.assertEqual(env._agent_pos, (1, 0))

    def test_keep_map_reset_preserves_random_map(self) -> None:
        config = GridWorldConfig(grid_size=5, frame_stack=1)
        env = GridWorldEnv(config, seed=3)
        env.reset(seed=3)
        walls = env._walls.copy()
        start = env._start_pos
        goal = env._goal_pos
        env.reset(options={"keep_map": True})
        self.assertTrue((env._walls == walls).all())
        self.assertEqual(env._start_pos, start)
        self.assertEqual(env._goal_pos, goal)

    def test_reward_noise_is_seeded(self) -> None:
        config = GridWorldConfig(
            grid_size=4,
            obstacle_prob=0.0,
            max_steps=4,
            frame_stack=1,
            start_corner=0,
            goal_corner=2,
            reward_noise_std=0.1,
        )
        first = GridWorldEnv(config, seed=11)
        second = GridWorldEnv(config, seed=11)
        first.reset(seed=11)
        second.reset(seed=11)
        first_rewards = [first.step(1)[1] for _ in range(3)]
        second_rewards = [second.step(1)[1] for _ in range(3)]
        self.assertEqual(first_rewards, second_rewards)


if __name__ == "__main__":
    unittest.main()
