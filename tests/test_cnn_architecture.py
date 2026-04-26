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


if __name__ == "__main__":
    unittest.main()
