from __future__ import annotations

import unittest

import numpy as np
import torch

from rl_dd.bandit import ContextualBanditConfig, ContextualBanditEnv
from rl_dd.continuous_bandit import (
    ContinuousContextualBanditConfig,
    ContinuousContextualBanditEnv,
)
from rl_dd.train import build_network
from rl_dd.trpo import build_policy, build_value
from rl_dd.train import count_parameters
from rl_dd.lstd import OnlineLSTDQ
from rl_dd.delayed_mdp import DelayedContextMDPConfig, DelayedContextMDPEnv


class ContextualBanditTests(unittest.TestCase):
    def test_seed_selects_context_and_keep_context_preserves_it(self) -> None:
        env = ContextualBanditEnv(
            ContextualBanditConfig(context_dim=8, action_dim=3, reward_noise_std=0.0),
            seed=0,
        )
        first, first_info = env.reset(seed=11)
        env.step(first_info["optimal_action"])
        repeated, repeated_info = env.reset(options={"keep_context": True})
        other, other_info = env.reset(seed=12)

        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(first_info["optimal_action"], repeated_info["optimal_action"])
        self.assertFalse(np.array_equal(first, other))
        self.assertIn("optimal_action", other_info)

    def test_reward_noise_is_sampled_online_between_same_context_episodes(self) -> None:
        env = ContextualBanditEnv(
            ContextualBanditConfig(context_dim=8, action_dim=3, reward_noise_std=1.0),
            seed=0,
        )
        _, info = env.reset(seed=11)
        reward_one = env.step(info["optimal_action"])[1]
        env.reset(options={"keep_context": True})
        reward_two = env.step(info["optimal_action"])[1]

        self.assertNotEqual(reward_one, reward_two)

    def test_episode_is_one_step_and_reports_correct_action(self) -> None:
        env = ContextualBanditEnv(
            ContextualBanditConfig(context_dim=8, action_dim=3, reward_noise_std=0.0),
            seed=0,
        )
        _, info = env.reset(seed=11)
        _, reward, terminated, truncated, step_info = env.step(info["optimal_action"])

        self.assertEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["correct"], 1.0)

    def test_continuous_bandit_reward_is_live_and_bounded_without_noise(self) -> None:
        env = ContinuousContextualBanditEnv(
            ContinuousContextualBanditConfig(
                context_dim=4, action_dim=5, reward_noise_std=0.0
            ),
            seed=0,
        )
        _, info = env.reset(seed=11)
        reward = env.step(info["optimal_action"])[1]
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
        env.reset(options={"keep_context": True})
        noisy_env = ContinuousContextualBanditEnv(
            ContinuousContextualBanditConfig(
                context_dim=4, action_dim=5, reward_noise_std=1.0
            ),
            seed=0,
        )
        _, noisy_info = noisy_env.reset(seed=11)
        reward_one = noisy_env.step(noisy_info["optimal_action"])[1]
        noisy_env.reset(options={"keep_context": True})
        reward_two = noisy_env.step(noisy_info["optimal_action"])[1]
        self.assertNotEqual(reward_one, reward_two)

    def test_random_feature_policy_has_frozen_map_and_trainable_head(self) -> None:
        policy = build_policy(4, 4, [16], torch.device("cpu"), arch="random_features")
        value = build_value(4, [16], torch.device("cpu"), arch="random_features")

        self.assertFalse(any(parameter.requires_grad for parameter in policy.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in policy.head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in value.features.parameters()))
        self.assertGreater(count_parameters(policy), count_parameters(build_policy(4, 4, [8], torch.device("cpu"), arch="random_features")))

    def test_random_feature_q_network_has_frozen_map(self) -> None:
        q_network = build_network(
            4, 4, [16], torch.device("cpu"), arch="random_features"
        )

        self.assertFalse(
            any(parameter.requires_grad for parameter in q_network.features.parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in q_network.head.parameters())
        )

    def test_lstd_updates_sufficient_statistics_online(self) -> None:
        estimator = OnlineLSTDQ(4, 3, 8, ridge=1e-2, seed=0)
        observation = np.ones(4, dtype=np.float32)
        before = estimator.q_values(observation).copy()
        estimator.update(observation, 1, 1.0, None, True)
        after = estimator.q_values(observation)

        self.assertEqual(after.shape, (3,))
        self.assertFalse(np.array_equal(before, after))

    def test_lstd_random_features_are_nested_across_widths(self) -> None:
        narrow = OnlineLSTDQ(4, 3, 8, ridge=1e-2, seed=7)
        wide = OnlineLSTDQ(4, 3, 16, ridge=1e-2, seed=7)
        np.testing.assert_array_equal(narrow._projection, wide._projection[:, :8])

    def test_delayed_mdp_has_bootstrap_transition(self) -> None:
        env = DelayedContextMDPEnv(
            DelayedContextMDPConfig(context_dim=4, action_dim=3, reward_noise_std=0.0),
            seed=0,
        )
        observation, _ = env.reset(seed=11)
        next_observation, first_reward, first_terminated, _, _ = env.step(0)
        _, second_reward, second_terminated, truncated, _ = env.step(1)

        self.assertEqual(observation.shape, (8,))
        self.assertEqual(next_observation.shape, (8,))
        self.assertEqual(first_reward, 0.0)
        self.assertFalse(first_terminated)
        self.assertTrue(second_terminated)
        self.assertFalse(truncated)
        self.assertGreaterEqual(second_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
