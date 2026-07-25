from __future__ import annotations

import unittest

import numpy as np

from rl_dd.bandit import ContextualBanditConfig, ContextualBanditEnv


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


if __name__ == "__main__":
    unittest.main()
