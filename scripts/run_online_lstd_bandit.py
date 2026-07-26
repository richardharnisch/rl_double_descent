from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from rl_dd.bandit import ContextualBanditConfig, ContextualBanditEnv
from rl_dd.continuous_bandit import (
    ContinuousContextualBanditConfig,
    ContinuousContextualBanditEnv,
)
from rl_dd.delayed_mdp import DelayedContextMDPConfig, DelayedContextMDPEnv
from rl_dd.lstd import OnlineLSTDQ


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            step = 1 if start <= end else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(part))
    return result


def parse_widths(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def mean_return_and_accuracy(
    env,
    agent: OnlineLSTDQ,
    seeds: Iterable[int],
    episodes_per_seed: int,
) -> tuple[float, float, float, float]:
    returns: list[float] = []
    correct: list[float] = []
    for seed in seeds:
        for episode_idx in range(episodes_per_seed):
            if episode_idx == 0 or getattr(env, "is_delayed_mdp", False):
                reset_seed = int(seed)
                options = None
            else:
                reset_seed = None
                options = {"keep_context": True}
            observation, _ = env.reset(seed=reset_seed, options=options)
            done = False
            total_return = 0.0
            while not done:
                q_values = agent.q_values(observation)
                action = int(np.argmax(q_values))
                observation, reward, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                done = terminated or truncated
                if "correct" in info:
                    correct.append(float(info["correct"]))
            returns.append(total_return)
    return (
        float(np.mean(returns)),
        float(np.std(returns, ddof=0)),
        float(np.mean(correct)),
        float(np.std(correct, ddof=0)),
    )


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    env,
    agent: OnlineLSTDQ,
    train_seeds: list[int],
    episodes: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        observation, _ = env.reset(seed=int(rng.choice(train_seeds)))
        done = False
        while not done:
            fraction = max(0.0, (epsilon_decay - episode) / max(1, epsilon_decay))
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * fraction
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.action_space.n))
            else:
                action = int(np.argmax(agent.q_values(observation)))
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(
                observation,
                action,
                reward,
                next_observation,
                terminated,
            )
            observation = next_observation
            done = terminated or truncated


def main() -> None:
    parser = argparse.ArgumentParser(description="Online random-feature LSTD-Q bandit sweep")
    parser.add_argument("--widths", default="2,4,8,16,32,64,128,256")
    parser.add_argument(
        "--task", choices=["bandit", "continuous_bandit", "delayed_mdp"], default="bandit"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--train-seeds", default="1-20")
    parser.add_argument("--test-seeds", default="21-220")
    parser.add_argument("--context-dim", type=int, default=4)
    parser.add_argument("--bandit-actions", type=int, default=4)
    parser.add_argument("--bandit-teacher-hidden", type=int, default=2)
    parser.add_argument("--bandit-teacher-seed", type=int, default=3)
    parser.add_argument("--reward-noise-std", type=float, default=0.5)
    parser.add_argument("--continuous-reward-distance-scale", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--solve-every", type=int, default=1)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()

    train_seeds = parse_int_list(args.train_seeds)
    test_seeds = parse_int_list(args.test_seeds)
    widths = parse_widths(args.widths)
    root = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    for width in widths:
        for run in range(args.runs):
            run_seed = args.base_seed + run
            if args.task == "delayed_mdp":
                env = DelayedContextMDPEnv(
                    DelayedContextMDPConfig(
                        context_dim=args.context_dim,
                        action_dim=args.bandit_actions,
                        teacher_hidden=args.bandit_teacher_hidden,
                        teacher_seed=args.bandit_teacher_seed,
                        reward_noise_std=args.reward_noise_std,
                    ),
                    seed=run_seed,
                )
            elif args.task == "continuous_bandit":
                env = ContinuousContextualBanditEnv(
                    ContinuousContextualBanditConfig(
                        context_dim=args.context_dim,
                        action_dim=args.bandit_actions,
                        teacher_hidden=args.bandit_teacher_hidden,
                        teacher_seed=args.bandit_teacher_seed,
                        reward_noise_std=args.reward_noise_std,
                        reward_distance_scale=args.continuous_reward_distance_scale,
                    ),
                    seed=run_seed,
                )
            else:
                env = ContextualBanditEnv(
                    ContextualBanditConfig(
                        context_dim=args.context_dim,
                        action_dim=args.bandit_actions,
                        teacher_hidden=args.bandit_teacher_hidden,
                        teacher_seed=args.bandit_teacher_seed,
                        reward_noise_std=args.reward_noise_std,
                    ),
                    seed=run_seed,
                )
            agent = OnlineLSTDQ(
                input_dim=int(env.observation_space.shape[0]),
                action_dim=args.bandit_actions,
                feature_dim=width,
                gamma=args.gamma,
                ridge=args.ridge,
                seed=run_seed,
                solve_interval=args.solve_every,
            )
            train_one(
                env,
                agent,
                train_seeds,
                args.episodes,
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay,
                run_seed,
            )
            agent.solve()
            train_return, train_std, train_accuracy, train_accuracy_std = mean_return_and_accuracy(
                env, agent, train_seeds, args.eval_episodes
            )
            test_return, test_std, test_accuracy, test_accuracy_std = mean_return_and_accuracy(
                env, agent, test_seeds, args.eval_episodes
            )
            row = {
                "width": float(width),
                "depth": 1.0,
                "run": float(run),
                "num_params": float(agent.effective_parameter_count),
                "train_return": train_return,
                "test_return": test_return,
                "train_return_std": train_std,
                "test_return_std": test_std,
                "train_optimal_action_rate": train_accuracy,
                "test_optimal_action_rate": test_accuracy,
                "train_optimal_action_std": train_accuracy_std,
                "test_optimal_action_std": test_accuracy_std,
                "train_entropy": float("nan"),
                "test_entropy": float("nan"),
                "train_coverage": 1.0,
                "test_coverage": 1.0,
                "fim_trace": float("nan"),
            }
            run_dir = root / f"w{width}_d1_run{run}"
            write_rows(run_dir / "metrics.csv", [row])
            rows.append(row)
            print(
                f"width={width} run={run} train={train_accuracy:.3f} "
                f"test={test_accuracy:.3f}"
            )

    write_rows(root / "metrics.csv", rows)


if __name__ == "__main__":
    main()
