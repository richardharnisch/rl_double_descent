from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from rl_dd.env import GridWorldConfig, GridWorldEnv
from rl_dd.dqn import ReplayBuffer
from rl_dd.train import (
    TrainConfig,
    build_network,
    count_parameters,
    evaluate_policy,
    estimate_fim_trace,
    make_optimizer,
    set_global_seeds,
    train_dqn,
)
from rl_dd.trpo import (
    TRPOConfig,
    build_policy,
    build_value,
    evaluate_policy as evaluate_policy_trpo,
    train_trpo,
)


def save_gif(frames: List[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    duration = max(1, int(1000 / max(1, fps)))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


def rollout_episode(
    env, q_net, seed: int, device: torch.device, max_steps: int
) -> Tuple[List[np.ndarray], float]:
    was_training = q_net.training
    q_net.eval()
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_return = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        with torch.no_grad():
            q_vals = q_net(obs_t)
        action = int(torch.argmax(q_vals, dim=1).item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        frames.append(env.render())
        done = terminated or truncated
        steps += 1
    if was_training:
        q_net.train()
    return frames, total_return


def parse_int_list(value: str) -> List[int]:
    items: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            step = 1 if start <= end else -1
            items.extend(list(range(start, end + step, step)))
        else:
            items.append(int(part))
    return items


def parse_widths(value: str) -> List[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def parse_depths(value: str) -> List[int]:
    return parse_int_list(value)


def aggregate_results(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[int, Dict[str, List[float]]] = {}
    for row in rows:
        key = int(row["num_params"])
        grouped.setdefault(
            key,
            {"train_return": [], "test_return": [], "fim_trace": []},
        )
        grouped[key]["train_return"].append(float(row["train_return"]))
        grouped[key]["test_return"].append(float(row["test_return"]))
        if "fim_trace" in row and not np.isnan(float(row["fim_trace"])):
            grouped[key]["fim_trace"].append(float(row["fim_trace"]))

    summary = []
    for num_params, metrics in grouped.items():
        train_vals = np.array(metrics["train_return"])
        test_vals = np.array(metrics["test_return"])
        fim_vals = np.array(metrics["fim_trace"], dtype=np.float64)
        summary_row = {
            "num_params": float(num_params),
            "train_mean": float(train_vals.mean()),
            "train_std": float(train_vals.std(ddof=0)),
            "test_mean": float(test_vals.mean()),
            "test_std": float(test_vals.std(ddof=0)),
        }
        if fim_vals.size:
            summary_row["fim_mean"] = float(fim_vals.mean())
            summary_row["fim_std"] = float(fim_vals.std(ddof=0))
        summary.append(summary_row)
    summary.sort(key=lambda x: x["num_params"])
    return summary


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(summary: List[Dict[str, float]], path: Path, log_x: bool) -> None:
    import matplotlib.pyplot as plt

    if not summary:
        return

    params = [row["num_params"] for row in summary]
    train_mean = [row["train_mean"] for row in summary]
    train_std = [row["train_std"] for row in summary]
    test_mean = [row["test_mean"] for row in summary]
    test_std = [row["test_std"] for row in summary]

    has_fim = any("fim_mean" in row for row in summary)
    if has_fim:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax_perf, ax_fim = axes
    else:
        fig, ax_perf = plt.subplots(figsize=(8, 5))
        ax_fim = None

    ax_perf.errorbar(params, train_mean, yerr=train_std, label="train", marker="o")
    ax_perf.errorbar(params, test_mean, yerr=test_std, label="test", marker="o")
    if log_x:
        ax_perf.set_xscale("log")
    ax_perf.set_ylabel("Average return")
    ax_perf.set_title("Performance vs parameter count")
    ax_perf.legend()

    if ax_fim is not None:
        fim_params = []
        fim_per_param = []
        fim_std = []
        for row in summary:
            if "fim_mean" not in row:
                continue
            num_params = float(row["num_params"])
            fim_params.append(num_params)
            fim_per_param.append(float(row["fim_mean"]) / num_params)
            fim_std.append(float(row.get("fim_std", 0.0)) / num_params)
        ax_fim.errorbar(
            fim_params, fim_per_param, yerr=fim_std, label="fim/param", marker="o"
        )
        if log_x:
            ax_fim.set_xscale("log")
        ax_fim.set_xlabel("Number of parameters")
        ax_fim.set_ylabel("FIM trace per parameter")
        ax_fim.set_title("FIM trace per parameter vs parameter count")
        ax_fim.legend()
    else:
        ax_perf.set_xlabel("Number of parameters")

    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def collect_results(log_root: Path, log_x: bool) -> None:
    metrics_files = sorted(log_root.glob("w*_d*_run*/metrics.csv"))
    print(f"Found {len(metrics_files)} metrics files.")
    results: List[Dict[str, float]] = []
    for metrics_path in tqdm(metrics_files):
        with metrics_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed = {key: float(value) for key, value in row.items()}
                results.append(parsed)
    if not results:
        return
    save_csv(log_root / "metrics.csv", results)
    summary = aggregate_results(results)
    save_csv(log_root / "summary.csv", summary)
    plot_results(summary, log_root / "curve.png", log_x=log_x)


def save_episode_metrics(
    returns: List[float],
    lengths: List[int],
    path: Path,
    run_meta: Dict[str, float],
) -> None:
    rows: List[Dict[str, object]] = []
    for idx, (ret, length) in enumerate(zip(returns, lengths)):
        row = {
            "episode": idx,
            "return": float(ret),
            "length": int(length),
        }
        row.update(run_meta)
        rows.append(row)
    save_csv(path, rows)


def plot_episode_metrics(
    returns: List[float],
    lengths: List[int],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not returns:
        return

    total_episodes = len(returns)
    num_points = min(200, total_episodes)
    return_chunks = np.array_split(np.array(returns, dtype=np.float32), num_points)
    length_chunks = np.array_split(np.array(lengths, dtype=np.float32), num_points)
    avg_returns = np.array([chunk.mean() for chunk in return_chunks], dtype=np.float32)
    avg_lengths = np.array([chunk.mean() for chunk in length_chunks], dtype=np.float32)
    chunk_sizes = [len(chunk) for chunk in return_chunks]
    episodes = []
    start_idx = 1
    for size in chunk_sizes:
        mid = start_idx + (size - 1) / 2.0
        episodes.append(mid)
        start_idx += size
    avg_per_point = total_episodes / max(1, num_points)
    if abs(avg_per_point - round(avg_per_point)) < 1e-6:
        avg_label = f"{int(round(avg_per_point))}"
    else:
        avg_label = f"{avg_per_point:.1f}"
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_ret, ax_len = axes
    ax_ret.plot(episodes, avg_returns, label=f"return (avg over ~{avg_label} eps)")
    ax_ret.set_ylabel("Return")
    ax_ret.set_title("Training return per episode")
    ax_ret.legend()

    ax_len.plot(
        episodes,
        avg_lengths,
        label=f"length (avg over ~{avg_label} eps)",
        color="tab:orange",
    )
    ax_len.set_xlabel("Episode")
    ax_len.set_ylabel("Length")
    ax_len.set_title("Training length per episode")
    ax_len.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def save_model_state(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def build_hidden_sizes(width: int, depth: int) -> List[int]:
    return [width for _ in range(depth)]


def run_sanity_check(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    train_seeds = [0]
    depth_list = parse_depths(args.depths)
    env_config = GridWorldConfig(
        grid_size=5,
        obstacle_prob=0.0,
        max_steps=32,
    )
    train_config = TrainConfig(
        episodes=args.sanity_episodes,
        max_steps=32,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        buffer_size=50_000,
        target_update_interval=500,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=30_000,
    )

    set_global_seeds(0)
    rng = np.random.default_rng(0)
    env = GridWorldEnv(env_config, seed=0)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_sizes = build_hidden_sizes(64, depth_list[0])
    if args.algo == "trpo":
        trpo_config = TRPOConfig(
            episodes=args.sanity_episodes,
            batch_episodes=args.trpo_batch_episodes,
            max_steps=32,
            gamma=0.99,
            gae_lambda=args.trpo_gae_lambda,
            max_kl=args.trpo_max_kl,
            cg_iters=args.trpo_cg_iters,
            cg_damping=args.trpo_cg_damping,
            backtrack_coeff=args.trpo_backtrack_coeff,
            backtrack_iters=args.trpo_backtrack_iters,
            vf_iters=args.trpo_vf_iters,
            vf_lr=args.trpo_vf_lr,
            early_stop_return=args.early_stop_return,
            early_stop_episodes=args.early_stop_episodes,
        )
        policy_net = build_policy(obs_dim, action_dim, hidden_sizes, device)
        value_net = build_value(obs_dim, hidden_sizes, device)
        train_trpo(
            env,
            policy_net,
            value_net,
            train_seeds,
            trpo_config,
            device,
            rng,
        )
        train_return = evaluate_policy_trpo(
            env,
            policy_net,
            train_seeds,
            episodes_per_seed=5,
            device=device,
        )
    else:
        q_net = build_network(obs_dim, action_dim, hidden_sizes, device)
        target_net = build_network(obs_dim, action_dim, hidden_sizes, device)
        optimizer = make_optimizer(q_net, train_config)
        buffer = ReplayBuffer(train_config.buffer_size, obs_dim, seed=0)

        train_dqn(
            env,
            q_net,
            target_net,
            optimizer,
            buffer,
            train_seeds,
            train_config,
            device,
            rng,
        )
        train_return = evaluate_policy(
            env, q_net, train_seeds, episodes_per_seed=5, device=device
        )
    if train_return < args.sanity_threshold:
        raise RuntimeError(
            f"Sanity check failed: expected return >= {args.sanity_threshold}, got {train_return:.3f}"
        )


def run_experiment(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    train_seeds = parse_int_list(args.train_seeds)
    test_seeds = parse_int_list(args.test_seeds)
    widths = parse_widths(args.widths)
    depths = parse_depths(args.depths)
    if args.run_id is not None and args.runs != 1:
        raise ValueError("Use --runs 1 when --run-id is set.")
    log_root = Path(args.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)

    env_config = GridWorldConfig(
        grid_size=args.grid_size,
        obstacle_prob=args.obstacle_prob,
        max_steps=args.max_steps,
    )
    train_config = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        buffer_size=args.buffer_size,
        target_update_interval=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        early_stop_return=args.early_stop_return,
        early_stop_episodes=args.early_stop_episodes,
    )

    results: List[Dict[str, float]] = []

    run_ids = [args.run_id] if args.run_id is not None else list(range(args.runs))
    total_runs = len(widths) * len(depths) * len(run_ids)
    progress = tqdm(total=total_runs, desc="runs")

    for depth in depths:
        for width in widths:
            hidden_sizes = build_hidden_sizes(width, depth)
            for run_id in run_ids:
                run_seed = args.base_seed + int(run_id)
                set_global_seeds(run_seed)
                rng = np.random.default_rng(run_seed)

                env = GridWorldEnv(env_config, seed=run_seed)
                obs_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n

                episode_progress = tqdm(
                    total=args.episodes,
                    desc=f"episodes w{width} d{depth} run{run_id}",
                    leave=False,
                )
                try:
                    if args.algo == "trpo":
                        trpo_config = TRPOConfig(
                            episodes=args.episodes,
                            batch_episodes=args.trpo_batch_episodes,
                            max_steps=args.max_steps,
                            gamma=args.gamma,
                            gae_lambda=args.trpo_gae_lambda,
                            max_kl=args.trpo_max_kl,
                            cg_iters=args.trpo_cg_iters,
                            cg_damping=args.trpo_cg_damping,
                            backtrack_coeff=args.trpo_backtrack_coeff,
                            backtrack_iters=args.trpo_backtrack_iters,
                            vf_iters=args.trpo_vf_iters,
                            vf_lr=args.trpo_vf_lr,
                            early_stop_return=args.early_stop_return,
                            early_stop_episodes=args.early_stop_episodes,
                        )
                        policy_net = build_policy(
                            obs_dim, action_dim, hidden_sizes, device
                        )
                        value_net = build_value(obs_dim, hidden_sizes, device)
                        train_info = train_trpo(
                            env,
                            policy_net,
                            value_net,
                            train_seeds,
                            trpo_config,
                            device,
                            rng,
                            progress=episode_progress,
                        )
                        model_for_eval = policy_net
                    else:
                        q_net = build_network(obs_dim, action_dim, hidden_sizes, device)
                        target_net = build_network(
                            obs_dim, action_dim, hidden_sizes, device
                        )
                        optimizer = make_optimizer(q_net, train_config)
                        buffer = ReplayBuffer(
                            train_config.buffer_size, obs_dim, seed=run_seed
                        )

                        train_info = train_dqn(
                            env,
                            q_net,
                            target_net,
                            optimizer,
                            buffer,
                            train_seeds,
                            train_config,
                            device,
                            rng,
                            progress=episode_progress,
                        )
                        model_for_eval = q_net
                finally:
                    episode_progress.close()
                run_dir = log_root / f"w{width}_d{depth}_run{run_id}"
                run_dir.mkdir(parents=True, exist_ok=True)
                if args.algo == "trpo":
                    save_model_state(policy_net, run_dir / "policy.pt")
                    save_model_state(value_net, run_dir / "value.pt")
                else:
                    save_model_state(model_for_eval, run_dir / "model.pt")
                meta = {
                    "width": float(width),
                    "depth": float(depth),
                    "run": float(run_id),
                    "num_params": float(count_parameters(model_for_eval)),
                }
                save_episode_metrics(
                    train_info["episode_returns"],
                    train_info["episode_lengths"],
                    run_dir / "episodes.csv",
                    meta,
                )
                plot_episode_metrics(
                    train_info["episode_returns"],
                    train_info["episode_lengths"],
                    run_dir / "episodes.png",
                )
                if args.algo == "trpo" and train_info.get("update_kl"):
                    update_rows: List[Dict[str, object]] = []
                    for idx, (kl_val, ent_val) in enumerate(
                        zip(train_info["update_kl"], train_info["update_entropy"])
                    ):
                        update_rows.append(
                            {
                                "update": idx,
                                "kl": float(kl_val),
                                "entropy": float(ent_val),
                            }
                        )
                    save_csv(run_dir / "trpo_updates.csv", update_rows)

                if args.algo == "trpo":
                    train_return = evaluate_policy_trpo(
                        env,
                        model_for_eval,
                        train_seeds,
                        episodes_per_seed=args.eval_episodes,
                        device=device,
                    )
                    test_return = evaluate_policy_trpo(
                        env,
                        model_for_eval,
                        test_seeds,
                        episodes_per_seed=args.eval_episodes,
                        device=device,
                    )
                else:
                    train_return = evaluate_policy(
                        env,
                        model_for_eval,
                        train_seeds,
                        episodes_per_seed=args.eval_episodes,
                        device=device,
                    )
                    test_return = evaluate_policy(
                        env,
                        model_for_eval,
                        test_seeds,
                        episodes_per_seed=args.eval_episodes,
                        device=device,
                    )

                fim_trace = estimate_fim_trace(
                    env,
                    model_for_eval,
                    train_seeds,
                    device,
                    args.max_steps,
                    args.fim_samples,
                    args.fim_hutchinson,
                    rng,
                )

                if train_seeds and test_seeds:
                    if args.video_seeds:
                        seeds_to_render = parse_int_list(args.video_seeds)
                        for seed in seeds_to_render:
                            frames, _ = rollout_episode(
                                env,
                                model_for_eval,
                                int(seed),
                                device,
                                args.max_steps,
                            )
                            save_gif(
                                frames, run_dir / f"seed{seed}.gif", args.video_fps
                            )

                run_row = {
                    "width": float(width),
                    "depth": float(depth),
                    "run": float(run_id),
                    "num_params": float(count_parameters(model_for_eval)),
                    "train_return": float(train_return),
                    "test_return": float(test_return),
                    "fim_trace": float(fim_trace),
                }
                save_csv(run_dir / "metrics.csv", [run_row])
                results.append(run_row)
                progress.update(1)

    progress.close()
    summary = aggregate_results(results)
    return results, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DQN double descent experiment")
    parser.add_argument("--train-seeds", default="1-25")
    parser.add_argument("--test-seeds", default="26-30")
    parser.add_argument("--widths", default="16,32,64,128,256,512")
    parser.add_argument("--depths", default="2")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--algo", choices=["dqn", "trpo"], default="dqn")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--obstacle-prob", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=30_000)
    parser.add_argument("--early-stop-return", type=float, default=0.7)
    parser.add_argument("--early-stop-episodes", type=int, default=10)
    parser.add_argument("--trpo-max-kl", type=float, default=1e-2)
    parser.add_argument("--trpo-cg-iters", type=int, default=10)
    parser.add_argument("--trpo-cg-damping", type=float, default=0.1)
    parser.add_argument("--trpo-backtrack-coeff", type=float, default=0.5)
    parser.add_argument("--trpo-backtrack-iters", type=int, default=10)
    parser.add_argument("--trpo-vf-iters", type=int, default=5)
    parser.add_argument("--trpo-vf-lr", type=float, default=1e-3)
    parser.add_argument("--trpo-gae-lambda", type=float, default=0.95)
    parser.add_argument("--trpo-batch-episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--log-x", dest="log_x", action="store_true", default=True)
    parser.add_argument("--no-log-x", dest="log_x", action="store_false")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--collect-only", action="store_true", default=False)
    parser.add_argument("--video-seeds", default="")
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--fim-samples", type=int, default=64)
    parser.add_argument("--fim-hutchinson", type=int, default=4)
    parser.add_argument("--sanity-check", action="store_true", default=False)
    parser.add_argument("--sanity-only", action="store_true", default=False)
    parser.add_argument("--sanity-episodes", type=int, default=800)
    parser.add_argument("--sanity-threshold", type=float, default=0.8)
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = str(Path("results") / timestamp)

    if args.collect_only:
        collect_results(Path(args.log_dir), log_x=args.log_x)
        return

    if args.sanity_check:
        run_sanity_check(args)
        if args.sanity_only:
            return

    results, summary = run_experiment(args)
    log_root = Path(args.log_dir)
    save_csv(log_root / "metrics.csv", results)
    save_csv(log_root / "summary.csv", summary)

    plot_results(summary, log_root / "curve.png", log_x=args.log_x)


if __name__ == "__main__":
    main()
