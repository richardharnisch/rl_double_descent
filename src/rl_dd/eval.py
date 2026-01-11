from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from rl_dd.env import GridWorldConfig, GridWorldEnv
from rl_dd.train import build_network
from rl_dd.trpo import build_policy


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
    env: GridWorldEnv,
    model: torch.nn.Module,
    seed: int,
    device: torch.device,
    max_steps: int,
) -> Tuple[List[np.ndarray], float, int, int]:
    was_training = model.training
    model.eval()
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_return = 0.0
    done = False
    steps = 0
    success = 0
    while not done and steps < max_steps:
        obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        with torch.no_grad():
            logits = model(obs_t)
        action = int(torch.argmax(logits, dim=1).item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total_return += reward
        frames.append(env.render())
        done = terminated or truncated
        if terminated:
            success = 1
        steps += 1
    if was_training:
        model.train()
    return frames, total_return, success, steps


def infer_arch(model_path: Path) -> Optional[Tuple[int, int]]:
    metrics_path = model_path.parent / "metrics.csv"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            width = int(float(row["width"]))
            depth = int(float(row["depth"]))
            return width, depth
    return None


def build_model(
    algo: str,
    obs_dim: int,
    action_dim: int,
    width: int,
    depth: int,
    device: torch.device,
) -> torch.nn.Module:
    hidden_sizes = [width for _ in range(depth)]
    if algo == "trpo":
        return build_policy(obs_dim, action_dim, hidden_sizes, device)
    return build_network(obs_dim, action_dim, hidden_sizes, device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on seeded maps."
    )
    parser.add_argument("--algo", choices=["dqn", "trpo"], default="dqn")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--obstacle-prob", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=2)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--start-corner", dest="start", type=int, default=None)
    parser.add_argument("--goal-corner", dest="end", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if args.width is None or args.depth is None:
        inferred = infer_arch(model_path)
        if inferred is None:
            raise ValueError(
                "Provide --width and --depth (could not infer from metrics.csv)."
            )
        args.width, args.depth = inferred

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    if args.start is not None and args.end is not None:
        if args.start == args.end:
            raise ValueError("--start and --end must be different.")

    env_config = GridWorldConfig(
        grid_size=args.grid_size,
        obstacle_prob=args.obstacle_prob,
        max_steps=args.max_steps,
        frame_stack=args.frame_stack,
        start_corner=args.start,
        goal_corner=args.end,
    )
    env = GridWorldEnv(env_config, seed=0)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = build_model(args.algo, obs_dim, action_dim, args.width, args.depth, device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    seeds = parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        frames, total_return, success, steps = rollout_episode(
            env,
            model,
            int(seed),
            device,
            args.max_steps,
        )
        save_gif(frames, out_dir / f"seed{seed}.gif", args.video_fps)
        results.append(
            {
                "seed": int(seed),
                "return": float(total_return),
                "success": int(success),
                "length": int(steps),
            }
        )

    if results:
        with (out_dir / "eval_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
