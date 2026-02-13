from __future__ import annotations

import numpy as np

from rl_dd.env import GridWorldConfig, GridWorldEnv

from tqdm import trange


def main() -> None:
    config = GridWorldConfig()
    returns = []
    suboptimal_maps = 0
    perfect_maps = 0
    n = 50
    for seed in trange(n):
        env = GridWorldEnv(config, seed=seed)
        env.reset(seed=seed)
        suboptimal_maps += int(len(env._shortest_path()) > 15)
        perfect_maps += int(env.optimal_return() == 1)
        returns.append(env.optimal_return())
    mean_return = float(np.mean(returns)) if returns else float("nan")
    print(f"Mean optimal return over {len(returns)} envs: {mean_return:.6f}")
    print(f"Proportion of optimal maps (==1 return): {100* perfect_maps / n:.6f}%")
    print(
        f"Proportion of suboptimal maps (requiring >14 steps): {100* suboptimal_maps / n:.6f}%"
    )


if __name__ == "__main__":
    main()
