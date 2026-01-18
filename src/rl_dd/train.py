from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn

from rl_dd.dqn import QNetwork, ReplayBuffer


@dataclass
class TrainConfig:
    episodes: int = 2000
    max_steps: int = 64
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 50_000
    target_update_interval: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 600
    early_stop_return: float = 0.7
    early_stop_episodes: int = 10


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    frac = max(0.0, (decay_steps - step) / decay_steps)
    return end + (start - end) * frac


def train_dqn(
    env,
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    train_seeds: Iterable[int],
    config: TrainConfig,
    device: torch.device,
    rng: np.random.Generator,
    progress: Optional[object] = None,
    log_every: int = 0,
    log_callback: Optional[Callable[[Dict[str, List[float]]], None]] = None,
) -> Dict[str, List[float]]:
    q_net.train()
    target_net.load_state_dict(q_net.state_dict())

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    global_step = 0
    train_seed_list = list(train_seeds)
    early_stop_count = 0

    for episode_idx in range(config.episodes):
        env_seed = int(rng.choice(train_seed_list))
        obs, _ = env.reset(seed=env_seed)
        episode_return = 0.0
        episode_length = 0

        for _ in range(config.max_steps):
            episode_length += 1
            eps = linear_epsilon(
                episode_idx,
                config.eps_start,
                config.eps_end,
                config.eps_decay_episodes,
            )
            if rng.random() < eps:
                action = int(rng.integers(0, env.action_space.n))
            else:
                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = q_net(obs_t)
                action = int(torch.argmax(q_vals, dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            global_step += 1

            if buffer.size >= config.batch_size:
                batch = buffer.sample(config.batch_size, device)
                q_values = (
                    q_net(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
                )
                with torch.no_grad():
                    next_q = target_net(batch.next_obs).max(1)[0]
                    target = batch.rewards + config.gamma * (1.0 - batch.dones) * next_q
                loss = nn.functional.smooth_l1_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (
                config.target_update_interval > 0
                and global_step % config.target_update_interval == 0
            ):
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        if progress is not None:
            progress.update(1)
        if (
            log_every > 0
            and log_callback is not None
            and (episode_idx + 1) % log_every == 0
        ):
            log_callback(
                {
                    "episode_returns": episode_returns,
                    "episode_lengths": episode_lengths,
                }
            )
        if config.early_stop_episodes > 0:
            if episode_return >= config.early_stop_return:
                early_stop_count += 1
            else:
                early_stop_count = 0
            if early_stop_count >= config.early_stop_episodes:
                break

    return {"episode_returns": episode_returns, "episode_lengths": episode_lengths}


def evaluate_policy(
    env,
    q_net: QNetwork,
    seeds: Iterable[int],
    episodes_per_seed: int,
    device: torch.device,
) -> float:
    q_net.eval()
    returns = []
    for seed in seeds:
        for _ in range(episodes_per_seed):
            obs, _ = env.reset(seed=int(seed))
            total = 0.0
            done = False
            while not done:
                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = q_net(obs_t)
                action = int(torch.argmax(q_vals, dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            returns.append(total)
    q_net.train()
    if not returns:
        return 0.0
    return float(np.mean(returns))


def build_network(
    input_dim: int,
    action_dim: int,
    hidden_sizes: Iterable[int],
    device: torch.device,
) -> QNetwork:
    net = QNetwork(input_dim, action_dim, hidden_sizes)
    return net.to(device)


def make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=config.lr)


def _flatten_grads(params: Iterable[nn.Parameter]) -> torch.Tensor:
    grads = []
    for param in params:
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.detach().reshape(-1))
    return torch.cat(grads)


def estimate_fim_trace(
    env,
    q_net: QNetwork,
    seeds: Iterable[int],
    device: torch.device,
    max_steps: int,
    sample_count: int,
    hutchinson_samples: int,
    rng: np.random.Generator,
) -> float:
    if sample_count <= 0 or hutchinson_samples <= 0:
        return float("nan")
    seed_list = list(seeds)
    if not seed_list:
        return float("nan")

    q_net.eval()
    params = [p for p in q_net.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)
    if num_params == 0:
        return float("nan")

    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(int(rng.integers(0, 2**31 - 1)))

    accum = 0.0
    state_samples = 0
    seed_idx = 0

    while state_samples < sample_count:
        seed = seed_list[seed_idx % len(seed_list)]
        seed_idx += 1
        obs, _ = env.reset(seed=int(seed))
        done = False
        steps = 0

        while not done and steps < max_steps and state_samples < sample_count:
            obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            logits = q_net(obs_t)
            probs = torch.softmax(logits, dim=1)
            action = torch.multinomial(probs, num_samples=1, generator=torch_rng).item()
            log_prob = torch.log(probs[0, action] + 1e-8)

            q_net.zero_grad(set_to_none=True)
            log_prob.backward()
            grads = _flatten_grads(params)

            for _ in range(hutchinson_samples):
                z = torch.randint(
                    0,
                    2,
                    (num_params,),
                    generator=torch_rng,
                    device=device,
                    dtype=torch.int8,
                )
                z = (z.to(dtype=grads.dtype) * 2) - 1
                accum += float(torch.dot(grads, z).item() ** 2)

            state_samples += 1
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            steps += 1

    q_net.train()
    return accum / float(state_samples * hutchinson_samples)
