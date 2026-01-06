from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


@dataclass
class TRPOConfig:
    episodes: int = 2000
    batch_episodes: int = 20
    max_steps: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_kl: float = 1e-2
    cg_iters: int = 10
    cg_damping: float = 0.1
    backtrack_coeff: float = 0.5
    backtrack_iters: int = 10
    vf_iters: int = 5
    vf_lr: float = 1e-3
    early_stop_return: float = 0.7
    early_stop_episodes: int = 10


def build_policy(input_dim: int, action_dim: int, hidden_sizes: Iterable[int], device: torch.device) -> PolicyNetwork:
    net = PolicyNetwork(input_dim, action_dim, hidden_sizes)
    return net.to(device)


def build_value(input_dim: int, hidden_sizes: Iterable[int], device: torch.device) -> ValueNetwork:
    net = ValueNetwork(input_dim, hidden_sizes)
    return net.to(device)


def evaluate_policy(
    env,
    policy_net: PolicyNetwork,
    seeds: Iterable[int],
    episodes_per_seed: int,
    device: torch.device,
) -> float:
    policy_net.eval()
    returns = []
    for seed in seeds:
        for _ in range(episodes_per_seed):
            obs, _ = env.reset(seed=int(seed))
            total = 0.0
            done = False
            while not done:
                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    logits = policy_net(obs_t)
                action = int(torch.argmax(logits, dim=1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            returns.append(total)
    policy_net.train()
    if not returns:
        return 0.0
    return float(np.mean(returns))


def _flat_params(params: Iterable[nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in params])


def _set_params(params: Iterable[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        param.data.copy_(flat[offset : offset + numel].view_as(param))
        offset += numel


def _flat_grad(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.contiguous().view(-1) for g in grads])


def _conjugate_gradient(fvp, b, nsteps: int, residual_tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(nsteps):
        Ap = fvp(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


def _compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros(len(rewards), dtype=np.float32)
    returns = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for idx in reversed(range(len(rewards))):
        next_value = values[idx + 1] if idx + 1 < len(values) else 0.0
        next_non_terminal = 0.0 if dones[idx] else 1.0
        delta = rewards[idx] + gamma * next_value * next_non_terminal - values[idx]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        adv[idx] = last_gae
        returns[idx] = adv[idx] + values[idx]
    return returns, adv


def train_trpo(
    env,
    policy_net: PolicyNetwork,
    value_net: ValueNetwork,
    train_seeds: Iterable[int],
    config: TRPOConfig,
    device: torch.device,
    rng: np.random.Generator,
    progress: Optional[object] = None,
) -> dict[str, List[float]]:
    policy_net.train()
    value_net.train()

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    update_kls: List[float] = []
    update_entropies: List[float] = []
    train_seed_list = list(train_seeds)
    vf_optimizer = torch.optim.Adam(value_net.parameters(), lr=config.vf_lr)
    early_stop_count = 0
    total_episodes = 0

    while total_episodes < config.episodes:
        batch_obs: List[np.ndarray] = []
        batch_actions: List[int] = []
        batch_log_probs: List[float] = []
        batch_returns: List[float] = []
        batch_advantages: List[float] = []

        episodes_in_batch = 0
        batch_steps = 0
        stop_training = False

        while episodes_in_batch < config.batch_episodes and total_episodes < config.episodes:
            env_seed = int(rng.choice(train_seed_list))
            obs, _ = env.reset(seed=env_seed)
            done = False
            steps = 0

            obs_list: List[np.ndarray] = []
            actions_list: List[int] = []
            rewards_list: List[float] = []
            dones_list: List[bool] = []
            values_list: List[float] = []
            log_probs_list: List[float] = []

            episode_return = 0.0

            while not done and steps < config.max_steps:
                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    logits = policy_net(obs_t)
                    value = value_net(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated

                obs_list.append(obs)
                actions_list.append(int(action.item()))
                rewards_list.append(float(reward))
                dones_list.append(bool(done))
                values_list.append(float(value.item()))
                log_probs_list.append(float(log_prob.item()))

                obs = next_obs
                steps += 1
                episode_return += reward

            if obs_list:
                returns, advantages = _compute_gae(
                    rewards_list,
                    values_list,
                    dones_list,
                    config.gamma,
                    config.gae_lambda,
                )
                batch_obs.extend(obs_list)
                batch_actions.extend(actions_list)
                batch_log_probs.extend(log_probs_list)
                batch_returns.extend(returns.tolist())
                batch_advantages.extend(advantages.tolist())
                batch_steps += len(obs_list)

            episode_returns.append(float(episode_return))
            episode_lengths.append(int(steps))
            if progress is not None:
                progress.update(1)
            total_episodes += 1
            episodes_in_batch += 1

            if config.early_stop_episodes > 0:
                if episode_return >= config.early_stop_return:
                    early_stop_count += 1
                else:
                    early_stop_count = 0
                if early_stop_count >= config.early_stop_episodes:
                    stop_training = True
                    break

        if batch_steps == 0:
            break

        obs_batch = torch.from_numpy(np.array(batch_obs, dtype=np.float32)).to(device)
        actions_batch = torch.tensor(batch_actions, dtype=torch.int64, device=device)
        old_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32, device=device)

        returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(config.vf_iters):
            values_pred = value_net(obs_batch)
            vf_loss = torch.mean((values_pred - returns_t) ** 2)
            vf_optimizer.zero_grad()
            vf_loss.backward()
            vf_optimizer.step()

        with torch.no_grad():
            old_logits = policy_net(obs_batch)
            old_probs = torch.softmax(old_logits, dim=1)
            entropy = -(old_probs * torch.log(old_probs + 1e-8)).sum(dim=1).mean()
        update_entropies.append(float(entropy.item()))

        params = [p for p in policy_net.parameters() if p.requires_grad]

        def surrogate_loss() -> torch.Tensor:
            logits = policy_net(obs_batch)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_batch)
            ratio = torch.exp(log_probs - old_log_probs)
            return torch.mean(ratio * adv_t)

        def kl_divergence() -> torch.Tensor:
            logits = policy_net(obs_batch)
            new_probs = torch.softmax(logits, dim=1)
            kl = old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_probs + 1e-8))
            return kl.sum(dim=1).mean()

        loss = surrogate_loss()
        grads = torch.autograd.grad(loss, params)
        g = _flat_grad(grads).detach()

        def fvp(v: torch.Tensor) -> torch.Tensor:
            kl = kl_divergence()
            grads_kl = torch.autograd.grad(kl, params, create_graph=True)
            flat_kl = _flat_grad(grads_kl)
            kl_v = torch.dot(flat_kl, v)
            grads_hvp = torch.autograd.grad(kl_v, params)
            flat_hvp = _flat_grad(grads_hvp).detach()
            return flat_hvp + config.cg_damping * v

        step_dir = _conjugate_gradient(fvp, g, config.cg_iters)
        shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
        if shs <= 0:
            update_kls.append(0.0)
            if stop_training:
                break
            continue
        step_size = torch.sqrt(config.max_kl / (shs + 1e-8))
        full_step = step_dir * step_size

        old_params = _flat_params(params)
        old_loss = float(loss.item())
        success = False
        final_kl = float("nan")

        for step in range(config.backtrack_iters):
            step_frac = config.backtrack_coeff ** step
            new_params = old_params + step_frac * full_step
            _set_params(params, new_params)
            new_loss = float(surrogate_loss().item())
            kl_val = float(kl_divergence().item())
            if new_loss > old_loss and kl_val <= config.max_kl:
                success = True
                final_kl = kl_val
                break

        if not success:
            _set_params(params, old_params)
            final_kl = float(kl_divergence().item())

        update_kls.append(final_kl)

        if stop_training:
            break

    return {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "update_kl": update_kls,
        "update_entropy": update_entropies,
    }
