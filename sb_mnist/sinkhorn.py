from __future__ import annotations

import math

import torch


def pairwise_squared_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared Euclidean cost between flattened batches."""
    x_flat = x.flatten(1)
    y_flat = y.flatten(1)
    return torch.cdist(x_flat, y_flat, p=2).pow(2) / x_flat.shape[1]


def log_sinkhorn(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 80,
) -> torch.Tensor:
    """Return an entropic OT coupling with uniform marginals.

    The log-domain updates are stable for the small regularization values that
    are useful for image batches.
    """
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D tensor")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    rows, cols = cost.shape
    log_mu = cost.new_full((rows,), -math.log(rows))
    log_nu = cost.new_full((cols,), -math.log(cols))
    log_kernel = -cost / epsilon
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)

    for _ in range(n_iters):
        u = log_mu - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)

    return torch.exp(log_kernel + u.unsqueeze(1) + v.unsqueeze(0))


@torch.no_grad()
def sinkhorn_match_targets(
    source: torch.Tensor,
    target_pool: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 80,
    hard: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sample target images from a mini-batch entropic OT plan."""
    cost = pairwise_squared_cost(source.detach(), target_pool.detach())
    plan = log_sinkhorn(cost, epsilon=epsilon, n_iters=n_iters)
    row_probs = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)

    if hard:
        indices = row_probs.argmax(dim=1)
    else:
        indices = torch.multinomial(row_probs, num_samples=1).squeeze(1)

    matched = target_pool.index_select(0, indices)
    plan_cost = (plan * cost).sum()
    entropy = -(plan * plan.clamp_min(1e-12).log()).sum()
    stats = {
        "sinkhorn_cost": float(plan_cost.detach().cpu()),
        "sinkhorn_entropy": float(entropy.detach().cpu()),
        "matched_mse": float(pairwise_squared_cost(source, matched).diag().mean().detach().cpu()),
    }
    return matched, stats


@torch.no_grad()
def random_match_targets(
    source: torch.Tensor,
    target_pool: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    indices = torch.randperm(target_pool.shape[0], device=target_pool.device)[: source.shape[0]]
    matched = target_pool.index_select(0, indices)
    stats = {
        "sinkhorn_cost": 0.0,
        "sinkhorn_entropy": 0.0,
        "matched_mse": float(pairwise_squared_cost(source, matched).diag().mean().detach().cpu()),
    }
    return matched, stats

