from __future__ import annotations

import math

import torch
from torch import nn


def bridge_std(t: torch.Tensor, sigma: float) -> torch.Tensor:
    return sigma * torch.sqrt((t * (1.0 - t)).clamp_min(0.0))


def sample_brownian_bridge(
    source: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Sample X_t from a Brownian bridge between paired endpoints."""
    while t.ndim < source.ndim:
        t = t.unsqueeze(-1)
    mean = (1.0 - t) * source + t * target
    std = bridge_std(t, sigma)
    return mean + std * torch.randn_like(source)


def terminal_drift(
    x_t: torch.Tensor,
    terminal_prediction: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    while t.ndim < x_t.ndim:
        t = t.unsqueeze(-1)
    return (terminal_prediction - x_t) / (1.0 - t).clamp_min(eps)


@torch.no_grad()
def euler_transport(
    model: nn.Module,
    source: torch.Tensor,
    steps: int,
    sigma: float,
    eta: float = 0.25,
    eps: float = 1e-3,
    clamp: tuple[float, float] | None = (-1.5, 1.5),
    return_trajectory: bool = False,
    trajectory_points: int = 6,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Transport source samples toward the learned target marginal."""
    if steps < 2:
        raise ValueError("steps must be at least 2")

    model_was_training = model.training
    model.eval()
    x = source.clone()
    trajectory: list[torch.Tensor] = [x.detach().cpu()] if return_trajectory else []
    save_indices = set()
    if return_trajectory:
        save_indices = set(torch.linspace(1, steps, trajectory_points).round().long().tolist())

    device = source.device
    grid = torch.linspace(0.0, 1.0 - eps, steps + 1, device=device)
    for step in range(steps):
        t0 = grid[step]
        t1 = grid[step + 1]
        dt = t1 - t0
        t_batch = torch.full((source.shape[0],), float(t0), device=device)
        terminal = model(x, source, t_batch).clamp(-1.0, 1.0)
        drift = terminal_drift(x, terminal, t_batch, eps=eps)
        noise = 0.0
        if eta > 0.0 and step < steps - 1:
            noise = eta * sigma * math.sqrt(float(dt)) * torch.randn_like(x)
        x = x + dt * drift + noise
        if clamp is not None:
            x = x.clamp(*clamp)
        if return_trajectory and (step + 1) in save_indices:
            trajectory.append(x.detach().cpu())

    final_t = torch.full((source.shape[0],), 1.0 - eps, device=device)
    x = model(x, source, final_t).clamp(-1.0, 1.0)
    if return_trajectory:
        trajectory.append(x.detach().cpu())
    if model_was_training:
        model.train()
    return x, trajectory

