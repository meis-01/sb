from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from .bridge import euler_transport, sample_brownian_bridge
from .config import TrainConfig, save_json_config
from .data import make_digit_loaders
from .models import ConditionalUNet
from .sinkhorn import random_match_targets, sinkhorn_match_targets
from .utils import EMA, append_metrics, count_parameters, cycle, denormalize_mnist, seed_everything, select_device


def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a MNIST 2-to-8 Schrodinger bridge.")
    defaults = TrainConfig()
    for field_name, value in asdict(defaults).items():
        arg_name = "--" + field_name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def _match_targets(
    cfg: TrainConfig,
    source: torch.Tensor,
    target_pool: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if cfg.coupling == "sinkhorn":
        return sinkhorn_match_targets(
            source,
            target_pool,
            epsilon=cfg.sinkhorn_epsilon,
            n_iters=cfg.sinkhorn_iters,
            hard=cfg.hard_coupling,
        )
    if cfg.coupling == "random":
        return random_match_targets(source, target_pool)
    raise ValueError("coupling must be 'sinkhorn' or 'random'")


def save_checkpoint(
    path: str | Path,
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "config": asdict(cfg),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


@torch.no_grad()
def save_preview(
    path: str | Path,
    model: nn.Module,
    ema: EMA,
    source: torch.Tensor,
    cfg: TrainConfig,
) -> None:
    from torchvision.utils import save_image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = source[: cfg.preview_count]
    ema.apply_to(model)
    try:
        _, trajectory = euler_transport(
            model,
            source,
            steps=cfg.sample_steps,
            sigma=cfg.bridge_sigma,
            eta=cfg.sample_eta,
            eps=cfg.t_eps,
            return_trajectory=True,
            trajectory_points=5,
        )
    finally:
        ema.restore(model)
    grid = torch.cat(trajectory, dim=0)
    save_image(denormalize_mnist(grid), path, nrow=source.shape[0])


def run_training(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json_config(cfg, out_dir / "config.json")

    device = select_device(cfg.device)
    source_loader, target_loader = make_digit_loaders(
        cfg.data_dir,
        cfg.source_digit,
        cfg.target_digit,
        cfg.batch_size,
        cfg.num_workers,
        train=True,
        download=cfg.download,
    )
    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)

    model = ConditionalUNet(cfg.base_channels, cfg.time_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_path = out_dir / "metrics.csv"
    start = time.time()
    print(f"device={device} parameters={count_parameters(model):,} output={out_dir}")

    progress = trange(1, cfg.steps + 1, dynamic_ncols=True)
    for step in progress:
        source = next(source_iter).to(device, non_blocking=True)
        target_pool = next(target_iter).to(device, non_blocking=True)
        target, match_stats = _match_targets(cfg, source, target_pool)
        t = torch.rand(source.shape[0], device=device) * (1.0 - 2.0 * cfg.t_eps) + cfg.t_eps
        x_t = sample_brownian_bridge(source, target, t, sigma=cfg.bridge_sigma)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x_t, source, t)
            loss = F.mse_loss(pred, target)

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        progress.set_description(f"loss={loss.item():.4f}")
        if step % cfg.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "elapsed_sec": round(time.time() - start, 2),
                **match_stats,
            }
            append_metrics(metrics_path, row)
        if step % cfg.sample_every == 0 or step == cfg.steps:
            save_preview(out_dir / f"preview_step_{step:06d}.png", model, ema, source, cfg)
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            save_checkpoint(out_dir / "latest.pt", cfg, model, optimizer, ema, step)
            save_checkpoint(out_dir / f"step_{step:06d}.pt", cfg, model, optimizer, ema, step)


def main() -> None:
    run_training(parse_train_args())


if __name__ == "__main__":
    main()

