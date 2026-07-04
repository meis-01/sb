from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .bridge import euler_transport
from .config import SampleConfig, TrainConfig
from .data import DigitMNIST
from .models import ConditionalUNet
from .utils import denormalize_mnist, seed_everything, select_device


def parse_sample_args() -> SampleConfig:
    parser = argparse.ArgumentParser(description="Sample a trained MNIST 2-to-8 bridge.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=SampleConfig.__dataclass_fields__["output"].default)
    parser.add_argument("--data-dir", default=SampleConfig.__dataclass_fields__["data_dir"].default)
    parser.add_argument("--source-digit", type=int, default=SampleConfig.__dataclass_fields__["source_digit"].default)
    parser.add_argument("--num-samples", type=int, default=SampleConfig.__dataclass_fields__["num_samples"].default)
    parser.add_argument("--batch-size", type=int, default=SampleConfig.__dataclass_fields__["batch_size"].default)
    parser.add_argument("--steps", type=int, default=SampleConfig.__dataclass_fields__["steps"].default)
    parser.add_argument("--eta", type=float, default=SampleConfig.__dataclass_fields__["eta"].default)
    parser.add_argument("--split", choices=["train", "test"], default=SampleConfig.__dataclass_fields__["split"].default)
    parser.add_argument("--seed", type=int, default=SampleConfig.__dataclass_fields__["seed"].default)
    parser.add_argument("--device", default=SampleConfig.__dataclass_fields__["device"].default)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return SampleConfig(**vars(parser.parse_args()))


def _load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[ConditionalUNet, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = TrainConfig(**checkpoint["config"])
    model = ConditionalUNet(cfg.base_channels, cfg.time_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    if "ema" in checkpoint:
        state = model.state_dict()
        for name, value in checkpoint["ema"].items():
            if name in state:
                state[name].copy_(value.to(device=device, dtype=state[name].dtype))
    model.eval()
    return model, cfg


@torch.no_grad()
def run_sampling(cfg: SampleConfig) -> None:
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    model, train_cfg = _load_model(cfg.checkpoint, device)

    dataset = DigitMNIST(
        cfg.data_dir,
        digit=cfg.source_digit,
        train=cfg.split == "train",
        download=cfg.download,
    )
    loader = DataLoader(dataset, batch_size=cfg.num_samples, shuffle=True, drop_last=False)
    source = next(iter(loader)).to(device)
    source = source[: cfg.num_samples]

    _, trajectory = euler_transport(
        model,
        source,
        steps=cfg.steps,
        sigma=train_cfg.bridge_sigma,
        eta=cfg.eta,
        eps=train_cfg.t_eps,
        return_trajectory=True,
        trajectory_points=6,
    )
    grid = torch.cat(trajectory, dim=0)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize_mnist(grid), output, nrow=source.shape[0])
    print(f"saved {output}")


def main() -> None:
    run_sampling(parse_sample_args())


if __name__ == "__main__":
    main()

