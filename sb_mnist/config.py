from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    data_dir: str = "data"
    output_dir: str = "runs/mnist_2_to_8"
    source_digit: int = 2
    target_digit: int = 8
    batch_size: int = 128
    num_workers: int = 2
    steps: int = 5000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    base_channels: int = 48
    time_dim: int = 128
    bridge_sigma: float = 0.45
    t_eps: float = 1e-3
    coupling: str = "sinkhorn"
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 80
    hard_coupling: bool = False
    ema_decay: float = 0.999
    sample_steps: int = 64
    sample_eta: float = 0.35
    preview_count: int = 8
    log_every: int = 50
    sample_every: int = 500
    checkpoint_every: int = 1000
    seed: int = 7
    device: str = "auto"
    amp: bool = True
    download: bool = True


@dataclass(slots=True)
class SampleConfig:
    checkpoint: str
    output: str = "runs/samples_2_to_8.png"
    data_dir: str = "data"
    source_digit: int = 2
    num_samples: int = 16
    batch_size: int = 16
    steps: int = 96
    eta: float = 0.25
    split: str = "test"
    seed: int = 11
    device: str = "auto"
    download: bool = True


def save_json_config(config: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2, sort_keys=True)
        file.write("\n")


def load_json_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)

