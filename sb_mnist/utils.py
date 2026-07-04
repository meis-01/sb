from __future__ import annotations

import csv
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cycle(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def denormalize_mnist(x: torch.Tensor) -> torch.Tensor:
    """Convert MNIST tensors from [-1, 1] to [0, 1]."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def append_metrics(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


class EMA:
    """Exponential moving average for evaluation and sampling."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in state.items():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: value.clone() for name, value in state.items()}

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for name, value in state.items():
            if name in self.shadow:
                self.backup[name] = value.detach().clone()
                value.copy_(self.shadow[name].to(device=value.device, dtype=value.dtype))

    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        state = model.state_dict()
        for name, value in self.backup.items():
            state[name].copy_(value)
        self.backup = {}

