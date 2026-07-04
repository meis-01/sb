from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def _load_torchvision():
    try:
        from torchvision import datasets, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for MNIST loading. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return datasets, transforms


class DigitMNIST(Dataset):
    """MNIST filtered to one digit and normalized to [-1, 1]."""

    def __init__(
        self,
        root: str | Path,
        digit: int,
        train: bool,
        download: bool = True,
    ) -> None:
        datasets, transforms = _load_torchvision()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.dataset = datasets.MNIST(
            root=str(root),
            train=train,
            download=download,
            transform=transform,
        )
        targets = torch.as_tensor(self.dataset.targets)
        self.indices = torch.nonzero(targets == int(digit), as_tuple=False).flatten().tolist()
        if not self.indices:
            raise ValueError(f"No MNIST samples found for digit {digit}.")
        self.digit = int(digit)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset[self.indices[index]]
        return image


def make_digit_loaders(
    data_dir: str | Path,
    source_digit: int,
    target_digit: int,
    batch_size: int,
    num_workers: int,
    train: bool = True,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    source = DigitMNIST(data_dir, source_digit, train=train, download=download)
    target = DigitMNIST(data_dir, target_digit, train=train, download=download)
    common_kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "drop_last": train,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return DataLoader(source, **common_kwargs), DataLoader(target, **common_kwargs)

