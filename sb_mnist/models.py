from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        torch.linspace(
            0.0,
            math.log(10000.0),
            half,
            device=t.device,
            dtype=t.dtype,
        )
    )
    args = t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(t, self.dim))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    """Small conditional U-Net that predicts the target endpoint x_1."""

    def __init__(self, base_channels: int = 48, time_dim: int = 128) -> None:
        super().__init__()
        c = base_channels
        self.time = TimeMLP(time_dim)
        self.in_conv = nn.Conv2d(2, c, kernel_size=3, padding=1)
        self.block1 = ResBlock(c, c, time_dim)
        self.down1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.block2 = ResBlock(c * 2, c * 2, time_dim)
        self.down2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.mid1 = ResBlock(c * 4, c * 4, time_dim)
        self.mid2 = ResBlock(c * 4, c * 4, time_dim)
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.up_block1 = ResBlock(c * 4, c * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up_block2 = ResBlock(c * 2, c, time_dim)
        self.out_norm = nn.GroupNorm(_groups(c), c)
        self.out = nn.Conv2d(c, 1, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, source: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.flatten()
        time_emb = self.time(t.to(dtype=x_t.dtype))
        h = self.in_conv(torch.cat([x_t, source], dim=1))
        h1 = self.block1(h, time_emb)
        h = self.down1(h1)
        h2 = self.block2(h, time_emb)
        h = self.down2(h2)
        h = self.mid1(h, time_emb)
        h = self.mid2(h, time_emb)
        h = self.up1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up_block1(h, time_emb)
        h = self.up2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up_block2(h, time_emb)
        return torch.tanh(self.out(F.silu(self.out_norm(h))))

