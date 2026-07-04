import torch
from torch import nn

from sb_mnist.bridge import euler_transport, sample_brownian_bridge
from sb_mnist.models import ConditionalUNet


class ZeroTerminal(nn.Module):
    def forward(self, x_t, source, t):
        return torch.zeros_like(x_t)


def test_bridge_sample_shape():
    torch.manual_seed(0)
    source = torch.randn(3, 1, 28, 28)
    target = torch.randn(3, 1, 28, 28)
    t = torch.tensor([0.25, 0.5, 0.75])

    sample = sample_brownian_bridge(source, target, t, sigma=0.4)

    assert sample.shape == source.shape


def test_model_forward_shape():
    torch.manual_seed(0)
    model = ConditionalUNet(base_channels=8, time_dim=32)
    x_t = torch.randn(2, 1, 28, 28)
    source = torch.randn(2, 1, 28, 28)
    t = torch.rand(2)

    output = model(x_t, source, t)

    assert output.shape == x_t.shape
    assert output.min() >= -1.0
    assert output.max() <= 1.0


def test_euler_transport_shape_and_trajectory():
    torch.manual_seed(0)
    source = torch.randn(2, 1, 28, 28)
    model = ZeroTerminal()

    final, trajectory = euler_transport(
        model,
        source,
        steps=8,
        sigma=0.2,
        eta=0.0,
        return_trajectory=True,
        trajectory_points=3,
    )

    assert final.shape == source.shape
    assert len(trajectory) >= 2

