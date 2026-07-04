import torch

from sb_mnist.sinkhorn import log_sinkhorn, sinkhorn_match_targets


def test_log_sinkhorn_has_uniform_marginals():
    torch.manual_seed(0)
    cost = torch.rand(5, 7)
    plan = log_sinkhorn(cost, epsilon=0.3, n_iters=200)

    assert torch.allclose(plan.sum(dim=1), torch.full((5,), 1 / 5), atol=2e-4)
    assert torch.allclose(plan.sum(dim=0), torch.full((7,), 1 / 7), atol=2e-4)


def test_sinkhorn_match_returns_batch_shape():
    torch.manual_seed(1)
    source = torch.randn(4, 1, 28, 28)
    target = torch.randn(4, 1, 28, 28)
    matched, stats = sinkhorn_match_targets(source, target, epsilon=0.5, n_iters=40)

    assert matched.shape == source.shape
    assert "sinkhorn_cost" in stats
    assert "matched_mse" in stats

