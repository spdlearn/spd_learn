# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for GaussianSPD and RHOP.

Verifies the mathematical properties and empirical behavior of RHOP (ICLR 2026):
  - Strict SPD-ness (positive eigenvalues, Hermitian symmetry) of GaussianSPD
  - Riemannian mappings: LEM, PEM, and LCM
  - Forward pass with and without class token (including token shape squeezing)
  - Full end-to-end gradient flow through all learnable parameters and input tokens
  - Optimization dynamics in a mock classifier head
"""

import pytest
import torch

from torch import nn

from spd_learn.modules import RHOP, GaussianSPD


# ---------------------------------------------------------------------------
# GaussianSPD Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(8, 6), (2, 10, 8), (2, 3, 5, 4)])
@pytest.mark.parametrize("embedding_type", ["quotient", "lovric"])
def test_gaussian_spd_descriptor_shape_and_spdness(shape, embedding_type):
    """Verify that GaussianSPD produces symmetric positive definite matrices."""
    torch.manual_seed(42)
    layer = GaussianSPD(embedding_type=embedding_type, alpha=1.2, beta=0.8, eps=1e-5)
    x = torch.randn(*shape, dtype=torch.float64)

    G = layer(x)

    # Check output shape: (..., D + 1, D + 1)
    d = shape[-1]
    expected_shape = (*shape[:-2], d + 1, d + 1)
    assert G.shape == expected_shape

    # Check exact symmetry
    diff = G - G.transpose(-2, -1)
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-7)

    # Check positive definiteness (all eigenvalues > 0)
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals > 0).all(), (
        f"Found non-positive eigenvalues: {eigvals.min().item()}"
    )


def test_gaussian_spd_descriptor_single_token():
    """Verify descriptor behavior when sequence length T = 1."""
    layer = GaussianSPD(eps=1e-4)
    x = torch.randn(2, 1, 5, dtype=torch.float64)
    G = layer(x)

    assert G.shape == (2, 6, 6)
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals > 0).all()


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_gaussian_spd_descriptor_complex(dtype):
    """Verify that complex inputs produce Hermitian positive definite descriptors."""
    layer = GaussianSPD(eps=1e-4)
    x = torch.randn(3, 8, 4, dtype=dtype)
    G = layer(x)

    assert G.shape == (3, 5, 5)
    assert G.dtype == dtype

    # Hermitian check: G == G.conj().T
    diff = G - G.conj().transpose(-2, -1)
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-6)

    # All real eigenvalues must be strictly positive
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals > 0).all()


def test_gaussian_spd_descriptor_invalid_args():
    """Verify that invalid arguments raise informative ValueError."""
    with pytest.raises(ValueError, match="Unsupported embedding_type"):
        GaussianSPD(embedding_type="unsupported")

    with pytest.raises(ValueError, match="beta must be positive"):
        GaussianSPD(beta=-1.0)

    with pytest.raises(ValueError, match="alpha must be non-zero"):
        GaussianSPD(alpha=0.0)


# ---------------------------------------------------------------------------
# RHOP Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["LEM", "PEM", "LCM"])
def test_rhop_metrics_forward(metric):
    """Verify forward pass of RHOP for LEM, PEM, and LCM."""
    torch.manual_seed(42)
    in_features = 16
    reduced_dim = 6
    batch_size = 4
    n_tokens = 10

    rhop = RHOP(
        in_features=in_features,
        reduced_dim=reduced_dim,
        metric=metric,
        power=0.5,
    )
    x = torch.randn(batch_size, n_tokens, in_features)
    out = rhop(x)

    assert out.shape == (batch_size, in_features)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_rhop_with_and_without_cls_token():
    """Verify fusion with and without class token."""
    torch.manual_seed(42)
    in_features = 12
    batch_size = 3
    n_tokens = 8

    rhop = RHOP(in_features=in_features, reduced_dim=4)
    x = torch.randn(batch_size, n_tokens, in_features)

    # 1. Forward without cls_token (uses token average)
    out_no_cls = rhop(x)
    assert out_no_cls.shape == (batch_size, in_features)

    # 2. Forward with 2D cls_token (batch_size, in_features)
    cls_token_2d = torch.randn(batch_size, in_features)
    out_2d = rhop(x, cls_token=cls_token_2d)
    assert out_2d.shape == (batch_size, in_features)
    assert not torch.allclose(out_no_cls, out_2d)

    # 3. Forward with 3D cls_token (batch_size, 1, in_features)
    cls_token_3d = cls_token_2d.unsqueeze(1)
    out_3d = rhop(x, cls_token=cls_token_3d)
    assert out_3d.shape == (batch_size, in_features)
    assert torch.allclose(out_2d, out_3d)


@pytest.mark.parametrize("metric", ["LEM", "PEM", "LCM"])
def test_rhop_gradient_flow(metric):
    """Verify end-to-end gradient propagation through RHOP."""
    torch.manual_seed(42)
    in_features = 8
    reduced_dim = 4
    batch_size = 2
    n_tokens = 6

    rhop = RHOP(
        in_features=in_features,
        reduced_dim=reduced_dim,
        metric=metric,
        power=0.5,
        dtype=torch.float64,
    )

    x = torch.randn(
        batch_size, n_tokens, in_features, dtype=torch.float64, requires_grad=True
    )
    cls_token = torch.randn(
        batch_size, in_features, dtype=torch.float64, requires_grad=True
    )

    out = rhop(x, cls_token=cls_token)
    loss = (out**2).sum()
    loss.backward()

    # Verify input gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert x.grad.norm().item() > 0

    assert cls_token.grad is not None
    assert not torch.isnan(cls_token.grad).any()
    assert cls_token.grad.norm().item() > 0

    # Verify parameter gradients
    assert rhop.down_proj.weight.grad is not None
    assert rhop.down_proj.bias.grad is not None
    assert rhop.up_proj.weight.grad is not None
    assert rhop.up_proj.bias.grad is not None

    for param in rhop.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert param.grad.norm().item() > 0


def test_rhop_invalid_metric():
    """Verify that invalid metric or power raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported metric"):
        RHOP(in_features=16, metric="INVALID")

    with pytest.raises(ValueError, match="power must be non-zero"):
        RHOP(in_features=16, metric="PEM", power=0.0)


def test_rhop_mock_training():
    """Verify that RHOP can be trained end-to-end in a mock downstream classification task."""
    torch.manual_seed(42)
    in_features = 12
    n_classes = 3

    model = nn.Sequential(
        RHOP(in_features=in_features, reduced_dim=4, metric="LEM"),
        nn.Linear(in_features, n_classes),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, 10, in_features)
    labels = torch.randint(0, n_classes, (8,))

    initial_loss = None
    final_loss = None
    for step in range(10):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss
