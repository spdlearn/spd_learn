# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch

from spd_learn.models import RCEEGNet, STaRNet


def test_rceegnet_default_forward():
    """Test RCEEGNet default initialization and forward pass."""
    model = RCEEGNet(n_chans=22, n_outputs=2)
    x = torch.randn(2, 22, 1000)
    out = model(x)
    assert out.shape == (2, 2)
    assert not torch.isnan(out).any()


def test_rceegnet_backward_gradient_flow():
    """Test RCEEGNet backward pass and gradient flow through all trainable layers."""
    model = RCEEGNet(
        n_chans=8,
        n_outputs=3,
        n_spatial_filters=8,
        reduced_dim=6,
    )
    x = torch.randn(2, 8, 250, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 3)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"


@pytest.mark.parametrize("reduced_dim", [4, 6, 8])
def test_rceegnet_custom_reduced_dim(reduced_dim):
    """Test RCEEGNet with various reduced_dim configurations."""
    model = RCEEGNet(
        n_chans=10,
        n_outputs=4,
        n_spatial_filters=12,
        reduced_dim=reduced_dim,
    )
    x = torch.randn(2, 10, 500)
    out = model(x)
    assert out.shape == (2, 4)


def test_rceegnet_train_and_eval_modes():
    """Test RCEEGNet behaves correctly in train and eval modes."""
    model = RCEEGNet(n_chans=12, n_outputs=2, n_spatial_filters=12, reduced_dim=6)
    x = torch.randn(4, 12, 500)

    model.train()
    out_train = model(x)
    assert out_train.shape == (4, 2)

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (4, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rceegnet_dtypes(dtype):
    """Test RCEEGNet with float32 and float64 dtypes."""
    model = RCEEGNet(n_chans=8, n_outputs=2, n_spatial_filters=8, reduced_dim=6).to(
        dtype=dtype
    )
    x = torch.randn(2, 8, 300, dtype=dtype)
    out = model(x)
    assert out.shape == (2, 2)
    assert out.dtype == dtype


def test_starnet_default_forward():
    """Test STaRNet default initialization and forward pass."""
    model = STaRNet(n_chans=22, n_outputs=2)
    x = torch.randn(2, 22, 1000)
    out = model(x)
    assert out.shape == (2, 2)
    assert not torch.isnan(out).any()


def test_starnet_backward_gradient_flow():
    """Test STaRNet backward pass and gradient flow through all branches."""
    model = STaRNet(
        n_chans=8,
        n_outputs=3,
        spatial_kernel_sizes=(2, 4, 8),
        spatial_expansion_coeff=16,
        depth_spatial_fusion=4,
        depth_temporal_map=2,
        temporal_kernel_sizes=(15, 31),
        bimap_reduced_dimension=8,
    )
    x = torch.randn(2, 8, 200, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 3)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"


@pytest.mark.parametrize("bimap_reduced_dimension", [8, 12])
def test_starnet_custom_configs(bimap_reduced_dimension):
    """Test STaRNet with different BiMap reduced dimensions."""
    model = STaRNet(
        n_chans=12,
        n_outputs=4,
        spatial_kernel_sizes=(3, 6, 12),
        bimap_reduced_dimension=bimap_reduced_dimension,
    )
    x = torch.randn(2, 12, 500)
    out = model(x)
    assert out.shape == (2, 4)


def test_starnet_train_and_eval_modes():
    """Test STaRNet in train and eval modes."""
    model = STaRNet(
        n_chans=12,
        n_outputs=2,
        spatial_kernel_sizes=(3, 6, 12),
        bimap_reduced_dimension=12,
    )
    x = torch.randn(4, 12, 500)

    model.train()
    out_train = model(x)
    assert out_train.shape == (4, 2)

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (4, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_starnet_dtypes(dtype):
    """Test STaRNet with float32 and float64 dtypes."""
    model = STaRNet(
        n_chans=8,
        n_outputs=2,
        spatial_kernel_sizes=(2, 4, 8),
        spatial_expansion_coeff=16,
        depth_spatial_fusion=4,
        depth_temporal_map=2,
        temporal_kernel_sizes=(15, 31),
        bimap_reduced_dimension=6,
    ).to(dtype=dtype)
    x = torch.randn(2, 8, 300, dtype=dtype)
    out = model(x)
    assert out.shape == (2, 2)
    assert out.dtype == dtype
