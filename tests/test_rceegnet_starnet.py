# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch

from spd_learn.models import RCEEGNet, STaRNet, SqueezeExcitation


def test_rceegnet_default_forward():
    """Test RCEEGNet default initialization and forward pass with 3D and 4D inputs."""
    model = RCEEGNet(n_chans=22, n_outputs=2)
    # 3D input: (batch, n_chans, n_times)
    x3d = torch.randn(2, 22, 1000)
    out3d = model(x3d)
    assert out3d.shape == (2, 2)
    assert not torch.isnan(out3d).any()

    # 4D input: (batch, 1, n_chans, n_times)
    x4d = torch.randn(3, 1, 22, 500)
    out4d = model(x4d)
    assert out4d.shape == (3, 2)
    assert not torch.isnan(out4d).any()


def test_rceegnet_backward_gradient_flow():
    """Test RCEEGNet backward pass and gradient flow through all trainable layers."""
    model = RCEEGNet(
        n_chans=8,
        n_outputs=3,
        n_filters=8,
        bimap_dims=(8, 6),
        conv_channels=4,
        drop_prob=0.1,
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


@pytest.mark.parametrize("bimap_dims", [(8,), (12, 8), (16, 12, 8)])
def test_rceegnet_custom_bimap_dims(bimap_dims):
    """Test RCEEGNet with various bimap dimension configurations."""
    model = RCEEGNet(
        n_chans=10,
        n_outputs=4,
        n_filters=16,
        bimap_dims=bimap_dims,
        conv_channels=4,
    )
    x = torch.randn(2, 10, 500)
    out = model(x)
    assert out.shape == (2, 4)


def test_rceegnet_train_and_eval_modes():
    """Test RCEEGNet behaves correctly in train and eval modes."""
    model = RCEEGNet(n_chans=12, n_outputs=2)
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
    model = RCEEGNet(n_chans=8, n_outputs=2, n_filters=8, bimap_dims=(8, 6)).to(
        dtype=dtype
    )
    x = torch.randn(2, 8, 300, dtype=dtype)
    out = model(x)
    assert out.shape == (2, 2)
    assert out.dtype == dtype


def test_starnet_default_forward():
    """Test STaRNet default initialization and forward pass with 3D and 4D inputs."""
    model = STaRNet(n_chans=22, n_outputs=2)
    # 3D input: (batch, n_chans, n_times)
    x3d = torch.randn(2, 22, 1000)
    out3d = model(x3d)
    assert out3d.shape == (2, 2)
    assert not torch.isnan(out3d).any()

    # 4D input: (batch, 1, n_chans, n_times)
    x4d = torch.randn(3, 1, 22, 500)
    out4d = model(x4d)
    assert out4d.shape == (3, 2)
    assert not torch.isnan(out4d).any()


def test_starnet_backward_gradient_flow():
    """Test STaRNet backward pass and gradient flow through both branches."""
    model = STaRNet(
        n_chans=8,
        n_outputs=3,
        n_patches=2,
        temporal_filters=8,
        spatial_filters=8,
        bimap_dim=6,
        drop_prob=0.1,
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


@pytest.mark.parametrize("n_patches", [2, 4])
@pytest.mark.parametrize("bimap_dim", [8, 12])
def test_starnet_custom_configs(n_patches, bimap_dim):
    """Test STaRNet with different patches and BiMap dimensions."""
    model = STaRNet(
        n_chans=12,
        n_outputs=4,
        n_patches=n_patches,
        temporal_filters=16,
        spatial_filters=16,
        bimap_dim=bimap_dim,
    )
    x = torch.randn(2, 12, 500)
    out = model(x)
    assert out.shape == (2, 4)


def test_starnet_train_and_eval_modes():
    """Test STaRNet in train and eval modes."""
    model = STaRNet(n_chans=12, n_outputs=2)
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
        n_patches=2,
        temporal_filters=8,
        spatial_filters=8,
        bimap_dim=6,
    ).to(dtype=dtype)
    x = torch.randn(2, 8, 300, dtype=dtype)
    out = model(x)
    assert out.shape == (2, 2)
    assert out.dtype == dtype


def test_squeeze_excitation():
    """Test SqueezeExcitation module forward pass and shape preservation."""
    se = SqueezeExcitation(channels=16, reduction=4)
    x = torch.randn(2, 16, 1, 100)
    out = se(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
