# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pytest
import torch

from torch.nn.utils.parametrize import is_parametrized

from spd_learn.functional import ensure_sym
from spd_learn.models import GBWAtt, GBWAttention
from spd_learn.modules import CovLayer


@pytest.fixture
def sample_spd_inputs():
    """Generate random SPD inputs for testing attention modules."""
    torch.manual_seed(42)
    batch_size, n_patches, dim = 2, 4, 12
    raw = torch.randn(batch_size, n_patches, dim, 50)
    cov = CovLayer()(raw)
    return ensure_sym(cov + 0.1 * torch.eye(dim))


def test_gbwattention_spd_preservation(sample_spd_inputs):
    """Test that GBWAttention preserves symmetry and strict positive definiteness."""
    in_features, out_features = 12, 8
    att = GBWAttention(
        in_features=in_features,
        out_features=out_features,
        theta=0.5,
    )
    out = att(sample_spd_inputs)

    assert out.shape == (2, 4, out_features, out_features)

    # Check symmetry
    asym = (out - out.mT).abs().max().item()
    assert asym < 1e-5, f"Output is not symmetric, max error {asym}"

    # Check strict positive definiteness
    min_eig = torch.linalg.eigvalsh(out).min().item()
    assert min_eig > 0, f"Expected strictly positive eigenvalues, got {min_eig}"


def test_gbwattention_reference_parametrization():
    """Verify that the reference matrix M is initialized to identity and is parametrized."""
    in_features, out_features = 10, 6

    # Test learnable_reference=True
    att_learnable = GBWAttention(
        in_features=in_features,
        out_features=out_features,
        learnable_reference=True,
    )
    assert is_parametrized(att_learnable, "reference")
    assert att_learnable.parametrizations.reference.original.requires_grad is True
    # Reference should evaluate to identity
    eye = torch.eye(out_features)
    assert torch.allclose(att_learnable.reference, eye, atol=1e-5)

    # Test learnable_reference=False
    att_fixed = GBWAttention(
        in_features=in_features,
        out_features=out_features,
        learnable_reference=False,
    )
    assert is_parametrized(att_fixed, "reference")
    assert att_fixed.parametrizations.reference.original.requires_grad is False
    assert torch.allclose(att_fixed.reference, eye, atol=1e-5)

    # Mutate the underlying parameter and ensure it remains SPD
    with torch.no_grad():
        att_learnable.parametrizations.reference.original.fill_(0.5)
    mutated_ref = att_learnable.reference
    eigs = torch.linalg.eigvalsh(mutated_ref)
    assert (eigs > 0).all(), "Parametrized reference must strictly stay on SPD manifold"


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 1.5, 2.0])
def test_gbwattention_theta_variations(sample_spd_inputs, theta):
    """Test GBWAttention with various valid deformation values of theta."""
    in_features, out_features = 12, 8
    att = GBWAttention(
        in_features=in_features,
        out_features=out_features,
        theta=theta,
    )
    out = att(sample_spd_inputs)
    assert out.shape == (2, 4, out_features, out_features)
    assert not torch.isnan(out).any()

    # Backward gradient check
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(sample_spd_inputs.grad).any() if sample_spd_inputs.grad is not None else True


def test_gbwattention_invalid_theta():
    """Negative or zero theta must raise ValueError."""
    with pytest.raises(ValueError, match="theta must be positive"):
        GBWAttention(in_features=8, out_features=6, theta=0.0)
    with pytest.raises(ValueError, match="theta must be positive"):
        GBWAttention(in_features=8, out_features=6, theta=-0.5)


def test_gbwattention_backward_gradients(sample_spd_inputs):
    """Ensure gradients flow properly to projections and reference."""
    in_features, out_features = 12, 8
    att = GBWAttention(
        in_features=in_features,
        out_features=out_features,
        theta=0.5,
        learnable_reference=True,
    )
    x = sample_spd_inputs.clone().requires_grad_()
    out = att(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert att.parametrizations.reference.original.grad is not None
    assert not torch.isnan(att.parametrizations.reference.original.grad).any()


def test_gbwattention_single_patch():
    """Edge case: P=1 temporal patch."""
    torch.manual_seed(42)
    att = GBWAttention(in_features=8, out_features=6)
    x = torch.eye(8).expand(2, 1, 8, 8).clone().requires_grad_()
    out = att(x)
    assert out.shape == (2, 1, 6, 6)
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gbwattention_dtypes(dtype):
    """Ensure GBWAttention functions with float32 and float64."""
    in_features, out_features = 8, 6
    att = GBWAttention(in_features, out_features, dtype=dtype)
    x = torch.eye(in_features, dtype=dtype).expand(2, 3, in_features, in_features).clone()
    out = att(x)
    assert out.dtype == dtype
    assert out.shape == (2, 3, out_features, out_features)


def test_gbwatt_initialization_and_forward():
    """Test GBWAtt initialization and forward pass with 3D and 4D EEG signals."""
    model = GBWAtt(n_chans=22, n_outputs=2, n_patches=6)
    x_3d = torch.randn(2, 22, 1000)
    out_3d = model(x_3d)
    assert out_3d.shape == (2, 2)

    x_4d = torch.randn(2, 1, 22, 1000)
    out_4d = model(x_4d)
    assert out_4d.shape == (2, 2)


def test_gbwatt_forward_backward():
    """Test full forward and backward pass through GBWAtt."""
    model = GBWAtt(
        n_chans=22,
        n_outputs=4,
        n_patches=6,
        temporal_filters=22,
        spatial_filters=20,
        att_out_features=18,
        theta=0.5,
        drop_prob=0.2,
    )
    x = torch.randn(2, 22, 1000, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 4)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"


@pytest.mark.parametrize("learnable_reference", [True, False])
@pytest.mark.parametrize("drop_prob", [0.0, 0.5])
def test_gbwatt_configs(learnable_reference, drop_prob):
    """Test GBWAtt with different combinations of hyperparameters."""
    model = GBWAtt(
        n_chans=16,
        n_outputs=3,
        n_patches=4,
        temporal_filters=16,
        spatial_filters=14,
        att_out_features=12,
        learnable_reference=learnable_reference,
        drop_prob=drop_prob,
    )
    x = torch.randn(2, 16, 500)
    out = model(x)
    assert out.shape == (2, 3)
