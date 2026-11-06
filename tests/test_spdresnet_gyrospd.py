# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch

from spd_learn.models import (
    AdaptiveLogEuclidean,
    GyroSPD,
    GyroSPDLayer,
    SPDResBlock,
    SPDResNet,
)


def _generate_spd_matrices(batch_shape: tuple[int, ...], dim: int) -> torch.Tensor:
    """Helper to generate random SPD matrices."""
    A = torch.randn(*batch_shape, dim, dim)
    spd = A @ A.mT + 0.1 * torch.eye(dim)
    return spd


# =============================================================================
# AdaptiveLogEuclidean Tests
# =============================================================================


def test_adaptive_log_euclidean_tangent():
    """Test AdaptiveLogEuclidean returning symmetric tangent matrices."""
    alem = AdaptiveLogEuclidean(num_features=6, return_spd=False)
    X = _generate_spd_matrices((4,), 6)
    X.requires_grad_(True)

    out = alem(X)
    assert out.shape == (4, 6, 6)
    # Check exact symmetry
    torch.testing.assert_close(out, out.mT, atol=1e-5, rtol=1e-5)

    loss = out.sum()
    loss.backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    assert alem.p_raw.grad is not None and torch.isfinite(alem.p_raw.grad).all()
    assert alem.weight.grad is not None and torch.isfinite(alem.weight.grad).all()
    assert alem.bias.grad is not None and torch.isfinite(alem.bias.grad).all()


def test_adaptive_log_euclidean_spd():
    """Test AdaptiveLogEuclidean returning strictly SPD matrices."""
    alem = AdaptiveLogEuclidean(num_features=6, return_spd=True)
    X = _generate_spd_matrices((4,), 6)
    X.requires_grad_(True)

    out = alem(X)
    assert out.shape == (4, 6, 6)
    # Check exact symmetry
    torch.testing.assert_close(out, out.mT, atol=1e-5, rtol=1e-5)
    # Check positive definiteness
    eigvals = torch.linalg.eigvalsh(out)
    assert (eigvals > 0).all()

    loss = out.sum()
    loss.backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()


def test_adaptive_log_euclidean_reset_parameters():
    """Test reset_parameters initializes weights to identity-like."""
    alem = AdaptiveLogEuclidean(num_features=4)
    with torch.no_grad():
        alem.p_raw.fill_(2.0)
        alem.weight.fill_(5.0)
        alem.bias.fill_(3.0)
    alem.reset_parameters()
    assert torch.allclose(alem.p_raw, torch.zeros(1))
    assert torch.allclose(alem.weight, torch.ones(4))
    assert torch.allclose(alem.bias, torch.zeros(4))


def test_adaptive_log_euclidean_multidim():
    """Test AdaptiveLogEuclidean with arbitrary batch dimensions."""
    alem = AdaptiveLogEuclidean(num_features=5, return_spd=False)
    X = _generate_spd_matrices((2, 3), 5)
    out = alem(X)
    assert out.shape == (2, 3, 5, 5)


# =============================================================================
# SPDResBlock Tests
# =============================================================================


def test_spd_res_block_forward_backward():
    """Test SPDResBlock preserves SPD property and computes gradients."""
    block = SPDResBlock(num_features=8, hidden_channels=16)
    X = _generate_spd_matrices((3,), 8)
    X.requires_grad_(True)

    out = block(X)
    assert out.shape == (3, 8, 8)
    # Check exact symmetry
    torch.testing.assert_close(out, out.mT, atol=1e-5, rtol=1e-5)
    # Check positive definiteness
    eigvals = torch.linalg.eigvalsh(out)
    assert (eigvals > 0).all()

    loss = out.sum()
    loss.backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_spd_res_block_multidim():
    """Test SPDResBlock with higher-dimensional batches."""
    block = SPDResBlock(num_features=6)
    X = _generate_spd_matrices((2, 3), 6)
    out = block(X)
    assert out.shape == (2, 3, 6, 6)
    eigvals = torch.linalg.eigvalsh(out)
    assert (eigvals > 0).all()


# =============================================================================
# SPDResNet Tests
# =============================================================================


@pytest.mark.parametrize("use_alem", [True, False])
@pytest.mark.parametrize("num_res_blocks", [0, 1, 2])
def test_spdresnet_configurations(use_alem, num_res_blocks):
    """Test SPDResNet with combinations of use_alem and num_res_blocks."""
    model = SPDResNet(
        n_chans=10,
        n_outputs=3,
        bimap_dims=(8, 6),
        num_res_blocks=num_res_blocks,
        use_alem=use_alem,
    )
    # Raw EEG input
    x_raw = torch.randn(2, 10, 200)
    out = model(x_raw)
    assert out.shape == (2, 3)

    # Precomputed covariance input
    x_cov = _generate_spd_matrices((2,), 10)
    out_cov = model(x_cov)
    assert out_cov.shape == (2, 3)


def test_spdresnet_backward():
    """Test gradient flow across the entire SPDResNet model."""
    model = SPDResNet(n_chans=8, n_outputs=2, bimap_dims=(6, 4), num_res_blocks=1)
    x = torch.randn(3, 8, 300, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_spdresnet_single_sample_batch():
    """Test SPDResNet handles a batch of size 1."""
    model = SPDResNet(n_chans=8, n_outputs=2, bimap_dims=(6, 4))
    x = torch.randn(1, 8, 200)
    out = model(x)
    assert out.shape == (1, 2)


# =============================================================================
# GyroSPDLayer Tests
# =============================================================================


def test_gyrospd_layer_forward_backward():
    """Test GyroSPDLayer applies left gyro-translation and supports gradients."""
    layer = GyroSPDLayer(in_features=8, out_features=5)
    X = _generate_spd_matrices((3,), 8)
    X.requires_grad_(True)

    out = layer(X)
    assert out.shape == (3, 5, 5)
    # Check exact symmetry
    torch.testing.assert_close(out, out.mT, atol=1e-5, rtol=1e-5)
    # Check strictly positive definite
    eigvals = torch.linalg.eigvalsh(out)
    assert (eigvals > 0).all()

    loss = out.sum()
    loss.backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    assert (
        layer.bimap.parametrizations.weight.original.grad is not None
        and torch.isfinite(layer.bimap.parametrizations.weight.original.grad).all()
    )
    assert (
        layer.parametrizations.bias.original.grad is not None
        and torch.isfinite(layer.parametrizations.bias.original.grad).all()
    )


def test_gyrospd_layer_reset_parameters():
    """Test reset_parameters resets bias to identity and bimap parameters."""
    layer = GyroSPDLayer(in_features=6, out_features=4)
    with torch.no_grad():
        layer.parametrizations.bias.original.fill_(1.5)
    layer.reset_parameters()
    torch.testing.assert_close(layer.bias, torch.eye(4), atol=1e-5, rtol=1e-5)


# =============================================================================
# GyroSPD Tests
# =============================================================================


@pytest.mark.parametrize("drop_prob", [0.0, 0.2])
def test_gyrospd_forward_eval(drop_prob):
    """Test GyroSPD in train and eval modes with raw and covariance inputs."""
    model = GyroSPD(n_chans=10, n_outputs=2, bimap_dims=(8, 6), drop_prob=drop_prob)

    # Raw signal
    x_raw = torch.randn(4, 10, 200)
    out_train = model(x_raw)
    assert out_train.shape == (4, 2)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x_raw)
    assert out_eval.shape == (4, 2)

    # Precomputed covariance
    x_cov = _generate_spd_matrices((4,), 10)
    with torch.no_grad():
        out_cov = model(x_cov)
    assert out_cov.shape == (4, 2)


def test_gyrospd_backward():
    """Test gradient propagation across the entire GyroSPD architecture."""
    model = GyroSPD(n_chans=8, n_outputs=2, bimap_dims=(6, 4))
    x = torch.randn(3, 8, 250, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No grad for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_spdresnet_gyrospd_normalized_params():
    """Test SPDResNet and GyroSPD with input_type='cov', upper=False, and custom threshold."""
    # SPDResNet with input_type='cov' and upper=False (with and without alem)
    x_cov = _generate_spd_matrices((2,), 8)
    for use_alem in [True, False]:
        model_res = SPDResNet(
            n_chans=8,
            n_outputs=2,
            bimap_dims=(6, 4),
            use_alem=use_alem,
            input_type="cov",
            upper=False,
            threshold=1e-3,
        )
        out_res = model_res(x_cov)
        assert out_res.shape == (2, 2)

    # GyroSPD with input_type='cov' and upper=False
    model_gyro = GyroSPD(
        n_chans=8,
        n_outputs=2,
        bimap_dims=(6, 4),
        input_type="cov",
        upper=False,
        threshold=1e-3,
    )
    out_gyro = model_gyro(x_cov)
    assert out_gyro.shape == (2, 2)

    # Invalid input_type
    with pytest.raises(ValueError, match="Unknown input_type"):
        SPDResNet(input_type="invalid")
    with pytest.raises(ValueError, match="Unknown input_type"):
        GyroSPD(input_type="invalid")
