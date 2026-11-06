# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from spd_learn.models import (
    SPDDecoderBlock,
    SPDNetAE,
    SSGCNet,
    TangentGraphSPDAggregation,
)


# =============================================================================
# Tests for SPDDecoderBlock
# =============================================================================


def test_spd_decoder_block_shape_and_spd():
    """Test SPDDecoderBlock outputs correct shape and strictly positive definite matrices."""
    in_features, out_features = 8, 16
    dec = SPDDecoderBlock(in_features=in_features, out_features=out_features, eps=1e-4)

    # Input SPD batch
    B = 3
    A = torch.randn(B, in_features, in_features)
    Z = A @ A.mT + 0.1 * torch.eye(in_features)

    Y = dec(Z)
    assert Y.shape == (B, out_features, out_features)

    # Check symmetry
    assert torch.allclose(Y, Y.mT, atol=1e-5)

    # Check positive definiteness
    eigvals = torch.linalg.eigvalsh(Y)
    assert (eigvals > 0).all()
    # Check that minimum eigenvalues are on the order of eps
    assert (eigvals >= 0.9 * 1e-4).all()


def test_spd_decoder_block_stiefel():
    """Test that SPDDecoderBlock weight satisfies the Stiefel constraint W^T W = I."""
    dec = SPDDecoderBlock(in_features=8, out_features=16)
    W = dec.bimap.weight.squeeze(0)
    WtW = W.mT @ W
    eye = torch.eye(W.shape[-1], dtype=W.dtype, device=W.device)
    assert torch.allclose(WtW, eye, atol=1e-5)


def test_spd_decoder_block_broadcasting():
    """Test SPDDecoderBlock with multiple leading batch dimensions."""
    dec = SPDDecoderBlock(in_features=4, out_features=10)
    Z = torch.eye(4).expand(2, 3, 4, 4).clone()
    Y = dec(Z)
    assert Y.shape == (2, 3, 10, 10)


def test_spd_decoder_block_backward():
    """Test gradients flow through SPDDecoderBlock to input and parameters."""
    dec = SPDDecoderBlock(in_features=4, out_features=8)
    A = torch.randn(2, 4, 4, requires_grad=True)
    Z = A @ A.mT + 0.2 * torch.eye(4)
    Y = dec(Z)
    loss = Y.sum()
    loss.backward()

    assert A.grad is not None
    original_param = dec.bimap.get_parameter("parametrizations.weight.original")
    assert original_param.grad is not None


# =============================================================================
# Tests for SPDNetAE
# =============================================================================


@pytest.mark.parametrize("bimap_dims", [(16, 8), (12,), (16, 12, 6)])
def test_spdnet_ae_initialization(bimap_dims):
    """Test SPDNetAE initialization with various encoder architectures."""
    model = SPDNetAE(
        n_chans=20,
        n_outputs=3,
        bimap_dims=bimap_dims,
        eps=1e-4,
        drop_prob=0.1,
    )
    assert model.bimap_dims == tuple(bimap_dims)
    assert model.n_chans == 20
    assert model.n_outputs == 3


def test_spdnet_ae_encode_decode_autoencode():
    """Test encode, decode, and autoencode routines on raw EEG signals and covariance inputs."""
    model = SPDNetAE(n_chans=22, n_outputs=2, bimap_dims=(16, 8))

    # 1. From raw EEG signals
    x_raw = torch.randn(3, 22, 500)
    Z_from_raw = model.encode(x_raw)
    assert Z_from_raw.shape == (3, 8, 8)

    # Check that latent Z is SPD
    assert torch.allclose(Z_from_raw, Z_from_raw.mT, atol=1e-5)
    eigvals_z = torch.linalg.eigvalsh(Z_from_raw)
    assert (eigvals_z > 0).all()

    # Reconstruct from latent
    S_recon_from_raw = model.decode(Z_from_raw)
    assert S_recon_from_raw.shape == (3, 22, 22)
    assert torch.allclose(S_recon_from_raw, S_recon_from_raw.mT, atol=1e-5)
    eigvals_recon = torch.linalg.eigvalsh(S_recon_from_raw)
    assert (eigvals_recon > 0).all()

    # Check autoencode matches decode(encode(x))
    auto_recon = model.autoencode(x_raw)
    assert torch.allclose(auto_recon, S_recon_from_raw, atol=1e-5)

    # 2. From SPD covariance matrix input
    S_cov = model.cov_frontend(x_raw)
    Z_from_cov = model.encode(S_cov)
    assert Z_from_cov.shape == (3, 8, 8)
    S_recon_from_cov = model.decode(Z_from_cov)
    assert S_recon_from_cov.shape == (3, 22, 22)


def test_spdnet_ae_reconstruction_loss():
    """Test reconstruction loss value, reductions, and identity properties."""
    model = SPDNetAE(n_chans=10, n_outputs=2, bimap_dims=(8, 4))
    x_raw = torch.randn(4, 10, 300)
    S_in = model.cov_frontend(x_raw)
    S_recon = model.autoencode(S_in)

    # Mean reduction
    loss_mean = model.reconstruction_loss(S_in, S_recon, reduction="mean")
    assert loss_mean.ndim == 0
    assert loss_mean.item() >= 0.0

    # Sum reduction
    loss_sum = model.reconstruction_loss(S_in, S_recon, reduction="sum")
    assert torch.allclose(loss_sum, loss_mean * 4, rtol=1e-4)

    # None reduction
    loss_none = model.reconstruction_loss(S_in, S_recon, reduction="none")
    assert loss_none.shape == (4,)
    assert torch.allclose(loss_none.mean(), loss_mean, rtol=1e-4)

    # Reconstruction loss on identical matrices must be 0
    loss_zero = model.reconstruction_loss(S_in, S_in, reduction="mean")
    assert torch.allclose(loss_zero, torch.tensor(0.0), atol=1e-6)

    # Invalid reduction raises ValueError
    with pytest.raises(ValueError, match="Unknown reduction"):
        model.reconstruction_loss(S_in, S_recon, reduction="invalid")


def test_spdnet_ae_forward_and_backward():
    """Test end-to-end forward classification and reconstruction loss backpropagation."""
    model = SPDNetAE(n_chans=12, n_outputs=3, bimap_dims=(8, 4))
    x = torch.randn(2, 12, 400, requires_grad=True)

    # Classifier forward & backward
    logits = model(x)
    assert logits.shape == (2, 3)
    loss_cls = logits.sum()
    loss_cls.backward()
    assert x.grad is not None

    # Reconstruction loss backward
    x2 = torch.randn(2, 12, 400, requires_grad=True)
    S_in = model.cov_frontend(x2)
    S_recon = model.autoencode(S_in)
    loss_rec = model.reconstruction_loss(S_in, S_recon)
    loss_rec.backward()
    assert x2.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spdnet_ae_dtypes(dtype):
    """Test SPDNetAE with single and double precision floats."""
    model = SPDNetAE(n_chans=8, n_outputs=2, bimap_dims=(6, 4)).to(dtype)
    x = torch.randn(2, 8, 200, dtype=dtype)
    out = model(x)
    assert out.dtype == dtype
    assert out.shape == (2, 2)


def test_spdnet_ae_input_type_cov():
    """Test SPDNetAE with precomputed covariance input."""
    model = SPDNetAE(n_chans=8, n_outputs=2, bimap_dims=(6, 4), input_type="cov")
    A = torch.randn(2, 8, 8)
    cov = A @ A.mT + 0.1 * torch.eye(8)
    out = model(cov)
    assert out.shape == (2, 2)
    Z = model.encode(cov)
    assert Z.shape == (2, 4, 4)
    recon = model.decode(Z)
    assert recon.shape == (2, 8, 8)


# =============================================================================
# Tests for TangentGraphSPDAggregation
# =============================================================================


def test_tangent_graph_spd_aggregation_basic():
    """Test tangent-space graph aggregation preserves SPD structure and attention."""
    B, V, d = 3, 4, 8
    agg = TangentGraphSPDAggregation(k_neighbors=2)

    A = torch.randn(B, V, d, d)
    S = A @ A.mT + 0.2 * torch.eye(d)

    S_out = agg(S)
    assert S_out.shape == (B, V, d, d)

    # Must be symmetric
    assert torch.allclose(S_out, S_out.mT, atol=1e-5)

    # Must be positive definite
    eigvals = torch.linalg.eigvalsh(S_out)
    assert (eigvals > 0).all()


def test_tangent_graph_spd_aggregation_broadcasting_and_backward():
    """Test tangent aggregation with leading batch dimensions and gradient flow."""
    agg = TangentGraphSPDAggregation(k_neighbors=2)
    A = torch.randn(2, 3, 4, 6, 6, requires_grad=True)
    S = A @ A.mT + 0.2 * torch.eye(6)

    S_out = agg(S)
    assert S_out.shape == (2, 3, 4, 6, 6)

    loss = S_out.sum()
    loss.backward()
    assert A.grad is not None


def test_tangent_graph_spd_aggregation_k_clamping():
    """Test tangent aggregation behaves gracefully when k_neighbors >= V."""
    V, d = 3, 6
    agg = TangentGraphSPDAggregation(k_neighbors=10)
    A = torch.randn(2, V, d, d)
    S = A @ A.mT + 0.1 * torch.eye(d)

    S_out = agg(S)
    assert S_out.shape == (2, V, d, d)


# =============================================================================
# Tests for SSGCNet
# =============================================================================


@pytest.mark.parametrize("n_nodes", [2, 4])
@pytest.mark.parametrize("stiefel_dim", [8, 12])
def test_ssgcnet_initialization_and_forward(n_nodes, stiefel_dim):
    """Test SSGCNet forward pass with different node counts and Stiefel dimensions."""
    model = SSGCNet(
        n_chans=16,
        n_outputs=3,
        n_nodes=n_nodes,
        temporal_filters=14,
        stiefel_dim=stiefel_dim,
        k_neighbors=2,
    )
    x = torch.randn(2, 16, 600)
    out = model(x)
    assert out.shape == (2, 3)

    # Also test 4D input (batch, 1, channels, times)
    x_4d = x.unsqueeze(1)
    out_4d = model(x_4d)
    assert out_4d.shape == (2, 3)


def test_ssgcnet_backward():
    """Test that gradients flow end-to-end through all components of SSGCNet."""
    model = SSGCNet(
        n_chans=10,
        n_outputs=2,
        n_nodes=3,
        temporal_filters=8,
        stiefel_dim=6,
        k_neighbors=2,
    )
    x = torch.randn(2, 10, 450, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert model.temp_conv.weight.grad is not None
    assert model.spat_conv.weight.grad is not None
    assert model.classifier.weight.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ssgcnet_dtypes(dtype):
    """Test SSGCNet with float32 and float64 dtypes."""
    model = SSGCNet(
        n_chans=8,
        n_outputs=2,
        n_nodes=2,
        temporal_filters=6,
        stiefel_dim=4,
    ).to(dtype)
    x = torch.randn(2, 8, 300, dtype=dtype)
    out = model(x)
    assert out.dtype == dtype
    assert out.shape == (2, 2)
