# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for correlation manifold operations, CorAtt, and CorNet."""

import pytest
import torch

from spd_learn.functional import (
    corr_lsm_exp,
    corr_lsm_log,
    corr_olm_exp,
    corr_olm_log,
    cov_to_corr,
    off_diag,
    tril_embed,
    untril_embed,
)
from spd_learn.models import CorAtt, CorNet, CorrAttention


def _make_spd(shape, dtype=torch.float32, device="cpu"):
    """Generate random SPD matrices of given shape `(..., n, n)`."""
    torch.manual_seed(42)
    A = torch.randn(*shape, dtype=dtype, device=device)
    S = A @ A.mT + torch.eye(shape[-1], dtype=dtype, device=device) * 0.1
    return S


def _make_hollow_sym(shape, dtype=torch.float32, device="cpu"):
    """Generate random hollow symmetric matrices of shape `(..., n, n)`."""
    torch.manual_seed(42)
    A = torch.randn(*shape, dtype=dtype, device=device) * 0.2
    S = 0.5 * (A + A.mT)
    return off_diag(S)


# =============================================================================
# Functional Correlation Tests
# =============================================================================


def test_cov_to_corr_properties():
    """cov_to_corr must produce unit diagonal, symmetric, SPD correlation matrices."""
    S = _make_spd((4, 5, 5))
    C = cov_to_corr(S)

    # Unit diagonal
    diag = torch.diagonal(C, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.ones_like(diag))

    # Symmetry
    torch.testing.assert_close(C, C.mT)

    # Positive definiteness
    eigvals = torch.linalg.eigvalsh(C)
    assert (eigvals > 0).all()

    # Off-diagonal range [-1, 1]
    assert (C >= -1.0 - 1e-5).all() and (C <= 1.0 + 1e-5).all()


def test_cov_to_corr_backward():
    """Gradients must flow backward through cov_to_corr."""
    S = _make_spd((2, 4, 4)).requires_grad_()
    C = cov_to_corr(S)
    loss = C.sum()
    loss.backward()
    assert S.grad is not None
    assert not torch.isnan(S.grad).any()


def test_off_diag():
    """off_diag must zero out the diagonal and preserve off-diagonals."""
    X = torch.randn(3, 4, 4)
    H = off_diag(X)

    # Diagonal is exactly 0
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.zeros_like(diag))

    # Off-diagonals are unchanged
    mask = ~torch.eye(4, dtype=torch.bool)
    torch.testing.assert_close(H[:, mask], X[:, mask])


def test_corr_olm_log_and_exp_properties():
    """OLM log and exp maps must satisfy geometric properties and roundtrip."""
    S = _make_spd((3, 4, 4))
    C = cov_to_corr(S)

    # OLM log produces hollow symmetric matrices
    H = corr_olm_log(C)
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.zeros_like(diag), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(H, H.mT)

    # OLM exp produces correlation matrices
    C_rec = corr_olm_exp(H)
    diag_c = torch.diagonal(C_rec, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag_c, torch.ones_like(diag_c))
    eigvals = torch.linalg.eigvalsh(C_rec)
    assert (eigvals > 0).all()

    # Roundtrip: log(exp(X)) == X for hollow symmetric X
    X = _make_hollow_sym((2, 4, 4))
    C_from_x = corr_olm_exp(X)
    X_rec = corr_olm_log(C_from_x)
    torch.testing.assert_close(X_rec, X, atol=1e-3, rtol=1e-3)


def test_corr_olm_backward():
    """Gradients must propagate through corr_olm_log and corr_olm_exp."""
    X = _make_hollow_sym((2, 3, 3)).requires_grad_()
    C = corr_olm_exp(X)
    loss = C.sum()
    loss.backward()
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()

    C_in = cov_to_corr(_make_spd((2, 3, 3))).requires_grad_()
    H = corr_olm_log(C_in)
    H.sum().backward()
    assert C_in.grad is not None
    assert not torch.isnan(C_in.grad).any()


def test_corr_lsm_log_and_exp_properties():
    """LSM log and exp maps must satisfy geometric properties and roundtrip."""
    S = _make_spd((3, 4, 4))
    C = cov_to_corr(S)

    # LSM log produces hollow symmetric matrices
    H = corr_lsm_log(C)
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.zeros_like(diag), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(H, H.mT)

    # LSM exp produces correlation matrices
    C_rec = corr_lsm_exp(H)
    diag_c = torch.diagonal(C_rec, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag_c, torch.ones_like(diag_c))
    eigvals = torch.linalg.eigvalsh(C_rec)
    assert (eigvals > 0).all()

    # Roundtrip: log(exp(X)) == X for hollow symmetric X
    X = _make_hollow_sym((2, 4, 4))
    C_from_x = corr_lsm_exp(X)
    X_rec = corr_lsm_log(C_from_x)
    torch.testing.assert_close(X_rec, X, atol=1e-2, rtol=1e-2)


def test_corr_lsm_backward():
    """Gradients must propagate through corr_lsm_log and corr_lsm_exp."""
    X = _make_hollow_sym((2, 3, 3)).requires_grad_()
    C = corr_lsm_exp(X)
    loss = C.sum()
    loss.backward()
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()

    C_in = cov_to_corr(_make_spd((2, 3, 3))).requires_grad_()
    H = corr_lsm_log(C_in)
    H.sum().backward()
    assert C_in.grad is not None
    assert not torch.isnan(C_in.grad).any()


def test_tril_embed_isometry_and_roundtrip():
    """tril_embed must preserve Frobenius inner product and roundtrip with untril_embed."""
    n = 5
    X = _make_hollow_sym((2, 3, n, n))
    Y = _make_hollow_sym((2, 3, n, n))

    v_x = tril_embed(X)
    v_y = tril_embed(Y)

    expected_dim = n * (n - 1) // 2
    assert v_x.shape == (2, 3, expected_dim)

    # Frobenius inner product: tr(X Y) == <tril_embed(X), tril_embed(Y)>
    inner_matrix = (X * Y).sum(dim=(-2, -1))
    inner_embed = (v_x * v_y).sum(dim=-1)
    torch.testing.assert_close(inner_embed, inner_matrix)

    # Roundtrip
    X_recovered = untril_embed(v_x, n)
    torch.testing.assert_close(X_recovered, X)


# =============================================================================
# CorrAttention Layer Tests
# =============================================================================


@pytest.mark.parametrize("metric", ["mix", "OLM", "LSM"])
def test_corr_attention_forward_backward(metric):
    """CorrAttention must return valid correlation matrices and compute gradients."""
    batch, patches, dim = 2, 4, 6
    layer = CorrAttention(dim=dim, metric=metric)

    X = _make_spd((batch, patches, dim, dim)).requires_grad_()
    out = layer(X)

    assert out.shape == (batch, patches, dim, dim)

    # Output matrices must have unit diagonal and be SPD
    diag = torch.diagonal(out, dim1=-2, dim2=-1)
    torch.testing.assert_close(diag, torch.ones_like(diag))
    eigvals = torch.linalg.eigvalsh(out)
    assert (eigvals > 0).all()

    # Backward pass
    out.sum().backward()
    assert X.grad is not None
    assert layer.weight_q.grad is not None
    assert layer.weight_k.grad is not None
    assert layer.weight_v.grad is not None
    assert layer.bias.grad is not None


def test_corr_attention_invalid_metric():
    """CorrAttention must reject invalid metric strings."""
    with pytest.raises(ValueError, match="Unknown metric"):
        CorrAttention(dim=6, metric="invalid_metric")


def test_corr_attention_device_dtype():
    """CorrAttention must support custom device and dtype."""
    layer = CorrAttention(dim=4, dtype=torch.float64, device="cpu")
    assert layer.weight_q.dtype == torch.float64
    assert layer.bias.dtype == torch.float64

    X = _make_spd((2, 3, 4, 4), dtype=torch.float64)
    out = layer(X)
    assert out.dtype == torch.float64


# =============================================================================
# CorAtt Model Tests
# =============================================================================


@pytest.mark.parametrize("metric", ["mix", "OLM", "LSM"])
def test_coratt_forward_backward(metric):
    """CorAtt forward and backward passes across all metrics."""
    batch, n_chans, n_times, n_outputs = 2, 22, 500, 2
    model = CorAtt(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_patches=4,
        temporal_filters=16,
        spatial_filters=12,
        metric=metric,
        drop_prob=0.1,
    )

    x = torch.randn(batch, n_chans, n_times, requires_grad=True)
    out = model(x)
    assert out.shape == (batch, n_outputs)

    out.sum().backward()
    assert x.grad is not None
    assert model.classifier.weight.grad is not None


def test_coratt_eval_mode():
    """CorAtt must work in evaluation mode (dropout disabled)."""
    model = CorAtt(
        n_chans=10, n_outputs=3, n_patches=3, spatial_filters=8, drop_prob=0.5
    )
    model.eval()
    x = torch.randn(2, 10, 300)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)


# =============================================================================
# CorNet Model Tests
# =============================================================================


@pytest.mark.parametrize("metric", ["OLM", "LSM"])
def test_cornet_forward_backward(metric):
    """CorNet forward and backward passes with raw time-series inputs."""
    batch, n_chans, n_times, n_outputs = 2, 22, 500, 3
    model = CorNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        bimap_dims=(16, 12),
        metric=metric,
        drop_prob=0.1,
    )

    x = torch.randn(batch, n_chans, n_times, requires_grad=True)
    out = model(x)
    assert out.shape == (batch, n_outputs)

    out.sum().backward()
    assert x.grad is not None
    assert model.classifier.weight.grad is not None


def test_cornet_cov_input():
    """CorNet must accept precomputed covariance matrices as inputs."""
    batch, n_chans, n_outputs = 2, 16, 2
    model = CorNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        bimap_dims=(10, 8),
        input_type="cov",
    )

    cov = _make_spd((batch, n_chans, n_chans))
    with torch.no_grad():
        out = model(cov)
    assert out.shape == (batch, n_outputs)


def test_cornet_empty_bimap_dims():
    """CorNet must function correctly when bimap_dims is empty."""
    model = CorNet(n_chans=8, n_outputs=2, bimap_dims=())
    x = torch.randn(2, 8, 200)
    out = model(x)
    assert out.shape == (2, 2)


def test_cornet_invalid_metric():
    """CorNet must reject invalid metric strings."""
    with pytest.raises(ValueError, match="Unknown metric"):
        CorNet(n_chans=8, metric="invalid")
