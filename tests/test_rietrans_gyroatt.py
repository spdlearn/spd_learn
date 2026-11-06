# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for RieTrans (SPDRMLR, SPDLinear, SPNN) and GyroAtt (GyroSPDAtt)."""

import pytest
import torch
import torch.nn as nn

from spd_learn.models import (
    SPNN,
    GyroSPDAtt,
    GyroSPDAttention,
    SymSPDAttention,
)
from spd_learn.modules import SPDRMLR, SPDLinear


def _generate_spd_batch(*shape: int, dtype=torch.float32) -> torch.Tensor:
    """Generate a batch of well-conditioned SPD matrices."""
    torch.manual_seed(42)
    A = torch.randn(*shape, dtype=dtype)
    n = shape[-1]
    return A @ A.mT + 0.2 * torch.eye(n, dtype=dtype)


# =============================================================================
# SPDRMLR Tests
# =============================================================================


class TestSPDRMLR:
    """Tests for Riemannian Multinomial Logistic Regression layer."""

    @pytest.mark.parametrize("metric", ["LEM", "LCM", "AIM", "PEM"])
    def test_forward_metrics(self, metric: str):
        batch, n_features, n_outputs = 4, 3, 2
        X = _generate_spd_batch(batch, n_features, n_features)
        layer = SPDRMLR(n_features=n_features, n_outputs=n_outputs, metric=metric)
        out = layer(X)
        assert out.shape == (batch, n_outputs)
        assert torch.isfinite(out).all()

    def test_pem_theta_zero(self):
        batch, n_features, n_outputs = 3, 3, 2
        X = _generate_spd_batch(batch, n_features, n_features)
        layer = SPDRMLR(
            n_features=n_features, n_outputs=n_outputs, metric="PEM", theta=0.0
        )
        out = layer(X)
        assert out.shape == (batch, n_outputs)
        assert torch.isfinite(out).all()

    def test_extra_batch_dims(self):
        X = _generate_spd_batch(2, 3, 4, 3, 3)
        layer = SPDRMLR(n_features=3, n_outputs=5, metric="LEM")
        out = layer(X)
        assert out.shape == (2, 3, 4, 5)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            SPDRMLR(n_features=3, n_outputs=2, metric="INVALID")

    def test_input_tangent_matrix(self):
        batch, n_features, n_outputs = 4, 3, 2
        V = torch.randn(batch, n_features, n_features)
        V = (V + V.mT) / 2
        layer = SPDRMLR(n_features=n_features, n_outputs=n_outputs, input_tangent=True)
        out = layer(V)
        assert out.shape == (batch, n_outputs)

    def test_input_tangent_vector(self):
        batch, n_features, n_outputs = 4, 3, 2
        vec_dim = n_features * (n_features + 1) // 2
        v = torch.randn(batch, vec_dim)
        layer = SPDRMLR(n_features=n_features, n_outputs=n_outputs, input_tangent=True)
        out = layer(v)
        assert out.shape == (batch, n_outputs)

    def test_backward_gradient_flow(self):
        X = _generate_spd_batch(3, 4, 4)
        layer = SPDRMLR(n_features=4, n_outputs=2, metric="LEM")
        out = layer(X)
        loss = out.sum()
        loss.backward()

        assert layer.linear.weight.grad is not None
        assert torch.isfinite(layer.linear.weight.grad).all()
        # Original parameter of parametrized reference must have finite gradients
        ref_orig = layer.parametrizations.reference.original
        assert ref_orig.grad is not None
        assert torch.isfinite(ref_orig.grad).all()

    def test_reset_parameters(self):
        layer = SPDRMLR(n_features=3, n_outputs=2)
        layer.reset_parameters()
        assert torch.allclose(layer.reference, torch.eye(3), atol=1e-5)

    def test_dtype_support(self):
        X = _generate_spd_batch(2, 3, 3, dtype=torch.float64)
        layer = SPDRMLR(n_features=3, n_outputs=2, dtype=torch.float64)
        out = layer(X)
        assert out.dtype == torch.float64


# =============================================================================
# SPDLinear Tests
# =============================================================================


class TestSPDLinear:
    """Tests for Intrinsic SPD Transformation Layer."""

    @pytest.mark.parametrize("metric", ["LEM", "LCM", "AIM", "PEM"])
    def test_forward_dimension_reduction(self, metric: str):
        batch, in_features, out_features = 4, 4, 3
        X = _generate_spd_batch(batch, in_features, in_features)
        layer = SPDLinear(
            in_features=in_features, out_features=out_features, metric=metric
        )
        out = layer(X)
        assert out.shape == (batch, out_features, out_features)
        # Check output is strictly SPD
        assert torch.allclose(out, out.mT, atol=1e-5)
        evals = torch.linalg.eigvalsh(out)
        assert (evals > 0).all()

    def test_forward_dimension_expansion(self):
        batch, in_features, out_features = 4, 2, 4
        X = _generate_spd_batch(batch, in_features, in_features)
        layer = SPDLinear(
            in_features=in_features, out_features=out_features, metric="LEM"
        )
        out = layer(X)
        assert out.shape == (batch, out_features, out_features)
        evals = torch.linalg.eigvalsh(out)
        assert (evals > 0).all()

    def test_return_tangent(self):
        batch, in_features, out_features = 3, 4, 3
        X = _generate_spd_batch(batch, in_features, in_features)
        layer = SPDLinear(
            in_features=in_features,
            out_features=out_features,
            return_tangent=True,
        )
        out = layer(X)
        assert out.shape == (batch, out_features, out_features)
        # Tangent matrices must be symmetric
        assert torch.allclose(out, out.mT, atol=1e-5)

    def test_extra_batch_dims(self):
        X = _generate_spd_batch(2, 3, 4, 4)
        layer = SPDLinear(in_features=4, out_features=2)
        out = layer(X)
        assert out.shape == (2, 3, 2, 2)

    def test_backward_gradient_flow(self):
        X = _generate_spd_batch(3, 4, 4)
        layer = SPDLinear(in_features=4, out_features=3, metric="LEM")
        out = layer(X)
        loss = out.sum()
        loss.backward()
        assert layer.rmlr.linear.weight.grad is not None
        assert torch.isfinite(layer.rmlr.linear.weight.grad).all()

    def test_extra_repr(self):
        layer = SPDLinear(in_features=4, out_features=3, metric="LEM")
        rep = repr(layer)
        assert "in_features=4" in rep
        assert "out_features=3" in rep


# =============================================================================
# SPNN Model Tests
# =============================================================================


class TestSPNN:
    """Tests for the complete SPNN model."""

    def test_raw_eeg_forward(self):
        batch, n_chans, n_times, n_outputs = 3, 22, 500, 2
        x = torch.randn(batch, n_chans, n_times)
        model = SPNN(n_chans=n_chans, n_outputs=n_outputs, hidden_dims=(16, 12))
        out = model(x)
        assert out.shape == (batch, n_outputs)
        assert torch.isfinite(out).all()

    def test_cov_input_forward(self):
        batch, n_chans, n_outputs = 2, 22, 4
        x = _generate_spd_batch(batch, n_chans, n_chans)
        model = SPNN(n_chans=n_chans, n_outputs=n_outputs, hidden_dims=(14,))
        out = model(x)
        assert out.shape == (batch, n_outputs)

    def test_different_metrics(self):
        x = torch.randn(2, 8, 200)
        for metric in ["LEM", "AIM", "PEM"]:
            model = SPNN(n_chans=8, n_outputs=2, hidden_dims=(6,), metric=metric)
            out = model(x)
            assert out.shape == (2, 2)

    def test_gradient_flow(self):
        x = torch.randn(2, 10, 300)
        model = SPNN(n_chans=10, n_outputs=2, hidden_dims=(8, 6), drop_prob=0.1)
        model.train()
        out = model(x)
        loss = nn.functional.cross_entropy(out, torch.tensor([0, 1]))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert torch.isfinite(param.grad).all(), (
                    f"Parameter {name} gradient has NaNs"
                )

    def test_eval_mode(self):
        x = torch.randn(2, 10, 300)
        model = SPNN(n_chans=10, n_outputs=2, hidden_dims=(8,), drop_prob=0.5)
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


# =============================================================================
# GyroSPDAttention & SymSPDAttention Tests
# =============================================================================


class TestGyroSPDAttention:
    """Tests for Gyrovector SPD Attention mechanism."""

    def test_cayley_orthogonality(self):
        att = GyroSPDAttention(dim=4)
        W = torch.randn(4, 4)
        orth = att._cayley(W)
        eye = torch.eye(4)
        assert torch.allclose(orth @ orth.mT, eye, atol=1e-5)
        det = torch.linalg.det(orth)
        assert torch.isclose(det, torch.tensor(1.0), atol=1e-4)

    @pytest.mark.parametrize("metric", ["LEM", "LCM"])
    def test_forward_output_properties(self, metric: str):
        batch, n_patches, dim = 2, 4, 3
        X = _generate_spd_batch(batch, n_patches, dim, dim)
        att = GyroSPDAttention(dim=dim, metric=metric)
        out = att(X)
        assert out.shape == (batch, n_patches, dim, dim)
        assert torch.allclose(out, out.mT, atol=1e-5)
        evals = torch.linalg.eigvalsh(out)
        assert (evals > 0).all()

    def test_backward_gradient_flow(self):
        X = _generate_spd_batch(2, 3, 4, 4)
        att = GyroSPDAttention(dim=4, metric="LEM")
        out = att(X)
        loss = out.sum()
        loss.backward()
        assert (
            att.q_proj.parametrizations.weight.original.grad is not None
            and torch.isfinite(att.q_proj.parametrizations.weight.original.grad).all()
        )
        assert (
            att.k_proj.parametrizations.weight.original.grad is not None
            and torch.isfinite(att.k_proj.parametrizations.weight.original.grad).all()
        )
        assert (
            att.v_proj.parametrizations.weight.original.grad is not None
            and torch.isfinite(att.v_proj.parametrizations.weight.original.grad).all()
        )
        assert (
            att.parametrizations.bias.original.grad is not None
            and torch.isfinite(att.parametrizations.bias.original.grad).all()
        )

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            GyroSPDAttention(dim=4, metric="INVALID")


class TestSymSPDAttention:
    """Tests for Symmetric Tangent Space Attention."""

    def test_forward_output_properties(self):
        batch, n_patches, in_features, out_features = 2, 4, 4, 3
        X = _generate_spd_batch(batch, n_patches, in_features, in_features)
        att = SymSPDAttention(
            in_features=in_features, out_features=out_features, metric="LEM"
        )
        out = att(X)
        assert out.shape == (batch, n_patches, out_features, out_features)
        assert torch.allclose(out, out.mT, atol=1e-5)
        evals = torch.linalg.eigvalsh(out)
        assert (evals > 0).all()

    def test_backward_gradient_flow(self):
        X = _generate_spd_batch(2, 3, 4, 4)
        att = SymSPDAttention(in_features=4, out_features=3, metric="LEM")
        out = att(X)
        loss = out.sum()
        loss.backward()
        assert att.alpha.grad is not None and torch.isfinite(att.alpha.grad).all()
        assert att.beta.grad is not None and torch.isfinite(att.beta.grad).all()


# =============================================================================
# GyroSPDAtt Model Tests
# =============================================================================


class TestGyroSPDAtt:
    """Tests for the complete GyroSPDAtt model."""

    def test_gyro_attention_forward(self):
        batch, n_chans, n_times, n_outputs = 2, 22, 1000, 2
        x = torch.randn(batch, n_chans, n_times)
        model = GyroSPDAtt(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_patches=6,
            temporal_filters=22,
            spatial_filters=20,
            att_out_features=18,
            attention_type="gyro",
            metric="LEM",
        )
        out = model(x)
        assert out.shape == (batch, n_outputs)
        assert torch.isfinite(out).all()

    def test_sym_attention_forward(self):
        batch, n_chans, n_times, n_outputs = 2, 22, 1000, 2
        x = torch.randn(batch, n_chans, n_times)
        model = GyroSPDAtt(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_patches=4,
            temporal_filters=16,
            spatial_filters=12,
            att_out_features=10,
            attention_type="sym",
            metric="LEM",
        )
        out = model(x)
        assert out.shape == (batch, n_outputs)
        assert torch.isfinite(out).all()

    def test_cov_patches_forward(self):
        batch, n_patches, spatial_filters, n_outputs = 2, 4, 16, 2
        X_cov = _generate_spd_batch(batch, n_patches, spatial_filters, spatial_filters)
        model = GyroSPDAtt(
            n_chans=22,
            n_outputs=n_outputs,
            n_patches=n_patches,
            spatial_filters=spatial_filters,
            att_out_features=12,
            attention_type="gyro",
        )
        out = model(X_cov)
        assert out.shape == (batch, n_outputs)

    def test_invalid_attention_type(self):
        with pytest.raises(ValueError, match="attention_type must be 'gyro' or 'sym'"):
            GyroSPDAtt(attention_type="invalid")

    def test_backward_gradient_flow(self):
        x = torch.randn(2, 8, 400)
        model = GyroSPDAtt(
            n_chans=8,
            n_outputs=2,
            n_patches=4,
            temporal_filters=8,
            spatial_filters=6,
            att_out_features=6,
            attention_type="gyro",
            drop_prob=0.1,
        )
        model.train()
        out = model(x)
        loss = nn.functional.cross_entropy(out, torch.tensor([0, 1]))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert torch.isfinite(param.grad).all(), (
                    f"Parameter {name} gradient has NaNs"
                )
