# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Scaling, metric fidelity, and multi-channel tests for RieTrans (SPDRMLR, SPDLinear, SPNN)."""

import pytest
import torch

from spd_learn.models import SPNN
from spd_learn.modules import SPDRMLR, SPDLinear


def _make_spd(*shape: int, dtype=torch.float32) -> torch.Tensor:
    """Create a batch of well-conditioned SPD matrices."""
    torch.manual_seed(0)
    n = shape[-1]
    A = torch.randn(*shape, dtype=dtype) / (n**0.5)
    return A @ A.mT + 0.5 * torch.eye(n, dtype=dtype)


class TestRieTransMetricsAndChartMaps:
    """Tests for BWM, PEM, LCM, AIM, LEM chart/inverse-chart maps and (alpha, beta) isometry."""

    @pytest.mark.parametrize("metric", ["LEM", "LCM", "AIM", "PEM", "BWM"])
    @pytest.mark.parametrize("unidirectional", [False, True])
    def test_spdrmlr_all_metrics_and_modes(self, metric: str, unidirectional: bool):
        X = _make_spd(4, 5, 5)
        layer = SPDRMLR(
            n_features=5,
            n_outputs=3,
            metric=metric,
            theta=0.5 if metric in ("PEM", "BWM") else 1.0,
            unidirectional=unidirectional,
        )
        out = layer(X)
        assert out.shape == (4, 3)
        assert torch.isfinite(out).all()
        out.sum().backward()
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing grad for {name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

    @pytest.mark.parametrize("metric", ["LEM", "LCM", "AIM", "PEM", "BWM"])
    @pytest.mark.parametrize("unidirectional", [False, True])
    def test_spdlinear_all_metrics_spd_output(self, metric: str, unidirectional: bool):
        X = _make_spd(4, 6, 6)
        layer = SPDLinear(
            in_features=6,
            out_features=4,
            metric=metric,
            theta=0.5 if metric in ("PEM", "BWM") else 1.0,
            unidirectional=unidirectional,
        )
        Y = layer(X)
        assert Y.shape == (4, 4, 4)
        assert torch.allclose(Y, Y.mT, atol=1e-5)
        evals = torch.linalg.eigvalsh(Y)
        assert (evals > 0).all()
        Y.sum().backward()
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing grad for {name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

    @pytest.mark.parametrize("metric", ["LEM", "LCM", "AIM", "PEM", "BWM"])
    def test_chart_and_inverse_chart_exact_roundtrip(self, metric: str):
        """When W = I, b = 0, P = I, SPDLinear must be the exact identity map on SPD(d)."""
        d = 4
        X = _make_spd(3, d, d, dtype=torch.float64)
        layer = SPDLinear(
            in_features=d,
            out_features=d,
            metric=metric,
            theta=0.5,
            dtype=torch.float64,
        )
        with torch.no_grad():
            layer.rmlr.linear.weight.copy_(
                torch.eye(layer.out_vec_dim, dtype=torch.float64)
            )
            layer.rmlr.linear.bias.zero_()
        Y = layer(X)
        assert torch.allclose(Y, X, atol=1e-6), f"Roundtrip failed for {metric}"

    def test_alpha_beta_on_invariant_isometry(self):
        """Verify (alpha, beta) inner product isometry and v2V exact inverse."""
        d = 5
        alpha, beta = 2.0, 0.5
        layer = SPDLinear(
            in_features=d,
            out_features=d,
            metric="LEM",
            alpha=alpha,
            beta=beta,
            dtype=torch.float64,
        )
        A = torch.randn(4, d, d, dtype=torch.float64)
        B = torch.randn(4, d, d, dtype=torch.float64)
        U = 0.5 * (A + A.mT)
        V = 0.5 * (B + B.mT)

        u_vec = layer.rmlr._vectorize(U)
        v_vec = layer.rmlr._vectorize(V)

        # Euclidean dot product of vectorized forms must equal (alpha, beta) inner product
        dot_vec = (u_vec * v_vec).sum(dim=-1)
        tr_UV = (U * V).sum(dim=(-2, -1))
        tr_U = torch.diagonal(U, dim1=-2, dim2=-1).sum(dim=-1)
        tr_V = torch.diagonal(V, dim1=-2, dim2=-1).sum(dim=-1)
        expected = alpha * tr_UV + beta * tr_U * tr_V
        assert torch.allclose(dot_vec, expected, atol=1e-10)

        # _vec_to_tangent must be the exact inverse of _vectorize
        U_rec = layer._vec_to_tangent(u_vec)
        assert torch.allclose(U_rec, U, atol=1e-10)

    def test_bwm_general_2theta(self):
        """Test 2*theta-BWM with theta=0.25 and act_power=-0.25."""
        X = _make_spd(3, 5, 5)
        layer = SPDLinear(
            in_features=5,
            out_features=4,
            metric="BWM",
            theta=0.25,
            act_power=-0.25,
        )
        Y = layer(X)
        assert Y.shape == (3, 4, 4)
        assert (torch.linalg.eigvalsh(Y) > 0).all()


class TestMultiChannelProductSPDAndSPNN:
    """Tests for multi-channel product-SPD aggregation and SPNN enhancements."""

    @pytest.mark.parametrize("metric", ["LEM", "LCM", "AIM", "PEM", "BWM"])
    def test_multichannel_spdlinear_and_spdrmlr(self, metric: str):
        batch, c_in, c_out, d_in, d_out, n_classes = 3, 4, 2, 6, 4, 3
        X = _make_spd(batch, c_in, d_in, d_in)
        lin = SPDLinear(
            in_features=d_in,
            out_features=d_out,
            in_channels=c_in,
            out_channels=c_out,
            metric=metric,
            unidirectional=True,
        )
        clf = SPDRMLR(
            n_features=d_out,
            n_outputs=n_classes,
            in_channels=c_out,
            metric=metric,
            unidirectional=True,
        )
        H = lin(X)
        assert H.shape == (batch, c_out, d_out, d_out)
        assert (torch.linalg.eigvalsh(H) > 0).all()
        logits = clf(H)
        assert logits.shape == (batch, n_classes)
        logits.sum().backward()
        assert torch.isfinite(lin.rmlr.linear.weight.grad).all()
        assert torch.isfinite(clf.linear.weight.grad).all()

    @pytest.mark.parametrize("metric", ["LEM", "LCM", "AIM", "PEM", "BWM"])
    @pytest.mark.parametrize("use_chart_bypass", [False, True])
    def test_spnn_metrics_and_chart_bypass(self, metric: str, use_chart_bypass: bool):
        x = torch.randn(3, 8, 200)
        model = SPNN(
            n_chans=8,
            n_outputs=2,
            bimap_dims=(6, 4),
            metric=metric,
            use_chart_bypass=use_chart_bypass,
            act_power=-0.25 if metric in ("LCM", "BWM") else None,
        )
        out = model(x)
        assert out.shape == (3, 2)
        assert torch.isfinite(out).all()
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing grad for {name}"
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

    def test_spnn_multichannel_product_spd(self):
        X = _make_spd(3, 9, 10, 10)
        model = SPNN(
            n_chans=10,
            n_outputs=4,
            bimap_dims=(6,),
            in_channels=9,
            metric="LEM",
            unidirectional=True,
            use_chart_bypass=True,
        )
        out = model(X)
        assert out.shape == (3, 4)
        assert torch.isfinite(out).all()


class TestHighDimAndLargeSampleScaling:
    """Tests for d=400 high-dimensional SPD scaling and large-batch throughput."""

    def test_spdrmlr_high_dim_d400(self):
        d = 400
        X = _make_spd(2, d, d)
        layer = SPDRMLR(
            n_features=d,
            n_outputs=4,
            metric="LCM",
            unidirectional=True,
        )
        out = layer(X)
        assert out.shape == (2, 4)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(layer.linear.weight.grad).all()

    def test_spdlinear_high_dim_d400_to_d32(self):
        d_in, d_out = 400, 32
        X = _make_spd(2, d_in, d_in)
        layer = SPDLinear(
            in_features=d_in,
            out_features=d_out,
            metric="LCM",
            unidirectional=True,
        )
        Y = layer(X)
        assert Y.shape == (2, d_out, d_out)
        assert torch.isfinite(Y).all()
        evals = torch.linalg.eigvalsh(Y)
        assert (evals > 0).all()

    def test_spdrmlr_large_sample_batch(self):
        N, d = 2048, 8
        X = _make_spd(N, d, d)
        layer = SPDRMLR(n_features=d, n_outputs=4, metric="LCM", unidirectional=True)
        out = layer(X)
        assert out.shape == (N, 4)
        assert torch.isfinite(out).all()
