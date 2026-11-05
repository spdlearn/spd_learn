# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Comprehensive parametrized test suite for all SPD Learn models, modules, and functional metrics."""

from __future__ import annotations

import inspect

import pytest
import torch

from torch.testing import assert_close

import spd_learn.functional as F_spd

from spd_learn import models, modules
from spd_learn.modules import step_dynamic_trivialization


def _make_spd(
    batch_shape: tuple[int, ...],
    dim: int,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(*batch_shape, dim, dim, generator=g, dtype=dtype)
    return a @ a.mT / dim + 0.5 * torch.eye(dim, dtype=dtype)


# =============================================================================
# 1. Parametrized End-to-End Model Suite (All 22 Models + SPDTransNet)
# =============================================================================


@pytest.mark.parametrize("model_name", models.__all__)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("batch_size", [2, 4], ids=["b2", "b4"])
@pytest.mark.parametrize("n_outputs", [2, 4], ids=["c2", "c4"])
def test_all_models_parametrized_forward_backward_rebase(
    model_name: str,
    dtype: torch.dtype,
    batch_size: int,
    n_outputs: int,
) -> None:
    """Verify forward shape, dtype, eval determinism, finite gradients, and dynamic trivialization."""
    n_chans = 22
    model_cls = getattr(models, model_name)
    sig = inspect.signature(model_cls.__init__).parameters
    kwargs: dict = dict(n_chans=n_chans, n_outputs=n_outputs)
    if "dtype" in sig:
        kwargs["dtype"] = dtype
    if model_name == "Green":
        kwargs.update(sfreq=100, bi_out=[8], hidden_dim=(8,))
    elif model_name == "TensorCSPNet":
        kwargs.update(dims=(22, 16, 16, 8), parametrized=True)
    elif model_name == "GraphCSPNet":
        kwargs.update(dims=(22, 16, 8), n_patches=2)

    model = model_cls(**kwargs).to(dtype)

    g = torch.Generator().manual_seed(7)
    if model_name in models.__filter_bank_models__:
        x = torch.randn(batch_size, 9, n_chans, 160, generator=g, dtype=dtype)
    else:
        x = torch.randn(batch_size, n_chans, 160, generator=g, dtype=dtype)

    if model_name == "GraphCSPNet":
        model.fit_graph(x)

    # Train mode forward + backward + optimizer step + dynamic trivialization
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    assert out.shape == (batch_size, n_outputs)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()

    loss = out.square().mean()
    loss.backward()
    grad_norms = [
        p.grad.abs().sum().item()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    optimizer.step()
    step_dynamic_trivialization(model, optimizer)

    # Eval mode determinism
    model.eval()
    with torch.no_grad():
        out_eval_1 = model(x)
        out_eval_2 = model(x)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    assert_close(out_eval_1, out_eval_2, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("context_epochs", [1, 3], ids=["ctx1", "ctx3"])
@pytest.mark.parametrize("inter_heads", [1, 2], ids=["h1", "h2"])
def test_spdtransnet_parametrized(
    dtype: torch.dtype,
    context_epochs: int,
    inter_heads: int,
) -> None:
    """Parametrized verification of sequence-native SPDTransNet."""
    # token_matrix_size=4 -> D = 4*5//2 = 10 (triangular number), divisible by 1 and 2
    model = models.SPDTransNet(
        matrix_size=4,
        n_bands=2,
        n_subwindows=2,
        context_epochs=context_epochs,
        token_matrix_size=4,
        feature_tokens=2,
        intra_heads=2,
        inter_heads=inter_heads,
        intra_layers=1,
        inter_layers=1,
        feedforward_dim=10,
        classifier_dim=10,
        n_outputs=3,
        dtype=dtype,
    )
    x = _make_spd((2, context_epochs, 2, 2), dim=4, dtype=dtype)
    out = model(x)
    assert out.shape == (2, 3)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


# =============================================================================
# 2. Parametrized Functional Metrics Suite (PCM, BWCM, Bures-Wasserstein, Corr)
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [3, 6], ids=["d3", "d6"])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)], ids=["1d", "2d"])
def test_cholesky_and_bw_metrics_invariants(
    dtype: torch.dtype,
    dim: int,
    batch_shape: tuple[int, ...],
) -> None:
    """Verify roundtrip, symmetry, non-negativity, and geodesic midpoint laws for PCM, BWCM, and BW."""
    tol = 2e-4 if dtype == torch.float32 else 1e-9
    A = _make_spd(batch_shape, dim, dtype=dtype, seed=11)
    B = _make_spd(batch_shape, dim, dtype=dtype, seed=29)

    # Cholesky log/exp roundtrip (takes SPD matrix A directly)
    X_A = F_spd.cholesky_log.apply(A)
    A_rec = F_spd.cholesky_exp.apply(X_A)
    assert_close(A_rec, A, atol=tol, rtol=tol)

    # PCM, BWCM, Log-Cholesky & Bures-Wasserstein distance symmetry and zero self-distance
    for dist_fn, geo_fn in [
        (F_spd.log_cholesky_distance, F_spd.log_cholesky_geodesic),
        (F_spd.power_cholesky_distance, F_spd.power_cholesky_geodesic),
        (
            F_spd.bures_wasserstein_cholesky_distance,
            F_spd.bures_wasserstein_cholesky_geodesic,
        ),
        (F_spd.bures_wasserstein_distance, F_spd.bures_wasserstein_geodesic),
    ]:
        d_aa = dist_fn(A, A)
        d_ab = dist_fn(A, B)
        d_ba = dist_fn(B, A)
        zero_tol = 5e-3 if dtype == torch.float32 else 1e-6
        assert_close(d_aa, torch.zeros_like(d_aa), atol=zero_tol, rtol=zero_tol)
        assert_close(d_ab, d_ba, atol=tol, rtol=tol)
        assert (d_ab >= 0).all()

        # Geodesic endpoints & midpoint bisecting property
        g0 = geo_fn(A, B, 0.0)
        g1 = geo_fn(A, B, 1.0)
        g_half = geo_fn(A, B, 0.5)
        assert_close(g0, A, atol=tol, rtol=tol)
        assert_close(g1, B, atol=tol, rtol=tol)
        assert (torch.linalg.eigvalsh(g_half) > 0).all()
        d_half_a = dist_fn(A, g_half)
        d_half_b = dist_fn(g_half, B)
        assert_close(d_half_a, 0.5 * d_ab, atol=tol * 5, rtol=tol * 5)
        assert_close(d_half_b, 0.5 * d_ab, atol=tol * 5, rtol=tol * 5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [4, 7], ids=["d4", "d7"])
def test_correlation_manifold_diffeomorphism_parametrized(
    dtype: torch.dtype,
    dim: int,
) -> None:
    """Verify OffLog / OnExp diffeomorphism between Cor+(d) and Sym_0(d)."""
    tol = 1e-4 if dtype == torch.float32 else 1e-8
    S = _make_spd((3,), dim, dtype=dtype, seed=19)
    C = F_spd.cov_to_corr(S)

    # Unit diagonal and SPD
    diag_C = torch.diagonal(C, dim1=-2, dim2=-1)
    assert_close(diag_C, torch.ones_like(diag_C), atol=tol, rtol=tol)
    assert (torch.linalg.eigvalsh(C) > 0).all()

    # OffLog -> hollow symmetric matrix
    H = F_spd.corr_olm_log(C)
    diag_H = torch.diagonal(H, dim1=-2, dim2=-1)
    assert_close(diag_H, torch.zeros_like(diag_H), atol=tol, rtol=tol)
    assert_close(H, H.mT, atol=tol, rtol=tol)

    # OnExp roundtrip back to C
    C_rec = F_spd.corr_olm_exp(H)
    assert_close(C_rec, C, atol=tol * 5, rtol=tol * 5)


# =============================================================================
# 3. Parametrized Batch Normalization Suite (All 6 SPD BatchNorm Layers)
# =============================================================================


@pytest.mark.parametrize(
    "bn_name",
    [
        "SPDBatchNormMean",
        "SPDBatchNormMeanVar",
        "SPDBatchNormLie",
        "SPDBatchNormHarmonicArithmetic",
        "SPDBatchNormBW",
        "SPDBatchNormCholesky",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("dim", [4, 6], ids=["d4", "d6"])
def test_all_spd_batchnorms_parametrized(
    bn_name: str,
    dtype: torch.dtype,
    dim: int,
) -> None:
    """Verify train/eval forward, SPD cone preservation, finite gradients, and rebase."""
    bn_cls = getattr(modules, bn_name)
    layer = bn_cls(num_features=dim, dtype=dtype)
    X = _make_spd((4,), dim, dtype=dtype, seed=53).requires_grad_(True)

    layer.train()
    out_train = layer(X)
    assert out_train.shape == (4, dim, dim)
    assert out_train.dtype == dtype
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    assert_close(out_train, out_train.mT, atol=tol, rtol=tol)
    assert (torch.linalg.eigvalsh(out_train) > 0).all()

    out_train.square().mean().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()
    for p in layer.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()

    layer.eval()
    with torch.no_grad():
        out_eval_1 = layer(X)
        step_dynamic_trivialization(layer)
        out_eval_2 = layer(X)
    assert_close(out_eval_1, out_eval_2, atol=tol, rtol=tol)
