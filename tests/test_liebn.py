"""Tests for Lie Group Batch Normalization (LieBN) for SPD matrices.

Verifies the theoretical guarantees from Chen et al., ICLR 2024
(Proposition 4.2):
  - Mean property: after centering+biasing with bias=I, output Frechet mean ≈ I
  - Variance property: after scaling with shift=1, dispersion ≈ 1.0
  - Running statistics converge to population statistics
"""

from math import sqrt

import pytest
import torch

from spd_learn.functional import (
    ensure_sym,
    matrix_exp,
    matrix_log,
    vec_to_sym,
)
from spd_learn.functional.batchnorm import karcher_mean_iteration
from spd_learn.modules import SPDBatchNormLie


DTYPE = torch.float64

# ---------------------------------------------------------------------------
# Data fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def simulated_data():
    """Generate SPD data with known Frechet mean for testing.

    Strategy: zero-mean tangent vectors -> matrix_exp -> SPD at Identity,
    then apply linear mixing x = A z A^T so Frechet mean = A A^T.
    """
    ndim = 10
    nobs = 128
    generator = torch.Generator().manual_seed(42)

    # Zero-mean tangent vectors -> SPD matrices centered at Identity
    logz = vec_to_sym(
        torch.randn((nobs, ndim * (ndim + 1) // 2), generator=generator, dtype=DTYPE)
    )
    logz = logz - logz.mean(dim=0, keepdim=True)
    z = matrix_exp.apply(logz)

    # Linear mixing model: shifts Frechet mean to A @ A^T
    eps = 0.1
    forward_model = (
        torch.rand((ndim, ndim), generator=generator, dtype=DTYPE) - 0.5
    ) * (1 - eps) + eps * torch.eye(ndim, dtype=DTYPE)
    x = forward_model @ z @ forward_model.mT

    # Analytic Frechet mean (by invariance)
    x_mean_expected = (forward_model @ forward_model.mT).unsqueeze(0)

    return x, x_mean_expected, ndim, nobs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

METRICS = ["AIM", "LEM", "LCM"]
CONGRUENCES = ["cholesky", "eig"]


@pytest.mark.parametrize(
    "metric,theta,atol",
    [
        ("AIM", 1.0, 1e-10),
        ("LCM", 1.0, 1e-5),  # Cholesky + log/exp introduces numerical error
        ("LCM", 0.5, 1e-4),  # Additional matrix_power roundtrip error
    ],
)
def test_deform_inv_deform_roundtrip(simulated_data, metric, theta, atol):
    """_inv_deform(_deform(X)) should recover X."""
    x, _, ndim, _ = simulated_data
    layer = SPDBatchNormLie(ndim, metric=metric, theta=theta, dtype=DTYPE)

    X_def = layer._deform(x)
    X_recovered = layer._inv_deform(X_def)

    assert torch.allclose(X_recovered, x, atol=atol, rtol=0.0)


@pytest.mark.parametrize("congruence", CONGRUENCES)
@pytest.mark.parametrize("metric", METRICS)
def test_post_normalization_mean(simulated_data, metric, congruence):
    """After LieBN forward (bias=I, shift=1), codomain mean should be neutral.

    - AIM: Karcher mean of output ≈ Identity
    - LEM/LCM: arithmetic mean of deformed output ≈ zero matrix
    """
    x, _, ndim, nobs = simulated_data
    layer = SPDBatchNormLie(
        ndim, metric=metric, n_iter=64, congruence=congruence, dtype=DTYPE
    )
    layer.train()

    with torch.no_grad():
        output = layer(x)

    tol = 2 * sqrt(1.0 / nobs)

    if metric == "AIM":
        # Compute Karcher mean of output
        mean = output.mean(dim=0, keepdim=True)
        for _ in range(64):
            mean = karcher_mean_iteration(output, mean, detach=True)
        identity = torch.eye(ndim, dtype=DTYPE).unsqueeze(0)
        assert torch.allclose(mean, identity, atol=tol, rtol=0.0), (
            f"AIM: Karcher mean of output deviates from Identity by "
            f"{(mean - identity).abs().max().item():.6f}"
        )
    else:
        # In codomain, mean should be ≈ zero
        output_def = layer._deform(output)
        codomain_mean = output_def.mean(dim=0, keepdim=True)
        zeros = torch.zeros_like(codomain_mean)
        assert torch.allclose(codomain_mean, zeros, atol=tol, rtol=0.0), (
            f"{metric}: codomain mean deviates from zero by "
            f"{codomain_mean.abs().max().item():.6f}"
        )


@pytest.mark.parametrize("metric", METRICS)
def test_post_normalization_variance(simulated_data, metric):
    """After LieBN forward (shift=1), output variance should be ≈ 1.0.

    Theoretical: shift^2 * v^2 / (v^2 + eps). With shift=1 and large v^2,
    this is close to 1.0.
    """
    x, _, ndim, nobs = simulated_data
    layer = SPDBatchNormLie(ndim, metric=metric, n_iter=64, dtype=DTYPE)
    layer.train()

    with torch.no_grad():
        output = layer(x)

    # Compute variance of output in the same way as SPDBatchNormLie._frechet_variance
    # but on the re-centered output
    output_def = layer._deform(output)
    if metric == "AIM":
        output_mean = output_def.mean(dim=0, keepdim=True)
        for _ in range(64):
            output_mean = karcher_mean_iteration(output_def, output_mean, detach=True)
        L = torch.linalg.cholesky(output_mean)
        Y = torch.linalg.solve_triangular(L, output_def, upper=False)
        centered = ensure_sym(torch.linalg.solve_triangular(L, Y.mT, upper=False).mT)
        logX = matrix_log.apply(centered)
        frob_sq = (logX * logX).sum(dim=(-2, -1))
        output_var = frob_sq.mean()
    else:
        centered = output_def - output_def.mean(dim=0, keepdim=True)
        frob_sq = (centered * centered).sum(dim=(-2, -1))
        output_var = frob_sq.mean()

    # Expected: shift^2 * v^2 / (v^2 + eps) ≈ 1.0
    tol = 3 * sqrt(1.0 / nobs)
    assert abs(output_var.item() - 1.0) < tol, (
        f"{metric}: output variance = {output_var.item():.6f}, expected ≈ 1.0"
    )


@pytest.mark.parametrize("metric", METRICS)
def test_running_stats_single_batch(simulated_data, metric):
    """With momentum=1.0, running stats should match batch stats exactly."""
    x, _, ndim, nobs = simulated_data
    layer = SPDBatchNormLie(ndim, metric=metric, momentum=1.0, n_iter=64, dtype=DTYPE)
    layer.train()

    with torch.no_grad():
        layer(x)

    # Independently compute batch statistics
    X_def = layer._deform(x)
    if metric == "AIM":
        expected_mean = X_def.mean(dim=0, keepdim=True)
        for _ in range(64):
            expected_mean = karcher_mean_iteration(X_def, expected_mean, detach=True)
    else:
        expected_mean = X_def.mean(dim=0, keepdim=True)

    tol = sqrt(1.0 / nobs)
    assert torch.allclose(layer.running_mean, expected_mean, atol=tol, rtol=0.0), (
        f"{metric}: running_mean deviates from batch mean by "
        f"{(layer.running_mean - expected_mean).abs().max().item():.6f}"
    )

    # Compute expected variance from centered data
    if metric == "AIM":
        L = torch.linalg.cholesky(expected_mean)
        Y = torch.linalg.solve_triangular(L, X_def, upper=False)
        centered = ensure_sym(torch.linalg.solve_triangular(L, Y.mT, upper=False).mT)
        logX = matrix_log.apply(centered)
        frob_sq = (logX * logX).sum(dim=(-2, -1))
        expected_var = frob_sq.mean()
    else:
        centered = X_def - expected_mean
        frob_sq = (centered * centered).sum(dim=(-2, -1))
        expected_var = frob_sq.mean()

    assert torch.allclose(layer.running_var, expected_var, atol=tol, rtol=0.0), (
        f"{metric}: running_var = {layer.running_var.item():.6f}, "
        f"expected = {expected_var.item():.6f}"
    )


@pytest.mark.parametrize("metric", METRICS)
def test_running_stats_convergence(simulated_data, metric):
    """Running stats should converge to population stats over mini-batches."""
    x, _, ndim, nobs = simulated_data
    layer = SPDBatchNormLie(ndim, metric=metric, n_iter=1, dtype=DTYPE)

    # Full-batch reference statistics (high precision)
    with torch.no_grad():
        ref_layer = SPDBatchNormLie(
            ndim, metric=metric, momentum=1.0, n_iter=64, dtype=DTYPE
        )
        ref_layer.train()
        ref_layer(x)
        ref_mean = ref_layer.running_mean.clone()
        ref_var = ref_layer.running_var.clone()

    # Train with mini-batches and decaying momentum
    ds = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(ds, batch_size=nobs // 4, drop_last=True)

    layer.train()
    n_epochs = 64 // len(loader) * 4
    for epoch in range(n_epochs):
        layer.momentum = 1 / (epoch + 1)
        for batch in loader:
            with torch.no_grad():
                layer(batch[0])

    tol = 5 * sqrt(1.0 / nobs)
    assert torch.allclose(layer.running_mean, ref_mean, atol=tol, rtol=0.0), (
        f"{metric}: running_mean did not converge. "
        f"Max deviation: {(layer.running_mean - ref_mean).abs().max().item():.6f}"
    )
    assert torch.allclose(layer.running_var, ref_var, atol=tol, rtol=0.05), (
        f"{metric}: running_var did not converge. "
        f"running={layer.running_var.item():.4f}, ref={ref_var.item():.4f}"
    )


@pytest.mark.parametrize("metric", METRICS)
def test_gradient_flow(simulated_data, metric):
    """Verify gradients flow through LieBN to input and parameters."""
    x, _, ndim, _ = simulated_data
    # Use a small batch to keep computation fast
    x_small = x[:8].clone().requires_grad_(True)

    layer = SPDBatchNormLie(ndim, metric=metric, n_iter=1, dtype=DTYPE)
    layer.train()

    output = layer(x_small)
    loss = output.sum()
    loss.backward()

    # Input gradient
    assert x_small.grad is not None, f"{metric}: no gradient on input"
    assert x_small.grad.abs().sum() > 0, f"{metric}: zero gradient on input"

    # Bias parameter gradient (underlying unconstrained parameter)
    bias_param = layer.parametrizations.bias.original
    assert bias_param.grad is not None, f"{metric}: no gradient on bias"
    assert bias_param.grad.abs().sum() > 0, f"{metric}: zero gradient on bias"

    # Shift parameter gradient
    shift_param = layer.parametrizations.shift.original
    assert shift_param.grad is not None, f"{metric}: no gradient on shift"
    assert shift_param.grad.abs().sum() > 0, f"{metric}: zero gradient on shift"


@pytest.mark.parametrize("metric", METRICS)
def test_default_initialization(metric):
    """Verify default parameter initialization."""
    ndim = 4
    layer = SPDBatchNormLie(ndim, metric=metric, dtype=DTYPE)

    # Bias should be Identity
    identity = torch.eye(ndim, dtype=DTYPE).unsqueeze(0)
    assert torch.allclose(layer.bias, identity, atol=1e-10), (
        f"{metric}: bias not initialized to Identity"
    )

    # Shift should be 1.0
    assert torch.allclose(layer.shift, torch.ones((), dtype=DTYPE), atol=1e-10), (
        f"{metric}: shift not initialized to 1.0"
    )

    # Running mean: Identity for AIM, zeros for LEM/LCM
    if metric == "AIM":
        assert torch.allclose(layer.running_mean, identity, atol=1e-10)
    else:
        assert torch.allclose(
            layer.running_mean, torch.zeros(1, ndim, ndim, dtype=DTYPE), atol=1e-10
        )

    # Running var should be 1.0
    assert torch.allclose(layer.running_var, torch.ones((), dtype=DTYPE), atol=1e-10)

    # Verify dtype propagated correctly
    assert layer.bias.dtype == DTYPE
    assert layer.shift.dtype == DTYPE
    assert layer.running_mean.dtype == DTYPE
    assert layer.running_var.dtype == DTYPE
