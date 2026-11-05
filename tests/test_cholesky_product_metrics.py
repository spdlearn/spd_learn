# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from torch.autograd import gradcheck

from spd_learn.functional.metrics import (
    bures_wasserstein_cholesky_distance,
    bures_wasserstein_cholesky_exp,
    bures_wasserstein_cholesky_geodesic,
    bures_wasserstein_cholesky_log,
    bures_wasserstein_cholesky_mean,
    bures_wasserstein_distance,
    power_cholesky_coords,
    power_cholesky_distance,
    power_cholesky_exp,
    power_cholesky_geodesic,
    power_cholesky_inv_coords,
    power_cholesky_log,
    power_cholesky_mean,
)


def random_spd(n=3, batch_shape=(), dtype=torch.float64):
    shape = (*batch_shape, n, n)
    A = torch.randn(shape, dtype=dtype)
    return A @ A.mT + 2.0 * torch.eye(n, dtype=dtype)


# =============================================================================
# Metric Axioms & Distance Properties
# =============================================================================


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 2.0])
def test_power_cholesky_distance_identity(theta):
    A = random_spd(n=4)
    dist = power_cholesky_distance(A, A, theta=theta)
    torch.testing.assert_close(dist, torch.zeros((), dtype=torch.float64))


def test_bures_wasserstein_cholesky_distance_identity():
    A = random_spd(n=4)
    dist = bures_wasserstein_cholesky_distance(A, A)
    torch.testing.assert_close(dist, torch.zeros((), dtype=torch.float64))


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_power_cholesky_distance_symmetry(theta):
    A = random_spd(n=3)
    B = random_spd(n=3)
    d_AB = power_cholesky_distance(A, B, theta=theta)
    d_BA = power_cholesky_distance(B, A, theta=theta)
    torch.testing.assert_close(d_AB, d_BA)


def test_bures_wasserstein_cholesky_distance_symmetry():
    A = random_spd(n=3)
    B = random_spd(n=3)
    d_AB = bures_wasserstein_cholesky_distance(A, B)
    d_BA = bures_wasserstein_cholesky_distance(B, A)
    torch.testing.assert_close(d_AB, d_BA)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_power_cholesky_distance_triangle_inequality(theta):
    A = random_spd(n=3)
    B = random_spd(n=3)
    C = random_spd(n=3)

    d_AC = power_cholesky_distance(A, C, theta=theta)
    d_AB = power_cholesky_distance(A, B, theta=theta)
    d_BC = power_cholesky_distance(B, C, theta=theta)

    assert d_AC <= d_AB + d_BC + 1e-12


def test_bures_wasserstein_cholesky_distance_triangle_inequality():
    A = random_spd(n=3)
    B = random_spd(n=3)
    C = random_spd(n=3)

    d_AC = bures_wasserstein_cholesky_distance(A, C)
    d_AB = bures_wasserstein_cholesky_distance(A, B)
    d_BC = bures_wasserstein_cholesky_distance(B, C)

    assert d_AC <= d_AB + d_BC + 1e-12


def test_bures_wasserstein_cholesky_diagonal_equivalence():
    """BWCM on diagonal matrices matches the exact Bures-Wasserstein distance."""
    # Let D_A and D_B be diagonal SPD matrices with diagonal elements d_A, d_B
    d_A = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    d_B = torch.tensor([2.0, 2.0, 8.0, 4.0], dtype=torch.float64)

    D_A = torch.diag(d_A)
    D_B = torch.diag(d_B)

    # In Cholesky factorization of A = D_A^2, the Cholesky factor is D_A
    # Hence the diagonal factor is d_A.
    # The BW distance between D_A and D_B is ||sqrt(d_A) - sqrt(d_B)||_2
    bw_dist = bures_wasserstein_distance(D_A, D_B)
    bwcm_dist = bures_wasserstein_cholesky_distance(D_A @ D_A, D_B @ D_B)

    torch.testing.assert_close(bwcm_dist, bw_dist)


def test_distance_broadcasting():
    A = random_spd(n=3, batch_shape=(5, 1))
    B = random_spd(n=3, batch_shape=(1, 4))
    dist = power_cholesky_distance(A, B, theta=0.5)
    assert dist.shape == (5, 4)

    # Verify individual entries
    for i in range(5):
        for j in range(4):
            d_ij = power_cholesky_distance(A[i, 0], B[0, j], theta=0.5)
            torch.testing.assert_close(dist[i, j], d_ij)


# =============================================================================
# Coordinate Maps & Riemannian Log/Exp
# =============================================================================


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 2.0])
def test_power_cholesky_coords_roundtrip(theta):
    X = random_spd(n=4)
    Z = power_cholesky_coords(X, theta=theta)
    X_rec = power_cholesky_inv_coords(Z, theta=theta)
    torch.testing.assert_close(X_rec, X)


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 2.0])
def test_power_cholesky_log_exp_roundtrip_none_p(theta):
    X = random_spd(n=4, batch_shape=(3,))
    V = power_cholesky_log(X, P=None, theta=theta)
    X_rec = power_cholesky_exp(V, P=None, theta=theta)
    torch.testing.assert_close(X_rec, X)


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_power_cholesky_log_exp_roundtrip_given_p(theta):
    X = random_spd(n=3, batch_shape=(2,))
    P = random_spd(n=3, batch_shape=(2,))
    V = power_cholesky_log(X, P=P, theta=theta)
    X_rec = power_cholesky_exp(V, P=P, theta=theta)
    torch.testing.assert_close(X_rec, X)


def test_bures_wasserstein_cholesky_log_exp_roundtrip():
    X = random_spd(n=3)
    P = random_spd(n=3)

    # Reference None
    V_none = bures_wasserstein_cholesky_log(X, P=None)
    X_rec_none = bures_wasserstein_cholesky_exp(V_none, P=None)
    torch.testing.assert_close(X_rec_none, X)

    # Reference P
    V_p = bures_wasserstein_cholesky_log(X, P=P)
    X_rec_p = bures_wasserstein_cholesky_exp(V_p, P=P)
    torch.testing.assert_close(X_rec_p, X)


def test_log_at_reference_is_zero():
    P = random_spd(n=3)
    V = power_cholesky_log(P, P=P, theta=0.5)
    torch.testing.assert_close(V, torch.zeros_like(V))

    I = torch.eye(3, dtype=torch.float64)
    V_ident = power_cholesky_log(I, P=None, theta=0.5)
    torch.testing.assert_close(V_ident, torch.zeros_like(V_ident))


# =============================================================================
# Geodesics
# =============================================================================


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_power_cholesky_geodesic_endpoints(theta):
    A = random_spd(n=3)
    B = random_spd(n=3)

    g0 = power_cholesky_geodesic(A, B, 0.0, theta=theta)
    g1 = power_cholesky_geodesic(A, B, 1.0, theta=theta)

    torch.testing.assert_close(g0, A)
    torch.testing.assert_close(g1, B)


def test_bures_wasserstein_cholesky_geodesic_endpoints():
    A = random_spd(n=3)
    B = random_spd(n=3)

    g0 = bures_wasserstein_cholesky_geodesic(A, B, 0.0)
    g1 = bures_wasserstein_cholesky_geodesic(A, B, 1.0)

    torch.testing.assert_close(g0, A)
    torch.testing.assert_close(g1, B)


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_power_cholesky_geodesic_midpoint_equals_mean(theta):
    A = random_spd(n=3)
    B = random_spd(n=3)

    mid = power_cholesky_geodesic(A, B, 0.5, theta=theta)
    mean = power_cholesky_mean(torch.stack([A, B]), theta=theta)

    torch.testing.assert_close(mid, mean)


def test_bures_wasserstein_cholesky_geodesic_midpoint_equals_mean():
    A = random_spd(n=3)
    B = random_spd(n=3)

    mid = bures_wasserstein_cholesky_geodesic(A, B, 0.5)
    mean = bures_wasserstein_cholesky_mean(torch.stack([A, B]))

    torch.testing.assert_close(mid, mean)


@pytest.mark.parametrize("t", [0.25, 0.5, 0.75])
def test_power_cholesky_geodesic_distance_proportional(t):
    A = random_spd(n=3)
    B = random_spd(n=3)
    theta = 0.5

    d_total = power_cholesky_distance(A, B, theta=theta)
    gt = power_cholesky_geodesic(A, B, t, theta=theta)
    d_segment = power_cholesky_distance(A, gt, theta=theta)

    torch.testing.assert_close(d_segment, t * d_total)


# =============================================================================
# Fréchet Mean
# =============================================================================


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_power_cholesky_mean_single_matrix(theta):
    X = random_spd(n=3).unsqueeze(0)
    mean = power_cholesky_mean(X, theta=theta)
    torch.testing.assert_close(mean, X.squeeze(0))


def test_power_cholesky_mean_uniform_vs_none_weights():
    X = random_spd(n=3, batch_shape=(4,))
    uniform_weights = torch.ones(4, dtype=torch.float64) / 4.0

    mean_none = power_cholesky_mean(X, weights=None, theta=0.5)
    mean_uniform = power_cholesky_mean(X, weights=uniform_weights, theta=0.5)

    torch.testing.assert_close(mean_none, mean_uniform)


def test_power_cholesky_mean_weighted():
    A = random_spd(n=3)
    B = random_spd(n=3)
    X = torch.stack([A, B])
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)

    mean = power_cholesky_mean(X, weights=weights, theta=0.5)
    geodesic_pt = power_cholesky_geodesic(A, B, 0.75, theta=0.5)

    torch.testing.assert_close(mean, geodesic_pt)


def test_power_cholesky_mean_batched_dim():
    X = random_spd(n=3, batch_shape=(2, 5))
    mean = power_cholesky_mean(X, dim=1, keepdim=False, theta=0.5)
    assert mean.shape == (2, 3, 3)

    mean_keep = power_cholesky_mean(X, dim=1, keepdim=True, theta=0.5)
    assert mean_keep.shape == (2, 1, 3, 3)

    for b in range(2):
        expected_b = power_cholesky_mean(X[b], dim=0, theta=0.5)
        torch.testing.assert_close(mean[b], expected_b)


# =============================================================================
# Autograd Gradcheck
# =============================================================================


def test_power_cholesky_distance_gradcheck():
    A = random_spd(n=3).requires_grad_(True)
    B = random_spd(n=3).requires_grad_(True)

    assert gradcheck(lambda a, b: power_cholesky_distance(a, b, theta=0.5), (A, B))


def test_bures_wasserstein_cholesky_distance_gradcheck():
    A = random_spd(n=3).requires_grad_(True)
    B = random_spd(n=3).requires_grad_(True)

    assert gradcheck(bures_wasserstein_cholesky_distance, (A, B))


def test_power_cholesky_geodesic_gradcheck():
    A = random_spd(n=3).requires_grad_(True)
    B = random_spd(n=3).requires_grad_(True)
    t = 0.4

    assert gradcheck(lambda a, b: power_cholesky_geodesic(a, b, t, theta=0.5), (A, B))


def test_power_cholesky_mean_gradcheck():
    X = random_spd(n=3, batch_shape=(3,)).requires_grad_(True)
    assert gradcheck(lambda x: power_cholesky_mean(x, theta=0.5), (X,))


def test_power_cholesky_log_gradcheck():
    X = random_spd(n=3).requires_grad_(True)
    P = random_spd(n=3).requires_grad_(True)

    # P is None
    assert gradcheck(lambda x: power_cholesky_log(x, P=None, theta=0.5), (X,))
    # P is given
    assert gradcheck(lambda x, p: power_cholesky_log(x, P=p, theta=0.5), (X, P))


def test_power_cholesky_exp_gradcheck():
    X = random_spd(n=3)
    P = random_spd(n=3).requires_grad_(True)

    V_none = power_cholesky_log(X, None, theta=0.5).requires_grad_(True)
    assert gradcheck(lambda v: power_cholesky_exp(v, P=None, theta=0.5), (V_none,))

    V_p = power_cholesky_log(X, P.detach(), theta=0.5).requires_grad_(True)
    assert gradcheck(lambda v, p: power_cholesky_exp(v, P=p, theta=0.5), (V_p, P))


# =============================================================================
# Error Handling & Edge Cases
# =============================================================================


def test_zero_theta_raises():
    A = random_spd(n=3)
    B = random_spd(n=3)
    with pytest.raises(ValueError, match="theta must be non-zero"):
        power_cholesky_distance(A, B, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        power_cholesky_geodesic(A, B, 0.5, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        power_cholesky_mean(torch.stack([A, B]), theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        power_cholesky_log(A, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        power_cholesky_exp(A, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        power_cholesky_coords(A, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        power_cholesky_inv_coords(A, theta=0.0)


def test_invalid_mean_dim_raises():
    X = random_spd(n=3, batch_shape=(2,))
    with pytest.raises(ValueError, match="batch dimension"):
        power_cholesky_mean(X, dim=-1)

    with pytest.raises(ValueError, match="batch dimension"):
        power_cholesky_mean(X, dim=-2)

    with pytest.raises(ValueError, match="at least 3 dimensions"):
        power_cholesky_mean(random_spd(n=3))
