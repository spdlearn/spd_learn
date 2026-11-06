# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from torch.autograd import gradcheck

from spd_learn.functional.metrics import (
    bures_wasserstein_distance,
    bwcm_distance,
    bwcm_geodesic,
    bwcm_mean,
    exp_map_bwcm,
    exp_map_pcm,
    log_map_bwcm,
    log_map_pcm,
    pcm_coords,
    pcm_distance,
    pcm_geodesic,
    pcm_inv_coords,
    pcm_mean,
)


def random_spd(n=3, batch_shape=(), dtype=torch.float64):
    shape = (*batch_shape, n, n)
    A = torch.randn(shape, dtype=dtype)
    return A @ A.mT + 2.0 * torch.eye(n, dtype=dtype)


# =============================================================================
# Metric Axioms & Distance Properties
# =============================================================================


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 2.0])
def test_pcm_distance_identity(theta):
    A = random_spd(n=4)
    dist = pcm_distance(A, A, theta=theta)
    torch.testing.assert_close(dist, torch.zeros((), dtype=torch.float64))


def test_bwcm_distance_identity():
    A = random_spd(n=4)
    dist = bwcm_distance(A, A)
    torch.testing.assert_close(dist, torch.zeros((), dtype=torch.float64))


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_pcm_distance_symmetry(theta):
    A = random_spd(n=3)
    B = random_spd(n=3)
    d_AB = pcm_distance(A, B, theta=theta)
    d_BA = pcm_distance(B, A, theta=theta)
    torch.testing.assert_close(d_AB, d_BA)


def test_bwcm_distance_symmetry():
    A = random_spd(n=3)
    B = random_spd(n=3)
    d_AB = bwcm_distance(A, B)
    d_BA = bwcm_distance(B, A)
    torch.testing.assert_close(d_AB, d_BA)


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_pcm_distance_triangle_inequality(theta):
    A = random_spd(n=3)
    B = random_spd(n=3)
    C = random_spd(n=3)

    d_AC = pcm_distance(A, C, theta=theta)
    d_AB = pcm_distance(A, B, theta=theta)
    d_BC = pcm_distance(B, C, theta=theta)

    assert d_AC <= d_AB + d_BC + 1e-12


def test_bwcm_distance_triangle_inequality():
    A = random_spd(n=3)
    B = random_spd(n=3)
    C = random_spd(n=3)

    d_AC = bwcm_distance(A, C)
    d_AB = bwcm_distance(A, B)
    d_BC = bwcm_distance(B, C)

    assert d_AC <= d_AB + d_BC + 1e-12


def test_bwcm_diagonal_equivalence():
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
    bwcm_dist = bwcm_distance(D_A @ D_A, D_B @ D_B)

    torch.testing.assert_close(bwcm_dist, bw_dist)


def test_distance_broadcasting():
    A = random_spd(n=3, batch_shape=(5, 1))
    B = random_spd(n=3, batch_shape=(1, 4))
    dist = pcm_distance(A, B, theta=0.5)
    assert dist.shape == (5, 4)

    # Verify individual entries
    for i in range(5):
        for j in range(4):
            d_ij = pcm_distance(A[i, 0], B[0, j], theta=0.5)
            torch.testing.assert_close(dist[i, j], d_ij)


# =============================================================================
# Coordinate Maps & Riemannian Log/Exp
# =============================================================================


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 2.0])
def test_pcm_coords_roundtrip(theta):
    X = random_spd(n=4)
    Z = pcm_coords(X, theta=theta)
    X_rec = pcm_inv_coords(Z, theta=theta)
    torch.testing.assert_close(X_rec, X)


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0, 2.0])
def test_pcm_log_exp_roundtrip_none_p(theta):
    X = random_spd(n=4, batch_shape=(3,))
    V = log_map_pcm(X, P=None, theta=theta)
    X_rec = exp_map_pcm(V, P=None, theta=theta)
    torch.testing.assert_close(X_rec, X)


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_pcm_log_exp_roundtrip_given_p(theta):
    X = random_spd(n=3, batch_shape=(2,))
    P = random_spd(n=3, batch_shape=(2,))
    V = log_map_pcm(X, P=P, theta=theta)
    X_rec = exp_map_pcm(V, P=P, theta=theta)
    torch.testing.assert_close(X_rec, X)


def test_bwcm_log_exp_roundtrip():
    X = random_spd(n=3)
    P = random_spd(n=3)

    # Reference None
    V_none = log_map_bwcm(X, P=None)
    X_rec_none = exp_map_bwcm(V_none, P=None)
    torch.testing.assert_close(X_rec_none, X)

    # Reference P
    V_p = log_map_bwcm(X, P=P)
    X_rec_p = exp_map_bwcm(V_p, P=P)
    torch.testing.assert_close(X_rec_p, X)


def test_log_at_reference_is_zero():
    P = random_spd(n=3)
    V = log_map_pcm(P, P=P, theta=0.5)
    torch.testing.assert_close(V, torch.zeros_like(V))

    eye = torch.eye(3, dtype=torch.float64)
    V_ident = log_map_pcm(eye, P=None, theta=0.5)
    torch.testing.assert_close(V_ident, torch.zeros_like(V_ident))


# =============================================================================
# Geodesics
# =============================================================================


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_pcm_geodesic_endpoints(theta):
    A = random_spd(n=3)
    B = random_spd(n=3)

    g0 = pcm_geodesic(A, B, 0.0, theta=theta)
    g1 = pcm_geodesic(A, B, 1.0, theta=theta)

    torch.testing.assert_close(g0, A)
    torch.testing.assert_close(g1, B)


def test_bwcm_geodesic_endpoints():
    A = random_spd(n=3)
    B = random_spd(n=3)

    g0 = bwcm_geodesic(A, B, 0.0)
    g1 = bwcm_geodesic(A, B, 1.0)

    torch.testing.assert_close(g0, A)
    torch.testing.assert_close(g1, B)


@pytest.mark.parametrize("theta", [0.25, 0.5, 1.0])
def test_pcm_geodesic_midpoint_equals_mean(theta):
    A = random_spd(n=3)
    B = random_spd(n=3)

    mid = pcm_geodesic(A, B, 0.5, theta=theta)
    mean = pcm_mean(torch.stack([A, B]), theta=theta)

    torch.testing.assert_close(mid, mean)


def test_bwcm_geodesic_midpoint_equals_mean():
    A = random_spd(n=3)
    B = random_spd(n=3)

    mid = bwcm_geodesic(A, B, 0.5)
    mean = bwcm_mean(torch.stack([A, B]))

    torch.testing.assert_close(mid, mean)


@pytest.mark.parametrize("t", [0.25, 0.5, 0.75])
def test_pcm_geodesic_distance_proportional(t):
    A = random_spd(n=3)
    B = random_spd(n=3)
    theta = 0.5

    d_total = pcm_distance(A, B, theta=theta)
    gt = pcm_geodesic(A, B, t, theta=theta)
    d_segment = pcm_distance(A, gt, theta=theta)

    torch.testing.assert_close(d_segment, t * d_total)


# =============================================================================
# Fréchet Mean
# =============================================================================


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_pcm_mean_single_matrix(theta):
    X = random_spd(n=3).unsqueeze(0)
    mean = pcm_mean(X, theta=theta)
    torch.testing.assert_close(mean, X.squeeze(0))


def test_pcm_mean_uniform_vs_none_weights():
    X = random_spd(n=3, batch_shape=(4,))
    uniform_weights = torch.ones(4, dtype=torch.float64) / 4.0

    mean_none = pcm_mean(X, weights=None, theta=0.5)
    mean_uniform = pcm_mean(X, weights=uniform_weights, theta=0.5)

    torch.testing.assert_close(mean_none, mean_uniform)


def test_pcm_mean_weighted():
    A = random_spd(n=3)
    B = random_spd(n=3)
    X = torch.stack([A, B])
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)

    mean = pcm_mean(X, weights=weights, theta=0.5)
    geodesic_pt = pcm_geodesic(A, B, 0.75, theta=0.5)

    torch.testing.assert_close(mean, geodesic_pt)


def test_pcm_mean_batched_dim():
    X = random_spd(n=3, batch_shape=(2, 5))
    mean = pcm_mean(X, dim=1, keepdim=False, theta=0.5)
    assert mean.shape == (2, 3, 3)

    mean_keep = pcm_mean(X, dim=1, keepdim=True, theta=0.5)
    assert mean_keep.shape == (2, 1, 3, 3)

    for b in range(2):
        expected_b = pcm_mean(X[b], dim=0, theta=0.5)
        torch.testing.assert_close(mean[b], expected_b)


# =============================================================================
# Autograd Gradcheck
# =============================================================================


def test_pcm_distance_gradcheck():
    A = random_spd(n=3).requires_grad_(True)
    B = random_spd(n=3).requires_grad_(True)

    assert gradcheck(lambda a, b: pcm_distance(a, b, theta=0.5), (A, B))


def test_bwcm_distance_gradcheck():
    A = random_spd(n=3).requires_grad_(True)
    B = random_spd(n=3).requires_grad_(True)

    assert gradcheck(bwcm_distance, (A, B))


def test_pcm_geodesic_gradcheck():
    A = random_spd(n=3).requires_grad_(True)
    B = random_spd(n=3).requires_grad_(True)
    t = 0.4

    assert gradcheck(lambda a, b: pcm_geodesic(a, b, t, theta=0.5), (A, B))


def test_pcm_mean_gradcheck():
    X = random_spd(n=3, batch_shape=(3,)).requires_grad_(True)
    assert gradcheck(lambda x: pcm_mean(x, theta=0.5), (X,))


def test_pcm_log_gradcheck():
    X = random_spd(n=3).requires_grad_(True)
    P = random_spd(n=3).requires_grad_(True)

    # P is None
    assert gradcheck(lambda x: log_map_pcm(x, P=None, theta=0.5), (X,))
    # P is given
    assert gradcheck(lambda x, p: log_map_pcm(x, P=p, theta=0.5), (X, P))


def test_pcm_exp_gradcheck():
    X = random_spd(n=3)
    P = random_spd(n=3).requires_grad_(True)

    V_none = log_map_pcm(X, None, theta=0.5).requires_grad_(True)
    assert gradcheck(lambda v: exp_map_pcm(v, P=None, theta=0.5), (V_none,))

    V_p = log_map_pcm(X, P.detach(), theta=0.5).requires_grad_(True)
    assert gradcheck(lambda v, p: exp_map_pcm(v, P=p, theta=0.5), (V_p, P))


# =============================================================================
# Error Handling & Edge Cases
# =============================================================================


def test_zero_theta_raises():
    A = random_spd(n=3)
    B = random_spd(n=3)
    with pytest.raises(ValueError, match="theta must be non-zero"):
        pcm_distance(A, B, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        pcm_geodesic(A, B, 0.5, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        pcm_mean(torch.stack([A, B]), theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        log_map_pcm(A, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        exp_map_pcm(A, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        pcm_coords(A, theta=0.0)

    with pytest.raises(ValueError, match="theta must be non-zero"):
        pcm_inv_coords(A, theta=0.0)


def test_invalid_mean_dim_raises():
    X = random_spd(n=3, batch_shape=(2,))
    with pytest.raises(ValueError, match="batch dimension"):
        pcm_mean(X, dim=-1)

    with pytest.raises(ValueError, match="batch dimension"):
        pcm_mean(X, dim=-2)

    with pytest.raises(ValueError, match="at least 3 dimensions"):
        pcm_mean(random_spd(n=3))
