"""Tests for transport and metric implementations: Bures-Wasserstein, Log-Cholesky, and Parallel Transport."""

import pytest
import torch

from spd_learn.functional import (
    bures_wasserstein_distance,
    bures_wasserstein_geodesic,
    bures_wasserstein_mean,
    bures_wasserstein_transport,
    cholesky_exp,
    cholesky_log,
    log_cholesky_distance,
    log_cholesky_geodesic,
    log_cholesky_mean,
    log_euclidean_distance,
    parallel_transport_airm,
    parallel_transport_lem,
    pole_ladder,
    schild_ladder,
    transport_tangent_vector,
)


# --- Helper functions ---


def make_spd(n, batch_size=None, dtype=torch.float64, device="cpu"):
    """Create random SPD matrices."""
    if batch_size is not None:
        A = torch.randn(batch_size, n, n, dtype=dtype, device=device)
    else:
        A = torch.randn(n, n, dtype=dtype, device=device)
    return A @ A.transpose(-1, -2) + 0.1 * torch.eye(n, dtype=dtype, device=device)


def make_symmetric(n, batch_size=None, dtype=torch.float64, device="cpu"):
    """Create random symmetric matrices (tangent vectors)."""
    if batch_size is not None:
        A = torch.randn(batch_size, n, n, dtype=dtype, device=device)
    else:
        A = torch.randn(n, n, dtype=dtype, device=device)
    return (A + A.transpose(-1, -2)) / 2


def airm_inner_product(U, V, P):
    """Compute the AIRM inner product <U, V>_P = tr(P^{-1} U P^{-1} V)."""
    P_inv = torch.linalg.inv(P)
    return torch.trace(P_inv @ U @ P_inv @ V)


# --- Distance function tests ---


@pytest.mark.parametrize(
    "distance_fn",
    [bures_wasserstein_distance, log_cholesky_distance, log_euclidean_distance],
    ids=["bures_wasserstein", "log_cholesky", "log_euclidean"],
)
def test_distance_identity(distance_fn):
    """Distance from a matrix to itself should be zero."""
    A = make_spd(3)
    d = distance_fn(A, A)
    assert torch.allclose(d, torch.tensor(0.0, dtype=A.dtype), atol=1e-6)


@pytest.mark.parametrize(
    "distance_fn",
    [bures_wasserstein_distance, log_cholesky_distance, log_euclidean_distance],
    ids=["bures_wasserstein", "log_cholesky", "log_euclidean"],
)
def test_distance_symmetry(distance_fn):
    """Distance should be symmetric: d(A,B) = d(B,A)."""
    A = make_spd(3)
    B = make_spd(3)
    d_AB = distance_fn(A, B)
    d_BA = distance_fn(B, A)
    assert torch.allclose(d_AB, d_BA, atol=1e-6)


@pytest.mark.parametrize(
    "distance_fn",
    [bures_wasserstein_distance, log_cholesky_distance, log_euclidean_distance],
    ids=["bures_wasserstein", "log_cholesky", "log_euclidean"],
)
def test_distance_positivity(distance_fn):
    """Distance should be positive for distinct matrices."""
    A = make_spd(3)
    B = make_spd(3)
    d = distance_fn(A, B)
    assert d > 0


@pytest.mark.parametrize(
    "distance_fn",
    [bures_wasserstein_distance, log_cholesky_distance, log_euclidean_distance],
    ids=["bures_wasserstein", "log_cholesky", "log_euclidean"],
)
def test_distance_triangle_inequality(distance_fn):
    """Triangle inequality: d(A,C) <= d(A,B) + d(B,C)."""
    A = make_spd(3)
    B = make_spd(3)
    C = make_spd(3)
    d_AC = distance_fn(A, C)
    d_AB = distance_fn(A, B)
    d_BC = distance_fn(B, C)
    assert d_AC <= d_AB + d_BC + 1e-6  # Small tolerance for numerical errors


def test_bures_wasserstein_distance_batched():
    """Test batched distance computation."""
    A = make_spd(3, batch_size=10)
    B = make_spd(3, batch_size=10)
    d = bures_wasserstein_distance(A, B)
    assert d.shape == (10,)


def test_bures_wasserstein_distance_requires_grad():
    """Test that gradients flow through the distance."""
    A = make_spd(3).requires_grad_(True)
    B = make_spd(3)
    d = bures_wasserstein_distance(A, B)
    d.backward()
    assert A.grad is not None


# --- Geodesic tests ---


@pytest.mark.parametrize(
    "geodesic_fn",
    [bures_wasserstein_geodesic, log_cholesky_geodesic],
    ids=["bures_wasserstein", "log_cholesky"],
)
@pytest.mark.parametrize("t,expected_idx", [(0.0, 0), (1.0, 1)], ids=["t=0", "t=1"])
def test_geodesic_endpoints(geodesic_fn, t, expected_idx):
    """Geodesic at t=0 should return A, at t=1 should return B."""
    A = make_spd(3)
    B = make_spd(3)
    matrices = [A, B]
    result = geodesic_fn(A, B, t)
    assert torch.allclose(result, matrices[expected_idx], atol=1e-6)


@pytest.mark.parametrize(
    "geodesic_fn,distance_fn",
    [
        (bures_wasserstein_geodesic, bures_wasserstein_distance),
        (log_cholesky_geodesic, log_cholesky_distance),
    ],
    ids=["bures_wasserstein", "log_cholesky"],
)
def test_geodesic_distance_proportional_to_t(geodesic_fn, distance_fn):
    """Distance from A to gamma(t) should be proportional to t."""
    A = make_spd(3)
    B = make_spd(3)
    d_total = distance_fn(A, B)

    for t in [0.25, 0.5, 0.75]:
        G_t = geodesic_fn(A, B, t)
        d_t = distance_fn(A, G_t)
        expected = t * d_total
        assert torch.allclose(d_t, expected, rtol=0.15, atol=1e-6), (
            f"At t={t}: d(A, gamma(t))={d_t.item():.4f}, expected={expected.item():.4f}"
        )


@pytest.mark.parametrize("t", [0.25, 0.5, 0.75])
def test_bures_wasserstein_geodesic_spd(t):
    """Geodesic points should be SPD (positive eigenvalues)."""
    A = make_spd(3)
    B = make_spd(3)
    G = bures_wasserstein_geodesic(A, B, t)
    eigvals = torch.linalg.eigvalsh(G)
    assert (eigvals > 0).all()


# --- Mean tests ---


@pytest.mark.parametrize(
    "mean_fn",
    [bures_wasserstein_mean, log_cholesky_mean],
    ids=["bures_wasserstein", "log_cholesky"],
)
def test_mean_single_matrix(mean_fn):
    """Mean of a single matrix should be that matrix."""
    A = make_spd(3)
    mean = mean_fn(A.unsqueeze(0))
    assert torch.allclose(mean, A, atol=1e-5)


@pytest.mark.parametrize(
    "mean_fn",
    [bures_wasserstein_mean, log_cholesky_mean],
    ids=["bures_wasserstein", "log_cholesky"],
)
def test_mean_spd(mean_fn):
    """Mean should be SPD."""
    matrices = make_spd(3, batch_size=5)
    mean = mean_fn(matrices)
    eigvals = torch.linalg.eigvalsh(mean)
    assert (eigvals > 0).all()


@pytest.mark.parametrize(
    "mean_fn,distance_fn",
    [
        (bures_wasserstein_mean, bures_wasserstein_distance),
        (log_cholesky_mean, log_cholesky_distance),
    ],
    ids=["bures_wasserstein", "log_cholesky"],
)
def test_mean_minimizes_sum_of_squared_distances(mean_fn, distance_fn):
    """Mean should minimize sum of squared distances (Frechet mean property)."""
    matrices = make_spd(3, batch_size=5)
    mean = mean_fn(matrices)

    sum_sq_dist_mean = sum(distance_fn(matrices[i], mean) ** 2 for i in range(5))

    # Perturb mean slightly and check that sum increases
    perturbation = make_symmetric(3) * 0.01
    perturbed = mean + perturbation
    eigvals = torch.linalg.eigvalsh(perturbed)
    if (eigvals > 0).all():
        sum_sq_dist_perturbed = sum(
            distance_fn(matrices[i], perturbed) ** 2 for i in range(5)
        )
        assert sum_sq_dist_mean <= sum_sq_dist_perturbed + 1e-4


def test_bures_wasserstein_mean_identical_matrices():
    """Mean of identical matrices should be that matrix."""
    A = make_spd(3)
    matrices = A.unsqueeze(0).expand(5, -1, -1).clone()
    mean = bures_wasserstein_mean(matrices)
    assert torch.allclose(mean, A, atol=1e-5)


def test_bures_wasserstein_mean_with_weights():
    """Test weighted mean."""
    A = make_spd(3)
    B = make_spd(3)
    matrices = torch.stack([A, B])
    weights = torch.tensor([1.0, 0.0])
    mean = bures_wasserstein_mean(matrices, weights=weights)
    assert torch.allclose(mean, A, atol=1e-4)


# --- Bures-Wasserstein transport tests ---


def test_bures_wasserstein_transport_identity():
    """Transport A from A to B should give B."""
    A = make_spd(3)
    B = make_spd(3)
    transported = bures_wasserstein_transport(A, B, A)
    assert torch.allclose(transported, B, atol=1e-5)


# --- Cholesky log/exp tests ---


def test_cholesky_log_exp_inverse():
    """cholesky_exp should be the inverse of cholesky_log."""
    A = make_spd(4)
    log_chol = cholesky_log.apply(A)
    reconstructed = cholesky_exp.apply(log_chol)
    assert torch.allclose(A, reconstructed, atol=1e-6)


def test_cholesky_exp_log_inverse():
    """cholesky_log should be the inverse of cholesky_exp."""
    L = torch.randn(4, 4, dtype=torch.float64).tril()
    L = L + torch.diag(torch.abs(L.diag()) + 0.1)
    log_L = L.tril(-1) + torch.diag(L.diag().log())

    reconstructed_log = cholesky_log.apply(cholesky_exp.apply(log_L))
    assert torch.allclose(log_L, reconstructed_log, atol=1e-6)


def test_cholesky_log_batched():
    """Test batched Log-Cholesky."""
    A = make_spd(3, batch_size=5)
    log_chol = cholesky_log.apply(A)
    assert log_chol.shape == (5, 3, 3)


def test_cholesky_log_requires_grad():
    """Test gradient flow through cholesky_log."""
    A = make_spd(3).requires_grad_(True)
    log_chol = cholesky_log.apply(A)
    loss = log_chol.sum()
    loss.backward()
    assert A.grad is not None


# --- Parallel transport AIRM tests ---


def test_parallel_transport_airm_identity_reference():
    """Transport from P to P should be identity."""
    P = make_spd(3)
    V = make_symmetric(3)
    V_transported = parallel_transport_airm(V, P, P)
    assert torch.allclose(V, V_transported, atol=1e-5)


def test_parallel_transport_airm_preserves_symmetry():
    """Transported vector should be symmetric."""
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)
    V_transported = parallel_transport_airm(V, P, Q)
    assert torch.allclose(V_transported, V_transported.T, atol=1e-6)


def test_parallel_transport_airm_preserves_inner_product():
    """Parallel transport should preserve the Riemannian inner product.

    The AIRM inner product is: <U, V>_P = tr(P^{-1} U P^{-1} V)
    After transport: <Γ(U), Γ(V)>_Q = <U, V>_P
    """
    P = make_spd(3)
    Q = make_spd(3)
    U = make_symmetric(3)
    V = make_symmetric(3)

    inner_P = airm_inner_product(U, V, P)

    U_transported = parallel_transport_airm(U, P, Q)
    V_transported = parallel_transport_airm(V, P, Q)
    inner_Q = airm_inner_product(U_transported, V_transported, Q)

    assert torch.allclose(inner_P, inner_Q, rtol=1e-5, atol=1e-6), (
        f"Inner product not preserved: at P={inner_P.item():.6f}, at Q={inner_Q.item():.6f}"
    )


def test_parallel_transport_airm_roundtrip():
    """Transport P→Q then Q→P should recover the original vector."""
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)

    V_to_Q = parallel_transport_airm(V, P, Q)
    V_back = parallel_transport_airm(V_to_Q, Q, P)

    assert torch.allclose(V, V_back, atol=1e-5), (
        f"Roundtrip failed: max diff = {(V - V_back).abs().max().item():.2e}"
    )


def test_parallel_transport_airm_linearity():
    """Parallel transport should be linear: Γ(aU + bV) = aΓ(U) + bΓ(V)."""
    P = make_spd(3)
    Q = make_spd(3)
    U = make_symmetric(3)
    V = make_symmetric(3)
    a, b = 2.5, -1.3

    combined = a * U + b * V
    transported_combined = parallel_transport_airm(combined, P, Q)

    transported_U = parallel_transport_airm(U, P, Q)
    transported_V = parallel_transport_airm(V, P, Q)
    linear_combo_transported = a * transported_U + b * transported_V

    assert torch.allclose(transported_combined, linear_combo_transported, atol=1e-6)


def test_parallel_transport_airm_zero_vector():
    """Transport of zero vector should be zero."""
    P = make_spd(3)
    Q = make_spd(3)
    V = torch.zeros(3, 3, dtype=torch.float64)

    V_transported = parallel_transport_airm(V, P, Q)
    assert torch.allclose(V_transported, V, atol=1e-10)


def test_parallel_transport_airm_batched():
    """Test batched transport."""
    P = make_spd(3, batch_size=5)
    Q = make_spd(3, batch_size=5)
    V = make_symmetric(3, batch_size=5)
    V_transported = parallel_transport_airm(V, P, Q)
    assert V_transported.shape == (5, 3, 3)


@pytest.mark.parametrize(
    "grad_input",
    ["v", "p", "q", "all"],
    ids=["grad_wrt_v", "grad_wrt_p", "grad_wrt_q", "grad_wrt_all"],
)
def test_parallel_transport_airm_gradient(grad_input):
    """Gradient should flow through the specified inputs."""
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)

    if grad_input in ("v", "all"):
        V = V.requires_grad_(True)
    if grad_input in ("p", "all"):
        P = P.requires_grad_(True)
    if grad_input in ("q", "all"):
        Q = Q.requires_grad_(True)

    V_transported = parallel_transport_airm(V, P, Q)
    loss = V_transported.sum()
    loss.backward()

    if grad_input in ("v", "all"):
        assert V.grad is not None
    if grad_input in ("p", "all"):
        assert P.grad is not None
    if grad_input in ("q", "all"):
        assert Q.grad is not None


def test_parallel_transport_airm_gradcheck():
    """Numerical gradient check for tangent vector input."""
    P = make_spd(3, dtype=torch.float64)
    Q = make_spd(3, dtype=torch.float64)
    V = make_symmetric(3, dtype=torch.float64).requires_grad_(True)

    def transport_v_only(v):
        return parallel_transport_airm(v, P, Q)

    assert torch.autograd.gradcheck(transport_v_only, (V,), eps=1e-6)


# --- Parallel transport LEM tests ---


def test_parallel_transport_lem_is_identity():
    """LEM transport should be identity (flat geometry)."""
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)
    V_transported = parallel_transport_lem(V, P, Q)
    assert torch.allclose(V, V_transported, atol=1e-6)


# --- Schild's ladder tests ---


def test_schild_ladder_approximates_airm_for_small_vectors():
    """Schild's ladder should approximate AIRM transport for small tangent vectors.

    Note: Schild's ladder is a first-order approximation whose accuracy depends on
    the ratio |V|/d(P,Q). For small tangent vectors relative to the geodesic
    distance, the approximation is good.
    """
    P = make_spd(3)
    Q = make_spd(3)
    # Use very small tangent vector for good approximation
    V = make_symmetric(3) * 0.001

    V_airm = parallel_transport_airm(V, P, Q)
    V_schild = schild_ladder(V, P, Q, n_steps=10)

    rel_error = (V_airm - V_schild).norm() / V_airm.norm()
    assert rel_error < 0.05, (
        f"Schild ladder error too large for small V: {rel_error:.2%}"
    )


def test_schild_ladder_preserves_symmetry():
    """Schild's ladder should preserve symmetry."""
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)
    V_schild = schild_ladder(V, P, Q, n_steps=5)
    assert torch.allclose(V_schild, V_schild.T, atol=1e-5)


# --- Pole ladder tests ---


def test_pole_ladder_preserves_symmetry():
    """Pole ladder should preserve symmetry."""
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)
    V_pole = pole_ladder(V, P, Q)
    assert torch.allclose(V_pole, V_pole.T, atol=1e-5)


def test_pole_ladder_matches_airm():
    """Pole ladder should closely approximate AIRM transport.

    Pole ladder uses the geodesic midpoint for reflection, providing O(h^2)
    approximation error where h is the geodesic distance. For small distances
    (typical in tests), this gives excellent agreement with closed-form AIRM.
    """
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)

    V_airm = parallel_transport_airm(V, P, Q)
    V_pole = pole_ladder(V, P, Q)

    # Pole ladder should match AIRM transport very closely
    assert torch.allclose(V_airm, V_pole, rtol=1e-5, atol=1e-6)


# --- Transport tangent vector interface tests ---


@pytest.mark.parametrize(
    "metric", ["airm", "lem", "log_euclidean", "log_cholesky", "schild", "pole"]
)
def test_transport_tangent_vector_metric_selection(metric):
    """Test that different metrics can be selected."""
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)

    kwargs = {"n_steps": 5} if metric == "schild" else {}
    result = transport_tangent_vector(V, P, Q, metric=metric, **kwargs)
    assert result.shape == V.shape


def test_transport_tangent_vector_invalid_metric():
    """Test that invalid metric raises error."""
    P = make_spd(3)
    Q = make_spd(3)
    V = make_symmetric(3)

    with pytest.raises(ValueError):
        transport_tangent_vector(V, P, Q, metric="invalid")


# --- Metric consistency tests ---


def test_log_euclidean_vs_log_cholesky_scaling():
    """Both Log-Euclidean and Log-Cholesky should give nonzero distance for scaled identity."""
    eye = torch.eye(3, dtype=torch.float64)
    eye_scaled = 2 * eye

    d_le = log_euclidean_distance(eye, eye_scaled)
    d_lc = log_cholesky_distance(eye, eye_scaled)

    assert d_le > 0
    assert d_lc > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
