"""Comparison tests between spd_learn (PyTorch) and pyriemann (NumPy).

Validates that spd_learn's implementations produce numerically equivalent
results to pyriemann's established NumPy implementations for all shared
SPD matrix operations: matrix functions, distances, geodesics, means,
and log/exp maps.
"""

import numpy as np
import pytest
import torch


pyriemann = pytest.importorskip("pyriemann")

from pyriemann.utils.base import expm, invsqrtm, logm, powm, sqrtm  # noqa: E402
from pyriemann.utils.distance import (  # noqa: E402
    distance_logchol,
    distance_logeuclid,
    distance_riemann,
    distance_wasserstein,
)
from pyriemann.utils.geodesic import (  # noqa: E402
    geodesic_logchol,
    geodesic_logeuclid,
    geodesic_riemann,
    geodesic_wasserstein,
)
from pyriemann.utils.mean import (  # noqa: E402
    mean_logchol,
    mean_logeuclid,
    mean_riemann,
    mean_wasserstein,
)
from pyriemann.utils.tangentspace import exp_map_riemann, log_map_riemann  # noqa: E402

from spd_learn.functional import (  # noqa: E402
    airm_distance,
    airm_geodesic,
    bures_wasserstein_distance,
    bures_wasserstein_geodesic,
    bures_wasserstein_mean,
    exp_map_airm,
    log_cholesky_distance,
    log_cholesky_geodesic,
    log_cholesky_mean,
    log_euclidean_distance,
    log_euclidean_geodesic,
    log_map_airm,
    matrix_exp,
    matrix_inv_sqrt,
    matrix_log,
    matrix_power,
    matrix_sqrt,
)
from spd_learn.functional.batchnorm import karcher_mean_iteration  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZES = [3, 5]
SEED = 42

STRICT_TOL = dict(atol=1e-7, rtol=1e-6)
ITERATIVE_TOL = dict(atol=1e-6, rtol=1e-5)

# ---------------------------------------------------------------------------
# Mapping dicts: spd_learn <-> pyriemann
# ---------------------------------------------------------------------------

MATRIX_OPS = {
    "log": {"spd_learn": matrix_log, "pyriemann": logm},
    "exp": {"spd_learn": matrix_exp, "pyriemann": expm},
    "sqrt": {"spd_learn": matrix_sqrt, "pyriemann": sqrtm},
    "inv_sqrt": {"spd_learn": matrix_inv_sqrt, "pyriemann": invsqrtm},
}

DISTANCE_PAIRS = {
    "riemann": {"spd_learn": airm_distance, "pyriemann": distance_riemann},
    "logeuclid": {"spd_learn": log_euclidean_distance, "pyriemann": distance_logeuclid},
    "wasserstein": {
        "spd_learn": bures_wasserstein_distance,
        "pyriemann": distance_wasserstein,
    },
    "logchol": {"spd_learn": log_cholesky_distance, "pyriemann": distance_logchol},
}

GEODESIC_PAIRS = {
    "riemann": {"spd_learn": airm_geodesic, "pyriemann": geodesic_riemann},
    "logeuclid": {"spd_learn": log_euclidean_geodesic, "pyriemann": geodesic_logeuclid},
    "logchol": {"spd_learn": log_cholesky_geodesic, "pyriemann": geodesic_logchol},
    "wasserstein": {
        "spd_learn": bures_wasserstein_geodesic,
        "pyriemann": geodesic_wasserstein,
    },
}

MEAN_PAIRS = {
    "logchol": {"spd_learn": log_cholesky_mean, "pyriemann": mean_logchol},
    "wasserstein": {"spd_learn": bures_wasserstein_mean, "pyriemann": mean_wasserstein},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_spd_np(n, rng):
    """Generate a random SPD matrix as a NumPy array."""
    A = rng.randn(n, n)
    return A @ A.T + 0.1 * np.eye(n)


def to_torch(arr):
    """Convert a NumPy array to a float64 PyTorch tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float64)


def to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 1. Base matrix operations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "op_name", list(MATRIX_OPS.keys()), ids=list(MATRIX_OPS.keys())
)
def test_matrix_op(op_name, n):
    """spd_learn matrix op matches pyriemann on a random SPD matrix."""
    rng = np.random.RandomState(SEED)
    A_np = make_spd_np(n, rng)

    spd_fn = MATRIX_OPS[op_name]["spd_learn"]
    pyr_fn = MATRIX_OPS[op_name]["pyriemann"]

    # For exp, use a symmetric (log of SPD) input instead of SPD directly
    if op_name == "exp":
        A_np = logm(A_np)

    expected = pyr_fn(A_np)
    result = to_numpy(spd_fn.apply(to_torch(A_np)))

    np.testing.assert_allclose(result, expected, **STRICT_TOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("alpha", [0.5, 2.0, -1.0, 0.3])
def test_matrix_power(n, alpha):
    """spd_learn matrix_power matches pyriemann powm."""
    rng = np.random.RandomState(SEED)
    A_np = make_spd_np(n, rng)

    expected = powm(A_np, alpha)
    result = to_numpy(matrix_power.apply(to_torch(A_np), alpha))

    np.testing.assert_allclose(result, expected, **STRICT_TOL)


# ---------------------------------------------------------------------------
# 2. Distances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize(
    "metric", list(DISTANCE_PAIRS.keys()), ids=list(DISTANCE_PAIRS.keys())
)
def test_distance(metric, n):
    """spd_learn distance matches pyriemann for a pair of SPD matrices."""
    rng = np.random.RandomState(SEED)
    A_np, B_np = make_spd_np(n, rng), make_spd_np(n, rng)

    spd_fn = DISTANCE_PAIRS[metric]["spd_learn"]
    pyr_fn = DISTANCE_PAIRS[metric]["pyriemann"]

    expected = pyr_fn(A_np, B_np)
    result = to_numpy(spd_fn(to_torch(A_np), to_torch(B_np)))

    np.testing.assert_allclose(result, expected, **STRICT_TOL)


# ---------------------------------------------------------------------------
# 3. Geodesics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize(
    "metric", list(GEODESIC_PAIRS.keys()), ids=list(GEODESIC_PAIRS.keys())
)
def test_geodesic(metric, n, alpha):
    """spd_learn geodesic matches pyriemann at various interpolation points."""
    rng = np.random.RandomState(SEED)
    A_np, B_np = make_spd_np(n, rng), make_spd_np(n, rng)

    spd_fn = GEODESIC_PAIRS[metric]["spd_learn"]
    pyr_fn = GEODESIC_PAIRS[metric]["pyriemann"]

    expected = pyr_fn(A_np, B_np, alpha=alpha)
    result = to_numpy(spd_fn(to_torch(A_np), to_torch(B_np), alpha))

    np.testing.assert_allclose(result, expected, **STRICT_TOL)


# ---------------------------------------------------------------------------
# 4. Means
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_mean_logeuclid(n):
    """Manual log-euclidean mean matches pyriemann mean_logeuclid."""
    rng = np.random.RandomState(SEED)
    X_np = np.stack([make_spd_np(n, rng) for _ in range(6)])
    X_t = to_torch(X_np)

    expected = mean_logeuclid(X_np)
    result = to_numpy(matrix_exp.apply(matrix_log.apply(X_t).mean(dim=0)))

    np.testing.assert_allclose(result, expected, **STRICT_TOL)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("metric", list(MEAN_PAIRS.keys()), ids=list(MEAN_PAIRS.keys()))
def test_mean(metric, n):
    """spd_learn mean matches pyriemann for a batch of SPD matrices."""
    rng = np.random.RandomState(SEED)
    X_np = np.stack([make_spd_np(n, rng) for _ in range(6)])
    X_t = to_torch(X_np)

    spd_fn = MEAN_PAIRS[metric]["spd_learn"]
    pyr_fn = MEAN_PAIRS[metric]["pyriemann"]

    expected = pyr_fn(X_np)
    result = to_numpy(spd_fn(X_t))

    np.testing.assert_allclose(result, expected, **ITERATIVE_TOL)


@pytest.mark.parametrize("n", SIZES)
def test_mean_riemann(n):
    """Iterative Karcher mean matches pyriemann mean_riemann."""
    rng = np.random.RandomState(SEED)
    X_np = np.stack([make_spd_np(n, rng) for _ in range(6)])
    X_t = to_torch(X_np)

    expected = mean_riemann(X_np)

    current = X_t.mean(dim=0, keepdim=True)
    for _ in range(50):
        current = karcher_mean_iteration(X_t, current, detach=True)
    result = to_numpy(current.squeeze(0))

    np.testing.assert_allclose(result, expected, **ITERATIVE_TOL)


# ---------------------------------------------------------------------------
# 5. Log / Exp maps (AIRM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_log_map_airm(n):
    """spd_learn log_map_airm matches pyriemann log_map_riemann (C12=True)."""
    rng = np.random.RandomState(SEED)
    P_np, Q_np = make_spd_np(n, rng), make_spd_np(n, rng)

    expected = log_map_riemann(Q_np[None], P_np, C12=True)[0]
    result = to_numpy(log_map_airm(to_torch(P_np), to_torch(Q_np)))

    np.testing.assert_allclose(result, expected, **STRICT_TOL)


@pytest.mark.parametrize("n", SIZES)
def test_exp_map_airm(n):
    """spd_learn exp_map_airm matches pyriemann exp_map_riemann (Cm12=True)."""
    rng = np.random.RandomState(SEED)
    P_np, Q_np = make_spd_np(n, rng), make_spd_np(n, rng)

    V_np = log_map_riemann(Q_np[None], P_np, C12=True)[0]

    expected = exp_map_riemann(V_np[None], P_np, Cm12=True)[0]
    result = to_numpy(exp_map_airm(to_torch(P_np), to_torch(V_np)))

    np.testing.assert_allclose(result, expected, **STRICT_TOL)


@pytest.mark.parametrize("n", SIZES)
def test_log_exp_roundtrip(n):
    """Cross-library roundtrip: log with pyriemann, exp with spd_learn."""
    rng = np.random.RandomState(SEED)
    P_np, Q_np = make_spd_np(n, rng), make_spd_np(n, rng)

    V_np = log_map_riemann(Q_np[None], P_np, C12=True)[0]
    recovered = to_numpy(exp_map_airm(to_torch(P_np), to_torch(V_np)))

    np.testing.assert_allclose(recovered, Q_np, **STRICT_TOL)


# ---------------------------------------------------------------------------
# 6. Batch consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
def test_batched_matrix_log(n):
    """Batched matrix_log matches per-element pyriemann logm."""
    rng = np.random.RandomState(SEED)
    X_list = [make_spd_np(n, rng) for _ in range(4)]
    X_batch_t = to_torch(np.stack(X_list))

    result = to_numpy(matrix_log.apply(X_batch_t))

    for i, X_np in enumerate(X_list):
        np.testing.assert_allclose(result[i], logm(X_np), **STRICT_TOL)


@pytest.mark.parametrize("n", SIZES)
def test_batched_distances(n):
    """Batched airm_distance matches per-pair pyriemann distance_riemann."""
    rng = np.random.RandomState(SEED)
    A_list = [make_spd_np(n, rng) for _ in range(4)]
    B_list = [make_spd_np(n, rng) for _ in range(4)]

    result = to_numpy(
        airm_distance(to_torch(np.stack(A_list)), to_torch(np.stack(B_list)))
    )

    for i in range(4):
        np.testing.assert_allclose(
            result[i], distance_riemann(A_list[i], B_list[i]), **STRICT_TOL
        )
