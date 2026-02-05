# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Riemannian metrics for SPD manifolds.

This subpackage provides implementations of various Riemannian metrics
on the manifold of Symmetric Positive Definite (SPD) matrices.

Each metric module provides a consistent API with the following operations:

- ``{metric}_distance(A, B)``: Geodesic distance between SPD matrices
- ``{metric}_geodesic(A, B, t)``: Geodesic interpolation at parameter t
- ``{metric}_mean(matrices, weights)``: Weighted Fréchet mean
- ``exp_map_{metric}(P, V)``: Riemannian exponential map
- ``log_map_{metric}(P, Q)``: Riemannian logarithmic map

Available metrics:

- **AIRM** (Affine-Invariant Riemannian Metric): The natural metric with
  affine invariance, forming a Hadamard manifold with nonpositive curvature.
- **Log-Euclidean**: A computationally efficient metric that induces a flat
  (Euclidean) geometry via the matrix logarithm.
- **Bures-Wasserstein**: The optimal transport metric between centered
  Gaussians, with positive sectional curvature.
- **Log-Cholesky**: An efficient metric using the Cholesky decomposition,
  avoiding eigenvalue computations.
"""

from .affine_invariant import (
    airm_distance,
    airm_geodesic,
    exp_map_airm,
    log_map_airm,
    spd_egrad2rgrad,
)
from .bures_wasserstein import (
    bures_wasserstein_distance,
    bures_wasserstein_geodesic,
    bures_wasserstein_mean,
    bures_wasserstein_transport,
)
from .log_cholesky import (
    cholesky_exp,
    cholesky_log,
    log_cholesky_distance,
    log_cholesky_geodesic,
    log_cholesky_mean,
)
from .log_euclidean import (
    exp_map_lem,
    log_euclidean_distance,
    log_euclidean_geodesic,
    log_euclidean_mean,
    log_euclidean_multiply,
    log_euclidean_scalar_multiply,
    log_map_lem,
)


__all__ = [
    # AIRM metric
    "airm_distance",
    "airm_geodesic",
    "exp_map_airm",
    "log_map_airm",
    "spd_egrad2rgrad",
    # Log-Euclidean metric
    "log_euclidean_distance",
    "log_euclidean_geodesic",
    "log_euclidean_mean",
    "log_euclidean_multiply",
    "log_euclidean_scalar_multiply",
    "exp_map_lem",
    "log_map_lem",
    # Bures-Wasserstein metric
    "bures_wasserstein_distance",
    "bures_wasserstein_geodesic",
    "bures_wasserstein_mean",
    "bures_wasserstein_transport",
    # Log-Cholesky metric
    "cholesky_log",
    "cholesky_exp",
    "log_cholesky_distance",
    "log_cholesky_geodesic",
    "log_cholesky_mean",
]
