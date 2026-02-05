# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Functional API for SPD matrix operations.

This module provides differentiable operations for Symmetric Positive Definite (SPD)
matrices, organized into:

- **Core operations**: Matrix logarithm, exponential, power, square root
- **Metrics**: Riemannian metrics (AIRM, Log-Euclidean, Bures-Wasserstein, Log-Cholesky)
- **Transport**: Parallel transport operations
- **Covariance**: Covariance estimation functions
- **Regularization**: Covariance regularization utilities
- **Vectorization**: Batch vectorization and (un)vectorization helpers
- **Numerical**: Numerical stability configuration
"""

from .autograd import modeig_backward, modeig_forward
from .batchnorm import (
    karcher_mean_iteration,
    spd_centering,
    spd_rebiasing,
    tangent_space_variance,
)
from .bilinear import bimap_increase_dim, bimap_transform
from .core import (
    abs_eigvals,
    clamp_eigvals,
    inv_softplus,
    matrix_exp,
    matrix_inv_softplus,
    matrix_inv_sqrt,
    matrix_log,
    matrix_power,
    matrix_softplus,
    matrix_sqrt,
    matrix_sqrt_inv,
    orthogonal_polar_factor,
    softplus,
    sym_to_upper,
    vec_to_sym,
)
from .covariance import covariance, cross_covariance, real_covariance, sample_covariance
from .dropout import dropout_spd
from .metrics import (
    # AIRM metric
    airm_distance,
    airm_geodesic,
    # Bures-Wasserstein metric
    bures_wasserstein_distance,
    bures_wasserstein_geodesic,
    bures_wasserstein_mean,
    bures_wasserstein_transport,
    # Log-Cholesky metric
    cholesky_exp,
    cholesky_log,
    exp_map_airm,
    # Log-Euclidean metric
    exp_map_lem,
    log_cholesky_distance,
    log_cholesky_geodesic,
    log_cholesky_mean,
    log_euclidean_distance,
    log_euclidean_geodesic,
    log_euclidean_mean,
    log_euclidean_multiply,
    log_euclidean_scalar_multiply,
    log_map_airm,
    log_map_lem,
    spd_egrad2rgrad,
)
from .numerical import (
    NumericalConfig,
    NumericalContext,
    check_spd_eigenvalues,
    get_epsilon,
    get_epsilon_tensor,
    get_loewner_threshold,
    is_half_precision,
    numerical_config,
    recommend_dtype_for_spd,
    safe_clamp_eigenvalues,
)
from .parallel_transport import (
    parallel_transport_airm,
    parallel_transport_lem,
    parallel_transport_log_cholesky,
    pole_ladder,
    schild_ladder,
    transport_tangent_vector,
)
from .regularize import ledoit_wolf, shrinkage_covariance, trace_normalization
from .riemannian_pgd import spd_rpgd_attack
from .utils import ensure_sym, unvec_batch, vec_batch
from .wavelet import compute_gabor_wavelet


__all__ = [
    # Core operations
    "matrix_log",
    "matrix_exp",
    "matrix_softplus",
    "matrix_inv_softplus",
    "softplus",
    "inv_softplus",
    "clamp_eigvals",
    "abs_eigvals",
    "matrix_power",
    "matrix_sqrt",
    "matrix_inv_sqrt",
    "matrix_sqrt_inv",
    "orthogonal_polar_factor",
    "ensure_sym",
    "vec_to_sym",
    "sym_to_upper",
    # AIRM metric
    "airm_distance",
    "airm_geodesic",
    "exp_map_airm",
    "log_map_airm",
    # Log-Euclidean metric
    "log_euclidean_distance",
    "log_euclidean_geodesic",
    "log_euclidean_mean",
    "log_euclidean_multiply",
    "log_euclidean_scalar_multiply",
    "exp_map_lem",
    "log_map_lem",
    "spd_egrad2rgrad",
    # Log-Cholesky metric
    "cholesky_log",
    "cholesky_exp",
    "log_cholesky_distance",
    "log_cholesky_mean",
    "log_cholesky_geodesic",
    # Bures-Wasserstein metric
    "bures_wasserstein_distance",
    "bures_wasserstein_geodesic",
    "bures_wasserstein_mean",
    "bures_wasserstein_transport",
    # Parallel transport
    "parallel_transport_airm",
    "parallel_transport_lem",
    "parallel_transport_log_cholesky",
    "schild_ladder",
    "pole_ladder",
    "transport_tangent_vector",
    # Covariance estimation
    "covariance",
    "sample_covariance",
    "real_covariance",
    "cross_covariance",
    # Regularization
    "trace_normalization",
    "ledoit_wolf",
    "shrinkage_covariance",
    # Batch normalization
    "karcher_mean_iteration",
    "spd_centering",
    "spd_rebiasing",
    "tangent_space_variance",
    # Bilinear operations
    "bimap_transform",
    "bimap_increase_dim",
    # Wavelet operations
    "compute_gabor_wavelet",
    # Dropout
    "dropout_spd",
    # Autograd utilities
    "modeig_backward",
    "modeig_forward",
    # Vectorization utilities
    "vec_batch",
    "unvec_batch",
    # Numerical stability configuration
    "numerical_config",
    "NumericalConfig",
    "NumericalContext",
    "get_epsilon",
    "get_epsilon_tensor",
    "get_loewner_threshold",
    "safe_clamp_eigenvalues",
    "check_spd_eigenvalues",
    "is_half_precision",
    "recommend_dtype_for_spd",
    # Attacks
    "spd_rpgd_attack",
]
