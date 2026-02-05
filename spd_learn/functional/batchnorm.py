# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Functional operations for SPD batch normalization.

This module provides stateless mathematical operations for Riemannian batch
normalization on SPD manifolds. These functions implement the core computations
used by the batch normalization modules.

Functions
---------
karcher_mean_iteration
    Single iteration of the Karcher (Fréchet) mean algorithm.
spd_centering
    Center SPD matrices around a given mean via congruence transformation.
tangent_space_variance
    Compute variance of SPD matrices in the tangent space.

See Also
--------
:class:`~spd_learn.modules.SPDBatchNormMean` : Mean-only Riemannian batch normalization.
:class:`~spd_learn.modules.SPDBatchNormMeanVar` : Full Riemannian batch normalization.
"""

import torch

from .core import matrix_exp, matrix_log, matrix_sqrt_inv


def karcher_mean_iteration(
    X: torch.Tensor,
    current_mean: torch.Tensor,
    detach: bool = True,
) -> torch.Tensor:
    r"""Perform one iteration of the Karcher mean algorithm.

    The Karcher (Fréchet) mean on the SPD manifold is the minimizer of the sum
    of squared geodesic distances. This function performs one iteration of the
    iterative algorithm to compute it.

    Given a current estimate :math:`M` of the mean, the update is:

    .. math::

        M_{\text{new}} = M^{1/2} \exp\left(\frac{1}{N} \sum_{i=1}^N
        \log(M^{-1/2} X_i M^{-1/2})\right) M^{1/2}

    Parameters
    ----------
    X : torch.Tensor
        Batch of SPD matrices with shape `(batch_size, ..., n, n)`.
    current_mean : torch.Tensor
        Current estimate of the Karcher mean with shape `(1, ..., n, n)`.
    detach : bool, default=True
        If True, detaches ``current_mean`` from the computational graph before
        computing the update. Set to False when gradients with respect to the
        mean are needed.

    Returns
    -------
    torch.Tensor
        Updated Karcher mean estimate with shape `(1, ..., n, n)`.

    Notes
    -----
    For well-conditioned data, a single iteration often suffices. The algorithm
    converges quadratically near the solution.

    When ``detach=False``, gradients flow through the entire computation,
    including the matrix square root and inverse square root of the current mean.

    See Also
    --------
    :func:`spd_centering` : Center matrices around a mean.
    :func:`~spd_learn.functional.airm_geodesic` : Geodesic under AIRM.

    References
    ----------
    See :cite:p:`pennec2006riemannian` for details on Karcher mean computation.
    """
    mean_input = current_mean.detach() if detach else current_mean
    mean_sqrt, mean_invsqrt = matrix_sqrt_inv.apply(mean_input)
    # Transport to tangent space at identity
    X_tangent = matrix_log.apply(mean_invsqrt @ X @ mean_invsqrt)
    # Compute mean in tangent space
    mean_tangent = X_tangent.mean(dim=0, keepdim=True)
    # Map back to manifold
    new_mean = mean_sqrt @ matrix_exp.apply(mean_tangent) @ mean_sqrt
    return new_mean


def spd_centering(
    X: torch.Tensor,
    mean_invsqrt: torch.Tensor,
) -> torch.Tensor:
    r"""Center SPD matrices around a mean via congruence transformation.

    Applies the congruence transformation to center SPD matrices:

    .. math::

        \tilde{X}_i = M^{-1/2} X_i M^{-1/2}

    This corresponds to parallel transport from the mean :math:`M` to the
    identity matrix under the affine-invariant Riemannian metric.

    Parameters
    ----------
    X : torch.Tensor
        Batch of SPD matrices with shape `(..., n, n)`.
    mean_invsqrt : torch.Tensor
        Inverse square root of the mean with shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Centered SPD matrices with shape `(..., n, n)`.

    Notes
    -----
    After centering, the Fréchet mean of the batch is (approximately) the
    identity matrix.

    See Also
    --------
    :func:`karcher_mean_iteration` : Compute the Karcher mean.
    :func:`~spd_learn.functional.parallel_transport_airm` : Parallel transport under AIRM.
    """
    return mean_invsqrt @ X @ mean_invsqrt


def spd_rebiasing(
    X: torch.Tensor,
    bias_sqrt: torch.Tensor,
) -> torch.Tensor:
    r"""Apply learnable rebiasing to centered SPD matrices.

    Applies a congruence transformation to rebias centered SPD matrices:

    .. math::

        \hat{X}_i = B^{1/2} X_i B^{1/2}

    This corresponds to parallel transport from the identity to the bias
    matrix :math:`B` under the affine-invariant Riemannian metric.

    Parameters
    ----------
    X : torch.Tensor
        Batch of centered SPD matrices with shape `(..., n, n)`.
    bias_sqrt : torch.Tensor
        Square root of the bias SPD matrix with shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Rebiased SPD matrices with shape `(..., n, n)`.

    See Also
    --------
    :func:`spd_centering` : Center matrices around a mean.
    """
    return bias_sqrt @ X @ bias_sqrt


def tangent_space_variance(
    X_tangent: torch.Tensor,
    mean_tangent: torch.Tensor,
) -> torch.Tensor:
    r"""Compute scalar dispersion in the tangent space.

    Computes the mean squared Frobenius distance from the tangent space mean:

    .. math::

        \sigma^2 = \frac{1}{N} \sum_{i=1}^N \|V_i - \bar{V}\|_F^2

    where :math:`V_i = \log(M^{-1/2} X_i M^{-1/2})` are the tangent vectors.

    Parameters
    ----------
    X_tangent : torch.Tensor
        Batch of tangent vectors (symmetric matrices) with shape
        `(batch_size, ..., n, n)`.
    mean_tangent : torch.Tensor
        Mean tangent vector with shape `(1, ..., n, n)`.

    Returns
    -------
    torch.Tensor
        Scalar dispersion value (single number, not a variance matrix).
        This is the mean squared Frobenius distance from the tangent mean.

    Notes
    -----
    This scalar dispersion is used for dispersion normalization in SPD batch
    normalization.

    See Also
    --------
    :func:`karcher_mean_iteration` : Compute the Karcher mean.
    """
    diff = X_tangent - mean_tangent
    variance = (
        torch.norm(diff, p="fro", dim=(-2, -1), keepdim=True)
        .square()
        .mean(dim=0, keepdim=True)
        .squeeze(-1)
    )
    return variance


__all__ = [
    "karcher_mean_iteration",
    "spd_centering",
    "spd_rebiasing",
    "tangent_space_variance",
]
