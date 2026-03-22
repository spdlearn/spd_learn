# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Functional operations for SPD batch normalization.

This module provides stateless mathematical operations for Riemannian batch
normalization on SPD manifolds. These functions implement the core computations
used by the batch normalization modules.

Functions
---------
frechet_mean
    Fréchet mean of SPD matrices under the AIRM via Karcher flow.
karcher_mean_iteration
    Single iteration of the Karcher (Fréchet) mean algorithm.
spd_centering
    Center SPD matrices around a given mean via congruence transformation.
spd_cholesky_congruence
    Congruence transformation using the Cholesky factor of an SPD matrix.
tangent_space_variance
    Compute variance of SPD matrices in the tangent space.
lie_group_variance
    Fréchet variance under a Lie group structure on the SPD manifold.

See Also
--------
:class:`~spd_learn.modules.SPDBatchNormMean` : Mean-only Riemannian batch normalization.
:class:`~spd_learn.modules.SPDBatchNormMeanVar` : Full Riemannian batch normalization.
"""

from typing import Optional, Tuple, Union

import torch

from .core import matrix_exp, matrix_log, matrix_sqrt_inv
from .utils import ensure_sym


def karcher_mean_iteration(
    X: torch.Tensor,
    current_mean: torch.Tensor,
    detach: bool = True,
    return_tangent: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
    return_tangent : bool, default=False
        If True, also returns the mean tangent update used in this Karcher step.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Updated Karcher mean estimate with shape `(1, ..., n, n)`. When
        ``return_tangent=True``, also returns the mean tangent update with the
        same shape.

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
    if return_tangent:
        return new_mean, mean_tangent
    return new_mean


def frechet_mean(
    X: torch.Tensor,
    max_iter: int = 1,
    weights: Optional[torch.Tensor] = None,
    return_distances: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Fréchet mean of SPD matrices under the AIRM via Karcher flow.

    Computes the minimizer of the sum of squared geodesic distances:

    .. math::

        \bar{X} = \arg\min_{G \in \mathcal{S}_{++}^n}
        \sum_{i=1}^{N} w_i \, d_{\text{AIRM}}^2(G, X_i)

    using iterative Karcher flow initialized from the (weighted) Euclidean mean.

    Parameters
    ----------
    X : torch.Tensor
        Batch of SPD matrices with shape ``(batch_size, ..., n, n)``.
    max_iter : int, default=1
        Number of Karcher flow iterations. A single iteration is often
        sufficient for batch normalization; use more (e.g. 50) when a
        high-accuracy mean is needed.
    weights : torch.Tensor, optional
        Per-sample weights with shape broadcastable to ``X``. When ``None``,
        uniform weights ``1/N`` are used.
    return_distances : bool, default=False
        If True, also returns the geodesic distances from each sample to
        the mean.

    Returns
    -------
    mean : torch.Tensor
        Fréchet mean with shape ``(1, ..., n, n)``.
    distances : torch.Tensor
        Only returned when ``return_distances=True``. Geodesic distances
        from each sample to the mean, with shape ``(batch_size, ...)``.

    See Also
    --------
    :func:`karcher_mean_iteration` : Single Karcher step (lower-level).
    :func:`~spd_learn.functional.airm_distance` : Pairwise AIRM distance.

    References
    ----------
    See :cite:p:`pennec2006riemannian` for details on Karcher mean computation.
    """
    batch = X.detach()

    if weights is None:
        mean = batch.mean(dim=0, keepdim=True)
    else:
        mean = (batch * weights).sum(dim=0, keepdim=True)

    for _ in range(max_iter):
        mean_sqrt, mean_invsqrt = matrix_sqrt_inv.apply(mean)
        X_tangent = matrix_log.apply(mean_invsqrt @ batch @ mean_invsqrt)
        if weights is None:
            mean_tangent = X_tangent.mean(dim=0, keepdim=True)
        else:
            mean_tangent = (X_tangent * weights).sum(dim=0, keepdim=True)
        mean = mean_sqrt @ matrix_exp.apply(mean_tangent) @ mean_sqrt

    if return_distances:
        mean_sqrt, mean_invsqrt = matrix_sqrt_inv.apply(mean)
        X_tangent = matrix_log.apply(mean_invsqrt @ batch @ mean_invsqrt)
        distances = torch.norm(X_tangent, p="fro", dim=(-2, -1))
        return mean, distances

    return mean


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


def spd_cholesky_congruence(
    X: torch.Tensor,
    P: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    r"""Congruence transformation using the Cholesky factor of an SPD matrix.

    Given an SPD matrix :math:`P = LL^T`, applies:

    .. math::

        \text{forward: } Y = LXL^T, \qquad
        \text{inverse: } Y = L^{-1}X L^{-T}

    This implements the Lie group action of ``GL(n)`` on the SPD manifold and
    is used for centering and biasing under the affine-invariant metric.

    Parameters
    ----------
    X : torch.Tensor
        Batch of SPD matrices with shape `(..., n, n)`.
    P : torch.Tensor
        SPD matrix whose Cholesky factor defines the transformation,
        with shape broadcastable to ``X``.
    inverse : bool, default=False
        If True, applies the inverse congruence :math:`L^{-1}X L^{-T}`.

    Returns
    -------
    torch.Tensor
        Transformed SPD matrices with the same shape as ``X``.

    See Also
    --------
    :func:`spd_centering` : Eigendecomposition-based centering (uses :math:`M^{-1/2}`).
    """
    L = torch.linalg.cholesky(P)
    if inverse:
        Y = torch.linalg.solve_triangular(L, X, upper=False)
        return ensure_sym(torch.linalg.solve_triangular(L, Y.mT, upper=False).mT)
    return ensure_sym(L @ X @ L.mT)


def lie_group_variance(
    X_centered: torch.Tensor,
    metric: str,
    alpha: float = 1.0,
    beta: float = 0.0,
    theta: float = 1.0,
) -> torch.Tensor:
    r"""Fréchet variance under a Lie group structure on the SPD manifold.

    Computes the scalar dispersion of centered data in the Lie algebra,
    using the bi-invariant distance of Chen et al. :cite:p:`chen2024liebn`:

    .. math::

        \sigma^2 = \frac{1}{N} \sum_i
        \bigl(\alpha \lVert V_i \rVert_F^2 + \beta \, g(V_i)^2\bigr)
        \;/\; \theta^2

    where the auxiliary term :math:`g` depends on the metric:

    - **AIM**: :math:`V_i = \log(X_i)`, :math:`g(V) = \log\det(X)`
    - **LEM**: :math:`V_i = X_i` (already in log space), :math:`g(V) = \operatorname{tr}(V)`,
      no :math:`\theta` scaling
    - **LCM**: same as LEM but with :math:`\theta` scaling

    Parameters
    ----------
    X_centered : torch.Tensor
        Centered data in the Lie algebra with shape `(batch_size, ..., n, n)`.
        For AIM these are SPD matrices (centered around identity); for LEM/LCM
        these are symmetric / lower-triangular matrices.
    metric : {"AIM", "LEM", "LCM"}
        Lie group structure.
    alpha : float, default=1.0
        Frobenius-norm weight.
    beta : float, default=0.0
        Trace / log-determinant weight.
    theta : float, default=1.0
        Power deformation parameter.

    Returns
    -------
    torch.Tensor
        Scalar variance (0-d tensor).

    See Also
    --------
    :func:`tangent_space_variance` : Unweighted tangent-space dispersion used
        by :class:`~spd_learn.modules.SPDBatchNormMeanVar`.
    """
    if metric == "AIM":
        logX = matrix_log.apply(X_centered)
        frob_sq = (logX * logX).sum(dim=(-2, -1))
        dists = alpha * frob_sq
        if beta != 0:
            dists = dists + beta * torch.logdet(X_centered).square()
        return dists.mean() / (theta**2)

    frob_sq = (X_centered * X_centered).sum(dim=(-2, -1))
    dists = alpha * frob_sq
    if beta != 0:
        trace = X_centered.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        dists = dists + beta * trace.square()
    var = dists.mean()
    if metric == "LCM":
        var = var / (theta**2)
    elif metric != "LEM":
        raise ValueError(f"metric must be 'AIM', 'LEM', or 'LCM', got '{metric}'")
    return var


__all__ = [
    "frechet_mean",
    "karcher_mean_iteration",
    "lie_group_variance",
    "spd_centering",
    "spd_cholesky_congruence",
    "spd_rebiasing",
    "tangent_space_variance",
]
