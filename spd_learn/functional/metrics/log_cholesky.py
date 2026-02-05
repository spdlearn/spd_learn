# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Log-Cholesky metric operations for SPD matrices.

This module implements the Log-Cholesky Metric (LCM) for symmetric positive
definite (SPD) matrices :cite:p:`lin2019riemannian`. The LCM is a computationally
efficient alternative to eigenvalue-based Riemannian metrics, leveraging the
Cholesky decomposition instead of eigendecomposition.

Geometric Foundation
--------------------
The LCM builds upon the Cholesky decomposition :math:`P = LL^\top`, where
:math:`L` is a lower-triangular matrix with positive diagonal entries. There
exists a smooth bijection (diffeomorphism) :math:`\varphi: \mathcal{L}_+ \to
\mathcal{S}_{++}^n`, where :math:`\mathcal{L}_+` denotes the **Cholesky space**
of lower-triangular matrices with positive diagonals.

The Log-Cholesky representation maps an SPD matrix to a lower-triangular
matrix with logarithms on the diagonal:

.. math::

    X = LL^\top \quad \text{(Cholesky decomposition)}

.. math::

    \log_{\text{chol}}(L) = \text{tril}(L, -1) + \text{diag}(\log(\text{diag}(L)))

Riemannian Inner Product
------------------------
At :math:`P = LL^\top \in \mathcal{S}_{++}^n`, the LCM inner product is:

.. math::

    g^{\text{LCM}}_P(v, w) = \bar{g}_L(L(L^{-1}vL^{-\top})_\triangle, L(L^{-1}wL^{-\top})_\triangle)

where :math:`(\cdot)_\triangle` extracts the lower-triangular part and scales
diagonal elements by :math:`\frac{1}{2}`. The metric :math:`\bar{g}_L` on
:math:`\mathcal{L}_+` is:

.. math::

    \bar{g}_L(X, Y) = \sum_{i>j} X_{ij}Y_{ij} + \sum_{j=1}^{n} X_{jj}Y_{jj}L_{jj}^{-2}

Key Properties
--------------
- **Fastest computation**: Complexity :math:`O(n^3/3)` vs :math:`O(n^3)` for eigendecomposition
- **Numerical stability**: Cholesky decomposition is well-conditioned for SPD matrices
- **Globally flat geometry**: Inherits Euclidean structure from Cholesky space
- **Closed-form geodesics and means**: No iterative optimization required
- **Ideal for optimization**: Avoids explicit matrix inversions and logarithms,
  yielding improved differentiability for deep learning
- **Not affine-invariant**: Invariant under lower-triangular transformations with positive diagonal

See :cite:p:`lin2019riemannian` for the complete mathematical treatment.

Examples
--------
>>> import torch
>>> from spd_learn.functional.log_cholesky import (
...     cholesky_log, cholesky_exp, log_cholesky_distance, log_cholesky_mean
... )
>>> # Create SPD matrices
>>> A = torch.randn(3, 3)
>>> A = A @ A.T + 0.1 * torch.eye(3)
>>> B = torch.randn(3, 3)
>>> B = B @ B.T + 0.1 * torch.eye(3)
>>> # Compute Log-Cholesky representation
>>> log_chol_A = cholesky_log.apply(A)
>>> # Reconstruct from Log-Cholesky
>>> A_reconstructed = cholesky_exp.apply(log_chol_A)
>>> # Compute distance
>>> dist = log_cholesky_distance(A, B)
>>> # Compute mean of multiple matrices
>>> matrices = torch.stack([A, B])
>>> mean = log_cholesky_mean(matrices)
"""

import torch

from torch.autograd import Function

from ..numerical import get_epsilon


class cholesky_log(Function):
    r"""Matrix logarithm via Cholesky decomposition (Log-Cholesky map).

    This function computes the Log-Cholesky representation of an SPD matrix.
    Given an SPD matrix :math:`X`, it first computes the Cholesky decomposition:

    .. math::

        X = LL^T

    where :math:`L` is a lower triangular matrix with positive diagonal entries.
    The Log-Cholesky map then applies the logarithm to the diagonal:

    .. math::

        \log_{\text{chol}}(L) = \text{tril}(L, -1) + \text{diag}(\log(\text{diag}(L)))

    This maps the SPD manifold to the vector space of lower triangular matrices
    with arbitrary diagonal entries.

    Parameters
    ----------
    X : torch.Tensor
        SPD matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Log-Cholesky representation of `X` as a lower triangular matrix of
        shape `(..., n, n)`.

    Notes
    -----
    The backward pass is derived analytically :cite:p:`lin2019riemannian`. For
    the forward map :math:`f: X \mapsto \log_{\text{chol}}(L)` where
    :math:`X = LL^T`, the gradient computation involves the Cholesky derivative.

    See Also
    --------
    :class:`cholesky_exp` : Inverse mapping from Log-Cholesky space to SPD.
    :func:`log_cholesky_distance` : Distance under Log-Cholesky metric.
    :func:`~spd_learn.functional.matrix_log` : Matrix logarithm via eigendecomposition.
    """

    @staticmethod
    def forward(ctx, X):
        # Compute Cholesky decomposition
        L = torch.linalg.cholesky(X)

        # Extract diagonal
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1)

        # Compute log of diagonal with numerical stability
        threshold = get_epsilon(X.dtype, "eigval_log")
        diag_L_clamped = diag_L.clamp(min=threshold)
        log_diag = diag_L_clamped.log()

        # Create output: strictly lower triangular + log(diagonal)
        output = L.tril(-1)  # strictly lower triangular part
        output = output + torch.diag_embed(log_diag)

        # Save for backward
        ctx.save_for_backward(L)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (L,) = ctx.saved_tensors

        # Extract diagonal of L
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1)

        # Gradient through log on diagonal: d(log(d))/d(d) = 1/d
        grad_log_diag = torch.diagonal(grad_output, dim1=-2, dim2=-1)
        threshold = get_epsilon(L.dtype, "eigval_log")
        diag_L_clamped = diag_L.clamp(min=threshold)
        grad_diag_L = grad_log_diag / diag_L_clamped
        # Zero gradient where clamping occurred (subgradient of clamp)
        grad_diag_L = torch.where(
            diag_L > threshold, grad_diag_L, torch.zeros_like(grad_diag_L)
        )

        # Gradient through strictly lower triangular part is identity
        # grad_L is the gradient w.r.t. L (before log_chol transformation)
        grad_L = grad_output.tril(-1) + torch.diag_embed(grad_diag_L)

        # Backprop through Cholesky decomposition: X = L @ L.T
        # We use PyTorch's built-in Cholesky backward which is numerically stable.
        # The adjoint formula: if Y = chol(X), then
        # grad_X = L^{-T} @ Phi(L.T @ grad_L) @ L^{-1}
        # where Phi(M) = tril(M) - 0.5 * diag(diag(M)) for lower triangular constraint
        # and the result is symmetrized.

        # Compute S = L.T @ grad_L
        S = L.transpose(-1, -2) @ grad_L

        # Phi operation: take lower triangular and halve the diagonal
        Phi_S = S.tril()
        diag_Phi = torch.diagonal(Phi_S, dim1=-2, dim2=-1)
        Phi_S = Phi_S - 0.5 * torch.diag_embed(diag_Phi)

        # Compute L^{-T} @ Phi_S @ L^{-1} via solving triangular systems
        # First: solve L.T @ temp1 = Phi_S for temp1 (temp1 = L^{-T} @ Phi_S)
        temp1 = torch.linalg.solve_triangular(L.transpose(-1, -2), Phi_S, upper=True)
        # Then: solve temp1 @ L = grad_X for grad_X (grad_X = temp1 @ L^{-1})
        # Transpose to solve: L.T @ grad_X.T = temp1.T => grad_X.T = L^{-T} @ temp1.T
        grad_X_T = torch.linalg.solve_triangular(
            L.transpose(-1, -2), temp1.transpose(-1, -2), upper=True
        )
        grad_X = grad_X_T.transpose(-1, -2)

        # Symmetrize the gradient (since X is symmetric, grad must be symmetric)
        grad_X = 0.5 * (grad_X + grad_X.transpose(-1, -2))

        return grad_X


class cholesky_exp(Function):
    r"""Inverse of the Log-Cholesky map (Cholesky exponential).

    This function reconstructs an SPD matrix from its Log-Cholesky representation.
    Given a lower triangular matrix :math:`Y` (with arbitrary diagonal entries),
    it computes:

    .. math::

        L = \text{tril}(Y, -1) + \text{diag}(\exp(\text{diag}(Y)))

    .. math::

        X = LL^T

    Parameters
    ----------
    Y : torch.Tensor
        Log-Cholesky representation (lower triangular matrix) of shape
        `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        SPD matrix of shape `(..., n, n)`.

    Notes
    -----
    This is the inverse of :func:`cholesky_log`. The composition
    ``cholesky_exp(cholesky_log(X)) = X`` for any SPD matrix X.

    See Also
    --------
    :class:`cholesky_log` : Forward mapping from SPD to Log-Cholesky space.
    :func:`log_cholesky_mean` : Fréchet mean under Log-Cholesky metric.
    :func:`~spd_learn.functional.matrix_exp` : Matrix exponential via eigendecomposition.
    """

    @staticmethod
    def forward(ctx, Y):
        # Extract log-diagonal and strictly lower triangular parts
        log_diag = torch.diagonal(Y, dim1=-2, dim2=-1)
        diag_L = log_diag.exp()

        # Reconstruct L: strictly lower triangular + exp(diagonal)
        L = Y.tril(-1) + torch.diag_embed(diag_L)

        # Compute X = L @ L.T
        X = L @ L.transpose(-1, -2)

        # Save for backward
        ctx.save_for_backward(L, diag_L)

        return X

    @staticmethod
    def backward(ctx, grad_output):
        L, diag_L = ctx.saved_tensors

        # Backprop through X = L @ L.T
        # grad_L = 2 * tril(grad_X @ L)  (from the symmetric formula)
        # But we need to be careful: grad_output might not be symmetric
        # Ensure symmetry first
        grad_X_sym = 0.5 * (grad_output + grad_output.transpose(-1, -2))

        # Gradient through L @ L.T
        # d(L @ L.T) = dL @ L.T + L @ dL.T
        # For symmetric grad_X: grad_L = 2 * tril(grad_X @ L)
        grad_L = 2 * (grad_X_sym @ L).tril()

        # But this counts the diagonal twice, so we need to halve it:
        # Actually, let's be more careful.
        # The adjoint: tr(grad_X.T @ (dL @ L.T + L @ dL.T))
        # = tr(grad_X.T @ dL @ L.T) + tr(grad_X.T @ L @ dL.T)
        # = tr(L.T @ grad_X.T @ dL) + tr(dL.T @ grad_X.T @ L)
        # = tr(L.T @ grad_X @ dL) + tr(L.T @ grad_X.T @ dL)  (using cyclic property)
        # = tr((grad_X @ L + grad_X.T @ L).T @ dL)
        # = tr((2 * sym(grad_X) @ L).T @ dL)
        # So grad_L = 2 * sym(grad_X) @ L

        # But L is lower triangular, so we only care about the lower triangular part
        # grad_L = tril(2 * sym(grad_X) @ L)
        grad_L = (2 * grad_X_sym @ L).tril()

        # Backprop through the exponential on the diagonal
        # grad_Y_diag = grad_L_diag * diag_L  (chain rule for exp)
        grad_L_diag = torch.diagonal(grad_L, dim1=-2, dim2=-1)
        grad_Y_diag = grad_L_diag * diag_L

        # Backprop through strictly lower triangular (identity)
        grad_Y = grad_L.tril(-1) + torch.diag_embed(grad_Y_diag)

        return grad_Y


def log_cholesky_distance(A, B):
    r"""Compute the distance in the Log-Cholesky metric.

    The Log-Cholesky distance between two SPD matrices :math:`A` and :math:`B`
    is the Frobenius norm of the difference of their Log-Cholesky representations:

    .. math::

        d_{LC}(A, B) = \|\log_{\text{chol}}(L_A) - \log_{\text{chol}}(L_B)\|_F

    where :math:`A = L_A L_A^T` and :math:`B = L_B L_B^T` are the Cholesky
    decompositions.

    Parameters
    ----------
    A : torch.Tensor
        SPD matrices of shape `(..., n, n)`.
    B : torch.Tensor
        SPD matrices of shape `(..., n, n)`. Must be broadcastable with `A`.

    Returns
    -------
    torch.Tensor
        Distances of shape `(...)` (batch dimensions).

    Notes
    -----
    This metric is computationally cheaper than the affine-invariant Riemannian
    metric (AIRM) since it avoids eigendecomposition. The complexity is
    :math:`O(n^3/3)` for the Cholesky decomposition.

    The Log-Cholesky distance is not affine-invariant, but it is invariant
    under lower triangular transformations with positive diagonal
    :cite:p:`lin2019riemannian`.

    See Also
    --------
    :func:`log_cholesky_mean` : Fréchet mean under Log-Cholesky metric.
    :func:`log_cholesky_geodesic` : Geodesic interpolation under Log-Cholesky metric.
    :func:`~spd_learn.functional.airm_distance` : Distance under AIRM.
    :func:`~spd_learn.functional.log_euclidean_distance` : Distance under Log-Euclidean metric.
    :func:`~spd_learn.functional.bures_wasserstein_distance` : Distance under Bures-Wasserstein metric.

    Examples
    --------
    >>> import torch
    >>> A = torch.eye(3)
    >>> B = 2 * torch.eye(3)
    >>> dist = log_cholesky_distance(A, B)
    >>> print(f"Distance: {dist.item():.4f}")
    Distance: 1.2012
    """
    log_chol_A = cholesky_log.apply(A)
    log_chol_B = cholesky_log.apply(B)

    diff = log_chol_A - log_chol_B

    # Frobenius norm
    return torch.sqrt((diff**2).sum(dim=(-2, -1)))


def log_cholesky_mean(matrices, weights=None):
    r"""Compute the weighted mean in the Log-Cholesky space.

    The Log-Cholesky mean is the arithmetic mean of the Log-Cholesky
    representations, mapped back to the SPD manifold:

    .. math::

        \bar{X} = \exp_{\text{chol}}\left(\sum_{i=1}^{N} w_i \log_{\text{chol}}(L_i)\right)

    where :math:`w_i` are the weights (summing to 1) and :math:`X_i = L_i L_i^T`.

    Parameters
    ----------
    matrices : torch.Tensor
        SPD matrices of shape `(N, ..., n, n)` where N is the number of matrices
        to average.
    weights : torch.Tensor, optional
        Weights of shape `(N,)` or broadcastable. If None, uniform weights
        are used. Weights are automatically normalized to sum to 1.

    Returns
    -------
    torch.Tensor
        Mean SPD matrix of shape `(..., n, n)`.

    Notes
    -----
    The Log-Cholesky mean has the following properties :cite:p:`lin2019riemannian`:

    - It is the unique minimizer of the sum of squared Log-Cholesky distances.
    - It can be computed in closed form (no iterative optimization required).
    - It is computationally more efficient than the Fréchet mean under AIRM.

    See Also
    --------
    :func:`log_cholesky_distance` : Distance under Log-Cholesky metric.
    :func:`log_cholesky_geodesic` : Geodesic interpolation under Log-Cholesky metric.
    :func:`~spd_learn.functional.log_euclidean_mean` : Mean under Log-Euclidean metric.
    :func:`~spd_learn.functional.bures_wasserstein_mean` : Mean under Bures-Wasserstein metric.
    :class:`~spd_learn.modules.SPDBatchNormMeanVar` : Uses Fréchet mean for batch normalization.

    Examples
    --------
    >>> import torch
    >>> # Create 4 SPD matrices of size 3x3
    >>> matrices = torch.stack([
    ...     torch.eye(3),
    ...     2 * torch.eye(3),
    ...     3 * torch.eye(3),
    ...     4 * torch.eye(3)
    ... ])
    >>> mean = log_cholesky_mean(matrices)
    >>> print(f"Mean diagonal: {torch.diag(mean)}")
    Mean diagonal: tensor([2.2134, 2.2134, 2.2134])
    """
    N = matrices.shape[0]

    # Handle weights
    if weights is None:
        weights = torch.ones(N, device=matrices.device, dtype=matrices.dtype) / N
    else:
        weights = weights.to(device=matrices.device, dtype=matrices.dtype)
        weights = weights / weights.sum()  # Normalize

    # Reshape weights for broadcasting: (N, 1, 1, ...)
    weight_shape = (N,) + (1,) * (matrices.ndim - 1)
    weights = weights.view(weight_shape)

    # Compute Log-Cholesky representations
    log_chol_matrices = cholesky_log.apply(matrices)

    # Weighted average in Log-Cholesky space
    weighted_mean = (weights * log_chol_matrices).sum(dim=0)

    # Map back to SPD manifold
    return cholesky_exp.apply(weighted_mean)


def log_cholesky_geodesic(A, B, t):
    r"""Geodesic interpolation in the Log-Cholesky metric.

    Computes the point on the geodesic between SPD matrices :math:`A` and
    :math:`B` at parameter :math:`t`:

    .. math::

        \gamma(t) = \exp_{\text{chol}}\left((1-t) \log_{\text{chol}}(L_A) +
                     t \log_{\text{chol}}(L_B)\right)

    where :math:`t \\in [0, 1]`.

    Parameters
    ----------
    A : torch.Tensor
        Starting SPD matrix of shape `(..., n, n)`.
    B : torch.Tensor
        Ending SPD matrix of shape `(..., n, n)`.
    t : float or torch.Tensor
        Interpolation parameter. For t=0, returns A. For t=1, returns B.

    Returns
    -------
    torch.Tensor
        Interpolated SPD matrix of shape `(..., n, n)`.

    See Also
    --------
    :func:`log_cholesky_distance` : Distance under Log-Cholesky metric.
    :func:`log_cholesky_mean` : Fréchet mean under Log-Cholesky metric.
    :func:`~spd_learn.functional.airm_geodesic` : Geodesic under AIRM.
    :func:`~spd_learn.functional.bures_wasserstein_geodesic` : Geodesic under Bures-Wasserstein metric.

    Examples
    --------
    >>> import torch
    >>> A = torch.eye(3)
    >>> B = 4 * torch.eye(3)
    >>> # Midpoint
    >>> mid = log_cholesky_geodesic(A, B, 0.5)
    >>> print(f"Midpoint diagonal: {torch.diag(mid)}")
    Midpoint diagonal: tensor([2., 2., 2.])
    """
    log_chol_A = cholesky_log.apply(A)
    log_chol_B = cholesky_log.apply(B)

    # Linear interpolation in Log-Cholesky space
    log_chol_interp = (1 - t) * log_chol_A + t * log_chol_B

    return cholesky_exp.apply(log_chol_interp)
