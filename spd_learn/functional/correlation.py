# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Correlation manifold operations and Riemannian metrics.

This module implements differential geometric operations on the manifold of
full-rank correlation matrices :math:`\\mathcal{C}_{++}^n`, including:

- Conversion between covariance and correlation matrices (:func:`cov_to_corr`)
- Extraction of hollow symmetric matrices (:func:`off_diag`)
- Off-Log Metric (OLM) logarithmic and exponential maps (:func:`log_map_olm`, :func:`exp_map_olm`)
- Log-Scaled Metric (LSM) logarithmic and exponential maps (:func:`log_map_lsm`, :func:`exp_map_lsm`)
- Isometric Frobenius embedding for hollow symmetric matrices (:func:`tril_embed`, :func:`untril_embed`)

References
----------
.. [1] C. Hu, R. Wang, X. Song, T. Zhou, X.-J. Wu, N. Sebe, and Z. Chen,
   "A Correlation Manifold Self-Attention Network for EEG Decoding,"
   in Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2025.
.. [2] Z. Chen, X. Wu, B. Schölkopf, and N. Sebe,
   "Riemannian Networks over Full-Rank Correlation Matrices,"
   arXiv:2605.19073, 2026.
"""

from __future__ import annotations

import math

import torch

from .core import matrix_exp, matrix_log
from .utils import ensure_sym


def off_diag(X: torch.Tensor) -> torch.Tensor:
    r"""Extract the hollow off-diagonal part of a matrix (zeroing the diagonal).

    Given a square matrix :math:`X`, this returns:

    .. math::

        \text{off\_diag}(X) = X - \text{diag}(\text{diag}(X))

    preserving the shape and off-diagonal entries while setting all diagonal
    elements to zero.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Hollow matrix with zero diagonal, shape `(..., n, n)`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import off_diag
    >>> X = torch.ones(3, 3)
    >>> off_diag(X)
    tensor([[0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.]])
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    return X - torch.diag_embed(diag)


def cov_to_corr(S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Convert an SPD covariance matrix to a full-rank correlation matrix.

    Converts an SPD covariance matrix :math:`S \in \mathcal{S}_{++}^n` to a
    full-rank correlation matrix :math:`C \in \mathcal{C}_{++}^n` with unit
    diagonal:

    .. math::

        d = \text{clamp}(\text{diag}(S), \min=\epsilon)^{-1/2}

    .. math::

        C = \text{diag}(d) \, S \, \text{diag}(d)

    with the diagonal explicitly set to 1.

    Parameters
    ----------
    S : torch.Tensor
        SPD covariance matrices of shape `(..., n, n)`.
    eps : float, default=1e-6
        Small positive constant for numerical stability of division.

    Returns
    -------
    torch.Tensor
        Full-rank correlation matrix with unit diagonal, shape `(..., n, n)`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import cov_to_corr
    >>> S = torch.tensor([[4.0, 1.0], [1.0, 9.0]])
    >>> cov_to_corr(S)
    tensor([[1.0000, 0.1667],
            [0.1667, 1.0000]])
    """
    n = S.shape[-1]
    S_sym = ensure_sym(S)
    d = torch.diagonal(S_sym, dim1=-2, dim2=-1).clamp(min=eps).rsqrt()
    C_sym = ensure_sym(d.unsqueeze(-1) * S_sym * d.unsqueeze(-2))
    return off_diag(C_sym) + torch.eye(n, dtype=S.dtype, device=S.device)


def log_map_olm(C: torch.Tensor) -> torch.Tensor:
    r"""Off-Log Metric (OLM) logarithmic map.

    Maps a correlation matrix :math:`C \in \mathcal{C}_{++}^n` to the hollow
    symmetric tangent space :math:`\text{Hol}(\mathcal{S}^n)`:

    .. math::

        \Phi_{\text{OLM}}(C) = \text{off\_diag}(\log(C))

    Parameters
    ----------
    C : torch.Tensor
        Correlation matrices of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Hollow symmetric tangent matrix of shape `(..., n, n)`.

    References
    ----------
    .. [1] C. Hu et al., "A Correlation Manifold Self-Attention Network for EEG
       Decoding," IJCAI 2025.
    """
    return off_diag(matrix_log.apply(ensure_sym(C)))


def exp_map_olm(
    X: torch.Tensor,
    max_iter: int = 15,
    tol: float = 1e-6,
) -> torch.Tensor:
    r"""Off-Log Metric (OLM) exponential map.

    Maps a hollow symmetric matrix :math:`X \in \text{Hol}(\mathcal{S}^n)` to
    the correlation manifold :math:`\mathcal{C}_{++}^n` by finding a diagonal
    scaling vector :math:`d` such that :math:`\text{diag}(\exp(X + \text{diag}(d))) = I`.

    The diagonal vector :math:`d` is solved via fixed-point iteration, and
    gradients flow cleanly through :math:`X` via :func:`matrix_exp` and
    :func:`cov_to_corr`.

    Parameters
    ----------
    X : torch.Tensor
        Hollow symmetric tangent matrices of shape `(..., n, n)`.
    max_iter : int, default=15
        Maximum number of fixed-point iterations.
    tol : float, default=1e-6
        Convergence tolerance on the maximum diagonal deviation.

    Returns
    -------
    torch.Tensor
        Correlation matrix with unit diagonal, shape `(..., n, n)`.

    References
    ----------
    .. [1] C. Hu et al., "A Correlation Manifold Self-Attention Network for EEG
       Decoding," IJCAI 2025.
    """
    X_hollow = off_diag(ensure_sym(X))

    with torch.no_grad():
        d = torch.zeros(
            X_hollow.shape[:-1], dtype=X_hollow.dtype, device=X_hollow.device
        )
        for _ in range(max_iter):
            M = matrix_exp.apply(X_hollow + torch.diag_embed(d))
            diag_M = torch.diagonal(M, dim1=-2, dim2=-1).clamp(min=1e-6)
            delta = -torch.log(diag_M)
            d = d + delta
            if torch.max(torch.abs(delta)) < tol:
                break

    return cov_to_corr(matrix_exp.apply(X_hollow + torch.diag_embed(d)))


def log_map_lsm(
    C: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> torch.Tensor:
    r"""Log-Scaled Metric (LSM) logarithmic map.

    Scales correlation matrix :math:`C \in \mathcal{C}_{++}^n` via diagonal
    :math:`D = \text{diag}(d)` such that :math:`\text{diag}(\log(D C D)) = 0`.

    Parameters
    ----------
    C : torch.Tensor
        Correlation matrices of shape `(..., n, n)`.
    max_iter : int, default=10
        Maximum number of scaling iterations.
    tol : float, default=1e-6
        Convergence tolerance on the diagonal elements of the logarithm.

    Returns
    -------
    torch.Tensor
        Hollow symmetric tangent matrix of shape `(..., n, n)`.

    References
    ----------
    .. [1] C. Hu et al., "A Correlation Manifold Self-Attention Network for EEG
       Decoding," IJCAI 2025.
    """
    C_sym = ensure_sym(C)
    with torch.no_grad():
        d = torch.ones(C_sym.shape[:-1], dtype=C_sym.dtype, device=C_sym.device)
        for _ in range(max_iter):
            D_C_D = d.unsqueeze(-1) * C_sym * d.unsqueeze(-2)
            L = matrix_log.apply(D_C_D)
            diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
            if torch.max(torch.abs(diag_L)) < tol:
                break
            d = d * torch.exp(-0.5 * diag_L)

    D_C_D = d.unsqueeze(-1) * C_sym * d.unsqueeze(-2)
    return off_diag(matrix_log.apply(D_C_D))


def exp_map_lsm(X: torch.Tensor) -> torch.Tensor:
    r"""Log-Scaled Metric (LSM) exponential map.

    Maps a hollow symmetric matrix :math:`X \in \text{Hol}(\mathcal{S}^n)`
    to the correlation manifold :math:`\mathcal{C}_{++}^n`:

    .. math::

        \Phi_{\text{LSM}}^{-1}(X) = \text{cov\_to\_corr}(\exp(\text{off\_diag}(X)))

    Parameters
    ----------
    X : torch.Tensor
        Hollow symmetric tangent matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Correlation matrix with unit diagonal, shape `(..., n, n)`.

    References
    ----------
    .. [1] C. Hu et al., "A Correlation Manifold Self-Attention Network for EEG
       Decoding," IJCAI 2025.
    """
    X_hollow = off_diag(ensure_sym(X))
    return cov_to_corr(matrix_exp.apply(X_hollow))


def tril_embed(X: torch.Tensor) -> torch.Tensor:
    r"""Extract strictly lower-triangular entries scaled by :math:`\sqrt{2}`.

    Provides an isometric embedding of hollow symmetric matrices
    :math:`\text{Hol}(\mathcal{S}^n)` into Euclidean space
    :math:`\mathbb{R}^{n(n-1)/2}` under the Frobenius metric:

    .. math::

        \langle X, Y \rangle_F = \text{tr}(X Y) = \langle \text{tril\_embed}(X), \text{tril\_embed}(Y) \rangle

    Parameters
    ----------
    X : torch.Tensor
        Hollow symmetric matrices of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Vectorized representation of shape `(..., n*(n-1)//2)`.
    """
    n = X.shape[-1]
    tril_idx = torch.tril_indices(row=n, col=n, offset=-1, device=X.device)
    scale = math.sqrt(2.0)
    return X[..., tril_idx[0], tril_idx[1]] * scale


def untril_embed(v: torch.Tensor, n: int) -> torch.Tensor:
    r"""Inverse of :func:`tril_embed`.

    Reconstructs a hollow symmetric matrix from its scaled lower-triangular
    embedding vector.

    Parameters
    ----------
    v : torch.Tensor
        Vectorized hollow symmetric matrix of shape `(..., n*(n-1)//2)`.
    n : int
        Dimension of the square matrix to reconstruct.

    Returns
    -------
    torch.Tensor
        Hollow symmetric matrix of shape `(..., n, n)`.
    """
    tril_idx = torch.tril_indices(row=n, col=n, offset=-1, device=v.device)
    scale = math.sqrt(2.0)
    unscaled = v / scale
    batch_shape = v.shape[:-1]
    X = torch.zeros((*batch_shape, n, n), dtype=v.dtype, device=v.device)
    X[..., tril_idx[0], tril_idx[1]] = unscaled
    X[..., tril_idx[1], tril_idx[0]] = unscaled
    return X
