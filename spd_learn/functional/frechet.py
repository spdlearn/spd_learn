# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Frechet (directional) derivatives of matrix functions.

This module provides Frechet derivatives of the matrix logarithm and matrix
exponential, computed via the Daleckii-Krein theorem. These are used for
non-trivial parallel transport under the Log-Euclidean metric.

The Frechet derivative :math:`Df(P)[V]` of a matrix function :math:`f` at
:math:`P` in direction :math:`V` is computed as:

.. math::

    Df(P)[V] = U (K \odot (U^T V U)) U^T

where :math:`P = U \Lambda U^T` is the eigendecomposition and :math:`K` is
the Loewner matrix of divided differences of :math:`f`.
"""

import torch

from .numerical import get_loewner_threshold
from .utils import ensure_sym


def frechet_derivative_log(P, V):
    r"""Frechet derivative of the matrix logarithm at P in direction V.

    Computes :math:`D\log(P)[V]` using the Daleckii-Krein theorem:

    .. math::

        D\log(P)[V] = U (K \odot (U^T V U)) U^T

    where :math:`P = U \operatorname{diag}(\lambda) U^T` and the coefficient
    matrix :math:`K` has entries:

    .. math::

        K_{ij} = \begin{cases}
            \frac{\log \lambda_i - \log \lambda_j}{\lambda_i - \lambda_j}
            & \text{if } \lambda_i \neq \lambda_j \\
            \frac{1}{\lambda_i}
            & \text{if } \lambda_i = \lambda_j
        \end{cases}

    Parameters
    ----------
    P : torch.Tensor
        SPD matrix, shape (..., n, n).
    V : torch.Tensor
        Symmetric matrix (tangent vector), shape (..., n, n).

    Returns
    -------
    torch.Tensor
        The Frechet derivative :math:`D\log(P)[V]`, shape (..., n, n).
    """
    # Eigendecompose P
    eigvals, U = torch.linalg.eigh(P)

    # Build the Loewner matrix K for log
    log_eigvals = eigvals.log()

    # Denominator: lambda_i - lambda_j
    denom = eigvals.unsqueeze(-1) - eigvals.unsqueeze(-2)

    # Adaptive threshold for detecting equal eigenvalues
    threshold = get_loewner_threshold(eigvals)
    is_eq = denom.abs() < threshold
    denom_safe = denom.clone()
    denom_safe[is_eq] = 1.0

    # Numerator: log(lambda_i) - log(lambda_j)
    numer = log_eigvals.unsqueeze(-1) - log_eigvals.unsqueeze(-2)

    # K_ij = (log(lambda_i) - log(lambda_j)) / (lambda_i - lambda_j)
    K = numer / denom_safe

    # Diagonal (and near-equal): K_ii = 1 / lambda_i
    diag_vals = 1.0 / eigvals
    # For equal eigenvalues, use average of the diagonal values
    K[is_eq] = (
        0.5
        * (diag_vals.unsqueeze(-1) + diag_vals.unsqueeze(-2))[is_eq]
    )

    # Apply: U @ (K * (U^T V U)) @ U^T
    Ut = U.transpose(-2, -1)
    inner = Ut @ V @ U
    result = U @ (K * inner) @ Ut

    return ensure_sym(result)


def frechet_derivative_exp(X, W):
    r"""Frechet derivative of the matrix exponential at X in direction W.

    Computes :math:`D\exp(X)[W]` using the Daleckii-Krein theorem:

    .. math::

        D\exp(X)[W] = U (M \odot (U^T W U)) U^T

    where :math:`X = U \operatorname{diag}(d) U^T` (X is symmetric, not
    necessarily SPD) and the coefficient matrix :math:`M` has entries:

    .. math::

        M_{ij} = \begin{cases}
            \frac{e^{d_i} - e^{d_j}}{d_i - d_j}
            & \text{if } d_i \neq d_j \\
            e^{d_i}
            & \text{if } d_i = d_j
        \end{cases}

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix, shape (..., n, n). Need not be SPD.
    W : torch.Tensor
        Symmetric matrix (tangent direction), shape (..., n, n).

    Returns
    -------
    torch.Tensor
        The Frechet derivative :math:`D\exp(X)[W]`, shape (..., n, n).
    """
    # Eigendecompose X
    eigvals, U = torch.linalg.eigh(X)

    # Build the Loewner matrix M for exp
    exp_eigvals = eigvals.exp()

    # Denominator: d_i - d_j
    denom = eigvals.unsqueeze(-1) - eigvals.unsqueeze(-2)

    # Adaptive threshold for detecting equal eigenvalues
    threshold = get_loewner_threshold(eigvals)
    is_eq = denom.abs() < threshold
    denom_safe = denom.clone()
    denom_safe[is_eq] = 1.0

    # Numerator: exp(d_i) - exp(d_j)
    numer = exp_eigvals.unsqueeze(-1) - exp_eigvals.unsqueeze(-2)

    # M_ij = (exp(d_i) - exp(d_j)) / (d_i - d_j)
    M = numer / denom_safe

    # Diagonal (and near-equal): M_ii = exp(d_i)
    # For equal eigenvalues, use average of exp values
    M[is_eq] = (
        0.5
        * (exp_eigvals.unsqueeze(-1) + exp_eigvals.unsqueeze(-2))[is_eq]
    )

    # Apply: U @ (M * (U^T W U)) @ U^T
    Ut = U.transpose(-2, -1)
    inner = Ut @ W @ U
    result = U @ (M * inner) @ Ut

    return ensure_sym(result)
