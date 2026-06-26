# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Frechet (directional) derivatives of matrix functions.

Thin wrappers over pyriemann's :func:`~pyriemann.geometry.base.ddlogm` /
:func:`~pyriemann.geometry.base.ddexpm` (Array API, run on torch tensors with
autograd), which compute the Frechet derivatives of the matrix logarithm and
exponential via the Daleckii-Krein theorem. Used for non-trivial parallel
transport under the Log-Euclidean metric.
"""

from pyriemann.geometry.base import ddexpm, ddlogm


def frechet_derivative_log(P, V):
    r"""Frechet derivative of the matrix logarithm at P in direction V.

    Computes :math:`D\log(P)[V]`, the directional derivative of the matrix
    logarithm at SPD matrix :math:`P` in direction :math:`V`. Delegates to
    :func:`pyriemann.geometry.base.ddlogm`, which evaluates the Daleckii-Krein
    (Loewner divided-difference) form on the Array API backend.

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
    # Delegated to pyriemann (Array API). pyriemann's ddlogm(X, Cref) computes
    # the directional derivative of logm at Cref in direction X.
    return ddlogm(V, P)


def frechet_derivative_exp(X, W):
    r"""Frechet derivative of the matrix exponential at X in direction W.

    Computes :math:`D\exp(X)[W]`, the directional derivative of the matrix
    exponential at symmetric matrix :math:`X` (need not be SPD) in direction
    :math:`W`. Delegates to :func:`pyriemann.geometry.base.ddexpm`, which
    evaluates the Daleckii-Krein (Loewner divided-difference) form on the
    Array API backend.

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
    # Delegated to pyriemann (Array API). pyriemann's ddexpm(X, Cref) computes
    # the directional derivative of expm at Cref in direction X.
    return ddexpm(W, X)
