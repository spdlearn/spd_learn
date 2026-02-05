# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Optional

import torch

from torch import Tensor

from .numerical import numerical_config


def trace_normalization(covariances: Tensor, epsilon: Optional[float] = None) -> Tensor:
    """Performs trace normalization on a batch of covariance matrices.

    This function normalizes a batch of covariance matrices by their trace.
    It also adds a small identity matrix for numerical stability.

    Parameters
    ----------
    covariances : Tensor
        A batch of covariance matrices with shape `(..., n, n)`.
    epsilon : float, optional
        A small value to add to the diagonal of the covariance matrices for
        numerical stability. If None, uses the value from the unified
        numerical configuration. Default: None.

    Returns
    -------
    Tensor
        The trace-normalized covariance matrices.
    """
    # Use unified config if epsilon not specified
    if epsilon is None:
        epsilon = numerical_config.trace_norm_eps

    trace = covariances.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
    trace = trace.unsqueeze(-1)

    diagonal_dim = covariances.shape[-1]

    covariances = covariances / (trace / diagonal_dim)

    identity = torch.eye(
        diagonal_dim, device=covariances.device, dtype=covariances.dtype
    )
    covariances = covariances + epsilon * identity

    return covariances


def ledoit_wolf(
    covariances: Tensor,
    shrinkage: Tensor,
    shrink_mat: Tensor,
    size: int,
) -> Tensor:
    """Applies Ledoit-Wolf shrinkage to a batch of covariance matrices.

    Parameters
    ----------
    covariances : Tensor
        A batch of covariance matrices with shape `(..., n, n)`.
    shrinkage : Tensor
        Unconstrained shrinkage parameters, which will be passed through a
        sigmoid function.
    shrink_mat : Tensor
        The target "shrink" matrices, which are often identity matrices.
    size : int
        The dimensionality of the covariance matrices.

    Returns
    -------
    Tensor
        The shrunken covariance matrices.
    """
    trace = covariances.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
    mu = trace / size

    alpha = torch.sigmoid(shrinkage)

    return (1 - alpha)[..., None, None] * covariances + alpha[..., None, None] * (
        mu[..., None] * shrink_mat
    )


def shrinkage_covariance(
    X: Tensor,
    alpha: Tensor,
    n_chans: int,
    identity: Optional[Tensor] = None,
) -> Tensor:
    r"""Apply shrinkage regularization to covariance matrices.

    Computes the shrinkage estimator:

    .. math::

        \hat{C} = (1 - \alpha) C + \alpha \cdot \frac{\text{tr}(C)}{n} \cdot I_n

    This convex combination interpolates between the empirical covariance
    :math:`C` and a scaled identity matrix.

    Parameters
    ----------
    X : Tensor
        Batch of covariance matrices with shape `(..., n_chans, n_chans)`.
    alpha : Tensor
        Shrinkage intensity in range [0, 1]. Can be a scalar or broadcastable
        with `X`.
    n_chans : int
        Number of channels (matrix dimension).
    identity : Tensor, optional
        Pre-computed identity matrix. If None, one is created.

    Returns
    -------
    Tensor
        Regularized covariance matrices with shape `(..., n_chans, n_chans)`.

    Notes
    -----
    For :math:`\alpha = 0`, returns the original covariance.
    For :math:`\alpha = 1`, returns a scaled identity matrix.

    See Also
    --------
    :func:`ledoit_wolf` : Alternative shrinkage formulation.
    :func:`trace_normalization` : Trace-based normalization.
    :class:`~spd_learn.modules.Shrinkage` : Module wrapper for this function.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import shrinkage_covariance
    >>> X = torch.randn(4, 8, 8)
    >>> X = X @ X.mT  # Make SPD
    >>> alpha = torch.tensor(0.5)
    >>> X_shrunk = shrinkage_covariance(X, alpha, n_chans=8)
    """
    trace = torch.sum(torch.diagonal(X, dim1=-2, dim2=-1), dim=-1, keepdim=True)
    mu = (trace / n_chans).unsqueeze(-1)

    if identity is None:
        identity = torch.eye(n_chans, device=X.device, dtype=X.dtype)

    shrunk_target = alpha * mu * identity
    return (1 - alpha) * X + shrunk_target
