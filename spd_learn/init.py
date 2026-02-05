# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Initialization functions for SPD-aware neural networks.

This module provides functions to initialize tensors with SPD-specific methods,
following PyTorch's `torch.nn.init` pattern.

All functions operate in-place and return the modified tensor for convenience.

Functions
---------
``stiefel_``
    Initialize tensor on the Stiefel manifold (orthonormal columns).
``spd_identity_``
    Initialize tensor as identity matrix.

Examples
--------
>>> import torch
>>> from spd_learn import init as spd_init
>>> W = torch.empty(8, 4)
>>> spd_init.stiefel_(W, seed=42)
>>> # W is now on the Stiefel manifold: W^T @ W = I
>>> torch.allclose(W.T @ W, torch.eye(4), atol=1e-5)
True
"""

import logging

from typing import Optional

import torch

from .functional.core import orthogonal_polar_factor


logger = logging.getLogger(__name__)


@torch.no_grad()
def stiefel_(tensor: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    r"""Initialize tensor on the Stiefel manifold (in-place).

    Projects a random matrix to orthonormal columns via polar decomposition.
    The Stiefel manifold :math:`\text{St}(n, k)` is the set of matrices
    :math:`W \in \mathbb{R}^{n \times k}` with orthonormal columns:
    :math:`W^\top W = I_k`.

    The orthogonal polar factor is computed as:

    .. math::

        W_{\text{Stiefel}} = W (W^\top W)^{-1/2}

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to initialize, with shape `(..., n, k)` where `n >= k`.
    seed : int, optional
        Random seed for reproducibility. If None, a warning is logged and
        seed 0 is used.

    Returns
    -------
    torch.Tensor
        The initialized tensor (same object as input, modified in-place).

    Notes
    -----
    Uses dtype-aware eigenvalue clamping from the unified numerical
    configuration to ensure numerical stability during initialization.

    If the eigendecomposition fails (rare), falls back to QR decomposition.

    Examples
    --------
    >>> import torch
    >>> from spd_learn import init as spd_init
    >>> W = torch.empty(10, 5)
    >>> spd_init.stiefel_(W, seed=42)
    >>> # Verify orthonormality
    >>> torch.allclose(W.T @ W, torch.eye(5), atol=1e-5)
    True

    See Also
    --------
    :func:`spd_identity_` : Initialize as identity matrix.
    :class:`~spd_learn.modules.BiMap` : Bilinear mapping layer using
        Stiefel initialization.
    """
    if seed is None:
        logger.warning(
            "No seed provided for Stiefel initialization. "
            "Using default seed 0 for reproducibility."
        )
        seed = 0

    generator = torch.Generator(device=tensor.device).manual_seed(seed)
    _W = torch.randn(
        *tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator
    )
    try:
        stiefel_W = orthogonal_polar_factor(_W)
    except torch.linalg.LinAlgError as e:
        logger.warning(
            f"Stiefel initialization via eigh failed ({e}). "
            f"Falling back to QR decomposition for orthogonalization."
        )
        Q_list = []
        if _W.ndim > 2:
            for i in range(_W.shape[0]):
                q, _ = torch.linalg.qr(_W[i])
                Q_list.append(q[..., : _W.shape[-1]])
            stiefel_W = torch.stack(Q_list, dim=0)
        else:
            q, _ = torch.linalg.qr(_W)
            stiefel_W = q[..., : _W.shape[-1]]

    tensor.copy_(stiefel_W)
    return tensor


@torch.no_grad()
def spd_identity_(tensor: torch.Tensor) -> torch.Tensor:
    """Initialize tensor as identity matrix (in-place).

    Sets the tensor to the identity matrix. The tensor must be square
    (last two dimensions equal).

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to initialize, with shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        The initialized tensor (same object as input, modified in-place).

    Examples
    --------
    >>> import torch
    >>> from spd_learn import init as spd_init
    >>> X = torch.empty(3, 4, 4)
    >>> spd_init.spd_identity_(X)
    >>> torch.allclose(X[0], torch.eye(4))
    True

    See Also
    --------
    :func:`stiefel_` : Initialize on Stiefel manifold.
    """
    if tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(
            f"spd_identity_ requires square matrices, got shape {tensor.shape}"
        )

    tensor.zero_()
    n = tensor.shape[-1]
    # Fill diagonal with ones
    tensor[..., range(n), range(n)] = 1.0
    return tensor


__all__ = [
    "stiefel_",
    "spd_identity_",
]
