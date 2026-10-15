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
>>> spd_init.stiefel_(W)
>>> # W is now on the Stiefel manifold: W^T @ W = I
>>> torch.allclose(W.T @ W, torch.eye(4), atol=1e-5)
True
"""

import logging

import torch

from .functional.core import orthogonal_polar_factor


logger = logging.getLogger(__name__)


@torch.no_grad()
def stiefel_(
    tensor: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
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
    generator : torch.Generator, optional
        Generator used for random sampling. If None, the default PyTorch
        random number generator is used.

    Returns
    -------
    torch.Tensor
        The initialized tensor (same object as input, modified in-place).

    Notes
    -----
    Computes the polar factor in float64 to improve numerical accuracy,
    then casts the result to the input dtype. For MPS tensors, the
    decomposition runs on CPU because MPS does not support float64.

    If the eigendecomposition fails (rare), falls back to QR decomposition.

    Examples
    --------
    >>> import torch
    >>> from spd_learn import init as spd_init
    >>> W = torch.empty(10, 5)
    >>> spd_init.stiefel_(W)
    >>> # Verify orthonormality
    >>> torch.allclose(W.T @ W, torch.eye(5), atol=1e-5)
    True

    See Also
    --------
    :func:`spd_identity_` : Initialize as identity matrix.
    :class:`~spd_learn.modules.BiMap` : Bilinear mapping layer using
        Stiefel initialization.
    """
    _W = torch.randn(
        *tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator
    )
    work = _W.cpu() if _W.device.type == "mps" else _W
    work = work.to(torch.float64)

    try:
        stiefel_W = orthogonal_polar_factor(work)
    except torch.linalg.LinAlgError as e:
        logger.warning(
            f"Stiefel initialization via eigh failed ({e}). "
            "Falling back to QR decomposition for orthogonalization."
        )
        stiefel_W, _ = torch.linalg.qr(work, mode="reduced")

    tensor.copy_(stiefel_W.to(dtype=tensor.dtype, device=tensor.device))
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
