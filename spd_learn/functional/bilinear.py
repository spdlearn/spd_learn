# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Functional operations for bilinear mappings on SPD matrices.

This module provides stateless mathematical operations for bilinear
transformations that preserve the SPD property of covariance matrices.

Functions
---------
bimap_transform
    Apply bilinear transformation to SPD matrices.
bimap_increase_dim
    Increase the dimension of SPD matrices via embedding.

See Also
--------
:class:`~spd_learn.modules.BiMap` : Bilinear mapping layer.
:class:`~spd_learn.modules.BiMapIncreaseDim` : Dimension expansion layer.
"""

from torch import Tensor


def bimap_transform(X: Tensor, W: Tensor) -> Tensor:
    r"""Apply bilinear transformation to SPD matrices.

    Computes the bilinear mapping:

    .. math::

        Y = W^\top X W

    where :math:`X` is an SPD matrix and :math:`W` is a (semi-)orthogonal
    weight matrix. When :math:`W` has orthonormal columns, the transformation
    preserves the SPD property.

    Parameters
    ----------
    X : Tensor
        Input SPD matrices with shape `(..., n, n)`.
    W : Tensor
        Weight matrix with shape `(..., n, k)` where `k <= n`.
        Should have orthonormal columns for SPD preservation.

    Returns
    -------
    Tensor
        Transformed SPD matrices with shape `(..., k, k)`.

    Notes
    -----
    This operation is the core of spatial filtering methods like Common
    Spatial Patterns (CSP) and is used in SPD neural networks to reduce
    the dimensionality of covariance matrices while preserving geometric
    structure.

    See Also
    --------
    :func:`bimap_increase_dim` : For dimension expansion.
    :class:`~spd_learn.modules.BiMap` : Module wrapper for this function.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import bimap_transform
    >>> X = torch.eye(8)  # 8x8 identity
    >>> W = torch.eye(8, 4)  # Semi-orthogonal projection
    >>> Y = bimap_transform(X, W)
    >>> Y.shape
    torch.Size([4, 4])
    """
    return W.mT @ X @ W


def bimap_increase_dim(
    X: Tensor,
    projection_matrix: Tensor,
    padding_matrix: Tensor,
) -> Tensor:
    r"""Increase the dimension of SPD matrices via embedding.

    Computes the dimension expansion:

    .. math::

        Y = P + W X W^\top

    where :math:`X` is the input SPD matrix, :math:`W` is a semi-orthogonal
    projection matrix, and :math:`P` is an identity padding matrix that
    ensures the output is SPD.

    Parameters
    ----------
    X : Tensor
        Input SPD matrices with shape `(..., n_in, n_in)`.
    projection_matrix : Tensor
        Semi-orthogonal projection matrix with shape `(n_out, n_in)`.
    padding_matrix : Tensor
        Identity padding matrix with shape `(n_out, n_out)`.
        Should be a diagonal matrix with ones for indices >= n_in.

    Returns
    -------
    Tensor
        Expanded SPD matrices with shape `(..., n_out, n_out)`.

    Notes
    -----
    The padding matrix ensures that the output remains SPD by adding
    identity elements in the expanded dimensions.

    See Also
    --------
    :func:`bimap_transform` : For dimension reduction.
    :class:`~spd_learn.modules.BiMapIncreaseDim` : Module wrapper for this function.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import bimap_increase_dim
    >>> X = torch.eye(4)  # 4x4 identity
    >>> proj = torch.eye(8, 4)  # Projection from 4 to 8
    >>> pad = torch.diag(torch.tensor([0,0,0,0,1,1,1,1], dtype=torch.float))
    >>> Y = bimap_increase_dim(X, proj, pad)
    >>> Y.shape
    torch.Size([8, 8])
    """
    return padding_matrix + projection_matrix @ X @ projection_matrix.mT


__all__ = [
    "bimap_transform",
    "bimap_increase_dim",
]
