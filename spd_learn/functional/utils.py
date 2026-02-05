# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch


def ensure_sym(matrix):
    """Ensures that a matrix is symmetric.

    This function ensures that the last two dimensions of a tensor represent a
    symmetric matrix by averaging the matrix with its transpose.

    Parameters
    ----------
    matrix : torch.Tensor
        A tensor with at least two dimensions, where the last two dimensions
        form a square matrix.

    Returns
    -------
    torch.Tensor
        The symmetric version of the input matrix.
    """
    return (matrix + matrix.mT) / 2


def vec_batch(X: torch.Tensor) -> torch.Tensor:
    """Vectorizes a batch of tensors along the last two dimensions.

    Parameters
    ----------
    X : torch.Tensor
        A batch of matrices with shape `(..., n, k)`.

    Returns
    -------
    torch.Tensor
        A batch of vectorized matrices with shape `(..., n * k)`.
    """
    return X.reshape(*X.shape[:-2], -1)


def unvec_batch(X_vec: torch.Tensor, n: int) -> torch.Tensor:
    """Unvectorizes a batch of tensors along the last dimension.

    Parameters
    ----------
    X_vec : torch.Tensor
        A batch of vectorized matrices with shape `(..., n * k)`.
    n : int
        The number of rows in the output matrices.

    Returns
    -------
    torch.Tensor
        A batch of matrices with shape `(..., n, k)`.
    """
    return X_vec.reshape(*X_vec.shape[:-1], n, -1)
