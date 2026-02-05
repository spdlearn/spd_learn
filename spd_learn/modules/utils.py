# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Optional

import torch

from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from ..functional.core import sym_to_upper, vec_to_sym
from ..functional.utils import unvec_batch, vec_batch


class PatchEmbeddingLayer(nn.Module):
    """Patch Embedding Layer.

    This layer extracts patches from an input signal using an unfolding
    operation, which is similar to a convolution.

    Parameters
    ----------
    n_chans : int
        Number of input channels.
    n_patches : int
        Number of patches to extract.
    stride : int, optional
        The step size between patches. If `None`, it defaults to the patch
        size, resulting in non-overlapping patches.
    """

    def __init__(
        self,
        n_chans: int,
        n_patches: int,
        stride: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_patches = n_patches
        self.stride = stride
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PatchEmbeddingLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch, channels, time)`.

        Returns
        -------
        torch.Tensor
            A tensor of patches with shape
            `(batch, n_patches, channels, patch_size)`.
        """
        x = x.to(device=self.device, dtype=self.dtype)

        time = x.shape[-1]
        patch_size = time // self.n_patches
        stride = (self.stride, 1) if self.stride is not None else (patch_size, 1)

        x_unsqueezed = x.unsqueeze(-1)
        patches = F.unfold(
            input=x_unsqueezed, kernel_size=(patch_size, 1), stride=stride
        )
        patches = Rearrange(
            "batch (chans time) patches -> batch patches chans time",
            time=patch_size,
            chans=self.n_chans,
        )(patches)
        return patches


class Vec(nn.Module):
    """Vectorization Layer.

    This layer vectorizes a batch of matrices along the last two dimensions.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Vec layer.

        Parameters
        ----------
        X : torch.Tensor
            A batch of matrices with shape `(..., n, k)`.

        Returns
        -------
        torch.Tensor
            A batch of vectorized matrices with shape `(..., n * k)`.
        """
        return vec_batch(X).to(device=self.device, dtype=self.dtype)

    def inverse_transform(self, X: torch.Tensor, n_rows: int) -> torch.Tensor:
        """Inverse transform of the Vec layer.

        Parameters
        ----------
        X : torch.Tensor
            A batch of vectorized matrices with shape `(..., n_rows * k)`.
        n_rows : int
            The number of rows in the original matrices.

        Returns
        -------
        torch.Tensor
            A batch of matrices with shape `(..., n_rows, k)`.
        """
        return unvec_batch(X, n_rows)


class Vech(nn.Module):
    """Vectorize Triangular Part Layer.

    This layer vectorizes the triangular part of a batch of symmetric matrices.
    By default, extracts the upper triangular with norm-preserving scaling.

    Parameters
    ----------
    preserve_norm : bool, default=True
        If True, applies sqrt(2) scaling to off-diagonal elements so that
        ``||vec(X)||_2 = ||X||_F``. If False, extracts raw triangular elements.
    upper : bool, default=True
        If True, extracts upper triangular elements. If False, extracts lower
        triangular elements.
    device : torch.device, optional
        Device to place the output tensor on.
    dtype : torch.dtype, optional
        Data type of the output tensor.
    """

    def __init__(
        self,
        preserve_norm: bool = True,
        upper: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.preserve_norm = preserve_norm
        self.upper = upper
        self.device = device
        self.dtype = dtype

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Vech layer.

        Parameters
        ----------
        X : torch.Tensor
            A batch of symmetric matrices with shape `(..., n, n)`.

        Returns
        -------
        torch.Tensor
            A batch of vectorized matrices with shape
            `(..., n * (n + 1) // 2)`.
        """
        result = sym_to_upper(X, preserve_norm=self.preserve_norm, upper=self.upper)
        return result.to(device=self.device, dtype=self.dtype)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Inverse transform of the Vech layer.

        Parameters
        ----------
        X : torch.Tensor
            A batch of vectorized matrices with shape
            `(..., n * (n + 1) // 2)`.

        Returns
        -------
        torch.Tensor
            A batch of symmetric matrices with shape `(..., n, n)`.
        """
        return vec_to_sym(X, preserve_norm=self.preserve_norm, upper=self.upper)
