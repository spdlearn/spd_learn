# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Optional

import torch

from torch import Tensor

from .numerical import numerical_config


def dropout_spd(
    input_mat: Tensor,
    p: float = 0.5,
    use_scaling: bool = True,
    epsilon: Optional[float] = None,
    device=None,
    dtype=None,
) -> Tensor:
    """Applies dropout to a batch of SPD matrices.

    This function applies dropout to a batch of SPD matrices. Dropped
    channels have their diagonal set to `epsilon` and off-diagonal entries
    zeroed.

    The input is expected to have shape `(..., dim, dim)`.

    Parameters
    ----------
    input_mat : Tensor
        Input SPD matrices with shape `(..., dim, dim)`.
    p : float, default=0.5
        Dropout probability.
    use_scaling : bool, default=True
        If `True`, the output is scaled by `1 / (1 - p)` to maintain the
        expected value.
    epsilon : float, optional
        Value for the diagonal of dropped channels. If None, uses the value
        from the unified numerical configuration. Default: None.
    device : torch.device, optional
        The device to move the output tensor to.
    dtype : torch.dtype, optional
        The data type to cast the output tensor to.

    Returns
    -------
    Tensor
        Dropped-out SPD matrices of the same shape and dtype as the input.
    """
    if not (0 <= p <= 1):
        raise ValueError(f"dropout probability must be in [0,1], got {p}")

    if input_mat.dim() < 2 or input_mat.shape[-2] != input_mat.shape[-1]:
        raise ValueError(
            f"input_mat must have shape (..., dim, dim), got {input_mat.shape}"
        )

    device = device or input_mat.device
    dtype = dtype or input_mat.dtype

    # Use unified config if epsilon not specified
    if epsilon is None:
        epsilon = numerical_config.dropout_eps

    *batch_shape, dim, _ = input_mat.shape

    mask = (
        torch.empty(*batch_shape, dim, device=device).bernoulli_(1 - p).to(dtype=dtype)
    )

    mask_outer = torch.einsum("...i,...j->...ij", mask, mask)

    dropped = 1 - mask
    # Set dropped diagonals to epsilon (independent of original diagonal values).
    diag_scale = dropped * epsilon
    mask_diag = torch.diag_embed(diag_scale)

    output = input_mat * mask_outer + mask_diag

    if use_scaling and p != 1.0:
        output = output.mul_(1.0 / (1.0 - p))

    return output
