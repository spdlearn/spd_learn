# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of Graph-CSPNet (Ju and Guan, IEEE TNNLS, 2023).
# Reference implementation: https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet

import torch

from torch import Tensor


def normalize_spd_adjacency(adjacency: Tensor) -> Tensor:
    r"""Normalize nonnegative adjacency weights with unit self-loops.

    Parameters
    ----------
    adjacency : torch.Tensor
        Finite nonnegative adjacency matrices of shape (..., nodes,
        nodes).

    Returns
    -------
    torch.Tensor
        Row-stochastic propagation matrices with the input shape, dtype
        and device.

    Notes
    -----
    Existing diagonal weights are retained before adding the self-loops.
    """
    if (
        adjacency.ndim < 2
        or adjacency.shape[-1] == 0
        or adjacency.shape[-1] != adjacency.shape[-2]
    ):
        raise ValueError("adjacency must have shape (..., nodes, nodes)")
    if not adjacency.is_floating_point() or not torch.isfinite(adjacency).all():
        raise ValueError("adjacency must be finite and floating point")
    if (adjacency < 0).any():
        raise ValueError("SPD aggregation requires nonnegative edge weights")
    augmented = adjacency + torch.eye(
        adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype
    )
    # Rescale before summing to avoid overflow for finite large edge strengths.
    scaled = augmented / augmented.amax(-1, keepdim=True)
    return scaled / scaled.sum(-1, keepdim=True)


def spd_graph_aggregate(input: Tensor, propagation: Tensor) -> Tensor:
    r"""Aggregate SPD graph nodes with nonnegative propagation weights.

    Parameters
    ----------
    input : torch.Tensor
        SPD nodes of shape (..., nodes, channels, channels).
    propagation : torch.Tensor
        Row-stochastic weights of shape (..., nodes, nodes), matching
        dtype/device.

    Returns
    -------
    torch.Tensor
        Aggregated nodes with broadcast leading axes and unchanged matrix
        dimensions.

    Notes
    -----
    Aggregation is independent across samples. PSD inputs need only produce
    PSD outputs.
    """
    if input.ndim < 3 or input.shape[-1] != input.shape[-2]:
        raise ValueError("input must have shape (..., nodes, channels, channels)")
    if propagation.ndim < 2 or propagation.shape[-2:] != (
        input.shape[-3],
        input.shape[-3],
    ):
        raise ValueError("propagation and input node counts must match")
    if (
        not input.is_floating_point()
        or input.dtype != propagation.dtype
        or input.device != propagation.device
    ):
        raise ValueError("input and propagation must share floating dtype/device")
    if not torch.isfinite(propagation).all() or (propagation < 0).any():
        raise ValueError("propagation must be finite and nonnegative")
    if not torch.allclose(propagation.sum(-1), torch.ones_like(propagation.sum(-1))):
        raise ValueError("propagation rows must sum to one")
    return (propagation @ input.flatten(-2)).reshape(
        *torch.broadcast_shapes(input.shape[:-3], propagation.shape[:-2]),
        *input.shape[-3:],
    )
