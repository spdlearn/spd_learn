# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Tuple

import torch
import torch.nn as nn

from spd_learn.modules import (
    BiMap,
    CovLayer,
    LogEig,
    ReEig,
    SPDBatchNormMean,
    Shrinkage,
    TraceNorm,
)


class RCEEGNet(nn.Module):
    r"""Geometry-Aware Convolutional Neural Network on SPD Manifolds (RCEEGNet).

    This class implements the RCEEGNet model :cite:p:`tibermacine2025geometry`,
    a geometry-aware architecture for EEG decoding that integrates Riemannian
    geometry on the manifold of Symmetric Positive Definite (SPD) matrices with
    2D convolutional refinement.

    .. figure:: /_static/models/rceegnet.png
       :align: center
       :alt: RCEEGNet Architecture

    The model consists of three main stages:

    1. **Temporal Filtering & Regularized Covariance Pooling:** A 1D temporal
       convolution extracts ``n_filters`` feature maps from multivariate EEG
       signals. Covariance pooling (:class:`~spd_learn.modules.CovLayer`) followed
       by trace normalization (:class:`~spd_learn.modules.TraceNorm`) and
       shrinkage regularization (:class:`~spd_learn.modules.Shrinkage`) yields
       well-conditioned SPD covariance matrices of shape
       ``(batch, n_filters, n_filters)``.
    2. **Riemannian SPD Network:** Stacked :class:`~spd_learn.modules.BiMap`,
       :class:`~spd_learn.modules.ReEig`, and
       :class:`~spd_learn.modules.SPDBatchNormMean` layers transform and reduce
       the dimensionality of the SPD matrices across ``(n_filters, *bimap_dims)``
       while preserving positive-definiteness on the manifold.
    3. **Tangent Space Convolutional Refinement & Classification:** A logarithmic
       mapping (:class:`~spd_learn.modules.LogEig` with ``upper=False``) projects the
       SPD matrices to full symmetric matrices in the tangent space at the identity.
       A 2D convolutional block (:class:`nn.Conv2d`, :class:`nn.BatchNorm2d`,
       :class:`nn.ELU`, :class:`nn.AdaptiveAvgPool2d`) refines geometry-aware
       spatial features before flattening, dropout, and linear classification.

    Parameters
    ----------
    n_chans : int, default=22
        Number of input EEG channels (electrodes).
    n_outputs : int, default=2
        Number of output classes for classification.
    n_filters : int, default=16
        Number of temporal convolutional filters.
    bimap_dims : tuple of int, default=(16, 12)
        Dimensionality progression for the stacked BiMap layers.
    conv_channels : int, default=8
        Number of channels in the 2D convolutional refinement block.
    drop_prob : float, default=0.25
        Dropout probability before the final linear layer.
    filter_time_length : int, default=31
        Kernel size for the temporal 1D convolution.
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        n_filters: int = 16,
        bimap_dims: Tuple[int, ...] = (16, 12),
        conv_channels: int = 8,
        drop_prob: float = 0.25,
        filter_time_length: int = 31,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_filters = n_filters
        self.bimap_dims = bimap_dims
        self.conv_channels = conv_channels
        self.drop_prob = drop_prob
        self.filter_time_length = filter_time_length

        padding = filter_time_length // 2
        self.temporal_conv = nn.Conv1d(
            in_channels=n_chans,
            out_channels=n_filters,
            kernel_size=filter_time_length,
            padding=padding,
            bias=False,
        )
        self.cov = CovLayer()
        self.trace_norm = TraceNorm()
        self.shrinkage = Shrinkage(n_chans=n_filters, init_shrinkage=-2.0)

        # Stacked BiMap + ReEig + SPDBatchNormMean
        dims = (n_filters, *bimap_dims)
        spd_layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            spd_layers.append(BiMap(in_features=d_in, out_features=d_out))
            spd_layers.append(ReEig())
            spd_layers.append(SPDBatchNormMean(num_features=d_out))
        self.spdnet = nn.Sequential(*spd_layers)

        # Tangent space projection LogEig(upper=False, flatten=False)
        self.logeig = LogEig(upper=False, flatten=False)

        # 2D geometry-aware convolutional refinement
        self.conv2d_refinement = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(p=drop_prob),
            nn.Linear(conv_channels * 16, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RCEEGNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, n_chans, n_times)`` or
            ``(batch, 1, n_chans, n_times)``.

        Returns
        -------
        torch.Tensor
            Classification logits of shape ``(batch, n_outputs)``.
        """
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = self.temporal_conv(x)
        x = self.cov(x)
        x = self.trace_norm(x)
        x = self.shrinkage(x)
        x = self.spdnet(x)
        x = self.logeig(x)
        x = x.unsqueeze(1)
        x = self.conv2d_refinement(x)
        return x
