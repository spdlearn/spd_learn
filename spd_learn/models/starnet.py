# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from warnings import warn

import torch
import torch.nn as nn

from spd_learn.modules import BiMap, CovLayer, LogEig


class STaRNet(nn.Module):
    r"""Spatio-Temporal Riemannian Network (STaRNet).

    This class implements the STaRNet :cite:p:`wang2024starnet`.
    STaRNet is a neural network model designed for EEG signal classification,
    combining multi-scale spatial and temporal convolution branches with a
    covariance-based SPD feature extraction module.

    .. figure:: /_static/models/starnet.png
       :align: center
       :alt: STaRNet Architecture

    Parameters
    ----------
    n_chans : int
        Number of input EEG channels (electrodes).
    n_outputs : int
        Number of output classes for classification.
    spatial_kernel_sizes : Sequence[int], optional, default=(3, 11, 22)
        List of kernel sizes for the spatial convolutional layers.
    spatial_expansion_coeff : int, optional, default=44
        Expansion coefficient for the spatial convolutional layers.
    depth_spatial_fusion : int, optional, default=8
        Depth of the spatial fusion convolutional layer.
    depth_temporal_map : int, optional, default=4
        Depth of the temporal mapping convolutional layers.
    temporal_kernel_sizes : Sequence[int], optional, default=(25, 65, 125)
        List of kernel sizes for the temporal convolutional layers.
    bimap_reduced_dimension : int, optional, default=48
        Reduced dimension for the BiMap layer in the SPD feature extraction module.

    Notes
    -----
    The default dimensions are an engineering configuration for 22-channel EEG.
    The STaRNet paper does not report every architecture dimension, so these
    values should not be treated as an exact reproduction of the authors' model.
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        spatial_kernel_sizes: Sequence[int] = (3, 11, 22),
        spatial_expansion_coeff: int = 44,
        depth_spatial_fusion: int = 8,
        depth_temporal_map: int = 4,
        temporal_kernel_sizes: Sequence[int] = (25, 65, 125),
        bimap_reduced_dimension: int = 48,
    ):
        super().__init__()
        assert all(kernel_size <= n_chans for kernel_size in spatial_kernel_sizes)

        spatial_dimensions = [n_chans - h_i + 1 for h_i in spatial_kernel_sizes]
        raw_depths_spatial_maps = [
            spatial_expansion_coeff // S_i for S_i in spatial_dimensions
        ]
        depths_spatial_maps = [max(1, depth) for depth in raw_depths_spatial_maps]

        if depths_spatial_maps != raw_depths_spatial_maps:
            warn(
                "spatial_expansion_coeff produced zero spatial filters for some "
                f"branches. Using depths {depths_spatial_maps} instead of "
                f"{raw_depths_spatial_maps}.",
                UserWarning,
                stacklevel=2,
            )

        self.conv_spatial = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=d,
                    kernel_size=(h, 1),
                    bias=False,
                )
                for h, d in zip(spatial_kernel_sizes, depths_spatial_maps)
            ]
        )

        spatial_height = sum(
            d * s for d, s in zip(depths_spatial_maps, spatial_dimensions)
        )
        self.conv_spatial_fusion = nn.Conv2d(
            in_channels=1,
            out_channels=depth_spatial_fusion,
            kernel_size=(spatial_height, 1),
            bias=False,
        )

        self.conv_temporal = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=depth_temporal_map,
                    kernel_size=(1, w_j),
                    padding="same",
                    bias=False,
                )
                for w_j in temporal_kernel_sizes
            ]
        )
        temporal_height = (
            len(temporal_kernel_sizes) * depth_spatial_fusion * depth_temporal_map
        )
        self.spd_features = nn.Sequential(
            CovLayer(),
            BiMap(temporal_height, bimap_reduced_dimension),
            LogEig(),
        )
        self.classifier = nn.Linear(
            in_features=(bimap_reduced_dimension * (bimap_reduced_dimension + 1)) // 2,
            out_features=n_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network to raw EEG epochs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch_size, n_chans, n_times)``.

        Returns
        -------
        torch.Tensor
            Class logits with shape ``(batch_size, n_outputs)``.
        """
        x = x.unsqueeze(1)
        branches = [conv(x).flatten(1, 2).unsqueeze(1) for conv in self.conv_spatial]
        x = torch.cat(branches, dim=2)
        x = self.conv_spatial_fusion(x)
        x = x.flatten(1, 2).unsqueeze(1)
        branches = [conv(x).flatten(1, 2).unsqueeze(1) for conv in self.conv_temporal]
        x = torch.cat(branches, dim=2).squeeze(1)
        x = self.spd_features(x)
        x = self.classifier(x)
        return x
