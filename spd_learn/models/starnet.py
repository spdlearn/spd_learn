# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from warnings import warn

import torch
import torch.nn as nn

from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, TraceNorm


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
    bimap_dims : Sequence[int] or int, optional
        Optional alias for ``bimap_reduced_dimension`` for API consistency
        across SPDNet-family models.

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
        bimap_dims: Sequence[int] | int | None = None,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        if bimap_dims is not None:
            bimap_reduced_dimension = (
                bimap_dims[0] if isinstance(bimap_dims, Sequence) else int(bimap_dims)
            )
        self.bimap_dims = (bimap_reduced_dimension,)
        if (
            tuple(spatial_kernel_sizes) == (3, 11, 22)
            and spatial_expansion_coeff == 44
            and n_chans != 22
        ):
            spatial_kernel_sizes = (
                min(3, n_chans),
                max(1, n_chans // 2),
                n_chans,
            )
            spatial_expansion_coeff = max(44, 2 * n_chans)
        spatial_kernel_sizes = tuple(
            max(1, min(int(k), n_chans)) for k in spatial_kernel_sizes
        )
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
            TraceNorm(epsilon=1e-4),
            BiMap(temporal_height, bimap_reduced_dimension),
            ReEig(threshold=1e-4),
            LogEig(),
        )
        self.classifier = nn.Linear(
            in_features=(bimap_reduced_dimension * (bimap_reduced_dimension + 1)) // 2,
            out_features=n_outputs,
        )

    @staticmethod
    def _normalize_branch(branch: torch.Tensor) -> torch.Tensor:
        branch = branch - branch.mean(dim=-1, keepdim=True)
        rms = branch.square().mean(dim=(-2, -1), keepdim=True).add(1e-6).sqrt()
        return branch / rms

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
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = x.unsqueeze(1)
        branches = [
            self._normalize_branch(conv(x).flatten(1, 2).unsqueeze(1))
            for conv in self.conv_spatial
        ]
        x = torch.cat(branches, dim=2)
        x = self._normalize_branch(
            self.conv_spatial_fusion(x).flatten(1, 2).unsqueeze(1)
        )
        branches = [
            self._normalize_branch(conv(x).flatten(1, 2).unsqueeze(1))
            for conv in self.conv_temporal
        ]
        x = torch.cat(branches, dim=2).squeeze(1)
        x = self.spd_features(x)
        x = self.classifier(x)
        return x
