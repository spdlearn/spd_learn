# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn

from spd_learn.modules import (
    BiMap,
    CovLayer,
    LogEig,
    PatchEmbeddingLayer,
    ReEig,
    Shrinkage,
    TraceNorm,
)


class SqueezeExcitation(nn.Module):
    r"""Squeeze-and-Excitation channel attention block.

    This module recalibrates channel-wise feature responses adaptively by
    explicitly modeling interdependencies between convolutional channels.

    Parameters
    ----------
    channels : int
        Number of input channels.
    reduction : int, default=4
        Reduction ratio for the bottleneck layer.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SqueezeExcitation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Channel-reweighted tensor of shape ``(batch, channels, height, width)``.
        """
        return x * self.fc(x)


class STaRNet(nn.Module):
    r"""Spatio-Temporal and Riemannian Network (STaRNet).

    This class implements the STaRNet model :cite:p:`wang2024starnet`, a hybrid
    dual-branch architecture for high-performance motor imagery decoding in EEG
    brain-computer interfaces.

    .. figure:: /_static/models/starnet.png
       :align: center
       :alt: STaRNet Architecture

    The model synergistically couples a multi-scale spatio-temporal convolution
    frontend with two complementary processing streams:

    1. **Frontend:** Temporal convolution followed by depthwise spatial
       filtering across EEG electrodes, batch normalization, and ELU activation.
    2. **Riemannian Branch:** Temporal segmentation into patches via
       :class:`~spd_learn.modules.PatchEmbeddingLayer`, regularized covariance
       estimation via :class:`~spd_learn.modules.CovLayer`,
       :class:`~spd_learn.modules.TraceNorm`, and :class:`~spd_learn.modules.Shrinkage`,
       followed by patch-wise bilinear dimensionality reduction
       (:class:`~spd_learn.modules.BiMap` with ``depthwise=n_patches``),
       eigenvalue rectification (:class:`~spd_learn.modules.ReEig`), and tangent
       space projection (:class:`~spd_learn.modules.LogEig` with ``upper=True``).
    3. **Euclidean Temporal Branch:** Temporal average pooling, 2D convolution,
       batch normalization, ELU, and Squeeze-and-Excitation (SE) channel
       attention, pooled into compact temporal representations via
       :class:`nn.AdaptiveAvgPool2d((1, 8))`.
    4. **Feature Fusion & Classification:** Concatenation of Riemannian tangent
       features and Euclidean temporal features, followed by dropout and a
       linear classification head.

    Parameters
    ----------
    n_chans : int, default=22
        Number of input EEG channels (electrodes).
    n_outputs : int, default=2
        Number of output classes for classification.
    n_patches : int, default=4
        Number of temporal patches for the Riemannian branch.
    temporal_filters : int, default=16
        Number of temporal convolution filters in the frontend.
    spatial_filters : int, default=16
        Number of spatial filters in the frontend.
    bimap_dim : int, default=12
        Output dimension for the BiMap layer in the Riemannian branch.
    drop_prob : float, default=0.2
        Dropout probability before the final linear layer.
    temporal_kernel_size : int, default=31
        Kernel size for the temporal convolution in the frontend.
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        n_patches: int = 4,
        temporal_filters: int = 16,
        spatial_filters: int = 16,
        bimap_dim: int = 12,
        drop_prob: float = 0.2,
        temporal_kernel_size: int = 31,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_patches = n_patches
        self.temporal_filters = temporal_filters
        self.spatial_filters = spatial_filters
        self.bimap_dim = bimap_dim
        self.drop_prob = drop_prob
        self.temporal_kernel_size = temporal_kernel_size

        # 1. Spatio-temporal frontend
        temporal_padding = temporal_kernel_size // 2
        self.temporal_conv = nn.Conv2d(
            1,
            temporal_filters,
            kernel_size=(1, temporal_kernel_size),
            padding=(0, temporal_padding),
            bias=False,
        )
        groups = (
            temporal_filters
            if (spatial_filters % temporal_filters == 0)
            else 1
        )
        self.spatial_conv = nn.Conv2d(
            temporal_filters,
            spatial_filters,
            kernel_size=(n_chans, 1),
            groups=groups,
            bias=False,
        )
        self.bn_frontend = nn.BatchNorm2d(spatial_filters)
        self.elu = nn.ELU()

        # 2. Riemannian Branch
        self.patch_embed = PatchEmbeddingLayer(
            n_chans=spatial_filters, n_patches=n_patches
        )
        self.cov = CovLayer()
        self.trace_norm = TraceNorm()
        self.shrinkage = Shrinkage(n_chans=spatial_filters)
        self.bimap = BiMap(spatial_filters, bimap_dim, depthwise=n_patches)
        self.reeig = ReEig()
        self.logeig = LogEig(upper=True)
        self.flatten_riemann = nn.Flatten()

        tangent_dim = bimap_dim * (bimap_dim + 1) // 2
        self.n_riemann_features = n_patches * tangent_dim

        # 3. Euclidean Temporal Branch
        self.euclid_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 15),
                padding=(0, 7),
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            SqueezeExcitation(spatial_filters),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Flatten(),
        )
        self.n_euclid_features = spatial_filters * 8

        # 4. Fusion and classification
        total_features = self.n_riemann_features + self.n_euclid_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_prob),
            nn.Linear(total_features, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of STaRNet.

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
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Frontend: (batch, spatial_filters, 1, n_times)
        feat = self.elu(self.bn_frontend(self.spatial_conv(self.temporal_conv(x))))

        # Riemannian branch
        x_r = feat.squeeze(2)  # (batch, spatial_filters, n_times)
        x_r = self.patch_embed(x_r)
        x_r = self.cov(x_r)
        x_r = self.trace_norm(x_r)
        x_r = self.shrinkage(x_r)
        x_r = self.bimap(x_r)
        x_r = self.reeig(x_r)
        x_r = self.logeig(x_r)
        x_r = self.flatten_riemann(x_r)

        # Euclidean branch
        x_e = self.euclid_branch(feat)

        # Fusion
        x_fused = torch.cat([x_r, x_e], dim=1)
        return self.classifier(x_fused)
