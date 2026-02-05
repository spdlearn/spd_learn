# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from spd_learn.functional import (
    log_euclidean_distance,
    log_euclidean_mean,
    sample_covariance,
)
from spd_learn.modules import (
    BiMap,
    CovLayer,
    LogEig,
    PatchEmbeddingLayer,
    ReEig,
    TraceNorm,
)


class MAtt(nn.Module):
    r"""Manifold Attention Network for EEG Decoding (MAtt).

    This class implements the MAtt model :cite:p:`pan2022matt`, a manifold
    attention network for EEG decoding. The architecture integrates Riemannian
    geometry with attention mechanisms on the manifold of SPD matrices.

    .. figure:: /_static/models/matt.png
        :align: center
        :alt: MAtt Architecture

    The model consists of the following stages:

    1. **Feature Extraction:** Raw EEG signals are processed by two
       convolutional layers with batch normalization to extract spatial and
       spatiotemporal features.
    2. **Euclidean-to-Riemannian (E2R) Mapping:** The extracted features are
       segmented into patches, and a sample covariance matrix is computed for
       each patch to create SPD data points.
    3. **Manifold Attention Module:** An attention mechanism is applied to the
       SPD matrices, using bilinear mappings and the Log-Euclidean distance
       to compute attention scores.
    4. **Riemannian-to-Euclidean (R2E) Mapping:** A ReEig layer and a
       Log-Euclidean mapping project the SPD data back to a Euclidean space.
    5. **Classification:** A fully connected linear layer outputs the final
       class scores.

    Parameters
    ----------
    n_patches : int, default=2
        Number of patches for the time dimension.
    n_chans : int, default=22
        Number of output channels for the first convolutional layer.
    n_outputs : int, default=4
        Number of classes for classification.
    temporal_out_channels : int, default=20
        Number of output channels for the second convolutional layer.
    temporal_kernel_size : int, default=12
        Kernel size for the second convolutional layer.
    temporal_padding : int, default=6
        Padding for the second convolutional layer.
    attention_in_features : int, default=20
        Input feature dimension for the manifold attention module.
    attention_out_features : int, default=18
        Output feature dimension for the manifold attention module.
    covariance_method : callable, default=sample_covariance
        The method to use for computing covariance matrices.
    """

    def __init__(
        self,
        n_patches: int = 2,
        n_chans: int = 22,
        n_outputs: int = 4,
        temporal_out_channels: int = 20,
        temporal_kernel_size: int = 12,
        temporal_padding: int = 6,
        attention_in_features: int = 20,
        attention_out_features: int = 18,
        covariance_method=sample_covariance,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.temporal_out_channels = temporal_out_channels
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_padding = temporal_padding
        self.attention_in_features = attention_in_features
        self.attention_out_features = attention_out_features
        self.n_patches = n_patches

        self.add_extra_dim = Rearrange("batch chan time -> batch 1 chan time")

        self.spatial_filter = nn.Conv2d(
            in_channels=1,
            out_channels=n_chans,
            kernel_size=(n_chans, 1),
        )

        self.spatial_batch_norm = nn.BatchNorm2d(num_features=n_chans)
        self.temporal_feature_extractor = nn.Conv2d(
            in_channels=n_chans,
            out_channels=temporal_out_channels,
            kernel_size=(1, temporal_kernel_size),
            padding=(0, temporal_padding),
        )

        self.temporal_batch_norm = nn.BatchNorm2d(num_features=temporal_out_channels)
        self.squeeze_channel = Rearrange("batch kernel 1 time -> batch kernel time")

        self.patch_cov_layer = nn.Sequential(
            PatchEmbeddingLayer(n_patches=n_patches, n_chans=temporal_out_channels),
            CovLayer(
                method=covariance_method,
            ),
            TraceNorm(),
        )

        self.manifold_attention = AttentionManifold(
            in_features=attention_in_features,
            out_features=attention_out_features,
        )
        self.re_eig = ReEig()

        self.tangent = LogEig()

        self.flatten = nn.Flatten()
        tangent_dim = (attention_out_features * (attention_out_features + 1)) / 2
        in_features = int(n_patches * tangent_dim)
        self.linear = nn.Linear(
            in_features=in_features, out_features=n_outputs, bias=True
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MAtt model.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_outputs)`.
        """
        x = self.add_extra_dim(input)
        x = self.spatial_filter(x)
        x = self.spatial_batch_norm(x)
        x = self.temporal_feature_extractor(x)
        x = self.temporal_batch_norm(x)
        x = self.squeeze_channel(x)
        x = self.patch_cov_layer(x)
        x = self.manifold_attention(x)
        x = self.re_eig(x)
        x = self.tangent(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class AttentionManifold(nn.Module):
    """Manifold Attention Module.

    This module implements an attention mechanism over a sequence of SPD
    matrices, where the operations are performed on the SPD manifold using
    the Log-Euclidean metric :cite:p:`pan2022matt`.

    Parameters
    ----------
    in_features : int
        Dimensionality of input SPD matrices.
    out_features : int
        Dimensionality of the output features after mapping.
    """

    def __init__(self, in_features, out_features):
        super(AttentionManifold, self).__init__()

        self._in_features = in_features
        self._out_features = out_features

        self.q_trans = BiMap(self._in_features, self._out_features)
        self.k_trans = BiMap(self._in_features, self._out_features)
        self.v_trans = BiMap(self._in_features, self._out_features)

    def forward(self, x):
        """Forward pass of the manifold attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape `(batch, num_patch, ...)` where `...`
            represents the dimensions of the SPD matrices.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(-1, out_features, out_features)`.
        """
        Q = self.q_trans(x)
        K = self.k_trans(x)
        V = self.v_trans(x)

        Q_expand = Q.unsqueeze(2)
        K_expand = K.unsqueeze(1)

        atten_energy = log_euclidean_distance(Q_expand, K_expand)
        atten_weights = 1 / (1 + torch.log1p(atten_energy))
        atten_prob = F.softmax(atten_weights, dim=-2)
        atten_prob = atten_prob.permute(0, 2, 1)

        output = log_euclidean_mean(atten_prob, V)

        return output
