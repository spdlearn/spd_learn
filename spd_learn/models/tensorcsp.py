# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import List, Tuple

import torch

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from spd_learn.modules import (
    BiMap,
    CovLayer,
    LogEig,
    PatchEmbeddingLayer,
    ReEig,
    SPDBatchNormMean,
)


class TensorCSPNet(nn.Module):
    """Tensor-CSPNet.

    This class implements the Tensor-CSPNet model :cite:p:`ju2022tensor`, an
    SPDNet framework for EEG-based motor imagery classification.

    .. figure:: /_static/models/tensorcspnet.png
       :align: center
       :alt: TensorCSPNet Architecture

    The architecture consists of four stages:

    1. **Tensor Stacking:** Segments the EEG signals into
       temporospatial-frequency tensors.
    2. **Common Spatial Pattern (CSP):** Uses modified SPDNet layers to
       capture spatial patterns.
    3. **Temporal Convolution:** Captures temporal dynamics using 2D CNN
       layers.
    4. **Classification:** Uses a linear layer or a multi-layer perceptron
       for final classification.

    Parameters
    ----------
    n_chans : int, default=22
        Number of input channels.
    n_outputs : int, default=4
        Number of output classes.
    n_patches : int, default=4
        Number of patches to split the temporal dimension into.
    n_freqs : int, default=9
        Number of frequency bands.
    use_mlp : bool, default=False
        Whether to use a multi-layer perceptron in the final layer.
    tcn_channels : int, default=16
        Number of channels for the temporal convolutional network.
    dims : tuple[int, int, int, int], default=(22, 36, 36, 22)
        Dimensions for the BiMap layers.
    momentum : float, default=0.1
        Momentum factor for the Riemannian Brooks Batch Normalization.
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 4,
        n_patches: int = 4,
        n_freqs: int = 9,
        use_mlp: bool = False,
        tcn_channels: int = 16,
        dims: Tuple[int, int, int, int] = (22, 36, 36, 22),
        momentum: float = 0.1,
    ):
        super().__init__()

        assert len(dims) % 2 == 0, "dims must have an even number of elements."

        self.use_mlp = use_mlp
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_patches = n_patches
        self.dims = dims
        self.tcn_channels = tcn_channels
        self.tcn_width = n_freqs
        self.momentum = momentum
        self.depthwise = self.tcn_width * self.n_patches

        self.temporal_segmenter = nn.Sequential(
            Rearrange("batch freq chans time-> (batch freq) chans time"),
            PatchEmbeddingLayer(n_chans=self.n_chans, n_patches=self.n_patches),
        )

        self.cov = CovLayer()

        self.tensor_stack = Rearrange(
            "batch windows_index freq_index chans1 chans2 ->"
            "batch (windows_index freq_index) chans1 chans2"
        )

        layers: List[nn.Module] = []
        layer_num = len(self.dims) // 2

        if layer_num > 1:
            for i in range(layer_num - 1):
                dim_in, dim_out = self.dims[2 * i], self.dims[2 * i + 1]
                layers.append(
                    BiMap(
                        depthwise=self.depthwise,
                        in_features=dim_in,
                        out_features=dim_out,
                        parametrized=False,
                    )
                )
                layers.append(ReEig())

        dim_in, dim_out = self.dims[-2], self.dims[-1]

        layers.append(
            BiMap(
                depthwise=self.depthwise,
                in_features=dim_in,
                out_features=dim_out,
                parametrized=False,
            )
        )
        layers.append(Rearrange("batch depth cov1 cov2 -> (batch depth) cov1 cov2"))
        layers.append(SPDBatchNormMean(momentum=self.momentum, num_features=dim_out))
        layers.append(ReEig())
        self.bimap_block = nn.Sequential(*layers)

        self.log_eig = LogEig(upper=False, flatten=False)

        self.temporal_block = nn.Conv2d(
            1,
            self.tcn_channels,
            (self.n_patches, self.tcn_width * self.dims[-1] ** 2),
            stride=(1, self.dims[-1] ** 2),
            padding=0,
        )

        self.final_layers: nn.Module
        if self.use_mlp:
            self.final_layers = nn.Sequential(
                nn.Linear(
                    in_features=self.tcn_channels, out_features=self.tcn_channels
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.tcn_channels, self.tcn_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.tcn_channels, self.n_outputs),
            )
        else:
            self.final_layers = nn.Linear(
                in_features=self.tcn_channels, out_features=self.n_outputs
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TensorCSPNet model.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape
            `(batch_size, n_freqs, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_outputs)`.
        """
        batch_size, freq_index, _, _ = input.shape

        x = self.temporal_segmenter(input)
        _, patch, _, _ = x.shape

        x = rearrange(
            x,
            "(batch freq) patch chans time -> (batch freq patch) chans time",
            batch=batch_size,
            freq=freq_index,
        )

        x = self.cov(x)
        x = rearrange(
            x,
            "(batch freq patch) ... -> batch (freq patch) ...",
            batch=batch_size,
            freq=freq_index,
            patch=patch,
        )

        x_csp = self.bimap_block(x)

        x_csp = rearrange(
            tensor=x_csp,
            pattern="(batch depth) cov1 cov2 -> batch depth cov1 cov2",
            batch=batch_size,
        )

        x_log = self.log_eig(x_csp)

        x_vec = x_log.view(batch_size, 1, patch, -1)

        x_temp = self.temporal_block(x_vec).view(batch_size, -1)

        y = self.final_layers(x_temp)

        return y
