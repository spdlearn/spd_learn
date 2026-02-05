# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from collections import OrderedDict

import torch.nn as nn

from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDDropout


class EEGSPDNet(nn.Module):
    r"""EE(G) SPDNet.

    This class implements the EE(G) SPDNet model :cite:p:`wilson2025deep`.
    EE(G) SPDNet is designed for EEG signal classification, featuring
    channel-specific convolution, covariance matrix pooling, and an SPDNet
    for processing the resulting SPD matrices.

    .. figure:: /_static/models/eegspdnet.jpeg
       :align: center
       :alt: EE(G) SPDNet Architecture

    The model consists of the following layers:

    - A channel-specific convolutional layer that applies a 1D convolution to
      the input EEG data, using separate filters for each channel.
    - A sample covariance matrix (SCM) pooling layer that computes SCMs from
      the convolved signals.
    - An SPDNet that learns representations from the SPD matrices via a
      series of BiMap, SPD dropout, and ReEig layers, optionally followed by
      a LogEig layer.

    Parameters
    ----------
    n_chans : int
        Number of input EEG channels (electrodes).
    n_outputs : int
        Number of output classes for classification.
    n_filters : int, default=10
        Number of convolutional filters per channel.
    bimap_sizes : tuple, default=(2, 3)
        A tuple defining the scaling factor and number of BiMap layers.
    filter_time_length : int, default=25
        Length of the convolutional filter.
    final_layer_drop_prob : float, default=0
        Dropout probability applied before the final linear layer.
    spd_drop_prob : float, default=0
        Dropout probability used in the SPDNet layers.
    spd_drop_scaling : bool, default=True
        Whether to use scaling in the SPD dropout.
    """

    def __init__(
        self,
        n_chans,
        n_outputs,
        n_filters=10,
        bimap_sizes=(2, 3),
        filter_time_length=25,
        final_layer_drop_prob=0,
        spd_drop_prob=0,
        spd_drop_scaling=True,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.bimap_sizes = bimap_sizes
        self.final_layer_drop_prob = final_layer_drop_prob
        self.spd_drop_prob = spd_drop_prob
        self.filter_time_length = filter_time_length
        self.spd_drop_scaling = spd_drop_scaling

        self.conv = nn.Conv1d(
            in_channels=self.n_chans,
            out_channels=n_filters * self.n_chans,
            kernel_size=self.filter_time_length,
            groups=self.n_chans,
        )

        self.cov_pool = CovLayer()

        bimap_sizes_ls = _parse_bimap_sizes(
            bimap_sizes, n_chans=self.n_chans, n_filters=self.n_filters
        )
        self.spdnet = self.create_spdnet(
            bimap_sizes_ls=bimap_sizes_ls,
            dropout=self.spd_drop_prob,
            dropout_scaling=self.spd_drop_scaling,
        )
        self.dropout = nn.Dropout(p=self.final_layer_drop_prob)

        in_features = int(bimap_sizes_ls[-1] * (bimap_sizes_ls[-1] + 1) / 2)

        self.linear = nn.Linear(in_features=in_features, out_features=self.n_outputs)

    def forward(self, x):
        x = self.conv(x)
        x = self.cov_pool(x)
        x = self.spdnet(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def create_spdnet(
        self,
        bimap_sizes_ls,
        threshold=5e-4,
        log=True,
        dropout=0,
        dropout_scaling=True,
    ):
        n_bimap_reeig = len(bimap_sizes_ls) - 1

        layers = OrderedDict()

        for i in range(n_bimap_reeig):
            size_in, size_out = bimap_sizes_ls[i], bimap_sizes_ls[i + 1]

            layers[f"bimap{i}"] = BiMap(in_features=size_in, out_features=size_out)

            if dropout > 0:
                layers[f"spd_dropout{i}"] = SPDDropout(
                    epsilon=threshold, use_scaling=dropout_scaling
                )

            layers[f"reeig{i}"] = ReEig(threshold=threshold)

        if log:
            layers["logeig"] = LogEig()

        spdnet = nn.Sequential(layers)
        return spdnet


def _parse_bimap_sizes(bimap_sizes: tuple, n_chans: int, n_filters: int) -> list[int]:
    assert n_chans is not None
    assert n_filters is not None
    k, n_bimap_reeig = bimap_sizes

    computed_sizes = [
        int((n_chans * n_filters) / k**i) for i in range(n_bimap_reeig + 1)
    ]

    return computed_sizes
