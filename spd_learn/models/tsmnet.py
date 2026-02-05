# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch

from torch import nn

from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDBatchNormMeanVar


class TSMNet(nn.Module):
    """Tangent Space Mapping Network (TSMNet).

    This class implements the TSMNet model :cite:p:`kobler2022spd`, which
    combines a convolutional feature extractor, latent covariance pooling,
    an SPDNet, and a Tangent Space Mapping (TSM).

    .. figure:: /_static/models/tsmnet.png
       :align: center
       :alt: TSMNet Architecture

    The model consists of the following layers:

    - A convolutional layer that applies the first two layers of
      ShallowConvNet :cite:p:`schirrmeister2017deep`.
    - A sample covariance matrix (SCM) pooling layer that creates SCMs from
      the convolved signals.
    - An SPDNet that includes a SPDBatchNormMeanVar layer before the LogEig layer.
    - A linear projection head.

    Parameters
    ----------
    n_chans : int
        Number of input channels.
    n_temp_filters : int, default=4
        Number of filters in the temporal convolution layer.
    temp_kernel_length : int, default=25
        Length of the 1D kernels in the temporal convolution layer.
    n_spatiotemp_filters : int, default=40
        Number of filters in the spatiotemporal convolution layer.
    n_bimap_filters : int, default=20
        Number of filters in the BiMap layer.
    reeig_threshold : float, default=1e-4
        Threshold for the ReEig layer.
    n_outputs : int
        Number of output dimensions.
    """

    def __init__(
        self,
        n_chans=None,
        n_temp_filters=4,
        temp_kernel_length=25,
        n_spatiotemp_filters=40,
        n_bimap_filters=20,
        reeig_threshold=1e-4,
        n_outputs=None,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_temp_filters = n_temp_filters
        self.n_temp_kernel = temp_kernel_length
        self.n_spatiotemp_filters = n_spatiotemp_filters
        self.n_bimap_filters = n_bimap_filters
        self.reeig_threshold = reeig_threshold

        n_tangent_dim = int(n_bimap_filters * (n_bimap_filters + 1) / 2)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                1,
                self.n_temp_filters,
                kernel_size=(1, self.n_temp_kernel),
                padding="same",
                padding_mode="reflect",
            ),
            nn.Conv2d(
                self.n_temp_filters, self.n_spatiotemp_filters, (self.n_chans, 1)
            ),
            nn.Flatten(start_dim=2),
        )
        self.covpool = CovLayer()
        self.spdnet = nn.Sequential(
            BiMap(
                in_features=self.n_spatiotemp_filters, out_features=self.n_bimap_filters
            ),
            ReEig(threshold=self.reeig_threshold),
        )
        self.spdbnorm = SPDBatchNormMeanVar(
            self.n_bimap_filters,
            affine=True,
            bias_requires_grad=False,
            weight_requires_grad=True,
        )
        self.logeig = nn.Sequential(
            LogEig(),
            nn.Flatten(start_dim=1),
        )
        self.head = nn.Linear(in_features=n_tangent_dim, out_features=self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TSMNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_outputs)`.
        """
        x_filtered = self.cnn(x[:, None, ...])
        x_cov = self.covpool(x_filtered)
        x_spd = self.spdnet(x_cov)
        x_tangent = self.logeig(self.spdbnorm(x_spd))
        return self.head(x_tangent)
