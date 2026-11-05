# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch
import torch.nn as nn

from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig


class RCEEGNet(nn.Module):
    """RCEEGNet.

    This class implements the RCEEGNet model :cite:p:`tibermacine2025rceegnet`.

    Parameters
    ----------
    n_chans : int
        Number of input EEG channels (electrodes).
    n_outputs : int, default=2
        Number of output classes for classification.
    n_spatial_filters : int, default=22
        Number of spatial filters learned by the convolutional frontend
        (dimension D in the paper). Sets the dimension of the initial
        covariance matrix.
    reduced_dim : int, default=8
        Target subspace dimension for the BiMap layer (dimension d
        in the paper, where reduced_dim <= n_spatial_filters).
    reg_cov : float, default=1e-5
        Diagonal regularization parameter (epsilon in Eq. 8 of the paper)
        ensuring positive definiteness of the sample covariance matrix.
    reeig_threshold : float, default=1e-12
        Floor threshold for eigenvalue clamping (epsilon_eig in Section
        III-E of the paper) for numerical stability before matrix logarithm.
    """

    def __init__(
        self,
        n_outputs: int = 2,
        n_chans: int = 22,
        n_spatial_filters: int = 22,
        reduced_dim: int = 8,
        reg_cov: float = 1e-5,
        reeig_threshold: float = 1e-12,
    ):
        super().__init__()

        self.conv_spatial = nn.Sequential(
                nn.Conv2d(
                in_channels=1,
                out_channels=n_spatial_filters,
                kernel_size=(n_chans, 1),
                bias=False,
            ),
            nn.BatchNorm2d(n_spatial_filters),
            nn.ReLU(),
        )

        self.conv_temporal = nn.Sequential(
            nn.Conv2d(
                in_channels=n_spatial_filters,
                out_channels=n_spatial_filters,
                kernel_size=(1, 5),
                padding=(0,2),
                bias=False,
            ),
            nn.BatchNorm2d(n_spatial_filters),
            nn.ReLU(),
        )
        self.covariance = CovLayer()
        self.register_buffer("identity", torch.eye(n_spatial_filters)*reg_cov)

        self.spd_features = nn.Sequential(
            BiMap(n_spatial_filters, reduced_dim, init_method="stiefel"),
            ReEig(threshold=reeig_threshold),
            LogEig()
        )

        self.classifier = nn.Linear(
            in_features = (reduced_dim * (reduced_dim + 1)) // 2,
            out_features = n_outputs,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize weights according to Section H.a of Tibermacine et al. (2025)."""
        nn.init.kaiming_uniform_(self.conv_spatial[0].weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv_temporal[0].weight, nonlinearity="relu")

        if hasattr(self.spd_features[0], "reset_parameters"):
            self.spd_features[0].reset_parameters()

        m = self.classifier.in_features
        bound = (6.0 / m) ** 0.5
        nn.init.uniform_(self.classifier.weight, -bound, bound)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RCEEGNet model.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_outputs)`.
        """
        x = x.unsqueeze(1)
        x = self.conv_spatial(x)
        x = self.conv_temporal(x)
        x = self.covariance(x.squeeze(2)) + self.identity
        x = self.spd_features(x)
        x = self.classifier(x)
        return x
