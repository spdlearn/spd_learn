# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Sequence

import torch
import torch.nn as nn

from spd_learn.functional import trace_normalization
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
    bimap_dims : Sequence[int] or int, optional
        Optional alias for ``reduced_dim`` for API consistency across
        SPDNet-family models.
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
        bimap_dims: Sequence[int] | int | None = None,
    ):
        super().__init__()
        if bimap_dims is not None:
            reduced_dim = (
                bimap_dims[0] if isinstance(bimap_dims, Sequence) else int(bimap_dims)
            )
        self.bimap_dims = (reduced_dim,)

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
                padding=(0, 2),
                bias=False,
            ),
            nn.BatchNorm2d(n_spatial_filters),
            nn.ReLU(),
        )
        self.covariance = CovLayer()
        self.identity: torch.Tensor
        self.register_buffer("identity", torch.eye(n_spatial_filters) * reg_cov)
        self.reeig_threshold = reeig_threshold
        eff_threshold = max(float(reeig_threshold), 1e-4)

        self.spd_features = nn.Sequential(
            BiMap(n_spatial_filters, reduced_dim, init_method="stiefel"),
            ReEig(threshold=eff_threshold),
            LogEig(),
        )

        self.classifier = nn.Linear(
            in_features=(reduced_dim * (reduced_dim + 1)) // 2,
            out_features=n_outputs,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initialize weights according to Section H.a of Tibermacine et al. (2025)."""
        for block in (self.conv_spatial, self.conv_temporal):
            if isinstance(block[0], nn.Conv2d):
                nn.init.kaiming_uniform_(block[0].weight, nonlinearity="relu")
        if isinstance(self.spd_features[0], BiMap):
            self.spd_features[0].reset_parameters()

        bound = (6.0 / self.classifier.in_features) ** 0.5
        nn.init.uniform_(self.classifier.weight, -bound, bound)
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
        x = (
            trace_normalization(self.covariance(x.squeeze(2)), epsilon=1e-3)
            + self.identity
        )
        x = self.spd_features(x)
        x = self.classifier(x)
        return x
