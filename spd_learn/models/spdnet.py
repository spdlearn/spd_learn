# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from warnings import warn

import torch
import torch.nn as nn

from spd_learn.functional import covariance
from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig


class SPDNet(nn.Module):
    """Symmetric Positive Definite Neural Network (SPDNet).

    This class implements the SPDNet model :cite:p:`huang2017riemannian`. SPDNet
    is a deep learning architecture designed for processing SPD matrices, which
    are common in applications like diffusion tensor imaging and covariance-based
    representations of EEG signals.

    The network consists of a series of layers that operate directly on SPD
    matrices, preserving their geometric structure. The main layers are:

    - **BiMap**: A bilinear mapping layer for linear transformation on the SPD manifold.
    - **ReEig**: A rectified eigenvalue layer that clamps eigenvalues as non-linearity.
    - **LogEig**: Logarithmic eigenvalue layer mapping SPD matrices to Euclidean space.

    .. figure:: /_static/models/spdnet.png
       :align: center
       :alt: SPDNet Architecture

    Parameters
    ----------
    input_type : {"raw", "cov"}, default="raw"
        The type of input data. If "raw", the input is expected to be raw
        signals (batch, channels, time) and a `CovLayer` is automatically
        added to compute covariance matrices. If "cov", the input is expected
        to be already valid SPD matrices (batch, channels, channels).
    cov_method : Callable, default=covariance
        The covariance estimation method to use when `input_type` is "raw".
        Can be one of `covariance`, `sample_covariance`, `real_covariance`,
        or `cross_covariance`.
    subspacedim : int, optional
        The subspace dimension for the BiMap layer. If `None`, it is set to
        `n_chans`.
    threshold : float, default=1e-4
        The threshold for the rectified eigenvalue layer (ReEig).
    upper : bool, default=True
        If `True`, only the upper triangular part of the matrix is used in
        the LogEig layer.
    n_chans : int
        The number of channels in the input data. This is used to define the
        size of the covariance matrix and the BiMap layer.
    n_outputs : int
        The number of outputs for the final classification layer.
    """

    def __init__(
        self,
        input_type="raw",
        cov_method=covariance,
        subspacedim=None,
        threshold=1e-4,
        upper=True,
        n_chans=None,
        n_outputs=None,
    ):
        super().__init__()

        if subspacedim is None:
            warn(
                "subspacedim is None, using the default value of "
                "the number of channels",
                UserWarning,
            )
            subspacedim = n_chans

        if input_type == "raw":
            self.cov = CovLayer(method=cov_method)
        elif input_type == "cov":
            self.cov = nn.Identity()

        self.bimap = BiMap(n_chans, subspacedim)
        self.reeig = ReEig(threshold)
        self.logeig = LogEig(upper=upper)
        self.len_last_layer = (
            subspacedim * (subspacedim + 1) // 2 if upper else subspacedim**2
        )
        self.classifier = nn.Linear(self.len_last_layer, n_outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPDNet model.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. If `input_type` is "raw", the shape should be
            `(batch_size, n_channels, n_times)`. If `input_type` is "cov",
            the shape should be `(batch_size, n_channels, n_channels)`.

        Returns
        -------
        torch.Tensor
            Output of the classifier, with shape `(batch_size, n_outputs)`.
        """
        X = self.cov(X)
        X = self.bimap(X)
        X = self.reeig(X)
        X = self.logeig(X)
        X = self.classifier(X)

        return X
