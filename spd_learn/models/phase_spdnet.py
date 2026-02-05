# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from warnings import warn

import torch
import torch.nn as nn

from spd_learn.models.spdnet import SPDNet


class PhaseSPDNet(nn.Module):
    """Phase SPDNet.

    This class implements the Phase SPDNet model :cite:p:`carrara2024eegspd`.
    This model first applies a Phase-Space Embedding (PSE) to the input time
    series data, followed by an SPDNet for classification.

    .. figure:: /_static/models/phase_spdnet.png
       :align: center
       :alt: Phase SPDNet Architecture

    Parameters
    ----------
    subspacedim : int, optional
        The dimension of the subspace for the SPDNet's bilinear mapping
        layers. If `None`, it defaults to `n_chans * order`.
    input_type : str, default="raw"
        Specifies the input type for the SPDNet component.
    threshold : float, default=1e-4
        Regularization threshold used within SPDNet.
    upper : bool, default=True
        If `True`, uses the upper triangular part for certain operations
        within SPDNet.
    n_chans : int
        Number of channels in the original input time series.
    n_outputs : int
        Number of output units for the final classification or regression
        layer.
    order : int, default=1
        The embedding order (dimension) for the Phase-Space Embedding.
    lag : int, default=1
        The time lag used between consecutive samples in the Phase-Space
        Embedding.
    """

    def __init__(
        self,
        subspacedim=None,
        input_type="raw",
        threshold=1e-4,
        upper=True,
        n_chans=None,
        n_outputs=None,
        order=1,
        lag=1,
    ):
        super().__init__()

        if subspacedim is None:
            warn(
                "subspacedim is None, using the default value of "
                "the number of channels",
                UserWarning,
            )

            subspacedim = int(n_chans * order)

        self.phase = PhaseDelay(order=order, lag=lag)

        self.spdnet = SPDNet(
            subspacedim=subspacedim,
            input_type="raw",
            threshold=threshold,
            upper=upper,
            n_chans=int(n_chans * order),
            n_outputs=n_outputs,
        )

    def forward(self, input):
        """Forward pass of the PhaseSPDNet model.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_outputs)`.
        """
        x_delay = self.phase(input)
        out = self.spdnet(x_delay)

        return out


class PhaseDelay(nn.Module):
    """Phase-Space Embedding Layer.

    This class implements the phase-space embedding layer
    :cite:p:`carrara2024eegspd`.

    Parameters
    ----------
    order : int, default=1
        Order of the phase space.
    lag : int, default=1
        Delay of the phase space.
    """

    def __init__(self, order: int = 1, lag: int = 1, device=None, dtype=None):
        super().__init__()
        if order < 1:
            raise ValueError("order must be >= 1.")
        if lag < 1:
            raise ValueError("lag (delay) must be >= 1.")
        self.order = order
        self.lag = lag

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Performs lagged concatenation on the input tensor.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch, channels, time_length)`.

        Returns
        -------
        torch.Tensor
            Augmented tensor of shape `(batch, channels * order, time_out)`,
            where `time_out = n_times - (order * lag)`.
        """
        batch_size, channels, time = input.shape
        if self.order == 1:
            return input
        else:
            return torch.concat(
                [
                    input[:, :, p * self.lag : -(self.order - p) * self.lag]
                    for p in range(0, self.order)
                ],
                dim=1,
            )
