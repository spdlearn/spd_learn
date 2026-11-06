# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Symmetric Positive Definite Neural Network (SPNN).

This module implements the SPNN architecture based on:
Ziheng Chen, Rui Wang, and Nicu Sebe,
"Building Transformation Layers for Riemannian Neural Networks,"
NeurIPS 2026 (RieTrans).
"""

from typing import Tuple

import torch

from torch import nn

from ..modules import (
    SPDRMLR,
    BiMap,
    CovLayer,
    ReEig,
    SPDDropout,
    SPDLinear,
)


class SPNN(nn.Module):
    r"""Symmetric Positive Definite Neural Network (SPNN).

    Implements the deep Riemannian architecture introduced in RieTrans
    :cite:p:`chen2026rietrans`. SPNN transforms covariance representations
    through a cascade of intrinsic Riemannian linear layers
    (:class:`~spd_learn.modules.SPDLinear`)
    interleaved with Riemannian non-linear activation
    (:class:`~spd_learn.modules.ReEig`),
    and performs final classification via Riemannian Multinomial Logistic
    Regression (:class:`~spd_learn.modules.SPDRMLR`):

    .. math::

        X \xrightarrow{\text{Cov}} C \xrightarrow{\text{TraceNorm}} C_1
        \xrightarrow{\text{Shrinkage}} C_2
        \xrightarrow{\text{SPDLinear} + \text{ReEig}} \dots
        \xrightarrow{\text{SPDRMLR}} \hat{y}

    Parameters
    ----------
    n_chans : int, default=22
        Number of EEG channels (input dimensionality).
    n_outputs : int, default=2
        Number of output classification classes.
    bimap_dims : tuple of int, default=(16, 12)
        Intermediate SPD manifold dimensions for the stacked BiMap and SPDLinear layers.
    metric : {"LEM", "LCM", "AIM", "PEM"}, default="LEM"
        Riemannian metric for the SPDLinear and SPDRMLR layers.
    theta : float, default=0.5
        Power deformation parameter for PEM or RieTrans transformations.
    drop_prob : float, default=0.0
        Dropout probability. If > 0, applies :class:`~spd_learn.modules.SPDDropout`
        between intermediate SPD layers.

    References
    ----------
    .. bibliography::
       :filter: key in {"chen2024rmlr", "chen2026rietrans"}

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SPNN
    >>> model = SPNN(n_chans=22, n_outputs=2)
    >>> x = torch.randn(4, 22, 1000)
    >>> out = model(x)
    >>> out.shape
    torch.Size([4, 2])
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        bimap_dims: Tuple[int, ...] = (16, 12),
        metric: str = "LEM",
        theta: float = 0.5,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.bimap_dims = tuple(bimap_dims)
        self.metric = metric
        self.theta = theta
        self.drop_prob = drop_prob

        # Frontend
        self.cov = CovLayer()

        # Stack of BiMap stem + SPDLinear transformation layers with ReEig non-linearity
        layers: list[nn.Module] = []
        dims = [n_chans, *self.bimap_dims]
        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            if i == 0:
                layers.append(BiMap(in_features=d_in, out_features=d_out))
            else:
                layers.append(
                    SPDLinear(
                        in_features=d_in,
                        out_features=d_out,
                        metric=metric,
                        theta=theta,
                        return_tangent=False,
                    )
                )
            layers.append(ReEig())
            if drop_prob > 0.0:
                layers.append(SPDDropout(p=drop_prob))

        self.spd_layers = nn.Sequential(*layers)

        last_dim = self.bimap_dims[-1] if len(self.bimap_dims) > 0 else n_chans
        self.classifier = SPDRMLR(
            n_features=last_dim,
            n_outputs=n_outputs,
            metric=metric,
            theta=theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(..., n_chans, n_times)` or
            `(..., n_chans, n_chans)`.

        Returns
        -------
        torch.Tensor
            Classification logits of shape `(..., n_outputs)`.
        """
        if x.shape[-1] != x.shape[-2]:
            x = self.cov(x)
        x = self.spd_layers(x)
        return self.classifier(x)
