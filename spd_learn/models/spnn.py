# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Symmetric Positive Definite Neural Network (SPNN).

This module implements the SPNN architecture based on:
Ziheng Chen, Rui Wang, and Nicu Sebe,
"Building Transformation Layers for Riemannian Neural Networks,"
NeurIPS 2026 (RieTrans).
"""

from typing import Optional, Tuple

import torch

from torch import nn

from ..modules import (
    SPDRMLR,
    BiMap,
    CovLayer,
    ReEig,
    Shrinkage,
    SPDBatchNormMean,
    SPDDropout,
    SPDLinear,
    TraceNorm,
)


class SPNN(nn.Module):
    r"""Symmetric Positive Definite Neural Network (SPNN).

    Implements the deep Riemannian architecture introduced in RieTrans
    :cite:p:`chen2026rietrans`. SPNN transforms raw multivariate time series or
    precomputed SPD covariance representations through trace-normalized
    shrinkage regularization, a cascade of bilinear/intrinsic Riemannian linear
    layers (:class:`~spd_learn.modules.BiMap`,
    :class:`~spd_learn.modules.SPDLinear`), and final classification via
    Riemannian Multinomial Logistic Regression
    (:class:`~spd_learn.modules.SPDRMLR`):

    .. math::

        X \xrightarrow{\text{MultiBandCov}} C \xrightarrow{\text{TraceNorm}} C_1
        \xrightarrow{\text{Shrinkage}} C_2
        \xrightarrow{\text{BiMap / SPDLinear}} \dots
        \xrightarrow{\text{SPDRMLR}} \hat{y}

    Parameters
    ----------
    n_chans : int, default=22
        Number of EEG channels (input dimensionality).
    n_outputs : int, default=2
        Number of output classification classes.
    bimap_dims : tuple of int, default=(16, 12)
        Intermediate SPD manifold dimensions for the stacked BiMap and SPDLinear layers.
    metric : {"LEM", "LCM", "AIM", "PEM", "BWM"}, default="LEM"
        Riemannian metric for the SPDLinear and SPDRMLR layers.
    theta : float, default=0.5
        Power deformation parameter for PEM or BWM transformations.
    drop_prob : float, default=0.0
        Dropout probability. If > 0, applies :class:`~spd_learn.modules.SPDDropout`
        between intermediate SPD layers.
    in_channels : int, default=1
        Number of input product-SPD channels.
    use_conv_stem : bool, default=True
        If True and input is a 3D time series ``(batch, n_chans, n_times)``,
        applies a multi-band temporal filterbank and aggregates trace-normalized
        band covariances to mitigate $1/f$ spectral dominance and rank deficiency.
    use_chart_bypass : bool, default=False
        If True, the final :class:`~spd_learn.modules.SPDLinear` layer returns
        tangent matrices (``return_tangent=True``) and the classifier
        :class:`~spd_learn.modules.SPDRMLR` consumes tangent matrices directly
        (``input_tangent=True``), bypassing redundant inverse/forward chart maps.
    act_power : float, optional
        Optional power deformation :math:`X \mapsto X^{\text{act\_power}}`
        applied before the first RieTrans tangent projection.
    unidirectional : bool, default=False
        If True, uses unidirectional geodesic hyperplanes in
        :class:`~spd_learn.modules.SPDLinear` and :class:`~spd_learn.modules.SPDRMLR`.
    alpha : float, default=1.0
        Weight of :math:`\operatorname{tr}(UV)` in the :math:`O(n)`-invariant
        inner product.
    beta : float, default=0.0
        Weight of :math:`\operatorname{tr}(U)\operatorname{tr}(V)` in the
        :math:`O(n)`-invariant inner product.
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type of parameters.

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
        in_channels: int = 1,
        use_conv_stem: bool = True,
        use_chart_bypass: bool = False,
        act_power: Optional[float] = None,
        unidirectional: bool = False,
        alpha: float = 1.0,
        beta: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.bimap_dims = tuple(bimap_dims)
        self.metric = metric
        self.theta = theta
        self.drop_prob = drop_prob
        self.in_channels = in_channels
        self.use_conv_stem = use_conv_stem
        self.use_chart_bypass = use_chart_bypass
        self.act_power = act_power
        self.unidirectional = unidirectional
        self.alpha = alpha
        self.beta = beta

        # Covariance & Regularization Frontend
        self.cov = CovLayer(device=device, dtype=dtype)
        self.trace_norm = TraceNorm(epsilon=1e-5, device=device, dtype=dtype)
        self.shrinkage = Shrinkage(
            n_chans=n_chans,
            init_shrinkage=-2.5,
            learnable=True,
            device=device,
            dtype=dtype,
        )

        if self.use_conv_stem:
            # ponytail: fixed k=25 4-band stem, scale with sfreq when sfreq >> 250 Hz
            n_bands = max(self.in_channels, 4)
            self.conv_stem = nn.Sequential(
                nn.Conv2d(
                    1,
                    n_bands,
                    kernel_size=(1, 25),
                    padding=(0, 12),
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                nn.BatchNorm2d(n_bands, device=device, dtype=dtype),
                nn.ELU(),
            )
            if self.in_channels == 1:
                init_logits = torch.zeros(n_bands + 1, device=device, dtype=dtype)
                init_logits[0] = 1.0
                self.band_logits = nn.Parameter(init_logits)

        # Stack of BiMap stem + SPDLinear transformation layers
        layers: list[nn.Module] = []
        dims = [n_chans, *self.bimap_dims]
        n_layers = len(dims) - 1
        used_act_power = False
        for i in range(n_layers):
            d_in, d_out = dims[i], dims[i + 1]
            is_last = i == n_layers - 1
            if (
                i == 0
                and self.in_channels == 1
                and not (is_last and self.use_chart_bypass)
            ):
                layers.append(
                    BiMap(
                        in_features=d_in,
                        out_features=d_out,
                        device=device,
                        dtype=dtype,
                    )
                )
                layers.append(ReEig(threshold=1e-4, device=device, dtype=dtype))
                if self.use_conv_stem:
                    layers.append(SPDBatchNormMean(d_out, device=device, dtype=dtype))
                if drop_prob > 0.0:
                    layers.append(
                        SPDDropout(
                            p=drop_prob, epsilon=1e-4, device=device, dtype=dtype
                        )
                    )
            else:
                layer_in_chans = self.in_channels if i == 0 else 1
                ret_tan = is_last and self.use_chart_bypass
                layer_act_power = act_power if not used_act_power else None
                used_act_power = True
                layers.append(
                    SPDLinear(
                        in_features=d_in,
                        out_features=d_out,
                        metric=metric,
                        theta=theta,
                        input_tangent=False,
                        return_tangent=ret_tan,
                        in_channels=layer_in_chans,
                        out_channels=1,
                        alpha=alpha,
                        beta=beta,
                        unidirectional=unidirectional,
                        act_power=layer_act_power,
                        device=device,
                        dtype=dtype,
                    )
                )
                if drop_prob > 0.0 and not ret_tan:
                    layers.append(
                        SPDDropout(
                            p=drop_prob, epsilon=1e-4, device=device, dtype=dtype
                        )
                    )

        self.spd_layers = nn.Sequential(*layers)

        last_dim = self.bimap_dims[-1] if n_layers > 0 else n_chans
        clf_in_chans = self.in_channels if n_layers == 0 else 1
        clf_input_tangent = self.use_chart_bypass and (n_layers > 0)
        self.classifier = SPDRMLR(
            n_features=last_dim,
            n_outputs=n_outputs,
            metric=metric,
            theta=theta,
            input_tangent=clf_input_tangent,
            in_channels=clf_in_chans,
            alpha=alpha,
            beta=beta,
            unidirectional=unidirectional,
            act_power=act_power if not used_act_power else None,
            device=device,
            dtype=dtype,
        )

    def _extract_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute regularized SPD covariance from time series or SPD input."""
        if x.shape[-1] == x.shape[-2]:
            return self.shrinkage(self.trace_norm(x))

        if not self.use_conv_stem:
            return self.shrinkage(self.trace_norm(self.cov(x)))

        x_centered = x - x.mean(dim=-1, keepdim=True)
        rms = x_centered.pow(2).mean(dim=(-2, -1), keepdim=True).add(1e-6).sqrt()
        x_norm = x_centered / rms

        if x_norm.ndim == 3:
            x_bands = self.conv_stem(x_norm.unsqueeze(1))
            if self.in_channels == 1:
                x_all = torch.cat([x_norm.unsqueeze(1), x_bands], dim=1)
                cov_all = self.trace_norm(self.cov(x_all))
                weights = torch.softmax(self.band_logits, dim=0).view(1, -1, 1, 1)
                return self.shrinkage((weights * cov_all).sum(dim=1))
            return self.shrinkage(
                self.trace_norm(self.cov(x_bands[:, : self.in_channels]))
            )

        return self.shrinkage(self.trace_norm(self.cov(x_norm)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(..., n_chans, n_times)` or
            `(..., n_chans, n_chans)` (or with ``in_channels`` dimension).

        Returns
        -------
        torch.Tensor
            Classification logits of shape `(..., n_outputs)`.
        """
        x = self._extract_covariance(x)
        x = self.spd_layers(x)
        return self.classifier(x)
