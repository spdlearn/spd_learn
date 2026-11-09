# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import torch.nn as nn

from torch.nn.utils.parametrize import is_parametrized, register_parametrization

from spd_learn.functional import (
    covariance,
    ensure_sym,
    matrix_sqrt,
    spd_rebiasing,
)
from spd_learn.modules import (
    BiMap,
    CovLayer,
    LogEig,
    ReEig,
    Shrinkage,
    SPDBatchNormMean,
    SymmetricPositiveDefinite,
    TraceNorm,
)


class GyroSPD(nn.Module):
    r"""Gyrovector Space Network on the SPD Manifold (GyroSPD).

    This class implements the GyroSPD architecture :cite:p:`nguyen2023building`,
    which formulates neural network operations on the Symmetric Positive Definite
    (SPD) manifold using gyrovector space calculus and left gyro-translations.

    The model consists of:

    1. **Preprocessing**: Covariance estimation via
       :class:`~spd_learn.modules.CovLayer`,
       trace normalization via :class:`~spd_learn.modules.TraceNorm`, and learnable
       shrinkage regularization via :class:`~spd_learn.modules.Shrinkage`.
    2. **Gyro-Transformation Stages**: Stacked :class:`GyroSPDLayer` layers across
       subspace dimensions `(n_chans, *bimap_dims)`, each combining bilinear mapping,
       rectified eigenvalue non-linearity, and left gyro-translation with a learnable
       SPD bias.
    3. **Riemannian Normalization**: Batch centering on the SPD manifold via
       :class:`~spd_learn.modules.SPDBatchNormMean`.
    4. **Riemannian-to-Euclidean Projection**: Tangent space projection and upper
       triangular vectorization via :class:`~spd_learn.modules.LogEig`.
    5. **Classification Head**: Dropout regularization and a fully connected
       linear layer.

    Parameters
    ----------
    n_chans : int, default=22
        Number of input channels (electrodes in EEG).
    n_outputs : int, default=2
        Number of output classes for classification.
    bimap_dims : tuple of int, default=(16, 12)
        Subspace dimensions for successive :class:`GyroSPDLayer` stages.
    drop_prob : float, default=0.0
        Dropout probability applied before the linear classification layer.
    input_type : {"raw", "cov"}, default="raw"
        Format of input data ('raw' time series or precomputed 'cov' matrices).
    cov_method : callable, default=covariance
        Method to compute covariance when `input_type` is 'raw'.
    threshold : float, default=1e-4
        Eigenvalue rectification threshold for :class:`~spd_learn.modules.ReEig`.
    upper : bool, default=True
        If `True`, only upper-triangular elements are used in tangent projection.
    device : torch.device, optional
        Device on which module parameters are allocated. Default is None.
    dtype : torch.dtype, optional
        Data type of module parameters. Default is None.

    See Also
    --------
    :class:`GyroSPDLayer` : Building block layer applying left gyro-translation.
    :class:`~spd_learn.models.SPDResNet` : Riemannian ResNet on the SPD manifold.

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import GyroSPD
    >>> model = GyroSPD(n_chans=22, n_outputs=2)
    >>> x = torch.randn(2, 22, 1000)
    >>> out = model(x)
    >>> out.shape
    torch.Size([2, 2])
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        bimap_dims: tuple[int, ...] = (16, 12),
        drop_prob: float = 0.0,
        input_type: str = "raw",
        cov_method=covariance,
        threshold: float = 1e-4,
        upper: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if input_type not in ("raw", "cov"):
            raise ValueError(
                f"Unknown input_type: '{input_type}'. Expected 'raw' or 'cov'."
            )
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.bimap_dims = tuple(bimap_dims)
        self.drop_prob = drop_prob
        self.input_type = input_type
        self.cov_method = cov_method
        self.threshold = threshold
        self.upper = upper

        self.cov: nn.Module
        if input_type == "raw":
            self.cov = CovLayer(method=cov_method, device=device, dtype=dtype)
        else:
            self.cov = nn.Identity()

        self.trace_norm = TraceNorm()
        self.shrinkage = Shrinkage(n_chans, device=device, dtype=dtype)

        layers: list[nn.Module] = []
        in_dim = n_chans
        for out_dim in self.bimap_dims:
            layers.append(
                GyroSPDLayer(
                    in_dim,
                    out_dim,
                    threshold=threshold,
                    device=device,
                    dtype=dtype,
                )
            )
            in_dim = out_dim
        self.layers = nn.Sequential(*layers)

        final_dim = self.bimap_dims[-1] if len(self.bimap_dims) > 0 else n_chans
        self.batch_norm = SPDBatchNormMean(final_dim, device=device, dtype=dtype)
        self.logeig = LogEig(upper=upper)
        self.dropout = nn.Dropout(drop_prob)

        n_features = final_dim * (final_dim + 1) // 2 if upper else final_dim**2
        self.classifier = nn.Linear(n_features, n_outputs, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GyroSPD model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. If raw signal, shape is `(batch_size, n_chans, n_times)`.
            If precomputed covariance, shape is `(batch_size, n_chans, n_chans)`.

        Returns
        -------
        torch.Tensor
            Class logits of shape `(batch_size, n_outputs)`.
        """
        if self.input_type != "raw" or (x.ndim == 3 and x.shape[-1] != x.shape[-2]):
            x = self.cov(x)
        x = self.trace_norm(x)
        x = self.shrinkage(x)
        x = self.layers(x)
        x = self.batch_norm(x)
        x = self.logeig(x)
        x = self.dropout(x)
        return self.classifier(x)


class GyroSPDLayer(nn.Module):
    r"""Gyrovector Space Layer on the SPD Manifold.

    This module applies a bilinear projection :class:`~spd_learn.modules.BiMap`
    followed by eigenvalue rectification :class:`~spd_learn.modules.ReEig`, and
    then applies **left gyro-translation** on the SPD manifold
    :cite:p:`nguyen2023building`:

    .. math::

        X \oplus B = B^{1/2} X B^{1/2}

    where :math:`B \in \mathcal{S}_{++}^{\text{out\_features}}` is a learnable SPD bias
    matrix parameterized via :class:`~spd_learn.modules.SymmetricPositiveDefinite`.

    Parameters
    ----------
    in_features : int
        Input SPD matrix dimension.
    out_features : int
        Output SPD matrix dimension.
    threshold : float, default=1e-4
        Rectification threshold for :class:`~spd_learn.modules.ReEig`.
    device : torch.device, optional
        Device on which module parameters are allocated. Default is None.
    dtype : torch.dtype, optional
        Data type of module parameters. Default is None.

    See Also
    --------
    :class:`GyroSPD` : Full architecture combining stacked gyro layers.
    :class:`~spd_learn.modules.BiMap` : Bilinear projection layer.
    :class:`~spd_learn.modules.SymmetricPositiveDefinite` : SPD parametrization.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import GyroSPDLayer
    >>> layer = GyroSPDLayer(in_features=8, out_features=4)
    >>> X = torch.randn(2, 8, 8)
    >>> X = X @ X.mT + 0.1 * torch.eye(8)
    >>> out = layer(X)
    >>> out.shape
    torch.Size([2, 4, 4])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float = 1e-4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.bimap = BiMap(in_features, out_features, device=device, dtype=dtype)
        self.reeig = ReEig(threshold=threshold, device=device, dtype=dtype)

        self.bias = nn.Parameter(torch.eye(out_features, device=device, dtype=dtype))
        register_parametrization(
            self, "bias", SymmetricPositiveDefinite(device=device, dtype=dtype)
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset layer parameters to their initial states."""
        self.bimap.reset_parameters()
        if is_parametrized(self, "bias"):
            setattr(
                self,
                "bias",
                torch.eye(
                    self.out_features, device=self.bias.device, dtype=self.bias.dtype
                ),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass applying BiMap, ReEig, and left gyro-translation.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD tensor of shape `(..., in_features, in_features)`.

        Returns
        -------
        torch.Tensor
            Transformed SPD tensor of shape `(..., out_features, out_features)`.
        """
        X = self.reeig(self.bimap(X))
        B_sqrt = matrix_sqrt.apply(self.bias)
        return ensure_sym(spd_rebiasing(X, B_sqrt))
