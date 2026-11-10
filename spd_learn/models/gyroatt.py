# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Gyrovector and Riemannian Attention Networks for SPD Manifolds.

This module implements the Gyrovector SPD Attention framework and architectures:
Rui Wang, Chen Hu, Xiaoning Song, Xiaojun Wu, Nicu Sebe, and Ziheng Chen,
"Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds,"
NeurIPS 2025 (GyroAtt).
"""

import math

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrize import is_parametrized, register_parametrization

from ..functional import (
    distance_euclid,
    ensure_sym,
    log_cholesky_distance,
    log_euclidean_distance,
    log_euclidean_mean,
    matrix_exp,
    matrix_sqrt,
    sample_covariance,
    spd_rebiasing,
)
from ..functional.metrics import cholesky_exp, cholesky_log
from ..modules import (
    BiMap,
    SPDLinear,
    SymmetricPositiveDefinite,
    TraceNorm,
)
from .matt import MAtt


class GyroSPDAttention(nn.Module):
    r"""Gyrovector Self-Attention Layer for SPD Matrices.

    Implements the general gyrovector attention mechanism on the SPD manifold
    introduced in GyroAtt :cite:p:`wang2025towards`.

    Query, Key, and Value representations are parameterized via orthogonal
    gyro-automorphisms :math:`O \in \mathrm{O}(d)` generated through the
    Cayley transform of skew-symmetric matrices:

    .. math::

        S = W - W^\top, \quad O = (I - S)(I + S)^{-1}

    Each patch SPD token :math:`X_i \in \mathcal{S}_{++}^d` is transformed as:

    .. math::

        Q_i = O_Q X_i O_Q^\top, \quad K_i = O_K X_i O_K^\top,
        \quad V_i = O_V X_i O_V^\top

    The pairwise manifold distance :math:`D_{ij}` between patch tokens is computed
    under either the Log-Euclidean Metric (LEM) or the Log-Cholesky Metric (LCM):

    .. math::

        D_{ij} = \lVert \log(Q_i) - \log(K_j) \rVert_F

    Attention weights are computed via logarithmic decay:

    .. math::

        A_{ij} = \operatorname{softmax}_j \left( \frac{1}{1 + \log(1 + D_{ij})} \right)

    The aggregated output is the weighted Fréchet mean in tangent/Cholesky space,
    followed by a learnable Riemannian left gyro-translation bias
    :math:`B \in \mathcal{S}_{++}^d`:

    .. math::

        \bar{V}_i = \exp \left( \sum_j A_{ij} \log(V_j) \right), \quad
        Y_i = B^{1/2} \bar{V}_i B^{1/2}

    Parameters
    ----------
    dim : int
        Dimension of the input SPD matrices (:math:`d \times d`).
    metric : {"LEM", "LCM"}, default="LEM"
        Riemannian metric for pairwise distance and Fréchet aggregation.
    power : float, default=1.0
        Power parameter for metric deformation.
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type of parameters.

    Attributes
    ----------
    bias : nn.Parameter
        Learnable SPD bias matrix :math:`B \in \mathcal{S}_{++}^d`,
        parameterized via :class:`~spd_learn.modules.SymmetricPositiveDefinite`.
        Initialized to the identity matrix.

    References
    ----------
    .. bibliography::
       :filter: key == "wang2025towards"

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import GyroSPDAttention
    >>> att = GyroSPDAttention(dim=4, metric="LEM")
    >>> X = torch.randn(2, 6, 4, 4)
    >>> X = X @ X.mT + 0.1 * torch.eye(4)
    >>> out = att(X)
    >>> out.shape
    torch.Size([2, 6, 4, 4])
    """

    def __init__(
        self,
        dim: int,
        metric: str = "LEM",
        power: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        valid_metrics = {"LEM", "LCM"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

        self.dim = dim
        self.metric = metric
        self.power = power

        self.q_proj = BiMap(
            dim,
            dim,
            parametrized=True,
            orthogonal_map="cayley",
            device=device,
            dtype=dtype,
        )
        self.k_proj = BiMap(
            dim,
            dim,
            parametrized=True,
            orthogonal_map="cayley",
            device=device,
            dtype=dtype,
        )
        self.v_proj = BiMap(
            dim,
            dim,
            parametrized=True,
            orthogonal_map="cayley",
            device=device,
            dtype=dtype,
        )

        self.bias = nn.Parameter(torch.eye(dim, device=device, dtype=dtype))
        register_parametrization(self, "bias", SymmetricPositiveDefinite())
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset layer parameters to initial state."""
        eye = torch.eye(self.dim, device=self.bias.device, dtype=self.bias.dtype)
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            proj.reset_parameters()
            if is_parametrized(proj, "weight"):
                setattr(proj, "weight", eye.unsqueeze(0))
                nn.init.kaiming_uniform_(
                    getattr(proj.parametrizations, "weight").original, a=0.01
                )
        if is_parametrized(self, "bias"):
            setattr(self, "bias", eye)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of GyroSPDAttention.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD tokens of shape `(..., n_patches, dim, dim)`.

        Returns
        -------
        torch.Tensor
            Attended SPD tokens of shape `(..., n_patches, dim, dim)`.
        """
        Q, K, V = self.q_proj(X), self.k_proj(X), self.v_proj(X)
        dist_fn = (
            log_euclidean_distance if self.metric == "LEM" else log_cholesky_distance
        )
        D = dist_fn(Q.unsqueeze(-3), K.unsqueeze(-4)) / math.sqrt(self.dim)
        A = F.softmax(1.0 / (1.0 + torch.log1p(D)), dim=-1)

        if self.metric == "LEM":
            bar_V = ensure_sym(log_euclidean_mean(A, V))
        else:
            chol_V = cholesky_log.apply(V)
            bar_V = ensure_sym(
                cholesky_exp.apply(torch.einsum("...ij, ...jkl -> ...ikl", A, chol_V))
            )

        return spd_rebiasing(bar_V, matrix_sqrt.apply(self.bias))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, metric='{self.metric}', power={self.power}"


class SymSPDAttention(nn.Module):
    r"""Symmetric Tangent Space Self-Attention Layer for SPD Matrices.

    Implements the tangent-space attention mechanism for SPD matrices.
    Uses three :class:`~spd_learn.modules.SPDLinear` projections with
    ``return_tangent=True`` to map patch SPD tokens directly into symmetric
    tangent matrices :math:`Q, K, V \in \mathcal{S}^{d_{\text{out}}}`, computes
    scaled Frobenius distance attention:

    .. math::

        A_{ij} = \operatorname{softmax}_j \left(
            -e^\alpha \lVert Q_i - K_j \rVert_F - \beta
        \right)

    aggregates features in tangent space:

    .. math::

        \bar{V}_i = \sum_j A_{ij} V_j

    and maps back to the SPD manifold via matrix exponential :math:`\exp(\bar{V}_i)`.

    Parameters
    ----------
    in_features : int
        Dimension of the input SPD matrices (:math:`d_{\text{in}}`).
    out_features : int
        Dimension of the output SPD matrices (:math:`d_{\text{out}}`).
    metric : {"LEM", "LCM", "AIM", "PEM"}, default="LEM"
        Riemannian metric for the SPDLinear projections.
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type of parameters.

    References
    ----------
    .. bibliography::
       :filter: key == "wang2025towards"

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SymSPDAttention
    >>> att = SymSPDAttention(in_features=4, out_features=3, metric="LEM")
    >>> X = torch.randn(2, 6, 4, 4)
    >>> X = X @ X.mT + 0.1 * torch.eye(4)
    >>> out = att(X)
    >>> out.shape
    torch.Size([2, 6, 3, 3])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        metric: str = "LEM",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.metric = metric

        self.q_proj = SPDLinear(
            in_features,
            out_features,
            metric=metric,
            return_tangent=True,
            device=device,
            dtype=dtype,
        )
        self.k_proj = SPDLinear(
            in_features,
            out_features,
            metric=metric,
            return_tangent=True,
            device=device,
            dtype=dtype,
        )
        self.v_proj = SPDLinear(
            in_features,
            out_features,
            metric=metric,
            return_tangent=True,
            device=device,
            dtype=dtype,
        )

        self.alpha = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SymSPDAttention.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD tokens of shape `(..., n_patches, in_features, in_features)`.

        Returns
        -------
        torch.Tensor
            Attended SPD tokens of shape `(..., n_patches, out_features, out_features)`.
        """
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        D = distance_euclid(Q.unsqueeze(-3), K.unsqueeze(-4)) / math.sqrt(
            self.out_features
        )
        logits = -torch.exp(self.alpha) * D - self.beta
        A = F.softmax(logits, dim=-1)

        out_tan = torch.einsum("...ij, ...jkl -> ...ikl", A, V)
        return matrix_exp.apply(ensure_sym(out_tan))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"metric='{self.metric}'"
        )


class GyroAtt(MAtt):
    r"""Gyrovector Riemannian Attention Network for EEG Decoding.

    Implements the full deep architecture for EEG decoding introduced in
    GyroAtt :cite:p:`wang2025towards`.

    The architecture comprises:

    1. **Spatiotemporal Frontend**: Spatial and temporal 2D convolutions
       (inherited from MAtt).
    2. **Patch Covariance Embedding**: Signal segmentation into patches via
       :class:`~spd_learn.modules.PatchEmbeddingLayer`, sample covariance estimation
       via :class:`~spd_learn.modules.CovLayer`, and trace normalization via
       :class:`~spd_learn.modules.TraceNorm`.
    3. **Riemannian Attention**: Either :class:`GyroSPDAttention` (orthogonal
       gyro-automorphisms and Fréchet aggregation) or :class:`SymSPDAttention`
       (tangent space projections via :class:`~spd_learn.modules.SPDLinear`).
    4. **Classification Backend**: Riemannian ReEig activation, LogEig tangent
       projection, dropout, and linear classification.

    Parameters
    ----------
    n_patches : int, default=6
        Number of temporal patches.
    n_chans : int, default=22
        Number of EEG channels.
    n_outputs : int, default=2
        Number of output classification classes.
    temporal_out_channels : int, default=20
        Number of output channels for the temporal convolution.
    temporal_kernel_size : int, default=25
        Kernel size for the temporal convolution along the time dimension.
    temporal_padding : int, optional
        Padding along time for the temporal convolution.
        Defaults to ``temporal_kernel_size // 2``.
    attention_in_features : int, default=20
        Input feature dimension for the attention module.
    attention_out_features : int, default=18
        Feature dimensionality of the attention output SPD matrices.
    attention_type : {"gyro", "sym"}, default="gyro"
        Type of Riemannian attention module to use.
    metric : {"LEM", "LCM"}, default="LEM"
        Riemannian metric used in the attention mechanism.
    drop_prob : float, default=0.0
        Dropout probability before the final linear classifier.
    covariance_method : callable, default=sample_covariance
        Method used for computing sample covariances.
    spatial_filters : int, optional
        Alias for ``temporal_out_channels`` and ``attention_in_features``.
    att_out_features : int, optional
        Alias for ``attention_out_features``.
    temporal_filters : int, optional
        Alias for intermediate feature dimension.

    References
    ----------
    .. bibliography::
       :filter: key == "wang2025towards"

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import GyroAtt
    >>> model = GyroAtt(n_chans=22, n_outputs=2, attention_type="gyro")
    >>> x = torch.randn(4, 22, 1000)
    >>> out = model(x)
    >>> out.shape
    torch.Size([4, 2])
    """

    def __init__(
        self,
        n_patches: int = 6,
        n_chans: int = 22,
        n_outputs: int = 2,
        temporal_out_channels: int = 20,
        temporal_kernel_size: int = 25,
        temporal_padding: Optional[int] = None,
        attention_in_features: int = 20,
        attention_out_features: int = 18,
        attention_type: str = "gyro",
        metric: str = "LEM",
        drop_prob: float = 0.0,
        covariance_method=sample_covariance,
        spatial_filters: Optional[int] = None,
        att_out_features: Optional[int] = None,
        temporal_filters: Optional[int] = None,
    ):
        if attention_type not in ("gyro", "sym"):
            raise ValueError(
                f"attention_type must be 'gyro' or 'sym', got '{attention_type}'"
            )

        if spatial_filters is not None:
            temporal_out_channels = spatial_filters
            attention_in_features = spatial_filters
        if att_out_features is not None:
            attention_out_features = att_out_features

        if temporal_padding is None:
            temporal_padding = temporal_kernel_size // 2

        super().__init__(
            n_patches=n_patches,
            n_chans=n_chans,
            n_outputs=n_outputs,
            temporal_out_channels=temporal_out_channels,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_padding,
            attention_in_features=attention_in_features,
            attention_out_features=attention_out_features,
            covariance_method=covariance_method,
        )

        self.attention_type = attention_type
        self.metric = metric
        self.drop_prob = drop_prob
        self.spatial_filters = temporal_out_channels
        self.att_out_features = attention_out_features
        self.temporal_filters = (
            temporal_filters if temporal_filters is not None else n_chans
        )
        self.patch_cov_layer[2] = TraceNorm(epsilon=1e-4)

        # Attention block
        self.proj: nn.Module
        self.attention: nn.Module
        if attention_type == "gyro":
            self.proj = (
                BiMap(temporal_out_channels, attention_out_features)
                if temporal_out_channels != attention_out_features
                else nn.Identity()
            )
            self.attention = GyroSPDAttention(dim=attention_out_features, metric=metric)
            self.manifold_attention = nn.Sequential(  # type: ignore[assignment]
                self.proj, self.attention
            )
        else:  # sym
            self.proj = nn.Identity()
            self.attention = SymSPDAttention(
                in_features=temporal_out_channels,
                out_features=attention_out_features,
                metric=metric,
            )
            self.manifold_attention = self.attention  # type: ignore[assignment]

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of GyroAtt.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch, n_chans, n_times)` or precomputed
            covariance patches of shape
            `(batch, n_patches, spatial_filters, spatial_filters)`.

        Returns
        -------
        torch.Tensor
            Logits of shape `(batch, n_outputs)`.
        """
        if x.ndim >= 4 and x.shape[-1] == x.shape[-2]:
            x = self.patch_cov_layer[2](x)
        else:
            if x.ndim == 3:
                x = self.add_extra_dim(x)
            x = self.spatial_filter(x)
            x = self.spatial_batch_norm(x)
            x = self.temporal_feature_extractor(x)
            x = self.temporal_batch_norm(x)
            x = self.squeeze_channel(x)
            x = self.patch_cov_layer(x)

        x = self.manifold_attention(x)
        x = self.re_eig(x)
        x = self.tangent(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.linear(x)
