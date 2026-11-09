# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""SPD-MetaFormer for small-data brain decoding.

This module implements the SPD-MetaFormer architecture and its building blocks
as proposed in :cite:p:`wang2026spdmetaformer`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from spd_learn.functional import (
    geodesic_euclid,
    matrix_exp,
    matrix_log,
    mean_euclid,
    sample_covariance,
)
from spd_learn.models.matt import MAtt
from spd_learn.modules import (
    BiMap,
    ReEig,
    Shrinkage,
    Vech,
)


class SPDMetaFormer(MAtt):
    r"""SPD-MetaFormer for Small-Data Brain Decoding.

    This class implements the SPD-MetaFormer model :cite:p:`wang2026spdmetaformer`,
    an attention-free Riemannian architecture designed for small-data brain
    decoding from electroencephalography (EEG) signals.

    In standard manifold attention models such as MAtt :cite:p:`pan2022matt`,
    the bounded similarity metric :math:`\frac{1}{1 + \log(1 + d)}` causes softmax
    attention weights over short EEG patch sequences to remain nearly uniform,
    while query/key bilinear projections over-parameterize low-data regimes.
    SPD-MetaFormer replaces learned attention with an attention-free Fréchet token
    mixer and shared spectral feedforward maps equipped with Log-Euclidean
    geodesic skip connections.

    The model consists of three stages:

    1. **Frontend & Patch Covariance Extraction:**
       Raw EEG signals are processed by spatial and temporal 2D convolutions
       (inherited from MAtt), segmented into :math:`P` temporal patches via
       :class:`~spd_learn.modules.PatchEmbeddingLayer`, and mapped to SPD
       covariance matrices regularized by :class:`~spd_learn.modules.TraceNorm`
       and learnable :class:`~spd_learn.modules.Shrinkage`.
    2. **SPD-MetaFormer Backbone:**
       A stack of :math:`\text{depth}` :class:`SPDMetaFormerBlock` layers that perform
       attention-free token mixing via uniform Fréchet averaging and channel mixing
       via a shared SPD spectral feedforward network (SPD-FFN).
    3. **Learned Geometric Readout:**
       Patch tokens are aggregated via a learnable softmax-weighted Log-Euclidean
       tangent mean :math:`\bar{L}_{\text{readout}} = \sum_{p=1}^P w_p \log(Y_p)`,
       vectorized via :class:`~spd_learn.modules.Vech`, and classified using a
       linear projection with dropout.

    Parameters
    ----------
    n_patches : int, default=6
        Number of temporal patches extracted from the signal.
    n_chans : int, default=22
        Number of input EEG channels (electrodes).
    n_outputs : int, default=2
        Number of output classes for classification.
    depth : int, default=2
        Number of stacked :class:`SPDMetaFormerBlock` layers.
    temporal_out_channels : int, default=20
        Number of temporal convolution filters.
    temporal_kernel_size : int, default=25
        Kernel length of the temporal convolution layer.
    temporal_padding : int, optional
        Padding along time for temporal convolution.
    attention_in_features : int, default=20
        Input feature dimension for SPD tokens.
    attention_out_features : int, default=20
        Output feature dimension for SPD tokens.
    ffn_dim : int, optional
        Hidden dimension of the SPD feedforward network within each block.
    eps : float, default=1e-4
        Eigenvalue rectification threshold for :class:`~spd_learn.modules.ReEig`.
    drop_prob : float, default=0.0
        Dropout probability applied before the final classification head.
    covariance_method : callable, default=sample_covariance
        Covariance estimation method.
    init_shrinkage : float, default=-2.0
        Initial shrinkage pre-sigmoid logit for covariance regularization.
    pooling : {"mean", "concat"}, default="mean"
        Token readout pooling method in the tangent space.
    spatial_filters : int, optional
        Alias for ``temporal_out_channels``.
    temporal_filters : int, optional
        Alias for intermediate feature dimension.
    temp_kernel_length : int, optional
        Alias for ``temporal_kernel_size``.
    device : torch.device, optional
        Target device for layer parameters and buffers.
    dtype : torch.dtype, optional
        Data type for layer parameters and buffers.

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    See Also
    --------
    :class:`SPDMetaFormerBlock` : Attention-free SPD MetaFormer building block.
    :class:`~spd_learn.models.MAtt` : Manifold attention network for EEG decoding.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SPDMetaFormer
    >>> model = SPDMetaFormer(n_chans=22, n_outputs=2, n_patches=6, depth=2)
    >>> x = torch.randn(2, 22, 1000)
    >>> out = model(x)
    >>> out.shape
    torch.Size([2, 2])
    """

    def __init__(
        self,
        n_patches: int = 6,
        n_chans: int = 22,
        n_outputs: int = 2,
        depth: int = 2,
        temporal_out_channels: int = 20,
        temporal_kernel_size: int = 25,
        temporal_padding: Optional[int] = None,
        attention_in_features: int = 20,
        attention_out_features: int = 20,
        ffn_dim: Optional[int] = None,
        eps: float = 1e-4,
        drop_prob: float = 0.0,
        covariance_method=sample_covariance,
        init_shrinkage: float = -2.0,
        pooling: str = "mean",
        spatial_filters: Optional[int] = None,
        temporal_filters: Optional[int] = None,
        temp_kernel_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if spatial_filters is not None:
            temporal_out_channels = spatial_filters
            attention_in_features = spatial_filters
            attention_out_features = spatial_filters

        if temp_kernel_length is not None:
            temporal_kernel_size = temp_kernel_length

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

        self.depth = depth
        self.ffn_dim = ffn_dim if ffn_dim is not None else temporal_out_channels
        self.eps = eps
        self.drop_prob = drop_prob
        self.temp_kernel_length = temporal_kernel_size
        self.init_shrinkage = init_shrinkage
        self.pooling = pooling
        self.spatial_filters = temporal_out_channels
        self.temporal_filters = (
            temporal_filters if temporal_filters is not None else n_chans
        )

        param_dtype = dtype if dtype is not None else torch.float32

        # Regularized shrinkage for patch covariances
        self.shrinkage = Shrinkage(
            n_chans=temporal_out_channels,
            init_shrinkage=init_shrinkage,
            learnable=True,
            device=device,
            dtype=dtype,
        )

        # MetaFormer blocks
        self.blocks = nn.ModuleList(
            [
                SPDMetaFormerBlock(
                    dim=temporal_out_channels,
                    ffn_dim=self.ffn_dim,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )
        self.manifold_attention = (
            nn.Sequential(*self.blocks) if depth > 0 else nn.Identity()  # type: ignore[assignment]
        )

        # Learned Geometric Readout
        self.readout_logits = nn.Parameter(
            torch.zeros(n_patches, device=device, dtype=param_dtype)
        )
        self.vech = Vech(device=device, dtype=dtype)
        self.dropout = nn.Dropout(p=drop_prob)

        tangent_dim = int(temporal_out_channels * (temporal_out_channels + 1) // 2)
        if pooling == "concat":
            linear_in_features = n_patches * tangent_dim
        else:
            linear_in_features = tangent_dim

        self.linear = nn.Linear(
            in_features=linear_in_features,
            out_features=n_outputs,
            device=device,
            dtype=dtype,
        )

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPD-MetaFormer model.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG tensor of shape `(batch, n_chans, n_times)` or
            `(batch, 1, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output class logits of shape `(batch, n_outputs)`.
        """
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        x = self.add_extra_dim(x)
        x = self.spatial_filter(x)
        x = self.spatial_batch_norm(x)
        x = self.temporal_feature_extractor(x)
        x = self.temporal_batch_norm(x)
        x = self.squeeze_channel(x)

        x = self.patch_cov_layer(x)
        x = self.shrinkage(x)
        x = self.manifold_attention(x)

        log_tokens = matrix_log.apply(x)

        if self.pooling == "concat":
            feat = self.vech(log_tokens).flatten(start_dim=-2)
        else:
            w = torch.softmax(self.readout_logits, dim=0)
            l_readout = torch.einsum("p, ...pij -> ...ij", w, log_tokens)
            feat = self.vech(l_readout)

        feat = self.dropout(feat)
        return self.linear(feat)


class SPDMetaFormerBlock(nn.Module):
    r"""SPD MetaFormer Block.

    This module implements an attention-free MetaFormer block operating on
    sequences of Symmetric Positive Definite (SPD) patch tokens
    :cite:p:`wang2026spdmetaformer`.

    The block replaces computationally heavy and over-parameterized manifold
    attention mechanisms with two sub-blocks:

    1. **Uniform Fréchet Token Mixer with Geodesic Residual:**
       Tokens are mapped to the Log-Euclidean tangent space via matrix logarithm
       :math:`L = \log(X)`. The uniform Log-Euclidean Fréchet mean across patches
       is computed as :math:`\bar{L} = \frac{1}{P} \sum_{p=1}^P L_p`. Tokens are
       updated via learnable geodesic interpolation with rate
       :math:`\alpha = \sigma(\text{mix\_logit})`:

       .. math::
           U = \exp((1 - \alpha) L + \alpha \bar{L})

    2. **Shared Spectral Feedforward Map (SPD-FFN) with Geodesic Residual:**
       A shared 2-layer SPD spectral feedforward network processes each token:

       .. math::
           \text{FFN}(U) = W_2^\top \text{ReEig}(W_1^\top U W_1) W_2

       where :math:`W_1 \in \text{St}(d, d_{\text{ffn}})` and
       :math:`W_2 \in \text{St}(d_{\text{ffn}}, d)` are (semi-)orthogonal bilinear
       mappings. The output is combined with :math:`U` via a second learnable
       Log-Euclidean geodesic skip connection with rate
       :math:`\beta = \sigma(\text{ffn\_logit})`:

       .. math::
           Y = \exp((1 - \beta) \log(U) + \beta \log(\text{FFN}(U)))

    Parameters
    ----------
    dim : int
        Dimensionality :math:`d` of the input SPD matrices.
    ffn_dim : int, optional
        Hidden dimensionality :math:`d_{\text{ffn}}` of the spectral feedforward
        network. If None, defaults to `dim`.
    eps : float, default=1e-4
        Eigenvalue rectification threshold for :class:`~spd_learn.modules.ReEig`.
    device : torch.device, optional
        Target device for layer parameters.
    dtype : torch.dtype, optional
        Data type for layer parameters.

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    See Also
    --------
    :class:`SPDMetaFormer` : Full SPD-MetaFormer model for EEG decoding.
    :class:`~spd_learn.modules.BiMap` : Bilinear mapping layer for SPD matrices.
    :class:`~spd_learn.modules.ReEig` : Rectified eigenvalue layer.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SPDMetaFormerBlock
    >>> block = SPDMetaFormerBlock(dim=8, ffn_dim=12)
    >>> A = torch.randn(2, 6, 8, 8)
    >>> X = A @ A.mT + 0.1 * torch.eye(8)  # SPD tokens
    >>> Y = block(X)
    >>> Y.shape
    torch.Size([2, 6, 8, 8])
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: Optional[int] = None,
        eps: float = 1e-4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else dim
        self.eps = eps
        self.device = device
        self.dtype = dtype

        param_dtype = dtype if dtype is not None else torch.float32

        # Sub-block 1: Learnable geodesic mixing rate for uniform Fréchet mean
        self.mix_logit = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=param_dtype)
        )

        # Sub-block 2: Shared SPD spectral feedforward network (SPD-FFN)
        self.bimap1 = BiMap(dim, self.ffn_dim, device=device, dtype=dtype)
        self.reeig = ReEig(threshold=eps, device=device, dtype=dtype)
        self.bimap2 = BiMap(self.ffn_dim, dim, device=device, dtype=dtype)

        # Sub-block 2: Learnable geodesic residual rate for FFN
        self.ffn_logit = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=param_dtype)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Forward pass through the SPD MetaFormer block.

        Parameters
        ----------
        X : torch.Tensor
            Sequence of patch SPD tokens of shape `(..., P, dim, dim)`.

        Returns
        -------
        torch.Tensor
            Transformed SPD tokens of shape `(..., P, dim, dim)`.
        """
        # Sub-block 1 — Uniform Fréchet Token Mixer with Geodesic Residual
        L = matrix_log.apply(X)
        L_bar = mean_euclid(L).unsqueeze(-3).expand_as(L)
        alpha = torch.sigmoid(self.mix_logit).expand(L.shape[:-2])
        U = matrix_exp.apply(geodesic_euclid(L, L_bar, alpha=alpha))

        # Sub-block 2 — Shared Spectral Feedforward Map (SPD-FFN) with Geodesic Residual
        ffn_out = self.bimap2(self.reeig(self.bimap1(U)))
        beta = torch.sigmoid(self.ffn_logit).expand(U.shape[:-2])
        return matrix_exp.apply(
            geodesic_euclid(matrix_log.apply(U), matrix_log.apply(ffn_out), alpha=beta)
        )
