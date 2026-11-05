# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from spd_learn.modules import (
    CovLayer,
    LogEig,
    PatchEmbeddingLayer,
    SPDBatchNormMean,
    SPDTransformerEncoderLayer,
    Shrinkage,
    TraceNorm,
)


class SPDTransNet(nn.Module):
    r"""Structure-Preserving Transformer Network for SPD Sequences (SPDTransNet).

    This class implements the SPDTransNet model :cite:p:`seraphim2023structure`,
    a Transformer architecture designed for sequences of Symmetric Positive
    Definite (SPD) covariance matrices. The network preserves Riemannian
    manifold geometry through structure-preserving multihead attention (SP-MHA).

    The model consists of the following processing stages:

    1. **Temporal-Spatial Convolution Frontend:** Raw multivariate EEG time
       series are filtered across time and space to extract :math:`n_{\text{filters}}`
       feature channels, which are subsequently segmented into :math:`n_{\text{patches}}`
       temporal patches.
    2. **Per-Patch Regularized Covariance Pooling:** A sample covariance matrix is
       estimated for each patch via :class:`~spd_learn.modules.CovLayer`,
       normalized by its trace via :class:`~spd_learn.modules.TraceNorm`, and
       regularized by learnable shrinkage via :class:`~spd_learn.modules.Shrinkage`.
    3. **Riemannian Whitening and Tangent Mapping:** Per-patch SPD matrices are
       whitened across the batch using Riemannian Batch Normalization
       (:class:`~spd_learn.modules.SPDBatchNormMean`) and projected into the
       identity tangent space via :class:`~spd_learn.modules.LogEig` with upper-triangular
       isometric vectorization (:class:`~spd_learn.modules.Vech`), yielding token
       embeddings of dimension :math:`d_{\text{embed}} = n_{\text{filters}} (n_{\text{filters}} + 1) / 2`.
    4. **Positional Encoding:** Learnable positional embeddings are added to each
       patch token in the sequence.
    5. **Structure-Preserving Transformer Encoder:** A stack of
       :class:`~spd_learn.modules.SPDTransformerEncoderLayer` modules processes
       the patch sequence, computing attention maps that form affine combinations
       of tangent tokens without distorting geometric structure.
    6. **Classification Head:** Features across all patches are flattened and
       mapped to class logits via a linear layer.

    Parameters
    ----------
    n_chans : int, default=22
        Number of input channels (e.g. EEG electrodes).
    n_outputs : int, default=2
        Number of output classes for classification.
    n_patches : int, default=8
        Number of temporal patches along the time dimension.
    n_filters : int, default=16
        Number of spatial-temporal filters extracted by the convolutional frontend.
    num_layers : int, default=2
        Number of :class:`~spd_learn.modules.SPDTransformerEncoderLayer` layers.
    num_heads : int, default=4
        Number of attention heads in each encoder layer.
    dim_feedforward : int, optional
        Hidden dimension of the feedforward network in each encoder layer.
        Defaults to ``4 * embed_dim`` when ``None``.
    drop_prob : float, default=0.1
        Dropout probability applied in positional encoding, attention, and FFNs.
    temporal_kernel_size : int, default=25
        Kernel length of the 1D temporal convolution.

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SPDTransNet
    >>> model = SPDTransNet(n_chans=22, n_outputs=2)
    >>> X = torch.randn(2, 22, 1000)
    >>> out = model(X)
    >>> out.shape
    torch.Size([2, 2])
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        n_patches: int = 8,
        n_filters: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: Optional[int] = None,
        drop_prob: float = 0.1,
        temporal_kernel_size: int = 25,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_patches = n_patches
        self.n_filters = n_filters
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.temporal_kernel_size = temporal_kernel_size

        self.embed_dim = n_filters * (n_filters + 1) // 2

        # Stage 1: Temporal-spatial convolution frontend
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(1, temporal_kernel_size),
            padding="same",
        )
        self.temporal_bn = nn.BatchNorm2d(n_filters)
        self.spatial_conv = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=(n_chans, 1),
        )
        self.spatial_bn = nn.BatchNorm2d(n_filters)
        self.patch_embedding = PatchEmbeddingLayer(
            n_chans=n_filters,
            n_patches=n_patches,
        )

        # Stage 2: Per-patch regularized SPD covariance
        self.cov_layer = CovLayer()
        self.trace_norm = TraceNorm()
        self.shrinkage = Shrinkage(
            n_chans=n_filters,
            init_shrinkage=-2.0,
            learnable=True,
        )

        # Stage 3: Whitening and Tangent Space Mapping
        self.rbn = SPDBatchNormMean(n_filters)
        self.logeig = LogEig(upper=True)

        # Stage 4: Positional Embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, n_patches, self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.pos_dropout = nn.Dropout(drop_prob)

        # Stage 5: Stack of SPD Transformer Encoder Layers
        self.layers = nn.ModuleList([
            SPDTransformerEncoderLayer(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=drop_prob,
            )
            for _ in range(num_layers)
        ])

        # Stage 6: Linear classification head
        self.classifier = nn.Linear(n_patches * self.embed_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the SPDTransNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch, n_chans, n_times)` or
            `(batch, 1, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape `(batch, n_outputs)`.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Stage 1: Conv frontend
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = x.squeeze(2)

        x = self.patch_embedding(x)

        # Stage 2: Covariance + Regularization
        x = self.cov_layer(x)
        x = self.trace_norm(x)
        x = self.shrinkage(x)

        # Stage 3: Riemannian BatchNorm + LogEig
        batch_size, n_patches, n_filters, _ = x.shape
        x_reshaped = x.reshape(batch_size * n_patches, n_filters, n_filters)
        x_bn = self.rbn(x_reshaped)
        x_log = self.logeig(x_bn)
        x = x_log.reshape(batch_size, n_patches, self.embed_dim)

        # Stage 4: Positional embedding
        x = self.pos_dropout(x + self.pos_embedding)

        # Stage 5: Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Stage 6: Classifier
        x = x.flatten(start_dim=1)
        out = self.classifier(x)
        return out
