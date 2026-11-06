# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import torch.nn as nn

from ..functional import matrix_exp, matrix_log
from ..modules import (
    BiMap,
    CovLayer,
    LogEig,
    PatchEmbeddingLayer,
    ReEig,
    Shrinkage,
    TraceNorm,
)


class TangentGraphSPDAggregation(nn.Module):
    r"""Tangent-Space :math:`k`-NN Graph-SPD Attention Aggregation.

    This module performs geometric graph aggregation over a set of SPD node
    representations :cite:p:`tnsre2026ssgcnet`.

    1. **Tangent Space Mapping**: Node covariance matrices
       :math:`S \in \mathbb{R}^{B \times V \times d \times d}` are mapped into
       the Log-Euclidean tangent space:

       .. math::

           L_u = \log(S_u) \in \mathcal{S}^d

    2. **Pairwise Tangent Distance**: Computes pairwise Frobenius distances
       in the tangent space:

       .. math::

           D_{uv} = \|L_u - L_v\|_F = \sqrt{\sum_{i,j} (L_{u,ij} - L_{v,ij})^2}

    3. **Top-:math:`k` Nearest-Neighbor Masking**: For each node :math:`u`,
       identifies its :math:`k` nearest neighbors :math:`\mathcal{N}_k(u)`
       corresponding to the smallest distances :math:`D_{uv}`. Non-neighbor
       connections are masked with :math:`-\infty`.

    4. **Attention Weights**: Computes softmax attention scores over the neighbors:

       .. math::

           A_{uv} = \frac{\exp(-D_{uv})}{\sum_{w \in \mathcal{N}_k(u)} \exp(-D_{uw})},
           \quad v \in \mathcal{N}_k(u)

    5. **Tangent Aggregation and Manifold Retraction**: Combines the node's
       tangent vector with the attention-weighted neighbor tangent vectors,
       then projects back to the SPD manifold via matrix exponential:

       .. math::

           \tilde{L}_u = 0.5 L_u + 0.5 \sum_{v \in \mathcal{N}_k(u)} A_{uv} L_v

       .. math::

           \tilde{S}_u = \exp(\tilde{L}_u) \in \mathcal{S}_{++}^d

    Parameters
    ----------
    k_neighbors : int, default=2
        Number of nearest neighbors to aggregate for each node.
    eps : float, default=1e-4
        Numerical stability constant for distance computation.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models.ssgcnet import TangentGraphSPDAggregation
    >>> agg = TangentGraphSPDAggregation(k_neighbors=2)
    >>> S = torch.eye(6).expand(2, 4, 6, 6).clone()
    >>> S_out = agg(S)
    >>> S_out.shape
    torch.Size([2, 4, 6, 6])
    """

    def __init__(self, k_neighbors: int = 2, eps: float = 1e-4) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.eps = eps

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """Forward pass of the tangent-space graph-SPD aggregation.

        Parameters
        ----------
        S : torch.Tensor
            Input node SPD matrices of shape `(..., V, d, d)`.

        Returns
        -------
        torch.Tensor
            Aggregated node SPD matrices of shape `(..., V, d, d)`.
        """
        V = S.shape[-3]
        L = matrix_log.apply(S)

        # Compute pairwise difference and Frobenius norm
        diff = L.unsqueeze(-3) - L.unsqueeze(-4)  # (..., V, V, d, d)
        D = torch.sqrt(torch.sum(diff**2, dim=(-2, -1)) + 1e-8)  # (..., V, V)

        # Select top-k nearest neighbors per node (smallest distance -> largest -D)
        k = min(self.k_neighbors, V)
        _, topk_idx = torch.topk(-D, k=k, dim=-1)

        mask = torch.full_like(D, float("-inf"))
        mask.scatter_(-1, topk_idx, 0.0)
        scores = -D + mask
        A = torch.softmax(scores, dim=-1)  # (..., V, V)

        # Aggregate neighbor tangent matrices: sum_v A_uv L_v
        L_flat = L.flatten(start_dim=-2)
        L_neighbors_flat = torch.matmul(A, L_flat)
        L_neighbors = L_neighbors_flat.view_as(L)

        L_tilde = 0.5 * L + 0.5 * L_neighbors
        return matrix_exp.apply(L_tilde)


class SSGCNet(nn.Module):
    r"""Stiefel-SPD Manifold Graph Convolutional Network (SSGCNet).

    This class implements the SSGCNet model proposed by Tibermacine, Russo,
    and Napoli :cite:p:`tnsre2026ssgcnet`. SSGCNet provides an end-to-end
    geometry-consistent architecture for EEG decoding that connects depthwise
    temporal-spatial convolutions, Stiefel manifold projections, and tangent-space
    :math:`k`-NN graph attention.

    Architecture Stages
    -------------------
    1. **Temporal & Spatial Convolutions**:
       - 2D Temporal Conv: `Conv2d(1, temporal_filters, (1, 15), padding=(0, 7))`
       - Depthwise Spatial Conv:
         `Conv2d(temporal_filters, temporal_filters, (n_chans, 1), groups=...)`
    2. **Patch Covariance Embedding**:
       - :class:`~spd_learn.modules.PatchEmbeddingLayer`: splits temporal features
         into `n_nodes` patches.
       - :class:`~spd_learn.modules.CovLayer`: computes sample covariance per patch.
       - :class:`~spd_learn.modules.TraceNorm`: normalizes covariance scale.
       - :class:`~spd_learn.modules.Shrinkage`: regularizes conditioning.
    3. **Stiefel Projection**:
       - :class:`~spd_learn.modules.BiMap`: learnable orthonormal projection on
         :math:`\text{St}(\text{temporal\_filters}, \text{stiefel\_dim})`.
       - :class:`~spd_learn.modules.ReEig`: eigenvalue floor enforcement.
    4. **Tangent-Space :math:`k`-NN Graph-SPD Aggregation**:
       - Maps node covariances to Log-Euclidean tangent space.
       - Computes pairwise tangent distances and dynamically selects top-:math:`k`
         nearest neighbors.
       - Aggregates neighbor tangent representations via attention and retracts
         back to the SPD manifold via :func:`~spd_learn.functional.matrix_exp`.
    5. **Classification**:
       - :class:`~spd_learn.modules.LogEig`: vectorizes upper triangular elements.
       - Flattening, dropout, and a linear classifier.

    Parameters
    ----------
    n_chans : int, default=22
        Number of EEG electrode channels.
    n_outputs : int, default=2
        Number of output classes.
    n_nodes : int, default=4
        Number of temporal graph nodes (patches).
    temporal_filters : int, default=16
        Number of temporal convolutional filters.
    stiefel_dim : int, default=12
        Dimension of the reduced SPD matrices after Stiefel projection.
    k_neighbors : int, default=2
        Number of nearest neighbors aggregated in the tangent-space graph.
    eps : float, default=1e-4
        Eigenvalue threshold and regularizer.
    drop_prob : float, default=0.2
        Dropout probability before the final classification layer.

    References
    ----------
    .. bibliography::
       :filter: docname in %user

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SSGCNet
    >>> model = SSGCNet(n_chans=22, n_outputs=2)
    >>> x = torch.randn(4, 22, 1000)
    >>> out = model(x)
    >>> out.shape
    torch.Size([4, 2])
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        n_nodes: int = 4,
        temporal_filters: int = 16,
        stiefel_dim: int = 12,
        k_neighbors: int = 2,
        eps: float = 1e-4,
        drop_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.temporal_filters = temporal_filters
        self.stiefel_dim = stiefel_dim
        self.k_neighbors = k_neighbors
        self.eps = eps
        self.drop_prob = drop_prob

        # Temporal & spatial convolutional frontend
        self.temp_conv = nn.Conv2d(
            in_channels=1,
            out_channels=temporal_filters,
            kernel_size=(1, 15),
            padding=(0, 7),
            bias=False,
        )
        self.spat_conv = nn.Conv2d(
            in_channels=temporal_filters,
            out_channels=temporal_filters,
            kernel_size=(n_chans, 1),
            groups=temporal_filters,
            bias=False,
        )

        # Patch embedding and covariance frontend
        self.patch_embedding = PatchEmbeddingLayer(
            n_chans=temporal_filters,
            n_patches=n_nodes,
        )
        self.cov = CovLayer()
        self.trace_norm = TraceNorm()
        self.shrinkage = Shrinkage(n_chans=temporal_filters)

        # Learnable Stiefel projection
        self.bimap = BiMap(temporal_filters, stiefel_dim)
        self.reeig = ReEig(threshold=eps)

        # Tangent-Space k-NN Graph-SPD Attention Aggregation
        self.graph_agg = TangentGraphSPDAggregation(
            k_neighbors=k_neighbors,
            eps=eps,
        )

        # Classification Head
        self.logeig = LogEig(upper=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=drop_prob)
        self.feat_dim = n_nodes * stiefel_dim * (stiefel_dim + 1) // 2
        self.classifier = nn.Linear(self.feat_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SSGCNet.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG tensor of shape `(batch, n_chans, n_times)` or
            `(batch, 1, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output class logits with shape `(batch, n_outputs)`.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        h = self.temp_conv(x)
        h = self.spat_conv(h)
        h = h[:, :, 0, :]

        patches = self.patch_embedding(h)
        covs = self.cov(patches)
        covs = self.trace_norm(covs)
        covs = self.shrinkage(covs)

        stiefel_covs = self.reeig(self.bimap(covs))
        agg_covs = self.graph_agg(stiefel_covs)

        feats = self.logeig(agg_covs)
        feats = self.flatten(feats)
        feats = self.dropout(feats)
        return self.classifier(feats)
