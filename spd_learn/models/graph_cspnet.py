# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Optional, Tuple

import torch
import torch.nn as nn

from einops import rearrange

from spd_learn.modules import (
    CovLayer,
    GraphBiMap,
    LogEig,
    ReEig,
    SPDBatchNormMean,
    Shrinkage,
    TraceNorm,
)


class GraphCSPNet(nn.Module):
    r"""Graph-CSPNet: Graph Neural Networks on SPD Manifolds for Motor Imagery.

    This class implements the GraphCSPNet model :cite:p:`ju2023graph`, a geometric
    deep learning framework on Symmetric Positive Definite (SPD) manifolds that
    integrates time-frequency spatial representations via graph neural networks.

    The architecture consists of:

    1. **Time-Frequency Representation**: 3D raw EEG `(batch, n_chans, n_times)`
       is transformed into multi-frequency bands via a 2D temporal convolution
       ``nn.Conv2d(1, n_freq, kernel_size=(1, 25), padding=(0, 12), bias=False)``
       producing `(batch, n_freq, n_chans, n_times)`, or 4D filter-bank input
       is used directly.
    2. **Temporal Windowing**: The time dimension is partitioned into `n_temp`
       windows, yielding `n_nodes = n_freq * n_temp` nodes.
    3. **Node-wise SPD Covariance**: Spatial covariance matrices are estimated
       via :class:`~spd_learn.modules.CovLayer`, normalized via
       :class:`~spd_learn.modules.TraceNorm`, and regularized via
       :class:`~spd_learn.modules.Shrinkage`.
    4. **Graph Time-Frequency Lattice**: A 2D lattice connects neighboring time
       and frequency slices with a row-stochastic initial propagation matrix
       :math:`P_{\text{init}}`.
    5. **Graph Bilinear Mapping Blocks**: Consecutive :class:`~spd_learn.modules.GraphBiMap`
       + :class:`~spd_learn.modules.ReEig` blocks aggregate information across
       graph neighbors and reduce spatial dimensions. The first block operates
       on the graph lattice (optionally learnable), subsequent blocks use
       identity aggregation, and the final block applies
       :class:`~spd_learn.modules.SPDBatchNormMean` + :class:`~spd_learn.modules.ReEig`.
    6. **Tangent Space Classification**: SPD matrices are mapped to the tangent
       space via :class:`~spd_learn.modules.LogEig` (upper triangular vectorization),
       flattened, passed through dropout, and classified with a linear layer.

    Parameters
    ----------
    n_chans : int, default=22
        Number of EEG channels.
    n_outputs : int, default=2
        Number of output classes.
    n_temp : int, default=3
        Number of temporal segments (windows).
    n_freq : int, default=3
        Number of frequency bands.
    bimap_dims : tuple[int, ...], default=(36, 20, 22)
        Output dimensions for the sequence of BiMap transformations.
    learnable_graph : bool, default=True
        Whether the first GraphBiMap layer learns graph aggregation logits.
    drop_prob : float, default=0.0
        Dropout probability before the final linear classifier.
    momentum : float, default=0.1
        Momentum for Riemannian batch normalization (SPDBatchNormMean).

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import GraphCSPNet
    >>> model = GraphCSPNet(n_chans=22, n_outputs=2)
    >>> # 3D input: (batch, n_chans, n_times)
    >>> x = torch.randn(2, 22, 1000)
    >>> out = model(x)
    >>> out.shape
    torch.Size([2, 2])
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        n_temp: int = 3,
        n_freq: int = 3,
        bimap_dims: Tuple[int, ...] = (36, 20, 22),
        learnable_graph: bool = True,
        drop_prob: float = 0.0,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_temp = n_temp
        self.n_freq = n_freq
        self.bimap_dims = tuple(bimap_dims)
        self.learnable_graph = learnable_graph
        self.drop_prob = drop_prob
        self.momentum = momentum

        n_nodes = n_freq * n_temp
        self.n_nodes = n_nodes

        # Temporal convolution for 3D input
        self.temporal_conv = nn.Conv2d(
            1, n_freq, kernel_size=(1, 25), padding=(0, 12), bias=False
        )

        # Covariance and regularization pipeline
        self.cov = CovLayer()
        self.trace_norm = TraceNorm()
        self.shrinkage = Shrinkage(n_chans=n_chans, init_shrinkage=-2.0)

        # 2D time-frequency lattice adjacency matrix
        adj = torch.zeros(n_nodes, n_nodes)
        for f in range(n_freq):
            for t in range(n_temp):
                i = f * n_temp + t
                adj[i, i] = 1.0  # self-loop
                for df, dt in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nf, nt = f + df, t + dt
                    if 0 <= nf < n_freq and 0 <= nt < n_temp:
                        j = nf * n_temp + nt
                        adj[i, j] = 1.0
        P_init = adj / adj.sum(dim=-1, keepdim=True)

        all_dims = (n_chans, *self.bimap_dims)
        n_blocks = len(self.bimap_dims)

        initial_blocks = []
        for idx in range(n_blocks - 1):
            in_dim = all_dims[idx]
            out_dim = all_dims[idx + 1]
            g_mat = P_init if idx == 0 else None
            learnable = learnable_graph if idx == 0 else False
            initial_blocks.append(
                nn.Sequential(
                    GraphBiMap(
                        n_nodes=n_nodes,
                        in_features=in_dim,
                        out_features=out_dim,
                        graph_matrix=g_mat,
                        learnable_graph=learnable,
                    ),
                    ReEig(),
                )
            )
        self.initial_blocks = nn.ModuleList(initial_blocks)

        final_in = all_dims[-2]
        final_out = all_dims[-1]
        final_g_mat = P_init if n_blocks == 1 else None
        final_learnable = learnable_graph if n_blocks == 1 else False

        self.final_bimap = GraphBiMap(
            n_nodes=n_nodes,
            in_features=final_in,
            out_features=final_out,
            graph_matrix=final_g_mat,
            learnable_graph=final_learnable,
        )
        self.final_bn = SPDBatchNormMean(num_features=final_out, momentum=momentum)
        self.final_reeig = ReEig()

        self.log_eig = LogEig(upper=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(drop_prob)
        feat_dim = n_nodes * final_out * (final_out + 1) // 2
        self.fc = nn.Linear(feat_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GraphCSPNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG data of shape `(batch_size, n_chans, n_times)` (3D raw EEG)
            or `(batch_size, n_freq, n_chans, n_times)` (4D filter-bank EEG).

        Returns
        -------
        torch.Tensor
            Class logits of shape `(batch_size, n_outputs)`.
        """
        if x.ndim == 3:
            # 3D raw EEG -> apply multi-band temporal convolution
            x = self.temporal_conv(x.unsqueeze(1))
        elif x.ndim == 4:
            # 4D filter bank EEG
            if x.shape[1] != self.n_freq:
                raise ValueError(
                    f"Expected 4D input with n_freq={self.n_freq}, got {x.shape[1]}"
                )
        else:
            raise ValueError(
                f"Expected 3D or 4D input tensor, got shape {x.shape}"
            )

        window_len = x.shape[-1] // self.n_temp
        x_trimmed = x[..., : self.n_temp * window_len]
        segments = rearrange(
            x_trimmed,
            "b f c (t w) -> b (f t) c w",
            t=self.n_temp,
            w=window_len,
        )

        x_cov = self.cov(segments)
        x_cov = self.trace_norm(x_cov)
        x_cov = self.shrinkage(x_cov)

        for block in self.initial_blocks:
            x_cov = block(x_cov)

        x_cov = self.final_bimap(x_cov)
        b, n, d, _ = x_cov.shape
        x_flat = rearrange(x_cov, "b n d1 d2 -> (b n) d1 d2")
        x_norm = self.final_bn(x_flat)
        x_cov = rearrange(x_norm, "(b n) d1 d2 -> b n d1 d2", b=b, n=n)
        x_cov = self.final_reeig(x_cov)

        x_tangent = self.log_eig(x_cov)
        x_feat = self.flatten(x_tangent)
        x_drop = self.dropout(x_feat)
        return self.fc(x_drop)
