# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Riemannian Networks over Full-Rank Correlation Matrices (CorNet).

Implements the CorNet model :cite:p:`chen2026cornet`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from spd_learn.functional import (
    cov_to_corr,
    log_map_lsm,
    log_map_olm,
    sample_covariance,
    tril_embed,
)
from spd_learn.modules import BiMap, CovLayer, ReEig, Shrinkage


class CorNet(nn.Module):
    r"""Riemannian Neural Network on the Correlation Manifold (CorNet).

    This class implements the CorNet model :cite:p:`chen2026cornet`, an architecture
    operating directly on the manifold of full-rank correlation matrices
    :math:`\mathcal{C}_{++}^n`.

    The model consists of:
    1. Sample covariance estimation (:class:`~spd_learn.modules.CovLayer`) and
       shrinkage regularization (:class:`~spd_learn.modules.Shrinkage`).
    2. Conversion to correlation matrices via :func:`~spd_learn.functional.cov_to_corr`.
    3. Stacked correlation blocks, each comprising a bilinear mapping
       (:class:`~spd_learn.modules.BiMap`), correlation normalization
       (:func:`~spd_learn.functional.cov_to_corr`), rectified eigenvalue activation
       (:class:`~spd_learn.modules.ReEig`), and correlation normalization.
    4. Riemannian tangent projection to hollow symmetric space
       (:func:`~spd_learn.functional.log_map_olm` or
       :func:`~spd_learn.functional.log_map_lsm`).
    5. Isometric lower-triangular embedding (:func:`~spd_learn.functional.tril_embed`).
    6. Dropout and linear classification head.

    Parameters
    ----------
    n_chans : int, default=22
        Number of input channels (matrix dimension before first BiMap).
    n_outputs : int, default=2
        Number of output classes.
    bimap_dims : tuple of int, default=(16, 12)
        Subspace dimensions for successive correlation BiMap layers.
    metric : {"OLM", "LSM"}, default="OLM"
        Riemannian metric on the correlation manifold used for tangent projection.
    threshold : float, default=1e-4
        Eigenvalue rectification threshold for :class:`~spd_learn.modules.ReEig`.
    drop_prob : float, default=0.0
        Dropout probability before the final classification head.
    input_type : {"raw", "cov"}, default="raw"
        Format of input data ('raw' time series or precomputed 'cov' matrices).
    cov_method : callable, default=sample_covariance
        Method to compute covariance when `input_type` is 'raw'.
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type for model parameters.
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        bimap_dims: tuple[int, ...] = (16, 12),
        metric: str = "OLM",
        threshold: float = 1e-4,
        drop_prob: float = 0.0,
        input_type: str = "raw",
        cov_method=sample_covariance,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if metric not in ("OLM", "LSM"):
            raise ValueError(f"Unknown metric '{metric}'. Expected 'OLM' or 'LSM'.")
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.bimap_dims = bimap_dims
        self.metric = metric
        self.input_type = input_type
        self.threshold = threshold

        self.cov = CovLayer(method=cov_method, device=device, dtype=dtype)
        self.shrinkage = Shrinkage(n_chans=n_chans, device=device, dtype=dtype)

        dims = (n_chans, *bimap_dims)
        self.blocks = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            bimap = BiMap(
                in_features=in_d, out_features=out_d, device=device, dtype=dtype
            )
            reeig = ReEig(threshold=threshold)
            self.blocks.append(nn.ModuleDict({"bimap": bimap, "reeig": reeig}))

        final_dim = dims[-1]
        tangent_dim = final_dim * (final_dim - 1) // 2
        self.dropout = nn.Dropout(p=drop_prob)
        self.classifier = nn.Linear(
            in_features=tangent_dim,
            out_features=n_outputs,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CorNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. If `input_type` is 'raw', shape is
            `(batch_size, n_chans, n_times)`. If `input_type` is 'cov',
            shape is `(batch_size, n_chans, n_chans)`.

        Returns
        -------
        torch.Tensor
            Output class logits of shape `(batch_size, n_outputs)`.
        """
        if self.input_type == "raw" or (x.ndim == 3 and x.shape[-1] != x.shape[-2]):
            x = self.cov(x)
            x = self.shrinkage(x)

        x = cov_to_corr(x)

        for block in self.blocks:
            assert isinstance(block, nn.ModuleDict)
            x = block["bimap"](x)
            x = cov_to_corr(x)
            x = block["reeig"](x)
            x = cov_to_corr(x)

        if self.metric == "OLM":
            x = log_map_olm(x)
        elif self.metric == "LSM":
            x = log_map_lsm(x)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'.")

        x = tril_embed(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
