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
        self.shrinkage = Shrinkage(
            n_chans=n_chans,
            init_shrinkage=-3.0,
            learnable=True,
            device=device,
            dtype=dtype,
        )
        self.gamma = 1e-3

        dims = (n_chans, *bimap_dims)
        self.bimaps = nn.ModuleList(
            [
                BiMap(in_d, out_d, device=device, dtype=dtype)
                for in_d, out_d in zip(dims[:-1], dims[1:])
            ]
        )
        self.reeig = ReEig(threshold=threshold)

        final_dim = dims[-1]
        tangent_dim = final_dim * (final_dim - 1) // 2
        self.dropout = nn.Dropout(p=drop_prob)
        self.classifier = nn.Linear(
            in_features=tangent_dim,
            out_features=n_outputs,
            device=device,
            dtype=dtype,
        )

    def _regularized_cov_to_corr(self, S: torch.Tensor) -> torch.Tensor:
        """Apply scale normalization and diagonal shrinkage before cov_to_corr."""
        d = S.shape[-1]
        mean_diag = (
            S.diagonal(dim1=-2, dim2=-1)
            .abs()
            .sum(dim=-1, keepdim=True)
            .unsqueeze(-1)
            / d
        )
        S_scaled = S / (mean_diag + 1e-12)
        diag_S = torch.diag_embed(
            torch.diagonal(S_scaled, dim1=-2, dim2=-1).clamp(min=1e-6)
        )
        S_reg = (1.0 - self.gamma) * S_scaled + self.gamma * diag_S
        return cov_to_corr(S_reg)

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
        is_raw = self.input_type == "raw" and (
            x.ndim != 3
            or x.shape[-1] != x.shape[-2]
            or not torch.allclose(x, x.mT, atol=1e-4)
        )
        if is_raw:
            x = self.cov(x)
            d = x.shape[-1]
            mean_trace = (
                x.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) / d
            )
            eye = torch.eye(d, device=x.device, dtype=x.dtype)
            x = x / (mean_trace + 1e-8) + 1e-3 * eye
            x = self.shrinkage(x)

        x = self._regularized_cov_to_corr(x)

        for bimap in self.bimaps:
            x = self._regularized_cov_to_corr(
                self.reeig(self._regularized_cov_to_corr(bimap(x)))
            )

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
