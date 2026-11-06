# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Correlation Manifold Attention Network (CorAtt).

Implements the CorAtt model and CorrAttention building block :cite:p:`hu2025riemannian`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spd_learn.functional import (
    cov_to_corr,
    distance_euclid,
    exp_map_lsm,
    exp_map_olm,
    geodesic_euclid,
    log_map_lsm,
    log_map_olm,
    off_diag,
    sample_covariance,
    tril_embed,
)
from spd_learn.models.matt import MAtt
from spd_learn.modules import BiMap


class CorrAttention(nn.Module):
    r"""Correlation Manifold Attention Module (CorrAttention).

    This module implements a self-attention mechanism on the Riemannian manifold
    of full-rank correlation matrices :math:`\mathcal{C}_{++}^d`
    :cite:p:`hu2025riemannian`.

    Given a sequence of correlation matrices
    :math:`X \in \mathbb{R}^{B \times P \times d \times d}`,
    the attention mechanism:
    1. Converts inputs to correlation matrices via :func:`cov_to_corr`.
    2. Maps to the hollow symmetric tangent space :math:`\text{Hol}(\mathcal{S}^d)` via
       :func:`log_map_olm` (for ``metric in ("OLM", "mix")``) or :func:`log_map_lsm`
       (for ``metric == "LSM"``).
    3. Transforms queries, keys, and values via Cayley-parameterized orthogonal mappings
       via :class:`~spd_learn.modules.BiMap`, projected onto the hollow tangent space:
       :math:`Q = \text{off\_diag}(W_Q^\top L W_Q)`,
       :math:`K = \text{off\_diag}(W_K^\top L W_K)`,
       :math:`V = \text{off\_diag}(W_V^\top L W_V)`.
    4. Computes pairwise Frobenius distances :math:`D_{ij} = \|Q_i - K_j\|_F` and
       attention weights :math:`A_{ij} = \text{softmax}_j(1 / (1 + \log(1 + D_{ij})))`.
    5. Computes the weighted sum
       :math:`\sum_j A_{ij} V_j + \text{off\_diag}(B + B^\top)`,
       followed by geodesic residual interpolation
       :math:`0.5 L + 0.5 \text{attn\_tangent}`.
    6. Maps back to :math:`\mathcal{C}_{++}^d` via :func:`exp_map_lsm`
       (for `"mix"`/`"LSM"`)
       or :func:`exp_map_olm` (for `"OLM"`).

    Parameters
    ----------
    dim : int
        Dimension :math:`d` of the correlation matrices.
    metric : {"mix", "OLM", "LSM"}, default="mix"
        Riemannian metric on the correlation manifold. In `"mix"` mode,
        the logarithmic map uses OLM (fast, closed-form) and the exponential
        map uses LSM (fast, closed-form).
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type for model parameters.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "mix",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if metric not in ("mix", "OLM", "LSM"):
            raise ValueError(
                f"Unknown metric '{metric}'. Expected one of ('mix', 'OLM', 'LSM')."
            )
        self.dim = dim
        self.metric = metric

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
        self.bias = nn.Parameter(torch.zeros(dim, dim, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        """Reset layer parameters."""
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of correlation attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of patch matrices with shape `(..., n_patches, dim, dim)`.

        Returns
        -------
        torch.Tensor
            Attended correlation matrices of shape `(..., n_patches, dim, dim)`.
        """
        C = cov_to_corr(x)

        if self.metric in ("OLM", "mix"):
            L = log_map_olm(C)
        else:
            L = log_map_lsm(C)

        # Hollow tangent projections via BiMap
        Q = off_diag(self.q_proj(L))
        K = off_diag(self.k_proj(L))
        V = off_diag(self.v_proj(L))

        # Pairwise Frobenius distance between patches
        D = distance_euclid(Q.unsqueeze(-3), K.unsqueeze(-4))

        # Attention weights
        energy = 1.0 / (1.0 + torch.log1p(D.clamp(min=0.0)))
        A = F.softmax(energy, dim=-1)

        # Weighted sum: \sum_j A_{ij} V_j
        attn_tangent = torch.einsum("...ij,...jkl->...ikl", A, V)

        # Bias term in hollow space
        bias_term = off_diag(self.bias + self.bias.mT)
        attn_tangent = attn_tangent + bias_term

        # Geodesic residual in tangent space
        out_tangent = geodesic_euclid(L, attn_tangent, alpha=0.5)

        # Exponential map back to correlation manifold
        if self.metric in ("LSM", "mix"):
            out = exp_map_lsm(out_tangent)
        else:
            out = exp_map_olm(out_tangent)

        return out


class CorAtt(MAtt):
    r"""Correlation Attention Network for EEG Decoding (CorAtt).

    This class implements the CorAtt model :cite:p:`hu2025riemannian`, a Riemannian
    self-attention network for EEG decoding operating on the correlation manifold.

    The model consists of:
    1. Spatial and temporal 2D convolution feature frontend (inherited from MAtt).
    2. Segmenting features into temporal patches and computing patch covariance.
    3. Normalizing to full-rank correlation matrices and applying
       :class:`CorrAttention`.
    4. Tangent projection (:func:`~spd_learn.functional.log_map_lsm` or
       :func:`~spd_learn.functional.log_map_olm`).
    5. :func:`~spd_learn.functional.tril_embed` isometrically vectorizing
       hollow tangent matrices.
    6. Dropout and Linear classification head.

    Parameters
    ----------
    n_patches : int, default=6
        Number of temporal patches.
    n_chans : int, default=22
        Number of EEG channels.
    n_outputs : int, default=2
        Number of output classes.
    temporal_out_channels : int, default=20
        Number of temporal convolution filters.
    temporal_kernel_size : int, default=25
        Kernel size for the temporal convolution.
    temporal_padding : int, optional
        Padding along time for the temporal convolution.
        Defaults to ``temporal_kernel_size // 2``.
    attention_in_features : int, default=20
        Input feature dimension for the correlation attention module.
    attention_out_features : int, default=20
        Output feature dimension for the correlation attention module.
    metric : {"mix", "OLM", "LSM"}, default="mix"
        Riemannian metric for the correlation attention module and tangent projection.
    drop_prob : float, default=0.0
        Dropout probability before the linear classifier.
    covariance_method : callable, default=sample_covariance
        Method used for computing sample covariances.
    spatial_filters : int, optional
        Alias for ``temporal_out_channels`` and ``attention_out_features``.
    temporal_filters : int, optional
        Alias for intermediate feature dimension.
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type for model parameters.
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
        attention_out_features: int = 20,
        metric: str = "mix",
        drop_prob: float = 0.0,
        covariance_method=sample_covariance,
        spatial_filters: Optional[int] = None,
        temporal_filters: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if spatial_filters is not None:
            temporal_out_channels = spatial_filters
            attention_in_features = spatial_filters
            attention_out_features = spatial_filters

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

        self.metric = metric
        self.drop_prob = drop_prob
        self.spatial_filters = temporal_out_channels
        self.temporal_filters = (
            temporal_filters if temporal_filters is not None else n_chans
        )

        self.manifold_attention = CorrAttention(  # type: ignore[assignment]
            dim=attention_out_features,
            metric=metric,
            device=device,
            dtype=dtype,
        )

        tangent_dim = attention_out_features * (attention_out_features - 1) // 2
        in_features = n_patches * tangent_dim
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=n_outputs,
            device=device,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(p=drop_prob)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CorAtt.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of raw EEG signals `(batch_size, n_chans, n_times)`
            or `(batch_size, 1, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output class logits of shape `(batch_size, n_outputs)`.
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

        x = self.manifold_attention(x)

        if self.metric in ("mix", "LSM"):
            x = log_map_lsm(x)
        elif self.metric == "OLM":
            x = log_map_olm(x)
        else:
            raise ValueError(f"Unknown metric '{self.metric}'.")

        x = tril_embed(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
