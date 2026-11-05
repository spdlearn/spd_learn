# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from torch.nn.utils.parametrize import register_parametrization

from spd_learn.functional import (
    ensure_sym,
    sample_covariance,
    spd_centering,
    spd_rebiasing,
)
from spd_learn.functional.core import matrix_inv_sqrt, matrix_power, matrix_sqrt
from spd_learn.models.matt import MAtt
from spd_learn.modules import (
    BiMap,
    SymmetricPositiveDefinite,
)


class GBWAtt(MAtt):
    r"""Generalized Bures-Wasserstein Riemannian Self-Attention Network (GBWAtt).

    This class implements the GBWAtt model :cite:p:`zhou2026towards`, an EEG
    decoding architecture founded on Riemannian self-attention under the
    generalized :math:`(2\theta, M)`-Bures-Wasserstein (GBW) metric.

    The architecture upgrades Log-Euclidean manifold self-attention to deformed
    power geometry and congruence whitening against a learnable reference SPD
    matrix, with closed-form barycentric aggregation:

    1. **Frontend Feature Extraction:** Raw EEG signals are processed by a
       spatial convolutional filter followed by a temporal convolutional layer
       to extract spatiotemporal representations.
    2. **Euclidean-to-Riemannian (E2R) Mapping:** The feature representations
       are segmented into temporal patches, followed by sample covariance
       estimation and trace normalization.
    3. **Generalized Bures-Wasserstein Attention:** An attention mechanism is
       computed using pairwise generalized Bures-Wasserstein distances between
       whitened, power-deformed queries and keys, and aggregated via closed-form
       generalized BW barycenters.
    4. **Riemannian-to-Euclidean (R2E) Mapping:** ReEig rectification and
       tangent space projection via LogEig embed the attended SPD matrices into
       Euclidean vectors.
    5. **Classification:** A dropout layer and a linear classifier output class
       logits.

    Parameters
    ----------
    n_patches : int, default=2
        Number of temporal patches for patch embedding.
    n_chans : int, default=22
        Number of EEG input channels.
    n_outputs : int, default=2
        Number of output classes for classification.
    temporal_out_channels : int, default=20
        Number of output channels for the temporal convolution.
    temporal_kernel_size : int, default=12
        Kernel length along time for the temporal convolution.
    temporal_padding : int, default=6
        Padding along time for the temporal convolution.
    attention_in_features : int, default=20
        Input feature dimension for the GBW attention module.
    attention_out_features : int, default=18
        Output feature dimension for the GBW attention module.
    theta : float, default=0.5
        Deformation parameter for the :math:`(2\theta, M)`-BW metric.
    learnable_reference : bool, default=True
        Whether the metric reference matrix :math:`M` is learnable.
    drop_prob : float, default=0.0
        Dropout probability before the final classification head.
    covariance_method : callable, default=sample_covariance
        Method to estimate covariance matrices.
    spatial_filters : int, optional
        Alias for ``temporal_out_channels`` and ``attention_in_features``.
    att_out_features : int, optional
        Alias for ``attention_out_features``.
    temporal_filters : int, optional
        Alias for intermediate feature dimension.
    """

    def __init__(
        self,
        n_patches: int = 2,
        n_chans: int = 22,
        n_outputs: int = 2,
        temporal_out_channels: int = 20,
        temporal_kernel_size: int = 12,
        temporal_padding: int = 6,
        attention_in_features: int = 20,
        attention_out_features: int = 18,
        theta: float = 0.5,
        learnable_reference: bool = True,
        drop_prob: float = 0.0,
        covariance_method=sample_covariance,
        spatial_filters: Optional[int] = None,
        att_out_features: Optional[int] = None,
        temporal_filters: Optional[int] = None,
    ):
        if spatial_filters is not None:
            temporal_out_channels = spatial_filters
            attention_in_features = spatial_filters
        if att_out_features is not None:
            attention_out_features = att_out_features

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

        self.theta = theta
        self.learnable_reference = learnable_reference
        self.drop_prob = drop_prob
        self.spatial_filters = temporal_out_channels
        self.att_out_features = attention_out_features
        self.temporal_filters = (
            temporal_filters if temporal_filters is not None else n_chans
        )

        self.manifold_attention = GBWAttention(
            in_features=attention_in_features,
            out_features=attention_out_features,
            theta=theta,
            learnable_reference=learnable_reference,
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GBWAtt model.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape `(batch_size, n_chans, n_times)` or
            `(batch_size, 1, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(batch_size, n_outputs)`.
        """
        if input.ndim == 4 and input.shape[1] == 1:
            input = input.squeeze(1)
        x = self.add_extra_dim(input)
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
        x = self.linear(x)
        return x


class GBWAttention(nn.Module):
    r"""Generalized :math:`(2\theta, M)`-Bures-Wasserstein Riemannian Self-Attention.

    This module implements Riemannian self-attention on the SPD manifold under the
    generalized :math:`(2\theta, M)`-Bures-Wasserstein metric :cite:p:`zhou2026towards`.

    Given input SPD matrices :math:`X \in \mathcal{S}_{++}^{d_{\text{in}}}`, the module:

    1. Projects inputs via Stiefel-constrained bilinear mappings:
       :math:`Q = \text{q\_proj}(X)`, :math:`K = \text{k\_proj}(X)`,
       :math:`V = \text{v\_proj}(X) \in \mathcal{S}_{++}^{d_{\text{out}}}`.
    2. Applies power deformation with exponent :math:`\theta > 0`:
       :math:`Q_\theta = Q^\theta`, :math:`K_\theta = K^\theta`,
       :math:`V_\theta = V^\theta`.
    3. Whitens with respect to a learnable SPD reference matrix :math:`M \in \mathcal{S}_{++}^{d_{\text{out}}}`:
       :math:`\tilde{Q} = M^{-1/2} Q_\theta M^{-1/2}`,
       :math:`\tilde{K} = M^{-1/2} K_\theta M^{-1/2}`,
       :math:`\tilde{V} = M^{-1/2} V_\theta M^{-1/2}`.
    4. Computes pairwise Bures-Wasserstein distances:
       :math:`d_{\text{BW}}^2(\tilde{Q}_i, \tilde{K}_j) = \text{tr}(\tilde{Q}_i) + \text{tr}(\tilde{K}_j) - 2\,\text{tr}\left((\tilde{Q}_i^{1/2} \tilde{K}_j \tilde{Q}_i^{1/2})^{1/2}\right)`.
    5. Normalizes attention weights:
       :math:`A_{ij} = \text{softmax}_j\left(\frac{1}{1 + \log(1 + D_{ij})}\right)`.
    6. Aggregates via closed-form generalized BW barycenters:
       :math:`S_i = \sum_{j=1}^P A_{ij} \tilde{V}_j^{1/2}`,
       :math:`\tilde{Y}_i = S_i S_i^\top`,
       :math:`Y_{\theta, i} = M^{1/2} \tilde{Y}_i M^{1/2}`,
       :math:`Y_i = (Y_{\theta, i})^{1/\theta}`.

    Parameters
    ----------
    in_features : int
        Dimensionality of input SPD matrices.
    out_features : int
        Dimensionality of projected SPD matrices.
    theta : float, default=0.5
        Power deformation parameter :math:`\theta > 0`.
    learnable_reference : bool, default=True
        Whether the SPD metric reference matrix :math:`M` is learnable.
    eps : float, default=1e-5
        Numerical stability threshold for distance clamping and eigenvalue square roots.
    device : Optional[torch.device], default=None
        Device for layer parameters.
    dtype : Optional[torch.dtype], default=None
        Data type for layer parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        theta: float = 0.5,
        learnable_reference: bool = True,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")

        self.in_features = in_features
        self.out_features = out_features
        self.theta = float(theta)
        self.learnable_reference = learnable_reference
        self.eps = float(eps)

        self.q_proj = BiMap(in_features, out_features, device=device, dtype=dtype)
        self.k_proj = BiMap(in_features, out_features, device=device, dtype=dtype)
        self.v_proj = BiMap(in_features, out_features, device=device, dtype=dtype)

        self.reference = nn.Parameter(
            torch.eye(out_features, device=device, dtype=dtype),
            requires_grad=learnable_reference,
        )
        register_parametrization(self, "reference", SymmetricPositiveDefinite())
        with torch.no_grad():
            self.parametrizations.reference.original.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Generalized Bures-Wasserstein Attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(..., P, in_features, in_features)` where
            each matrix slice is Symmetric Positive Definite.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(..., P, out_features, out_features)`.
        """
        # Bilinear projections on the SPD manifold
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Power deformation
        Q_theta = matrix_power.apply(Q, self.theta)
        K_theta = matrix_power.apply(K, self.theta)
        V_theta = matrix_power.apply(V, self.theta)

        # Metric reference decomposition
        M = self.reference
        M_sqrt = matrix_sqrt.apply(M)
        M_inv_sqrt = matrix_inv_sqrt.apply(M)

        # Congruence whitening
        tilde_Q = spd_centering(Q_theta, M_inv_sqrt)
        tilde_K = spd_centering(K_theta, M_inv_sqrt)
        tilde_V = spd_centering(V_theta, M_inv_sqrt)

        # Pairwise Generalized Bures-Wasserstein Distance
        tilde_Q_sqrt = matrix_sqrt.apply(tilde_Q)
        tr_Q = torch.diagonal(tilde_Q, dim1=-2, dim2=-1).sum(dim=-1)
        tr_K = torch.diagonal(tilde_K, dim1=-2, dim2=-1).sum(dim=-1)

        # Expand to form tilde_Q_{sqrt, i} tilde_K_j tilde_Q_{sqrt, i}
        Q_s = tilde_Q_sqrt.unsqueeze(-3)
        K_s = tilde_K.unsqueeze(-4)
        M_ij = ensure_sym(Q_s @ K_s @ Q_s)

        eigvals = torch.linalg.eigvalsh(M_ij)
        tr_ABA_sqrt = torch.sum(torch.sqrt(eigvals.clamp(min=self.eps)), dim=-1)

        dist_sq = tr_Q.unsqueeze(-1) + tr_K.unsqueeze(-2) - 2.0 * tr_ABA_sqrt
        D_ij = torch.sqrt(dist_sq.clamp(min=self.eps))

        # Attention weights
        scores = 1.0 / (1.0 + torch.log1p(D_ij))
        A = F.softmax(scores, dim=-1)

        # Closed-form Generalized BW Barycenter Aggregation
        tilde_V_sqrt = matrix_sqrt.apply(tilde_V)
        S = torch.einsum("...pq,...qij->...pij", A, tilde_V_sqrt)
        tilde_Y = S @ S.mT
        Y_theta = spd_rebiasing(tilde_Y, M_sqrt)
        Y = matrix_power.apply(ensure_sym(Y_theta), 1.0 / self.theta)

        return ensure_sym(Y)
