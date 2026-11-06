# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Sequence, Union

import torch
import torch.nn as nn

from spd_learn.functional import (
    ensure_sym,
    matrix_inv_sqrt,
    matrix_log,
    sample_covariance,
    sym_to_upper,
)
from spd_learn.modules import (
    BiMap,
    CovLayer,
    LogEig,
    ReEig,
    Shrinkage,
    SPDBatchNormMean,
    TraceNorm,
)


class DeepGeoCCA(nn.Module):
    r"""Deep Geodesic Canonical Correlation Analysis (DeepGeoCCA).

    This model implements the DeepGeoCCA architecture proposed in Ju et al.
    (2024) :cite:p:`ju2024self`. It performs multi-view learning on Symmetric
    Positive Definite (SPD) manifolds by maximizing the geodesic correlation
    between two multi-scale representations of multivariate time series
    (e.g., EEG signals).

    Given a multi-channel signal, two view-specific temporal convolutions with
    complementary kernel sizes extract multi-scale temporal dynamics. Each view
    is then mapped to the SPD manifold via spatial covariance estimation,
    regularized with trace normalization and learnable shrinkage, and processed
    by stacked bilinear mapping (:class:`~spd_learn.modules.BiMap`) and
    eigenvalue rectification (:class:`~spd_learn.modules.ReEig`) layers,
    followed by Riemannian batch normalization
    (:class:`~spd_learn.modules.SPDBatchNormMean`).

    For classification, latent SPD representations from both views are mapped
    to Euclidean tangent vectors via :class:`~spd_learn.modules.LogEig`,
    concatenated, and fed to a linear classifier. During self-supervised
    pre-training or multi-task optimization, :class:`GeodesicCCALoss` maximizes
    the geodesic canonical correlation between the two views.

    Parameters
    ----------
    n_chans : int, default=22
        Number of input EEG channels.
    n_outputs : int, default=2
        Number of output classes for classification.
    bimap_dims : tuple of int, default=(16, 10)
        Sequence of output subspace dimensions for the BiMap layers in each
        encoder.
    outdim_size : int, default=10
        Number of top canonical correlation dimensions to extract and maximize
        in :class:`GeodesicCCALoss`.
    eps : float, default=1e-4
        Regularization constant for numerical stability in eigenvalue clamping,
        covariance regularization, and CCA whitening.
    drop_prob : float, default=0.0
        Dropout probability applied to the concatenated tangent features before
        the linear classifier.
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        bimap_dims: Sequence[int] = (16, 10),
        outdim_size: int = 10,
        eps: float = 1e-4,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.bimap_dims = tuple(bimap_dims)
        self.outdim_size = outdim_size
        self.eps = eps
        self.drop_prob = drop_prob

        # View-specific temporal convolutions (15 vs 31 samples for multi-scale views)
        self.view1_conv = nn.Conv1d(
            n_chans, n_chans, kernel_size=15, padding=7, groups=n_chans, bias=False
        )
        self.view2_conv = nn.Conv1d(
            n_chans, n_chans, kernel_size=31, padding=15, groups=n_chans, bias=False
        )

        # Build view-specific SPDNet encoders
        self.encoder1 = self._build_encoder(n_chans, self.bimap_dims, eps)
        self.encoder2 = self._build_encoder(n_chans, self.bimap_dims, eps)

        # Projection to tangent space and classification head
        d_out = self.bimap_dims[-1] if self.bimap_dims else n_chans
        self.vec_dim = d_out * (d_out + 1) // 2
        self.logeig = LogEig(upper=True)
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(2 * self.vec_dim, n_outputs)

        # Geodesic CCA loss for multi-view self-supervised alignment
        self.cca_loss = GeodesicCCALoss(outdim_size=outdim_size, eps=eps)

    @staticmethod
    def _build_encoder(
        n_chans: int, bimap_dims: tuple[int, ...], eps: float
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            CovLayer(),
            TraceNorm(),
            Shrinkage(n_chans, init_shrinkage=-2.0),
        ]
        dims = (n_chans, *bimap_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(BiMap(in_dim, out_dim))
            layers.append(ReEig(threshold=eps))
        if bimap_dims:
            layers.append(SPDBatchNormMean(bimap_dims[-1]))
        return nn.Sequential(*layers)

    def encode_views(
        self,
        x1: Union[torch.Tensor, Sequence[torch.Tensor]],
        x2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw signals into SPD representations for both views.

        Parameters
        ----------
        x1 : torch.Tensor or sequence of torch.Tensor
            If `x2` is None, either a single 3D EEG tensor of shape
            `(batch_size, n_chans, n_times)` (which will be filtered into two
            multi-scale views via `view1_conv` and `view2_conv`), or a pair
            `(v1, v2)` of two view tensors. If `x2` is provided, `x1` is the
            first view tensor.
        x2 : torch.Tensor, optional
            The second view tensor of shape `(batch_size, n_chans, n_times)`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            `(Y1, Y2)` where each is a batch of SPD matrices of shape
            `(batch_size, d_out, d_out)`.
        """
        if x2 is None:
            if isinstance(x1, (tuple, list)):
                v1, v2 = x1[0], x1[1]
                v1 = self.view1_conv(v1)
                v2 = self.view2_conv(v2)
            else:
                v1 = self.view1_conv(x1)
                v2 = self.view2_conv(x1)
        else:
            v1 = self.view1_conv(x1)
            v2 = self.view2_conv(x2)

        Y1 = self.encoder1(v1)
        Y2 = self.encoder2(v2)
        return Y1, Y2

    def compute_cca_loss(
        self,
        x1: Union[torch.Tensor, Sequence[torch.Tensor]],
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Geodesic CCA loss between the two views.

        Parameters
        ----------
        x1 : torch.Tensor or sequence of torch.Tensor
            Input EEG signals of shape `(batch_size, n_chans, n_times)` or paired views.
        x2 : torch.Tensor, optional
            Second view if not provided in `x1`.

        Returns
        -------
        torch.Tensor
            Scalar Geodesic CCA loss (negative total canonical correlation).
        """
        Y1, Y2 = self.encode_views(x1, x2)
        return self.cca_loss(Y1, Y2)

    def forward(
        self,
        x: Union[torch.Tensor, Sequence[torch.Tensor]],
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of DeepGeoCCA.

        Parameters
        ----------
        x : torch.Tensor or sequence of torch.Tensor
            Input EEG signals of shape `(batch_size, n_chans, n_times)` or
            paired views `(x1, x2)`.
        x2 : torch.Tensor, optional
            Second view if not provided in `x`.

        Returns
        -------
        torch.Tensor
            Class logits of shape `(batch_size, n_outputs)`.
        """
        Y1, Y2 = self.encode_views(x, x2)
        z1 = self.logeig(Y1)
        z2 = self.logeig(Y2)
        z = torch.cat([z1, z2], dim=-1)
        z = self.dropout(z)
        return self.classifier(z)


class GeodesicCCALoss(nn.Module):
    r"""Geodesic Canonical Correlation Analysis (GeodesicCCA) Loss.

    Computes the negative geodesic canonical correlation loss between two batches
    of Symmetric Positive Definite (SPD) matrices :cite:p:`ju2024self`.

    Each batch of SPD matrices :math:`Y_1, Y_2 \in \mathbb{P}_d` is mapped to
    the tangent space at their empirical Fréchet mean via the matrix logarithm:

    .. math::

        L_1 = \log(Y_1), \quad L_2 = \log(Y_2)

    The tangent matrices are centered across the batch and vectorized into
    Euclidean representations :math:`H_1, H_2 \in \mathbb{R}^{B \times m}`
    where :math:`m = d(d+1)/2`. Regularized covariance matrices are then computed:

    .. math::

        \Sigma_{11} = \frac{1}{N} H_1^\top H_1 + \epsilon I_m, \quad
        \Sigma_{22} = \frac{1}{N} H_2^\top H_2 + \epsilon I_m, \quad
        \Sigma_{12} = \frac{1}{N} H_1^\top H_2

    where :math:`N = \max(B - 1, 1)`. The whitened cross-correlation matrix
    :math:`T = \Sigma_{11}^{-1/2} \Sigma_{12} \Sigma_{22}^{-1/2}` is formed, and
    its top-:math:`k` singular values :math:`\rho_i`
    (:math:`k = \min(\text{outdim\_size}, m)`)
    are extracted via the eigendecomposition of :math:`T^\top T + \epsilon I_m`.
    The loss returned is the negative sum of these canonical correlations:

    .. math::

        \mathcal{L}_{\text{GeoCCA}} = -\sum_{i=1}^k \rho_i

    Parameters
    ----------
    outdim_size : int, default=10
        Number of top canonical correlation components to extract and maximize.
    eps : float, default=1e-4
        Regularization parameter added to covariance matrices and eigenvalue
        clamping for numerical stability.
    """

    def __init__(
        self,
        outdim_size: int = 10,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.outdim_size = outdim_size
        self.eps = eps

    def forward(self, Y1: torch.Tensor, Y2: torch.Tensor) -> torch.Tensor:
        """Compute the negative geodesic canonical correlation loss.

        Parameters
        ----------
        Y1 : torch.Tensor
            Batch of SPD matrices from view 1 of shape `(batch_size, d, d)`.
        Y2 : torch.Tensor
            Batch of SPD matrices from view 2 of shape `(batch_size, d, d)`.

        Returns
        -------
        torch.Tensor
            Scalar negative total geodesic correlation.
        """
        if Y1.shape != Y2.shape:
            raise ValueError(
                f"Y1 and Y2 must have the same shape, got {Y1.shape} and {Y2.shape}"
            )
        if Y1.ndim < 3 or Y1.shape[-1] != Y1.shape[-2]:
            raise ValueError(
                "Y1 and Y2 must be batches of square matrices "
                f"with at least 3 dimensions, got shape {Y1.shape}"
            )

        batch_size = Y1.shape[0]

        # Map to Log-Euclidean tangent space
        L1 = matrix_log.apply(ensure_sym(Y1))
        L2 = matrix_log.apply(ensure_sym(Y2))

        # Center across batch in tangent space
        L1_c = L1 - L1.mean(dim=0, keepdim=True)
        L2_c = L2 - L2.mean(dim=0, keepdim=True)

        # Vectorize symmetric matrices into tangent coordinates (batch_size, m)
        H1 = sym_to_upper(L1_c)
        H2 = sym_to_upper(L2_c)
        m = H1.shape[-1]

        N = max(batch_size - 1, 1)
        eye_m = torch.eye(m, device=Y1.device, dtype=Y1.dtype)

        if batch_size > 1:
            Sigma11 = ensure_sym(sample_covariance(H1.mT) + self.eps * eye_m)
            Sigma22 = ensure_sym(sample_covariance(H2.mT) + self.eps * eye_m)
        else:
            Sigma11 = ensure_sym(H1.mT @ H1 + self.eps * eye_m)
            Sigma22 = ensure_sym(H2.mT @ H2 + self.eps * eye_m)
        Sigma12 = (1.0 / N) * (H1.mT @ H2)

        Sigma11_inv_sqrt = matrix_inv_sqrt.apply(Sigma11)
        Sigma22_inv_sqrt = matrix_inv_sqrt.apply(Sigma22)
        T = Sigma11_inv_sqrt @ Sigma12 @ Sigma22_inv_sqrt

        TT = ensure_sym(T.transpose(-2, -1) @ T + self.eps * eye_m)
        eigvals = torch.linalg.eigvalsh(TT)

        k = min(self.outdim_size, m)
        topk_eigvals = eigvals[..., -k:]
        corr = torch.sqrt(torch.clamp(topk_eigvals, min=self.eps))

        return -corr.sum()
