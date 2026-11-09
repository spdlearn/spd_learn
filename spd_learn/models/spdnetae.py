# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from ..functional import covariance, log_euclidean_distance
from ..modules import BiMap, CovLayer, LogEig, ReEig, Shrinkage, TraceNorm


class SPDDecoderBlock(nn.Module):
    r"""SPD Decoder Block via Bilinear Dimension Expansion.

    This block performs dimensionality expansion from a latent SPD subspace
    to a higher-dimensional SPD space using :class:`~spd_learn.modules.BiMap`
    (which applies semi-orthogonal projection and identity padding via
    :class:`~spd_learn.modules.BiMapIncreaseDim`) followed by
    :class:`~spd_learn.modules.ReEig`.

    Parameters
    ----------
    in_features : int
        Dimensionality of input latent SPD matrices.
    out_features : int
        Target dimensionality of reconstructed SPD matrices.
    eps : float, default=1e-4
        Eigenvalue rectification threshold.
    device : torch.device, optional
        Device for layer parameters.
    dtype : torch.dtype, optional
        Data type for layer parameters.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models.spdnetae import SPDDecoderBlock
    >>> dec = SPDDecoderBlock(in_features=8, out_features=16)
    >>> Z = torch.eye(8).unsqueeze(0)
    >>> Y = dec(Z)
    >>> Y.shape
    torch.Size([1, 16, 16])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-4,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.bimap = BiMap(
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
        )
        self.reeig = ReEig(threshold=eps)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPD decoder block."""
        return self.reeig(self.bimap(Z))


class SPDNetAE(nn.Module):
    r"""Riemannian Autoencoder for SPD Matrices (SPDNet-AE).

    This class implements the Riemannian Autoencoder model proposed by
    Boucherie, de Surrel, and Yger :cite:p:`boucherie2026spdnetae`.
    SPDNet-AE learns compact latent SPD representations while preserving the
    intrinsic Riemannian geometry of Symmetric Positive Definite matrices
    throughout both the encoding and decoding pathways.

    Architecture Overview
    ---------------------
    1. **Covariance Frontend**: When raw signal data is provided, it is
       mapped to an SPD covariance matrix via `CovLayer()`, followed by
       scale normalization `TraceNorm()` and shrinkage regularization
       `Shrinkage(n_chans, init_shrinkage=-2.0)`.
    2. **SPD Encoder**: Stacks :class:`~spd_learn.modules.BiMap` and
       :class:`~spd_learn.modules.ReEig` blocks across `(n_chans, *bimap_dims)`
       to produce a compact latent SPD representation
       :math:`Z \in \mathcal{S}_{++}^{d_{\text{latent}}}`.
    3. **SPD Decoder**: Reconstructs the original
       :math:`(n_{\text{chans}} \times n_{\text{chans}})`
       SPD matrix from :math:`Z` using transposed bilinear congruence blocks:

       .. math::

           S_{\text{recon}} = W Z W^\top + \epsilon I

       with :math:`W \in \text{St}(d_{\text{out}}, d_{\text{in}})` followed by
       :class:`~spd_learn.modules.ReEig`, guaranteeing that the reconstructed
       matrix strictly remains on the SPD manifold
       :math:`\mathcal{S}_{++}^{n_{\text{chans}}}`.
    4. **Reconstruction Loss**: Evaluates the Log-Euclidean geodesic distance
       between the input covariance and reconstructed covariance:

       .. math::

           \mathcal{L}_{\text{recon}}(S_{\text{in}}, S_{\text{recon}}) =
           \|\log(S_{\text{in}}) - \log(S_{\text{recon}})\|_F^2

    5. **Classification**: When used in a classification pipeline (e.g. with
       `braindecode.EEGClassifier`), the latent representation :math:`Z` is projected
       to Euclidean space via :class:`~spd_learn.modules.LogEig`, followed by
       dropout and a linear classification head.

    Parameters
    ----------
    n_chans : int, default=22
        Number of EEG channels (dimension of the input covariance matrix).
    n_outputs : int, default=2
        Number of output classes for classification.
    bimap_dims : tuple[int, ...], default=(16, 8)
        Sequence of subspace dimensions for the SPD encoder layers.
        The latent SPD dimension is `bimap_dims[-1]`.
    eps : float, default=1e-4
        Eigenvalue rectification threshold and diagonal regularization parameter.
    drop_prob : float, default=0.0
        Dropout probability applied to the vectorized tangent features before
        the final linear classification layer.
    input_type : {"raw", "cov"}, default="raw"
        Format of input data ('raw' time series or precomputed 'cov' matrices).
    cov_method : callable, default=covariance
        Method to compute covariance when `input_type` is 'raw'.
    threshold : float, default=1e-4
        Eigenvalue rectification threshold for :class:`~spd_learn.modules.ReEig`.
    upper : bool, default=True
        If `True`, only upper-triangular elements are used in tangent projection.

    References
    ----------
    .. bibliography::
       :filter: docname in %user

    Examples
    --------
    >>> import torch
    >>> from spd_learn.models import SPDNetAE
    >>> model = SPDNetAE(n_chans=22, n_outputs=2, bimap_dims=(16, 8))
    >>> x = torch.randn(4, 22, 1000)  # (batch, channels, times)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([4, 2])
    >>> Z = model.encode(x)
    >>> Z.shape
    torch.Size([4, 8, 8])
    >>> S_recon = model.decode(Z)
    >>> S_recon.shape
    torch.Size([4, 22, 22])
    >>> S_in = model.cov_frontend(x)
    >>> loss = model.reconstruction_loss(S_in, S_recon)
    >>> loss.item() >= 0.0
    True
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 2,
        bimap_dims: Sequence[int] = (16, 8),
        eps: float = 1e-4,
        drop_prob: float = 0.0,
        input_type: str = "raw",
        cov_method=covariance,
        threshold: float = 1e-4,
        upper: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if input_type not in ("raw", "cov"):
            raise ValueError(
                f"Unknown input_type: '{input_type}'. Expected 'raw' or 'cov'."
            )
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.bimap_dims = tuple(bimap_dims)
        self.eps = threshold if threshold != 1e-4 else eps
        self.drop_prob = drop_prob
        self.input_type = input_type
        self.cov_method = cov_method
        self.upper = upper

        # Covariance Frontend
        self.cov_frontend: nn.Module
        if input_type == "raw":
            self.cov_frontend = nn.Sequential(
                CovLayer(method=cov_method, device=device, dtype=dtype),
                TraceNorm(),
                Shrinkage(
                    n_chans=n_chans,
                    init_shrinkage=-2.0,
                    learnable=True,
                    device=device,
                    dtype=dtype,
                ),
            )
        else:
            self.cov_frontend = nn.Identity()

        # SPD Encoder
        dims = (n_chans, *self.bimap_dims)
        encoder_layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            encoder_layers.append(BiMap(d_in, d_out, device=device, dtype=dtype))
            encoder_layers.append(ReEig(threshold=self.eps))
        self.encoder = nn.Sequential(*encoder_layers)

        # SPD Decoder
        dec_dims = (*reversed(self.bimap_dims), n_chans)
        decoder_layers: list[nn.Module] = []
        for d_in, d_out in zip(dec_dims[:-1], dec_dims[1:]):
            decoder_layers.append(
                SPDDecoderBlock(d_in, d_out, eps=self.eps, device=device, dtype=dtype)
            )
        self.decoder = nn.Sequential(*decoder_layers)

        # Classifier Head
        d_latent = self.bimap_dims[-1]
        self.logeig = LogEig(upper=upper)
        self.dropout = nn.Dropout(p=drop_prob)
        self.feat_dim = d_latent * (d_latent + 1) // 2 if upper else d_latent**2
        self.classifier = nn.Linear(
            self.feat_dim, n_outputs, device=device, dtype=dtype
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input signals or covariance matrices into latent SPD representations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. If raw signal data, expected shape is
            `(batch, n_chans, n_times)`. If already an SPD covariance matrix,
            expected shape is `(..., n_chans, n_chans)`.

        Returns
        -------
        torch.Tensor
            Latent SPD matrix with shape `(..., d_latent, d_latent)`
            where `d_latent = bimap_dims[-1]`.
        """
        if self.input_type != "raw" or (
            x.ndim == 3
            and (x.shape[-1] != self.n_chans or not torch.allclose(x, x.mT, atol=1e-4))
        ):
            x = self.cov_frontend(x)
        return self.encoder(x)

    def decode(self, Z_spd: torch.Tensor) -> torch.Tensor:
        """Reconstruct full-dimensional SPD matrices from latent SPD matrices.

        Parameters
        ----------
        Z_spd : torch.Tensor
            Latent SPD matrices with shape `(..., d_latent, d_latent)`.

        Returns
        -------
        torch.Tensor
            Reconstructed SPD matrices with shape `(..., n_chans, n_chans)`.
        """
        return self.decoder(Z_spd)

    def autoencode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct input to produce reconstructed SPD matrices.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of raw signals or SPD covariance matrices.

        Returns
        -------
        torch.Tensor
            Reconstructed SPD matrices with shape `(..., n_chans, n_chans)`.
        """
        Z_spd = self.encode(x)
        return self.decode(Z_spd)

    def reconstruction_loss(
        self,
        S_in: torch.Tensor,
        S_recon: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        r"""Compute the Log-Euclidean geodesic reconstruction loss.

        .. math::

            \mathcal{L}_{\text{recon}}(S_{\text{in}}, S_{\text{recon}}) =
            \|\log(S_{\text{in}}) - \log(S_{\text{recon}})\|_F^2

        Parameters
        ----------
        S_in : torch.Tensor
            Input SPD matrices with shape `(..., n_chans, n_chans)`.
        S_recon : torch.Tensor
            Reconstructed SPD matrices with shape `(..., n_chans, n_chans)`.
        reduction : {"mean", "sum", "none"}, default="mean"
            Reduction applied to the loss across the batch.

        Returns
        -------
        torch.Tensor
            Reconstruction loss.
        """
        dist_sq = log_euclidean_distance(S_in, S_recon).square()
        if reduction == "mean":
            return dist_sq.mean()
        if reduction == "sum":
            return dist_sq.sum()
        if reduction == "none":
            return dist_sq
        raise ValueError(
            f"Unknown reduction: '{reduction}'. Choose from 'mean', 'sum', 'none'."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPDNet-AE model as a classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of raw signals with shape `(batch, n_chans, n_times)`
            or covariance matrices with shape `(batch, n_chans, n_chans)`.

        Returns
        -------
        torch.Tensor
            Class logits with shape `(batch, n_outputs)`.
        """
        Z = self.encode(x)
        feats = self.logeig(Z)
        feats = self.dropout(feats)
        return self.classifier(feats)
