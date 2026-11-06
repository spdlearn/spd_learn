# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Riemannian Pooling Layers for Sequence and Foundation Models.

This module provides pooling operations that extract first- and second-order
Riemannian geometric statistics from sequence representations, transforming
token sequences into Symmetric Positive Definite (SPD) descriptors and projecting
them to classification features.

The implementations follow the Riemannian High-Order Pooling (RHOP) framework
proposed by Hu et al. :cite:p:`hu2026rhop` (ICLR 2026) for brain foundation models.
"""

from typing import Optional

import torch

from torch import nn

from ..functional import matrix_log, matrix_power, sample_covariance
from ..functional.metrics import cholesky_log
from .utils import Vech


class GaussianSPD(nn.Module):
    r"""Gaussian SPD Descriptor.

    Embeds empirical first-order (mean) and second-order (covariance) statistics
    of a sequence of token representations into an extended Symmetric Positive
    Definite (SPD) matrix using the quotient / Lovrić embedding
    :cite:p:`hu2026rhop`.

    Given a sequence of :math:`T` token representations
    :math:`X \in \mathbb{R}^{\dots \times T \times D}`, this layer computes the
    empirical mean :math:`\mu \in \mathbb{R}^{\dots \times D \times 1}`:

    .. math::

        \mu = \frac{1}{T} \sum_{t=1}^T X_t

    and centered token covariance matrix :math:`\Sigma \in \mathcal{S}_{++}^D`:

    .. math::

        \Sigma = \frac{1}{\max(T - 1, 1)} \sum_{t=1}^T (X_t - \mu)(X_t - \mu)^* + \epsilon I_D

    where :math:`\epsilon > 0` guarantees strict positive definiteness.

    The mean and covariance are embedded into a :math:`(D + 1) \times (D + 1)`
    SPD matrix :math:`\mathcal{G}(\mu, \Sigma) \in \mathcal{S}_{++}^{D+1}`:

    .. math::

        \mathcal{G}(\mu, \Sigma) = \begin{pmatrix}
            \Sigma + \beta \mu \mu^* & \alpha \mu \\
            \alpha \mu^* & \frac{\alpha^2}{\beta}
        \end{pmatrix}

    By the Schur complement condition, :math:`\mathcal{G}(\mu, \Sigma)` is strictly
    positive definite whenever :math:`\Sigma \succ 0`, :math:`\beta > 0`, and
    :math:`\alpha \neq 0`.

    Parameters
    ----------
    embedding_type : str, default="quotient"
        The type of Gaussian embedding into the SPD manifold. Supported:
        ``"quotient"`` (or ``"lovric"``).
    alpha : float, default=1.0
        Cross-term coupling coefficient between mean and covariance.
    beta : float, default=1.0
        Mean rank-one perturbation weight. Must be positive.
    eps : float, default=1e-5
        Regularization added to the covariance diagonal to ensure strict positive
        definiteness.
    device : torch.device, optional
        Device on which tensors will be allocated.
    dtype : torch.dtype, optional
        Data type of the module.

    See Also
    --------
    :class:`RHOP` : End-to-end pooling layer using this descriptor.
    :class:`~spd_learn.modules.CovLayer` : Standard sample covariance layer.

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import GaussianSPD
    >>> layer = GaussianSPD()
    >>> tokens = torch.randn(2, 10, 8)  # batch=2, tokens=10, features=8
    >>> G = layer(tokens)
    >>> G.shape
    torch.Size([2, 9, 9])
    """

    def __init__(
        self,
        embedding_type: str = "quotient",
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        embedding_type_lower = embedding_type.lower()
        if embedding_type_lower not in ("quotient", "lovric"):
            raise ValueError(
                f"Unsupported embedding_type '{embedding_type}'. "
                "Supported types: 'quotient', 'lovric'."
            )
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if alpha == 0:
            raise ValueError(f"alpha must be non-zero, got {alpha}")

        self.embedding_type = embedding_type_lower
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.device = device
        self.dtype = dtype

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute Gaussian SPD descriptor from sequence tensor.

        Parameters
        ----------
        X : torch.Tensor
            Sequence tensor of shape `(..., n_tokens, n_features)`.

        Returns
        -------
        torch.Tensor
            Gaussian SPD matrix of shape `(..., n_features + 1, n_features + 1)`.
        """
        T = X.shape[-2]
        D = X.shape[-1]

        # Empirical mean: mu_row is (..., 1, D), mu is (..., D, 1)
        mu_row = X.mean(dim=-2, keepdim=True)
        mu = mu_row.transpose(-2, -1)

        # Empirical covariance with epsilon regularization
        eye = torch.eye(D, device=X.device, dtype=X.dtype)
        if T > 1 and not X.is_complex():
            Sigma = sample_covariance(X.mT) + self.eps * eye
        else:
            X_centered = X - mu_row
            Sigma = (X_centered.conj().transpose(-2, -1) @ X_centered) / max(
                T - 1, 1
            ) + self.eps * eye

        # Quotient / Lovric embedding blocks
        top_left = Sigma + self.beta * (mu @ mu.conj().transpose(-2, -1))
        top_right = self.alpha * mu
        bottom_left = self.alpha * mu.conj().transpose(-2, -1)
        bottom_right = torch.full(
            (*X.shape[:-2], 1, 1),
            (self.alpha**2) / self.beta,
            device=X.device,
            dtype=X.dtype,
        )

        top = torch.cat([top_left, top_right], dim=-1)
        bottom = torch.cat([bottom_left, bottom_right], dim=-1)
        G = torch.cat([top, bottom], dim=-2)

        # Guarantee exact Hermitian symmetry
        G = 0.5 * (G + G.conj().transpose(-2, -1))
        return G


class RHOP(nn.Module):
    r"""Riemannian High-Order Pooling (RHOP) for Brain Foundation Models.

    Implements the Riemannian High-Order Pooling classification head proposed by
    Hu et al. :cite:p:`hu2026rhop` (ICLR 2026). RHOP augments transformer backbones
    (such as EEG foundation models BIOT and LaBraM) by capturing second-order
    Riemannian geometric statistics across sequence tokens, rather than relying
    exclusively on first-order average pooling or a single class token.

    **Algorithm Formulation**

    1. **Dimension Reduction**: Sequence tokens :math:`X \in \mathbb{R}^{\dots \times T \times D_{\text{in}}}`
       are projected to a reduced dimension :math:`d = \text{reduced\_dim}`:

       .. math::

           Z = X W_{\text{down}}^\top + b_{\text{down}} \in \mathbb{R}^{\dots \times T \times d}

    2. **Gaussian SPD Descriptor**: The empirical mean and covariance of :math:`Z`
       are embedded into a :math:`(d + 1) \times (d + 1)` SPD matrix:

       .. math::

           G = \mathcal{G}(\mu_Z, \Sigma_Z) \in \mathcal{S}_{++}^{d+1}

    3. **Riemannian Geometry Mapping**: The matrix :math:`G` is mapped to Euclidean /
       tangent space via one of three Riemannian metrics:

       - **LEM (Log-Euclidean Metric)**:

         .. math::

             S = \log(G)

       - **PEM (Power-Euclidean Metric)**:

         .. math::

             S = \frac{G^p - I}{p}

       - **LCM (Log-Cholesky Metric)**:

         .. math::

             L = \text{chol}(G), \quad S = \lfloor L \rfloor + \log(\mathbb{D}(L))

    4. **Norm-Preserving Vectorization**: Vectorizes the non-redundant symmetric
       coefficients via :class:`~spd_learn.modules.Vech`:

       .. math::

           v = \operatorname{vech}(S) \in \mathbb{R}^{\dots \times \frac{(d+1)(d+2)}{2}}

    5. **Up-projection & Fusion**: Vector :math:`v` is projected back to the backbone
       dimension :math:`D_{\text{in}}` and fused with the summary token:

       .. math::

           f_{\text{RHOP}} = v W_{\text{up}}^\top + b_{\text{up}}

       .. math::

           y = \begin{cases}
               x_{\text{cls}} + f_{\text{RHOP}} & \text{if } x_{\text{cls}} \text{ is provided} \\
               \bar{X} + f_{\text{RHOP}} & \text{otherwise}
           \end{cases}

    Parameters
    ----------
    in_features : int
        Input token feature dimension :math:`D_{\text{in}}`.
    reduced_dim : int, default=16
        Reduced dimension :math:`d` for token projection before forming the SPD descriptor.
    metric : str, default="LEM"
        Riemannian metric mapping. Options: ``"LEM"`` (Log-Euclidean Metric),
        ``"PEM"`` (Power-Euclidean Metric), ``"LCM"`` (Log-Cholesky Metric).
    power : float, default=0.5
        Power parameter :math:`p` used when ``metric="PEM"``.
    alpha : float, default=1.0
        Coupling scale factor for the Gaussian SPD descriptor.
    beta : float, default=1.0
        Mean rank-one perturbation weight for the Gaussian SPD descriptor.
    eps : float, default=1e-5
        Covariance regularization diagonal loading.
    device : torch.device, optional
        Device on which module parameters and buffers are created.
    dtype : torch.dtype, optional
        Data type of module parameters.

    See Also
    --------
    :class:`GaussianSPD` : Forms the quotient Gaussian SPD matrix.
    :class:`~spd_learn.modules.Vech` : Vectorization of symmetric matrix.

    References
    ----------
    .. bibliography::
       :filter: docname in docnames

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import RHOP
    >>> rhop = RHOP(in_features=32, reduced_dim=8, metric="LEM")
    >>> tokens = torch.randn(4, 16, 32)  # batch=4, tokens=16, dim=32
    >>> out = rhop(tokens)
    >>> out.shape
    torch.Size([4, 32])
    """

    def __init__(
        self,
        in_features: int,
        reduced_dim: int = 16,
        metric: str = "LEM",
        power: float = 0.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        metric_upper = metric.upper()
        if metric_upper not in ("LEM", "PEM", "LCM"):
            raise ValueError(
                f"Unsupported metric '{metric}'. Expected 'LEM', 'PEM', or 'LCM'."
            )
        if power == 0:
            raise ValueError("power must be non-zero for PEM")

        self.in_features = in_features
        self.reduced_dim = reduced_dim
        self.metric = metric_upper
        self.power = float(power)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.device = device
        self.dtype = dtype

        self.down_proj = nn.Linear(in_features, reduced_dim, device=device, dtype=dtype)
        self.descriptor = GaussianSPD(
            embedding_type="quotient",
            alpha=alpha,
            beta=beta,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        self.spd_dim = (reduced_dim + 1) * (reduced_dim + 2) // 2
        self.vech = Vech(preserve_norm=True, upper=True, device=device, dtype=dtype)
        self.up_proj = nn.Linear(self.spd_dim, in_features, device=device, dtype=dtype)

    def forward(
        self,
        X: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of Riemannian High-Order Pooling.

        Parameters
        ----------
        X : torch.Tensor
            Sequence tensor of shape `(..., n_tokens, in_features)`.
        cls_token : torch.Tensor, optional
            Optional class token of shape `(..., in_features)` or
            `(..., 1, in_features)`. If not provided, empirical token mean
            `X.mean(dim=-2)` is used for fusion.

        Returns
        -------
        torch.Tensor
            Pooled representation of shape `(..., in_features)`.
        """
        # 1. Project tokens to reduced dimension
        Z = self.down_proj(X)

        # 2. Form Gaussian SPD descriptor G of shape (..., reduced_dim + 1, reduced_dim + 1)
        G = self.descriptor(Z)

        # 3. Riemannian mapping to tangent / Euclidean representation
        is_complex = G.is_complex()
        G_in = G.real if is_complex else G

        if self.metric == "LEM":
            S = matrix_log.apply(G_in)
        elif self.metric == "PEM":
            eye = torch.eye(G_in.shape[-1], device=G_in.device, dtype=G_in.dtype)
            S = (matrix_power.apply(G_in, self.power) - eye) / self.power
        elif self.metric == "LCM":
            S = cholesky_log.apply(G_in)
            # S is lower-triangular with log-diagonal; symmetrize for Vech
            S = S + S.tril(-1).transpose(-2, -1)

        # 4. Vectorize non-redundant symmetric elements
        v = self.vech(S)
        if is_complex:
            v = v.to(dtype=X.dtype)

        # 5. Project back to backbone feature dimension
        rhop_feat = self.up_proj(v)

        # 6. Fuse with class token or token average
        if cls_token is not None:
            if cls_token.ndim == X.ndim and cls_token.shape[-2] == 1:
                cls_token = cls_token.squeeze(-2)
            fused = cls_token + rhop_feat
        else:
            token_mean = X.mean(dim=-2)
            fused = token_mean + rhop_feat

        return fused
