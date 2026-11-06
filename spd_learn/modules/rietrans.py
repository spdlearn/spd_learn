# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Riemannian Multinomial Logistic Regression and Transformation Layers.

This module implements Riemannian deep learning layers based on:
Ziheng Chen, Yue Song, Yunmei Liu, and Nicu Sebe,
"Riemannian Multinomial Logistic Regression for Symmetric Positive Definite Matrices,"
NeurIPS 2024 (RMLR), and
Ziheng Chen, Rui Wang, and Nicu Sebe,
"Building Transformation Layers for Riemannian Neural Networks,"
NeurIPS 2026 (RieTrans).
"""

from typing import Optional

import torch
import torch.nn as nn

from torch.nn.utils.parametrize import is_parametrized, register_parametrization

from ..functional import ensure_sym
from ..functional.core import (
    matrix_exp,
    matrix_inv_sqrt,
    matrix_log,
    matrix_power,
    sym_to_upper,
    vec_to_sym,
)
from ..functional.metrics import cholesky_log
from .manifold import SymmetricPositiveDefinite


class SPDRMLR(nn.Module):
    r"""Riemannian Multinomial Logistic Regression for SPD matrices.

    Implements the Riemannian Multinomial Logistic Regression (RMLR) framework
    :cite:p:`chen2024rmlr` for Symmetric Positive Definite (SPD) manifolds.
    RMLR maps SPD matrices :math:`X \in \mathcal{S}_{++}^n` to a vectorized
    tangent space relative to a learnable reference SPD matrix :math:`P \in \mathcal{S}_{++}^n`,
    followed by a linear multinomial regression layer:

    .. math::

        V = \operatorname{Log}_P(X), \quad v = \operatorname{vech}(V), \quad y = W v + b

    Under the supported Riemannian geometries, the mapping :math:`\operatorname{Log}_P(X)`
    is computed as follows:

    - **LEM** (Log-Euclidean Metric):
      :math:`V = \log(X) - \log(P)`
    - **PEM** (Power-Euclidean Metric):
      :math:`V = \frac{X^\theta - P^\theta}{\theta}`
    - **AIM** (Affine-Invariant Metric):
      :math:`V = \log(P^{-1/2} X P^{-1/2})`
    - **LCM** (Log-Cholesky Metric):
      :math:`V = \log_{\text{chol}}(X) - \log_{\text{chol}}(P)`

    Parameters
    ----------
    n_features : int
        Size of the input SPD matrices (:math:`n \times n`).
    n_outputs : int
        Number of output classes or target features.
    metric : {"LEM", "LCM", "AIM", "PEM"}, default="LEM"
        Riemannian metric used to project matrices onto the tangent space.
    theta : float, default=1.0
        Power deformation parameter for the "PEM" metric. When :math:`\theta = 0`,
        the limit corresponds to the "LEM" metric.
    input_tangent : bool, default=False
        If True, input is assumed to already reside in the tangent space
        (as symmetric matrices or pre-vectorized vectors).
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type of parameters.

    Attributes
    ----------
    reference : nn.Parameter
        Learnable reference SPD matrix :math:`P \in \mathcal{S}_{++}^n`,
        parameterized via :class:`~spd_learn.modules.SymmetricPositiveDefinite`.
        Initialized to the identity matrix.
    linear : nn.Linear
        Linear projection layer from the vectorized tangent space of dimension
        :math:`n(n+1)/2` to ``n_outputs``.

    References
    ----------
    .. bibliography::
       :filter: key == "chen2024rmlr"

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import SPDRMLR
    >>> rmlr = SPDRMLR(n_features=4, n_outputs=2, metric="LEM")
    >>> X = torch.randn(8, 4, 4)
    >>> X = X @ X.mT + 0.1 * torch.eye(4)
    >>> y = rmlr(X)
    >>> y.shape
    torch.Size([8, 2])
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        metric: str = "LEM",
        theta: float = 1.0,
        input_tangent: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        valid_metrics = {"LEM", "LCM", "AIM", "PEM"}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}, got '{metric}'")

        self.n_features = n_features
        self.n_outputs = n_outputs
        self.metric = metric
        self.theta = theta
        self.input_tangent = input_tangent
        self.vec_dim = n_features * (n_features + 1) // 2

        self.reference = nn.Parameter(
            torch.empty(n_features, n_features, device=device, dtype=dtype)
        )
        self.linear = nn.Linear(
            self.vec_dim, n_outputs, bias=True, device=device, dtype=dtype
        )

        self.reset_parameters()
        self._parametrize()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset layer parameters to initial state."""
        identity = torch.eye(
            self.n_features,
            device=self.linear.weight.device,
            dtype=self.linear.weight.dtype,
        )
        if is_parametrized(self, "reference"):
            setattr(self, "reference", identity)
        else:
            self.reference.copy_(identity)
        self.linear.reset_parameters()

    def _parametrize(self) -> None:
        register_parametrization(self, "reference", SymmetricPositiveDefinite())

    def _to_tangent(self, X: torch.Tensor) -> torch.Tensor:
        """Map SPD matrices to tangent space at the reference point."""
        P = self.reference
        if self.metric == "LEM":
            return matrix_log.apply(X) - matrix_log.apply(P)
        elif self.metric == "PEM":
            if self.theta == 0.0:
                return matrix_log.apply(X) - matrix_log.apply(P)
            return (
                matrix_power.apply(X, self.theta) - matrix_power.apply(P, self.theta)
            ) / self.theta
        elif self.metric == "AIM":
            P_inv_sqrt = matrix_inv_sqrt.apply(P)
            return matrix_log.apply(P_inv_sqrt @ X @ P_inv_sqrt)
        elif self.metric == "LCM":
            return cholesky_log.apply(X) - cholesky_log.apply(P)
        raise ValueError(f"Unsupported metric: {self.metric}")

    def _vectorize(self, V: torch.Tensor) -> torch.Tensor:
        """Vectorize tangent matrix."""
        if self.metric == "LCM":
            return sym_to_upper(V, preserve_norm=False, upper=False)
        return sym_to_upper(ensure_sym(V), preserve_norm=True, upper=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPDRMLR.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. If ``input_tangent=False``, shape `(..., n_features, n_features)`.
            If ``input_tangent=True``, shape `(..., n_features, n_features)` or
            `(..., vec_dim)`.

        Returns
        -------
        torch.Tensor
            Classification logits or transformed features of shape `(..., n_outputs)`.
        """
        if self.input_tangent:
            if (
                X.ndim >= 2
                and X.shape[-1] == self.n_features
                and X.shape[-2] == self.n_features
            ):
                v = self._vectorize(X)
            else:
                v = X
        else:
            V = self._to_tangent(X)
            v = self._vectorize(V)

        return self.linear(v)

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, n_outputs={self.n_outputs}, "
            f"metric='{self.metric}', theta={self.theta}, "
            f"input_tangent={self.input_tangent}"
        )


class SPDLinear(nn.Module):
    r"""Intrinsic SPD Transformation Layer.

    Implements the intrinsic SPD transformation layer for Riemannian neural networks
    introduced in RieTrans :cite:p:`chen2026rietrans`.

    This layer transforms input SPD matrices in :math:`\mathcal{S}_{++}^{d_{\text{in}}}`
    to output SPD matrices in :math:`\mathcal{S}_{++}^{d_{\text{out}}}` (or symmetric
    tangent matrices in :math:`\mathcal{S}^{d_{\text{out}}}`). It leverages an internal
    :class:`SPDRMLR` module to project matrices into an unconstrained vectorized space
    of dimension :math:`d_{\text{out}}(d_{\text{out}}+1)/2`, reconstructs the symmetric
    matrix, and optionally applies the matrix exponential to guarantee positive definiteness.

    Parameters
    ----------
    in_features : int
        Dimension of the input SPD matrices (:math:`d_{\text{in}}`).
    out_features : int
        Dimension of the output SPD matrices (:math:`d_{\text{out}}`).
    metric : {"LEM", "LCM", "AIM", "PEM"}, default="LEM"
        Riemannian metric for the tangent space projection.
    theta : float, default=1.0
        Deformation parameter for the "PEM" metric.
    input_tangent : bool, default=False
        If True, expects input to already reside in the tangent space.
    return_tangent : bool, default=False
        If True, returns symmetric tangent matrices in :math:`\mathcal{S}^{d_{\text{out}}}`.
        If False, applies matrix exponential to return strictly SPD matrices.
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type of parameters.

    References
    ----------
    .. bibliography::
       :filter: key == "chen2026rietrans"

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import SPDLinear
    >>> layer = SPDLinear(in_features=4, out_features=3, metric="LEM")
    >>> X = torch.randn(8, 4, 4)
    >>> X = X @ X.mT + 0.1 * torch.eye(4)
    >>> Y = layer(X)
    >>> Y.shape
    torch.Size([8, 3, 3])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        metric: str = "LEM",
        theta: float = 1.0,
        input_tangent: bool = False,
        return_tangent: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.metric = metric
        self.theta = theta
        self.input_tangent = input_tangent
        self.return_tangent = return_tangent

        self.out_vec_dim = out_features * (out_features + 1) // 2
        self.rmlr = SPDRMLR(
            n_features=in_features,
            n_outputs=self.out_vec_dim,
            metric=metric,
            theta=theta,
            input_tangent=input_tangent,
            device=device,
            dtype=dtype,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPDLinear.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD tensor of shape `(..., in_features, in_features)` (or tangent
            tensor if ``input_tangent=True``).

        Returns
        -------
        torch.Tensor
            Transformed SPD tensor of shape `(..., out_features, out_features)`,
            or tangent matrix if ``return_tangent=True``.
        """
        v = self.rmlr(X)
        Y_tan = vec_to_sym(v, preserve_norm=True, upper=True)
        if self.return_tangent:
            return Y_tan
        return matrix_exp.apply(Y_tan)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"metric='{self.metric}', theta={self.theta}, "
            f"input_tangent={self.input_tangent}, return_tangent={self.return_tangent}"
        )
