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

import math

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrize import is_parametrized, register_parametrization

from ..functional import ensure_sym
from ..functional.core import (
    clamp_eigvals,
    matrix_exp,
    matrix_inv_sqrt,
    matrix_log,
    matrix_power,
    matrix_sqrt,
    matrix_sqrt_inv,
    sym_to_upper,
    vec_to_sym,
)
from ..functional.metrics import cholesky_exp, cholesky_log
from ..functional.numerical import get_epsilon
from .manifold import SymmetricPositiveDefinite


VALID_RIETRANS_METRICS = {"LEM", "LCM", "AIM", "PEM", "BWM"}


class SPDRMLR(nn.Module):
    r"""Riemannian Multinomial Logistic Regression for SPD matrices.

    Implements the Riemannian Multinomial Logistic Regression (RMLR) framework
    :cite:p:`chen2024rmlr` and its unidirectional product-SPD extension from
    RieTrans :cite:p:`chen2026rietrans` for Symmetric Positive Definite (SPD)
    manifolds. RMLR maps SPD matrices :math:`X \in \mathcal{S}_{++}^n` to a
    vectorized tangent space relative to a reference SPD matrix
    :math:`P \in \mathcal{S}_{++}^n` (or along unidirectional geodesics from
    :math:`I_n`), followed by a linear multinomial regression layer:

    .. math::

        V = \operatorname{Log}_P(X), \quad v = \operatorname{vech}_{\alpha,\beta}(V), \quad y = W v + b

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
    - **BWM** (Bures-Wasserstein Metric, :math:`2\theta`-BWM):
      :math:`V = (PX)^{1/2} + (XP)^{1/2} - 2P` for standard BWM (:math:`\theta \in \{0.5, 1.0\}`)

    Parameters
    ----------
    n_features : int
        Size of the input SPD matrices (:math:`n \times n`).
    n_outputs : int
        Number of output classes or target features.
    metric : {"LEM", "LCM", "AIM", "PEM", "BWM"}, default="LEM"
        Riemannian metric used to project matrices onto the tangent space.
    theta : float, default=1.0
        Power deformation parameter for the "PEM" or "BWM" metrics. When
        :math:`\theta = 0` under "PEM", the limit corresponds to "LEM".
    input_tangent : bool, default=False
        If True, input is assumed to already reside in the tangent space
        (as symmetric matrices or pre-vectorized vectors).
    in_channels : int, default=1
        Number of input SPD channels in the product manifold
        :math:`(\mathcal{S}_{++}^n)^{c_{\text{in}}}`.
    alpha : float, default=1.0
        Weight of :math:`\operatorname{tr}(UV)` in the :math:`O(n)`-invariant
        tangent inner product :math:`\alpha \operatorname{tr}(UV) + \beta \operatorname{tr}(U)\operatorname{tr}(V)`.
    beta : float, default=0.0
        Weight of :math:`\operatorname{tr}(U)\operatorname{tr}(V)` in the
        :math:`O(n)`-invariant tangent inner product. Requires
        :math:`\beta > -\alpha / n`.
    unidirectional : bool, default=False
        If True, uses unidirectional geodesic hyperplanes passing through
        :math:`\operatorname{Exp}_I(\gamma_k \tilde{Z}_k)` as in RieTrans
        :cite:p:`chen2026rietrans`.
    act_power : float, optional
        Optional power deformation :math:`X \mapsto X^{\text{act\_power}}`
        applied before the tangent projection.
    device : torch.device, optional
        Device on which to allocate parameters.
    dtype : torch.dtype, optional
        Data type of parameters.

    Attributes
    ----------
    reference : nn.Parameter
        Learnable reference SPD matrix :math:`P \in \mathcal{S}_{++}^n` (when
        ``input_tangent=False`` and ``unidirectional=False``), parameterized via
        :class:`~spd_learn.modules.SymmetricPositiveDefinite`.
    linear : nn.Linear
        Linear projection layer from the vectorized tangent space of dimension
        :math:`c_{\text{in}} \cdot n(n+1)/2` to ``n_outputs``.

    References
    ----------
    .. bibliography::
       :filter: key in {"chen2024rmlr", "chen2026rietrans"}

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
        in_channels: int = 1,
        alpha: float = 1.0,
        beta: float = 0.0,
        unidirectional: bool = False,
        act_power: Optional[float] = None,
        eps: float = 1e-4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if metric not in VALID_RIETRANS_METRICS:
            raise ValueError(
                f"metric must be one of {VALID_RIETRANS_METRICS}, got '{metric}'"
            )
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if alpha <= 0.0 or beta <= -alpha / n_features:
            raise ValueError(
                f"Invalid (alpha, beta)=({alpha}, {beta}) for n_features={n_features}: "
                "require alpha > 0 and beta > -alpha / n_features."
            )

        self.n_features = n_features
        self.n_outputs = n_outputs
        self.metric = metric
        self.theta = theta
        self.input_tangent = input_tangent
        self.in_channels = in_channels
        self.alpha = alpha
        self.beta = beta
        self.unidirectional = unidirectional
        self.act_power = act_power
        self.eps = eps
        self.vec_dim = n_features * (n_features + 1) // 2

        self._p = math.sqrt(alpha + n_features * beta)
        self._q = math.sqrt(alpha)
        self._iso_diag_shift = (self._p - self._q) / n_features

        if not self.input_tangent and not self.unidirectional:
            eye = torch.eye(n_features, device=device, dtype=dtype)
            if in_channels > 1:
                eye = eye.unsqueeze(0).repeat(in_channels, 1, 1)
            self.reference = nn.Parameter(eye)
            register_parametrization(self, "reference", SymmetricPositiveDefinite())

        self.linear = nn.Linear(
            in_channels * self.vec_dim,
            n_outputs,
            bias=not unidirectional,
            device=device,
            dtype=dtype,
        )
        if unidirectional:
            self.gamma = nn.Parameter(
                torch.zeros(n_outputs, in_channels, device=device, dtype=dtype)
            )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset layer parameters to initial state."""
        if hasattr(self, "reference") and is_parametrized(self, "reference"):
            eye = torch.eye(
                self.n_features,
                device=self.linear.weight.device,
                dtype=self.linear.weight.dtype,
            )
            if self.in_channels > 1:
                eye = eye.unsqueeze(0).repeat(self.in_channels, 1, 1)
            setattr(self, "reference", eye)
        self.linear.reset_parameters()
        if hasattr(self, "gamma"):
            nn.init.zeros_(self.gamma)

    def _to_tangent(self, X: torch.Tensor) -> torch.Tensor:
        """Map SPD matrices to tangent space at the reference point or identity."""
        if self.act_power is not None and self.act_power != 1.0:
            X = matrix_power.apply(X, self.act_power)

        if self.unidirectional:
            if self.metric in ("LEM", "AIM") or (
                self.metric == "PEM" and self.theta == 0.0
            ):
                return matrix_log.apply(X)
            if self.metric == "PEM":
                eye = torch.eye(self.n_features, device=X.device, dtype=X.dtype)
                return (matrix_power.apply(X, self.theta) - eye) / self.theta
            if self.metric == "BWM":
                eye = torch.eye(self.n_features, device=X.device, dtype=X.dtype)
                if self.theta in (0.5, 1.0):
                    return 2.0 * (matrix_sqrt.apply(X) - eye)
                return (matrix_power.apply(X, 2.0 * self.theta) - eye) / abs(
                    self.theta
                )
            return cholesky_log.apply(X)

        P = self.reference
        if self.metric == "LEM" or (self.metric == "PEM" and self.theta == 0.0):
            return matrix_log.apply(X) - matrix_log.apply(P)
        if self.metric == "PEM":
            return (
                matrix_power.apply(X, self.theta) - matrix_power.apply(P, self.theta)
            ) / self.theta
        if self.metric == "AIM":
            P_inv_sqrt = matrix_inv_sqrt.apply(P)
            return matrix_log.apply(ensure_sym(P_inv_sqrt @ X @ P_inv_sqrt))
        if self.metric == "BWM":
            if self.theta in (0.5, 1.0):
                P_sqrt, P_inv_sqrt = matrix_sqrt_inv.apply(P)
                inter = ensure_sym(P_sqrt @ X @ P_sqrt)
                sqrt_PX = P_sqrt @ matrix_sqrt.apply(inter) @ P_inv_sqrt
                return sqrt_PX + sqrt_PX.mT - 2.0 * P
            P_pow = matrix_power.apply(P, self.theta)
            P_inv_pow = matrix_power.apply(P, -self.theta)
            X_pow = matrix_power.apply(X, 2.0 * self.theta)
            inter = ensure_sym(P_pow @ X_pow @ P_pow)
            sqrt_term = P_pow @ matrix_sqrt.apply(inter) @ P_inv_pow
            return (sqrt_term + sqrt_term.mT - 2.0 * (P_pow @ P_pow)) / (
                2.0 * abs(self.theta)
            )
        return cholesky_log.apply(X) - cholesky_log.apply(P)

    def _vectorize(self, V: torch.Tensor) -> torch.Tensor:
        """Vectorize tangent matrix under the configured metric and (alpha, beta) inner product."""
        if self.metric == "LCM":
            return sym_to_upper(V, preserve_norm=False, upper=False)
        V_sym = ensure_sym(V)
        if self.alpha != 1.0 or self.beta != 0.0:
            tr_V = torch.diagonal(V_sym, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            diag_shift = torch.diag_embed(
                tr_V.expand(*V_sym.shape[:-2], self.n_features) * self._iso_diag_shift
            )
            V_sym = self._q * V_sym + diag_shift
        return sym_to_upper(V_sym, preserve_norm=True, upper=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPDRMLR.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor. If ``input_tangent=False``, shape `(..., n_features, n_features)`
            or `(..., in_channels, n_features, n_features)`.
            If ``input_tangent=True``, shape `(..., n_features, n_features)`,
            `(..., in_channels, n_features, n_features)`, or `(..., in_channels * vec_dim)`.

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
                if self.in_channels > 1 and X.ndim == 3:
                    X = X.unsqueeze(-3).expand(
                        *X.shape[:-2],
                        self.in_channels,
                        self.n_features,
                        self.n_features,
                    )
                v = self._vectorize(X)
            else:
                v = X
        else:
            if self.in_channels > 1 and X.ndim == 3:
                X = X.unsqueeze(-3).expand(
                    *X.shape[:-2], self.in_channels, self.n_features, self.n_features
                )
            V = self._to_tangent(X)
            v = self._vectorize(V)

        if self.in_channels > 1 and v.shape[-1] == self.vec_dim:
            v = v.flatten(start_dim=-2)

        if not self.unidirectional:
            return self.linear(v)

        W = self.linear.weight.view(self.n_outputs, self.in_channels, self.vec_dim)
        if self.metric == "LCM":
            ixs = torch.tril_indices(self.n_features, self.n_features, device=W.device)
            d_mask = torch.where(ixs[0] == ixs[1], 0.5, 1.0).to(dtype=W.dtype)
            W_eff = W * d_mask
            W_norm = torch.linalg.norm(W, dim=-1).clamp_min(1e-8)
            W_eff_sq = (W_eff * W_eff).sum(dim=-1)
            shift = (self.gamma * (W_eff_sq / W_norm)).sum(dim=-1)
            return F.linear(v, W_eff.reshape(self.n_outputs, -1), -shift)

        W_norm = torch.linalg.norm(W, dim=-1)
        shift = (self.gamma * W_norm).sum(dim=-1)
        return F.linear(v, self.linear.weight, -shift)

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, n_outputs={self.n_outputs}, "
            f"metric='{self.metric}', theta={self.theta}, "
            f"in_channels={self.in_channels}, unidirectional={self.unidirectional}, "
            f"input_tangent={self.input_tangent}"
        )


class SPDLinear(nn.Module):
    r"""Intrinsic SPD Transformation Layer.

    Implements the intrinsic SPD transformation layer for Riemannian neural networks
    introduced in RieTrans :cite:p:`chen2026rietrans`.

    This layer transforms input SPD matrices in
    :math:`(\mathcal{S}_{++}^{d_{\text{in}}})^{c_{\text{in}}}` to output SPD
    matrices in :math:`(\mathcal{S}_{++}^{d_{\text{out}}})^{c_{\text{out}}}` (or
    symmetric tangent matrices in :math:`\mathcal{S}^{d_{\text{out}}}`). It
    leverages an internal :class:`SPDRMLR` module to project matrices into an
    unconstrained vectorized space, reconstructs the tangent matrix via the
    isometric :math:`(\alpha, \beta)` map, and applies the metric-specific
    inverse chart map :math:`\operatorname{V2SPD}`:

    - **LEM / AIM**: :math:`\exp(V)`
    - **LCM**: :math:`L L^\top` where :math:`L = \operatorname{tril}(V, -1) + \operatorname{diag}(\exp(\operatorname{diag}(V)))` in :math:`O(d_{\text{out}}^3)` without eigendecomposition
    - **PEM**: :math:`\operatorname{ReEig}(I + \theta V, \epsilon)^{1/\theta}`
    - **BWM**: :math:`\operatorname{ReEig}(I + \frac{1}{2}V, \epsilon)^2`

    Parameters
    ----------
    in_features : int
        Dimension of the input SPD matrices (:math:`d_{\text{in}}`).
    out_features : int
        Dimension of the output SPD matrices (:math:`d_{\text{out}}`).
    metric : {"LEM", "LCM", "AIM", "PEM", "BWM"}, default="LEM"
        Riemannian metric for the tangent space projection and inverse chart map.
    theta : float, default=1.0
        Deformation parameter for the "PEM" or "BWM" metrics.
    input_tangent : bool, default=False
        If True, expects input to already reside in the tangent space.
    return_tangent : bool, default=False
        If True, returns symmetric tangent matrices in :math:`\mathcal{S}^{d_{\text{out}}}`
        (chart bypass mode).
    in_channels : int, default=1
        Number of input product-SPD channels (:math:`c_{\text{in}}`).
    out_channels : int, default=1
        Number of output product-SPD channels (:math:`c_{\text{out}}`).
    alpha : float, default=1.0
        Weight of :math:`\operatorname{tr}(UV)` in the :math:`O(n)`-invariant
        inner product.
    beta : float, default=0.0
        Weight of :math:`\operatorname{tr}(U)\operatorname{tr}(V)` in the
        :math:`O(n)`-invariant inner product.
    unidirectional : bool, default=False
        If True, uses unidirectional geodesic hyperplanes in the internal RMLR.
    act_power : float, optional
        Optional power deformation :math:`X \mapsto X^{\text{act\_power}}`
        applied before tangent projection.
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
        in_channels: int = 1,
        out_channels: int = 1,
        alpha: float = 1.0,
        beta: float = 0.0,
        unidirectional: bool = False,
        act_power: Optional[float] = None,
        eps: float = 1e-4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")
        if alpha <= 0.0 or beta <= -alpha / out_features:
            raise ValueError(
                f"Invalid (alpha, beta)=({alpha}, {beta}) for out_features={out_features}: "
                "require alpha > 0 and beta > -alpha / out_features."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.metric = metric
        self.theta = theta
        self.input_tangent = input_tangent
        self.return_tangent = return_tangent
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.beta = beta
        self.unidirectional = unidirectional
        self.act_power = act_power

        self._out_p = math.sqrt(alpha + out_features * beta)
        self._out_q = math.sqrt(alpha)
        self._out_mu = (1.0 / self._out_p - 1.0 / self._out_q) / out_features

        self.out_vec_dim = out_features * (out_features + 1) // 2
        self.rmlr = SPDRMLR(
            n_features=in_features,
            n_outputs=out_channels * self.out_vec_dim,
            metric=metric,
            theta=theta,
            input_tangent=input_tangent,
            in_channels=in_channels,
            alpha=alpha,
            beta=beta,
            unidirectional=unidirectional,
            act_power=act_power,
            device=device,
            dtype=dtype,
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        """Initialize tangent projection with principal-submatrix identity plus scaled noise."""
        in_vec_dim = self.in_features * (self.in_features + 1) // 2
        fan = self.in_channels * in_vec_dim * self.out_channels * self.out_vec_dim
        std = (
            1e-3
            if self.metric in ("AIM", "BWM")
            else min(1e-2, math.sqrt(0.5 / max(fan, 1)))
        )
        W = self.rmlr.linear.weight.view(
            self.out_channels, self.out_vec_dim, self.in_channels, in_vec_dim
        )
        nn.init.normal_(W, mean=0.0, std=std)

        m = min(self.in_features, self.out_features)
        if self.metric == "LCM":
            in_ixs = torch.tril_indices(
                self.in_features, self.in_features, device=W.device
            )
            out_ixs = torch.tril_indices(
                self.out_features, self.out_features, device=W.device
            )
        else:
            in_ixs = torch.triu_indices(
                self.in_features, self.in_features, device=W.device
            )
            out_ixs = torch.triu_indices(
                self.out_features, self.out_features, device=W.device
            )
        in_mask = (in_ixs[0] < m) & (in_ixs[1] < m)
        out_mask = (out_ixs[0] < m) & (out_ixs[1] < m)
        in_order = torch.argsort(in_ixs[0, in_mask] * m + in_ixs[1, in_mask])
        out_order = torch.argsort(out_ixs[0, out_mask] * m + out_ixs[1, out_mask])
        k_in = torch.where(in_mask)[0][in_order]
        k_out = torch.where(out_mask)[0][out_order]
        W[:, k_out, :, k_in] += 1.0 / self.in_channels

        if self.rmlr.linear.bias is not None:
            nn.init.zeros_(self.rmlr.linear.bias)
        if hasattr(self.rmlr, "gamma"):
            nn.init.zeros_(self.rmlr.gamma)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset layer parameters to initial state."""
        self.rmlr.reset_parameters()
        self._init_weights()

    def _vec_to_tangent(self, v: torch.Tensor) -> torch.Tensor:
        """Reconstruct symmetric tangent matrix from vectorized coordinates (v2V)."""
        if self.metric == "LCM":
            return vec_to_sym(v, preserve_norm=False, upper=False)
        Y_tan = vec_to_sym(v, preserve_norm=True, upper=True)
        if self.alpha != 1.0 or self.beta != 0.0:
            tr_Y = torch.diagonal(Y_tan, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            diag_shift = torch.diag_embed(
                tr_Y.expand(*Y_tan.shape[:-2], self.out_features) * self._out_mu
            )
            Y_tan = (1.0 / self._out_q) * Y_tan + diag_shift
        return Y_tan

    def _from_tangent(self, Y_tan: torch.Tensor) -> torch.Tensor:
        """Map tangent matrix back to the SPD manifold via metric-specific V2SPD."""
        if self.return_tangent:
            return Y_tan
        if self.metric == "LCM":
            # ponytail: [-8, 8] log-Cholesky diagonal clamp caps condition number at exp(16), widen if extreme dynamic range is required
            log_diag = torch.diagonal(Y_tan, dim1=-2, dim2=-1).clamp(min=-8.0, max=8.0)
            Y_safe = Y_tan.tril(-1) + torch.diag_embed(log_diag)
            eps = get_epsilon(Y_tan.dtype, "eigval_log")
            eye = torch.eye(self.out_features, device=Y_tan.device, dtype=Y_tan.dtype)
            return cholesky_exp.apply(Y_safe) + eps * eye
        if self.metric == "PEM" and self.theta != 0.0:
            eye = torch.eye(self.out_features, device=Y_tan.device, dtype=Y_tan.dtype)
            M = ensure_sym(eye + self.theta * Y_tan)
            if self.theta == 1.0:
                eps = get_epsilon(Y_tan.dtype, "eigval_power")
                return clamp_eigvals.apply(M, eps)
            return matrix_power.apply(M, 1.0 / self.theta)
        if self.metric == "BWM":
            eye = torch.eye(self.out_features, device=Y_tan.device, dtype=Y_tan.dtype)
            eps = get_epsilon(Y_tan.dtype, "eigval_power")
            tmp = clamp_eigvals.apply(ensure_sym(eye + 0.5 * Y_tan), eps)
            return tmp @ tmp
        return matrix_exp.apply(Y_tan)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of SPDLinear.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD tensor of shape `(..., in_features, in_features)` or
            `(..., in_channels, in_features, in_features)` (or tangent tensor
            if ``input_tangent=True``).

        Returns
        -------
        torch.Tensor
            Transformed SPD tensor of shape `(..., out_features, out_features)`
            (or `(..., out_channels, out_features, out_features)` when
            ``out_channels > 1``), or tangent matrix if ``return_tangent=True``.
        """
        v = self.rmlr(X)
        if self.out_channels > 1:
            v = v.view(*v.shape[:-1], self.out_channels, self.out_vec_dim)
        Y_tan = self._vec_to_tangent(v)
        return self._from_tangent(Y_tan)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"metric='{self.metric}', theta={self.theta}, "
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"unidirectional={self.unidirectional}, "
            f"input_tangent={self.input_tangent}, return_tangent={self.return_tangent}"
        )
