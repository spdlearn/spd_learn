# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Lie Group Batch Normalization for SPD matrices.

This module implements SPDBatchNormLie based on:
Ziheng Chen, Yue Song, Yunmei Liu, and Nicu Sebe,
"A Lie Group Approach to Riemannian Batch Normalization," ICLR 2024.

The implementation is integrated into ``spd_learn`` from the original LieBN
repository: https://github.com/GitZH-Chen/LieBN/tree/main/LieBN
"""

import torch

from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from ..functional import (
    airm_geodesic,
    ensure_sym,
    matrix_exp,
    matrix_inv_sqrt,
    matrix_log,
    matrix_power,
    matrix_sqrt,
)
from ..functional.batchnorm import (
    frechet_mean,
    lie_group_variance,
    spd_centering,
    spd_cholesky_congruence,
    spd_rebiasing,
)
from ..functional.metrics import (
    cholesky_exp,
    cholesky_log,
    log_euclidean_scalar_multiply,
)
from .manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite


class SPDBatchNormLie(nn.Module):
    r"""Lie Group Batch Normalization for SPD matrices.

    Implements the LieBN framework :cite:p:`chen2024liebn` for SPD manifolds.
    Unlike :class:`SPDBatchNormMeanVar`, which normalizes under a single
    Riemannian metric (AIRM), this layer exploits the **Lie group structure**
    of three classical SPD geometries to define centering, scaling, and biasing
    as group-theoretic operations with formal statistical guarantees.

    **Algorithm.**
    Given a batch :math:`\{P_i\}_{i=1}^N \subset \mathcal{S}_{++}^n`, the
    forward pass applies three steps in the Lie algebra selected by ``metric``:

    1. **Centering** -- translate the batch mean :math:`M` to the group
       identity :math:`E` via the inverse left translation:

       .. math::

           \bar{P}_i = L_{M_\odot^{-1}}(P_i)

    2. **Scaling** -- normalize the Fréchet variance :math:`v^2` with a
       learnable shift :math:`s \in \mathbb{R}_{>0}`:

       .. math::

           \hat{P}_i = \operatorname{Exp}_E
           \!\left[\frac{s}{\sqrt{v^2 + \epsilon}}\,
           \operatorname{Log}_E(\bar{P}_i)\right]

    3. **Biasing** -- translate to the learnable SPD parameter :math:`B`:

       .. math::

           \tilde{P}_i = L_B(\hat{P}_i)

    **Theoretical guarantees** (Proposition 4.2 of the paper):

    * *Mean control*: after centering and biasing with :math:`B = E`,
      the Fréchet mean of the output batch equals :math:`E`.
    * *Variance control*: after scaling, the output dispersion satisfies
      :math:`\sum_i w_i\,d^2(\hat{P}_i, E) = s^2`.

    **Supported metrics.**
    The ``metric`` parameter selects one of three Lie group structures, each
    inducing a family of parameterized metrics via the power deformation
    :math:`\mathrm{P}_\theta`.  The table below summarizes how each step is
    realized (see Table 2 in :cite:p:`chen2024liebn`):

    .. list-table::
       :header-rows: 1
       :widths: 25 25 25 25

       * - Operation
         - :math:`(\theta,\alpha,\beta)`-AIM
         - :math:`(\alpha,\beta)`-LEM
         - :math:`\theta`-LCM
       * - Pullback map
         - :math:`\mathrm{P}_\theta`
         - :math:`\operatorname{mlog}`
         - :math:`\psi_{\mathrm{LC}} \circ \mathrm{P}_\theta`
       * - Left translation :math:`L_Q(P)`
         - :math:`Q^{1/2} P\, Q^{1/2}`
         - :math:`P + Q`
         - :math:`P + Q`
       * - Scaling
         - :math:`\operatorname{Exp}_I[s\,\operatorname{Log}_I(P)]`
         - :math:`s \cdot P`
         - :math:`s \cdot P`
       * - Fréchet mean
         - Karcher flow
         - Arithmetic mean
         - Arithmetic mean
       * - Running mean update
         - AIRM geodesic
         - Linear interpolation
         - Linear interpolation

    **Bi-invariant distance.**
    The Fréchet variance uses the :math:`(\alpha, \beta)` bi-invariant metric
    (Definition 3 and Eq. 3 of the paper):

    .. math::

        d^2(P, Q) = \alpha \lVert V \rVert_F^2
                   + \beta \, g(V)^2

    where :math:`V` is the tangent representation (log-map) and
    :math:`g(V) = \log\det(P)` for AIM or :math:`\operatorname{tr}(V)`
    for LEM/LCM.  The variance is normalized by :math:`\theta^2` for AIM
    and LCM.

    Parameters
    ----------
    num_features : int
        Size of the SPD matrices (:math:`n \times n`).
    metric : {"AIM", "LEM", "LCM"}, default="AIM"
        Lie group invariant metric.
    theta : float, default=1.0
        Power deformation parameter :math:`\theta`.  When
        :math:`\theta = 1`, no deformation is applied.
    alpha : float, default=1.0
        Frobenius norm weight :math:`\alpha` in the bi-invariant distance.
    beta : float, default=0.0
        Trace / log-determinant weight :math:`\beta` in the bi-invariant
        distance.  Must satisfy :math:`\min(\alpha, \alpha + n\beta) > 0`.
    momentum : float, default=0.1
        Momentum :math:`\gamma` for exponential moving average of running
        statistics.
    eps : float, default=1e-5
        Numerical stability constant :math:`\epsilon` added to the variance
        before taking the square root.
    n_iter : int, default=1
        Number of Karcher flow iterations for the AIM Fréchet mean.
        Ignored by LEM and LCM (which use arithmetic means).
    congruence : {"cholesky", "eig"}, default="cholesky"
        Implementation of the AIM congruence action (centering/biasing).
        ``"cholesky"`` uses the Cholesky factor :math:`L` of :math:`P` to
        compute :math:`L X L^\top` (as in the original LieBN paper).
        ``"eig"`` uses eigendecomposition-based :math:`M^{-1/2} X M^{-1/2}`
        (matching :func:`~spd_learn.functional.spd_centering`).
        Both are mathematically equivalent; Cholesky is typically faster,
        while eigendecomposition reuses the infrastructure of
        :class:`~spd_learn.modules.SPDBatchNormMeanVar`.
        Only affects the AIM metric.
    device : torch.device or str, optional
        Device on which to create parameters and buffers.
    dtype : torch.dtype, optional
        Data type of parameters and buffers.

    Attributes
    ----------
    bias : nn.Parameter
        Learnable SPD bias matrix :math:`B \in \mathcal{S}_{++}^n`,
        parametrized via :class:`~spd_learn.modules.SymmetricPositiveDefinite`.
        Initialized to the identity.
    shift : nn.Parameter
        Learnable positive scalar :math:`s > 0`,
        parametrized via :class:`~spd_learn.modules.PositiveDefiniteScalar`.
        Initialized to 1.
    running_mean : torch.Tensor
        Exponential moving average of the batch Fréchet mean.
    running_var : torch.Tensor
        Exponential moving average of the batch variance.

    See Also
    --------
    :class:`SPDBatchNormMean` :
        Mean-only Riemannian batch normalization (AIRM centering without
        variance normalization) :cite:p:`brooks2019riemannian`.
    :class:`SPDBatchNormMeanVar` :
        Full Riemannian batch normalization under the AIRM
        :cite:p:`kobler2022spd`.
    :func:`~spd_learn.functional.frechet_mean` :
        Fréchet mean via Karcher flow (used internally for AIM).
    :func:`~spd_learn.functional.lie_group_variance` :
        Bi-invariant Fréchet variance computation.

    References
    ----------
    .. bibliography::
       :filter: key == "chen2024liebn"

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import SPDBatchNormLie
    >>> bn = SPDBatchNormLie(num_features=4, metric="AIM")
    >>> X = torch.randn(8, 4, 4, dtype=torch.float64)
    >>> X = X @ X.mT + 0.1 * torch.eye(4, dtype=torch.float64)
    >>> bn = bn.to(dtype=torch.float64)
    >>> Y = bn(X)
    >>> Y.shape
    torch.Size([8, 4, 4])
    """

    def __init__(
        self,
        num_features,
        metric="AIM",
        theta=1.0,
        alpha=1.0,
        beta=0.0,
        momentum=0.1,
        eps=1e-5,
        n_iter=1,
        congruence="cholesky",
        device=None,
        dtype=None,
    ):
        super().__init__()
        supported_metrics = ("AIM", "LEM", "LCM")
        if metric not in supported_metrics:
            raise ValueError(
                f"metric must be one of {supported_metrics}, got '{metric}'"
            )
        if congruence not in ("cholesky", "eig"):
            raise ValueError(
                f"congruence must be 'cholesky' or 'eig', got '{congruence}'"
            )
        self.num_features = num_features
        self.metric = metric
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.momentum = momentum
        self.eps = eps
        self.n_iter = n_iter
        self.congruence = congruence

        self.bias = nn.Parameter(
            torch.empty(1, num_features, num_features, device=device, dtype=dtype)
        )
        self.shift = nn.Parameter(torch.empty((), device=device, dtype=dtype))

        if metric == "AIM":
            self.register_buffer(
                "running_mean",
                torch.eye(num_features, device=device, dtype=dtype).unsqueeze(0),
            )
        else:
            self.register_buffer(
                "running_mean",
                torch.zeros(1, num_features, num_features, device=device, dtype=dtype),
            )
        self.register_buffer("running_var", torch.ones((), device=device, dtype=dtype))

        self.reset_parameters()
        self._parametrize()

    @torch.no_grad()
    def reset_parameters(self):
        self.bias.zero_()
        self.bias[0].fill_diagonal_(1.0)
        self.shift.fill_(1.0)

    def _parametrize(self):
        register_parametrization(self, "bias", SymmetricPositiveDefinite())
        register_parametrization(self, "shift", PositiveDefiniteScalar())

    # ------------------------------------------------------------------
    # Thin dispatch helpers — each delegates to existing functional ops.
    # ------------------------------------------------------------------

    def _deform(self, X):
        """Map SPD matrices to the Lie algebra."""
        if self.metric == "AIM":
            return X if self.theta == 1.0 else matrix_power.apply(X, self.theta)
        if self.metric == "LEM":
            return matrix_log.apply(X)
        # LCM
        Xp = X if self.theta == 1.0 else matrix_power.apply(X, self.theta)
        return cholesky_log.apply(Xp)

    def _inv_deform(self, S):
        """Map from the Lie algebra back to SPD matrices."""
        if self.metric == "AIM":
            return S if self.theta == 1.0 else matrix_power.apply(S, 1.0 / self.theta)
        if self.metric == "LEM":
            return matrix_exp.apply(S)
        # LCM
        spd = ensure_sym(cholesky_exp.apply(S))
        return spd if self.theta == 1.0 else matrix_power.apply(spd, 1.0 / self.theta)

    def _translate(self, X, P, inverse=False):
        """Group translation (centering / biasing) in the Lie algebra."""
        if self.metric == "AIM":
            if self.congruence == "cholesky":
                return spd_cholesky_congruence(X, P, inverse=inverse)
            # Eigendecomposition path
            if inverse:
                return spd_centering(X, matrix_inv_sqrt.apply(P))
            return spd_rebiasing(X, matrix_sqrt.apply(P))
        return X - P if inverse else X + P

    def _frechet_mean(self, X_def):
        """Fréchet mean in the deformed space."""
        if self.metric == "AIM":
            return frechet_mean(X_def, max_iter=self.n_iter)
        return X_def.detach().mean(dim=0, keepdim=True)

    def _scale(self, X, var):
        """Variance normalization in the Lie algebra."""
        factor = self.shift / (var + self.eps).sqrt()
        if self.metric == "AIM":
            return log_euclidean_scalar_multiply(factor, X)
        return X * factor

    # ------------------------------------------------------------------
    # Running statistics & forward
    # ------------------------------------------------------------------

    def _update_running_stats(self, batch_mean, batch_var):
        with torch.no_grad():
            if self.metric == "AIM":
                self.running_mean = airm_geodesic(
                    self.running_mean, batch_mean, self.momentum
                )
            else:
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

    def forward(self, X):
        X_def = self._deform(X)
        bias_def = self._deform(self.bias)

        if self.training:
            batch_mean = self._frechet_mean(X_def)
            X_centered = self._translate(X_def, batch_mean, inverse=True)
            if X.shape[0] > 1:
                batch_var = lie_group_variance(
                    X_centered.detach(),
                    self.metric,
                    self.alpha,
                    self.beta,
                    self.theta,
                )
                X_scaled = self._scale(X_centered, batch_var)
            else:
                batch_var = self.running_var.clone()
                X_scaled = X_centered
            self._update_running_stats(batch_mean.detach(), batch_var.detach())
        else:
            X_centered = self._translate(X_def, self.running_mean, inverse=True)
            X_scaled = self._scale(X_centered, self.running_var)

        X_biased = self._translate(X_scaled, bias_def, inverse=False)
        return self._inv_deform(X_biased)

    def extra_repr(self):
        return (
            f"num_features={self.num_features}, metric={self.metric}, "
            f"theta={self.theta}, alpha={self.alpha}, beta={self.beta}, "
            f"momentum={self.momentum}, congruence={self.congruence}"
        )
