# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Literal, cast

import torch

from torch import nn
from torch.nn.utils import parametrize

from ..functional import ensure_sym, matrix_exp, matrix_log
from ..functional.core import (
    inv_softplus,
    matrix_inv_softplus,
    matrix_softplus,
    softplus,
)
from .bilinear import BiMap


class SymmetricPositiveDefinite(nn.Module):
    """Symmetric Positive Definite Manifold parametrization.

    This module projects a matrix onto the SPD manifold by first ensuring
    it is symmetric and then applying either the matrix exponential or
    the matrix softplus function.

    Parameters
    ----------
    mapping : str, optional
        Mapping from symmetric matrices to SPD matrices.
        Default is "exp".
        Options are: "exp" and "softplus"
    device : torch.device, optional
        Device for the module. Default is None.
        Accepted for API consistency; the base follows assigned tensors.
    dtype : torch.dtype, optional
        Data type for the module. Default is None.
        Accepted for API consistency; the base follows assigned tensors.
    use_trivialization : bool, default=False
        Enable a moving exponential chart. Assignment stores a Cholesky base
        factor and resets tangent coordinates. Only supported for ``"exp"``.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import SymmetricPositiveDefinite
    >>> # Using matrix exponential (default)
    >>> spd_exp = SymmetricPositiveDefinite(mapping="exp")
    >>> X = torch.randn(2, 3, 3)
    >>> S = spd_exp(X)  # Projects to SPD manifold
    >>> # Using softplus mapping
    >>> spd_softplus = SymmetricPositiveDefinite(mapping="softplus")
    >>> S = spd_softplus(X)
    """

    def __init__(
        self, mapping="exp", device=None, dtype=None, *, use_trivialization=False
    ) -> None:
        super().__init__()
        if mapping not in ["softplus", "exp"]:
            raise ValueError(f"mapping must be 'softplus' or 'exp', got '{mapping}'")
        if use_trivialization and mapping != "exp":
            raise ValueError("Dynamic trivialization requires mapping='exp'")
        self.mapping = mapping
        self.use_trivialization = use_trivialization
        self.register_buffer("base_factor", None)
        self.device = device
        self.dtype = dtype
        if self.mapping == "softplus":
            self._spd_fun = matrix_softplus.apply
            self._tangent_fun = matrix_inv_softplus.apply
        else:  # exp
            self._spd_fun = matrix_exp.apply
            self._tangent_fun = matrix_log.apply

    def forward(self, X):
        """Forward pass projecting input onto the SPD manifold.

        Parameters
        ----------
        X : torch.Tensor
            Input matrix of shape (..., n, n).

        Returns
        -------
        torch.Tensor
            The projected SPD matrix of shape (..., n, n).
        """
        output = self._spd_fun(ensure_sym(X))
        if self.use_trivialization and self.base_factor is not None:
            output = self.base_factor @ output @ self.base_factor.mT
        return ensure_sym(output) if self.use_trivialization else output

    def right_inverse(self, X):
        """Map from SPD matrices to coordinates, rebasing if enabled.

        This is useful for initializing parameters from SPD matrices
        when using this module as a parametrization.

        Parameters
        ----------
        X : torch.Tensor
            Input SPD matrix of shape (..., n, n).

        Returns
        -------
        torch.Tensor
            The corresponding tangent symmetric matrix.
        """
        if self.use_trivialization:
            # PyTorch validates returned coordinates only after this method runs.
            if self.base_factor is not None and (
                X.shape != self.base_factor.shape
                or X.dtype != self.base_factor.dtype
                or X.device != self.base_factor.device
            ):
                raise ValueError(
                    "SPD assignment must match the base shape, dtype and device; use .to() to change dtype/device"
                )
            if X.ndim < 2 or X.shape[-2] != X.shape[-1]:
                raise ValueError("Expected square SPD matrices")
            if not torch.isfinite(X).all() or not torch.allclose(X, X.mT):
                raise ValueError("Expected finite symmetric SPD matrices")
            with torch.no_grad():
                self.base_factor = torch.linalg.cholesky(ensure_sym(X))
            return torch.zeros_like(X)
        return self._tangent_fun(X)


def _recenter_orthogonal(
    orth: nn.Module, original: torch.Tensor
) -> tuple[bool, torch.Tensor | None]:
    """Recenter a BiMap orthogonal parametrization by horizontal group multiplication."""
    X = original
    n, k = X.size(-2), X.size(-1)
    orth_map = orth.orthogonal_map.name
    old_base = orth.base.detach().clone()
    if orth_map in ("matrix_exp", "cayley"):
        X_tril = X.tril()
        if n != k:
            X_tril = torch.cat(
                [
                    X_tril,
                    X_tril.new_zeros(n, n - k).expand(*X_tril.shape[:-2], -1, -1),
                ],
                dim=-1,
            )
        A = X_tril - X_tril.mH
        if orth_map == "matrix_exp":
            U = torch.matrix_exp(A)
        else:
            Id = torch.eye(n, dtype=A.dtype, device=A.device)
            U = torch.linalg.solve(
                torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5)
            )
    else:
        signs = X.diagonal(dim1=-2, dim2=-1).int()
        if (signs.abs() != 1).any():
            return False, None
        A = X.tril(diagonal=-1)
        tau = 2.0 / (1.0 + (A * A).sum(dim=-2))
        if n != k:
            A = torch.cat(
                [A, A.new_zeros(n, n - k).expand(*A.shape[:-2], -1, -1)], dim=-1
            )
            tau = torch.cat(
                [tau, tau.new_zeros(n - k).expand(*tau.shape[:-1], -1)], dim=-1
            )
            signs = torch.cat(
                [signs, signs.new_ones(n - k).expand(*signs.shape[:-1], -1)], dim=-1
            )
        U = torch.linalg.householder_product(A, tau) * signs.unsqueeze(-2)
    q, r = torch.linalg.qr(orth.base @ U)
    orth.base = q * r.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    original.zero_()
    original.diagonal(dim1=-2, dim2=-1).fill_(-1.0)
    return True, old_base.mH @ orth.base


def _update_optimizer_state(
    optimizer: torch.optim.Optimizer,
    original: torch.Tensor,
    optimizer_state: Literal["reset", "transport", "keep"],
    *,
    is_spd: bool,
    old_factor: torch.Tensor | None = None,
    new_factor: torch.Tensor | None = None,
    q_rel: torch.Tensor | None = None,
    orth_map: str = "cayley",
) -> None:
    if optimizer_state == "reset":
        optimizer.state.pop(original, None)
        return
    if optimizer_state == "keep" or original not in optimizer.state:
        return
    state = optimizer.state[original]
    n, k = original.size(-2), original.size(-1)
    for key in ("exp_avg", "momentum_buffer"):
        buf = state.get(key)
        if not isinstance(buf, torch.Tensor) or buf.shape != original.shape:
            continue
        if is_spd and new_factor is not None:
            rel = (
                new_factor
                if old_factor is None
                else torch.linalg.solve_triangular(old_factor, new_factor, upper=False)
            )
            buf.copy_(ensure_sym(rel.mH @ buf @ rel))
        elif not is_spd and q_rel is not None:
            if orth_map in ("matrix_exp", "cayley"):
                m_tril = buf.tril(diagonal=-1)
                if n != k:
                    m_tril = torch.cat(
                        [
                            m_tril,
                            m_tril.new_zeros(n, n - k).expand(
                                *m_tril.shape[:-2], -1, -1
                            ),
                        ],
                        dim=-1,
                    )
                skew = q_rel.mH @ (m_tril - m_tril.mH) @ q_rel
                buf.copy_(skew[..., :, :k].tril(diagonal=-1))
            else:
                buf.diagonal(dim1=-2, dim2=-1).zero_()


@torch.no_grad()
def step_dynamic_trivialization(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    threshold: float = 0.0,
    optimizer_state: Literal["reset", "transport", "keep"] = "reset",
) -> None:
    """Recenter native BiMap and exponential SPD charts, enabling SPD on first call.

    Call after K completed optimizer updates and backward, outside
    ``parametrize.cached()``. Effective values and Parameter identities remain
    unchanged, including frozen parameters. Unsupported target chains raise.
    Nonparametrized BiMaps, softplus and positive scalars stay unchanged.

    Recentering is deterministic and preserves the horizontal Stiefel frame
    without consuming RNG state. Gradients of recentered Parameters are cleared.

    Parameters
    ----------
    model : nn.Module
        Model containing parametrized :class:`~spd_learn.modules.BiMap` or
        :class:`~spd_learn.modules.SymmetricPositiveDefinite` modules.
    optimizer : torch.optim.Optimizer or None, default=None
        Optimizer whose state should be updated according to ``optimizer_state``.
    threshold : float, default=0.0
        Minimum tangent displacement norm required to trigger recentering for a
        given parameter. When ``0.0`` (default), all eligible parameters are
        recentered unconditionally.
    optimizer_state : {"reset", "transport", "keep"}, default="reset"
        How to handle ``optimizer.state[original]`` when ``optimizer`` is
        provided: ``"transport"`` parallel-transports first-moment buffers
        (``exp_avg`` / ``momentum_buffer``) into the new tangent frame while
        preserving ``step`` and ``exp_avg_sq``; ``"keep"`` leaves state
        untouched; ``"reset"`` clears ``optimizer.state[original]``.

    Notes
    -----
    - **Single-subject / cross-session EEG (short travel):** When weights stay
      close to initialization (:math:`\\lesssim 25^\\circ`), static
      trivialization (:math:`K = \\infty`, the default) is already nearly
      isometric, so dynamic updates are not needed.
    - **Cross-subject, continual adaptation, or long-horizon training (large
      travel):** When covariance shifts push weights far from initialization,
      periodic recentering (:math:`K \\in [10, 50]` or ``threshold > 0``)
      prevents differential distortion. Just as
      :class:`~spd_learn.modules.SPDBatchNormMeanVar` uses parallel transport
      (:math:`G^{-1/2} P G^{-1/2}`) to re-center batch covariances to the
      identity, each dynamic trivialization step updates the base point
      (:math:`B \\leftarrow \\phi_B(A), A \\leftarrow 0`) and transports the
      parameter tangent space back to the identity (:math:`\\Gamma_{B \\to I}`),
      acting as Riemannian metric preconditioning.
    """
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0.0, got {threshold}")
    if optimizer_state not in ("reset", "transport", "keep"):
        raise ValueError(
            "optimizer_state must be 'reset', 'transport', or 'keep', "
            f"got '{optimizer_state}'"
        )
    targets = []
    for module in model.modules():
        if not parametrize.is_parametrized(module):
            continue
        for name, parametrization in cast(
            nn.ModuleDict, module.parametrizations
        ).items():
            chain = cast(parametrize.ParametrizationList, parametrization)
            is_bimap = isinstance(module, BiMap) and name == "weight"
            is_spd = any(
                isinstance(p, SymmetricPositiveDefinite) and p.mapping == "exp"
                for p in chain
            )
            if not (is_bimap or is_spd):
                continue
            if len(chain) != 1:
                raise ValueError(
                    "Dynamic trivialization requires a single parametrization"
                )
            if is_bimap and not hasattr(chain[0], "base"):
                raise ValueError("BiMap requires use_trivialization=True")
            targets.append((module, name, chain, is_spd))

    for module, name, chain, is_spd in targets:
        original = cast(torch.Tensor, chain.original)
        travel = float(
            (ensure_sym(original) if is_spd else original.tril(diagonal=-1))
            .norm()
            .item()
        )
        if threshold > 0.0 and travel <= threshold:
            continue
        if is_spd:
            spd_param = cast(SymmetricPositiveDefinite, chain[0])
            old_factor = (
                spd_param.base_factor.detach().clone()
                if spd_param.use_trivialization and spd_param.base_factor is not None
                else None
            )
            spd_param.use_trivialization = True
            setattr(module, name, getattr(module, name))
            original.grad = None
            if optimizer is not None:
                _update_optimizer_state(
                    optimizer,
                    original,
                    optimizer_state,
                    is_spd=True,
                    old_factor=old_factor,
                    new_factor=spd_param.base_factor,
                )
        else:
            orth_param = chain[0]
            ok, q_rel = _recenter_orthogonal(orth_param, original)
            if not ok:
                setattr(module, name, getattr(module, name))
            original.grad = None
            if optimizer is not None:
                _update_optimizer_state(
                    optimizer,
                    original,
                    optimizer_state,
                    is_spd=False,
                    q_rel=q_rel,
                    orth_map=orth_param.orthogonal_map.name,
                )


class PositiveDefiniteScalar(nn.Module):
    """Positive definite scalars parametrization.

    This module projects real scalars onto the space of positive definite scalars
    by applying either the exponential or SoftPlus functions.

    Parameters
    ----------
    mapping : str, optional
        Mapping from real scalars to positive definite scalars.
        Default is "exp".
        Options are: "exp" and "softplus"
    device : torch.device, optional
        Device for the module. Default is None.
        Note: This parametrization class has no parameters or buffers,
        so device is accepted for API consistency but not used.
    dtype : torch.dtype, optional
        Data type for the module. Default is None.
        Note: This parametrization class has no parameters or buffers,
        so dtype is accepted for API consistency but not used.
    """

    def __init__(self, mapping="exp", device=None, dtype=None) -> None:
        super().__init__()
        if mapping not in ["softplus", "exp"]:
            raise ValueError(f"mapping must be 'softplus' or 'exp', got '{mapping}'")
        self.mapping = mapping
        # device and dtype are accepted for API consistency but not used
        # since this parametrization class has no parameters or buffers
        self.device = device
        self.dtype = dtype
        if self.mapping == "softplus":
            self._pd_fun = softplus
            self._tangent_fun = inv_softplus
        else:  # exp
            self._pd_fun = torch.exp
            self._tangent_fun = torch.log

    def forward(self, s):
        """Forward pass projecting input onto positive definite scalars.

        Parameters
        ----------
        s : torch.Tensor of shape ()
            Real scalar

        Returns
        -------
        torch.Tensor of shape ()
            Positive definite scalar
        """
        return self._pd_fun(s)

    def right_inverse(self, s):
        """Map from positive definite scalar onto real scalars.

        Parameters
        ----------
        s : torch.Tensor of shape ()
            Positive definite scalar

        Returns
        -------
        torch.Tensor of shape ()
            Real scalar
        """
        return self._tangent_fun(s)
