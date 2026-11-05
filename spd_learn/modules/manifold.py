# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import cast

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


@torch.no_grad()
def step_dynamic_trivialization(
    model: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> None:
    """Recenter native BiMap and exponential SPD charts, enabling SPD on first call.

    Call after K completed optimizer updates and backward, outside
    ``parametrize.cached()``. Effective values and Parameter identities remain
    unchanged, including frozen parameters. Unsupported target chains raise.
    Nonparametrized BiMaps, softplus and positive scalars stay unchanged.

    Passing ``optimizer`` restarts only recentered Parameters' state; their
    gradients are always cleared. Omit only before training or for stateless
    optimizers. Enable a fresh model before loading dynamic checkpoint bases;
    save the update count/RNG separately to resume K and rectangular QR.
    """
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
        if is_spd:
            cast(SymmetricPositiveDefinite, chain[0]).use_trivialization = True
        setattr(module, name, getattr(module, name))
        original = cast(torch.Tensor, chain.original)
        original.grad = None
        if optimizer is not None:
            optimizer.state.pop(original, None)


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
