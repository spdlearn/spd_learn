# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch

from torch import nn

from ..functional import ensure_sym, matrix_exp, matrix_log
from ..functional.core import (
    inv_softplus,
    matrix_inv_softplus,
    matrix_softplus,
    softplus,
)


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
        Note: This parametrization class has no parameters or buffers,
        so device is accepted for API consistency but not used.
    dtype : torch.dtype, optional
        Data type for the module. Default is None.
        Note: This parametrization class has no parameters or buffers,
        so dtype is accepted for API consistency but not used.

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
        return self._spd_fun(ensure_sym(X))

    def right_inverse(self, X):
        """Map from the SPD manifold onto the tangent space at identity.

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
        return self._tangent_fun(X)


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
