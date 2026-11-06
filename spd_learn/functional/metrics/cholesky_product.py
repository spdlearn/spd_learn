# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Power-Cholesky and Bures-Wasserstein-Cholesky product metrics on SPD manifolds.

Implements the Power-Cholesky Metric (PCM) and Bures-Wasserstein-Cholesky
Metric (BWCM) for symmetric positive definite (SPD) matrices :cite:p:`chen2026fast`.
Decomposes the Cholesky factor :math:`X = LL^\top` into strictly lower
triangular part :math:`\lfloor L \rfloor` and diagonal :math:`d = \text{diag}(L)`.
"""

from typing import Optional, Union

import torch

from pyriemann.geometry.distance import distance_euclid
from pyriemann.geometry.geodesic import geodesic_euclid
from pyriemann.geometry.mean import mean_euclid

from ..utils import ensure_sym


def pcm_coords(X: torch.Tensor, theta: float = 0.5) -> torch.Tensor:
    r"""Compute global deformed coordinates under Power-Cholesky geometry.

    .. math::

        \tilde{\phi}_\theta(X) = \lfloor L \rfloor + \text{Diag}\left(\frac{1}{\theta} d^\theta\right)

    Parameters
    ----------
    X : torch.Tensor
        SPD matrices of shape `(..., n, n)`.
    theta : float, default=0.5
        Power deformation parameter (:math:`\theta \neq 0`).

    Returns
    -------
    torch.Tensor
        Deformed coordinates of shape `(..., n, n)`.
    """
    if theta == 0:
        raise ValueError("theta must be non-zero for Power-Cholesky metric.")

    X = ensure_sym(X)
    L = torch.linalg.cholesky(X)
    tril_L = L.tril(-1)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
    deformed_diag = diag_L.pow(theta) / theta
    return tril_L + torch.diag_embed(deformed_diag)


def pcm_inv_coords(
    Z: torch.Tensor, theta: float = 0.5, eps: float = 1e-6
) -> torch.Tensor:
    r"""Reconstruct SPD matrix from deformed Power-Cholesky coordinates.

    Parameters
    ----------
    Z : torch.Tensor
        Deformed coordinates of shape `(..., n, n)`.
    theta : float, default=0.5
        Power deformation parameter (:math:`\theta \neq 0`).
    eps : float, default=1e-6
        Numerical floor for clamping before power.

    Returns
    -------
    torch.Tensor
        Reconstructed SPD matrix of shape `(..., n, n)`.
    """
    if theta == 0:
        raise ValueError("theta must be non-zero for Power-Cholesky metric.")

    tril_Z = Z.tril(-1)
    diag_Z = torch.diagonal(Z, dim1=-2, dim2=-1)
    inner = (theta * diag_Z).clamp(min=eps)
    d = inner.pow(1.0 / theta)
    L = tril_Z + torch.diag_embed(d)
    return L @ L.transpose(-1, -2)


def pcm_distance(
    A: torch.Tensor,
    B: torch.Tensor,
    theta: float = 0.5,
    diagonal_weight: float = 1.0,
) -> torch.Tensor:
    r"""Compute Riemannian distance under the Power-Cholesky Metric (PCM).

    .. math::

        d_{\text{PCM}}(A, B) = \sqrt{\|\lfloor L_A \rfloor - \lfloor L_B \rfloor\|_F^2 +
        w_{\text{diag}} \left\|\frac{1}{\theta}(d_A^\theta - d_B^\theta)\right\|_2^2}

    Parameters
    ----------
    A, B : torch.Tensor
        SPD matrices of shape `(..., n, n)`.
    theta : float, default=0.5
        Power parameter (:math:`\theta \neq 0`).
    diagonal_weight : float, default=1.0
        Weight :math:`w_{\text{diag}}` on diagonal distance.

    Returns
    -------
    torch.Tensor
        Distances of shape `(...)`.
    """
    if theta == 0:
        raise ValueError("theta must be non-zero for Power-Cholesky metric.")

    Z_A = pcm_coords(A, theta=theta)
    Z_B = pcm_coords(B, theta=theta)
    if diagonal_weight == 1.0:
        return distance_euclid(Z_A, Z_B)
    tril_diff = Z_A.tril(-1) - Z_B.tril(-1)
    diff_diag = torch.diagonal(Z_A, dim1=-2, dim2=-1) - torch.diagonal(
        Z_B, dim1=-2, dim2=-1
    )
    sq_tril = tril_diff.pow(2).sum(dim=(-2, -1))
    sq_diag = diff_diag.pow(2).sum(dim=-1)
    return torch.sqrt((sq_tril + diagonal_weight * sq_diag).clamp(min=0.0))


def pcm_geodesic(
    A: torch.Tensor,
    B: torch.Tensor,
    t: Union[float, torch.Tensor],
    theta: float = 0.5,
) -> torch.Tensor:
    r"""Compute geodesic interpolation under the Power-Cholesky Metric (PCM).

    Parameters
    ----------
    A, B : torch.Tensor
        Starting and ending SPD matrices of shape `(..., n, n)`.
    t : float or torch.Tensor
        Interpolation parameter :math:`t \in [0, 1]`.
    theta : float, default=0.5
        Power parameter (:math:`\theta \neq 0`).

    Returns
    -------
    torch.Tensor
        Interpolated SPD matrix of shape `(..., n, n)`.
    """
    Z_A = pcm_coords(A, theta=theta)
    Z_B = pcm_coords(B, theta=theta)
    if torch.is_tensor(t) and t.ndim == 0 and Z_A.ndim > 2:
        t = t.expand(Z_A.shape[:-2])
    Z_t = geodesic_euclid(Z_A, Z_B, alpha=t)
    return pcm_inv_coords(Z_t, theta=theta)


def pcm_mean(
    X: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    theta: float = 0.5,
    dim: int = 0,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute closed-form Fréchet mean in Power-Cholesky coordinates.

    Parameters
    ----------
    X : torch.Tensor
        SPD matrices of shape `(..., n, n)` with at least 3 dims.
    weights : torch.Tensor, optional
        Weights along `dim`, normalized to sum to 1.
    theta : float, default=0.5
        Power parameter (:math:`\theta \neq 0`).
    dim : int, default=0
        Batch dimension along which mean is computed.
    keepdim : bool, default=False
        Whether to retain `dim`.

    Returns
    -------
    torch.Tensor
        Weighted Fréchet mean SPD matrix.
    """
    if X.ndim < 3:
        raise ValueError(
            f"X must have at least 3 dimensions (batch of matrices), got {X.ndim}"
        )
    if dim < 0:
        dim = X.ndim + dim
    if dim >= X.ndim - 2 or dim < 0:
        raise ValueError(f"dim must be a batch dimension (< {X.ndim - 2}), got {dim}")

    Z = pcm_coords(X, theta=theta)
    if weights is None:
        Z_moved = Z.movedim(dim, -3)
        Z_bar = mean_euclid(Z_moved)
        if keepdim:
            Z_bar = Z_bar.unsqueeze(dim)
        return pcm_inv_coords(Z_bar, theta=theta)

    N = X.shape[dim]
    weights = weights.to(device=X.device, dtype=X.dtype)
    if weights.ndim == 1 and weights.shape[0] == N:
        w_shape = [1] * X.ndim
        w_shape[dim] = N
        w = weights.view(w_shape)
    elif weights.ndim == X.ndim - 2:
        w = weights.unsqueeze(-1).unsqueeze(-1)
    elif weights.ndim == X.ndim - 1:
        w = weights.unsqueeze(-1)
    else:
        w = weights
    w = w / w.sum(dim=dim, keepdim=True)
    Z_bar = (w * Z).sum(dim=dim, keepdim=keepdim)
    return pcm_inv_coords(Z_bar, theta=theta)


def log_map_pcm(
    X: torch.Tensor,
    P: Optional[torch.Tensor] = None,
    theta: float = 0.5,
) -> torch.Tensor:
    r"""Compute Riemannian logarithmic map under the Power-Cholesky Metric.

    Parameters
    ----------
    X : torch.Tensor
        Target SPD matrix of shape `(..., n, n)`.
    P : torch.Tensor, optional
        Base SPD matrix of shape `(..., n, n)`. If None, identity :math:`I` is used.
    theta : float, default=0.5
        Power parameter (:math:`\theta \neq 0`).

    Returns
    -------
    torch.Tensor
        Deformed Cholesky coordinates of shape `(..., n, n)`.
    """
    if theta == 0:
        raise ValueError("theta must be non-zero for Power-Cholesky metric.")

    X = ensure_sym(X)
    L_X = torch.linalg.cholesky(X)
    tril_X = L_X.tril(-1)
    d_X = torch.diagonal(L_X, dim1=-2, dim2=-1)

    if P is None:
        v_diag = (d_X.pow(theta) - 1.0) / theta
        return tril_X + torch.diag_embed(v_diag)

    P = ensure_sym(P)
    L_P = torch.linalg.cholesky(P)
    tril_P = L_P.tril(-1)
    d_P = torch.diagonal(L_P, dim1=-2, dim2=-1)

    v_tril = tril_X - tril_P
    v_diag = (1.0 / theta) * d_P.pow(1.0 - theta) * (d_X.pow(theta) - d_P.pow(theta))
    return v_tril + torch.diag_embed(v_diag)


def exp_map_pcm(
    V: torch.Tensor,
    P: Optional[torch.Tensor] = None,
    theta: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Compute Riemannian exponential map under the Power-Cholesky Metric.

    Parameters
    ----------
    V : torch.Tensor
        Tangent matrix / deformed coordinates of shape `(..., n, n)`.
    P : torch.Tensor, optional
        Base SPD matrix of shape `(..., n, n)`. If None, identity :math:`I` is used.
    theta : float, default=0.5
        Power parameter (:math:`\theta \neq 0`).
    eps : float, default=1e-6
        Numerical threshold to clamp diagonal.

    Returns
    -------
    torch.Tensor
        Reconstructed SPD matrix of shape `(..., n, n)`.
    """
    if theta == 0:
        raise ValueError("theta must be non-zero for Power-Cholesky metric.")

    v_tril = V.tril(-1)
    v_diag = torch.diagonal(V, dim1=-2, dim2=-1)

    if P is None:
        inner = (1.0 + theta * v_diag).clamp(min=eps)
        d_X = inner.pow(1.0 / theta)
        L = v_tril + torch.diag_embed(d_X)
        return L @ L.transpose(-1, -2)

    P = ensure_sym(P)
    L_P = torch.linalg.cholesky(P)
    tril_P = L_P.tril(-1)
    d_P = torch.diagonal(L_P, dim1=-2, dim2=-1)

    inner = (d_P.pow(theta) + theta * d_P.pow(theta - 1.0) * v_diag).clamp(min=eps)
    d_X = inner.pow(1.0 / theta)
    L = (tril_P + v_tril) + torch.diag_embed(d_X)
    return L @ L.transpose(-1, -2)


def bwcm_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""Compute Riemannian distance under the Bures-Wasserstein-Cholesky Metric (BWCM)."""
    return pcm_distance(A, B, theta=0.5, diagonal_weight=0.25)


def bwcm_geodesic(
    A: torch.Tensor,
    B: torch.Tensor,
    t: Union[float, torch.Tensor],
) -> torch.Tensor:
    r"""Compute geodesic interpolation under the Bures-Wasserstein-Cholesky Metric (BWCM)."""
    return pcm_geodesic(A, B, t, theta=0.5)


def bwcm_mean(
    X: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: int = 0,
    keepdim: bool = False,
) -> torch.Tensor:
    r"""Compute closed-form Fréchet mean under the Bures-Wasserstein-Cholesky Metric (BWCM)."""
    return pcm_mean(X, weights=weights, theta=0.5, dim=dim, keepdim=keepdim)


def log_map_bwcm(
    X: torch.Tensor,
    P: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute Riemannian logarithmic map under the Bures-Wasserstein-Cholesky Metric (BWCM)."""
    return log_map_pcm(X, P=P, theta=0.5)


def exp_map_bwcm(
    V: torch.Tensor,
    P: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Compute Riemannian exponential map under the Bures-Wasserstein-Cholesky Metric (BWCM)."""
    return exp_map_pcm(V, P=P, theta=0.5)


__all__ = [
    "pcm_coords",
    "pcm_inv_coords",
    "pcm_distance",
    "pcm_geodesic",
    "pcm_mean",
    "log_map_pcm",
    "exp_map_pcm",
    "bwcm_distance",
    "bwcm_geodesic",
    "bwcm_mean",
    "log_map_bwcm",
    "exp_map_bwcm",
]
