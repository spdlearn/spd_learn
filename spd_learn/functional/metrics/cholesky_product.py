# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Power-Cholesky and Bures-Wasserstein-Cholesky product metrics on SPD manifolds.

Implements the Power-Cholesky Metric (PCM) and Bures-Wasserstein-Cholesky
Metric (BWCM) for symmetric positive definite (SPD) matrices :cite:p:`chen2026fast`.
Decomposes the Cholesky factor :math:`X = LL^\top` into strictly lower
triangular part :math:`\lfloor L \rfloor` and diagonal :math:`d = \text{diag}(L)`.
"""

from functools import partial
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

    w = weights.to(device=X.device, dtype=X.dtype)
    if w.ndim == 1:
        w = w.view(*[X.shape[dim] if i == dim else 1 for i in range(X.ndim)])
    else:
        w = w.view(*w.shape, *(1,) * (X.ndim - w.ndim))
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
    Z_X = pcm_coords(X, theta=theta)
    if P is None:
        return Z_X - torch.eye(X.shape[-1], device=X.device, dtype=X.dtype) / theta

    L_P = torch.linalg.cholesky(ensure_sym(P))
    d_P = torch.diagonal(L_P, dim1=-2, dim2=-1)
    diff = Z_X - (L_P.tril(-1) + torch.diag_embed(d_P.pow(theta) / theta))
    v_diag = d_P.pow(1.0 - theta) * torch.diagonal(diff, dim1=-2, dim2=-1)
    return diff.tril(-1) + torch.diag_embed(v_diag)


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
    if P is None:
        eye = torch.eye(V.shape[-1], device=V.device, dtype=V.dtype)
        return pcm_inv_coords(V + eye / theta, theta=theta, eps=eps)

    L_P = torch.linalg.cholesky(ensure_sym(P))
    d_P = torch.diagonal(L_P, dim1=-2, dim2=-1)
    z_diag = d_P.pow(theta) / theta + d_P.pow(theta - 1.0) * torch.diagonal(
        V, dim1=-2, dim2=-1
    )
    Z = (L_P + V).tril(-1) + torch.diag_embed(z_diag)
    return pcm_inv_coords(Z, theta=theta, eps=eps)


bwcm_distance = partial(pcm_distance, theta=0.5, diagonal_weight=0.25)
bwcm_geodesic = partial(pcm_geodesic, theta=0.5)
bwcm_mean = partial(pcm_mean, theta=0.5)
log_map_bwcm = partial(log_map_pcm, theta=0.5)
exp_map_bwcm = partial(exp_map_pcm, theta=0.5)


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
