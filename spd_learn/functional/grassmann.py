# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Distances between linear subspaces (Grassmann manifold)."""

import torch

from torch import Tensor


def grassmann_distance(U: Tensor, V: Tensor, metric: str = "geodesic") -> Tensor:
    r"""Distance between the column spans of two bases.

    With principal angles :math:`\theta_i` between the subspaces
    :math:`\operatorname{span}(U)` and :math:`\operatorname{span}(V)`
    :cite:p:`knyazev2002principal`:

    .. math::

        d_{\text{geodesic}} = \|\theta\|_2, \quad
        d_{\text{chordal}} = \|\sin\theta\|_2, \quad
        d_{\text{FS}} = \arccos\prod_i \cos\theta_i, \quad
        d_{\text{BC}} = \sqrt{1 - \prod_i \cos^2\theta_i}.

    The chordal distance equals the projection distance
    :math:`\|P_U - P_V\|_F / \sqrt{2}` between the orthogonal projectors.

    Bases are orthonormalised by QR; full column rank is assumed. Complex
    bases are supported. The computation runs outside autocast, because
    half-precision matmuls destroy the small-angle accuracy. First derivatives
    are finite everywhere; second derivatives are not at repeated angles.

    Parameters
    ----------
    U, V : Tensor
        Bases of shape ``(..., n, k)`` with ``k <= n``, with identical shapes.
    metric : {"geodesic", "chordal", "fubini_study", "binet_cauchy"}
        Distance to compute.

    Returns
    -------
    Tensor
        Distances of shape ``(...)``.

    Raises
    ------
    ValueError
        If the shapes differ, ``k > n`` (bases passed as rows), or ``metric``
        is unknown.
    """
    if U.shape != V.shape:
        raise ValueError(f"U and V must have the same shape, got {U.shape}, {V.shape}")
    if U.shape[-1] > U.shape[-2]:
        raise ValueError(
            f"bases must be (..., n, k) with k <= n, got {tuple(U.shape)}; "
            "pass column bases, e.g. components_.T"
        )
    if metric not in ("geodesic", "chordal", "fubini_study", "binet_cauchy"):
        raise ValueError(f"Unknown Grassmann metric: {metric!r}")
    with torch.autocast(U.device.type, enabled=False):
        return _grassmann(U, V, metric)


def _grassmann(U: Tensor, V: Tensor, metric: str) -> Tensor:
    Qu, Qv = torch.linalg.qr(U).Q, torch.linalg.qr(V).Q
    M = Qu.mH @ Qv
    cos = torch.linalg.svdvals(M)
    # sines sorted descending pair with cosines sorted descending once flipped
    sin = torch.linalg.svdvals(Qv - Qu @ M).flip(-1)
    theta = torch.atan2(sin, cos)
    if metric == "geodesic":
        return theta.norm(dim=-1)
    if metric == "chordal":
        return theta.sin().norm(dim=-1)
    # 1 - prod(cos^2) without cancellation at small angles
    # (the clamp keeps log1p and its gradient finite when an angle is pi/2)
    s2 = -torch.expm1(
        torch.log1p(
            -(theta.sin() ** 2).clamp(max=1 - torch.finfo(theta.dtype).eps)
        ).sum(-1)
    )
    # sqrt with a zero subgradient at coincident subspaces (sqrt'(0) is inf)
    positive = s2 > 0
    s = torch.where(positive, torch.where(positive, s2, 1.0).sqrt(), 0.0)
    if metric == "binet_cauchy":
        return s
    return torch.atan2(s, theta.cos().prod(-1))
