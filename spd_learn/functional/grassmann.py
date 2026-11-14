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

    ``"projection"`` is computed directly from the orthogonal projectors
    :math:`P_U`, :math:`P_V`, without principal angles, and is a squared
    distance:

    .. math::

        d_{\text{projection}} = \|P_U - P_V\|_F^2 = 2\, d_{\text{chordal}}^2.

    Bases are orthonormalised by QR; full column rank is assumed.

    Parameters
    ----------
    U, V : Tensor
        Bases of shape ``(..., n, k)``, with identical shapes.
    metric : {"geodesic", "chordal", "fubini_study", "binet_cauchy", "projection"}
        Distance to compute.

    Returns
    -------
    Tensor
        Distances of shape ``(...)``.

    Raises
    ------
    ValueError
        If the shapes differ or ``metric`` is unknown.
    """
    if U.shape != V.shape:
        raise ValueError(f"U and V must have the same shape, got {U.shape}, {V.shape}")
    if metric not in (
        "geodesic",
        "chordal",
        "fubini_study",
        "binet_cauchy",
        "projection",
    ):
        raise ValueError(f"Unknown Grassmann metric: {metric!r}")
    Qu, Qv = torch.linalg.qr(U).Q, torch.linalg.qr(V).Q
    if metric == "projection":
        return (Qu @ Qu.mT - Qv @ Qv.mT).square().sum(dim=(-2, -1))
    cos = torch.linalg.svdvals(Qu.mT @ Qv)
    # sines sorted descending pair with cosines sorted descending once flipped
    sin = torch.linalg.svdvals(Qv - Qu @ (Qu.mT @ Qv)).flip(-1)
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
    s = s2.clamp(min=0).sqrt()
    if metric == "binet_cauchy":
        return s
    return torch.atan2(s, theta.cos().prod(-1))
