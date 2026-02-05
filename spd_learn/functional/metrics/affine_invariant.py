# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Affine-Invariant Riemannian Metric (AIRM) for SPD matrices.

This module implements operations based on the Affine-Invariant Riemannian
Metric (AIRM), also known as the Fisher-Rao metric or canonical metric on
the SPD manifold :cite:p:`pennec2006riemannian`, :cite:p:`bhatia2007positive`.

Geometric Foundation
--------------------
The AIRM endows the SPD manifold :math:`\mathcal{S}_{++}^n` with a Riemannian
structure that is invariant under congruence transformations:

.. math::

    d(WAW^\top, WBW^\top) = d(A, B)

for any invertible matrix :math:`W`.

Riemannian Inner Product
------------------------
At :math:`P \in \mathcal{S}_{++}^n`, for tangent vectors :math:`U, V`:

.. math::

    g_P(U, V) = \text{tr}(P^{-1} U P^{-1} V)

Distance
--------
The geodesic distance between SPD matrices :math:`A` and :math:`B` is:

.. math::

    d_{\text{AIRM}}(A, B) = \|\log(A^{-1/2} B A^{-1/2})\|_F

Geodesic
--------
The geodesic between :math:`A` and :math:`B` is given by:

.. math::

    \gamma(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}

Key Properties
--------------
- **Affine invariance**: Distance is unchanged by congruence transformations
- **Hadamard manifold**: Complete, simply connected, nonpositive sectional
  curvature
- **Unique Fréchet mean**: Always exists and is unique
- **Boundary at infinity**: Singular matrices are at infinite distance

See :cite:p:`pennec2006riemannian` and :cite:p:`bhatia2007positive` for
more details.
"""

import torch

from ..core import (
    matrix_exp,
    matrix_inv_sqrt,
    matrix_log,
    matrix_power,
    matrix_sqrt_inv,
)
from ..numerical import get_epsilon
from ..utils import ensure_sym


def airm_distance(A, B):
    r"""Compute the geodesic distance under the Affine-Invariant Riemannian
    Metric (AIRM).

    The AIRM distance between two SPD matrices :math:`A` and :math:`B` is
    defined as the length of the geodesic connecting them:

    .. math::

        d_{\text{AIRM}}(A, B) = \| \log(A^{-1/2} B A^{-1/2}) \|_F
                              = \sqrt{\sum_{i=1}^{n} \log^2(\lambda_i)}

    where :math:`\lambda_i` are the eigenvalues of :math:`A^{-1/2} B A^{-1/2}`.

    This metric is **affine-invariant**: for any invertible matrix :math:`W`,

    .. math::

        d_{\text{AIRM}}(WAW^\top, WBW^\top) = d_{\text{AIRM}}(A, B)

    Parameters
    ----------
    A : torch.Tensor
        SPD matrices with shape `(..., n, n)`.
    B : torch.Tensor
        SPD matrices with shape `(..., n, n)`. Must be broadcastable with `A`.

    Returns
    -------
    torch.Tensor
        Geodesic distances with shape `(...)`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.metrics import airm_distance
    >>> A = torch.eye(3)
    >>> B = 2 * torch.eye(3)
    >>> d = airm_distance(A, B)
    >>> print(f"Distance: {d.item():.4f}")
    Distance: 1.2012

    See Also
    --------
    :func:`airm_geodesic` : Geodesic interpolation under AIRM.
    :func:`exp_map_airm` : Riemannian exponential map.
    :func:`log_map_airm` : Riemannian logarithmic map.
    :func:`~spd_learn.functional.metrics.log_euclidean_distance` : Distance under Log-Euclidean metric.
    :func:`~spd_learn.functional.metrics.bures_wasserstein_distance` : Distance under Bures-Wasserstein metric.

    References
    ----------
    See :cite:p:`pennec2006riemannian`, :cite:p:`bhatia2007positive` for more details.
    """
    Ainvsqrt = matrix_inv_sqrt.apply(A)
    eigenvalues = torch.linalg.eigvalsh(Ainvsqrt @ B @ Ainvsqrt)
    threshold = get_epsilon(eigenvalues.dtype, "eigval_log")
    return eigenvalues.clamp(min=threshold).log().square().sum(dim=-1).sqrt()


def airm_geodesic(A, B, t):
    r"""Geodesic interpolation on the SPD manifold under the Affine-Invariant
    Riemannian Metric (AIRM).

    The AIRM endows the SPD manifold with a geometry that is invariant under
    congruence transformations :math:`P \mapsto WPW^\top` for any invertible
    matrix :math:`W`. The geodesic (shortest path) between two SPD matrices
    :math:`A` and :math:`B` is given by:

    .. math::

        \gamma(t) = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}

    for :math:`t \in [0, 1]`.

    Parameters
    ----------
    A : torch.Tensor
        Starting point SPD matrices with shape `(..., n, n)`.
    B : torch.Tensor
        End point SPD matrices with shape `(..., n, n)`.
    t : float
        Interpolation parameter. For `t = 0`, returns `A`.
        For `t = 1`, returns `B`. For `t = 0.5`, returns the geodesic midpoint
        (Riemannian mean of two matrices).

    Returns
    -------
    torch.Tensor
        Interpolated SPD matrices on the geodesic with shape `(..., n, n)`.

    Notes
    -----
    The geodesic midpoint at :math:`t = 0.5` corresponds to the matrix geometric
    mean :math:`A \# B = A^{1/2}(A^{-1/2}BA^{-1/2})^{1/2}A^{1/2}`, which is the
    unique positive definite solution to the Riccati equation :math:`XA^{-1}X = B`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.metrics import airm_geodesic
    >>> A = torch.eye(3)
    >>> B = 4 * torch.eye(3)
    >>> mid = airm_geodesic(A, B, 0.5)
    >>> print(f"Midpoint diagonal: {torch.diag(mid)}")
    Midpoint diagonal: tensor([2., 2., 2.])

    See Also
    --------
    :func:`airm_distance` : Computes the geodesic distance under AIRM.
    :func:`exp_map_airm` : Riemannian exponential map under AIRM.
    :func:`~spd_learn.functional.metrics.bures_wasserstein_geodesic` : Geodesic under Bures-Wasserstein metric.
    :func:`~spd_learn.functional.metrics.log_cholesky_geodesic` : Geodesic under Log-Cholesky metric.

    References
    ----------
    See :cite:p:`pennec2006riemannian`, :cite:p:`bhatia2007positive` for more details.
    """
    rm_sq, rm_invsq = matrix_sqrt_inv.apply(A)
    return (
        rm_sq
        @ matrix_power.apply(rm_invsq @ B @ rm_invsq, torch.tensor(t).to(A))
        @ rm_sq
    )


def exp_map_airm(P, V, t=1.0):
    r"""Riemannian exponential map under the Affine-Invariant metric.

    Maps a tangent vector :math:`V` at base point :math:`P` to a point on the
    SPD manifold by shooting along the geodesic in direction :math:`V`.

    Under the Affine-Invariant Riemannian Metric (AIRM), the exponential map is:

    .. math::

        \text{Exp}_P(tV) = P^{1/2} \exp(t P^{-1/2} V P^{-1/2}) P^{1/2}

    where :math:`\exp` denotes the matrix exponential and :math:`V` is a
    symmetric matrix representing a tangent vector at :math:`P`.

    Parameters
    ----------
    P : torch.Tensor
        Base point on the SPD manifold with shape `(..., n, n)`.
    V : torch.Tensor
        Tangent vector at P (symmetric matrix) with shape `(..., n, n)`.
    t : float, optional
        Scaling factor for the tangent vector. Default is 1.0.

    Returns
    -------
    torch.Tensor
        Point on the SPD manifold with shape `(..., n, n)`.

    Notes
    -----
    The AIRM exponential map is a diffeomorphism from the tangent space at
    :math:`P` to the entire SPD manifold, because the SPD manifold with AIRM
    forms a Hadamard manifold (complete, simply connected, nonpositive
    sectional curvature).

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.metrics import exp_map_airm, log_map_airm
    >>> P = torch.eye(3)
    >>> V = 0.1 * torch.randn(3, 3)
    >>> V = (V + V.T) / 2  # Symmetrize
    >>> Q = exp_map_airm(P, V)  # Shoot from P in direction V
    >>> V_back = log_map_airm(P, Q)  # Should recover V
    >>> torch.allclose(V, V_back, atol=1e-5)
    True

    See Also
    --------
    :func:`log_map_airm` : Inverse operation (logarithmic map).
    :func:`airm_geodesic` : Geodesic interpolation.
    :func:`airm_distance` : Geodesic distance.

    References
    ----------
    See :cite:p:`pennec2006riemannian` for more details.
    """
    P_sqrt, P_invsqrt = matrix_sqrt_inv.apply(P)
    inner = P_invsqrt @ V @ P_invsqrt
    return P_sqrt @ matrix_exp.apply(t * inner) @ P_sqrt


def log_map_airm(P, Q):
    r"""Riemannian logarithmic map under the Affine-Invariant metric.

    Maps a point :math:`Q` on the SPD manifold to a tangent vector at base
    point :math:`P`. This is the inverse of the exponential map.

    Under the Affine-Invariant Riemannian Metric (AIRM), the logarithmic map is:

    .. math::

        \text{Log}_P(Q) = P^{1/2} \log(P^{-1/2} Q P^{-1/2}) P^{1/2}

    where :math:`\log` denotes the matrix logarithm. The result is a symmetric
    matrix representing the tangent vector at :math:`P` that points towards
    :math:`Q`.

    Parameters
    ----------
    P : torch.Tensor
        Base point on the SPD manifold with shape `(..., n, n)`.
    Q : torch.Tensor
        Target point on the SPD manifold with shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Tangent vector at P (symmetric matrix) with shape `(..., n, n)`.

    Notes
    -----
    The norm of the tangent vector under the AIRM inner product equals the
    geodesic distance:

    .. math::

        \|\text{Log}_P(Q)\|_P = d_{\text{AIRM}}(P, Q)

    where :math:`\|V\|_P = \|P^{-1/2} V P^{-1/2}\|_F` is the AIRM norm.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.metrics import exp_map_airm, log_map_airm
    >>> P = 2 * torch.eye(3)
    >>> Q = 3 * torch.eye(3)
    >>> V = log_map_airm(P, Q)  # Tangent vector from P to Q
    >>> Q_back = exp_map_airm(P, V)  # Should recover Q
    >>> torch.allclose(Q, Q_back, atol=1e-5)
    True

    See Also
    --------
    :func:`exp_map_airm` : Inverse operation (exponential map).
    :func:`airm_distance` : Geodesic distance.

    References
    ----------
    See :cite:p:`pennec2006riemannian` for more details.
    """
    P_sqrt, P_invsqrt = matrix_sqrt_inv.apply(P)
    inner = P_invsqrt @ Q @ P_invsqrt
    return P_sqrt @ matrix_log.apply(inner) @ P_sqrt


def spd_egrad2rgrad(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    r"""Compute the Riemannian gradient from the Euclidean gradient under the AIRM metric.

    The relationship between the Euclidean gradient :math:`\nabla_X \mathcal{J}`
    and the Riemannian gradient :math:`\text{grad}_X \mathcal{J}` under the
    AIRM metric is given by:

    .. math::

        \text{grad}_X \mathcal{J} = X (\nabla_X \mathcal{J})_{sym} X

    where :math:`(\cdot)_{sym}` denotes the symmetric part.

    Parameters
    ----------
    x : torch.Tensor
        Base point on the manifold, shape `(..., n, n)`.
    u : torch.Tensor
        Euclidean gradient at x, shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Riemannian gradient at x, shape `(..., n, n)`.
    """
    return x @ ensure_sym(u) @ x.transpose(-1, -2)
