# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Parallel transport operations on SPD manifolds.

This module implements parallel transport operations for tangent vectors on the
manifold of Symmetric Positive Definite (SPD) matrices.

**Geometric Definition**

Given a smooth curve :math:`\gamma: [0, 1] \to \mathcal{M}` on a Riemannian manifold,
a vector field :math:`X` along :math:`\gamma` is said to be **parallel** if:

.. math::

    \nabla_{\dot{\gamma}(t)} X = 0 \quad \text{for all } t

where :math:`\nabla` is the Levi-Civita connection (covariant derivative). For any
initial tangent vector :math:`v \in T_{\gamma(0)}\mathcal{M}`, there exists a unique
parallel vector field along :math:`\gamma` with :math:`X(0) = v`. The terminal vector
:math:`X(1)` is the **parallel transport** of :math:`v` along :math:`\gamma`:

.. math::

    P_\gamma: T_{\gamma(0)}\mathcal{M} \to T_{\gamma(1)}\mathcal{M}

This defines a linear isomorphism that preserves inner products.

**Applications**

Parallel transport is essential for:

- **Domain adaptation**: Riemannian Procrustes Analysis aligns covariance matrices
  across subjects/sessions :cite:p:`rodrigues2019riemannian`
- **Tangent space methods**: Aligning data from different reference points
- **Transfer learning**: Adapting BCI models between recording conditions
- **Batch normalization**: Centering SPD features via transport to identity

The module provides implementations for different Riemannian metrics:

- **AIRM**: Uses the congruence transport formula (non-trivial transport)
- **LEM**: Uses Frechet derivatives of matrix log/exp, see table 4 from :cite:p:`thanwerdas2023`
- **Log-Cholesky**: Uses Cholesky decomposition with log-diagonal transport :cite:p:`lin2019riemannian`

It also provides numerical approximation methods:

- **Schild's ladder**: Iterative geodesic-based approximation
- **Pole ladder**: A more efficient variant using midpoints

See :cite:p:`pennec2006riemannian` and :cite:p:`lorenzi2014efficient` for more
details on parallel transport and numerical approximation methods.
"""

import torch

from .core import (
    matrix_inv_sqrt,
    matrix_log,
    matrix_sqrt,
    matrix_sqrt_inv,
)
from .frechet import frechet_derivative_exp, frechet_derivative_log
from .metrics.affine_invariant import exp_map_airm, log_map_airm
from .utils import ensure_sym


def _parallel_transport_airm_functional(v, p, q):
    r"""Parallel transport under AIRM with full autograd support.

    This functional implementation leverages PyTorch's autograd by using
    operations that already have proper backward passes defined. This allows
    gradients to flow through all inputs (v, p, q).

    **Formula Relationship**

    The transport operator is :math:`E = (Q P^{-1})^{1/2}` (principal square
    root). Since :math:`Q P^{-1}` is non-symmetric when :math:`P \neq Q`, we
    use the equivalent stable formula:

    .. math::

        E = Q^{1/2} (Q^{-1/2} P Q^{-1/2})^{-1/2} Q^{-1/2}

    This computes only symmetric matrix square roots, which are well-defined
    for SPD matrices. The equivalence follows from :math:`E^2 = Q P^{-1}`,
    which can be verified by direct computation (see :func:`parallel_transport_airm`).

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p, shape (..., n, n). Must be symmetric.
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n).
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n).

    Returns
    -------
    torch.Tensor
        Transported tangent vector at q, shape (..., n, n).
    """
    # Compute E = Q^{1/2} @ (Q^{-1/2} @ P @ Q^{-1/2})^{-1/2} @ Q^{-1/2}
    # This is the principal square root of Q @ P^{-1}
    q_sqrt, q_invsqrt = matrix_sqrt_inv.apply(q)
    inner = q_invsqrt @ p @ q_invsqrt
    inner_invsqrt = matrix_inv_sqrt.apply(inner)
    E = q_sqrt @ inner_invsqrt @ q_invsqrt

    # Transport: v' = E @ v @ E^T
    v_transported = E @ v @ E.transpose(-2, -1)
    return ensure_sym(v_transported)


def parallel_transport_airm(v, p, q):
    r"""Parallel transport of tangent vector under the Affine-Invariant metric.

    Transports a tangent vector :math:`V \in T_P \mathcal{M}` from the tangent
    space at :math:`P` to the tangent space at :math:`Q` along the geodesic
    connecting them, using the Affine-Invariant Riemannian Metric (AIRM)
    :cite:p:`pennec2006riemannian`.

    **Transport Formula**

    The transport is given by:

    .. math::

        \Gamma_{P \rightarrow Q}(V) = E V E^T

    where :math:`E = (Q P^{-1})^{1/2}` is the **principal square root** of
    :math:`Q P^{-1}`.

    **Mathematical Derivation**

    The matrix :math:`Q P^{-1}` is generally **non-symmetric** (even when
    :math:`P` and :math:`Q` are SPD), so computing its principal square root
    requires care. The principal square root is defined as the unique matrix
    :math:`E` such that:

    1. :math:`E^2 = Q P^{-1}`
    2. All eigenvalues of :math:`E` have positive real parts

    **Numerically Stable Formula**

    We compute :math:`E` using the equivalent stable formula:

    .. math::

        E = Q^{1/2} (Q^{-1/2} P Q^{-1/2})^{-1/2} Q^{-1/2}

    This formula avoids computing :math:`P^{-1}` directly and uses only
    symmetric matrix square roots. To verify this is correct:

    .. math::

        E^2 &= Q^{1/2} (Q^{-1/2} P Q^{-1/2})^{-1/2} Q^{-1/2}
               \cdot Q^{1/2} (Q^{-1/2} P Q^{-1/2})^{-1/2} Q^{-1/2} \\
            &= Q^{1/2} (Q^{-1/2} P Q^{-1/2})^{-1} Q^{-1/2} \\
            &= Q^{1/2} Q^{1/2} P^{-1} Q^{1/2} Q^{-1/2} \\
            &= Q P^{-1}

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p, shape (..., n, n). Must be symmetric.
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n).
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n).

    Returns
    -------
    torch.Tensor
        Transported tangent vector at q, shape (..., n, n).

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import parallel_transport_airm
    >>> # Create random SPD matrices
    >>> n = 3
    >>> A = torch.randn(n, n)
    >>> p = A @ A.T + torch.eye(n)
    >>> B = torch.randn(n, n)
    >>> q = B @ B.T + torch.eye(n)
    >>> # Create a tangent vector at p (symmetric matrix)
    >>> v = torch.randn(n, n)
    >>> v = (v + v.T) / 2
    >>> # Transport v from T_P to T_Q
    >>> v_transported = parallel_transport_airm(v, p, q)

    Notes
    -----
    **Isometry Property**

    Parallel transport preserves the AIRM inner product:

    .. math::

        \langle \Gamma(U), \Gamma(V) \rangle_Q = \langle U, V \rangle_P

    where :math:`\langle U, V \rangle_P = \text{tr}(P^{-1} U P^{-1} V)`.

    **Gradient Support**

    This function supports full gradient computation through all inputs (v, p, q).
    Gradients flow correctly for use in optimization when learning reference points.

    **References**

    - :cite:p:`pennec2006riemannian` for the Riemannian framework on SPD manifolds
    - :cite:p:`skovgaard1984riemannian` for the original derivation of AIRM transport
    - :cite:p:`yair2019parallel` for applications in domain adaptation

    See Also
    --------
    :func:`parallel_transport_lem` : Parallel transport under Log-Euclidean metric.
    :func:`schild_ladder` : Numerical approximation via Schild's ladder.
    :func:`pole_ladder` : Numerical approximation via pole ladder.
    :func:`~spd_learn.functional.airm_geodesic` : Geodesic under AIRM.
    :func:`~spd_learn.functional.airm_distance` : Distance under AIRM.
    :class:`~spd_learn.modules.SPDBatchNormMeanVar` : Uses parallel transport for centering.
    """
    return _parallel_transport_airm_functional(v, p, q)


def parallel_transport_lem(v, p, q):
    r"""Parallel transport of tangent vector under the Log-Euclidean metric.

    Transports a tangent vector :math:`V \in T_P \mathcal{M}` from the tangent
    space at :math:`P` to the tangent space at :math:`Q` using the
    Log-Euclidean metric :cite:p:`thanwerdas2023`.

    **Transport Formula**

    .. math::

        \Gamma_{P \rightarrow Q}^{LEM}(V) = D\exp(\log Q)\bigl[D\log(P)[V]\bigr]

    where :math:`D\log(P)` is the Frechet derivative of the matrix logarithm
    at :math:`P` and :math:`D\exp(\log Q)` is the Frechet derivative of the
    matrix exponential at :math:`\log Q`. The intermediate step maps the
    ambient tangent vector into the flat log-space, where transport is trivial,
    then maps back to the ambient tangent space at :math:`Q`.

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p, shape (..., n, n). Must be symmetric.
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n).
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n).

    Returns
    -------
    torch.Tensor
        Transported tangent vector at q, shape (..., n, n).

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import parallel_transport_lem
    >>> n = 3
    >>> A = torch.randn(n, n, dtype=torch.float64)
    >>> p = A @ A.T + torch.eye(n, dtype=torch.float64)
    >>> B = torch.randn(n, n, dtype=torch.float64)
    >>> q = B @ B.T + torch.eye(n, dtype=torch.float64)
    >>> v = torch.randn(n, n, dtype=torch.float64)
    >>> v = (v + v.T) / 2
    >>> v_transported = parallel_transport_lem(v, p, q)
    >>> # Self-transport should be identity
    >>> v_self = parallel_transport_lem(v, p, p)
    >>> torch.allclose(v, v_self, atol=1e-6)
    True

    Notes
    -----
    While the Log-Euclidean metric makes the SPD manifold globally flat
    (zero curvature), so that parallel transport is trivial **in the log
    space**, it is non-trivial when expressed in the ambient SPD space for
    tangent vectors represented as symmetric matrices. The Frechet
    derivatives handle the coordinate change between these representations.

    See Also
    --------
    :func:`parallel_transport_airm` : Parallel transport under AIRM.
    :func:`~spd_learn.functional.frechet.frechet_derivative_log` : Frechet derivative of log.
    :func:`~spd_learn.functional.frechet.frechet_derivative_exp` : Frechet derivative of exp.
    :func:`~spd_learn.functional.log_euclidean_distance` : Distance under Log-Euclidean metric.
    """
    # Step 1: Map v from ambient T_P to log-space via D_log(P)[V]
    w = frechet_derivative_log(p, v)
    # Step 2: Map w from log-space back to ambient T_Q via D_exp(log(Q))[W]
    log_q = matrix_log.apply(q)
    v_transported = frechet_derivative_exp(log_q, w)
    return ensure_sym(v_transported)


def parallel_transport_log_cholesky(v, p, q):
    r"""Parallel transport of tangent vector under the Log-Cholesky metric.

    Transports a tangent vector :math:`V \in T_P \mathcal{M}` from the tangent
    space at :math:`P` to the tangent space at :math:`Q` using the
    Log-Cholesky metric :cite:p:`lin2019riemannian` (Proposition 7).

    **Transport Formula**

    Given Cholesky decompositions :math:`P = L_P L_P^T` and
    :math:`Q = L_Q L_Q^T`:

    1. Pull back :math:`V` to the Cholesky tangent space:
       :math:`S = L_P^{-1} V L_P^{-T}`, then
       :math:`B = \operatorname{strictly\_lower}(S) + \frac{1}{2}\operatorname{diag}(S)`
       and :math:`dL = L_P B`.

    2. Convert to log-Cholesky coordinates:
       :math:`dY = \operatorname{strictly\_lower}(dL) + \operatorname{diag}(\operatorname{diag}(dL) / \operatorname{diag}(L_P))`

    3. Transport in flat log-Cholesky space is the identity: :math:`dY` stays
       the same.

    4. Convert back at :math:`Q`:
       :math:`dL_Q = \operatorname{strictly\_lower}(dY) + \operatorname{diag}(\operatorname{diag}(dY) \cdot \operatorname{diag}(L_Q))`

    5. Push forward:
       :math:`V' = dL_Q L_Q^T + L_Q dL_Q^T`

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p, shape (..., n, n). Must be symmetric.
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n).
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n).

    Returns
    -------
    torch.Tensor
        Transported tangent vector at q, shape (..., n, n).

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import parallel_transport_log_cholesky
    >>> n = 3
    >>> A = torch.randn(n, n, dtype=torch.float64)
    >>> p = A @ A.T + torch.eye(n, dtype=torch.float64)
    >>> B = torch.randn(n, n, dtype=torch.float64)
    >>> q = B @ B.T + torch.eye(n, dtype=torch.float64)
    >>> v = torch.randn(n, n, dtype=torch.float64)
    >>> v = (v + v.T) / 2
    >>> v_transported = parallel_transport_log_cholesky(v, p, q)
    >>> # Self-transport should be identity
    >>> v_self = parallel_transport_log_cholesky(v, p, p)
    >>> torch.allclose(v, v_self, atol=1e-6)
    True

    Notes
    -----
    While the Log-Cholesky metric makes the SPD manifold globally flat in
    the log-Cholesky coordinates, transport is non-trivial when expressed
    in the ambient SPD space for tangent vectors represented as symmetric
    matrices. This implementation follows Proposition 7 of
    :cite:p:`lin2019riemannian`.

    See Also
    --------
    :func:`parallel_transport_airm` : Parallel transport under AIRM.
    :func:`parallel_transport_lem` : Parallel transport under Log-Euclidean.
    :func:`~spd_learn.functional.log_cholesky_distance` : Distance under Log-Cholesky metric.
    :func:`~spd_learn.functional.log_cholesky_mean` : Mean under Log-Cholesky metric.
    """
    # Cholesky decompose P and Q
    L_P = torch.linalg.cholesky(p)
    L_Q = torch.linalg.cholesky(q)

    # Step 1: Pull V back to the Cholesky tangent space
    # Compute S = L_P^{-1} V L_P^{-T} (symmetric)
    # First: X = L_P^{-1} V  (solve L_P X = V)
    X = torch.linalg.solve_triangular(L_P, v, upper=False)
    # Then: S = X L_P^{-T}  (solve L_P S^T = X^T, then transpose)
    S = torch.linalg.solve_triangular(
        L_P, X.transpose(-2, -1), upper=False
    ).transpose(-2, -1)

    # B = strictly_lower(S) + 0.5 * diag(S)
    B = S.tril(-1) + torch.diagonal(S, dim1=-2, dim2=-1).unsqueeze(-2) * 0.5 * torch.eye(
        S.shape[-1], dtype=S.dtype, device=S.device
    )

    # dL = L_P @ B (lower triangular)
    dL = L_P @ B

    # Step 2: Convert to log-Cholesky tangent coordinates
    diag_L_P = torch.diagonal(L_P, dim1=-2, dim2=-1)
    diag_dL = torch.diagonal(dL, dim1=-2, dim2=-1)
    # dY = strictly_lower(dL) + diag(diag(dL) / diag(L_P))
    dY = dL.tril(-1) + torch.diag_embed(diag_dL / diag_L_P)

    # Step 3: Transport in flat log-Cholesky space is identity (dY stays the same)

    # Step 4: Convert back at Q
    diag_L_Q = torch.diagonal(L_Q, dim1=-2, dim2=-1)
    diag_dY = torch.diagonal(dY, dim1=-2, dim2=-1)
    # dL_Q = strictly_lower(dY) + diag(diag(dY) * diag(L_Q))
    dL_Q = dY.tril(-1) + torch.diag_embed(diag_dY * diag_L_Q)

    # Step 5: Push forward to ambient tangent space at Q
    # V' = dL_Q @ L_Q^T + L_Q @ dL_Q^T
    v_transported = dL_Q @ L_Q.transpose(-2, -1) + L_Q @ dL_Q.transpose(-2, -1)

    return ensure_sym(v_transported)


def _geodesic_midpoint_airm(p, q):
    r"""Compute the geodesic midpoint between p and q under AIRM.

    The geodesic midpoint is:

    .. math::

        M = P^{1/2} (P^{-1/2} Q P^{-1/2})^{1/2} P^{1/2}

    Parameters
    ----------
    p : torch.Tensor
        First SPD matrix, shape (..., n, n).
    q : torch.Tensor
        Second SPD matrix, shape (..., n, n).

    Returns
    -------
    torch.Tensor
        Geodesic midpoint, shape (..., n, n).
    """
    p_sqrt, p_invsqrt = matrix_sqrt_inv.apply(p)
    inner = p_invsqrt @ q @ p_invsqrt
    inner_sqrt = matrix_sqrt.apply(inner)
    return p_sqrt @ inner_sqrt @ p_sqrt


def schild_ladder(v, p, q, n_steps=5):
    r"""Parallel transport via Schild's ladder approximation.

    Schild's ladder is a numerical scheme for approximating parallel transport
    along geodesics. It constructs a sequence of geodesic parallelograms to
    iteratively transport the tangent vector.

    The algorithm:
    1. Divide the geodesic from P to Q into n_steps segments
    2. For each step, construct a parallelogram using midpoints
    3. The opposite vertex of the parallelogram gives the transported vector

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p to be transported, shape (..., n, n).
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n).
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n).
    n_steps : int, optional
        Number of ladder rungs (iterations). More steps give better accuracy.
        Default is 5.

    Returns
    -------
    torch.Tensor
        Approximately transported tangent vector at q, shape (..., n, n).

    Notes
    -----
    Schild's ladder converges to the true parallel transport as n_steps -> inf.
    The approximation error is O(1/n_steps^2) for smooth geodesics.

    This method is metric-agnostic and works for any Riemannian metric where
    geodesics and exponential/logarithmic maps are available
    :cite:p:`ehlers1972geometry`, :cite:p:`lorenzi2014efficient`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import schild_ladder, parallel_transport_airm
    >>> n = 3
    >>> A = torch.randn(n, n)
    >>> p = A @ A.T + torch.eye(n)
    >>> B = torch.randn(n, n)
    >>> q = B @ B.T + torch.eye(n)
    >>> v = torch.randn(n, n)
    >>> v = (v + v.T) / 2
    >>> # Compare Schild's ladder with closed-form transport
    >>> v_schild = schild_ladder(v, p, q, n_steps=10)
    >>> v_exact = parallel_transport_airm(v, p, q)

    See Also
    --------
    :func:`parallel_transport_airm` : Closed-form parallel transport under AIRM.
    :func:`pole_ladder` : Alternative numerical scheme (more efficient).
    :func:`~spd_learn.functional.airm_geodesic` : Geodesic under AIRM.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    # Current position and tangent vector
    current_p = p
    current_v = v.clone()

    # Step size along the geodesic
    t_step = 1.0 / n_steps

    for i in range(n_steps):
        # Target point for this step along the main geodesic
        t_target = (i + 1) * t_step
        next_p = exp_map_airm(p, log_map_airm(p, q), t_target)

        # Schild's ladder construction:
        # 1. Compute endpoint of tangent vector: x1 = exp_p(v)
        x1 = exp_map_airm(current_p, current_v)

        # 2. Compute midpoint m between x1 and next_p
        m = _geodesic_midpoint_airm(x1, next_p)

        # 3. Reflect current_p through m to get x2
        # x2 = exp_m(log_m(current_p)) with opposite sign
        # This is equivalent to: x2 = exp_m(-log_m(current_p))
        log_m_p = log_map_airm(m, current_p)
        x2 = exp_map_airm(m, -log_m_p)

        # 4. The transported vector is log_{next_p}(x2)
        current_v = log_map_airm(next_p, x2)
        current_p = next_p

    return ensure_sym(current_v)


def pole_ladder(v, p, q):
    r"""Parallel transport via pole ladder approximation.

    Pole ladder is a more efficient variant of Schild's ladder that uses a
    single iteration with the geodesic midpoint as a "pole". It provides
    a good approximation with less computation than multi-step Schild's ladder.

    The algorithm (following Lorenzi & Pennec 2014):
    1. Compute x1 = exp_P(V), the endpoint of the tangent vector
    2. Compute the geodesic midpoint M between P and Q (the "pole")
    3. Compute x1' = exp_M(-log_M(x1)), reflecting x1 through M
    4. Transported vector is log_Q(x1')

    This is essentially one step of Schild's ladder but uses the full
    geodesic midpoint as the reflection center, which gives better accuracy.

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p to be transported, shape (..., n, n).
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n).
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n).

    Returns
    -------
    torch.Tensor
        Approximately transported tangent vector at q, shape (..., n, n).

    Notes
    -----
    Pole ladder has O(h^2) approximation error where h is the geodesic
    distance between P and Q. For small distances, it provides a good
    balance between accuracy and computational cost :cite:p:`lorenzi2014efficient`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import pole_ladder, parallel_transport_airm
    >>> n = 3
    >>> A = torch.randn(n, n)
    >>> p = A @ A.T + torch.eye(n)
    >>> B = torch.randn(n, n)
    >>> q = B @ B.T + torch.eye(n)
    >>> v = torch.randn(n, n)
    >>> v = (v + v.T) / 2
    >>> # Compare pole ladder with closed-form transport
    >>> v_pole = pole_ladder(v, p, q)
    >>> v_exact = parallel_transport_airm(v, p, q)

    See Also
    --------
    :func:`parallel_transport_airm` : Closed-form parallel transport under AIRM.
    :func:`schild_ladder` : Multi-step numerical approximation.
    :func:`~spd_learn.functional.airm_geodesic` : Geodesic under AIRM.
    """
    # 1. Compute x1 = exp_P(V), endpoint of tangent vector at p
    x1 = exp_map_airm(p, v)

    # 2. Compute geodesic midpoint M between P and Q (the "pole")
    m = _geodesic_midpoint_airm(p, q)

    # 3. Reflect x1 through M: x1' = exp_M(-log_M(x1))
    log_m_x1 = log_map_airm(m, x1)
    x1_prime = exp_map_airm(m, -log_m_x1)

    # 4. Transported vector is -log_Q(x1')
    # The negation accounts for the reflection: x1' is on the opposite side of M
    # from where the parallel-transported endpoint should be
    v_transported = -log_map_airm(q, x1_prime)

    return ensure_sym(v_transported)


def transport_tangent_vector(v, p, q, metric="airm", **kwargs):
    r"""Parallel transport of tangent vector with metric selection.

    A convenience function that allows selecting the transport method via
    a string argument.

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p to be transported, shape (..., n, n).
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n).
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n).
    metric : str, optional
        The Riemannian metric to use. Options are:
        - "airm": Affine-Invariant Riemannian Metric (closed-form)
        - "lem" or "log_euclidean": Log-Euclidean Metric (Frechet derivatives)
        - "log_cholesky": Log-Cholesky Metric (Cholesky decomposition)
        - "schild": Schild's ladder approximation
        - "pole": Pole ladder approximation
        Default is "airm".
    **kwargs : dict
        Additional keyword arguments passed to the transport function.
        For example, n_steps for schild_ladder.

    Returns
    -------
    torch.Tensor
        Transported tangent vector at q, shape (..., n, n).

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import transport_tangent_vector
    >>> n = 3
    >>> A = torch.randn(n, n)
    >>> p = A @ A.T + torch.eye(n)
    >>> B = torch.randn(n, n)
    >>> q = B @ B.T + torch.eye(n)
    >>> v = torch.randn(n, n)
    >>> v = (v + v.T) / 2
    >>> # Use different transport methods
    >>> v_airm = transport_tangent_vector(v, p, q, metric="airm")
    >>> v_lem = transport_tangent_vector(v, p, q, metric="lem")
    >>> v_schild = transport_tangent_vector(v, p, q, metric="schild", n_steps=10)

    See Also
    --------
    :func:`parallel_transport_airm` : Direct AIRM transport.
    :func:`parallel_transport_lem` : Direct LEM transport.
    :func:`schild_ladder` : Numerical Schild's ladder.
    :func:`pole_ladder` : Numerical pole ladder.
    """
    metric = metric.lower()

    if metric == "airm":
        return parallel_transport_airm(v, p, q)
    elif metric in ("lem", "log_euclidean", "log-euclidean"):
        return parallel_transport_lem(v, p, q)
    elif metric in ("log_cholesky", "log-cholesky", "lc"):
        return parallel_transport_log_cholesky(v, p, q)
    elif metric == "schild":
        return schild_ladder(v, p, q, **kwargs)
    elif metric == "pole":
        return pole_ladder(v, p, q)
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Supported metrics are: "
            "'airm', 'lem', 'log_euclidean', 'log_cholesky', 'schild', 'pole'"
        )
