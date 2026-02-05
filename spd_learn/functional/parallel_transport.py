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
- **LEM**: Identity transport in the log-space (trivial/flat geometry)

It also provides numerical approximation methods:

- **Schild's ladder**: Iterative geodesic-based approximation
- **Pole ladder**: A more efficient variant using midpoints

See :cite:p:`pennec2006riemannian` and :cite:p:`lorenzi2014efficient` for more
details on parallel transport and numerical approximation methods.
"""

from torch.autograd import Function

from .core import (
    matrix_inv_sqrt,
    matrix_sqrt,
    matrix_sqrt_inv,
)
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


class ParallelTransportLEM(Function):
    r"""Parallel transport under the Log-Euclidean Metric (LEM).

    Under the Log-Euclidean metric, the SPD manifold is isometric to a flat
    Euclidean space via the matrix logarithm. In this flat space, parallel
    transport is simply the identity operation.

    The transport formula in the original SPD space is:

    .. math::

        \Gamma_{P \rightarrow Q}^{LEM}(V) = D\exp_Q(D\log_P(V))

    where :math:`D\log_P` and :math:`D\exp_Q` are the differentials of the
    logarithm and exponential maps at P and Q respectively.

    For tangent vectors represented as symmetric matrices in the ambient space,
    this simplifies to:

    .. math::

        \Gamma_{P \rightarrow Q}^{LEM}(V) = V

    i.e., the identity transport, because in the log-Euclidean framework,
    tangent vectors are already represented in a shared vector space.

    Notes
    -----
    The Log-Euclidean metric makes the SPD manifold globally flat (zero curvature),
    which means parallel transport is path-independent and reduces to identity.
    This is computationally advantageous but may not capture the intrinsic
    geometry of SPD matrices as well as AIRM.
    """

    @staticmethod
    def forward(ctx, v, p, q):
        """Forward pass for parallel transport under LEM.

        Parameters
        ----------
        v : torch.Tensor
            Tangent vector at p, shape (..., n, n). Must be symmetric.
        p : torch.Tensor
            Source point on SPD manifold, shape (..., n, n). (unused)
        q : torch.Tensor
            Target point on SPD manifold, shape (..., n, n). (unused)

        Returns
        -------
        torch.Tensor
            Transported tangent vector at q (same as v), shape (..., n, n).
        """
        # Under LEM, parallel transport is identity
        ctx.save_for_backward(v)
        return v.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for parallel transport under LEM.

        Since the forward pass is identity, the backward pass is also identity
        for the tangent vector gradient, and zero for the base points.
        """
        (v,) = ctx.saved_tensors
        return grad_output.clone(), None, None


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

    Under the Log-Euclidean metric, the SPD manifold is isometric to a flat
    Euclidean space. Parallel transport in flat space is the identity operation,
    so this function returns the input tangent vector unchanged.

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p, shape (..., n, n). Must be symmetric.
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n). (unused, for API
        consistency)
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n). (unused, for API
        consistency)

    Returns
    -------
    torch.Tensor
        Transported tangent vector at q (identical to v), shape (..., n, n).

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import parallel_transport_lem
    >>> n = 3
    >>> A = torch.randn(n, n)
    >>> p = A @ A.T + torch.eye(n)
    >>> B = torch.randn(n, n)
    >>> q = B @ B.T + torch.eye(n)
    >>> v = torch.randn(n, n)
    >>> v = (v + v.T) / 2
    >>> v_transported = parallel_transport_lem(v, p, q)
    >>> torch.allclose(v, v_transported)
    True

    Notes
    -----
    The Log-Euclidean metric makes the SPD manifold globally flat, which means:

    1. Parallel transport is path-independent
    2. The transported vector equals the original vector
    3. Computationally very efficient (O(1) operation)

    However, this simplicity comes at the cost of not fully capturing the
    intrinsic Riemannian structure of SPD matrices.

    See Also
    --------
    :func:`parallel_transport_airm` : Parallel transport under AIRM (non-trivial).
    :func:`~spd_learn.functional.log_euclidean_distance` : Distance under Log-Euclidean metric.
    :func:`~spd_learn.functional.log_euclidean_mean` : Mean under Log-Euclidean metric.
    """
    return ParallelTransportLEM.apply(v, p, q)


def parallel_transport_log_cholesky(v, p, q):
    r"""Parallel transport of tangent vector under the Log-Cholesky metric.

    Under the Log-Cholesky metric, the SPD manifold inherits a flat (Euclidean)
    geometry from the Cholesky space via the Log-Cholesky map. In flat space,
    parallel transport is the identity operation.

    Parameters
    ----------
    v : torch.Tensor
        Tangent vector at p, shape (..., n, n). Must be symmetric.
    p : torch.Tensor
        Source point on SPD manifold, shape (..., n, n). (unused, for API
        consistency)
    q : torch.Tensor
        Target point on SPD manifold, shape (..., n, n). (unused, for API
        consistency)

    Returns
    -------
    torch.Tensor
        Transported tangent vector at q (identical to v), shape (..., n, n).

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import parallel_transport_log_cholesky
    >>> n = 3
    >>> A = torch.randn(n, n)
    >>> p = A @ A.T + torch.eye(n)
    >>> B = torch.randn(n, n)
    >>> q = B @ B.T + torch.eye(n)
    >>> v = torch.randn(n, n)
    >>> v = (v + v.T) / 2
    >>> v_transported = parallel_transport_log_cholesky(v, p, q)
    >>> torch.allclose(v, v_transported)
    True

    Notes
    -----
    The Log-Cholesky metric makes the SPD manifold globally flat via the
    diffeomorphism to Cholesky space :cite:p:`lin2019riemannian`. This means:

    1. Parallel transport is path-independent
    2. The transported vector equals the original vector
    3. Computationally very efficient (O(1) operation)

    See Also
    --------
    :func:`parallel_transport_airm` : Parallel transport under AIRM (non-trivial).
    :func:`parallel_transport_lem` : Parallel transport under Log-Euclidean (also identity).
    :func:`~spd_learn.functional.log_cholesky_distance` : Distance under Log-Cholesky metric.
    :func:`~spd_learn.functional.log_cholesky_mean` : Mean under Log-Cholesky metric.
    """
    # Under Log-Cholesky metric, parallel transport is identity (flat geometry)
    return v.clone()


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
        - "lem" or "log_euclidean": Log-Euclidean Metric (identity)
        - "log_cholesky": Log-Cholesky Metric (identity)
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
