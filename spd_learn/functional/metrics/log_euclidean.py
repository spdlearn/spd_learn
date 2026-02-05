# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch

from ..core import matrix_exp, matrix_log


def log_euclidean_distance(A, B):
    r"""Computes the Log-Euclidean distance between SPD matrices.

    The Log-Euclidean Metric (LEM) simplifies computations by using the matrix
    logarithm to map the SPD manifold diffeomorphically to the Euclidean vector
    space of symmetric matrices. The mapping :math:`\log: \mathcal{S}_{++}^n \to
    \mathcal{S}^n` is a global diffeomorphism.

    The Log-Euclidean distance is defined as the Frobenius norm of the difference
    of the matrix logarithms:

    .. math::

       d_{\text{LEM}}(A, B) = \| \log(A) - \log(B) \|_F

    This metric endows SPD matrices with a **commutative Lie group structure**,
    where the group operation is :math:`A \odot B = \exp(\log(A) + \log(B))`.

    Parameters
    ----------
    A : torch.Tensor
        SPD matrices with shape `(..., n, n)`.
    B : torch.Tensor
        SPD matrices with shape `(..., n, n)`. Must be broadcastable with `A`.

    Returns
    -------
    torch.Tensor
        Distances with shape `(...)`.

    Notes
    -----
    Unlike the Affine-Invariant Riemannian Metric (AIRM), the Log-Euclidean
    metric is **not affine-invariant**. It is invariant only under orthogonal
    transformations (rotations): :math:`d(QAQ^\top, QBQ^\top) = d(A, B)` for
    orthogonal :math:`Q`. However, it offers computational advantages as all
    operations reduce to standard Euclidean operations in the log-domain.

    See Also
    --------
    :func:`~spd_learn.functional.airm_distance` : Distance under affine-invariant metric.
    :func:`~spd_learn.functional.bures_wasserstein_distance` : Distance under Bures-Wasserstein metric.
    :func:`~spd_learn.functional.log_cholesky_distance` : Distance under Log-Cholesky metric.
    :func:`log_euclidean_mean` : Computes the weighted mean under Log-Euclidean metric.

    References
    ----------
    See :cite:p:`arsigny2007geometric` for more details.
    """
    inner_term = matrix_log.apply(A) - matrix_log.apply(B)
    final = torch.linalg.norm(inner_term, ord="fro", dim=(-2, -1))
    return final


def log_euclidean_mean(weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Computes the weighted Log-Euclidean mean of a batch of SPD matrices.

    The weighted Log-Euclidean mean is computed by taking the weighted average
    of the matrix logarithms of the SPD matrices, and then taking the matrix
    exponential of the result.

    .. math::

       \\text{mean}(V) = \\exp\\left( \\sum_i w_i \\log(V_i) \\right)

    Parameters
    ----------
    weights : torch.Tensor
        Attention probabilities with shape `(..., n, n)`.
    V : torch.Tensor
        SPD matrices with shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        The weighted Log-Euclidean mean of the SPD matrices, with shape
        `(..., n, n)`.
    """
    log_V = matrix_log.apply(V)
    weighted_log = torch.einsum("...pq,...qij->...pij", weights, log_V)
    return matrix_exp.apply(weighted_log)


def log_euclidean_geodesic(A, B, t):
    r"""Geodesic interpolation under the Log-Euclidean metric.

    Computes the point on the geodesic between SPD matrices :math:`A` and
    :math:`B` at parameter :math:`t` under the Log-Euclidean metric:

    .. math::

        \gamma(t) = \exp\left((1-t) \log(A) + t \log(B)\right)

    Since the Log-Euclidean metric induces a flat (Euclidean) geometry on the
    log-domain, geodesics are simply straight lines in that space.

    Parameters
    ----------
    A : torch.Tensor
        Starting SPD matrices with shape `(..., n, n)`.
    B : torch.Tensor
        Ending SPD matrices with shape `(..., n, n)`.
    t : float or torch.Tensor
        Interpolation parameter. For `t=0`, returns `A`. For `t=1`, returns `B`.
        For `t=0.5`, returns the geodesic midpoint (Log-Euclidean mean of two
        matrices).

    Returns
    -------
    torch.Tensor

        Interpolated SPD matrices on the geodesic with shape `(..., n, n)`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import log_euclidean_geodesic
    >>> A = torch.eye(3)
    >>> B = 4 * torch.eye(3)
    >>> # Midpoint
    >>> mid = log_euclidean_geodesic(A, B, 0.5)
    >>> print(f"Midpoint diagonal: {torch.diag(mid)}")
    Midpoint diagonal: tensor([2., 2., 2.])

    See Also
    --------
    :func:`log_euclidean_distance` : Distance under Log-Euclidean metric.
    :func:`log_euclidean_mean` : Weighted mean under Log-Euclidean metric.
    :func:`~spd_learn.functional.airm_geodesic` : Geodesic under AIRM.
    :func:`~spd_learn.functional.bures_wasserstein_geodesic` : Geodesic under Bures-Wasserstein metric.
    :func:`~spd_learn.functional.log_cholesky_geodesic` : Geodesic under Log-Cholesky metric.

    References
    ----------
    See :cite:p:`arsigny2007geometric` for more details.
    """
    log_A = matrix_log.apply(A)
    log_B = matrix_log.apply(B)
    return matrix_exp.apply((1 - t) * log_A + t * log_B)


def exp_map_lem(P, V):
    r"""Riemannian exponential map under the Log-Euclidean metric.

    Maps a tangent vector :math:`V` at base point :math:`P` to a point on the
    SPD manifold by shooting along the geodesic in direction :math:`V`.

    Under the Log-Euclidean metric, the exponential map is:

    .. math::

        \text{Exp}_P(V) = \exp(\log(P) + V)

    where :math:`V` is a symmetric matrix representing a tangent vector at
    :math:`P`.

    Parameters
    ----------
    P : torch.Tensor
        Base point on the SPD manifold with shape `(..., n, n)`.
    V : torch.Tensor
        Tangent vector at P (symmetric matrix) with shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor

        Point on the SPD manifold with shape `(..., n, n)`.

    Notes
    -----
    Under the Log-Euclidean metric, the SPD manifold is globally flat (zero
    curvature), so the exponential map is a global diffeomorphism. The tangent
    space at any point can be identified with the space of symmetric matrices.

    See Also
    --------
    :func:`log_map_lem` : Inverse operation (logarithmic map).
    :func:`~spd_learn.functional.parallel_transport_lem` : Parallel transport under LEM.
    :func:`log_euclidean_geodesic` : Geodesic interpolation under LEM.

    References
    ----------
    See :cite:p:`arsigny2007geometric` for more details.
    """
    log_P = matrix_log.apply(P)
    return matrix_exp.apply(log_P + V)


def log_map_lem(P, Q):
    r"""Riemannian logarithmic map under the Log-Euclidean metric.

    Maps a point :math:`Q` on the SPD manifold to a tangent vector at base
    point :math:`P`. This is the inverse of the exponential map.

    Under the Log-Euclidean metric, the logarithmic map is:

    .. math::

        \text{Log}_P(Q) = \log(Q) - \log(P)

    The result is a symmetric matrix representing the tangent vector at
    :math:`P` that points towards :math:`Q`.

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
    The norm of the tangent vector equals the Log-Euclidean distance:

    .. math::

        \|\text{Log}_P(Q)\|_F = d_{\text{LEM}}(P, Q)

    See Also
    --------
    :func:`exp_map_lem` : Inverse operation (exponential map).
    :func:`log_euclidean_distance` : Distance under Log-Euclidean metric.
    :func:`~spd_learn.functional.parallel_transport_lem` : Parallel transport under LEM.

    References
    ----------
    See :cite:p:`arsigny2007geometric` for more details.
    """
    log_P = matrix_log.apply(P)
    log_Q = matrix_log.apply(Q)
    return log_Q - log_P


def log_euclidean_multiply(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""Logarithmic multiplication of SPD matrices under the Log-Euclidean metric.

    Computes the **logarithmic multiplication** (denoted :math:`\odot` in the
    literature) of two SPD matrices:

    .. math::

        X \odot Y = \exp(\log(X) + \log(Y))

    This operation endows the SPD manifold with a **commutative Lie group
    structure** :cite:p:`arsigny2007geometric`. The Lie group
    :math:`(\mathcal{S}_{++}^n, \odot)` is isomorphic to the additive group
    of symmetric matrices :math:`(\mathcal{S}^n, +)` via the matrix logarithm.

    The Log-Euclidean framework, introduced by Arsigny et al. (2006, 2007),
    provides two algebraic structures on SPD matrices:

    1. A **Lie group structure** via logarithmic multiplication :math:`\odot`
    2. A **vector space structure** by adding logarithmic scalar multiplication
       :math:`\circledast` (see :func:`log_euclidean_scalar_multiply`)

    This operation is useful for implementing geometrically principled
    residual/skip connections in SPD neural networks
    :cite:p:`katsman2023riemannian`.

    Parameters
    ----------
    x : torch.Tensor
        First SPD tensor with shape `(..., n, n)`.
    y : torch.Tensor
        Second SPD tensor with shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Product SPD tensor with shape `(..., n, n)`.

    Notes
    -----
    The logarithmic multiplication satisfies the following properties:

    - **Commutative**: :math:`X \odot Y = Y \odot X`
    - **Associative**: :math:`(X \odot Y) \odot Z = X \odot (Y \odot Z)`
    - **Identity element**: :math:`X \odot I = X` (identity matrix is the
      neutral element)
    - **Inverse**: :math:`X \odot X^{-1} = I`
    - **SPD-preserving**: Output is SPD if inputs are SPD

    The logarithmic multiplication coincides with standard matrix multiplication
    when the two matrices commute in the matrix sense.

    See Also
    --------
    :func:`log_euclidean_scalar_multiply` : Scalar multiplication in Log-Euclidean space.
    :func:`log_euclidean_geodesic` : Weighted combination (geodesic interpolation).
    :func:`log_euclidean_mean` : Weighted mean of multiple SPD matrices.
    :class:`~spd_learn.modules.LogEuclideanResidual` : Module wrapper for this function.

    References
    ----------
    The logarithmic multiplication was introduced by :cite:t:`arsigny2007geometric`
    as part of the Log-Euclidean framework. The original papers are:

    - Arsigny, V., Fillard, P., Pennec, X., and Ayache, N. "Log-Euclidean
      metrics for fast and simple calculus on diffusion tensors."
      *Magnetic Resonance in Medicine*, 56(2):411-421, 2006.
    - Arsigny, V., Fillard, P., Pennec, X., and Ayache, N. "Geometric means
      in a novel vector space structure on symmetric positive-definite
      matrices." *SIAM Journal on Matrix Analysis and Applications*,
      29(1):328-347, 2007.

    For applications to residual neural networks on Riemannian manifolds,
    see :cite:t:`katsman2023riemannian`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import log_euclidean_multiply
    >>> X = torch.eye(3) * 2
    >>> Y = torch.eye(3) * 3
    >>> Z = log_euclidean_multiply(X, Y)
    >>> # Z = exp(log(2*I) + log(3*I)) = exp(log(6)*I) = 6*I
    >>> torch.allclose(Z, torch.eye(3) * 6, atol=1e-5)
    True
    """
    log_x = matrix_log.apply(x)
    log_y = matrix_log.apply(y)
    return matrix_exp.apply(log_x + log_y)


def log_euclidean_scalar_multiply(alpha: float, x: torch.Tensor) -> torch.Tensor:
    r"""Logarithmic scalar multiplication of an SPD matrix.

    Computes the **logarithmic scalar multiplication** (denoted :math:`\circledast`
    in the literature) of a scalar and an SPD matrix:

    .. math::

        \alpha \circledast X = \exp(\alpha \cdot \log(X))

    This operation, together with logarithmic multiplication :math:`\odot`,
    extends the Lie group structure on SPD matrices to a **vector space
    structure** :cite:p:`arsigny2007geometric`.

    The logarithmic scalar multiplication generalizes the notion of matrix
    power to the Log-Euclidean framework and provides a geometrically
    meaningful way to scale SPD matrices.

    Parameters
    ----------
    alpha : float or torch.Tensor
        Scalar multiplier. Can be any real number.
    x : torch.Tensor
        SPD tensor with shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Scaled SPD tensor with shape `(..., n, n)`.

    Notes
    -----
    The logarithmic scalar multiplication satisfies the following properties:

    - **Distributive over** :math:`\odot`: :math:`\alpha \circledast (X \odot Y)
      = (\alpha \circledast X) \odot (\alpha \circledast Y)`
    - **Compatible with scalar multiplication**:
      :math:`(\alpha \beta) \circledast X = \alpha \circledast (\beta \circledast X)`
    - **Identity**: :math:`1 \circledast X = X`
    - **Zero**: :math:`0 \circledast X = I` (identity matrix)
    - **Inverse**: :math:`(-1) \circledast X = X^{-1}`
    - **SPD-preserving**: Output is SPD for any real :math:`\alpha`

    For diagonal matrices, this reduces to element-wise power:
    :math:`\alpha \circledast \text{diag}(d_1, \ldots, d_n) =
    \text{diag}(d_1^\alpha, \ldots, d_n^\alpha)`.

    See Also
    --------
    :func:`log_euclidean_multiply` : Logarithmic multiplication of SPD matrices.
    :func:`log_euclidean_geodesic` : Geodesic interpolation (uses scalar multiplication).
    :func:`log_euclidean_mean` : Weighted mean of multiple SPD matrices.

    References
    ----------
    The logarithmic scalar multiplication was introduced by
    :cite:t:`arsigny2007geometric` (Definition 3.12) to extend the Lie group
    structure on SPD matrices to a vector space structure. See:

    - Arsigny, V., Fillard, P., Pennec, X., and Ayache, N. "Geometric means
      in a novel vector space structure on symmetric positive-definite
      matrices." *SIAM Journal on Matrix Analysis and Applications*,
      29(1):328-347, 2007.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import log_euclidean_scalar_multiply
    >>> X = torch.eye(3) * 4
    >>> # 0.5 ⊛ X = exp(0.5 * log(X)) = X^0.5 = 2*I
    >>> Y = log_euclidean_scalar_multiply(0.5, X)
    >>> torch.allclose(Y, torch.eye(3) * 2, atol=1e-5)
    True
    >>> # -1 ⊛ X = X^{-1}
    >>> Z = log_euclidean_scalar_multiply(-1, X)
    >>> torch.allclose(Z, torch.eye(3) * 0.25, atol=1e-5)
    True
    """
    log_x = matrix_log.apply(x)
    return matrix_exp.apply(alpha * log_x)
