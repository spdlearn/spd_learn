# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
r"""Bures-Wasserstein metric operations for SPD matrices.

This module implements operations based on the Bures-Wasserstein Metric (BWM),
also known as the Wasserstein-2 distance for Gaussian distributions
:cite:p:`bhatia2019bures`. The BWM provides an alternative geometric structure
on the SPD manifold that originates from quantum information theory and optimal
transport.

Geometric Foundation
--------------------
Unlike the Affine-Invariant Riemannian Metric (AIRM) which has non-positive
sectional curvature, the BWM endows :math:`\mathcal{S}_{++}^n` with a
**positively curved** Riemannian structure.

Riemannian Inner Product
------------------------
At :math:`P \in \mathcal{S}_{++}^n`, for tangent matrices :math:`V, W`:

.. math::

    g^{\text{BW}}_P(V, W) = \text{tr}(\mathcal{L}_P[V] W)

where :math:`\mathcal{L}_P` is the **Lyapunov operator** that assigns to each
:math:`V \in \mathcal{S}^n` the unique solution :math:`X` of the Lyapunov equation:

.. math::

    PX + XP = V

Distance
--------
The Bures-Wasserstein distance between two SPD matrices :math:`A` and :math:`B`
is defined as:

.. math::

    d_{\text{BW}}(A, B) = \sqrt{\text{tr}(A) + \text{tr}(B) - 2\text{tr}\left((A^{1/2} B A^{1/2})^{1/2}\right)}

Geodesic
--------
The geodesic between :math:`A` and :math:`B` is given by:

.. math::

    \gamma(t) = (1-t)^2 A + t^2 B + t(1-t)(M + M^\top)

where :math:`M = (A^{1/2} B A^{1/2})^{1/2}`.

Key Properties
--------------
- **Positively curved**: Unlike AIRM (non-positive curvature), BWM has positive
  sectional curvature.
- **Optimal transport interpretation**: The distance equals the 2-Wasserstein
  distance between centered Gaussian distributions :math:`\mathcal{N}(0, A)`
  and :math:`\mathcal{N}(0, B)`.
- **Quantum fidelity**: Related to the fidelity measure in quantum information.
- **Closed-form expressions**: Distances, geodesics, and barycenters have
  closed-form solutions (barycenters via fixed-point iteration).
- **No eigendecomposition**: Computation avoids eigenvalue decomposition, using
  matrix square roots instead.
- **Not affine-invariant**: Invariant only under unitary transformations.

See :cite:p:`bhatia2019bures`, :cite:p:`malago2018wasserstein`, and
:cite:p:`agueh2011barycenters` for more details.
"""

import warnings

import torch

from torch.autograd import Function

from ..core import matrix_inv_sqrt, matrix_sqrt, matrix_sqrt_inv
from ..numerical import get_epsilon, numerical_config
from ..utils import ensure_sym


class _BuresWassersteinDistanceFunction(Function):
    r"""Autograd function for Bures-Wasserstein distance computation.

    This implements the forward and backward passes for the BW distance
    with proper gradient computation through the matrix square root.
    """

    @staticmethod
    def forward(ctx, A, B):
        r"""Compute the Bures-Wasserstein distance.

        Parameters
        ----------
        A : torch.Tensor
            SPD matrices of shape `(..., n, n)`.
        B : torch.Tensor
            SPD matrices of shape `(..., n, n)`.

        Returns
        -------
        torch.Tensor
            Distances of shape `(...)`.
        """
        # Compute A^{1/2}
        A_sqrt = matrix_sqrt.apply(A)

        # Compute A^{1/2} B A^{1/2}
        ABA = A_sqrt @ B @ A_sqrt

        # Compute (A^{1/2} B A^{1/2})^{1/2}
        ABA_sqrt = matrix_sqrt.apply(ABA)

        # Compute traces
        trace_A = torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)
        trace_B = torch.diagonal(B, dim1=-2, dim2=-1).sum(dim=-1)
        trace_ABA_sqrt = torch.diagonal(ABA_sqrt, dim1=-2, dim2=-1).sum(dim=-1)

        # Compute distance
        distance_sq = trace_A + trace_B - 2 * trace_ABA_sqrt

        # Clamp for numerical stability (should be non-negative)
        threshold = get_epsilon(A.dtype, "eigval_sqrt")
        distance_sq = distance_sq.clamp(min=threshold)
        distance = distance_sq.sqrt()

        # Save for backward
        ctx.save_for_backward(A, B, A_sqrt, ABA_sqrt, distance)

        return distance

    @staticmethod
    def backward(ctx, grad_output):
        r"""Backward pass for Bures-Wasserstein distance.

        The gradient is computed using the chain rule through the trace
        and matrix square root operations.
        """
        A, B, A_sqrt, ABA_sqrt, distance = ctx.saved_tensors

        # Compute A^{-1/2}
        A_inv_sqrt = matrix_inv_sqrt.apply(A)

        # Gradient scaling factor: d(sqrt(x))/dx = 1/(2*sqrt(x))
        threshold = get_epsilon(distance.dtype, "eigval_sqrt")
        safe_distance = distance.clamp(min=threshold)
        grad_scale = grad_output / (2 * safe_distance)

        # For very small distances, use subgradient 0
        grad_scale = torch.where(
            distance < threshold, torch.zeros_like(grad_scale), grad_scale
        )

        # Expand grad_scale for matrix operations
        grad_scale_expanded = grad_scale[..., None, None]

        # Compute (A^{1/2} B A^{1/2})^{-1/2}
        ABA_inv_sqrt = matrix_inv_sqrt.apply(A_sqrt @ B @ A_sqrt)

        # Gradient w.r.t. A:
        # d/dA [tr(A) - 2*tr((A^{1/2} B A^{1/2})^{1/2})]
        # = I - (A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2} +
        #        A^{-1/2} B^{1/2} (B^{1/2} A B^{1/2})^{-1/2} B^{1/2} A^{-1/2}) / 2
        # Simplified using the relation with the geometric mean:
        # = I - A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}
        eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        eye = eye.expand_as(A)

        grad_A = grad_scale_expanded * (eye - A_inv_sqrt @ ABA_sqrt @ A_inv_sqrt)
        grad_A = ensure_sym(grad_A)

        # Gradient w.r.t. B:
        # d/dB [tr(B) - 2*tr((A^{1/2} B A^{1/2})^{1/2})]
        # = I - A^{1/2} (A^{1/2} B A^{1/2})^{-1/2} A^{1/2}
        grad_B = grad_scale_expanded * (eye - A_sqrt @ ABA_inv_sqrt @ A_sqrt)
        grad_B = ensure_sym(grad_B)

        return grad_A, grad_B


def bures_wasserstein_distance(A, B):
    r"""Compute the Bures-Wasserstein distance between SPD matrices.

    The Bures-Wasserstein (BW) distance, also known as the Wasserstein-2
    distance for centered Gaussian distributions, is defined as:

    .. math::

        d_{BW}(A, B) = \sqrt{\text{tr}(A) + \text{tr}(B) - 2\text{tr}\left((A^{1/2} B A^{1/2})^{1/2}\right)}

    This metric is symmetric and satisfies the triangle inequality. It is
    equivalent to the optimal transport distance between centered Gaussian
    distributions :math:`\mathcal{N}(0, A)` and :math:`\mathcal{N}(0, B)`
    :cite:p:`bhatia2019bures`.

    Parameters
    ----------
    A : torch.Tensor
        SPD matrices of shape `(..., n, n)`.
    B : torch.Tensor
        SPD matrices of shape `(..., n, n)`. Must be broadcastable with `A`.

    Returns
    -------
    torch.Tensor
        Bures-Wasserstein distances of shape `(...)`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.bures_wasserstein import bures_wasserstein_distance
    >>> # Single pair of 3x3 SPD matrices
    >>> A = torch.eye(3)
    >>> B = 2 * torch.eye(3)
    >>> d = bures_wasserstein_distance(A, B)
    >>> print(f"Distance: {d.item():.4f}")
    Distance: 0.8787

    >>> # Batch of matrices
    >>> A = torch.eye(3).unsqueeze(0).expand(10, 3, 3)
    >>> B = torch.randn(10, 3, 3)
    >>> B = B @ B.transpose(-1, -2) + 0.1 * torch.eye(3)  # Make SPD
    >>> distances = bures_wasserstein_distance(A, B)
    >>> print(f"Shape: {distances.shape}")
    Shape: torch.Size([10])

    See Also
    --------
    :func:`bures_wasserstein_geodesic` : Geodesic interpolation under BWM.
    :func:`bures_wasserstein_mean` : Fréchet mean under BWM.
    :func:`bures_wasserstein_transport` : Optimal transport map under BWM.
    :func:`~spd_learn.functional.airm_distance` : Distance under AIRM.
    :func:`~spd_learn.functional.log_euclidean_distance` : Distance under Log-Euclidean metric.
    :func:`~spd_learn.functional.log_cholesky_distance` : Distance under Log-Cholesky metric.
    """
    return _BuresWassersteinDistanceFunction.apply(A, B)


def bures_wasserstein_geodesic(A, B, t):
    r"""Compute the geodesic interpolation under the Bures-Wasserstein metric.

    The geodesic between two SPD matrices :math:`A` and :math:`B` under the
    Bures-Wasserstein metric is given by:

    .. math::

        \gamma(t) = \left((1-t) A^{1/2} + t A^{-1/2} \# B \right)^2

    where :math:`A^{-1/2} \# B` denotes the matrix geometric mean:

    .. math::

        A^{-1/2} \# B = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}

    Alternatively, this can be written as:

    .. math::

        \gamma(t) = (1-t)^2 A + t^2 B + t(1-t) \left(M + M^T\right)

    where :math:`M = (A^{1/2} B A^{1/2})^{1/2}`.

    Parameters
    ----------
    A : torch.Tensor
        Starting point SPD matrices of shape `(..., n, n)`.
    B : torch.Tensor
        End point SPD matrices of shape `(..., n, n)`.
    t : float or torch.Tensor
        Interpolation parameter(s). When `t = 0`, returns `A`.
        When `t = 1`, returns `B`. Can be a scalar or tensor
        broadcastable with the batch dimensions.

    Returns
    -------
    torch.Tensor
        Interpolated SPD matrices of shape `(..., n, n)`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.bures_wasserstein import bures_wasserstein_geodesic
    >>> A = torch.eye(3)
    >>> B = 2 * torch.eye(3)
    >>> # Midpoint
    >>> C = bures_wasserstein_geodesic(A, B, 0.5)
    >>> # Endpoints
    >>> torch.allclose(bures_wasserstein_geodesic(A, B, 0.0), A)
    True
    >>> torch.allclose(bures_wasserstein_geodesic(A, B, 1.0), B)
    True

    Notes
    -----
    The geodesic is computed using the formula:

    .. math::

        \gamma(t) = (1-t)^2 A + t^2 B + t(1-t)(M + M^T)

    where :math:`M = (A^{1/2} B A^{1/2})^{1/2}`. This formulation is
    numerically stable and efficient to compute :cite:p:`bhatia2019bures`.

    See Also
    --------
    :func:`bures_wasserstein_distance` : Distance under BWM.
    :func:`bures_wasserstein_mean` : Fréchet mean under BWM.
    :func:`~spd_learn.functional.airm_geodesic` : Geodesic under AIRM.
    :func:`~spd_learn.functional.log_cholesky_geodesic` : Geodesic under Log-Cholesky metric.
    """
    # Ensure t is a tensor
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=A.dtype, device=A.device)
    else:
        t = t.to(dtype=A.dtype, device=A.device)

    # Handle edge cases exactly for scalar t
    if t.numel() == 1:
        t_item = t.item()
        if t_item == 0.0:
            return A.clone()
        if t_item == 1.0:
            return B.clone()

    # Expand t for broadcasting
    t = t.reshape(t.shape + (1,) * 2)

    # Compute A^{1/2}
    A_sqrt = matrix_sqrt.apply(A)

    # Compute M = (A^{1/2} B A^{1/2})^{1/2}
    ABA = A_sqrt @ B @ A_sqrt
    M = matrix_sqrt.apply(ABA)

    # Geodesic formula: (1-t)^2 A + t^2 B + t(1-t)(M + M^T)
    result = (1 - t) ** 2 * A + t**2 * B + t * (1 - t) * (M + M.transpose(-1, -2))

    # Ensure symmetry
    return ensure_sym(result)


def bures_wasserstein_mean(
    matrices,
    weights=None,
    max_iter=50,
    tol=1e-8,
    init=None,
    return_info=False,
):
    r"""Compute the Bures-Wasserstein barycenter of SPD matrices.

    The Bures-Wasserstein barycenter (also known as the Fréchet mean under the
    BW metric) of a set of SPD matrices :math:`\{A_1, \ldots, A_K\}` with
    weights :math:`\{w_1, \ldots, w_K\}` is defined as:

    .. math::

        \bar{A} = \arg\min_{M \succ 0} \sum_{i=1}^{K} w_i \, d_{BW}^2(M, A_i)

    The barycenter is computed using the fixed-point iteration:

    .. math::

        M_{k+1} = \sum_{i=1}^{K} w_i \left(M_k^{1/2} A_i M_k^{1/2}\right)^{1/2}

    which converges to the unique barycenter.

    Parameters
    ----------
    matrices : torch.Tensor
        SPD matrices of shape `(K, ..., n, n)` where `K` is the number of
        matrices to average.
    weights : torch.Tensor, optional
        Non-negative weights of shape `(K,)`. If None, uniform weights
        are used. Weights are automatically normalized to sum to 1.
    max_iter : int, default=50
        Maximum number of fixed-point iterations.
    tol : float, default=1e-8
        Convergence tolerance. Iteration stops when the relative change
        in the Frobenius norm is below this threshold.
    init : torch.Tensor, optional
        Initial estimate of the barycenter of shape `(..., n, n)`. If None,
        the arithmetic mean is used as initialization.
    return_info : bool, default=False
        If True, return a dictionary with convergence information.

    Returns
    -------
    barycenter : torch.Tensor
        The Bures-Wasserstein barycenter of shape `(..., n, n)`.
    info : dict, optional
        Convergence information (only if `return_info=True`):
        - ``"n_iter"``: Number of iterations performed
        - ``"converged"``: Whether the algorithm converged
        - ``"relative_change"``: Final relative change in Frobenius norm

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.bures_wasserstein import bures_wasserstein_mean
    >>> # Create 5 random SPD matrices
    >>> K, n = 5, 3
    >>> matrices = torch.randn(K, n, n)
    >>> matrices = matrices @ matrices.transpose(-1, -2) + 0.1 * torch.eye(n)
    >>> # Compute barycenter
    >>> mean = bures_wasserstein_mean(matrices)
    >>> print(f"Barycenter shape: {mean.shape}")
    Barycenter shape: torch.Size([3, 3])

    >>> # With custom weights
    >>> weights = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.1])
    >>> mean = bures_wasserstein_mean(matrices, weights=weights)

    >>> # With convergence info
    >>> mean, info = bures_wasserstein_mean(matrices, return_info=True)
    >>> print(f"Converged in {info['n_iter']} iterations")

    Notes
    -----
    The fixed-point iteration :cite:p:`agueh2011barycenters` is guaranteed to
    converge for any initialization. However, the convergence rate depends on
    the condition numbers of the input matrices. For ill-conditioned matrices,
    more iterations may be needed :cite:p:`alvarez2016fixed`.

    See Also
    --------
    :func:`bures_wasserstein_distance` : Distance under BWM.
    :func:`bures_wasserstein_geodesic` : Geodesic interpolation under BWM.
    :func:`~spd_learn.functional.log_euclidean_mean` : Mean under Log-Euclidean metric.
    :func:`~spd_learn.functional.log_cholesky_mean` : Mean under Log-Cholesky metric.
    :class:`~spd_learn.modules.SPDBatchNormMeanVar` : Uses Fréchet mean for batch normalization.
    """
    K = matrices.shape[0]
    device = matrices.device
    dtype = matrices.dtype

    # Handle weights
    if weights is None:
        weights = torch.ones(K, dtype=dtype, device=device) / K
    else:
        weights = weights.to(dtype=dtype, device=device)
        weights = weights / weights.sum()  # Normalize

    # Validate weights shape
    if weights.shape[0] != K:
        raise ValueError(
            f"Number of weights ({weights.shape[0]}) must match "
            f"number of matrices ({K})"
        )

    # Initialize with arithmetic mean if not provided
    if init is None:
        M = (weights.view(K, *([1] * (matrices.ndim - 1))) * matrices).sum(dim=0)
    else:
        M = init.clone()

    # Ensure M is symmetric
    M = ensure_sym(M)

    # Fixed-point iteration
    converged = False
    relative_change = float("inf")

    for iteration in range(max_iter):
        M_prev = M.clone()

        # Compute M^{1/2}
        M_sqrt = matrix_sqrt.apply(M)

        # Compute weighted sum: sum_i w_i (M^{1/2} A_i M^{1/2})^{1/2}
        accumulator = torch.zeros_like(M)
        for i in range(K):
            # M^{1/2} A_i M^{1/2}
            MAM = M_sqrt @ matrices[i] @ M_sqrt
            # (M^{1/2} A_i M^{1/2})^{1/2}
            MAM_sqrt = matrix_sqrt.apply(MAM)
            # Weighted accumulation
            accumulator = accumulator + weights[i] * MAM_sqrt

        M = accumulator

        # Ensure symmetry
        M = ensure_sym(M)

        # Check convergence
        M_norm = torch.linalg.norm(M, ord="fro", dim=(-2, -1))
        diff_norm = torch.linalg.norm(M - M_prev, ord="fro", dim=(-2, -1))

        # Handle batched case: use maximum relative change
        if M_norm.numel() > 1:
            threshold = get_epsilon(dtype, "eigval_sqrt")
            safe_norm = M_norm.clamp(min=threshold)
            relative_change = (diff_norm / safe_norm).max().item()
        else:
            threshold = get_epsilon(dtype, "eigval_sqrt")
            safe_norm = M_norm.clamp(min=threshold)
            relative_change = (diff_norm / safe_norm).item()

        if relative_change < tol:
            converged = True
            break

    if not converged and numerical_config.warn_on_clamp:
        warnings.warn(
            f"Bures-Wasserstein mean did not converge after {max_iter} iterations. "
            f"Final relative change: {relative_change:.2e}, tolerance: {tol:.2e}",
            UserWarning,
        )

    if return_info:
        info = {
            "n_iter": iteration + 1,
            "converged": converged,
            "relative_change": relative_change,
        }
        return M, info

    return M


def bures_wasserstein_transport(A, B, X):
    r"""Compute the optimal transport map from A to B applied to X.

    Given SPD matrices A and B, the optimal transport map T that pushes
    the Gaussian :math:`\mathcal{N}(0, A)` to :math:`\mathcal{N}(0, B)` is:

    .. math::

        T_{A \to B} = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}

    When applied to a covariance matrix X (representing a Gaussian with
    covariance X), the transported covariance is:

    .. math::

        T_{A \to B}(X) = T_{A \to B} \, X \, T_{A \to B}^T

    This is useful for domain adaptation and covariance alignment tasks.

    Parameters
    ----------
    A : torch.Tensor
        Source SPD matrices of shape `(..., n, n)`.
    B : torch.Tensor
        Target SPD matrices of shape `(..., n, n)`.
    X : torch.Tensor
        SPD matrices to transport of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Transported SPD matrices of shape `(..., n, n)`.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.bures_wasserstein import bures_wasserstein_transport
    >>> # Transport A to B (should give B)
    >>> A = torch.eye(3)
    >>> B = 2 * torch.eye(3)
    >>> transported = bures_wasserstein_transport(A, B, A)
    >>> torch.allclose(transported, B, atol=1e-6)
    True

    Notes
    -----
    When X = A, the transport map gives exactly B. This property can be
    used to verify the correctness of the implementation :cite:p:`givens1984class`.

    See Also
    --------
    :func:`bures_wasserstein_distance` : Distance under BWM.
    :func:`bures_wasserstein_geodesic` : Geodesic interpolation under BWM.
    :func:`~spd_learn.functional.parallel_transport_airm` : Parallel transport under AIRM.
    """
    # Compute A^{1/2} and A^{-1/2}
    A_sqrt, A_inv_sqrt = matrix_sqrt_inv.apply(A)

    # Compute (A^{1/2} B A^{1/2})^{1/2}
    ABA = A_sqrt @ B @ A_sqrt
    ABA_sqrt = matrix_sqrt.apply(ABA)

    # Transport map: T = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}
    T = A_inv_sqrt @ ABA_sqrt @ A_inv_sqrt

    # Apply transport to X: T X T^T
    transported = T @ X @ T.transpose(-1, -2)

    # Ensure symmetry
    return ensure_sym(transported)
