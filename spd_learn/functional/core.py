# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import warnings

from math import sqrt

import torch

from torch.autograd import Function

from .autograd import modeig_backward, modeig_forward
from .numerical import get_epsilon, numerical_config


def softplus(s):
    """
    Scaled SoftPlus function.
    It is scaled so that: f(0) = 1, f(x) -> 0 as x -> -inf and
    f'(x) -> 1 as x -> +inf

    Parameters
    ----------
    s : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    torch.Tensor
        SoftPlus of s
    """
    return torch.log2(
        1.0 + torch.pow(torch.tensor(2.0, dtype=s.dtype, device=s.device), s)
    )


def inv_softplus(s):
    """
    Inverse of the scaled SoftPlus function

    Parameters
    ----------
    s : torch.Tensor
        Scalar or array of scalars.

    Returns
    -------
    torch.Tensor
        Inverse of SoftPlus of s
    """
    return torch.log2(
        torch.pow(torch.tensor(2.0, device=s.device, dtype=s.dtype), s) - 1.0
    )


class matrix_softplus(Function):
    r"""
    Matrix (scaled) SoftPlus of a symmetric matrix.
    It is scaled so that: f(0) = 1, f(x) -> 0 as x -> -inf and
    f'(x) -> 1 as x -> +inf.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Matrix SoftPlus of `X`.
    """

    @staticmethod
    def applied_fct(s):
        return softplus(s)

    @staticmethod
    def derivative(s):
        return 1 / (
            1.0 + torch.pow(torch.tensor(2.0, device=s.device, dtype=s.dtype), -s)
        )

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_softplus.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(
            grad_output, s, U, s_modified, matrix_softplus.derivative
        )


class matrix_inv_softplus(Function):
    r"""
    Matrix inverse (scaled) SoftPlus of a symmetric matrix.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Matrix inverse SoftPlus of `X`.
    """

    @staticmethod
    def applied_fct(s):
        return inv_softplus(s)

    @staticmethod
    def derivative(s):
        return 1 / (
            1.0 - torch.pow(torch.tensor(2.0, device=s.device, dtype=s.dtype), -s)
        )

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_inv_softplus.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(
            grad_output, s, U, s_modified, matrix_inv_softplus.derivative
        )


class matrix_log(Function):
    r"""Matrix logarithm of a symmetric matrix.

    This function computes the matrix logarithm of a symmetric matrix.
    Given a SPD matrix :math:`X`, its eigendecomposition is computed as:

    .. math::

        X = U \Lambda U^{T}

    where :math:`U` is an orthogonal matrix of eigenvectors and :math:`\Lambda`
    is a diagonal matrix of positive eigenvalues.

    The matrix logarithm is then defined as:

    .. math::

        \log(X) = U \log(\Lambda) U^{T}

    This approach is central to the Log-Euclidean framework on SPD manifolds,
    where operations such as distance computation and averaging are performed
    in the vector space after applying the matrix logarithm.

    This class adapts the backpropagation according to the chain rule
    :cite:p:`ionescu2015matrix`, :cite:p:`huang2017riemannian`.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Matrix logarithm of `X`.

    See Also
    --------
    :class:`matrix_exp` : Inverse operation, maps back to SPD manifold.
    :func:`log_euclidean_distance` : Distance using matrix logarithm.
    :func:`log_euclidean_mean` : Mean under Log-Euclidean metric.
    :class:`~spd_learn.modules.LogEig` : Neural network layer using matrix logarithm.
    :class:`~spd_learn.functional.log_cholesky.cholesky_log` : Alternative via Cholesky decomposition.
    """

    @staticmethod
    def applied_fct(s):
        threshold = get_epsilon(s.dtype, "eigval_log")
        return s.clamp(min=threshold).log()

    @staticmethod
    def derivative(s):
        threshold = get_epsilon(s.dtype, "eigval_log")
        s_deriv = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        s_deriv[s <= threshold] = 0
        return s_deriv

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_log.applied_fct)
        threshold = get_epsilon(s.dtype, "eigval_log")
        min_eigenvalue = s.min()
        if numerical_config.warn_on_clamp and threshold > min_eigenvalue:
            warnings.warn(
                f"Eigenvalue clamping occurred in matrix_log: threshold "
                f"({threshold:.2e}) > min eigenvalue ({min_eigenvalue:.2e}). "
                f"This might lead to inaccurate results.",
                UserWarning,
            )
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_log.derivative)


class matrix_exp(Function):
    r"""Matrix exponential of a symmetric matrix.

    This function computes the matrix exponential of a symmetric matrix.
    For each symmetric matrix :math:`X`, the matrix exponential is computed
    via eigenvalue decomposition:

    .. math::
        X = U \Lambda U^\top

    .. math::
        \exp(X) = U \exp(\Lambda) U^\top

    where :math:`\Lambda` is the diagonal matrix of eigenvalues and :math:`U`
    is the matrix of eigenvectors. The exponential is applied element-wise to
    the eigenvalues.

    This function is used to map matrices back to the SPD manifold after
    operations have been performed in the vector space using the matrix
    logarithm.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Matrix exponential of `X`.

    See Also
    --------
    :class:`matrix_log` : Inverse operation, maps SPD to tangent space.
    :func:`log_euclidean_mean` : Mean under Log-Euclidean metric uses exp/log.
    :class:`~spd_learn.modules.ExpEig` : Neural network layer using matrix exponential.
    :class:`~spd_learn.functional.log_cholesky.cholesky_exp` : Alternative via Cholesky decomposition.
    """

    @staticmethod
    def applied_fct(s):
        return s.exp()

    @staticmethod
    def derivative(s):
        return s.exp()

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_exp.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_exp.derivative)


class clamp_eigvals(Function):
    """Rectification of the eigenvalues of a symmetric matrix.

    This function computes the regularized matrix logarithm of a symmetric
    matrix `X`. It also adapts the backpropagation according to the chain
    rule :cite:p:`ionescu2015matrix`, :cite:p:`huang2017riemannian`.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.
    threshold : float
        Threshold for numerical stability.

    Returns
    -------
    torch.Tensor
        Regularized matrix.

    See Also
    --------
    :class:`~spd_learn.modules.ReEig` : Neural network layer using eigenvalue clamping.
    :class:`matrix_log` : Matrix logarithm (often used after clamping).
    """

    @staticmethod
    def applied_fct(s, threshold):
        return s.clamp(min=threshold)

    @staticmethod
    def applied_fct_complex(s, threshold):
        return clamp_complex(s, threshold)

    @staticmethod
    def derivative(s, threshold):
        s_deriv = torch.zeros_like(s)
        s_deriv[s > threshold] = 1
        return s_deriv

    @staticmethod
    def forward(ctx, X, threshold):
        if torch.is_complex(X):
            function_clamp = clamp_eigvals.applied_fct_complex
        else:
            function_clamp = clamp_eigvals.applied_fct

        output, s, U, s_modified = modeig_forward(X, function_clamp, threshold)
        ctx.save_for_backward(s, U, s_modified)
        ctx.threshold = threshold
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        threshold = ctx.threshold
        return modeig_backward(
            grad_output, s, U, s_modified, clamp_eigvals.derivative, threshold
        ), None


class abs_eigvals(Function):
    """Absolute value of the eigenvalues of a symmetric matrix.

    This function applies the absolute value function to the eigenvalues of a
    symmetric matrix and returns the modified matrix.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Modified matrix whose eigenvalues are `abs(eig(X))`.
    """

    @staticmethod
    def applied_fct(s):
        return s.abs()

    @staticmethod
    def derivative(s):
        return s.sign()

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, abs_eigvals.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, abs_eigvals.derivative)


class matrix_power(Function):
    """Computes the matrix power.

    This function computes the matrix power of a symmetric matrix `X` via
    Hermitian eigen decomposition.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.
    exponent : float
        Exponent to raise the matrix to.

    Returns
    -------
    torch.Tensor
        `X` raised to the power of `exponent`.

    Notes
    -----
    For numerical stability, eigenvalues are clamped to a minimum threshold
    before applying the power operation. This prevents NaN/Inf values for
    fractional or negative exponents with small eigenvalues.

    See Also
    --------
    :class:`matrix_sqrt` : Special case with exponent 0.5.
    :class:`matrix_inv_sqrt` : Special case with exponent -0.5.
    :func:`~spd_learn.functional.airm_geodesic` : Uses matrix power for geodesics.
    """

    @staticmethod
    def applied_fct(s, exponent):
        threshold = get_epsilon(s.dtype, "eigval_power")
        return s.clamp(min=threshold).pow(exponent=exponent)

    @staticmethod
    def derivative(s, exponent):
        threshold = get_epsilon(s.dtype, "eigval_power")
        s_clamped = s.clamp(min=threshold)
        s_deriv = exponent * s_clamped.pow(exponent=exponent - 1.0)
        # pick subgradient 0 for clamped eigenvalues
        s_deriv[s <= threshold] = 0
        return s_deriv

    @staticmethod
    def forward(ctx, X, exponent):
        output, s, U, s_modified = modeig_forward(X, matrix_power.applied_fct, exponent)
        ctx.save_for_backward(s, U, s_modified)
        ctx.exponent = exponent
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        exponent = ctx.exponent
        return modeig_backward(
            grad_output, s, U, s_modified, matrix_power.derivative, exponent
        ), None


class matrix_sqrt(Function):
    """Matrix square root.

    This function computes the matrix square root of a symmetric positive
    definite matrix `X` via Hermitian eigen decomposition.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric positive definite matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Matrix square root of `X`.

    See Also
    --------
    :class:`matrix_sqrt_inv` : Computes both square root and inverse square root.
    :class:`matrix_inv_sqrt` : Computes only inverse square root.
    :func:`~spd_learn.functional.airm_geodesic` : Uses matrix square root for geodesics.
    :func:`~spd_learn.functional.bures_wasserstein_distance` : Uses matrix square root.
    """

    @staticmethod
    def applied_fct(s):
        threshold = get_epsilon(s.dtype, "eigval_sqrt")
        return s.clamp(min=threshold).sqrt()

    @staticmethod
    def derivative(s):
        threshold = get_epsilon(s.dtype, "eigval_sqrt")
        sder = s.rsqrt() / 2
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= threshold] = 0
        return sder

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_sqrt.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, matrix_sqrt.derivative)


class matrix_inv_sqrt(Function):
    """Inverse matrix square root.

    This function computes the inverse of the matrix square root of a symmetric
    positive definite matrix `X` via Hermitian eigen decomposition.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric positive definite matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Inverse matrix square root of `X`.
    """

    @staticmethod
    def applied_fct(s):
        threshold = get_epsilon(s.dtype, "eigval_inv_sqrt")
        return s.clamp(min=threshold).rsqrt()

    @staticmethod
    def derivative(s):
        threshold = get_epsilon(s.dtype, "eigval_inv_sqrt")
        sder = -0.5 * s.pow(-1.5)
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= threshold] = 0
        return sder

    @staticmethod
    def forward(ctx, X):
        output, s, U, s_modified = modeig_forward(X, matrix_inv_sqrt.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(
            grad_output, s, U, s_modified, matrix_inv_sqrt.derivative
        )


class matrix_sqrt_inv(Function):
    """Matrix square root and inverse matrix square root.

    This function computes the matrix square root and its inverse for a
    symmetric positive definite matrix `X` via Hermitian eigen decomposition.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric positive definite matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        Matrix square root of `X`.
    torch.Tensor
        Inverse of the matrix square root of `X`.

    See Also
    --------
    :class:`matrix_sqrt` : Computes only matrix square root.
    :class:`matrix_inv_sqrt` : Computes only inverse square root.
    :func:`~spd_learn.functional.airm_geodesic` : Uses matrix sqrt/invsqrt for geodesics.
    :func:`parallel_transport_airm` : Uses matrix sqrt/invsqrt for transport.
    """

    @staticmethod
    def forward(ctx, X):
        output_sqrt, s, U, s_sqrt = modeig_forward(X, matrix_sqrt.applied_fct)
        s_invsqrt = matrix_inv_sqrt.applied_fct(s)
        output_invsqrt = (
            U @ torch.diag_embed(s_invsqrt).to(dtype=X.dtype) @ U.transpose(-1, -2)
        )
        ctx.save_for_backward(s, U, s_sqrt, s_invsqrt)
        return output_sqrt, output_invsqrt

    @staticmethod
    def backward(ctx, grad_output_sqrt, grad_output_invsqrt):
        s, U, s_sqrt, s_invsqrt = ctx.saved_tensors
        return modeig_backward(
            grad_output_sqrt, s, U, s_sqrt, matrix_sqrt.derivative
        ) + modeig_backward(
            grad_output_invsqrt, s, U, s_invsqrt, matrix_inv_sqrt.derivative
        )


def sym_to_upper(X, preserve_norm=True, upper=True):
    r"""Vectorizes symmetric matrices by extracting triangular elements.

    This function extracts the upper (or lower) triangular elements of symmetric
    matrices. When ``preserve_norm=True`` (the default), a :math:`\sqrt{2}` scaling
    is applied to off-diagonal elements so that the Euclidean norm of the resulting
    vector equals the Frobenius norm of the original matrix:

    .. math::

        \|z\|_2 = \|V\|_F

    This norm-preserving property is essential for tangent-space machine learning,
    ensuring that distances computed in the vectorized Euclidean space correspond
    to intrinsic Riemannian distances on the manifold.

    When ``preserve_norm=False``, no scaling is applied and the raw triangular
    elements are extracted (equivalent to the classical ``vech`` operation).

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrices with shape `(..., n, n)`.
    preserve_norm : bool, default=True
        If True, applies sqrt(2) scaling to off-diagonal elements so that
        ``||vec(X)||_2 = ||X||_F``. If False, extracts raw triangular elements.
    upper : bool, default=True
        If True, extracts upper triangular elements. If False, extracts lower
        triangular elements.

    Returns
    -------
    torch.Tensor
        Vectorized triangular part with shape `(..., n(n+1)/2)`.

    See Also
    --------
    :func:`vec_to_sym` : Inverse operation, reconstructs symmetric matrix.
    :class:`~spd_learn.modules.LogEig` : Uses this vectorization after matrix log.

    References
    ----------
    See :cite:p:`barachant2013classification` for tangent space classification.

    Examples
    --------
    >>> import torch
    >>> X = torch.tensor([[1., 2.], [2., 3.]])
    >>> # With norm preservation (default)
    >>> v = sym_to_upper(X)  # [1., 2*sqrt(2), 3.]
    >>> # Without norm preservation
    >>> v_raw = sym_to_upper(X, preserve_norm=False)  # [1., 2., 3.]
    """
    assert X.ndim >= 2
    assert X.shape[-1] == X.shape[-2]
    ndim = X.shape[-1]

    if upper:
        ixs = torch.triu_indices(ndim, ndim, offset=0)
    else:
        ixs = torch.tril_indices(ndim, ndim, offset=0)

    x_vec = X[..., ixs[0], ixs[1]]

    if preserve_norm:
        # multiply off-diagonal elements to preserve the norm
        off_diagonal_mask = ixs[0] != ixs[1]
        multipliers = torch.ones_like(x_vec)
        multipliers[..., off_diagonal_mask] = sqrt(2)
        x_vec = x_vec * multipliers

    return x_vec


def vec_to_sym(x_vec, preserve_norm=True, upper=True):
    r"""Reconstructs symmetric matrices from vectorization.

    This function is the inverse of :func:`sym_to_upper`. It reconstructs
    symmetric matrices from their vectorized triangular representation.

    When ``preserve_norm=True``, the inverse :math:`1/\sqrt{2}` scaling is applied
    to off-diagonal elements to recover the original matrix values from a
    norm-preserving vectorization.

    When ``preserve_norm=False``, no scaling is applied (inverse of raw ``vech``).

    Parameters
    ----------
    x_vec : torch.Tensor
        Vectorized triangular matrices with shape `(..., n(n+1)/2)`.
    preserve_norm : bool, default=True
        If True, applies inverse sqrt(2) scaling to off-diagonal elements.
        If False, uses raw values without scaling.
    upper : bool, default=True
        If True, reconstructs from upper triangular representation.
        If False, reconstructs from lower triangular representation.

    Returns
    -------
    torch.Tensor
        Symmetric matrices with shape `(..., n, n)`.

    See Also
    --------
    :func:`sym_to_upper` : Forward operation, vectorizes symmetric matrices.
    :class:`~spd_learn.modules.ExpEig` : Maps tangent vectors back to SPD manifold.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import sym_to_upper, vec_to_sym
    >>> X = torch.tensor([[1., 2.], [2., 3.]])
    >>> v = sym_to_upper(X)
    >>> X_reconstructed = vec_to_sym(v)
    >>> torch.allclose(X, X_reconstructed)
    True
    """
    ndim = (sqrt(1 + 8 * x_vec.shape[-1]) - 1) / 2
    assert ndim == int(ndim)
    ndim = int(ndim)

    if upper:
        ixs = torch.triu_indices(ndim, ndim, offset=0)
    else:
        ixs = torch.tril_indices(ndim, ndim, offset=0)

    od_mask = ixs[0] != ixs[1]

    X = torch.empty(
        (*x_vec.shape[:-1], ndim, ndim), device=x_vec.device, dtype=x_vec.dtype
    )
    X[..., ixs[0], ixs[1]] = x_vec

    if preserve_norm:
        # divide off-diagonal elements to undo norm-preserving scaling
        X[..., ixs[0, od_mask], ixs[1, od_mask]] /= sqrt(2)

    # Mirror to make symmetric
    X[..., ixs[1, od_mask], ixs[0, od_mask]] = X[..., ixs[0, od_mask], ixs[1, od_mask]]
    return X


def clamp_complex(x: torch.Tensor, min_mag: float) -> torch.Tensor:
    """Clamp the magnitude of a complex tensor `x` so that `|x[i]| >= min_mag`.

    `min_mag` can be a Python float or a real-valued `torch.Tensor`.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor.
    min_mag : float
        Minimum magnitude.

    Returns
    -------
    torch.Tensor
        Clamped complex tensor.
    """
    mag = x.abs()
    if isinstance(min_mag, torch.Tensor):
        if min_mag.is_complex():
            min_mag = min_mag.real
        min_mag = min_mag.to(mag.dtype)
    else:
        min_mag = float(min_mag)
    mag_clamped = mag.clamp(min=min_mag)
    safe_mag = mag.clamp(min=1e-12)
    return x * (mag_clamped / safe_mag)


def orthogonal_polar_factor(W: torch.Tensor) -> torch.Tensor:
    r"""Compute the orthogonal polar factor of a matrix.

    Projects a matrix onto the Stiefel manifold :math:`\text{St}(n, k)` via
    polar decomposition. The orthogonal polar factor is:

    .. math::

        W_{\perp} = W (W^\top W)^{-1/2}

    This is the unique matrix with orthonormal columns that is closest to
    :math:`W` in Frobenius norm.

    Parameters
    ----------
    W : torch.Tensor
        Input matrix with shape `(..., n, k)` where `n >= k`.

    Returns
    -------
    torch.Tensor
        Orthogonal polar factor with shape `(..., n, k)` satisfying
        :math:`W_{\perp}^\top W_{\perp} = I_k`.

    Notes
    -----
    This function uses the matrix inverse square root computed via
    eigendecomposition for numerical stability.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import orthogonal_polar_factor
    >>> W = torch.randn(8, 4)
    >>> W_orth = orthogonal_polar_factor(W)
    >>> # Verify orthonormality
    >>> torch.allclose(W_orth.T @ W_orth, torch.eye(4), atol=1e-5)
    True

    See Also
    --------
    :class:`~spd_learn.functional.matrix_inv_sqrt` : Matrix inverse square root.
    :func:`~spd_learn.init.stiefel_` : In-place Stiefel initialization.
    """
    return W @ matrix_inv_sqrt.apply(W.mT @ W)
