# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch

from .numerical import get_epsilon, get_loewner_threshold
from .utils import ensure_sym


def modeig_forward(X, applied_fct, *args):
    """Forward pass for the modified eigenvalue of a symmetric matrix.

    This function computes the forward pass for a function that modifies the
    eigenvalues of a symmetric matrix.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.
    applied_fct : callable
        Function to apply to the eigenvalues.
    *args : tuple
        Additional arguments for the applied function.

    Returns
    -------
    output : torch.Tensor
        Modified matrix after applying the function to the eigenvalues.
    s : torch.Tensor
        Eigenvalues of `X`.
    U : torch.Tensor
        Eigenvectors of `X`.
    s_modified : torch.Tensor
        Modified eigenvalues after applying the function.

    """
    s, U = torch.linalg.eigh(X)
    s_modified = applied_fct(s, *args)
    output = U @ torch.diag_embed(s_modified).to(dtype=X.dtype) @ U.transpose(-1, -2)
    return output, s, U, s_modified


def modeig_backward(grad_output, s, U, s_modified, derivative, *args):
    """Backward pass for the modified eigenvalue of a symmetric matrix.

    This function computes the backward pass for a function that modifies the
    eigenvalues of a symmetric matrix using the Loewner matrix formulation.

    Parameters
    ----------
    grad_output : torch.Tensor
        Gradient of the loss with respect to the output.
    s : torch.Tensor
        Eigenvalues of the input matrix.
    U : torch.Tensor
        Eigenvectors of the input matrix.
    s_modified : torch.Tensor
        Modified eigenvalues after applying the function.
    derivative : callable
        Derivative of the applied function with respect to the eigenvalues.
    *args : tuple
        Additional arguments for the derivative of the applied function.

    Returns
    -------
    grad_input : torch.Tensor
        Gradient of the loss with respect to the input.

    Notes
    -----
    The Loewner matrix L is computed as:

    .. math::
        L_{ij} = \\begin{cases}
            \\frac{f(\\lambda_i) - f(\\lambda_j)}{\\lambda_i - \\lambda_j}
            & \\text{if } \\lambda_i \\neq \\lambda_j \\\\
            f'(\\lambda_i) & \\text{if } \\lambda_i = \\lambda_j
        \\end{cases}

    For numerical stability, we use an adaptive threshold for detecting
    "equal" eigenvalues that scales with the magnitude of the eigenvalues.
    """
    # Compute Loewner matrix with adaptive threshold
    denominator = s.unsqueeze(-1) - s.unsqueeze(-1).transpose(-1, -2)

    # Use adaptive threshold that scales with eigenvalue magnitude
    threshold = get_loewner_threshold(s)
    is_eq = denominator.abs() < threshold
    denominator[is_eq] = 1.0

    # Case: sigma_i != sigma_j
    numerator = s_modified.unsqueeze(-1) - s_modified.unsqueeze(-1).transpose(-1, -2)

    # Case: sigma_i == sigma_j (use derivative instead)
    s_derivative = derivative(s, *args)
    numerator[is_eq] = (
        0.5
        * (s_derivative.unsqueeze(-1) + s_derivative.unsqueeze(-1).transpose(-1, -2))[
            is_eq
        ]
    )
    L = numerator / denominator

    grad_input = (
        U
        @ (L * (U.transpose(-1, -2) @ ensure_sym(grad_output) @ U))
        @ U.transpose(-1, -2)
    )

    return grad_input


def clamp_eigvals_func(X, threshold):
    """Clamps the eigenvalues of a symmetric matrix.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.
    threshold : float
        The threshold to clamp the eigenvalues.

    Returns
    -------
    torch.Tensor
        The matrix with clamped eigenvalues.
    """
    return modeig_forward(X, lambda eigvals: eigvals.clamp(min=threshold))[0]


def matrix_log_func(X):
    """Computes the matrix logarithm of a symmetric matrix.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        The matrix logarithm of `X`.

    Notes
    -----
    Eigenvalues are clamped to ensure numerical stability when taking
    the logarithm. The threshold is dtype-aware.
    """
    threshold = get_epsilon(X.dtype, "eigval_log")
    return modeig_forward(X, lambda eigvals: eigvals.clamp(min=threshold).log())[0]


def matrix_exp_func(X):
    """Computes the matrix exponential of a symmetric matrix.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape `(..., n, n)`.

    Returns
    -------
    torch.Tensor
        The matrix exponential of `X`.
    """
    return modeig_forward(X, lambda eigvals: eigvals.exp())[0]
