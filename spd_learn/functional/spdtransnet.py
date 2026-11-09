# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of SPDTransNet / SP-MHA (Seraphim et al., EUSIPCO, 2024).
# Reference implementation: https://github.com/MathieuSeraphim/SPDTransNet

from math import isfinite, isqrt

import torch

from .core import clamp_eigvals, matrix_exp, matrix_log, sym_to_upper, vec_to_sym


def triangular_dimension(matrix_size: int) -> int:
    r"""Return the symmetric-coordinate dimension of a matrix.

    Parameters
    ----------
    matrix_size : int
        Input SPD matrix dimension.

    Returns
    -------
    int
        matrix_size * (matrix_size + 1) // 2.
    """
    if (
        not isinstance(matrix_size, int)
        or isinstance(matrix_size, bool)
        or matrix_size < 1
    ):
        raise ValueError("matrix_size must be a positive integer")
    return matrix_size * (matrix_size + 1) // 2


def _check_real_finite(x: torch.Tensor) -> None:
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("SPDTransNet supports real float32 and float64 tensors")
    if x.numel() == 0:
        raise ValueError(
            "Empty inputs are not supported by SPDTransNet coordinate maps"
        )
    if not torch.isfinite(x).all():
        raise ValueError("input must be finite")


def spd_to_log_tokens(
    x: torch.Tensor,
    *,
    preserve_norm: bool = False,
    eigenvalue_floor: float | None = None,
    matrix_scale: float = 1.0,
) -> torch.Tensor:
    r"""Map SPD matrices to upper-triangle log coordinates.

    Parameters
    ----------
    x : torch.Tensor
        Real matrices of shape (..., n, n); SPD unless an explicit floor
        is supplied.
    preserve_norm : bool, default=False
        Scale off-diagonal coordinates by sqrt(2).
    eigenvalue_floor : float or None
        Explicit eigenvalue floor before the logarithm; None disables it.
    matrix_scale : float, default=1.0
        Positive matrix multiplier before the logarithm.

    Returns
    -------
    torch.Tensor
        Log coordinates of shape (..., n * (n + 1) // 2).

    Notes
    -----
    Scaling precedes flooring and the logarithm. An explicit floor is lossy;
    the host numerical log floor also applies.
    """
    _check_real_finite(x)
    if x.ndim < 2 or x.shape[-1] != x.shape[-2] or x.shape[-1] < 1:
        raise ValueError("expected nonempty square matrices on the last two axes")
    if not isfinite(matrix_scale) or matrix_scale <= 0:
        raise ValueError("matrix_scale must be finite and positive")
    if eigenvalue_floor is not None and (
        not isfinite(eigenvalue_floor) or eigenvalue_floor <= 0
    ):
        raise ValueError("eigenvalue_floor must be finite and positive, or None")
    tolerance = 10 * torch.finfo(x.dtype).eps
    scale = x.detach().abs().amax(dim=(-2, -1), keepdim=True)
    scale = scale.clamp_min(torch.finfo(x.dtype).tiny)
    if ((x - x.mT).abs() > tolerance * scale).any():
        raise ValueError("input matrices must be symmetric")
    x = (x + x.mT) * (0.5 * matrix_scale)
    with torch.no_grad():
        spectrum = torch.linalg.eigvalsh(x)
        if eigenvalue_floor is None:
            invalid = spectrum[..., 0] <= 0
        else:
            scale = spectrum.abs().amax(-1).clamp_min(torch.finfo(x.dtype).tiny)
            invalid = spectrum[..., 0] < -tolerance * scale
        if invalid.any():
            raise ValueError(
                "input must be SPD (or PSD with an explicit eigenvalue_floor)"
            )
    if eigenvalue_floor is not None:
        x = clamp_eigvals.apply(x, eigenvalue_floor)
    return sym_to_upper(matrix_log.apply(x), preserve_norm=preserve_norm)


def log_tokens_to_spd(
    tokens: torch.Tensor, *, preserve_norm: bool = False, matrix_scale: float = 1.0
) -> torch.Tensor:
    r"""Reconstruct matrices from symmetric log coordinates.

    Parameters
    ----------
    tokens : torch.Tensor
        Log coordinates of shape (..., n * (n + 1) // 2).
    preserve_norm : bool, default=False
        Scale off-diagonal coordinates by sqrt(2).
    matrix_scale : float, default=1.0
        Positive matrix multiplier before the logarithm.

    Returns
    -------
    torch.Tensor
        Reconstructed matrices of shape (..., n, n), after undoing
        matrix_scale.

    Notes
    -----
    Use the same basis and scale as the forward map. Previously floored
    eigenvalues cannot be recovered.
    """
    _check_real_finite(tokens)
    if tokens.ndim < 1:
        raise ValueError("expected tokens with a final coordinate axis")
    d = tokens.shape[-1]
    n = (isqrt(8 * d + 1) - 1) // 2
    if d < 1 or triangular_dimension(n) != d:
        raise ValueError("last dimension must be a positive triangular number")
    if not isfinite(matrix_scale) or matrix_scale <= 0:
        raise ValueError("matrix_scale must be finite and positive")
    return (
        matrix_exp.apply(vec_to_sym(tokens, preserve_norm=preserve_norm)) / matrix_scale
    )
