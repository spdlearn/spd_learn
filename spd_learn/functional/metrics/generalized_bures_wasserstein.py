# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of GBWAtt (Jin et al., KDD, 2026).
# Reference implementation: https://github.com/jjscc/GBWAtt

import math

import torch

from torch.autograd.function import once_differentiable

from ..core import matrix_inv_sqrt, matrix_power, matrix_sqrt
from ..utils import ensure_sym


def _domain_error(stage):
    return (
        f"GBW {stage} cannot be represented as finite SPD at this precision; "
        "use float64, rescale covariance units, or explicitly increase covariance "
        "regularization. No hidden SPD repair is applied."
    )


def _require_spd(x, stage):
    """Certify numerical SPD without repairing spectra or changing dtype."""
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("GBW geometry requires real float32/float64 matrices")
    if x.ndim < 2 or x.shape[-1] != x.shape[-2] or x.shape[-1] < 1 or x.numel() == 0:
        raise ValueError("GBW geometry requires nonempty square matrices")
    if not torch.isfinite(x).all():
        raise ValueError(_domain_error(stage))
    scale = x.detach().abs().amax(dim=(-2, -1), keepdim=True)
    scale = scale.clamp_min(torch.finfo(x.dtype).tiny)
    if ((x - x.mT).abs() > 100 * torch.finfo(x.dtype).eps * scale).any():
        raise ValueError(f"GBW {stage} must be symmetric")
    x = ensure_sym(x)
    with torch.no_grad():
        try:
            spectrum = torch.linalg.eigvalsh(x)
            _, info = torch.linalg.cholesky_ex(x)
        except RuntimeError as error:
            raise ValueError(_domain_error(stage)) from error
        if (
            not torch.isfinite(spectrum).all()
            or (spectrum <= 0).any()
            or (info != 0).any()
        ):
            raise ValueError(_domain_error(stage))
    return x


def _spectral_spd(operator, x, stage, *parameters):
    try:
        result = operator.apply(x, *parameters)
    except RuntimeError as error:
        raise ValueError(_domain_error(stage)) from error
    return _require_spd(ensure_sym(result), stage)


def _positive_power(power):
    if isinstance(power, torch.Tensor):
        if (
            power.ndim != 0
            or power.is_complex()
            or not torch.isfinite(power)
            or power <= 0
        ):
            raise ValueError("power must be a finite positive scalar")
    elif not math.isfinite(power) or power <= 0:
        raise ValueError("power must be a finite positive scalar")


def gbw_chart(x: torch.Tensor, metric: torch.Tensor, power=1.0) -> torch.Tensor:
    r"""Map SPD matrices into power-deformed BW coordinates.

    Parameters
    ----------
    x : torch.Tensor
        Real SPD matrices of shape (..., n, n).
    metric : torch.Tensor
        SPD metric of shape (..., n, n), broadcastable with x.
    power : float or torch.Tensor, default=1.0
        Positive scalar deformation exponent.

    Returns
    -------
    torch.Tensor
        Real SPD matrices with broadcast leading axes.

    Notes
    -----
    Computes M^(-1/2) X^power M^(-1/2). The common metric must broadcast
    against the input; host spectral floors apply.
    """
    _positive_power(power)
    x = _require_spd(x, "chart input")
    metric = _require_spd(metric, "metric")
    whitening = _spectral_spd(matrix_inv_sqrt, metric, "inverse metric square root")
    powered = _spectral_spd(matrix_power, x, "forward power", power)
    return _require_spd(ensure_sym(whitening @ powered @ whitening), "chart output")


def gbw_inverse_chart(x: torch.Tensor, metric: torch.Tensor, power=1.0):
    r"""Invert the power-deformed BW chart.

    Parameters
    ----------
    x : torch.Tensor
        Real SPD matrices of shape (..., n, n).
    metric : torch.Tensor
        SPD metric of shape (..., n, n), broadcastable with x.
    power : float or torch.Tensor, default=1.0
        Positive scalar deformation exponent.

    Returns
    -------
    torch.Tensor
        Real SPD matrices with broadcast leading axes.

    Notes
    -----
    Computes (M^(1/2) X M^(1/2))^(1/power). Use the same metric and power as
    the forward chart.
    """
    _positive_power(power)
    x = _require_spd(x, "inverse chart input")
    metric = _require_spd(metric, "metric")
    root = _spectral_spd(matrix_sqrt, metric, "metric square root")
    congruence = _require_spd(ensure_sym(root @ x @ root), "inverse chart congruence")
    return _spectral_spd(matrix_power, congruence, "inverse chart output", 1 / power)


class _AlignedSquaredDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, same):
        # Bhatia/Jain/Lim, arXiv:1712.01504, Theorem 1: minimize ||a-bO||_F.
        # Positive factor scaling leaves the polar optimizer unchanged and avoids
        # overflow in its cross product. No differentiation of SVD eigenvectors.
        unit_a = a / a.abs().amax(dim=(-2, -1), keepdim=True)
        unit_b = b / b.abs().amax(dim=(-2, -1), keepdim=True)
        try:
            u, _, vh = torch.linalg.svd(unit_b.mT @ unit_a, full_matrices=False)
        except RuntimeError as error:
            raise ValueError(_domain_error("distance alignment")) from error
        rotation = u @ vh
        residual = a - b @ rotation
        residual = torch.where(same, torch.zeros_like(residual), residual)
        ctx.save_for_backward(residual, rotation)
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return residual.square().sum(dim=(-2, -1))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Envelope theorem: at the optimal O its constrained first variation is
        # zero. These are exact first derivatives; higher derivatives unsupported.
        residual, rotation = ctx.saved_tensors
        gradient = 2 * grad_output[..., None, None] * residual
        return (
            gradient.sum_to_size(ctx.a_shape),
            (-gradient @ rotation.mT).sum_to_size(ctx.b_shape),
            None,
        )


def _bw_squared(x: torch.Tensor, y: torch.Tensor):
    a = _spectral_spd(matrix_sqrt, x, "distance x square root")
    b = _spectral_spd(matrix_sqrt, y, "distance y square root")
    same = (x == y).all(dim=(-2, -1), keepdim=True)
    squared = _AlignedSquaredDistance.apply(a, b, same)
    if not torch.isfinite(squared).all():
        raise ValueError(_domain_error("squared distance"))
    return squared


def _safe_distance(squared: torch.Tensor):
    # The nonnegative residual form needs no arbitrary distance dead zone.
    # Keep the inactive sqrt branch nonsingular so zero chooses zero subgradient.
    positive = squared > 0
    safe = torch.where(positive, squared, torch.ones_like(squared)).sqrt()
    return torch.where(positive, safe, torch.zeros_like(squared))


def gbw_distance(x, y, metric, power=1.0, *, squared=False):
    r"""Compute the power-deformed generalized BW distance.

    Parameters
    ----------
    x : torch.Tensor
        Real SPD matrices of shape (..., n, n).
    y : torch.Tensor
        Real SPD matrices broadcastable with x.
    metric : torch.Tensor
        Common SPD metric broadcastable with both inputs.
    power : float or torch.Tensor, default=1.0
        Positive scalar deformation exponent.
    squared : bool, default=False
        Return squared distances instead of distances.

    Returns
    -------
    torch.Tensor
        Distances over the broadcast leading axes, preserving input
        dtype/device.

    Notes
    -----
    The distance includes division by power. First-order derivatives are
    supported; coincident points use a zero subgradient.
    """
    value = _bw_squared(gbw_chart(x, metric, power), gbw_chart(y, metric, power))
    value = torch.where((x == y).all(dim=(-2, -1)), torch.zeros_like(value), value)
    scale = torch.as_tensor(power, dtype=value.dtype, device=value.device)
    return value / scale**2 if squared else _safe_distance(value) / scale


def bw_attention_mean(values, weights, *, n_iter=10, method="fixed_point"):
    r"""Aggregate SPD values using convex attention weights.

    Parameters
    ----------
    values : torch.Tensor
        SPD values of shape (..., keys, n, n).
    weights : torch.Tensor
        Nonnegative weights of shape (..., queries, keys), summing to one
        over keys.
    n_iter : int, default=10
        Positive number of fixed-point updates.
    method : {'fixed_point', 'root'}, default='fixed_point'
        Fixed-point BW iterations or the weighted-root approximation.

    Returns
    -------
    torch.Tensor
        SPD means of shape (..., queries, n, n).

    Notes
    -----
    Fixed-point updates approximate the barycenter without a convergence
    guarantee. Float32 accumulation uses float64 internally in the current
    implementation; root aggregation retains its input dtype.
    """
    if method not in {"fixed_point", "root"}:
        raise ValueError("method must be 'fixed_point' or 'root'")
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    if values.ndim < 3 or weights.ndim < 2:
        raise ValueError(
            "expected (..., keys,n,n) values and (..., queries,keys) weights"
        )
    if weights.shape[-1] != values.shape[-3] or values.shape[-3] == 0:
        raise ValueError("weights must match a nonempty key axis")
    values = _require_spd(values, "mean input")
    if weights.dtype != values.dtype or weights.device != values.device:
        raise ValueError("weights and values must share dtype and device")
    if (
        not torch.isfinite(weights).all()
        or (weights < 0).any()
        or not torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    ):
        raise ValueError("weights must be finite convex rows")
    if method == "root":
        mean_root = torch.einsum(
            "...qk,...kij->...qij",
            weights,
            _spectral_spd(matrix_sqrt, values, "value square roots"),
        )
        return _require_spd(ensure_sym(mean_root @ mean_root.mT), "root mean output")
    output_dtype = values.dtype
    if output_dtype == torch.float32:
        values, weights = values.double(), weights.double()
    mean = _require_spd(
        torch.einsum("...qk,...kij->...qij", weights, values), "mean initialization"
    )
    for _ in range(n_iter):
        root = _spectral_spd(matrix_sqrt, mean, "mean square root")
        inv = _spectral_spd(matrix_inv_sqrt, mean, "inverse mean square root")
        cross = root.unsqueeze(-3) @ values.unsqueeze(-4) @ root.unsqueeze(-3)
        cross = _require_spd(ensure_sym(cross), "mean cross product")
        total = torch.einsum(
            "...qk,...qkij->...qij",
            weights,
            _spectral_spd(matrix_sqrt, cross, "mean cross square root"),
        )
        factor = inv @ total
        mean = _require_spd(ensure_sym(factor @ factor.mT), "mean iteration output")
    return _require_spd(mean.to(dtype=output_dtype), "mean output")


def gbw_mean(values, weights, metric, power=1.0, *, n_iter=10, method="fixed_point"):
    r"""Aggregate SPD values using convex attention weights.

    Parameters
    ----------
    values : torch.Tensor
        SPD values of shape (..., keys, n, n).
    weights : torch.Tensor
        Nonnegative weights of shape (..., queries, keys), summing to one
        over keys.
    metric : torch.Tensor
        Common SPD metric; singleton axes must broadcast over keys and
        queries.
    power : float or torch.Tensor, default=1.0
        Positive scalar deformation exponent.
    n_iter : int, default=10
        Positive number of fixed-point updates.
    method : {'fixed_point', 'root'}, default='fixed_point'
        Fixed-point BW iterations or the weighted-root approximation.

    Returns
    -------
    torch.Tensor
        SPD means of shape (..., queries, n, n).

    Notes
    -----
    Fixed-point updates approximate the barycenter without a convergence
    guarantee. Float32 accumulation uses float64 internally in the current
    implementation; root aggregation retains its input dtype.
    """
    mean = bw_attention_mean(
        gbw_chart(values, metric, power), weights, n_iter=n_iter, method=method
    )
    return gbw_inverse_chart(mean, metric, power)
