# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of GBWAtt (Jin et al., KDD, 2026).
# Reference implementation: https://github.com/jjscc/GBWAtt

import math

import torch

from torch import nn
from torch.nn.utils import parametrize

from spd_learn.functional import sample_covariance
from spd_learn.functional.metrics.generalized_bures_wasserstein import (
    _bw_squared,
    _require_spd,
    _safe_distance,
    bw_attention_mean,
    gbw_chart,
    gbw_distance,
    gbw_inverse_chart,
    gbw_mean,
)

from .bilinear import BiMap
from .manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite


class GBWGeometry(nn.Module):
    r"""Parametrize a shared SPD metric and positive deformation power.

    See :cite:p:`jin2026robust`.

    Parameters
    ----------
    n_features : int
        SPD matrix dimension. Metric starts at identity and is learned
        through the existing symmetric matrix exponential parametrization.
    power : float, default=1.5
        Positive deformation exponent; zero/negative powers are excluded
        here.
    learn_power : bool, default=False
        Learn power via softplus. Paper experiments use fixed 1 or 1.5.
    learn_metric : bool, default=True
        Whether the common metric is learned. False fixes it at identity.

    Notes
    -----
    Metric and power are registered, serializable state, follow .to(), and
    need no manifold optimizer. Ordinary optimizers operate on unconstrained
    originals; this differs from Geoopt retractions. No running statistics.
    """

    metric: torch.Tensor
    power: torch.Tensor

    def __init__(self, n_features, power=1.5, learn_power=False, learn_metric=True):
        super().__init__()
        if n_features < 1 or not math.isfinite(power) or power <= 0:
            raise ValueError("n_features and power must be positive and finite")
        self.register_parameter(
            "metric", nn.Parameter(torch.eye(n_features), requires_grad=learn_metric)
        )
        parametrize.register_parametrization(
            self, "metric", SymmetricPositiveDefinite()
        )
        self.register_parameter(
            "power", nn.Parameter(torch.tensor(float(power)), requires_grad=learn_power)
        )
        parametrize.register_parametrization(
            self, "power", PositiveDefiniteScalar(mapping="softplus")
        )

    def forward(self, x):
        r"""Map SPD matrices into generalized BW coordinates.

        Parameters
        ----------
        x : torch.Tensor
            SPD matrices of shape (..., n_features, n_features).

        Returns
        -------
        torch.Tensor
            Charted matrices with the input shape.
        """
        return gbw_chart(x, self.metric, self.power)

    def inverse(self, x):
        r"""Map charted SPD matrices back to the original coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Charted matrices of shape (..., n_features, n_features).

        Returns
        -------
        torch.Tensor
            SPD matrices with the input shape.
        """
        return gbw_inverse_chart(x, self.metric, self.power)

    def distance(self, x, y, *, squared=False):
        r"""Compute pairwise/broadcast distances using the current geometry.

        Parameters
        ----------
        x : torch.Tensor
            Real SPD matrices of shape (..., n, n).
        y : torch.Tensor
            Real SPD matrices broadcastable with x.
        squared : bool, default=False
            Return squared distances instead of distances.

        Returns
        -------
        torch.Tensor
            Distances over the broadcast leading axes, preserving input
            dtype/device.
        """
        return gbw_distance(x, y, self.metric, self.power, squared=squared)

    def mean(self, values, weights, *, n_iter=10, method="fixed_point"):
        r"""Weighted barycenters; see functional ``gbw_mean`` for axes/accuracy.

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
        """
        return gbw_mean(
            values, weights, self.metric, self.power, n_iter=n_iter, method=method
        )


class GBWAttention(nn.Module):
    r"""Apply generalized BW attention to SPD tokens.

    See :cite:p:`jin2026robust`.

    Parameters
    ----------
    in_features : int
        Input matrix dimension.
    out_features : int
        Output matrix dimension.
    power : float, default=1.5
        Positive power deformation, fixed unless learn_power=True.
    learn_power : bool, default=False
        Enable constrained learnable power.
    n_iter : int, default=10
        Unrolled BW barycenter iterations (approximation, not
        convergence).
    aggregation : {'fixed_point', 'root'}, default='fixed_point'
        Paper Eq. 22 iteration or author weighted-root approximation.
        Fixed-point accumulation uses float64 for float32 inputs,
        returning checked float32 output with differentiable casts. Root
        stays in input dtype.

    Notes
    -----
    Independent Stiefel Q/K/V projections precede the common chart.
    Attention aggregates tokens within each sample; no masks or padding are
    supported.
    """

    def __init__(
        self,
        in_features,
        out_features,
        power=1.5,
        learn_power=False,
        n_iter=10,
        aggregation="fixed_point",
        *,
        temperature=None,
    ):
        super().__init__()
        if not 1 <= out_features <= in_features:
            raise ValueError("require 1 <= out_features <= in_features")
        if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if aggregation not in {"fixed_point", "root"}:
            raise ValueError("aggregation must be 'fixed_point' or 'root'")
        if temperature is not None and (
            not math.isfinite(temperature) or temperature <= 0
        ):
            raise ValueError("temperature must be positive and finite")
        self.in_features = in_features
        self.out_features = out_features
        self.n_iter = n_iter
        self.aggregation = aggregation
        self.temperature = None if temperature is None else float(temperature)
        self.q_trans = BiMap(in_features, out_features, init_method="orthogonal")
        self.k_trans = BiMap(in_features, out_features, init_method="orthogonal")
        self.v_trans = BiMap(in_features, out_features, init_method="orthogonal")
        self.geometry = GBWGeometry(out_features, power, learn_power)

    def forward(self, x, *, return_weights=False):
        r"""Attend to a sequence of SPD tokens.

        Parameters
        ----------
        x : torch.Tensor
            SPD tokens of shape (batch, tokens, in_features, in_features).
        return_weights : bool, default=False
            Also return attention weights.

        Returns
        -------
        torch.Tensor
            SPD outputs of shape (batch, tokens, out_features, out_features).
        weights : torch.Tensor, optional
            Attention of shape (batch, tokens, tokens), returned when
            return_weights=True.
        """
        if x.ndim != 4 or x.shape[-2:] != (self.in_features, self.in_features):
            raise ValueError("expected (batch, tokens, in_features, in_features)")
        if x.shape[1] == 0:
            raise ValueError("at least one token is required")
        metric, power = self.geometry.metric, self.geometry.power
        q, k, v = [
            gbw_chart(proj(x), metric, power)
            for proj in (self.q_trans, self.k_trans, self.v_trans)
        ]
        distance = _safe_distance(_bw_squared(q.unsqueeze(2), k.unsqueeze(1))) / power
        if self.temperature is not None:
            distance = distance / self.temperature
        weights = (1 / (1 + torch.log1p(distance))).softmax(dim=-1)
        mean = bw_attention_mean(
            v, weights, n_iter=self.n_iter, method=self.aggregation
        )
        output = gbw_inverse_chart(mean, metric, power)
        return (output, weights) if return_weights else output


class GBWPatchCovariance(nn.Module):
    r"""Estimate chronological covariance tokens from learned time series.

    See :cite:p:`jin2026robust`.

    Parameters
    ----------
    n_patches : int, default=3
        Number of chronological patches.
    epsilon : float, default=1e-05
        Positive absolute diagonal loading after optional trace
        normalization.
    trace_normalize : bool, default=True
        Normalize each covariance by its trace before diagonal loading.

    Notes
    -----
    Sample covariance uses divisor T-1. All samples are retained; earlier
    patches receive the remainder. Each patch needs at least two samples.
    """

    def __init__(self, n_patches=3, epsilon=1e-5, trace_normalize=True):
        super().__init__()
        if not isinstance(n_patches, int) or n_patches < 1:
            raise ValueError("n_patches must be a positive integer")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        self.n_patches = n_patches
        self.epsilon = epsilon
        self.trace_normalize = trace_normalize

    def forward(self, x):
        r"""Estimate regularized covariance tokens.

        Parameters
        ----------
        x : torch.Tensor
            Features of shape (batch, features, time).

        Returns
        -------
        torch.Tensor
            SPD tokens of shape (batch, n_patches, features, features).
        """
        if x.ndim != 3 or x.shape[-1] < 2 * self.n_patches:
            raise ValueError("expected (batch,features,time), >=2 times per patch")
        covs = []
        for patch in torch.tensor_split(x, self.n_patches, dim=-1):
            cov = sample_covariance(patch)
            if self.trace_normalize:
                trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
                cov = cov / trace.clamp_min(torch.finfo(x.dtype).tiny)
            covs.append(
                cov
                + self.epsilon * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
            )
        return _require_spd(torch.stack(covs, dim=1), "patch covariance output")
