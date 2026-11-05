# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of Graph-CSPNet (Ju and Guan, IEEE TNNLS, 2023).
# Reference implementation: https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet

from collections.abc import Sequence
from copy import deepcopy
from math import isfinite
from typing import Literal

import torch

from torch import Tensor, nn

from spd_learn.functional.core import matrix_inv_sqrt, matrix_log
from spd_learn.functional.graph import normalize_spd_adjacency, spd_graph_aggregate
from spd_learn.functional.utils import ensure_sym
from spd_learn.modules.bilinear import BiMap


def _atomic_load_state_dict(module, state_dict, strict=True, assign=False):
    """Restore values and tensor ownership if PyTorch's partial load fails."""
    snapshot = deepcopy(module.state_dict())
    bindings = [
        (child, child._parameters.copy(), child._buffers.copy())
        for child in module.modules()
    ]
    try:
        return nn.Module.load_state_dict(
            module, state_dict, strict=strict, assign=assign
        )
    except Exception:
        # assign=True can replace Parameter/buffer objects. Restore bindings as
        # well as values, preserving references held by an existing optimizer.
        for child, parameters, buffers in bindings:
            child._parameters = parameters
            child._buffers = buffers
        nn.Module.load_state_dict(module, snapshot, strict=True, assign=False)
        raise


class TimeFrequencyCovariances(nn.Module):
    r"""Estimate covariance matrices from ordered filter-bank windows.

    See :cite:p:`Ju_2024`.

    Parameters
    ----------
    segments : sequence of tuple
        Ordered (band, start, stop) windows in samples; stop is exclusive.
    center : bool, default=False
        Subtract each channel mean within its window.
    shrinkage : float, default=0.0
        Mixture weight of the trace-scaled identity, in [0, 1].
    ridge : float, default=1e-05
        Positive absolute loading in squared signal units.

    Notes
    -----
    Without centering this computes a second moment, using divisor window
    length. Leading trials and channel order are preserved; filtering is
    external.
    """

    def __init__(
        self,
        segments: Sequence[tuple[int, int, int]],
        *,
        center: bool = False,
        shrinkage: float = 0.0,
        ridge: float = 1e-5,
    ) -> None:
        super().__init__()
        if not segments or any(b < 0 or s < 0 or e <= s for b, s, e in segments):
            raise ValueError("segments must contain valid (band, start, stop) entries")
        if not 0 <= shrinkage <= 1 or not isfinite(ridge) or not ridge > 0:
            raise ValueError("require 0 <= shrinkage <= 1 and ridge > 0")
        self.segments = tuple(segments)
        self.center = center
        self.shrinkage = shrinkage
        self.ridge = ridge

    def forward(self, input: Tensor) -> Tensor:
        r"""Estimate one covariance matrix per window.

        Parameters
        ----------
        input : torch.Tensor
            Filtered trials of shape (batch, bands, channels, samples).

        Returns
        -------
        torch.Tensor
            Covariances of shape (batch, nodes, channels, channels).
        """
        if input.ndim != 4 or not input.is_floating_point():
            raise ValueError("expected floating (trials, bands, channels, samples)")
        if not torch.isfinite(input).all():
            raise ValueError("signals must be finite")
        eye = torch.eye(input.shape[-2], device=input.device, dtype=input.dtype)
        nodes = []
        for band, start, stop in self.segments:
            if band >= input.shape[1] or stop > input.shape[-1]:
                raise ValueError("segment lies outside the signal")
            window = input[:, band, :, start:stop]
            if self.center:
                window = window - window.mean(-1, keepdim=True)
            covariance = window @ window.mT / (stop - start)
            scale = covariance.diagonal(dim1=-2, dim2=-1).mean(-1)
            nodes.append(
                (1 - self.shrinkage) * covariance
                + (self.shrinkage * scale[..., None, None] + self.ridge) * eye
            )
        output = ensure_sym(torch.stack(nodes, dim=1))
        if not torch.isfinite(output).all():
            raise ValueError("covariance overflow; rescale signal units or use float64")
        return output


class TimeFrequencyGraph(nn.Module):
    r"""Construct a time-frequency graph with training-fitted edge weights.

    See :cite:p:`Ju_2024`.

    Parameters
    ----------
    coordinates : torch.Tensor
        Node coordinates of shape (nodes, 2): time in seconds and
        frequency in Hz.
    time_radius : float
        Temporal neighborhood radius in seconds.
    frequency_radius : float
        Frequency neighborhood radius in Hz.
    components : torch.Tensor or None
        Optional component IDs of shape (nodes,); edges stay within a
        component.
    bandwidth : float, default=1.0
        Positive divisor of squared AIRM distances in Gaussian edge
        weights.
    neighborhood : {'box', 'forward'}, default='box'
        Symmetric midpoint box or the released forward-grid neighborhood.

    Notes
    -----
    fit estimates arithmetic training lattices and Gaussian AIRM edge
    weights. Forward returns the stored row-stochastic propagation without
    refitting. Coordinates must match covariance node order.
    """

    coordinates: Tensor
    propagation: Tensor
    lattice: Tensor
    fitted: Tensor
    topology: Tensor

    def __init__(
        self,
        coordinates: Tensor,
        *,
        time_radius: float,
        frequency_radius: float,
        components: Tensor | None = None,
        bandwidth: float = 1.0,
        neighborhood: Literal["box", "forward"] = "box",
    ) -> None:
        super().__init__()
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates):
            raise ValueError("coordinates must have shape (nodes, 2)")
        if not coordinates.is_floating_point() or not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite floating point")
        if (
            not all(map(isfinite, (time_radius, frequency_radius, bandwidth)))
            or time_radius < 0
            or frequency_radius < 0
            or not bandwidth > 0
        ):
            raise ValueError("radii must be nonnegative and bandwidth positive")
        if neighborhood not in ("box", "forward"):
            raise ValueError("neighborhood must be 'box' or 'forward'")
        delta = coordinates[:, None] - coordinates[None, :]
        topology = (delta[..., 0].abs() <= time_radius) & (
            delta[..., 1].abs() <= frequency_radius
        )
        if neighborhood == "forward":
            topology &= delta[..., 0] * delta[..., 1] >= 0
        if components is not None:
            if (
                components.shape != (len(coordinates),)
                or not torch.isfinite(components).all()
            ):
                raise ValueError("components must have one finite entry per node")
            components = components.to(coordinates.device)
            topology &= components[:, None] == components[None, :]
        topology.fill_diagonal_(False)
        self.bandwidth = bandwidth
        self.neighborhood = neighborhood
        self.register_buffer("coordinates", coordinates.detach().clone())
        self.register_buffer("topology", topology)
        self.register_buffer(
            "propagation", normalize_spd_adjacency(topology.to(coordinates))
        )
        # Empty lattice is resized when loading a fitted graph checkpoint.
        self.register_buffer("lattice", coordinates.new_empty(len(coordinates), 0, 0))
        self.register_buffer("fitted", torch.tensor(False, device=coordinates.device))

    @torch.no_grad()
    def fit(self, input: Tensor) -> "TimeFrequencyGraph":
        r"""Fit edge weights from training covariance nodes.

        Parameters
        ----------
        input : torch.Tensor
            Finite SPD matrices of shape (training_trials, nodes, channels,
            channels).

        Returns
        -------
        self : TimeFrequencyGraph
            The graph with fitted lattice and propagation buffers.
        """
        if (
            input.ndim != 4
            or input.shape[0] == 0
            or input.shape[1] != len(self.coordinates)
        ):
            raise ValueError(
                "expected nonempty (training_trials, nodes, channels, channels)"
            )
        if input.shape[-1] != input.shape[-2] or not input.is_floating_point():
            raise ValueError("expected real square covariance matrices")
        if (
            input.dtype != self.propagation.dtype
            or input.device != self.propagation.device
        ):
            raise ValueError("covariances and graph must have the same dtype/device")
        if not torch.isfinite(input).all() or not torch.allclose(input, input.mT):
            raise ValueError("covariances must be finite and symmetric")
        if (torch.linalg.eigvalsh(input) <= 0).any():
            raise ValueError("LGT fitting requires SPD, not PSD, covariances")
        lattice = input.mean(0)
        if not torch.isfinite(lattice).all():
            raise ValueError("training lattice overflow; rescale or use float64")
        inverse = matrix_inv_sqrt.apply(lattice)
        whitened = inverse[:, None] @ lattice[None, :] @ inverse[:, None]
        distance_squared = matrix_log.apply(whitened).square().sum((-2, -1))
        # Roundoff can make the two independently computed AIRM distances differ.
        distance_squared = (distance_squared + distance_squared.mT) / 2
        adjacency = (-distance_squared / self.bandwidth).exp() * self.topology.to(input)
        propagation = normalize_spd_adjacency(adjacency)
        # No persistent mutation until every numerical operation has succeeded.
        self.lattice = lattice.detach().clone()
        self.propagation = propagation
        self.fitted.fill_(True)
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        r"""Load a checkpoint while preserving state after a failed load.

        Parameters
        ----------
        state_dict : dict
            Parameters and buffers returned by state_dict.
        strict : bool, default=True
            Require the checkpoint keys to match the module state.
        assign : bool, default=False
            Use checkpoint tensor properties when assigning state.

        Returns
        -------
        NamedTuple
            Missing and unexpected checkpoint keys.

        Notes
        -----
        Reconstruct the same graph and segmentation configuration before
        loading. Failed loads restore tensor values and ownership, including
        assign=True.
        """
        return _atomic_load_state_dict(self, state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        keys = ("coordinates", "topology", "propagation", "lattice", "fitted")
        present = [prefix + key in state_dict for key in keys]
        if any(present):
            try:
                if not all(present):
                    raise ValueError("graph checkpoint requires all graph buffers")
                coordinates, topology, propagation, lattice, fitted = (
                    state_dict[prefix + key] for key in keys
                )
                n = len(self.coordinates)
                if (
                    coordinates.shape != (n, 2)
                    or topology.shape != (n, n)
                    or propagation.shape != (n, n)
                    or lattice.ndim != 3
                    or lattice.shape[0] != n
                    or lattice.shape[-1] != lattice.shape[-2]
                    or fitted.shape != ()
                    or fitted.dtype != torch.bool
                ):
                    raise ValueError(
                        "graph checkpoint has incompatible node/buffer shapes"
                    )
                if bool(fitted) != (lattice.shape[-1] > 0):
                    raise ValueError(
                        "graph checkpoint fitted flag and lattice disagree"
                    )
                if (
                    not torch.isfinite(coordinates).all()
                    or not torch.isfinite(lattice).all()
                ):
                    raise ValueError("graph checkpoint contains nonfinite values")
                # Validate propagation before resizing/copying any graph buffer.
                if (
                    not propagation.is_floating_point()
                    or not torch.isfinite(propagation).all()
                    or (propagation < 0).any()
                    or not torch.allclose(
                        propagation.sum(-1), torch.ones_like(propagation.sum(-1))
                    )
                ):
                    raise ValueError(
                        "graph checkpoint propagation must be row stochastic"
                    )
                if bool(fitted) and (
                    not torch.allclose(lattice, lattice.mT)
                    or (torch.linalg.eigvalsh(lattice) <= 0).any()
                ):
                    raise ValueError("graph checkpoint lattice must be SPD")
            except (ValueError, RuntimeError, AttributeError, TypeError) as error:
                error_msgs.append(prefix + str(error))
                return
            if self.lattice.shape != lattice.shape:
                self.lattice = self.lattice.new_empty(lattice.shape)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self) -> Tensor:
        r"""Return the stored graph propagation.

        Returns
        -------
        torch.Tensor
            Row-stochastic propagation of shape (nodes, nodes).
        """
        return self.propagation


class GraphBiMap(nn.Module):
    r"""Apply graph aggregation followed by node-specific bilinear maps.

    See :cite:p:`Ju_2024`.

    Parameters
    ----------
    n_nodes : int
        Number of ordered graph nodes.
    in_features : int
        Input matrix dimension.
    out_features : int
        Output matrix dimension.
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.

    Notes
    -----
    Nodes are never mixed across trials. Projections use the host Stiefel
    parametrization; dimension increases use identity padding.
    """

    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        out_features: int,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if min(n_nodes, in_features, out_features) <= 0:
            raise ValueError("node and matrix dimensions must be positive")
        self.n_nodes = n_nodes
        self.in_features = in_features
        self.projection = BiMap(
            in_features,
            out_features,
            depthwise=n_nodes,
            parametrized=True,
            init_method="stiefel",
            device=device,
            dtype=dtype,
        )

    def forward(self, input: Tensor, propagation: Tensor) -> Tensor:
        r"""Aggregate and project ordered SPD nodes.

        Parameters
        ----------
        input : torch.Tensor
            SPD nodes of shape (..., n_nodes, in_features, in_features).
        propagation : torch.Tensor
            Row-stochastic weights of shape (..., n_nodes, n_nodes).

        Returns
        -------
        torch.Tensor
            Projected nodes of shape (..., n_nodes, out_features,
            out_features).
        """
        if input.shape[-3:] != (self.n_nodes, self.in_features, self.in_features):
            raise ValueError("input node/channel shape does not match GraphBiMap")
        output = ensure_sym(self.projection(spd_graph_aggregate(input, propagation)))
        with torch.no_grad():
            valid = bool(torch.isfinite(output).all())
            if valid:
                try:
                    valid = bool((torch.linalg.eigvalsh(output) > 0).all()) and bool(
                        (torch.linalg.cholesky_ex(output).info == 0).all()
                    )
                except RuntimeError:
                    valid = False
            if not valid:
                raise ValueError(
                    "GraphBiMap lost SPD in finite precision; use float64 or "
                    "explicit regularization/rescaling of covariance units"
                )
        return output
