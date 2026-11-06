# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of Graph-CSPNet (Ju and Guan, IEEE TNNLS, 2023).
# Reference: https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet

from collections.abc import Sequence
from math import isfinite
from warnings import warn

import torch

from torch import Tensor, nn

from spd_learn.modules import LogEig, ReEig, SPDBatchNormMean
from spd_learn.modules.graph import (
    GraphBiMap,
    TimeFrequencyCovariances,
    TimeFrequencyGraph,
    _atomic_load_state_dict,
)


class GraphCSPNet(nn.Module):
    r"""Graph-CSPNet for externally filtered multiband EEG.

    See :cite:p:`Ju_2024`.

    Parameters
    ----------
    n_chans : int, default=22
        Number of ordered input electrodes.
    n_outputs : int, default=4
        Number of output classes.
    n_freqs : int, default=9
        Number of ordered filter-bank bands.
    n_patches : int, default=4
        Number of windows per low-frequency band.
    dims : sequence of int or None
        Consecutive matrix dimensions; None scales the default (22, 36,
        22).
    epoch_duration : float, default=1.0
        Duration of the complete input epoch in seconds.
    graph : TimeFrequencyGraph or None
        Optional graph matching the ordered segmentation and construction
        dtype/device.
    segments : sequence of tuple or None
        Explicit (band, start, stop) sample windows; requires a matching
        graph.
    bandwidth : float, default=50.0
        Positive divisor of squared AIRM distances in Gaussian edge
        weights.
    time_radius : float or None
        Temporal radius in seconds; None uses half the epoch duration.
    frequency_radius : float, optional, default=16.0
        Frequency neighborhood radius in Hz for the default graph.
    ridge : float, default=1e-05
        Absolute covariance loading in squared signal units.
    shrinkage : float, default=0.01
        Mixture weight of the trace-scaled identity, in [0, 1].
    center : bool, default=True
        Subtract each channel mean within its window.
    use_mlp : bool, default=False
        Use the optional two-hidden-layer classifier.
    require_fitted_graph : bool, default=True
        Require training-only graph fitting before forward.
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.

    Notes
    -----
    Input is (trial, band, channel, sample); output is class logits. The
    default nine-band segmentation uses four windows in the first six bands
    and eight in the last three when n_patches=4. Fit the graph on training
    trials only. Covariance ridge and the ReEig floor are absolute: use
    consistent signal units above these floors. The classifier uses all
    log-matrix entries. Adam updates the host constrained coordinates,
    rather than the released optimizer trajectory.

    See Also
    --------
    spd_learn.modules.TimeFrequencyCovariances
    spd_learn.modules.TimeFrequencyGraph
    spd_learn.modules.GraphBiMap

    Examples
    --------
    See :ref:`sphx_glr_generated_auto_examples_howto_plot_graphcspnet.py`
    for a small synthetic update. It does not reproduce dataset accuracy.
    """

    def __init__(
        self,
        n_chans: int = 22,
        n_outputs: int = 4,
        n_freqs: int = 9,
        n_patches: int = 4,
        *,
        dims: Sequence[int] | None = None,
        epoch_duration: float = 1.0,
        graph: TimeFrequencyGraph | None = None,
        segments: Sequence[tuple[int, int, int]] | None = None,
        bandwidth: float = 50.0,
        time_radius: float | None = None,
        frequency_radius: float = 16.0,
        ridge: float = 1e-5,
        shrinkage: float = 0.01,
        center: bool = True,
        use_mlp: bool = False,
        require_fitted_graph: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if (
            min(n_chans, n_outputs, n_freqs, n_patches) <= 0
            or not isfinite(epoch_duration)
            or epoch_duration <= 0
        ):
            raise ValueError("dimensions and epoch_duration must be positive")
        if segments is not None and graph is None:
            raise ValueError(
                "explicit segments require a graph with matching node order"
            )
        if not 0 <= shrinkage <= 1 or not isfinite(ridge) or ridge <= 0:
            raise ValueError("require 0 <= shrinkage <= 1 and finite ridge > 0")
        self.n_chans, self.n_freqs, self.n_patches = n_chans, n_freqs, n_patches
        self.segments = tuple(segments) if segments is not None else None
        self.ridge, self.shrinkage = ridge, shrinkage
        self.center = center
        self.require_fitted_graph = require_fitted_graph
        coordinates, components = [], []
        self.window_counts = tuple(
            n_patches * (2 if b >= 6 else 1) for b in range(n_freqs)
        )
        for band, count in enumerate(self.window_counts):
            for window in range(count):
                coordinates.append(
                    ((window + 0.5) * epoch_duration / count, 6 + 4 * band)
                )
                components.append(
                    0 if band == 0 else 1 if band == 1 else 2 if band < 6 else 3
                )
        self.graph = (
            graph
            if graph is not None
            else TimeFrequencyGraph(
                torch.tensor(
                    coordinates, device=device, dtype=dtype or torch.get_default_dtype()
                ),
                time_radius=epoch_duration / 2 if time_radius is None else time_radius,
                frequency_radius=frequency_radius,
                components=torch.tensor(components, device=device),
                bandwidth=bandwidth,
                neighborhood="forward",
            )
        )
        expected = torch.empty(0, device=device, dtype=dtype)
        if (
            self.graph.propagation.dtype != expected.dtype
            or self.graph.propagation.device != expected.device
        ):
            raise ValueError(
                "custom graph must match the model construction dtype/device; "
                "move the graph explicitly or supply matching device/dtype"
            )
        n_nodes = len(self.graph.coordinates)
        if self.segments is not None and len(self.segments) != n_nodes:
            raise ValueError("segments and graph must have the same node count")
        if self.segments is None and n_nodes != sum(self.window_counts):
            raise ValueError("graph node count differs from default segmentation")
        self.dims = (
            tuple(dims)
            if dims is not None
            else (n_chans, max(n_chans, round(36 * n_chans / 22)), n_chans)
        )
        if len(self.dims) < 2 or self.dims[0] != n_chans or min(self.dims) <= 0:
            raise ValueError(
                "dims must begin with n_chans and contain positive layer sizes"
            )
        self.layers = nn.ModuleList(
            [
                GraphBiMap(n_nodes, a, b, device=device, dtype=dtype)
                for a, b in zip(self.dims[:-1], self.dims[1:])
            ]
        )
        self.rectify = ReEig(threshold=1e-4, device=device, dtype=dtype)
        self.norm = SPDBatchNormMean(
            self.dims[-1], momentum=0.1, device=device, dtype=dtype
        )
        self.log = LogEig(upper=False, flatten=False, device=device, dtype=dtype)
        self.register_buffer(
            "identity_graph", torch.eye(n_nodes, device=device, dtype=dtype)
        )
        features = n_nodes * self.dims[-1] ** 2
        self.classifier: nn.Module = (
            nn.Sequential(
                nn.Linear(features, n_nodes, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(n_nodes, n_nodes, device=device, dtype=dtype),
                nn.ReLU(),
                nn.Linear(n_nodes, n_outputs, device=device, dtype=dtype),
            )
            if use_mlp
            else nn.Linear(features, n_outputs, device=device, dtype=dtype)
        )

    def covariances(self, input: Tensor) -> Tensor:
        r"""Estimate the ordered time-frequency covariance nodes.

        Parameters
        ----------
        input : torch.Tensor
            Filtered trials of shape (batch, bands, channels, samples).

        Returns
        -------
        torch.Tensor
            Covariances of shape (batch, nodes, channels, channels), in
            squared signal units.
        """
        if input.ndim != 4 or input.shape[1:3] != (self.n_freqs, self.n_chans):
            raise ValueError("expected (trials, n_freqs, n_chans, samples)")
        segments = self.segments
        if segments is None:
            if input.shape[-1] < max(self.window_counts):
                raise ValueError("not enough time samples for nonempty windows")
            segments = tuple(
                (band, w * input.shape[-1] // count, (w + 1) * input.shape[-1] // count)
                for band, count in enumerate(self.window_counts)
                for w in range(count)
            )
        output = TimeFrequencyCovariances(
            segments, ridge=self.ridge, shrinkage=self.shrinkage, center=self.center
        )(input)
        if torch.linalg.eigvalsh(output.detach()).median() < self.rectify.threshold_:
            warn(
                "Covariance scale is below the absolute ReEig floor; ridge and "
                "rectification can erase EEG variation. Pre-scale signal units "
                "consistently (e.g. volts to microvolts).",
                UserWarning,
                stacklevel=2,
            )
        return output

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
        lattice = state_dict.get("graph.lattice")
        if lattice is not None and (
            lattice.ndim != 3
            or lattice.shape[1:] not in ((0, 0), (self.n_chans, self.n_chans))
        ):
            raise RuntimeError("graph checkpoint lattice channels must match n_chans")
        return _atomic_load_state_dict(self, state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def fit_graph(self, training_signals: Tensor) -> "GraphCSPNet":
        r"""Fit the local graph topology weights on training trials.

        Parameters
        ----------
        training_signals : torch.Tensor
            Training-only filtered trials of shape (batch, bands, channels,
            samples).

        Returns
        -------
        self : GraphCSPNet
            The model with fitted graph weights.

        Notes
        -----
        This fits the graph lattice without updating batch-normalization
        statistics.
        """
        self.graph.fit(self.covariances(training_signals))
        return self

    def forward_covariances(self, input: Tensor) -> Tensor:
        r"""Classify precomputed covariance nodes.

        Parameters
        ----------
        input : torch.Tensor
            SPD nodes of shape (batch, nodes, channels, channels), in fitted
            graph order.

        Returns
        -------
        torch.Tensor
            Unnormalized class logits of shape (batch, n_outputs).
        """
        if input.ndim != 4:
            raise ValueError("expected (trials, nodes, channels, channels)")
        if self.require_fitted_graph and (
            not bool(self.graph.fitted)
            or self.graph.lattice.shape
            != (len(self.graph.coordinates), self.n_chans, self.n_chans)
        ):
            raise RuntimeError("Fit the LGT graph on training trials before forward")
        x = input
        for index, layer in enumerate(self.layers):
            x = layer(x, self.graph() if index == 0 else self.identity_graph)
            if index < len(self.layers) - 1:
                x = self.rectify(x)
        shape = x.shape
        x = self.norm(x.flatten(0, 1)).reshape(shape)
        x = self.rectify(x)
        return self.classifier(self.log(x).flatten(1))

    def forward(self, input: Tensor) -> Tensor:
        r"""Classify externally filtered EEG trials.

        Parameters
        ----------
        input : torch.Tensor
            Filtered trials of shape (batch, bands, channels, samples), in
            consistent units.

        Returns
        -------
        torch.Tensor
            Unnormalized class logits of shape (batch, n_outputs).
        """
        return self.forward_covariances(self.covariances(input))
