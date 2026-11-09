# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of GBWAtt (Jin et al., KDD, 2026).
# Reference implementation: https://github.com/jjscc/GBWAtt

import torch

from torch import nn

from spd_learn.modules import LogEig, ReEig, Vech
from spd_learn.modules.gbw import GBWAttention, GBWPatchCovariance

from .matt import MAtt


class GBWAtt(MAtt):
    r"""Generalized Bures-Wasserstein attention network for EEG.

    See :cite:p:`jin2026robust`.

    Parameters
    ----------
    n_chans : int, default=22
        Ordered input electrode channels. Input is
        ``(batch,n_chans,n_times)``.
    n_outputs : int, default=4
        Output classes; returns unnormalized ``(batch,n_outputs)`` logits
        for cross_entropy, with no softmax inside the model.
    n_patches : int, default=3
        Temporal subparts in acquisition order; each needs >=2 feature
        samples.
    temporal_out_channels : int, default=20
        Covariance dimension (learned convolution features, not electrode
        axes).
    attention_out_features : int, default=18
        SPD dimension after Stiefel Q/K/V projections.
    temporal_kernel_size : int, default=12
        Temporal kernel in samples; symmetric padding of kernel_size//2.
        Even kernels produce n_times+1 outputs, as in the paper MI
        architecture.
    power : float, default=1.5
        Fixed positive deformation exponent.
    learn_power : bool, default=False
        Optional constrained learning of the scalar power (not paper
        default).
    n_iter : int, default=10
        Fixed count of differentiable BW barycenter iterations.
    aggregation : {'fixed_point', 'root'}, default='fixed_point'
        Paper iteration or author weighted-root approximation. The
        fixed-point mean uses float64 internally for float32 inputs to
        protect cross-product spectra, returning checked float32 output.
    trace_normalize : bool, default=True
        Normalize each covariance before diagonal loading, as stated in
        paper. Author Signal2Spd default disables this.
    covariance_epsilon : float, default=1e-05
        Absolute positive diagonal loading after optional trace
        normalization.
    reeig_threshold : float, default=1e-05
        Paper rectification threshold (author code instead uses 1e-4).

    Notes
    -----
    Accepts (trial, channel, sample) EEG and returns class logits. The
    default frontend is the MI architecture. Trace normalization and
    registered metric parameters follow the stated geometry; the released
    code differs. Fixed-point aggregation is approximate. Train BatchNorm
    within the training partition and keep input units, sampling and channel
    order consistent.

    See Also
    --------
    spd_learn.modules.GBWPatchCovariance
    spd_learn.modules.GBWGeometry
    spd_learn.modules.GBWAttention

    Examples
    --------
    See :ref:`sphx_glr_generated_auto_examples_howto_plot_gbwatt.py`
    for a small synthetic update. It does not reproduce dataset accuracy.
    """

    def __init__(
        self,
        n_chans=22,
        n_outputs=4,
        n_patches=3,
        temporal_out_channels=20,
        attention_out_features=18,
        temporal_kernel_size=12,
        power=1.5,
        learn_power=False,
        n_iter=10,
        aggregation="fixed_point",
        trace_normalize=True,
        covariance_epsilon=1e-5,
        reeig_threshold=1e-5,
    ):
        if min(n_chans, n_outputs, temporal_out_channels, temporal_kernel_size) < 1:
            raise ValueError("channel, class and kernel dimensions must be positive")
        super().__init__(
            n_patches=n_patches,
            n_chans=n_chans,
            n_outputs=n_outputs,
            temporal_out_channels=temporal_out_channels,
            temporal_kernel_size=temporal_kernel_size,
            temporal_padding=temporal_kernel_size // 2,
            attention_in_features=temporal_out_channels,
            attention_out_features=attention_out_features,
        )
        self.patch_cov_layer = GBWPatchCovariance(
            n_patches, covariance_epsilon, trace_normalize
        )
        self.manifold_attention = GBWAttention(
            temporal_out_channels,
            attention_out_features,
            power,
            learn_power,
            n_iter,
            aggregation,
        )
        self.re_eig = ReEig(threshold=reeig_threshold)
        self.tangent = nn.Sequential(
            LogEig(upper=False, flatten=False), Vech(preserve_norm=False)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Classify EEG trials using generalized BW attention.

        Parameters
        ----------
        input : torch.Tensor
            EEG trials of shape (batch, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Unnormalized class logits of shape (batch, n_outputs).
        """
        if input.ndim != 3 or input.shape[1] != self.n_chans:
            raise ValueError("expected EEG (batch,n_chans,n_times)")
        return super().forward(input)
