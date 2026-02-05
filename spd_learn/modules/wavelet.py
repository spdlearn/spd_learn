# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from the GREEN project by Raphaël Music and Louis Music
# Original source: https://github.com/Music-Prediction/GREEN
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from ..functional.wavelet import compute_gabor_wavelet


class WaveletConv(nn.Module):
    """Parametrized Complex Gabor Wavelet Convolution Layer.

    This layer performs a 1D convolution of input signals with a bank of
    complex-valued Gabor wavelets (Morlet wavelets). The center frequency
    and temporal resolution of each wavelet are learnable parameters.

    .. figure:: https://roche.github.io/neuro-meeglet/background/0_tour_files/figure-html/cell-3-output-2.png
       :align: center
       :alt: Morlet wavelets

    Parameters
    ----------
    kernel_width_s : float
        The temporal width of the wavelet kernel in seconds.
    foi_init : Sequence[float]
        A sequence of initial center frequencies for the wavelets, in
        octaves.
    sfreq : int, default=100
        The sampling frequency of the input data in Hz.
    fwhm_init : Sequence[float], optional
        A sequence of initial Full Width at Half Maximums (FWHM) for the
        wavelets, in octaves.
    padding : int or str, default=0
        Padding mode for the convolution.
    stride : int, default=1
        The stride of the convolution.
    scaling : str, default="oct"
        The scaling method applied to the wavelets after L2 normalization.
    dtype : torch.dtype, default=torch.complex64
        The data type for the complex wavelet kernels.

    Notes
    -----
    See :cite:p:`paillard2024green` for more details.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import WaveletConv
    >>> # Create wavelet layer with 5 wavelets centered at 4, 8, 16, 32, 64 Hz
    >>> foi_init = [2.0, 3.0, 4.0, 5.0, 6.0]  # In octaves (2^n Hz)
    >>> wavelet = WaveletConv(kernel_width_s=0.5, foi_init=foi_init, sfreq=250)
    >>> X = torch.randn(2, 22, 500)  # (batch, channels, time)
    >>> Y = wavelet(X)
    >>> Y.shape
    torch.Size([2, 5, 22, 376])

    .. plot::
        :include-source:

        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from spd_learn.modules import WaveletConv
        from spd_learn.functional import compute_gabor_wavelet

        # Create wavelet filterbank
        foi_init = [2.0, 3.0, 4.0, 5.0]  # 4, 8, 16, 32 Hz
        sfreq = 250
        wavelet = WaveletConv(kernel_width_s=0.5, foi_init=foi_init, sfreq=sfreq)

        # Get wavelet kernels
        tt = wavelet.tt.numpy()
        wavelets = compute_gabor_wavelet(
            wavelet.tt, wavelet.foi, wavelet.fwhm, sfreq=sfreq
        ).detach().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        freq_labels = [f'{2**f:.0f} Hz' for f in foi_init]

        for i, (ax, wav, label) in enumerate(zip(axes.flat, wavelets, freq_labels)):
            ax.plot(tt * 1000, wav.real, 'b-', label='Real', alpha=0.8)
            ax.plot(tt * 1000, wav.imag, 'r-', label='Imag', alpha=0.8)
            ax.fill_between(tt * 1000, -np.abs(wav), np.abs(wav),
                            color='gray', alpha=0.2, label='Envelope')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Wavelet at {label}', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(tt[0]*1000, tt[-1]*1000)

        plt.suptitle('Gabor Wavelet Filterbank', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()
    """

    tt: Tensor  # Type annotation for registered buffer

    def __init__(
        self,
        kernel_width_s: float,
        foi_init: Union[Sequence[float], Tensor],
        sfreq: int = 100,
        fwhm_init: Optional[Union[Sequence[float], Tensor]] = None,
        padding: Union[int, str] = 0,
        stride: int = 1,
        scaling: str = "oct",
        dtype: torch.dtype = torch.complex64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.n_wavelets = len(foi_init)
        self.sfreq = sfreq
        self.kernel_width_s = kernel_width_s
        self.padding = padding
        self.stride = stride
        self.scaling = scaling
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        tmax = kernel_width_s / 2.0
        tmin = -tmax
        kernel_length = int(kernel_width_s * sfreq)
        self.register_buffer("tt", torch.linspace(tmin, tmax, kernel_length))

        # Convert foi_init to tensor if needed
        if isinstance(foi_init, Tensor):
            foi_tensor = foi_init.detach().clone()
        else:
            foi_tensor = torch.tensor(foi_init)

        # Generate default fwhm_init if not provided, then convert to tensor
        if fwhm_init is None:
            # Default: FWHM decreases with frequency (negative values in log scale)
            fwhm_tensor = -foi_tensor
        elif isinstance(fwhm_init, Tensor):
            fwhm_tensor = fwhm_init.detach().clone()
        else:
            fwhm_tensor = torch.tensor(fwhm_init)

        self.foi = nn.Parameter(foi_tensor, requires_grad=True)
        self.fwhm = nn.Parameter(fwhm_tensor, requires_grad=True)

    def forward(self, X: Tensor) -> Tensor:
        """Applies the wavelet convolution to the input signal.

        Parameters
        ----------
        X : torch.Tensor
            Input data tensor with shape `(N, C_in, T_in)` or
            `(N, E, C_in, T_in)`.

        Returns
        -------
        torch.Tensor
            Complex-valued output tensor after convolution.
        """
        wavelets = compute_gabor_wavelet(
            tt=self.tt,
            foi=self.foi,
            fwhm=self.fwhm,
            sfreq=self.sfreq,
            scaling=self.scaling,
            dtype=self.dtype,
        )
        n_freqs = wavelets.shape[0]

        X = X.to(dtype=self.dtype)

        if X.dim() == 3:
            batch_size, in_channels, times = X.shape
            X_conv = F.conv1d(
                X.to(self.dtype).view(-1, 1, times),
                wavelets.unsqueeze(1),
                padding=self.padding,
                stride=self.stride,
            )
            X_conv = X_conv.view(batch_size, in_channels, n_freqs, -1)
            X_conv = X_conv.swapaxes(1, 2)
        elif X.dim() == 4:
            batch_size, n_epochs, in_channels, times = X.shape
            X_conv = F.conv1d(
                X.to(self.dtype).view(batch_size * n_epochs * in_channels, 1, times),
                wavelets.unsqueeze(1),
                padding=self.padding,
                stride=self.stride,
            )
            X_conv = X_conv.view(batch_size, n_epochs, in_channels, n_freqs, -1)
            X_conv = X_conv.permute(0, 3, 2, 1, 4).contiguous()
            n_batch, n_freqs, n_sensors, n_epochs, n_times = X_conv.shape
            X_conv = X_conv.view(n_batch, n_freqs, n_sensors, n_epochs * n_times)

        return X_conv.to(device=self.device, dtype=self.dtype)
