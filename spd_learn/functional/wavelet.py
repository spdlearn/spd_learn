# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
# This file contains code derived from the GREEN project by Raphaël Music and Louis Music
# Original source: https://github.com/Music-Prediction/GREEN
# Licensed under the Apache License, Version 2.0
"""Functional operations for wavelet transforms.

This module provides stateless mathematical operations for computing
Gabor (Morlet) wavelet filterbanks used in time-frequency analysis.

Functions
---------
compute_gabor_wavelet
    Compute a complex Gabor wavelet filterbank.

See Also
--------
:class:`~spd_learn.modules.WaveletConv` : Learnable wavelet convolution layer.
"""

import math

import torch

from torch import Tensor


def compute_gabor_wavelet(
    tt: Tensor,
    foi: Tensor,
    fwhm: Tensor,
    sfreq: float = 250.0,
    scaling: str = "oct",
    dtype: torch.dtype = torch.complex64,
    min_foi_oct: float = -2.0,
    max_foi_oct: float = 6.0,
    min_fwhm_oct: float = -6.0,
    max_fwhm_oct: float = 1.0,
) -> Tensor:
    """Compute a complex Gabor (Morlet) wavelet filterbank.

    Creates a bank of complex-valued Gabor wavelets with learnable center
    frequencies and temporal resolutions. The wavelets are L2-normalized
    and optionally scaled for octave-based frequency analysis.

    Parameters
    ----------
    tt : Tensor
        Time vector with shape `(kernel_length,)`, typically centered at 0.
    foi : Tensor
        Center frequencies in octaves (log2 Hz) with shape `(n_wavelets,)`.
        For example, foi=3.0 corresponds to 2^3 = 8 Hz.
    fwhm : Tensor
        Full Width at Half Maximum in octaves with shape `(n_wavelets,)`.
        Controls the temporal resolution of each wavelet.
    sfreq : float, default=250.0
        Sampling frequency in Hz.
    scaling : str, default="oct"
        Scaling method. If "oct", applies octave-based amplitude scaling
        to achieve constant energy per octave.
    dtype : torch.dtype, default=torch.complex64
        Output data type for the complex wavelets.
    min_foi_oct : float, default=-2.0
        Minimum clamp value for center frequencies (in octaves).
    max_foi_oct : float, default=6.0
        Maximum clamp value for center frequencies (in octaves).
    min_fwhm_oct : float, default=-6.0
        Minimum clamp value for FWHM (in octaves).
    max_fwhm_oct : float, default=1.0
        Maximum clamp value for FWHM (in octaves).

    Returns
    -------
    Tensor
        Complex wavelet filterbank with shape `(n_wavelets, kernel_length)`.

    Notes
    -----
    The Gabor wavelet is defined as:

    .. math::

        \\psi(t) = \\exp(2\\pi i f t) \\cdot \\exp\\left(-\\frac{4 \\ln 2 \\cdot t^2}{h^2}\\right)

    where :math:`f` is the center frequency and :math:`h` is the FWHM of the
    Gaussian envelope.

    See Also
    --------
    :class:`~spd_learn.modules.WaveletConv` : Module wrapper using this function.

    References
    ----------
    See :cite:p:`paillard2024green` for details on learnable wavelet filterbanks.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional import compute_gabor_wavelet
    >>> # Create time vector for 0.5s kernel at 250 Hz
    >>> tt = torch.linspace(-0.25, 0.25, 125)
    >>> # Frequencies: 4, 8, 16 Hz in octave notation
    >>> foi = torch.tensor([2.0, 3.0, 4.0])
    >>> fwhm = torch.tensor([-2.0, -3.0, -4.0])
    >>> wavelets = compute_gabor_wavelet(tt, foi, fwhm, sfreq=250)
    >>> wavelets.shape
    torch.Size([3, 125])
    """
    # Convert from octave notation to Hz
    foi_oct = 2 ** torch.clamp(foi, min_foi_oct, max_foi_oct)
    fwhm_oct = 2 ** torch.clamp(fwhm, min_fwhm_oct, max_fwhm_oct)

    # Compute wavelets
    wavelets = torch.stack(
        [
            torch.exp(2j * math.pi * f * tt)
            * torch.exp(-4 * math.log(2) * tt**2 / h**2)
            for f, h in zip(foi_oct, fwhm_oct)
        ],
        dim=0,
    ).to(dtype)

    # L2 normalize
    wav_norm = wavelets / torch.linalg.norm(wavelets, dim=-1, keepdim=True)

    # Apply octave-based scaling if requested
    if scaling == "oct":
        wav_norm *= math.sqrt(2.0 / sfreq) * torch.sqrt(
            math.log(2) * foi_oct
        ).unsqueeze(1)

    return wav_norm


__all__ = [
    "compute_gabor_wavelet",
]
