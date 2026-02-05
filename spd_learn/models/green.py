# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from the GREEN project by Raphaël Music and Louis Music
# Original source: https://github.com/Music-Prediction/GREEN
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import torch

from torch import nn

from spd_learn.functional.covariance import cross_covariance, real_covariance
from spd_learn.modules import BatchReNorm, BiMap, CovLayer, LogEig, Shrinkage
from spd_learn.modules.wavelet import WaveletConv


class Green(nn.Module):
    r"""Gabor Riemann EEGNet.

    This class implements the Gabor Riemann EEGNet (GREEN) model
    :cite:p:`paillard2024green`. GREEN is a neural network model that
    processes EEG epochs using convolutional layers, followed by the
    computation of SPD features.

    .. figure:: /_static/models/green.png
       :align: center
       :alt: Gabor Riemann EEGNet Architecture

    Parameters
    ----------
    n_outputs : int
        Number of output classes for classification.
    n_chans : int
        Number of input EEG channels.
    sfreq : int, default=100
        Sampling frequency of the EEG data.
    n_freqs_init : int, default=10
        Number of main frequencies in the wavelet family.
    kernel_width_s : float, default=0.5
        Width of the wavelet kernel in seconds.
    conv_stride : int, default=1
        Stride of the wavelet convolution.
    oct_min : float, default=0
        Minimum frequency of interest in octaves.
    oct_max : float, default=3
        Maximum frequency of interest in octaves.
    random_f_init : bool, default=False
        Whether to randomly initialize the frequencies of interest.
    shrinkage_init : float, optional, default=-3.0
        Initial shrinkage value before applying the sigmoid function.
    logref : str, default="logeuclid"
        Reference matrix used for the LogEig layer.
    momentum : float, optional, default=0.9
        Momentum for the BatchReNorm layer.
    dropout : float, optional, default=0.5
        Dropout rate for the fully connected layers.
    hidden_dim : tuple[int] or list[int], optional, default=(8,)
        Dimensions of the hidden layers in the classification head.
    pool_method : Callable, default=real_covariance
        Method for pooling the covariance matrices.
    bi_out : tuple[int] or list[int], optional
        Output dimensions for the BiMap layers.
    dtype : torch.dtype, default=torch.float32
        Data type of the tensors.
    """

    foi_init: torch.Tensor  # Type annotation for registered buffer
    fwhm_init: torch.Tensor  # Type annotation for registered buffer

    def __init__(
        self,
        n_outputs: int,
        n_chans: int,
        sfreq: int = 100,
        n_freqs_init: int = 10,
        kernel_width_s: float = 0.5,
        conv_stride: int = 1,
        oct_min: float = 0,
        oct_max: float = 3,
        random_f_init: bool = False,
        shrinkage_init: Optional[float] = -3.0,
        logref: str = "logeuclid",
        momentum: Optional[float] = 0.9,
        dropout: Optional[float] = 0.5,
        hidden_dim: Optional[Union[List[int], Tuple[int]]] = (8,),
        pool_method: Callable = real_covariance,
        bi_out: Optional[Union[List[int], Tuple[int]]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        cplx_dtype = torch.complex128 if (dtype == torch.float64) else torch.complex64

        if random_f_init:
            foi_init = torch.rand(n_freqs_init) * (oct_max - oct_min) + oct_min
            fwhm_init = -(
                torch.rand(n_freqs_init) * (oct_max - oct_min) + (oct_min - 1)
            )
        else:
            foi_init = torch.linspace(oct_min, oct_max, steps=n_freqs_init)
            fwhm_init = -torch.linspace(oct_min - 1, oct_max - 1, steps=n_freqs_init)

        self.register_buffer("foi_init", foi_init.to(dtype=dtype))
        self.register_buffer("fwhm_init", fwhm_init.to(dtype=dtype))

        self.conv_layers = nn.Sequential(
            WaveletConv(
                kernel_width_s=kernel_width_s,
                sfreq=sfreq,
                foi_init=self.foi_init,
                fwhm_init=self.fwhm_init,
                stride=conv_stride,
                dtype=cplx_dtype,
                scaling="oct",
            )
        )

        n_freqs_effective = n_freqs_init

        if pool_method == real_covariance:
            n_compo = n_chans
        elif pool_method == cross_covariance:
            n_compo = n_chans * n_freqs_init
            n_freqs_effective = 1

        self.cov_layer = CovLayer(method=pool_method)

        spd_layers_list: List[nn.Module] = []
        if shrinkage_init is not None:
            spd_layers_list.append(
                Shrinkage(
                    n_chans=n_compo,
                    init_shrinkage=shrinkage_init,
                    learnable=True,
                    dtype=dtype,
                )
            )
        else:
            spd_layers_list.append(nn.Identity())

        current_spd_dim = n_compo
        if bi_out is not None:
            for bo in bi_out:
                bimap = BiMap(in_features=current_spd_dim, out_features=bo, dtype=dtype)
                spd_layers_list.append(bimap)
                current_spd_dim = bo

        self.spd_layers = nn.Sequential(*spd_layers_list)

        vectorized_dim = int(current_spd_dim * (current_spd_dim + 1) / 2)
        proj = nn.Sequential(LogEig(upper=True, autograd=True))

        if logref == "logeuclid":
            proj.append(
                BatchReNorm(
                    num_features=vectorized_dim,
                    momentum=momentum,
                    rebias=False,
                    renorm=True,
                )
            )

        self.proj = proj

        if n_freqs_effective > 1 and pool_method == real_covariance:
            feat_dim = n_freqs_effective * vectorized_dim
        else:
            feat_dim = vectorized_dim

        layers: List[nn.Module] = []
        all_dims = [feat_dim] + list(hidden_dim or []) + [n_outputs]

        for i in range(len(all_dims) - 1):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]

            layers.append(nn.BatchNorm1d(in_dim, dtype=dtype))
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(in_dim, out_dim, dtype=dtype))

            is_last_layer = i == len(all_dims) - 2
            if not is_last_layer:
                layers.append(nn.GELU())

        self.head = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GREEN model.

        Parameters
        ----------
        X : torch.Tensor
            Input EEG data tensor with shape `(batch_size, n_chans, n_times)`.

        Returns
        -------
        torch.Tensor
            Output tensor with shape `(batch_size, n_outputs)`.
        """
        X_conv = self.conv_layers(X)
        X_cov = self.cov_layer(X_conv)
        X_spd = self.spd_layers(X_cov)
        X_proj = self.proj(X_spd)
        X_flat = torch.flatten(X_proj, start_dim=1)
        output = self.head(X_flat)
        return output
