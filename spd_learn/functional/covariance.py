# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import torch


def covariance(input: torch.Tensor) -> torch.Tensor:
    """Computes the covariance matrix of multivariate data.

    The input tensor is assumed to have shape `(..., n_channels, n_times)`,
    where `...` represents any number of leading dimensions.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor with shape `(..., n_channels, n_times)`.

    Returns
    -------
    torch.Tensor
        Covariance matrices with shape `(..., n_channels, n_channels)`.
    """
    mean = input.mean(dim=-1, keepdim=True)
    input_centered = input - mean
    covariances = torch.einsum(
        "...ik,...jk->...ij", input_centered, input_centered
    ) / input.size(-1)
    return covariances


def sample_covariance(input: torch.Tensor) -> torch.Tensor:
    """Computes the sample covariance matrix of multivariate data.

    The input tensor is assumed to have shape `(..., n_channels, n_times)`,
    where `...` represents any number of leading dimensions. The sample
    covariance is scaled by `(n_times - 1)`.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor with shape `(..., n_channels, n_times)`.

    Returns
    -------
    torch.Tensor
        Sample covariance matrices with shape `(..., n_channels, n_channels)`.
    """
    mean = input.mean(dim=-1, keepdim=True)
    input_centered = input - mean
    n_times = input_centered.shape[-1]
    covariances = torch.einsum("...ik,...jk->...ij", input_centered, input_centered) / (
        n_times - 1
    )
    return covariances


def real_covariance(X: torch.Tensor) -> torch.Tensor:
    """Computes the real-valued covariance matrix of time series data.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape `(..., C, T)`, where `C` is the number of
        channels and `T` is the number of time samples.

    Returns
    -------
    torch.Tensor
        Real part of the covariance matrix, with shape `(..., C, C)`.
    """
    cov = X @ X.mT.conj() / X.size(-1)
    return cov.real


def cross_covariance(X: torch.Tensor) -> torch.Tensor:
    """Computes the real-valued cross-frequency covariance matrix.

    This function computes the covariance matrix across both frequency bands
    and channels for wavelet-transformed EEG signals. The frequency and channel
    dimensions are flattened into a single feature dimension before computing
    the covariance.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor of shape `(..., F, C, T)`, where `F` is the number of
        frequency bands, `C` is the number of channels, and `T` is the number
        of time samples.

    Returns
    -------
    torch.Tensor
        The real part of the cross-frequency covariance matrix, with shape
        `(..., F * C, F * C)`.
    """
    *batch_shape, n_freqs, n_chans, n_times = X.shape
    X_flat = X.reshape(*batch_shape, n_freqs * n_chans, n_times)
    cov = X_flat @ X_flat.mT.conj() / n_times
    return cov.real
