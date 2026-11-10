# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-implementation of SPDTransNet / SP-MHA (Seraphim et al., EUSIPCO, 2024).
# Reference implementation: https://github.com/MathieuSeraphim/SPDTransNet

# All linear maps are unconstrained maps between symmetric coordinate spaces;
# Stiefel/SPD parametrizations are neither required nor appropriate for these maps.
# There is no value projection, head concatenation, or output feature projection
# in SP-MHA. Masks/padding/causal operation are deliberately unsupported.


import copy
import math

import torch

from torch import nn

from spd_learn.functional.spdtransnet import (
    log_tokens_to_spd,
    spd_to_log_tokens,
    triangular_dimension,
)


def _triangular_size(d: int) -> None:
    if d < 1 or (math.isqrt(8 * d + 1) ** 2 != 8 * d + 1):
        raise ValueError("coordinate dimension must be a positive triangular number")


def _reject_masks(*masks: torch.Tensor | None, is_causal: bool = False) -> None:
    if any(mask is not None for mask in masks) or is_causal:
        raise ValueError("masks, padding and causal attention are not supported")


class SPDLogTokenizer(nn.Module):
    r"""Convert prepared SPD matrices to symmetric log tokens.

    See :cite:p:`seraphim2024structure`.

    Parameters
    ----------
    matrix_size : int, default=9
        Size of each prepared (already augmented/whitened, if desired)
        matrix.
    preserve_norm : bool, default=False
        Raw canonical upper triangle by default, matching the author code.
        True uses sqrt(2) off-diagonal scaling and the matching inverse.
    eigenvalue_floor : float or None
        Optional explicit PSD rectification. Core log numerical floor also
        applies.
    matrix_scale : float, default=1.0
        Positive multiplier before the log and spectral floor.
    device : torch.device or str, optional
        Accepted for API consistency; this stateless operation follows the
        input device.
    dtype : torch.dtype, optional
        Accepted for API consistency; this stateless operation follows the
        input dtype.

    Notes
    -----
    The optional eigenvalue floor is explicit regularization. inverse uses
    the same basis and scale but cannot undo flooring. Dtype and device
    follow the input.
    """

    def __init__(
        self,
        matrix_size=9,
        *,
        preserve_norm=False,
        eigenvalue_floor=None,
        matrix_scale=1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.token_dim = triangular_dimension(matrix_size)
        self.preserve_norm = preserve_norm
        self.eigenvalue_floor = eigenvalue_floor
        self.matrix_scale = matrix_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Tokenize prepared SPD matrices.

        Parameters
        ----------
        x : torch.Tensor
            Matrices of shape (..., matrix_size, matrix_size).

        Returns
        -------
        torch.Tensor
            Log tokens of shape (..., matrix_size * (matrix_size + 1) // 2).
        """
        if x.shape[-2:] != (self.matrix_size, self.matrix_size):
            raise ValueError("input matrix size does not match tokenizer")
        return spd_to_log_tokens(
            x,
            preserve_norm=self.preserve_norm,
            eigenvalue_floor=self.eigenvalue_floor,
            matrix_scale=self.matrix_scale,
        )

    def inverse(self, tokens: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct matrices from log tokens.

        Parameters
        ----------
        tokens : torch.Tensor
            Coordinates of shape (..., matrix_size * (matrix_size + 1) // 2).

        Returns
        -------
        torch.Tensor
            Matrices of shape (..., matrix_size, matrix_size).
        """
        if tokens.ndim < 1 or tokens.shape[-1] != self.token_dim:
            raise ValueError("input coordinate dimension does not match tokenizer")
        return log_tokens_to_spd(
            tokens,
            preserve_norm=self.preserve_norm,
            matrix_scale=self.matrix_scale,
        )


class SPMultiheadAttention(nn.Module):
    r"""Apply SP-MHA to symmetric log coordinates.

    See :cite:p:`seraphim2024structure`.

    Parameters
    ----------
    embed_dim : int, default=6
        Positive triangular coordinate dimension, divisible by num_heads.
    num_heads : int, default=2
        Number of independently normalized attention maps.
    dropout : float, default=0.0
        Dropout of each head's post-softmax probabilities during training.
    bias : bool, default=True
        Include biases in query/key projections (head mixing is
        bias-free).
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.

    Notes
    -----
    Q/K are projected per head. A signed linear combination of head maps
    multiplies the unprojected values. Raw covariance entries are not valid
    value tokens. Masks, padding and causal operation are unsupported.
    """

    def __init__(
        self,
        embed_dim=6,
        num_heads=2,
        dropout=0.0,
        bias=True,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        _triangular_size(embed_dim)
        if num_heads < 1 or embed_dim % num_heads:
            raise ValueError("num_heads must be positive and divide embed_dim")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        factory = {"device": device, "dtype": dtype}
        self.query = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)
        self.key = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)
        # Match the distribution of the active Q/K slices of packed MHA weights,
        # without retaining the author's unused V/output projection parameters.
        bound = math.sqrt(6 / (4 * embed_dim))
        nn.init.uniform_(self.query.weight, -bound, bound)
        nn.init.uniform_(self.key.weight, -bound, bound)
        if bias:
            nn.init.zeros_(self.query.bias)
            nn.init.zeros_(self.key.bias)
        self.head_mix = nn.Linear(num_heads, 1, bias=False, **factory)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        *,
        key_padding_mask=None,
        attn_mask=None,
        is_causal=False,
        need_weights=False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Apply structure-preserving multihead attention.

        Parameters
        ----------
        query : torch.Tensor
            Log tokens of shape (batch, tokens, embed_dim).
        key : torch.Tensor or None
            Keys with the query shape; None uses query.
        value : torch.Tensor or None
            Unprojected values with the query shape; None uses query.
        key_padding_mask : None
            Reserved argument; padding masks are not supported.
        attn_mask : None
            Reserved argument; masks are not supported.
        is_causal : bool, default=False
            Must be False; causal attention is not supported.
        need_weights : bool, default=False
            Return the combined attention map.

        Returns
        -------
        output : torch.Tensor
            Attended log tokens with the query shape.
        weights : torch.Tensor or None
            Combined map of shape (batch, tokens, tokens) when
            need_weights=True.
        """
        _reject_masks(key_padding_mask, attn_mask, is_causal=is_causal)
        key = query if key is None else key
        value = query if value is None else value
        if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
            raise ValueError(
                "Q, K, V must have identical (batch, tokens, embed_dim) shapes"
            )
        if query.shape[-1] != self.embed_dim or query.shape[-2] < 1:
            raise ValueError("wrong coordinate dimension or empty token sequence")
        if any(
            x.dtype not in (torch.float32, torch.float64) for x in (query, key, value)
        ):
            raise TypeError("SP-MHA requires real float32/float64 log coordinates")
        b, s, _ = query.shape
        q = (
            self.query(query)
            .reshape(b, s, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = self.key(key).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        probabilities = ((q / math.sqrt(self.head_dim)) @ k.mT).softmax(-1)
        maps = self.attention_dropout(probabilities).movedim(1, -1)
        combined = self.head_mix(maps).squeeze(-1)
        return combined @ value, combined if need_weights else None


class SPDTransformerEncoderLayer(nn.Module):
    r"""Encode log tokens with SP-MHA and a residual feedforward block.

    See :cite:p:`seraphim2024structure`.

    Parameters
    ----------
    embed_dim : int, default=6
        Triangular log-coordinate dimension.
    num_heads : int, default=2
        Number of attention heads; must divide embed_dim.
    feedforward_dim : int, default=10
        Width of the encoder feedforward layer.
    dropout : float, default=0.1
        Dropout probability.
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.
    """

    def __init__(
        self,
        embed_dim=6,
        num_heads=2,
        feedforward_dim=10,
        dropout=0.1,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        _triangular_size(feedforward_dim)
        factory = {"device": device, "dtype": dtype}
        self.attention = SPMultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            **factory,
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim, **factory),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim, **factory),
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5, **factory)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5, **factory)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        with torch.no_grad():
            self.attention.head_mix.weight.mul_(0.2)
            self.ff[3].weight.mul_(0.1)
            if self.ff[3].bias is not None:
                self.ff[3].bias.zero_()

    def forward(self, x: torch.Tensor, *, attn_mask=None, key_padding_mask=None):
        r"""Encode an ordered log-token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Log tokens of shape (batch, tokens, embed_dim).
        attn_mask : None
            Reserved argument; masks are not supported.
        key_padding_mask : None
            Reserved argument; padding masks are not supported.

        Returns
        -------
        torch.Tensor
            Encoded tokens with the input shape.
        """
        _reject_masks(attn_mask, key_padding_mask)
        attended, _ = self.attention(x)
        x = self.norm1(x + self.dropout1(attended))
        return self.norm2(x + self.dropout2(self.ff(x)))


class LearnedSinusoidalPosition(nn.Module):
    r"""Add learned sinusoidal positional features to log tokens.

    Parameters
    ----------
    sequence_length : int, default=30
        Number of ordered input tokens.
    embed_dim : int, default=6
        Triangular log-coordinate dimension.
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.
    """

    sinusoid: torch.Tensor

    def __init__(self, sequence_length=30, embed_dim=6, *, device=None, dtype=None):
        super().__init__()
        _triangular_size(embed_dim)
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        factory = {"device": device, "dtype": dtype}
        positions = torch.arange(sequence_length, **factory).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, embed_dim, 2, **factory) * (-math.log(10000) / embed_dim)
        )
        angles = positions * frequency
        table = torch.empty(sequence_length, embed_dim, **factory)
        table[:, 0::2] = angles.sin()
        table[:, 1::2] = angles[:, : embed_dim // 2].cos()
        self.register_buffer("sinusoid", table)
        self.mapping = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, **factory),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, **factory),
        )
        with torch.no_grad():
            eye = torch.eye(embed_dim, **factory) * math.sqrt(2.0)
            self.mapping[0].weight.mul_(0.02).add_(eye)
            self.mapping[0].bias.zero_()
            self.mapping[2].weight.mul_(0.02).add_(eye)
            self.mapping[2].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode an ordered log-token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Log tokens of shape (batch, tokens, embed_dim).

        Returns
        -------
        torch.Tensor
            Encoded tokens with the input shape.
        """
        if x.ndim != 3 or x.shape[-2:] != self.sinusoid.shape:
            raise ValueError("expected (batch, configured sequence_length, embed_dim)")
        return self.mapping(x + self.sinusoid)


class SPDTemporalEncoder(nn.Module):
    r"""Stack positional encoding and structure-preserving encoder layers.

    See :cite:p:`seraphim2024structure`.

    Parameters
    ----------
    sequence_length : int, default=30
        Number of ordered input tokens.
    embed_dim : int, default=6
        Triangular log-coordinate dimension.
    num_heads : int, default=2
        Number of attention heads; must divide embed_dim.
    feedforward_dim : int, default=10
        Width of the encoder feedforward layer.
    num_layers : int, default=1
        Number of independently registered encoder layers.
    dropout : float, default=0.1
        Dropout probability.
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.
    """

    def __init__(
        self,
        sequence_length=30,
        embed_dim=6,
        num_heads=2,
        feedforward_dim=10,
        num_layers=1,
        dropout=0.1,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        factory = {"device": device, "dtype": dtype}
        self.position = LearnedSinusoidalPosition(sequence_length, embed_dim, **factory)
        layer = SPDTransformerEncoderLayer(
            embed_dim,
            num_heads,
            feedforward_dim,
            dropout,
            **factory,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode an ordered log-token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Log tokens of shape (batch, tokens, embed_dim).

        Returns
        -------
        torch.Tensor
            Encoded tokens with the input shape.
        """
        x = self.position(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SPDIntraEpochEncoder(nn.Module):
    r"""Encode and pool contiguous tokens within each epoch.

    See :cite:p:`seraphim2024structure`.

    Parameters
    ----------
    tokens_per_epoch : int, default=210
        Number of band/window tokens per epoch.
    feature_tokens : int, default=10
        Number of pooled tokens per epoch; must divide the input count.
    embed_dim : int, default=351
        Triangular log-coordinate dimension.
    num_heads : int, default=9
        Number of attention heads; must divide embed_dim.
    feedforward_dim : int, default=1035
        Width of the encoder feedforward layer.
    num_layers : int, default=5
        Number of independently registered encoder layers.
    dropout : float, default=0.1
        Dropout probability.
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.

    Notes
    -----
    feature_tokens must divide tokens_per_epoch. Epochs are encoded
    independently before equal-bin averaging.
    """

    def __init__(
        self,
        tokens_per_epoch=210,
        feature_tokens=10,
        embed_dim=351,
        num_heads=9,
        feedforward_dim=1035,
        num_layers=5,
        dropout=0.1,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if (
            not 0 < feature_tokens < tokens_per_epoch
            or tokens_per_epoch % feature_tokens
        ):
            raise ValueError(
                "feature_tokens must divide and be smaller than tokens_per_epoch"
            )
        self.tokens_per_epoch = tokens_per_epoch
        self.feature_tokens = feature_tokens
        self.encoder = SPDTemporalEncoder(
            tokens_per_epoch,
            embed_dim,
            num_heads,
            feedforward_dim,
            num_layers,
            dropout,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode each epoch and average contiguous token groups.

        Parameters
        ----------
        x : torch.Tensor
            Log tokens of shape (batch, epochs, tokens_per_epoch, embed_dim).

        Returns
        -------
        torch.Tensor
            Features of shape (batch, epochs, feature_tokens, embed_dim).
        """
        if x.ndim != 4 or x.shape[-2] != self.tokens_per_epoch:
            raise ValueError("expected (batch, epochs, tokens_per_epoch, embed_dim)")
        b, epochs, tokens, d = x.shape
        encoded = self.encoder(x.reshape(b * epochs, tokens, d))
        return encoded.reshape(
            b,
            epochs,
            self.feature_tokens,
            tokens // self.feature_tokens,
            d,
        ).mean(-2)


class SPDInterEpochEncoder(nn.Module):
    r"""Encode log tokens jointly across a chronological context.

    See :cite:p:`seraphim2024structure`.

    Parameters
    ----------
    context_epochs : int, default=21
        Odd number of chronological epochs in a context.
    feature_tokens : int, default=10
        Number of pooled tokens per epoch; must divide the input count.
    embed_dim : int, default=351
        Triangular log-coordinate dimension.
    num_heads : int, default=3
        Number of attention heads; must divide embed_dim.
    feedforward_dim : int, default=1035
        Width of the encoder feedforward layer.
    num_layers : int, default=4
        Number of independently registered encoder layers.
    dropout : float, default=0.1
        Dropout probability.
    device : torch.device or str, optional
        Device for parameters and buffers.
    dtype : torch.dtype, optional
        Data type for parameters and buffers.

    Notes
    -----
    Positions follow the flattened epoch-major order. Future epochs may
    affect every output.
    """

    def __init__(
        self,
        context_epochs=21,
        feature_tokens=10,
        embed_dim=351,
        num_heads=3,
        feedforward_dim=1035,
        num_layers=4,
        dropout=0.1,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if context_epochs < 1 or feature_tokens < 1:
            raise ValueError("context_epochs and feature_tokens must be positive")
        self.context_epochs, self.feature_tokens = context_epochs, feature_tokens
        self.encoder = SPDTemporalEncoder(
            context_epochs * feature_tokens,
            embed_dim,
            num_heads,
            feedforward_dim,
            num_layers,
            dropout,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a complete epoch-major context.

        Parameters
        ----------
        x : torch.Tensor
            Log tokens of shape (batch, context_epochs, feature_tokens,
            embed_dim).

        Returns
        -------
        torch.Tensor
            Encoded tokens with the input shape.
        """
        if x.ndim != 4 or x.shape[1:3] != (self.context_epochs, self.feature_tokens):
            raise ValueError(
                "expected (batch, context_epochs, feature_tokens, embed_dim)"
            )
        b, epochs, features, d = x.shape
        return self.encoder(x.reshape(b, epochs * features, d)).reshape_as(x)


SPDPositionEmbedding = LearnedSinusoidalPosition
