# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPMultiheadAttention(nn.Module):
    r"""Structure-Preserving Multihead Attention (SP-MHA).

    This module implements Structure-Preserving Multi-Head Attention for
    sequences of Symmetric Positive Definite (SPD) representations or their
    isometric tangent vectors :cite:p:`seraphim2023structure`.

    Unlike standard Multi-Head Attention (which projects values through a
    learned linear matrix :math:`W_V` and concatenates head outputs followed by
    an output projection :math:`W_O`), SP-MHA preserves the algebraic and
    geometric structure of the input representations:

    1. Queries :math:`Q` and Keys :math:`K` are projected into head-specific
       subspaces to compute scaled dot-product attention maps for each head.
    2. Head attention weight matrices are combined linearly across the head
       dimension into a single attention matrix :math:`\tilde{A}` via a
       structure-preserving linear layer:

       .. math::

           \tilde{A} = \sum_{h=1}^H w_h A_h

    3. The combined attention matrix :math:`\tilde{A}` is applied directly to
       the unprojected Values :math:`V`:

       .. math::

           \text{out} = \tilde{A} V

    This ensures that each output token is an affine combination of the input
    tokens, preserving closure within Riemannian tangent spaces or Euclideanized
    SPD features.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the input feature embeddings.
    num_heads : int, default=4
        Number of attention heads.
    kdim : int, optional
        Dimensionality of projected keys and queries per head. If ``None``,
        defaults to ``max(1, embed_dim // num_heads)``.
    dropout : float, default=0.0
        Dropout probability applied to attention weights.
    bias : bool, default=True
        Whether to include additive bias in key and query projections.
    device : torch.device, optional
        Device on which module parameters are allocated.
    dtype : torch.dtype, optional
        Data type of module parameters.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import SPMultiheadAttention
    >>> mha = SPMultiheadAttention(embed_dim=16, num_heads=4)
    >>> X = torch.randn(2, 8, 16)
    >>> out = mha(X)
    >>> out.shape
    torch.Size([2, 8, 16])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        kdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim
        self.head_dim = kdim if kdim is not None else max(1, embed_dim // num_heads)

        self.q_proj = nn.Linear(
            embed_dim,
            num_heads * self.head_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            embed_dim,
            num_heads * self.head_dim,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.head_combination = nn.Linear(
            num_heads,
            1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            self.head_combination.weight.fill_(1.0 / num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""Compute structure-preserving multihead attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape `(..., seq_len_q, embed_dim)`.
        key : torch.Tensor, optional
            Key tensor of shape `(..., seq_len_k, embed_dim)`. Defaults to
            ``query`` when ``None``.
        value : torch.Tensor, optional
            Value tensor of shape `(..., seq_len_k, embed_dim)`. Defaults to
            ``query`` when ``None``.
        key_padding_mask : torch.Tensor, optional
            Optional boolean mask of shape `(..., seq_len_k)` where ``True``
            indicates elements to ignore in attention.
        need_weights : bool, default=False
            If ``True``, returns the combined attention matrix alongside
            the output tensor.
        attn_mask : torch.Tensor, optional
            Additive or boolean attention mask of shape
            `(..., seq_len_q, seq_len_k)`.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            If ``need_weights=False``, returns the attention output tensor of
            shape `(..., seq_len_q, embed_dim)`. If ``need_weights=True``,
            returns a tuple `(out, combined_attn)` where `combined_attn` has
            shape `(..., seq_len_q, seq_len_k)`.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        *batch_dims, seq_len_q, _ = query.shape
        *_, seq_len_k, _ = key.shape

        q = self.q_proj(query).view(
            *batch_dims, seq_len_q, self.num_heads, self.head_dim
        ).transpose(-3, -2)
        k = self.k_proj(key).view(
            *batch_dims, seq_len_k, self.num_heads, self.head_dim
        ).transpose(-3, -2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(-2).unsqueeze(-2),
                float("-inf"),
            )

        if scores.is_complex():
            attn_weights = F.softmax(scores.real, dim=-1).to(scores.dtype)
        else:
            attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # Move num_heads (at -3) to the last dimension: (..., seq_len_q, seq_len_k, num_heads)
        attn_perm = attn_weights.transpose(-3, -2).transpose(-2, -1)
        combined_attn = self.head_combination(attn_perm).squeeze(-1)

        out = torch.matmul(combined_attn, value)

        if need_weights:
            return out, combined_attn
        return out


class SPDTransformerEncoderLayer(nn.Module):
    r"""Transformer Encoder Layer for SPD sequences.

    This layer combines :class:`SPMultiheadAttention` with residual
    connections, layer normalization, and a two-layer feedforward network
    (FFN) :cite:p:`seraphim2023structure`.

    Parameters
    ----------
    embed_dim : int
        Feature dimensionality of the input sequence.
    num_heads : int, default=4
        Number of attention heads for :class:`SPMultiheadAttention`.
    dim_feedforward : int, optional
        Hidden dimension of the feedforward network. If ``None``, defaults
        to ``4 * embed_dim``.
    dropout : float, default=0.1
        Dropout probability applied across attention weights and activations.
    activation : str or callable, default="gelu"
        Nonlinear activation function for the feedforward network.
    layer_norm_eps : float, default=1e-5
        Epsilon value for numerical stability in :class:`torch.nn.LayerNorm`.
    device : torch.device, optional
        Device on which module parameters are allocated.
    dtype : torch.dtype, optional
        Data type of module parameters.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.modules import SPDTransformerEncoderLayer
    >>> layer = SPDTransformerEncoderLayer(embed_dim=16, num_heads=4)
    >>> X = torch.randn(2, 8, 16)
    >>> out = layer(X)
    >>> out.shape
    torch.Size([2, 8, 16])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = "gelu",
        layer_norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = (
            dim_feedforward if dim_feedforward is not None else 4 * embed_dim
        )

        self.self_attn = SPMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

        self.linear1 = nn.Linear(
            embed_dim,
            self.dim_feedforward,
            device=device,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            self.dim_feedforward,
            embed_dim,
            device=device,
            dtype=dtype,
        )

        self.norm1 = nn.LayerNorm(
            embed_dim,
            eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.norm2 = nn.LayerNorm(
            embed_dim,
            eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            if activation.lower() == "gelu":
                self.activation = nn.GELU()
            elif activation.lower() == "relu":
                self.activation = nn.ReLU()
            else:
                raise ValueError(f"Unsupported activation string: {activation}")
        else:
            self.activation = activation

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Parameters
        ----------
        src : torch.Tensor
            Input sequence of shape `(..., seq_len, embed_dim)`.
        src_mask : torch.Tensor, optional
            Optional attention mask for the self-attention layer.
        src_key_padding_mask : torch.Tensor, optional
            Optional key padding mask for the self-attention layer.

        Returns
        -------
        torch.Tensor
            Output sequence tensor of shape `(..., seq_len, embed_dim)`.
        """
        attn_out = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = src + self.dropout1(attn_out)
        x = self.norm1(x)

        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ffn_out)
        x = self.norm2(x)
        return x
