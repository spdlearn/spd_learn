# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch

from spd_learn.models import SPDTransNet
from spd_learn.modules import (
    SPDTransformerEncoderLayer,
    SPMultiheadAttention,
)


def test_sp_multihead_attention_basic():
    """Verify standard forward pass shape with default settings."""
    embed_dim = 16
    num_heads = 4
    mha = SPMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, embed_dim)

    out = mha(x)
    assert out.shape == (batch_size, seq_len, embed_dim)


def test_sp_multihead_attention_separate_qkv():
    """Verify separate query, key, value with different sequence lengths."""
    embed_dim = 16
    num_heads = 4
    mha = SPMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    batch_size = 3
    seq_len_q = 6
    seq_len_kv = 10

    q = torch.randn(batch_size, seq_len_q, embed_dim)
    k = torch.randn(batch_size, seq_len_kv, embed_dim)
    v = torch.randn(batch_size, seq_len_kv, embed_dim)

    out = mha(q, key=k, value=v)
    assert out.shape == (batch_size, seq_len_q, embed_dim)


def test_sp_multihead_attention_need_weights():
    """Verify returned attention weights when need_weights=True."""
    embed_dim = 8
    num_heads = 2
    mha = SPMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    batch_size = 2
    seq_len_q = 5
    seq_len_k = 7
    q = torch.randn(batch_size, seq_len_q, embed_dim)
    k = torch.randn(batch_size, seq_len_k, embed_dim)
    v = torch.randn(batch_size, seq_len_k, embed_dim)

    out, weights = mha(q, key=k, value=v, need_weights=True)
    assert out.shape == (batch_size, seq_len_q, embed_dim)
    assert weights.shape == (batch_size, seq_len_q, seq_len_k)


def test_sp_multihead_attention_broadcast_dims():
    """Verify SPMultiheadAttention on arbitrary leading batch dimensions."""
    embed_dim = 8
    num_heads = 2
    mha = SPMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    x = torch.randn(2, 3, 4, 6, embed_dim)
    out = mha(x)
    assert out.shape == (2, 3, 4, 6, embed_dim)


def test_sp_multihead_attention_affine_combination():
    """Verify that SP-MHA forms an affine combination of unprojected values."""
    embed_dim = 6
    num_heads = 2
    mha = SPMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # Set constant values
    batch_size = 2
    seq_len = 4
    c = torch.tensor([1.0, 2.0, 3.0, -1.0, 0.5, 2.5])
    v = c.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, embed_dim)
    q = torch.randn(batch_size, seq_len, embed_dim)

    # With initial weights summing to 1 across heads and softmax summing to 1 across keys,
    # any combination of identical vectors must equal the vector itself.
    out = mha(q, value=v)
    torch.testing.assert_close(out, v, atol=1e-5, rtol=1e-5)


def test_sp_multihead_attention_masks():
    """Verify attention masks and key padding masks."""
    embed_dim = 8
    num_heads = 2
    mha = SPMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Boolean key padding mask: mask out last 2 keys
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    key_padding_mask[:, -2:] = True

    out, weights = mha(x, key_padding_mask=key_padding_mask, need_weights=True)
    assert out.shape == (batch_size, seq_len, embed_dim)
    # The masked key positions should have negligible attention weights
    assert torch.all(weights[:, :, -2:] < 1e-4)


def test_sp_multihead_attention_complex():
    """Verify complex dtype support for SP-MHA."""
    embed_dim = 8
    num_heads = 2
    mha = SPMultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dtype=torch.complex64
    )

    x = torch.randn(2, 5, embed_dim, dtype=torch.complex64, requires_grad=True)
    out = mha(x)
    assert out.dtype == torch.complex64
    assert out.shape == (2, 5, embed_dim)

    loss = out.abs().sum()
    loss.backward()
    assert x.grad is not None


def test_spd_transformer_encoder_layer_basic():
    """Verify SPDTransformerEncoderLayer forward and backward pass."""
    embed_dim = 16
    num_heads = 4
    dim_feedforward = 32
    layer = SPDTransformerEncoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
    )

    x = torch.randn(2, 6, embed_dim, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 6, embed_dim)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_spd_transformer_encoder_layer_activations():
    """Verify both GELU and ReLU activation functions."""
    for act in ["gelu", "relu"]:
        layer = SPDTransformerEncoderLayer(
            embed_dim=8, num_heads=2, activation=act
        )
        x = torch.randn(2, 4, 8)
        out = layer(x)
        assert out.shape == (2, 4, 8)

    with pytest.raises(ValueError, match="Unsupported activation string"):
        SPDTransformerEncoderLayer(embed_dim=8, activation="invalid_activation")


def test_spdtransnet_default():
    """Verify SPDTransNet instantiation and forward with default arguments."""
    model = SPDTransNet(n_chans=22, n_outputs=2)
    x = torch.randn(2, 22, 1000)

    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)


def test_spdtransnet_4d_input():
    """Verify SPDTransNet accepts 4D inputs (batch, 1, n_chans, n_times)."""
    model = SPDTransNet(n_chans=22, n_outputs=2)
    x = torch.randn(2, 1, 22, 1000)

    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)


def test_spdtransnet_custom_params():
    """Verify SPDTransNet with custom architecture hyperparameters."""
    model = SPDTransNet(
        n_chans=16,
        n_outputs=4,
        n_patches=4,
        n_filters=8,
        num_layers=1,
        num_heads=2,
        dim_feedforward=32,
        drop_prob=0.0,
        temporal_kernel_size=15,
    )
    x = torch.randn(3, 16, 500)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3, 4)


def test_spdtransnet_backward():
    """Verify gradient flow across all modules in SPDTransNet."""
    model = SPDTransNet(
        n_chans=8,
        n_outputs=2,
        n_patches=4,
        n_filters=8,
        num_layers=1,
        num_heads=2,
    )
    x = torch.randn(2, 8, 200, requires_grad=True)

    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Check key parameter gradients
    assert model.temporal_conv.weight.grad is not None
    assert model.spatial_conv.weight.grad is not None
    assert model.pos_embedding.grad is not None
    assert model.classifier.weight.grad is not None


def test_spdtransnet_eval_mode():
    """Verify SPDTransNet behaves consistently in evaluation mode."""
    model = SPDTransNet(n_chans=8, n_outputs=2, n_patches=4, n_filters=8)
    model.eval()

    x = torch.randn(2, 8, 200)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    torch.testing.assert_close(out1, out2)


def test_spdtransnet_braindecode_integration():
    """Verify braindecode EEGClassifier integration."""
    try:
        from braindecode import EEGClassifier
    except ImportError:
        pytest.skip("braindecode is not installed")

    clf = EEGClassifier(
        module=SPDTransNet,
        module__n_chans=16,
        module__n_outputs=2,
        module__n_patches=4,
        module__n_filters=8,
        module__num_layers=1,
        module__num_heads=2,
    )
    clf.initialize()
    assert clf.module_ is not None
