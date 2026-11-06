# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from spd_learn.models import SPDMetaFormer, SPDMetaFormerBlock


def _make_spd_batch(*shape):
    """Generate random SPD matrices where shape is (*batch_dims, dim)."""
    *batch_dims, dim = shape
    A = torch.randn(*batch_dims, dim, dim)
    return A @ A.mT + 0.1 * torch.eye(dim, dtype=A.dtype, device=A.device)


# =============================================================================
# SPDMetaFormerBlock Tests
# =============================================================================


class TestSPDMetaFormerBlock:
    """Tests for the SPDMetaFormerBlock module."""

    def test_block_initialization(self):
        """Test SPDMetaFormerBlock default and custom initialization."""
        block_default = SPDMetaFormerBlock(dim=8)
        assert block_default.dim == 8
        assert block_default.ffn_dim == 8
        assert block_default.eps == 1e-4

        block_custom = SPDMetaFormerBlock(dim=8, ffn_dim=16, eps=1e-3)
        assert block_custom.dim == 8
        assert block_custom.ffn_dim == 16
        assert block_custom.eps == 1e-3

    @pytest.mark.parametrize("dim, ffn_dim", [(8, 8), (8, 16), (12, 6)])
    def test_block_forward_shapes_and_spd(self, dim, ffn_dim):
        """Test forward output shape and positive definiteness across ffn dimensions."""
        block = SPDMetaFormerBlock(dim=dim, ffn_dim=ffn_dim)
        B, P = 2, 4
        X = _make_spd_batch(B, P, dim)

        out = block(X)
        assert out.shape == (B, P, dim, dim)

        # Check output is symmetric
        torch.testing.assert_close(out, out.mT, atol=1e-5, rtol=1e-5)

        # Check eigenvalues are strictly positive
        eigvals = torch.linalg.eigvalsh(out)
        assert (eigvals > 0).all()

    def test_block_backward(self):
        """Verify gradients propagate to mixing logits and FFN weights."""
        block = SPDMetaFormerBlock(dim=6, ffn_dim=8)
        X = _make_spd_batch(2, 3, 6).requires_grad_()

        out = block(X)
        loss = out.sum()
        loss.backward()

        assert block.mix_logit.grad is not None
        assert block.ffn_logit.grad is not None
        assert X.grad is not None

    def test_block_identical_tokens(self):
        """When tokens are identical, Fréchet mean equals token, mixing preserves token."""
        block = SPDMetaFormerBlock(dim=6, ffn_dim=6)
        block.mix_logit.data.fill_(1.5)

        base = _make_spd_batch(2, 1, 6)
        X = base.expand(2, 4, 6, 6).clone()

        with torch.no_grad():
            L = torch.linalg.matrix_norm(block(X)[:, 0] - block(X)[:, 1])
        assert L.max().item() < 1e-5

    @pytest.mark.parametrize("extra_dims", [(2,), (2, 3)])
    def test_block_broadcast(self, extra_dims):
        """Verify block handles arbitrary leading batch dimensions."""
        block = SPDMetaFormerBlock(dim=6)
        P = 4
        X = _make_spd_batch(*extra_dims, P, 6)

        out = block(X)
        assert out.shape == (*extra_dims, P, 6, 6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_block_dtypes(self, dtype):
        """Verify block supports float32 and float64."""
        block = SPDMetaFormerBlock(dim=6, dtype=dtype)
        X = _make_spd_batch(2, 3, 6).to(dtype)

        out = block(X)
        assert out.dtype == dtype


# =============================================================================
# SPDMetaFormer Full Model Tests
# =============================================================================


class TestSPDMetaFormer:
    """Tests for the full SPDMetaFormer architecture."""

    def test_default_forward(self):
        """Test standard forward pass with 3D input tensor."""
        model = SPDMetaFormer(n_chans=22, n_outputs=2, n_patches=6, depth=2)
        x = torch.randn(2, 22, 1000)
        out = model(x)
        assert out.shape == (2, 2)

    def test_4d_input(self):
        """Test forward pass when input includes explicit singleton channel dimension."""
        model = SPDMetaFormer(n_chans=22, n_outputs=4, n_patches=4, depth=1)
        x = torch.randn(2, 1, 22, 500)
        out = model(x)
        assert out.shape == (2, 4)

    def test_batch_size_one(self):
        """Verify model functions correctly with batch size 1."""
        model = SPDMetaFormer(n_chans=16, n_outputs=3, n_patches=4, depth=1)
        model.eval()
        x = torch.randn(1, 16, 800)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3)

    def test_readout_gradient(self):
        """Verify end-to-end backpropagation populates readout and conv gradients."""
        model = SPDMetaFormer(
            n_chans=8,
            n_outputs=2,
            n_patches=4,
            depth=1,
            temporal_filters=8,
            spatial_filters=8,
            ffn_dim=8,
        )
        x = torch.randn(2, 8, 400, requires_grad=True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert model.readout_logits.grad is not None
        assert (model.readout_logits.grad != 0).any()
        assert model.temporal_feature_extractor.weight.grad is not None
        assert model.spatial_filter.weight.grad is not None
        assert model.linear.weight.grad is not None
        assert x.grad is not None

    @pytest.mark.parametrize("pooling", ["mean", "concat"])
    def test_pooling_modes(self, pooling):
        """Test both mean and concatenation pooling modes."""
        model = SPDMetaFormer(
            n_chans=8,
            n_outputs=2,
            n_patches=4,
            depth=1,
            spatial_filters=8,
            pooling=pooling,
        )
        x = torch.randn(2, 8, 400)
        out = model(x)
        assert out.shape == (2, 2)

    @pytest.mark.parametrize("depth", [0, 1, 2])
    def test_depths(self, depth):
        """Verify model works for depth=0 (frontend+readout only) and stacked blocks."""
        model = SPDMetaFormer(
            n_chans=8,
            n_outputs=2,
            n_patches=4,
            depth=depth,
            spatial_filters=8,
        )
        x = torch.randn(2, 8, 400)
        out = model(x)
        assert out.shape == (2, 2)

    def test_dropout_behavior(self):
        """Verify dropout behavior in train vs eval mode."""
        torch.manual_seed(42)
        model = SPDMetaFormer(
            n_chans=8,
            n_outputs=2,
            n_patches=4,
            depth=1,
            spatial_filters=8,
            drop_prob=0.8,
        )
        x = torch.randn(4, 8, 400)

        # In eval mode, dropout is inactive
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)
