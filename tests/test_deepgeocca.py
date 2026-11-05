# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch.nn as nn

from spd_learn.models import DeepGeoCCA, GeodesicCCALoss


def _generate_spd_batch(
    batch_size: int, d: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Helper to generate a batch of random well-conditioned SPD matrices."""
    torch.manual_seed(42)
    A = torch.randn(batch_size, d, d, dtype=dtype)
    spd = A @ A.transpose(-2, -1) + torch.eye(d, dtype=dtype) * 0.5
    return spd


class TestGeodesicCCALoss:
    """Tests for GeodesicCCALoss."""

    def test_geodesic_cca_loss_shape_and_value(self):
        """Test output shape and sanity of computed loss."""
        loss_fn = GeodesicCCALoss(outdim_size=5, eps=1e-4)
        Y1 = _generate_spd_batch(batch_size=8, d=6)
        Y2 = _generate_spd_batch(batch_size=8, d=6)

        loss = loss_fn(Y1, Y2)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        # Loss is negative canonical correlation, so it must be negative
        assert loss.item() < 0.0

    def test_identical_views_maximum_correlation(self):
        """Identical views should achieve near-maximal correlation (approx -outdim_size)."""
        d = 4
        m = d * (d + 1) // 2
        outdim_size = 5
        loss_fn = GeodesicCCALoss(outdim_size=outdim_size, eps=1e-5)

        # Batch with diverse variance to span tangent space
        torch.manual_seed(123)
        Y = _generate_spd_batch(batch_size=30, d=d, dtype=torch.float64)

        loss = loss_fn(Y, Y)
        k = min(outdim_size, m)
        # Each canonical correlation should be ~ 1.0, so sum is ~ k, loss is ~ -k
        assert pytest.approx(-k, abs=0.1) == loss.item()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_geodesic_cca_loss_backward(self, dtype):
        """Test gradient propagation through GeodesicCCALoss back to inputs."""
        loss_fn = GeodesicCCALoss(outdim_size=3, eps=1e-4)
        Y1 = _generate_spd_batch(batch_size=6, d=4, dtype=dtype).requires_grad_(True)
        Y2 = _generate_spd_batch(batch_size=6, d=4, dtype=dtype).requires_grad_(True)

        loss = loss_fn(Y1, Y2)
        loss.backward()

        assert Y1.grad is not None
        assert Y2.grad is not None
        assert not torch.isnan(Y1.grad).any()
        assert not torch.isnan(Y2.grad).any()
        assert not torch.isinf(Y1.grad).any()
        assert not torch.isinf(Y2.grad).any()

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_geodesic_cca_loss_small_batches(self, batch_size):
        """Test loss computation on edge-case small batch sizes."""
        loss_fn = GeodesicCCALoss(outdim_size=4, eps=1e-4)
        Y1 = _generate_spd_batch(batch_size=batch_size, d=3)
        Y2 = _generate_spd_batch(batch_size=batch_size, d=3)

        loss = loss_fn(Y1, Y2)
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_outdim_size_larger_than_tangent_dim(self):
        """When outdim_size > m, it should clamp k to m without crashing."""
        d = 2
        m = d * (d + 1) // 2  # 3
        loss_fn = GeodesicCCALoss(outdim_size=10, eps=1e-4)
        Y1 = _generate_spd_batch(batch_size=5, d=d)
        Y2 = _generate_spd_batch(batch_size=5, d=d)

        loss = loss_fn(Y1, Y2)
        assert torch.isfinite(loss)

    def test_input_validation_errors(self):
        """Check that invalid shapes raise ValueError."""
        loss_fn = GeodesicCCALoss()

        # Mismatched shapes
        with pytest.raises(ValueError, match="same shape"):
            loss_fn(torch.eye(3).expand(2, 3, 3), torch.eye(4).expand(2, 4, 4))

        # Not at least 3D
        with pytest.raises(ValueError, match="at least 3 dimensions"):
            loss_fn(torch.eye(3), torch.eye(3))

        # Not square
        with pytest.raises(ValueError, match="square"):
            loss_fn(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))


class TestDeepGeoCCA:
    """Tests for DeepGeoCCA model."""

    def test_initialization_defaults(self):
        """Verify model initializes with expected layers and defaults."""
        model = DeepGeoCCA(n_chans=22, n_outputs=2)
        assert model.n_chans == 22
        assert model.n_outputs == 2
        assert model.bimap_dims == (16, 10)
        assert model.outdim_size == 10
        assert isinstance(model.view1_conv, nn.Conv1d)
        assert isinstance(model.view2_conv, nn.Conv1d)
        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.out_features == 2
        # d_out = 10 -> vec_dim = 10 * 11 // 2 = 55 -> 2 * vec_dim = 110
        assert model.classifier.in_features == 110

    def test_encode_views_single_tensor(self):
        """encode_views with single 3D EEG input returns two batches of SPD matrices."""
        model = DeepGeoCCA(n_chans=8, bimap_dims=(6, 4))
        x = torch.randn(3, 8, 200)

        Y1, Y2 = model.encode_views(x)
        assert Y1.shape == (3, 4, 4)
        assert Y2.shape == (3, 4, 4)

        # Check positive definiteness (all eigenvalues > 0)
        eig1 = torch.linalg.eigvalsh(Y1)
        eig2 = torch.linalg.eigvalsh(Y2)
        assert (eig1 > 0).all()
        assert (eig2 > 0).all()

    def test_encode_views_paired_inputs(self):
        """encode_views supports paired views as tuple or separate args."""
        model = DeepGeoCCA(n_chans=8, bimap_dims=(6, 4))
        x1 = torch.randn(3, 8, 200)
        x2 = torch.randn(3, 8, 200)

        # As tuple
        Y1_t, Y2_t = model.encode_views((x1, x2))
        assert Y1_t.shape == (3, 4, 4)
        assert Y2_t.shape == (3, 4, 4)

        # As two arguments
        Y1_a, Y2_a = model.encode_views(x1, x2)
        assert torch.allclose(Y1_t, Y1_a)
        assert torch.allclose(Y2_t, Y2_a)

    def test_forward_single_and_paired_inputs(self):
        """Forward pass produces logits of shape (batch, n_outputs)."""
        model = DeepGeoCCA(n_chans=10, n_outputs=4, bimap_dims=(8, 5))
        x = torch.randn(4, 10, 250)

        # Single tensor input
        out1 = model(x)
        assert out1.shape == (4, 4)

        # Tuple input
        x2 = torch.randn(4, 10, 250)
        out2 = model((x, x2))
        assert out2.shape == (4, 4)

        # Two-argument input
        out3 = model(x, x2)
        assert torch.allclose(out2, out3)

    def test_compute_cca_loss_backward(self):
        """compute_cca_loss returns scalar loss and backpropagates through model."""
        model = DeepGeoCCA(n_chans=6, bimap_dims=(4,), outdim_size=4)
        x = torch.randn(5, 6, 150)

        loss = model.compute_cca_loss(x)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

        loss.backward()

        # Check gradients in both encoders and filter convolutions
        assert model.view1_conv.weight.grad is not None
        assert model.view2_conv.weight.grad is not None
        assert not torch.isnan(model.view1_conv.weight.grad).any()
        assert not torch.isnan(model.view2_conv.weight.grad).any()

    def test_classification_end_to_end_training_step(self):
        """Test full training step with supervised classification loss."""
        model = DeepGeoCCA(n_chans=8, n_outputs=3, bimap_dims=(6, 4), drop_prob=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(4, 8, 200)
        labels = torch.tensor([0, 1, 2, 1])

        # Step 1: Forward + CrossEntropy
        optimizer.zero_grad()
        logits = model(x)
        class_loss = criterion(logits, labels)
        cca_loss = model.compute_cca_loss(x)
        total_loss = class_loss + 0.1 * cca_loss

        total_loss.backward()
        optimizer.step()

        assert torch.isfinite(class_loss)
        assert torch.isfinite(cca_loss)
        assert model.classifier.weight.grad is not None

    @pytest.mark.parametrize("bimap_dims", [(12,), (16, 8), (14, 10, 6)])
    def test_custom_bimap_dims(self, bimap_dims):
        """Test DeepGeoCCA with various depths and widths of BiMap layers."""
        model = DeepGeoCCA(n_chans=18, n_outputs=2, bimap_dims=bimap_dims)
        x = torch.randn(2, 18, 150)
        out = model(x)
        assert out.shape == (2, 2)
