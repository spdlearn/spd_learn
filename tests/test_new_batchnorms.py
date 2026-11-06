# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for new Riemannian batch normalization layers:
- SPDBatchNormHarmonicArithmetic (ARMAGNAC, Gallet et al. 2026)
- SPDBatchNormBW (GBWBN, Zhou et al. CVPR 2025)
- SPDBatchNormCholesky (CBN, Zhou et al. TNNLS 2025)
"""

import pytest
import torch

from spd_learn.modules import (
    SPDBatchNormBW,
    SPDBatchNormCholesky,
    SPDBatchNormHarmonicArithmetic,
)


def _make_spd_batch(shape, dtype=torch.float64, seed=42):
    """Helper to generate a batch of random SPD matrices."""
    torch.manual_seed(seed)
    n = shape[-1]
    X = torch.randn(*shape, dtype=dtype)
    spd = X @ X.mT + 0.2 * torch.eye(n, dtype=dtype)
    return spd


# ===========================================================================
# Tests for SPDBatchNormHarmonicArithmetic (ARMAGNAC)
# ===========================================================================


class TestSPDBatchNormHarmonicArithmetic:
    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            SPDBatchNormHarmonicArithmetic(num_features=3, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            SPDBatchNormHarmonicArithmetic(num_features=3, alpha=1.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            SPDBatchNormHarmonicArithmetic(num_features=3, alpha=-0.5)

    def test_instantiation_and_reset(self):
        n = 4
        bn = SPDBatchNormHarmonicArithmetic(
            num_features=n,
            alpha=0.7,
            learnable_alpha=True,
            rebias=True,
        )
        assert isinstance(bn.alpha_logit, torch.nn.Parameter)
        assert bn.running_mean.shape == (1, n, n)
        assert torch.allclose(bn.running_mean[0], torch.eye(n))
        assert bn.bias is not None

        # Modify buffers and parameters, then verify reset
        bn.running_mean.fill_(2.0)
        bn.alpha_logit.data.fill_(10.0)
        bn.reset_parameters()
        assert torch.allclose(bn.running_mean[0], torch.eye(n))
        expected_logit = torch.tensor(0.7 / 0.3).log()
        assert torch.allclose(bn.alpha_logit, expected_logit, atol=1e-5)

    @pytest.mark.parametrize("shape", [(6, 3, 3), (2, 4, 3, 3)])
    def test_forward_shape_and_spd(self, shape):
        n = shape[-1]
        X = _make_spd_batch(shape)
        bn = SPDBatchNormHarmonicArithmetic(
            num_features=n, alpha=0.5, dtype=torch.float64
        )
        out = bn(X)

        assert out.shape == X.shape
        # Check symmetry
        assert torch.allclose(out, out.mT, atol=1e-6)
        # Check positive definiteness
        eigvals = torch.linalg.eigvalsh(out)
        assert (eigvals > 0).all()

    def test_training_vs_eval_mode(self):
        n = 3
        X = _make_spd_batch((8, n, n))
        bn = SPDBatchNormHarmonicArithmetic(
            num_features=n, momentum=0.2, dtype=torch.float64
        )

        # In training mode, running_mean should update
        initial_mean = bn.running_mean.clone()
        bn.train()
        _ = bn(X)
        assert not torch.allclose(bn.running_mean, initial_mean)

        # In eval mode, running_mean should remain unchanged
        eval_mean = bn.running_mean.clone()
        bn.eval()
        _ = bn(X)
        assert torch.allclose(bn.running_mean, eval_mean)

    def test_gradient_flow_learnable_alpha(self):
        n = 3
        X = _make_spd_batch((4, n, n))
        X.requires_grad_(True)
        bn = SPDBatchNormHarmonicArithmetic(
            num_features=n,
            alpha=0.5,
            learnable_alpha=True,
            rebias=True,
            dtype=torch.float64,
        )

        out = bn(X)
        loss = out.sum()
        loss.backward()

        assert X.grad is not None
        assert not torch.isnan(X.grad).any()
        assert bn.alpha_logit.grad is not None
        assert not torch.isnan(bn.alpha_logit.grad)

    def test_fixed_alpha(self):
        n = 3
        bn = SPDBatchNormHarmonicArithmetic(
            num_features=n,
            alpha=0.4,
            learnable_alpha=False,
            rebias=False,
            dtype=torch.float64,
        )
        assert not isinstance(bn.alpha_logit, torch.nn.Parameter)
        assert bn.bias is None

        X = _make_spd_batch((4, n, n))
        X.requires_grad_(True)
        out = bn(X)
        loss = out.sum()
        loss.backward()

        assert X.grad is not None
        assert not torch.isnan(X.grad).any()


# ===========================================================================
# Tests for SPDBatchNormBW (GBWBN)
# ===========================================================================


class TestSPDBatchNormBW:
    def test_instantiation_and_reset(self):
        n = 4
        bn = SPDBatchNormBW(num_features=n, theta=1.0, n_iter=3, rebias=True)
        assert bn.running_mean.shape == (1, n, n)
        assert torch.allclose(bn.running_mean[0], torch.eye(n))
        assert bn.bias is not None

        bn.running_mean.fill_(3.0)
        bn.reset_parameters()
        assert torch.allclose(bn.running_mean[0], torch.eye(n))

    @pytest.mark.parametrize("n_iter", [1, 3])
    @pytest.mark.parametrize("theta", [1.0, 0.5])
    @pytest.mark.parametrize("shape", [(6, 3, 3), (2, 3, 3, 3)])
    def test_forward_shape_and_spd(self, n_iter, theta, shape):
        n = shape[-1]
        X = _make_spd_batch(shape)
        bn = SPDBatchNormBW(
            num_features=n,
            theta=theta,
            n_iter=n_iter,
            dtype=torch.float64,
        )
        out = bn(X)

        assert out.shape == X.shape
        assert torch.allclose(out, out.mT, atol=1e-6)
        eigvals = torch.linalg.eigvalsh(out)
        assert (eigvals > 0).all()

    def test_training_vs_eval_mode(self):
        n = 3
        X = _make_spd_batch((6, n, n))
        bn = SPDBatchNormBW(num_features=n, momentum=0.2, n_iter=1, dtype=torch.float64)

        initial_mean = bn.running_mean.clone()
        bn.train()
        _ = bn(X)
        assert not torch.allclose(bn.running_mean, initial_mean)

        eval_mean = bn.running_mean.clone()
        bn.eval()
        _ = bn(X)
        assert torch.allclose(bn.running_mean, eval_mean)

    @pytest.mark.parametrize("n_iter", [1, 3])
    @pytest.mark.parametrize("theta", [1.0, 0.7])
    def test_gradient_flow(self, n_iter, theta):
        n = 3
        X = _make_spd_batch((4, n, n))
        X.requires_grad_(True)
        bn = SPDBatchNormBW(
            num_features=n,
            theta=theta,
            n_iter=n_iter,
            rebias=True,
            dtype=torch.float64,
        )

        out = bn(X)
        loss = out.sum()
        loss.backward()

        assert X.grad is not None
        assert not torch.isnan(X.grad).any()


# ===========================================================================
# Tests for SPDBatchNormCholesky (CBN)
# ===========================================================================


class TestSPDBatchNormCholesky:
    def test_instantiation_and_reset(self):
        n = 4
        bn = SPDBatchNormCholesky(num_features=n, rebias=True)
        assert bn.running_mean.shape == (1, n, n)
        assert torch.allclose(bn.running_mean, torch.zeros(1, n, n))
        assert bn.running_var.shape == (1, n, n)
        assert torch.allclose(bn.running_var, torch.ones(1, n, n))
        assert bn.weight is not None
        assert bn.bias is not None

        bn.running_mean.fill_(2.0)
        bn.running_var.fill_(5.0)
        bn.weight.data.fill_(0.5)
        bn.bias.data.fill_(1.0)
        bn.reset_parameters()

        assert torch.allclose(bn.running_mean, torch.zeros(1, n, n))
        assert torch.allclose(bn.running_var, torch.ones(1, n, n))
        assert torch.allclose(bn.weight, torch.ones(1, n, n))
        assert torch.allclose(bn.bias, torch.zeros(1, n, n))

    @pytest.mark.parametrize("rebias", [True, False])
    @pytest.mark.parametrize("shape", [(6, 3, 3), (2, 3, 3, 3)])
    def test_forward_shape_and_spd(self, rebias, shape):
        n = shape[-1]
        X = _make_spd_batch(shape)
        bn = SPDBatchNormCholesky(num_features=n, rebias=rebias, dtype=torch.float64)
        out = bn(X)

        assert out.shape == X.shape
        assert torch.allclose(out, out.mT, atol=1e-6)
        eigvals = torch.linalg.eigvalsh(out)
        assert (eigvals > 0).all()

    def test_training_vs_eval_mode(self):
        n = 3
        X = _make_spd_batch((6, n, n))
        bn = SPDBatchNormCholesky(num_features=n, momentum=0.2, dtype=torch.float64)

        initial_mean = bn.running_mean.clone()
        initial_var = bn.running_var.clone()
        bn.train()
        _ = bn(X)
        assert not torch.allclose(bn.running_mean, initial_mean)
        assert not torch.allclose(bn.running_var, initial_var)

        eval_mean = bn.running_mean.clone()
        eval_var = bn.running_var.clone()
        bn.eval()
        _ = bn(X)
        assert torch.allclose(bn.running_mean, eval_mean)
        assert torch.allclose(bn.running_var, eval_var)

    def test_gradient_flow(self):
        n = 3
        X = _make_spd_batch((4, n, n))
        X.requires_grad_(True)
        bn = SPDBatchNormCholesky(num_features=n, rebias=True, dtype=torch.float64)

        out = bn(X)
        loss = out.sum()
        loss.backward()

        assert X.grad is not None
        assert not torch.isnan(X.grad).any()
        assert bn.weight.grad is not None
        assert not torch.isnan(bn.weight.grad).any()
        assert bn.bias.grad is not None
        assert not torch.isnan(bn.bias.grad).any()
