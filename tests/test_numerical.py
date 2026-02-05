"""Tests for the unified numerical stability configuration."""

import pytest
import torch

from spd_learn.functional.numerical import (
    NumericalConfig,
    NumericalContext,
    check_spd_eigenvalues,
    get_epsilon,
    get_epsilon_tensor,
    get_loewner_threshold,
    is_half_precision,
    numerical_config,
    recommend_dtype_for_spd,
    safe_clamp_eigenvalues,
)


class TestGetEpsilon:
    """Tests for get_epsilon function."""

    def test_dtype_aware_scaling(self):
        """Test that epsilon scales with dtype precision."""
        eps_f32 = get_epsilon(torch.float32, "eigval_clamp")
        eps_f64 = get_epsilon(torch.float64, "eigval_clamp")

        # float64 should have smaller epsilon (more precise)
        assert eps_f64 < eps_f32

    def test_float16_larger_epsilon(self):
        """Test that float16 gets a larger epsilon."""
        eps_f16 = get_epsilon(torch.float16, "eigval_clamp")
        eps_f32 = get_epsilon(torch.float32, "eigval_clamp")

        # float16 should have larger epsilon (less precise)
        assert eps_f16 > eps_f32

    def test_different_threshold_names(self):
        """Test that different threshold names give different values."""
        eps_clamp = get_epsilon(torch.float32, "eigval_clamp")
        eps_log = get_epsilon(torch.float32, "eigval_log")
        eps_sqrt = get_epsilon(torch.float32, "eigval_sqrt")

        # These should be different due to different scales
        assert eps_clamp != eps_log or eps_clamp != eps_sqrt

    def test_absolute_thresholds(self):
        """Test that absolute thresholds don't scale with dtype."""
        # batchnorm_var is an absolute threshold
        eps_f32 = get_epsilon(torch.float32, "batchnorm_var")
        eps_f64 = get_epsilon(torch.float64, "batchnorm_var")

        # Should be the same (absolute value)
        assert eps_f32 == eps_f64

    def test_invalid_threshold_name(self):
        """Test that invalid threshold name raises error."""
        with pytest.raises(ValueError, match="Unknown threshold name"):
            get_epsilon(torch.float32, "invalid_name")

    def test_caching(self):
        """Test that caching works correctly."""
        config = NumericalConfig()

        # First call
        eps1 = get_epsilon(torch.float32, "eigval_clamp", config=config)
        # Second call should use cache
        eps2 = get_epsilon(torch.float32, "eigval_clamp", config=config)

        assert eps1 == eps2
        assert (torch.float32, "eigval_clamp") in config._threshold_cache


class TestGetEpsilonTensor:
    """Tests for get_epsilon_tensor function."""

    def test_returns_tensor(self):
        """Test that function returns a tensor."""
        eps = get_epsilon_tensor(torch.float32, "eigval_clamp")
        assert isinstance(eps, torch.Tensor)

    def test_correct_dtype(self):
        """Test that tensor has correct dtype."""
        eps_f32 = get_epsilon_tensor(torch.float32, "eigval_clamp")
        eps_f64 = get_epsilon_tensor(torch.float64, "eigval_clamp")

        assert eps_f32.dtype == torch.float32
        assert eps_f64.dtype == torch.float64

    def test_correct_device(self):
        """Test that tensor is on correct device."""
        eps = get_epsilon_tensor(torch.float32, "eigval_clamp", device="cpu")
        assert eps.device == torch.device("cpu")


class TestSafeClampEigenvalues:
    """Tests for safe_clamp_eigenvalues function."""

    def test_clamps_small_values(self):
        """Test that small eigenvalues are clamped."""
        eigvals = torch.tensor([1e-10, 1e-5, 1e-3, 1.0], dtype=torch.float32)
        clamped = safe_clamp_eigenvalues(eigvals, "eigval_log")

        threshold = get_epsilon(torch.float32, "eigval_log")
        assert (clamped >= threshold).all()

    def test_preserves_large_values(self):
        """Test that large eigenvalues are preserved."""
        eigvals = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float32)
        clamped = safe_clamp_eigenvalues(eigvals, "eigval_clamp")

        torch.testing.assert_close(eigvals, clamped)

    def test_return_mask(self):
        """Test that mask is returned when requested."""
        eigvals = torch.tensor([1e-10, 1.0], dtype=torch.float32)
        clamped, mask = safe_clamp_eigenvalues(eigvals, "eigval_log", return_mask=True)

        assert mask[0].item() is True  # First was clamped
        assert mask[1].item() is False  # Second was not


class TestCheckSpdEigenvalues:
    """Tests for check_spd_eigenvalues function."""

    def test_valid_eigenvalues(self):
        """Test that valid eigenvalues pass check."""
        eigvals = torch.tensor([0.1, 1.0, 10.0], dtype=torch.float32)
        is_valid, min_val, num_bad = check_spd_eigenvalues(eigvals, "eigval_clamp")

        assert is_valid is True
        assert num_bad == 0

    def test_invalid_eigenvalues(self):
        """Test that invalid eigenvalues fail check."""
        eigvals = torch.tensor([1e-10, 0.1, 1.0], dtype=torch.float32)
        is_valid, min_val, num_bad = check_spd_eigenvalues(eigvals, "eigval_clamp")

        assert is_valid is False
        assert num_bad >= 1
        assert min_val == pytest.approx(1e-10)

    def test_raise_on_failure(self):
        """Test that error is raised when requested."""
        eigvals = torch.tensor([1e-10, 1.0], dtype=torch.float32)

        with pytest.raises(ValueError, match="SPD check failed"):
            check_spd_eigenvalues(eigvals, "eigval_clamp", raise_on_failure=True)


class TestGetLoewnerThreshold:
    """Tests for get_loewner_threshold function."""

    def test_adaptive_scaling(self):
        """Test that threshold scales with eigenvalue magnitude."""
        small_eigvals = torch.tensor([0.1, 0.2], dtype=torch.float32)
        large_eigvals = torch.tensor([100.0, 200.0], dtype=torch.float32)

        thresh_small = get_loewner_threshold(small_eigvals)
        thresh_large = get_loewner_threshold(large_eigvals)

        # Larger eigenvalues should give larger threshold
        assert thresh_large > thresh_small

    def test_minimum_scale(self):
        """Test that there's a minimum scale of 1."""
        tiny_eigvals = torch.tensor([1e-10, 1e-9], dtype=torch.float32)
        thresh = get_loewner_threshold(tiny_eigvals)

        # Should be at least the base threshold (scale=1)
        base_thresh = get_epsilon(torch.float32, "loewner_equal")
        assert thresh >= base_thresh


class TestNumericalContext:
    """Tests for NumericalContext context manager."""

    def test_temporary_override(self):
        """Test that config is temporarily overridden."""
        original = numerical_config.eigval_clamp_scale

        with NumericalContext(eigval_clamp_scale=1e6):
            assert numerical_config.eigval_clamp_scale == 1e6

        assert numerical_config.eigval_clamp_scale == original

    def test_cache_cleared(self):
        """Test that cache is cleared on context changes."""
        # Populate cache
        get_epsilon(torch.float32, "eigval_clamp")
        assert len(numerical_config._threshold_cache) > 0

        with NumericalContext(eigval_clamp_scale=1e6):
            # Cache should be cleared
            assert len(numerical_config._threshold_cache) == 0

    def test_multiple_overrides(self):
        """Test that multiple parameters can be overridden."""
        orig_clamp = numerical_config.eigval_clamp_scale
        orig_log = numerical_config.eigval_log_scale

        with NumericalContext(eigval_clamp_scale=1e6, eigval_log_scale=1e3):
            assert numerical_config.eigval_clamp_scale == 1e6
            assert numerical_config.eigval_log_scale == 1e3

        assert numerical_config.eigval_clamp_scale == orig_clamp
        assert numerical_config.eigval_log_scale == orig_log

    def test_invalid_parameter(self):
        """Test that invalid parameter raises error."""
        with pytest.raises(ValueError, match="Unknown configuration parameter"):
            with NumericalContext(invalid_param=1.0):
                pass


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_half_precision(self):
        """Test is_half_precision function."""
        assert is_half_precision(torch.float16) is True
        assert is_half_precision(torch.bfloat16) is True
        assert is_half_precision(torch.float32) is False
        assert is_half_precision(torch.float64) is False

    def test_recommend_dtype_for_spd(self):
        """Test dtype recommendation."""
        # Well-conditioned matrices can use float32
        assert recommend_dtype_for_spd(1e3) == torch.float32

        # Ill-conditioned matrices need float64
        assert recommend_dtype_for_spd(1e10) == torch.float64

    def test_recommend_dtype_prefer_speed(self):
        """Test dtype recommendation with speed preference."""
        # With prefer_speed, moderate condition numbers use float32
        dtype = recommend_dtype_for_spd(1e5, prefer_speed=True)
        assert dtype == torch.float32

        # Without prefer_speed, same condition number might use float64
        dtype = recommend_dtype_for_spd(1e5, prefer_speed=False)
        assert dtype == torch.float64


class TestNumericalConfig:
    """Tests for NumericalConfig class."""

    def test_default_values(self):
        """Test that default values are reasonable."""
        config = NumericalConfig()

        assert config.eigval_clamp_scale > 0
        assert config.batchnorm_var_eps > 0
        assert config.warn_on_clamp is True
        assert config.strict_spd_check is False

    def test_clear_cache(self):
        """Test that clear_cache works."""
        config = NumericalConfig()

        # Populate cache
        get_epsilon(torch.float32, "eigval_clamp", config=config)
        assert len(config._threshold_cache) > 0

        config.clear_cache()
        assert len(config._threshold_cache) == 0

    def test_is_absolute(self):
        """Test is_absolute method."""
        config = NumericalConfig()

        assert config.is_absolute("batchnorm_var") is True
        assert config.is_absolute("dropout") is True
        assert config.is_absolute("eigval_clamp") is False
        assert config.is_absolute("eigval_log") is False

    def test_summary_returns_string(self):
        """Test that summary method returns a formatted string."""
        config = NumericalConfig()
        summary = config.summary(torch.float32)

        assert isinstance(summary, str)
        assert "Numerical Configuration Summary" in summary
        assert "torch.float32" in summary

    def test_summary_contains_all_thresholds(self):
        """Test that summary contains all threshold types."""
        config = NumericalConfig()
        summary = config.summary(torch.float32)

        # Check scaled thresholds
        for name in [
            "eigval_clamp",
            "eigval_log",
            "eigval_sqrt",
            "eigval_inv_sqrt",
            "eigval_power",
            "loewner_equal",
            "stiefel_init",
            "division_safe",
        ]:
            assert name in summary

        # Check absolute thresholds
        for name in ["batchnorm_var", "dropout", "trace_norm"]:
            assert name in summary

        # Check behavior flags
        assert "warn_on_clamp" in summary
        assert "strict_spd_check" in summary

    def test_summary_dtype_aware(self):
        """Test that summary shows different values for different dtypes."""
        config = NumericalConfig()
        summary_f32 = config.summary(torch.float32)
        summary_f64 = config.summary(torch.float64)

        # Different dtypes should produce different summaries
        assert summary_f32 != summary_f64
        assert "torch.float32" in summary_f32
        assert "torch.float64" in summary_f64


class TestIntegrationWithFunctional:
    """Integration tests with functional operations."""

    def test_matrix_log_uses_config(self):
        """Test that matrix_log uses the unified config."""
        from spd_learn.functional import matrix_log

        # Create a well-conditioned SPD matrix
        A = torch.randn(5, 5, dtype=torch.float64)
        X = A @ A.T + torch.eye(5, dtype=torch.float64)

        # Should not raise with default config
        result = matrix_log.apply(X)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_matrix_power_uses_config(self):
        """Test that matrix_power uses the unified config."""
        from spd_learn.functional import matrix_power

        # Create a well-conditioned SPD matrix
        A = torch.randn(5, 5, dtype=torch.float64)
        X = A @ A.T + torch.eye(5, dtype=torch.float64)

        # Fractional power should work with clamping
        result = matrix_power.apply(X, 0.5)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_config_affects_operations(self):
        """Test that changing config affects operations."""
        from spd_learn.functional import matrix_log

        # Create a matrix with explicitly controlled small eigenvalues
        # that will definitely be affected by clamping
        U = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))[0]
        # Eigenvalues: one very small (1e-20), others reasonable
        s = torch.tensor([1e-20, 0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
        X = U @ torch.diag(s) @ U.T

        # With default config (eigval_log_scale=1e2, so threshold ~ 2e-14 for float64)
        result1 = matrix_log.apply(X.clone())

        # With very conservative config (threshold ~ 2e-6 for float64)
        with NumericalContext(eigval_log_scale=1e10):
            result2 = matrix_log.apply(X.clone())

        # Results should be different due to different clamping of the 1e-20 eigenvalue
        assert not torch.allclose(result1, result2)


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_very_small_eigenvalues(self):
        """Test handling of very small eigenvalues."""
        eigvals = torch.tensor([1e-15, 1e-10, 1e-5], dtype=torch.float64)
        clamped = safe_clamp_eigenvalues(eigvals, "eigval_log")

        # All should be clamped to at least the threshold
        threshold = get_epsilon(torch.float64, "eigval_log")
        assert (clamped >= threshold).all()

    def test_negative_eigenvalues(self):
        """Test handling of negative eigenvalues (shouldn't happen for SPD)."""
        eigvals = torch.tensor([-0.1, 0.1, 1.0], dtype=torch.float32)
        clamped = safe_clamp_eigenvalues(eigvals, "eigval_clamp")

        # Negative should be clamped to threshold
        threshold = get_epsilon(torch.float32, "eigval_clamp")
        assert clamped[0] >= threshold

    def test_mixed_precision_consistency(self):
        """Test that operations are consistent across precisions."""
        # For well-conditioned matrices, results should be similar
        A = torch.randn(3, 3)
        X_f32 = (A @ A.T + torch.eye(3)).float()
        X_f64 = X_f32.double()

        from spd_learn.functional import matrix_log

        result_f32 = matrix_log.apply(X_f32)
        result_f64 = matrix_log.apply(X_f64)

        # Results should be close (but not identical due to precision)
        torch.testing.assert_close(
            result_f32.double(), result_f64, rtol=1e-5, atol=1e-5
        )
