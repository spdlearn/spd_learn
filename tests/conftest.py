"""Pytest configuration and shared fixtures for spd_learn tests.

This module provides common fixtures and configuration for testing SPD operations,
including random SPD matrix generation, device handling, and numerical tolerances.
"""

import pytest
import torch


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and command-line options."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# =============================================================================
# Device Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def device():
    """Return the default device for testing (CPU)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cuda_device():
    """Return CUDA device if available, otherwise skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# =============================================================================
# Random Seed Fixtures
# =============================================================================


@pytest.fixture(autouse=False)
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield


@pytest.fixture
def rng():
    """Return a seeded random number generator."""
    return torch.Generator().manual_seed(42)


# =============================================================================
# SPD Matrix Generation Fixtures
# =============================================================================


@pytest.fixture
def random_spd_matrix():
    """Factory fixture that generates random SPD matrices.

    Returns a function that can be called with custom parameters.

    Example
    -------
    >>> def test_something(random_spd_matrix):
    ...     X = random_spd_matrix(batch_size=4, n_channels=22)
    ...     assert X.shape == (4, 22, 22)
    """

    def _make_spd(
        batch_size: int = 1,
        n_channels: int = 10,
        dtype: torch.dtype = torch.float64,
        eps: float = 1e-3,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Generate a random SPD matrix.

        Parameters
        ----------
        batch_size : int
            Number of matrices in the batch.
        n_channels : int
            Size of the square matrix (n_channels x n_channels).
        dtype : torch.dtype
            Data type of the tensor.
        eps : float
            Minimum eigenvalue to ensure positive definiteness.
        device : str
            Device to place the tensor on.

        Returns
        -------
        torch.Tensor
            Random SPD matrix of shape (batch_size, n_channels, n_channels).
        """
        A = torch.randn(batch_size, n_channels, n_channels, dtype=dtype, device=device)
        # X = A @ A^T + eps * I to ensure positive definiteness
        X = A @ A.transpose(-1, -2) + eps * torch.eye(
            n_channels, dtype=dtype, device=device
        )
        return X

    return _make_spd


@pytest.fixture
def random_spd_batch(random_spd_matrix):
    """Generate a batch of random SPD matrices with default parameters."""
    return random_spd_matrix(batch_size=4, n_channels=10)


@pytest.fixture
def small_spd_matrix(random_spd_matrix):
    """Generate a small SPD matrix for quick tests."""
    return random_spd_matrix(batch_size=1, n_channels=5)


@pytest.fixture
def eeg_like_data():
    """Factory fixture that generates EEG-like time series data.

    Returns a function that can be called with custom parameters.
    """

    def _make_eeg(
        batch_size: int = 4,
        n_channels: int = 22,
        n_times: int = 500,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Generate random EEG-like data.

        Parameters
        ----------
        batch_size : int
            Number of trials.
        n_channels : int
            Number of EEG channels.
        n_times : int
            Number of time samples.
        dtype : torch.dtype
            Data type of the tensor.
        device : str
            Device to place the tensor on.

        Returns
        -------
        torch.Tensor
            Random EEG data of shape (batch_size, n_channels, n_times).
        """
        return torch.randn(batch_size, n_channels, n_times, dtype=dtype, device=device)

    return _make_eeg


@pytest.fixture
def eeg_batch(eeg_like_data):
    """Generate a batch of EEG-like data with default parameters."""
    return eeg_like_data(batch_size=4, n_channels=22, n_times=500)


# =============================================================================
# Numerical Tolerance Fixtures
# =============================================================================


@pytest.fixture
def tolerance():
    """Return default numerical tolerances for comparisons."""
    return {"atol": 1e-5, "rtol": 1e-4}


@pytest.fixture
def strict_tolerance():
    """Return strict numerical tolerances for precise comparisons."""
    return {"atol": 1e-7, "rtol": 1e-6}


@pytest.fixture
def relaxed_tolerance():
    """Return relaxed numerical tolerances for approximate comparisons."""
    return {"atol": 1e-3, "rtol": 1e-2}


# =============================================================================
# Model Configuration Fixtures
# =============================================================================


@pytest.fixture
def model_config():
    """Return default model configuration for testing."""
    return {
        "n_chans": 22,
        "n_outputs": 4,
        "n_times": 500,
    }


@pytest.fixture
def small_model_config():
    """Return small model configuration for quick tests."""
    return {
        "n_chans": 8,
        "n_outputs": 2,
        "n_times": 100,
    }


# =============================================================================
# Gradient Checking Fixtures
# =============================================================================


@pytest.fixture
def gradcheck_config():
    """Return configuration for torch.autograd.gradcheck."""
    return {
        "eps": 1e-6,
        "atol": 1e-4,
        "rtol": 1e-3,
        "raise_exception": True,
    }
