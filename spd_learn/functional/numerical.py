# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Unified numerical stability configuration for SPD operations.

This module provides a centralized configuration for numerical stability
thresholds used throughout the spd_learn library. It ensures consistent
behavior across different operations and supports dtype-aware thresholds.

Examples
--------
>>> from spd_learn.functional.numerical import get_epsilon, numerical_config
>>> # Get dtype-aware epsilon for eigenvalue clamping
>>> eps = get_epsilon(torch.float32, "eigval_clamp")
>>> # Modify global configuration
>>> numerical_config.eigval_clamp_scale = 1e3  # More conservative clamping
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union

import torch


# Type alias for threshold names
ThresholdName = Literal[
    "eigval_clamp",
    "eigval_log",
    "eigval_sqrt",
    "eigval_inv_sqrt",
    "eigval_power",
    "loewner_equal",
    "batchnorm_var",
    "dropout",
    "trace_norm",
    "stiefel_init",
    "division_safe",
]


@dataclass
class NumericalConfig:
    r"""Global configuration for numerical stability thresholds.

    This class provides centralized control over numerical stability parameters
    used throughout the spd_learn library. All thresholds are specified as
    multipliers of the machine epsilon for the given dtype.

    The actual threshold for a given dtype is computed as::

        threshold = scale * torch.finfo(dtype).eps

    For example, with ``eigval_clamp_scale=1e4`` and ``dtype=torch.float32``::

        threshold = 1e4 * 1.19e-7 ≈ 1.19e-3

    Parameters
    ----------
    eigval_clamp_scale : float
        Scale factor for general eigenvalue clamping (ReEig layer).
        Default: 1e4 (yields ~1e-3 for float32).
    eigval_log_scale : float
        Scale factor for eigenvalue clamping before log operation.
        Default: 1e2 (yields ~1e-5 for float32).
    eigval_sqrt_scale : float
        Scale factor for eigenvalue clamping before sqrt operation.
        Default: 1e2 (yields ~1e-5 for float32).
    eigval_inv_sqrt_scale : float
        Scale factor for eigenvalue clamping before inverse sqrt.
        Default: 1e3 (yields ~1e-4 for float32).
    eigval_power_scale : float
        Scale factor for eigenvalue clamping before power operation.
        Default: 1e3 (yields ~1e-4 for float32).
    loewner_equal_scale : float
        Scale factor for detecting equal eigenvalues in Loewner matrix.
        Default: 1e2 (yields ~1e-5 for float32).
    batchnorm_var_eps : float
        Absolute epsilon for batch normalization scalar dispersion.
        This is a scalar value (mean squared Frobenius norm in tangent space),
        not a variance matrix. Default: 1e-5.
    dropout_eps : float
        Absolute epsilon for dropout diagonal entries.
        Default: 1e-5.
    trace_norm_eps : float
        Absolute epsilon for trace normalization.
        Default: 1e-6.
    stiefel_init_scale : float
        Scale factor for Stiefel manifold initialization.
        Default: 1e3 (yields ~1e-4 for float32).
    division_safe_scale : float
        Scale factor for safe division operations.
        Default: 1e5 (yields ~1e-2 for float32).
    warn_on_clamp : bool
        Whether to emit warnings when eigenvalues are clamped.
        Default: True.
    strict_spd_check : bool
        Whether to perform strict SPD checks (slower but safer).
        Default: False.

    Notes
    -----
    The default scale factors are chosen to balance numerical stability with
    accuracy :cite:p:`higham2002accuracy`. More conservative (larger) values
    provide better stability but may reduce precision. Less conservative
    (smaller) values preserve more information but risk numerical issues.

    For mixed-precision training (fp16), consider using larger scale factors
    as the machine epsilon for fp16 is much larger (~9.77e-4).
    """

    # Scale factors (multiplied by machine epsilon)
    eigval_clamp_scale: float = 1e4
    eigval_log_scale: float = 1e2
    eigval_sqrt_scale: float = 1e2
    eigval_inv_sqrt_scale: float = 1e3
    eigval_power_scale: float = 1e3
    loewner_equal_scale: float = 1e2
    stiefel_init_scale: float = 1e3
    division_safe_scale: float = 1e5

    # Absolute epsilons (not scaled by machine epsilon)
    batchnorm_var_eps: float = 1e-5
    dropout_eps: float = 1e-5
    trace_norm_eps: float = 1e-6

    # Behavior flags
    warn_on_clamp: bool = True
    strict_spd_check: bool = False

    # Cache for computed thresholds per dtype
    _threshold_cache: Dict[tuple, float] = field(default_factory=dict, repr=False)

    def clear_cache(self) -> None:
        """Clear the threshold cache after configuration changes."""
        self._threshold_cache.clear()

    def summary(self, dtype: torch.dtype = torch.float32) -> str:
        """Return formatted string showing all thresholds for a given dtype.

        Parameters
        ----------
        dtype : torch.dtype, default=torch.float32
            The dtype to compute thresholds for.

        Returns
        -------
        str
            Formatted summary of all threshold values.

        Examples
        --------
        >>> from spd_learn.functional.numerical import numerical_config
        >>> print(numerical_config.summary(torch.float32))
        Numerical Configuration Summary (dtype=torch.float32)
        ==================================================
        ...
        """
        # Import get_epsilon locally to avoid circular import issues
        # (get_epsilon is defined later in this module)
        lines = [f"Numerical Configuration Summary (dtype={dtype})"]
        lines.append("=" * 50)

        # Machine epsilon info
        machine_eps = torch.finfo(dtype).eps
        lines.append(f"\nMachine epsilon: {machine_eps:.2e}")

        # Scaled thresholds
        scaled_names: list[ThresholdName] = [
            "eigval_clamp",
            "eigval_log",
            "eigval_sqrt",
            "eigval_inv_sqrt",
            "eigval_power",
            "loewner_equal",
            "stiefel_init",
            "division_safe",
        ]
        lines.append("\nScaled thresholds (scale × machine_eps):")
        for name in scaled_names:
            scale = self.get_scale(name)
            eps_value = scale * machine_eps
            lines.append(f"  {name:20s}: {eps_value:.2e} (scale={scale:.0e})")

        # Absolute thresholds
        lines.append("\nAbsolute thresholds (dtype-independent):")
        absolute_names: list[ThresholdName] = [
            "batchnorm_var",
            "dropout",
            "trace_norm",
        ]
        for name in absolute_names:
            eps_value = self.get_scale(name)
            lines.append(f"  {name:20s}: {eps_value:.2e}")

        # Behavior flags
        lines.append("\nBehavior flags:")
        lines.append(f"  warn_on_clamp:      {self.warn_on_clamp}")
        lines.append(f"  strict_spd_check:   {self.strict_spd_check}")

        return "\n".join(lines)

    def get_scale(self, name: ThresholdName) -> float:
        """Get the scale factor for a given threshold name.

        Parameters
        ----------
        name : ThresholdName
            The name of the threshold.

        Returns
        -------
        float
            The scale factor for the threshold.
        """
        scale_map = {
            "eigval_clamp": self.eigval_clamp_scale,
            "eigval_log": self.eigval_log_scale,
            "eigval_sqrt": self.eigval_sqrt_scale,
            "eigval_inv_sqrt": self.eigval_inv_sqrt_scale,
            "eigval_power": self.eigval_power_scale,
            "loewner_equal": self.loewner_equal_scale,
            "stiefel_init": self.stiefel_init_scale,
            "division_safe": self.division_safe_scale,
            # Absolute epsilons return themselves (will be handled specially)
            "batchnorm_var": self.batchnorm_var_eps,
            "dropout": self.dropout_eps,
            "trace_norm": self.trace_norm_eps,
        }
        if name not in scale_map:
            raise ValueError(
                f"Unknown threshold name: '{name}'. "
                f"Valid names are: {list(scale_map.keys())}"
            )
        return scale_map[name]

    def is_absolute(self, name: ThresholdName) -> bool:
        """Check if a threshold uses absolute values (not scaled by eps).

        Parameters
        ----------
        name : ThresholdName
            The name of the threshold.

        Returns
        -------
        bool
            True if the threshold is absolute, False if scaled.
        """
        return name in ("batchnorm_var", "dropout", "trace_norm")


# Global configuration instance
numerical_config = NumericalConfig()


def get_epsilon(
    dtype: torch.dtype,
    name: ThresholdName = "eigval_clamp",
    *,
    config: Optional[NumericalConfig] = None,
) -> float:
    """Get a dtype-aware epsilon value for numerical stability.

    This function returns an appropriate epsilon value based on the data type
    and the intended use case. It scales the machine epsilon by a factor that
    ensures numerical stability for the specific operation.

    Parameters
    ----------
    dtype : torch.dtype
        The PyTorch dtype to compute epsilon for.
    name : ThresholdName, default="eigval_clamp"
        The type of threshold to compute. Options are:

        - ``"eigval_clamp"``: General eigenvalue clamping (ReEig layer)
        - ``"eigval_log"``: Eigenvalue clamping before log operation
        - ``"eigval_sqrt"``: Eigenvalue clamping before sqrt operation
        - ``"eigval_inv_sqrt"``: Eigenvalue clamping before inverse sqrt
        - ``"eigval_power"``: Eigenvalue clamping before power operation
        - ``"loewner_equal"``: Detection of equal eigenvalues in Loewner matrix
        - ``"batchnorm_var"``: Batch normalization variance epsilon
        - ``"dropout"``: Dropout diagonal epsilon
        - ``"trace_norm"``: Trace normalization epsilon
        - ``"stiefel_init"``: Stiefel manifold initialization
        - ``"division_safe"``: Safe division operations

    config : NumericalConfig, optional
        Configuration to use. If None, uses the global ``numerical_config``.

    Returns
    -------
    float
        The computed epsilon value.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.numerical import get_epsilon
    >>> # Get epsilon for float32 eigenvalue clamping
    >>> eps32 = get_epsilon(torch.float32, "eigval_clamp")
    >>> print(f"float32 eigval_clamp: {eps32:.2e}")
    float32 eigval_clamp: 1.19e-03
    >>> # Get epsilon for float64 (more precise)
    >>> eps64 = get_epsilon(torch.float64, "eigval_clamp")
    >>> print(f"float64 eigval_clamp: {eps64:.2e}")
    float64 eigval_clamp: 2.22e-12
    >>> # float16 needs larger epsilon
    >>> eps16 = get_epsilon(torch.float16, "eigval_clamp")
    >>> print(f"float16 eigval_clamp: {eps16:.2e}")
    float16 eigval_clamp: 9.77e+00

    See Also
    --------
    get_epsilon_tensor : Returns epsilon as a tensor on the correct device.
    numerical_config : Global configuration for threshold scales.
    """
    if config is None:
        config = numerical_config

    # Check cache first
    cache_key = (dtype, name)
    if cache_key in config._threshold_cache:
        return config._threshold_cache[cache_key]

    # Compute threshold
    if config.is_absolute(name):
        # Absolute thresholds don't scale with dtype
        threshold = config.get_scale(name)
    else:
        # Scaled thresholds multiply machine epsilon by scale factor
        machine_eps = torch.finfo(dtype).eps
        scale = config.get_scale(name)
        threshold = scale * machine_eps

    # Cache and return
    config._threshold_cache[cache_key] = threshold
    return threshold


def get_epsilon_tensor(
    dtype: torch.dtype,
    name: ThresholdName = "eigval_clamp",
    *,
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[NumericalConfig] = None,
) -> torch.Tensor:
    """Get a dtype-aware epsilon value as a tensor.

    Similar to :func:`get_epsilon`, but returns a tensor on the specified
    device. This is useful when the epsilon needs to be used in tensor
    operations that require matching devices.

    Parameters
    ----------
    dtype : torch.dtype
        The PyTorch dtype to compute epsilon for.
    name : ThresholdName, default="eigval_clamp"
        The type of threshold to compute.
    device : str or torch.device, optional
        The device to place the tensor on. If None, uses CPU.
    config : NumericalConfig, optional
        Configuration to use. If None, uses the global ``numerical_config``.

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the epsilon value.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.numerical import get_epsilon_tensor
    >>> eps = get_epsilon_tensor(torch.float32, "eigval_clamp", device="cpu")
    >>> print(eps)
    tensor(0.0012)
    """
    eps_value = get_epsilon(dtype, name, config=config)
    return torch.tensor(eps_value, dtype=dtype, device=device)


def safe_clamp_eigenvalues(
    eigenvalues: torch.Tensor,
    name: ThresholdName = "eigval_clamp",
    *,
    config: Optional[NumericalConfig] = None,
    return_mask: bool = False,
) -> Union[torch.Tensor, tuple]:
    """Safely clamp eigenvalues with dtype-aware threshold.

    This function clamps eigenvalues to ensure they are positive and
    numerically stable. It uses a dtype-aware threshold to balance
    stability and precision.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        The eigenvalues to clamp.
    name : ThresholdName, default="eigval_clamp"
        The type of threshold to use.
    config : NumericalConfig, optional
        Configuration to use. If None, uses the global ``numerical_config``.
    return_mask : bool, default=False
        If True, also return a boolean mask indicating which eigenvalues
        were clamped.

    Returns
    -------
    torch.Tensor or tuple
        The clamped eigenvalues. If ``return_mask=True``, returns a tuple
        of (clamped_eigenvalues, clamped_mask).

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.numerical import safe_clamp_eigenvalues
    >>> eigvals = torch.tensor([1e-10, 1e-5, 1e-3, 1.0])
    >>> clamped = safe_clamp_eigenvalues(eigvals, "eigval_log")
    >>> print(clamped)
    tensor([1.1921e-05, 1.1921e-05, 1.0000e-03, 1.0000e+00])
    """
    if config is None:
        config = numerical_config

    threshold = get_epsilon(eigenvalues.dtype, name, config=config)
    clamped = eigenvalues.clamp(min=threshold)

    if return_mask:
        mask = eigenvalues < threshold
        return clamped, mask
    return clamped


def check_spd_eigenvalues(
    eigenvalues: torch.Tensor,
    name: ThresholdName = "eigval_clamp",
    *,
    config: Optional[NumericalConfig] = None,
    raise_on_failure: bool = False,
) -> tuple:
    """Check if eigenvalues satisfy SPD requirements.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        The eigenvalues to check.
    name : ThresholdName, default="eigval_clamp"
        The threshold to use for the positivity check.
    config : NumericalConfig, optional
        Configuration to use. If None, uses the global ``numerical_config``.
    raise_on_failure : bool, default=False
        If True, raise an error when eigenvalues fail the check.

    Returns
    -------
    tuple
        A tuple of (is_valid, min_eigenvalue, num_below_threshold).

    Raises
    ------
    ValueError
        If ``raise_on_failure=True`` and eigenvalues are not valid.

    Examples
    --------
    >>> import torch
    >>> from spd_learn.functional.numerical import check_spd_eigenvalues
    >>> eigvals = torch.tensor([1e-10, 0.1, 1.0])
    >>> is_valid, min_val, num_bad = check_spd_eigenvalues(eigvals)
    >>> print(f"Valid: {is_valid}, Min: {min_val:.2e}, Bad count: {num_bad}")
    Valid: False, Min: 1.00e-10, Bad count: 1
    """
    if config is None:
        config = numerical_config

    threshold = get_epsilon(eigenvalues.dtype, name, config=config)
    min_eigenvalue = eigenvalues.min().item()
    num_below = (eigenvalues < threshold).sum().item()
    is_valid = num_below == 0

    if raise_on_failure and not is_valid:
        raise ValueError(
            f"SPD check failed: {num_below} eigenvalues below threshold "
            f"{threshold:.2e}. Minimum eigenvalue: {min_eigenvalue:.2e}"
        )

    return is_valid, min_eigenvalue, num_below


def get_loewner_threshold(
    eigenvalues: torch.Tensor,
    *,
    config: Optional[NumericalConfig] = None,
) -> float:
    """Get threshold for detecting equal eigenvalues in Loewner matrix.

    The Loewner matrix computation requires special handling when eigenvalues
    are equal or nearly equal. This function returns an appropriate threshold
    for detecting such cases.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        The eigenvalues (used to determine dtype).
    config : NumericalConfig, optional
        Configuration to use. If None, uses the global ``numerical_config``.

    Returns
    -------
    float
        The threshold for eigenvalue equality detection.

    Notes
    -----
    The threshold is computed as::

        threshold = scale * max(1, |eigenvalues|.max()) * eps

    This adaptive threshold accounts for the magnitude of eigenvalues,
    providing better numerical stability for matrices with large eigenvalues.
    """
    if config is None:
        config = numerical_config

    base_threshold = get_epsilon(eigenvalues.dtype, "loewner_equal", config=config)

    # Adaptive scaling based on eigenvalue magnitude
    max_eigval = eigenvalues.abs().max().item()
    scale = max(1.0, max_eigval)

    return base_threshold * scale


class NumericalContext:
    """Context manager for temporarily modifying numerical configuration.

    This context manager allows temporary modification of the global
    numerical configuration. The original configuration is restored
    when exiting the context.

    Parameters
    ----------
    **kwargs
        Configuration parameters to temporarily override.

    Examples
    --------
    >>> from spd_learn.functional.numerical import (
    ...     numerical_config, NumericalContext, get_epsilon
    ... )
    >>> import torch
    >>> # Default epsilon
    >>> print(f"Default: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")
    Default: 1.19e-03
    >>> # Temporarily use more conservative threshold
    >>> with NumericalContext(eigval_clamp_scale=1e6):
    ...     print(f"Conservative: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")
    Conservative: 1.19e-01
    >>> # Back to default
    >>> print(f"Restored: {get_epsilon(torch.float32, 'eigval_clamp'):.2e}")
    Restored: 1.19e-03
    """

    def __init__(self, **kwargs):
        self.overrides = kwargs
        self.saved_values = {}

    def __enter__(self):
        # Save current values and apply overrides
        for key, value in self.overrides.items():
            if not hasattr(numerical_config, key):
                raise ValueError(f"Unknown configuration parameter: {key}")
            self.saved_values[key] = getattr(numerical_config, key)
            setattr(numerical_config, key, value)
        numerical_config.clear_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values
        for key, value in self.saved_values.items():
            setattr(numerical_config, key, value)
        numerical_config.clear_cache()
        return False


# Convenience function for common dtype checks
def is_half_precision(dtype: torch.dtype) -> bool:
    """Check if dtype is half precision (float16 or bfloat16).

    Parameters
    ----------
    dtype : torch.dtype
        The dtype to check.

    Returns
    -------
    bool
        True if the dtype is float16 or bfloat16.
    """
    return dtype in (torch.float16, torch.bfloat16)


def recommend_dtype_for_spd(
    condition_number: float,
    *,
    prefer_speed: bool = False,
) -> torch.dtype:
    """Recommend a dtype based on expected matrix condition number.

    Parameters
    ----------
    condition_number : float
        The expected condition number of the SPD matrices.
    prefer_speed : bool, default=False
        If True, prefer faster dtypes when possible.

    Returns
    -------
    torch.dtype
        The recommended dtype.

    Examples
    --------
    >>> from spd_learn.functional.numerical import recommend_dtype_for_spd
    >>> # Well-conditioned matrices can use float32
    >>> print(recommend_dtype_for_spd(1e3))
    torch.float32
    >>> # Ill-conditioned matrices need float64
    >>> print(recommend_dtype_for_spd(1e10))
    torch.float64
    """
    if condition_number > 1e8:
        return torch.float64
    elif condition_number > 1e4 and not prefer_speed:
        return torch.float64
    else:
        return torch.float32
