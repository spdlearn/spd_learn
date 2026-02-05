# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Logging configuration for spd_learn.

This module provides centralized logging utilities for the spd_learn library,
with support for Rich formatting when available.

Examples
--------
Basic usage:

>>> from spd_learn.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Training SPDNet model")

Configure logging level:

>>> from spd_learn.logging import set_log_level
>>> set_log_level("DEBUG")

Use with Rich formatting (if installed):

>>> from spd_learn.logging import configure_logging
>>> configure_logging(use_rich=True, level="INFO")
"""

from __future__ import annotations

import logging
import sys
import warnings

from contextlib import contextmanager
from typing import Literal


# Package-level logger name
LOGGER_NAME = "spd_learn"

# Default format for standard logging
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"

# Type alias for log levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for spd_learn.

    Parameters
    ----------
    name : str | None
        The name of the logger. If None, returns the root spd_learn logger.
        If provided, creates a child logger under spd_learn namespace.

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing SPD matrices")
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    if not name.startswith(LOGGER_NAME):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def set_log_level(level: LogLevel | int) -> None:
    """Set the logging level for all spd_learn loggers.

    Parameters
    ----------
    level : LogLevel | int
        The logging level. Can be a string ("DEBUG", "INFO", etc.) or
        an integer from the logging module.

    Examples
    --------
    >>> set_log_level("DEBUG")
    >>> set_log_level(logging.WARNING)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logging.getLogger(LOGGER_NAME).setLevel(level)


def configure_logging(
    level: LogLevel | int = "INFO",
    use_rich: bool = True,
    format_string: str | None = None,
    show_path: bool = False,
    show_time: bool = True,
) -> None:
    """Configure logging for the spd_learn package.

    Parameters
    ----------
    level : LogLevel | int
        The logging level. Default is "INFO".
    use_rich : bool
        Whether to use Rich for formatted output (if available).
        Default is True.
    format_string : str | None
        Custom format string for log messages. If None, uses default format.
        Ignored when use_rich is True.
    show_path : bool
        Whether to show the file path in log messages (Rich only).
        Default is False.
    show_time : bool
        Whether to show timestamps in log messages (Rich only).
        Default is True.

    Examples
    --------
    >>> configure_logging(level="DEBUG", use_rich=True)
    >>> configure_logging(level="WARNING", use_rich=False)
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Remove existing handlers
    logger.handlers.clear()

    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)

    # Try to use Rich if requested and available
    handler: logging.Handler | None = None
    if use_rich:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                level=level,
                show_path=show_path,
                show_time=show_time,
                rich_tracebacks=True,
                markup=True,
            )
        except ImportError:
            pass

    # Fall back to standard handler
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        fmt = format_string or DEFAULT_FORMAT
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel(level)

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False


def disable_logging() -> None:
    """Disable all logging for spd_learn.

    Examples
    --------
    >>> disable_logging()
    """
    logging.getLogger(LOGGER_NAME).setLevel(logging.CRITICAL + 1)


def enable_logging(level: LogLevel | int = "INFO") -> None:
    """Re-enable logging for spd_learn after disabling.

    Parameters
    ----------
    level : LogLevel | int
        The logging level to set. Default is "INFO".

    Examples
    --------
    >>> enable_logging("DEBUG")
    """
    set_log_level(level)


@contextmanager
def log_level(level: LogLevel | int):
    """Context manager to temporarily change the log level.

    Parameters
    ----------
    level : LogLevel | int
        The temporary logging level.

    Yields
    ------
    None

    Examples
    --------
    >>> with log_level("DEBUG"):
    ...     logger.debug("This will be shown")
    >>> logger.debug("This might not be shown")
    """
    logger = logging.getLogger(LOGGER_NAME)
    old_level = logger.level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def warn_once(message: str, category: type = UserWarning) -> None:
    """Issue a warning that will only be shown once.

    Parameters
    ----------
    message : str
        The warning message.
    category : type
        The warning category. Default is UserWarning.

    Examples
    --------
    >>> warn_once("This feature is deprecated")
    """
    warnings.warn(message, category, stacklevel=2)


class DeprecationHelper:
    """Helper class for managing deprecation warnings.

    Parameters
    ----------
    old_name : str
        The deprecated name/feature.
    new_name : str
        The new name/feature to use instead.
    version : str
        The version when the deprecation was introduced.
    removal_version : str | None
        The version when the feature will be removed.

    Examples
    --------
    >>> deprecation = DeprecationHelper("old_func", "new_func", "0.2.0", "1.0.0")
    >>> deprecation.warn()
    """

    def __init__(
        self,
        old_name: str,
        new_name: str,
        version: str,
        removal_version: str | None = None,
    ):
        self.old_name = old_name
        self.new_name = new_name
        self.version = version
        self.removal_version = removal_version
        self._warned = False

    def warn(self) -> None:
        """Issue the deprecation warning (only once per instance)."""
        if self._warned:
            return
        self._warned = True

        msg = (
            f"'{self.old_name}' is deprecated since version {self.version}. "
            f"Use '{self.new_name}' instead."
        )
        if self.removal_version:
            msg += f" It will be removed in version {self.removal_version}."

        warnings.warn(msg, DeprecationWarning, stacklevel=3)


def deprecated(
    old_name: str,
    new_name: str,
    version: str,
    removal_version: str | None = None,
):
    """Decorator to mark a function as deprecated.

    Parameters
    ----------
    old_name : str
        The deprecated name.
    new_name : str
        The new name to use.
    version : str
        The version when deprecation was introduced.
    removal_version : str | None
        The version when the function will be removed.

    Returns
    -------
    callable
        A decorator function.

    Examples
    --------
    >>> @deprecated("old_func", "new_func", "0.2.0", "1.0.0")
    ... def old_func():
    ...     return new_func()
    """
    import functools

    def decorator(func):
        helper = DeprecationHelper(old_name, new_name, version, removal_version)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            helper.warn()
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Initialize logging with defaults on import
# Users can reconfigure with configure_logging()
_logger = get_logger()
if not _logger.handlers:
    # Only add a NullHandler by default to avoid "No handler found" warnings
    _logger.addHandler(logging.NullHandler())
