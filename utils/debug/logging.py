"""
Compatibility module for old logging imports.

This module provides backward compatibility for code that was using the old
`utils.debug.logging` import path. It redirects to the new logging implementation
in `core.logging` and exposes minimal debug/perf toggles expected by callers.
"""

import warnings
import os
from typing import Any

from core.logging import get_logger as core_get_logger

# Show deprecation warning
warnings.warn(
    "The 'utils.debug.logging' module is deprecated. "
    "Please update your imports to use 'core.logging' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export the get_logger function
get_logger = core_get_logger

# For backward compatibility
# Default debug to OFF unless explicitly enabled via environment variable.
# Recognized truthy values: 1, true, yes, on (case-insensitive)
_debug_env = os.environ.get("SPQ_DEBUG", "")
debug_enabled = _debug_env.strip().lower() in ("1", "true", "yes", "on")

# Minimal perf logging toggle expected by utils.debug.performance
_perf_logging_enabled = False

def perf_logging_enabled() -> bool:
    """Return whether performance logging is enabled.

    Exposed for legacy callers. The authoritative logger remains in core.logging.
    """
    return _perf_logging_enabled

def set_perf_logging(enabled: bool = True) -> None:
    """Enable or disable performance logging for legacy modules."""
    global _perf_logging_enabled
    _perf_logging_enabled = enabled

def debug(message: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message."""
    logger = core_get_logger(__name__)
    logger.debug(message, *args, **kwargs)

def info(message: str, *args: Any, **kwargs: Any) -> None:
    """Log an info message."""
    logger = core_get_logger(__name__)
    logger.info(message, *args, **kwargs)

def warning(message: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    logger = core_get_logger(__name__)
    logger.warning(message, *args, **kwargs)

def error(message: str, *args: Any, **kwargs: Any) -> None:
    """Log an error message."""
    logger = core_get_logger(__name__)
    logger.error(message, *args, **kwargs)

def critical(message: str, *args: Any, **kwargs: Any) -> None:
    """Log a critical message."""
    logger = core_get_logger(__name__)
    logger.critical(message, *args, **kwargs)

def exception(message: str, *args: Any, **kwargs: Any) -> None:
    """Log an exception with stack trace."""
    logger = core_get_logger(__name__)
    logger.exception(message, *args, **kwargs)

# For backwards compatibility with code that might be checking __name__
__all__ = [
    'get_logger',
    'debug_enabled',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'exception',
    'perf_logging_enabled',
    'set_perf_logging',
]
