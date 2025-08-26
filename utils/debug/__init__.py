"""
Debugging and logging utilities for the application.

This package provides centralized debugging functionality including:
- Logging configuration with rotation
- Debug flags and state management
- Performance measurement tools
- Debug visualization helpers
"""

import warnings
from typing import Any, Optional
import os

# Core imports
from core.logging import get_logger as core_get_logger

# Local API (implemented here to avoid importing deprecated utils.debug.logging)
# Default debug to OFF unless explicitly enabled via environment variable.
# Recognized truthy values: 1, true, yes, on (case-insensitive)
_debug_env = os.environ.get("SPQ_DEBUG", "")
debug_enabled = _debug_env.strip().lower() in ("1", "true", "yes", "on")

def debug(message: str, *args: Any, **kwargs: Any) -> None:
    logger = core_get_logger(__name__)
    logger.debug(message, *args, **kwargs)

def info(message: str, *args: Any, **kwargs: Any) -> None:
    logger = core_get_logger(__name__)
    logger.info(message, *args, **kwargs)

def warning(message: str, *args: Any, **kwargs: Any) -> None:
    logger = core_get_logger(__name__)
    logger.warning(message, *args, **kwargs)

def error(message: str, *args: Any, **kwargs: Any) -> None:
    logger = core_get_logger(__name__)
    logger.error(message, *args, **kwargs)

def critical(message: str, *args: Any, **kwargs: Any) -> None:
    logger = core_get_logger(__name__)
    logger.critical(message, *args, **kwargs)

def exception(message: str, *args: Any, **kwargs: Any) -> None:
    logger = core_get_logger(__name__)
    logger.exception(message, *args, **kwargs)

# Re-export the logger functions
get_logger = core_get_logger

# Version information
__version__ = "1.0.0"

# For backward compatibility
def setup_logging() -> None:
    """Deprecated. Logging is now configured in core.logging."""
    warnings.warn(
        "setup_logging() is deprecated. Logging is now configured in core.logging",
        DeprecationWarning,
        stacklevel=2
    )

def set_debug_mode(enabled: bool = True) -> None:
    """Set the debug mode.
    
    Args:
        enabled: Whether to enable debug mode
    """
    global debug_enabled
    debug_enabled = enabled

def debug_print(*args: Any, **kwargs: Any) -> None:
    """Print a debug message if debug mode is enabled."""
    if debug_enabled:
        print(*args, **kwargs)

def log_exception(message: str, exc_info: Optional[BaseException] = None) -> None:
    """Log an exception with the given message.
    
    Args:
        message: The message to log
        exc_info: Optional exception info to include
    """
    logger = core_get_logger(__name__)
    logger.exception(message, exc_info=exc_info)

# Performance logging stubs (callable API)
_perf_logging_enabled = False

def perf_logging_enabled() -> bool:
    """Return whether performance logging is enabled."""
    return _perf_logging_enabled

def set_perf_logging(enabled: bool = True) -> None:
    """Enable or disable performance logging.
    
    Args:
        enabled: Whether to enable performance logging
    """
    global _perf_logging_enabled
    _perf_logging_enabled = enabled
    warnings.warn(
        "Performance logging is not implemented yet",
        RuntimeWarning,
        stacklevel=2
    )

# Public API
__all__ = [
    # Core logging functions
    'get_logger',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'exception',
    
    # Debug utilities
    'debug_enabled',
    'set_debug_mode',
    'debug_print',
    'log_exception',
    
    # Performance monitoring
    'perf_logging_enabled',
    'set_perf_logging',
    
    # Backward compatibility
    'setup_logging',
    
    # Note: log_perf and DebugTimer are not implemented yet
    # 'log_perf',
    # 'DebugTimer'
]