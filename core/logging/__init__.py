"""
Core logging module for the application.

This module provides centralized logging functionality including:
- Colored console output
- File-based logging with rotation
- Structured logging with context
- Exception handling integration
"""

from .logger_impl import AppLogger, get_logger, configure_logging, get_app_logger

# Public API
__all__ = ['AppLogger', 'get_logger', 'configure_logging', 'get_app_logger']
