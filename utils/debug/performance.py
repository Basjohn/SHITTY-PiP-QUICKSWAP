"""
Performance measurement and profiling utilities.

This module provides tools for measuring and logging the performance of code,
including decorators and context managers for timing execution.
"""

import logging
import time
from core.logging import get_logger
from utils.debug import perf_logging_enabled
from typing import Any, Callable, TypeVar, cast
from functools import wraps

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])

def log_perf(level: int = logging.DEBUG, threshold_ms: float = 0.0):
    """
    Decorator to log function execution time when performance logging is enabled.
    
    Args:
        level: Logging level to use for performance messages.
        threshold_ms: Only log if execution time exceeds this threshold in milliseconds.
                     Use 0 to log all calls when performance logging is enabled.
                      
    Returns:
        A decorator function that can be applied to other functions.
        
    Example:
        @log_perf(level=logging.INFO, threshold_ms=100)
        def slow_function():
            # This will be logged if it takes more than 100ms
            time.sleep(0.2)
    """
    def decorator(func: F) -> F:
        if not perf_logging_enabled():
            return func
            
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            if threshold_ms == 0 or elapsed_ms >= threshold_ms:
                logger = get_logger('perf')
                if logger.isEnabledFor(level):
                    logger.log(
                        level,
                        f"{func.__qualname__} executed in {elapsed_ms:.2f}ms"
                    )
            
            return result
            
        return cast(F, wrapper)
    return decorator

class DebugTimer:
    """
    Context manager for timing code blocks with optional logging.
    
    Example:
        with DebugTimer("Expensive operation"):
            # Code to time
            result = expensive_operation()
    """
    
    def __init__(self, name: str = None, level: int = logging.DEBUG):
        """
        Initialize the timer.
        
        Args:
            name: Name for this timer (used in log messages).
            level: Logging level to use for the timing message.
        """
        self.name = name or "Code block"
        self.level = level
        self.start_time = 0.0
        self.logger = get_logger('perf')
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        """Start the timer and return self."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the timer and log the elapsed time.
        
        Args:
            exc_type: Exception type if an exception was raised in the context.
            exc_val: Exception value if an exception was raised.
            exc_tb: Traceback if an exception was raised.
        """
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        
        if perf_logging_enabled() and self.logger.isEnabledFor(self.level):
            self.logger.log(
                self.level,
                f"{self.name} completed in {self.elapsed_ms:.2f}ms"
            )
    
    @property
    def elapsed_seconds(self) -> float:
        """Get the elapsed time in seconds."""
        return self.elapsed_ms / 1000
    
    @property
    def elapsed_milliseconds(self) -> float:
        """Get the elapsed time in milliseconds."""
        return self.elapsed_ms