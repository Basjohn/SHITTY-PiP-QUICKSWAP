"""
Debugging utilities for the Shitty PiP QuickSwap application.

This module provides centralized debugging functionality including:
- Logging configuration with rotation
- Debug flags and state management
- Debug print utilities
- Performance measurement tools
- Debug visualization helpers
"""

import logging
import logging.handlers
import os
import sys
import time
import errno
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast, Dict, Union
from functools import wraps

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])

# Debug flags - controlled via environment variables
# Enable debug for troubleshooting quick swap issues
_DEBUG_MODE = False  # Always enable debug mode by default
_PERF_LOGGING = False  # Enable performance logging by default


class RobustRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A more robust rotating file handler that handles file access issues gracefully."""
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        self._base_filename = os.path.abspath(filename)
        self._directory = os.path.dirname(self._base_filename)
        self._ensure_directory_exists()
        logging.handlers.RotatingFileHandler.__init__(
            self, filename, mode, maxBytes, backupCount, encoding, delay
        )
    
    def _ensure_directory_exists(self):
        """Ensure the log directory exists."""
        if not os.path.exists(self._directory):
            try:
                os.makedirs(self._directory, exist_ok=True)
            except OSError as e:
                # If we can't create the directory, fall back to temp directory
                if e.errno != errno.EEXIST:
                    self._directory = os.path.join(os.environ.get('TEMP', '.'), 'SPQ_LOGS')
                    os.makedirs(self._directory, exist_ok=True)
                    self._base_filename = os.path.join(self._directory, os.path.basename(self._base_filename))
    
    def _open(self):
        """Open the current base file with the (original) mode and encoding."""
        try:
            return open(self.baseFilename, self.mode, encoding=self.encoding)
        except IOError as e:
            # If we can't open the file, try to create it
            if e.errno == errno.ENOENT:  # No such file or directory
                try:
                    return open(self.baseFilename, 'w' + self.mode, encoding=self.encoding)
                except IOError:
                    pass
            # If we still can't open it, fall back to stderr
            return sys.stderr
    
    def emit(self, record):
        """Emit a record."""
        try:
            if self.shouldRollover(record):
                self.doRollover()
            logging.FileHandler.emit(self, record)
        except (IOError, OSError) as e:
            # If we can't write to the log file, write to stderr
            if not getattr(self, 'reported_error', False):
                sys.stderr.write(f"Failed to write to log file {self.baseFilename}: {e}\n")
                self.reported_error = True
        except Exception:
            self.handleError(record)
    
    def shouldRollover(self, record):
        """Determine if rollover should occur."""
        if self.stream is None:  # Delay was set.
            self.stream = self._open()
        if self.maxBytes > 0:  # Are we rolling over?
            try:
                self.stream.seek(0, 2)  # Seek to end
                if self.stream.tell() + len(record.getMessage()) >= self.maxBytes:
                    return 1
            except (IOError, OSError):
                # If we can't check the size, don't roll over
                return 0
        return 0

import threading
# Module-level state
_logger: Optional[logging.Logger] = None
_initialized: bool = False
_log_lock = threading.Lock()

# Default logging parameters (can be overridden by env vars)
LOG_FILE = os.environ.get('SPQ_LOG_FILE', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'spqrestore.log'))
LOG_FORMAT = os.environ.get('SPQ_LOG_FORMAT', '%(asctime)s - %(levelname)-8s - %(name)-20s - %(message)s')
LOG_LEVEL = int(os.environ.get('SPQ_LOG_LEVEL', logging.DEBUG if os.environ.get('SPQ_DEBUG', '').lower() in ('1', 'true', 'yes') else logging.INFO))

# Log rotation settings (env var overrides)
MAX_LOG_SIZE = int(os.environ.get('SPQ_MAX_LOG_SIZE', 2 * 1024 * 1024))  # 2 MB
MAX_LOG_BACKUPS = int(os.environ.get('SPQ_MAX_LOG_BACKUPS', 2))
MAX_LOG_AGE_DAYS = int(os.environ.get('SPQ_MAX_LOG_AGE_DAYS', 7))  # For time-based rotation
LOG_ROTATION_POLICY = os.environ.get('SPQ_LOG_ROTATION', 'size')  # 'size' or 'time'
LOG_ROTATION_WHEN = os.environ.get('SPQ_LOG_ROTATION_WHEN', 'midnight')  # for time-based
LOG_ROTATION_INTERVAL = int(os.environ.get('SPQ_LOG_ROTATION_INTERVAL', 1))  # for time-based

# Ensure we're using the correct log directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'logs'))
LOG_FILE = os.environ.get('SPQ_LOG_FILE', os.path.join(LOG_DIR, 'spq_debug.log'))

# Log level overrides for noisy modules
LOG_LEVEL_OVERRIDES: Dict[str, int] = {
    'PIL': logging.WARNING,
    'matplotlib': logging.WARNING,
    'PySide6': logging.WARNING,
    'shiboken6': logging.WARNING,
    'asyncio': logging.WARNING,
    'urllib3': logging.WARNING,
    'win32com': logging.WARNING,
    'comtypes': logging.WARNING,
}


def setup_logging(log_file: Optional[Union[str, Path]] = None, 
                 log_level: Optional[int] = None):
    """
    Configure logging for the application with rotation and proper formatting.
    
    Args:
        log_file: Path to the log file. If None, uses the default from config or environment.
        log_level: Logging level. If None, uses the default from config or environment.
    
    Environment Variables:
        SPQ_DEBUG: Set to '1', 'true', or 'yes' to enable debug mode
        SPQ_PERF: Set to '1', 'true', or 'yes' to enable performance logging
        SPQ_LOG_FILE: Override the log file path
        SPQ_LOG_LEVEL: Override the log level (int)
        SPQ_MAX_LOG_SIZE: Max log file size in bytes (for size rotation)
        SPQ_MAX_LOG_BACKUPS: Number of backup log files
        SPQ_MAX_LOG_AGE_DAYS: Max days to keep logs (for time rotation)
        SPQ_LOG_ROTATION: 'size' or 'time'
        SPQ_LOG_ROTATION_WHEN: When to rotate logs (for time rotation, e.g. 'midnight')
        SPQ_LOG_ROTATION_INTERVAL: Interval for time-based rotation
        SPQ_LOG_FORMAT: Log message format
    """
    global _logger, _initialized, LOG_LEVEL, LOG_FILE, _DEBUG_MODE, _PERF_LOGGING
    # Re-check debug mode in case environment variables changed
    _DEBUG_MODE = os.environ.get('SPQ_DEBUG', '').lower() in ('1', 'true', 'yes')
    _PERF_LOGGING = os.environ.get('SPQ_PERF', '').lower() in ('1', 'true', 'yes')
    # Set default log level to CRITICAL for release build
    LOG_LEVEL = int(os.environ.get('SPQ_LOG_LEVEL', logging.CRITICAL))
    log_file = log_file or os.environ.get('SPQ_LOG_FILE', LOG_FILE)
    log_level = log_level or LOG_LEVEL
    # Thread-safe logging setup
    with _log_lock:
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        # Set up console handler to show all debug messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Changed from CRITICAL to DEBUG
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        if log_level <= logging.CRITICAL:
            try:
                log_file_path = os.path.abspath(log_file)
                if LOG_ROTATION_POLICY == 'time':
                    file_handler = logging.handlers.TimedRotatingFileHandler(
                        log_file_path,
                        when=LOG_ROTATION_WHEN,
                        interval=LOG_ROTATION_INTERVAL,
                        backupCount=MAX_LOG_BACKUPS,
                        encoding='utf-8',
                        delay=True
                    )
                else:
                    file_handler = RobustRotatingFileHandler(
                        log_file_path,
                        maxBytes=MAX_LOG_SIZE,
                        backupCount=MAX_LOG_BACKUPS,
                        encoding='utf-8',
                        delay=True
                    )
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter(LOG_FORMAT)
                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)
                if _DEBUG_MODE:
                    _logger = root_logger
                    _logger.debug(f"Logging to file: {log_file_path}")
            except Exception as e:
                if _DEBUG_MODE:
                    print(f"Failed to set up file logging: {e}", file=sys.stderr)
                if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
                    console = logging.StreamHandler(sys.stderr)
                    console.setLevel(log_level)
                    console.setFormatter(logging.Formatter(LOG_FORMAT))
                    root_logger.addHandler(console)
        for module, level in LOG_LEVEL_OVERRIDES.items():
            logging.getLogger(module).setLevel(level)
        _initialized = True
        _logger = logging.getLogger(__name__)
        if _DEBUG_MODE:
            _logger.info("Debug mode enabled")
        if _PERF_LOGGING:
            _logger.info("Performance logging enabled")


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name. If None, returns the root logger.
        
    Returns:
        Configured logger instance with proper settings.
    """
    # Ensure logging is set up
    if not _initialized:
        setup_logging()
        
    logger = logging.getLogger(name)
    
    # Only set up debug logging if debug mode is enabled
    if _DEBUG_MODE and not logger.isEnabledFor(logging.DEBUG):
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled for logger: %s", name)
    
    # Add a null handler if no handlers are configured (prevents 'No handlers' warnings)
    if not logger.handlers and not logging.getLogger().handlers:
        return logging.getLogger()
    return logging.getLogger(name)


def debug_enabled() -> bool:
    """
    Check if debug mode is enabled.
    
    Returns:
        bool: True if debug mode is enabled, False otherwise.
    """
    return _DEBUG_MODE or LOG_LEVEL <= logging.DEBUG


def perf_logging_enabled() -> bool:
    """
    Check if performance logging is enabled.
    
    Returns:
        bool: True if performance logging is enabled, False otherwise.
    """
    return _PERF_LOGGING


def set_perf_logging(enabled: bool) -> None:
    """
    Enable or disable performance logging at runtime.
    
    Args:
        enabled: Whether to enable performance logging.
    """
    global _PERF_LOGGING
    _PERF_LOGGING = enabled
    logger = get_logger(__name__)
    logger.info(f"Performance logging {'enabled' if enabled else 'disabled'}")


def set_debug_mode(enabled: bool) -> None:
    """
    Enable or disable debug mode at runtime.
    
    Args:
        enabled: Whether to enable debug mode.
    """
    global _DEBUG_MODE, LOG_LEVEL, _logger
    
    _DEBUG_MODE = enabled
    LOG_LEVEL = logging.DEBUG if enabled else logging.INFO
    
    # Get logger if not already available
    if _logger is None:
        _logger = get_logger(__name__)
    
    # Log the change
    level_name = logging.getLevelName(LOG_LEVEL)
    _logger.info(
        "Debug mode %s, log level set to %s",
        "enabled" if enabled else "disabled",
        level_name
    )
    
    # Re-initialize logging to apply changes
    if _initialized:
        setup_logging()


def debug_print(*args, level: int = logging.DEBUG, exc_info: bool = False, **kwargs) -> None:
    """
    Print debug messages when debug mode is enabled.
    
    Args:
        *args: Positional arguments to format into the message.
        level: Logging level (default: DEBUG). Only messages at or above the current
              log level will be processed.
        exc_info: If True, includes exception info in the output.
        **kwargs: Additional keyword arguments passed to the logger.
    """
    if not _initialized:
        setup_logging()
    
    # Only process if debug is enabled for DEBUG level messages
    # Always process INFO and above regardless of debug mode
    if (_DEBUG_MODE and level == logging.DEBUG) or level >= logging.INFO:
        logger = get_logger(kwargs.pop('logger', None))
        if logger.isEnabledFor(level):
            # Format the message
            message = ' '.join(str(arg) for arg in args)
            logger.log(level, message, exc_info=exc_info, **kwargs)
    
    # Only process if the logger is enabled for this level
    if not logger.isEnabledFor(level):
        return
    
    # Format the message
    message = ' '.join(str(arg) for arg in args)
    
    # Log the message with appropriate stack level
    logger.log(level, message, exc_info=exc_info, stacklevel=2, **kwargs)

def log_exception(msg: str, exc: Exception = None, logger_name: str = None):
    """
    Log an exception with an optional message and exception object.
    Args:
        msg: Contextual message for the error.
        exc: Exception instance (optional, will use sys.exc_info if not provided).
        logger_name: Name of the logger to use (optional).
    """
    logger = get_logger(logger_name)
    if exc is not None:
        logger.error(msg, exc_info=exc)
    else:
        logger.error(msg, exc_info=True)



def log_perf(level: int = logging.DEBUG, threshold_ms: float = 0.0) -> Callable[[F], F]:
    """
    Decorator to log function execution time when performance logging is enabled.
    
    Args:
        level: Logging level to use for performance messages.
        threshold_ms: Only log if execution time exceeds this threshold in milliseconds.
                     Use 0 to log all calls when performance logging is enabled.
                     
    Returns:
        A decorator function that can be applied to other functions.
        
    Note:
        Respects the PERF_LOGGING flag. Set SPQ_PERF=1 to enable performance logging.
    """
    def decorator(func: F) -> F:
        if not _PERF_LOGGING:
            return func
            
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if elapsed_ms >= threshold_ms:
                    logger = get_logger(func.__module__)
                    if logger.isEnabledFor(level):
                        logger.log(
                            level,
                            f"{func.__qualname__} took {elapsed_ms:.2f}ms",
                            extra={"perf_time_ms": elapsed_ms, "func_name": func.__qualname__}
                        )
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
        self.logger = get_logger(__name__)
        self.elapsed_ms = 0.0
        
    def __enter__(self) -> 'DebugTimer':
        """Start the timer and return self."""
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Any) -> None:
        """
        Stop the timer and log the elapsed time.
        
        Args:
            exc_type: Exception type if an exception was raised in the context.
            exc_val: Exception value if an exception was raised.
            exc_tb: Traceback if an exception was raised.
        """
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        
        # Only log if performance logging is enabled or there was an error
        if exc_type is not None:
            self.logger.error(
                "%s failed after %.2fms", 
                self.name, 
                self.elapsed_ms,
                exc_info=(exc_type, exc_val, exc_tb) if exc_val else None
            )
        elif _PERF_LOGGING and self.logger.isEnabledFor(self.level):
            self.logger.log(
                self.level,
                "%s took %.2fms", 
                self.name, 
                self.elapsed_ms,
                extra={"perf_time_ms": self.elapsed_ms, "timer_name": self.name}
            )
    
    @property
    def elapsed_seconds(self) -> float:
        """Get the elapsed time in seconds."""
        if not self.start_time:
            return 0.0
        return (time.perf_counter() - self.start_time)
    
    @property
    def elapsed_milliseconds(self) -> float:
        """Get the elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000


def _safe_setup_logging():
    """Safely initialize logging with error handling."""
    try:
        setup_logging()
    except Exception as e:
        # If we can't even set up basic logging, print to stderr
        import traceback
        sys.stderr.write(f"Failed to initialize logging: {e}\n")
        sys.stderr.write(f"{traceback.format_exc()}\n")
        # Set up a basic console handler as fallback
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.basicConfig(handlers=[handler], level=logging.WARNING)
        logging.getLogger(__name__).error("Logging initialization failed: %s", e, exc_info=True)

# Initialize logging when module is imported
_safe_setup_logging()
