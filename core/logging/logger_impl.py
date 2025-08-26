"""
Logger implementation for the application.

This module contains the concrete implementation of the logging functionality.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import threading
import time

from core.interfaces import ILogger

# Best-effort enable ANSI colors on Windows consoles
try:  # pragma: no cover - environment dependent
    import colorama  # type: ignore
    # Prefer the light-touch helper that doesn't wrap streams
    if hasattr(colorama, "just_fix_windows_console"):
        colorama.just_fix_windows_console()
    else:
        colorama.init()  # Fallback init
except Exception:
    # Colors remain in output; some consoles may not show them. This is acceptable for logging.
    pass

# Module-level singleton configuration (logging may be configured early at app start)
_configured: bool = False
_app_logger: Optional["AppLogger"] = None

# Dynamic task highlighting: selected logger names can be emphasized in output.
# Use provided helper functions to add/remove names at runtime.
_highlighted_loggers: set[str] = set()

def set_highlighted_topics(names: "list[str] | tuple[str, ...]") -> None:
    """Replace the set of highlighted logger names."""
    global _highlighted_loggers
    _highlighted_loggers = {str(n) for n in names if n}

def add_highlighted_topic(name: str) -> None:
    """Add a logger name to the highlighted set."""
    if not name:
        return
    _highlighted_loggers.add(str(name))

def remove_highlighted_topic(name: str) -> None:
    """Remove a logger name from the highlighted set (no error if absent)."""
    if not name:
        return
    _highlighted_loggers.discard(str(name))

def clear_highlighted_topics() -> None:
    """Clear the highlighted logger names set."""
    _highlighted_loggers.clear()


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    # ANSI color codes
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    
    # Color sequences for different log levels
    COLORS = {
        'DEBUG': BLUE,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': MAGENTA,
    }
    
    # Reset sequence
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    DIM_SEQ = "\033[2m"
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """Initialize the formatter with optional format strings."""
        # Our formatter builds the line manually for precise coloring; fmt is unused.
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as text with color.

        Structure:
        <time> <colored level> <dim logger>: message
        """
        # Base message and time
        asctime = self.formatTime(record, self.datefmt)
        message = record.getMessage()

        # Level coloring
        levelname = record.levelname
        color_code = 30 + self.COLORS.get(levelname, self.WHITE)
        colored_level = f"{self.COLOR_SEQ % color_code}{levelname:>8}{self.RESET_SEQ}"

        # Highlighting: if this logger is marked for emphasis, color its name and message in yellow (orange-ish)
        if record.name in _highlighted_loggers:
            hl_color = 30 + self.YELLOW
            name_part = f"{self.COLOR_SEQ % hl_color}{record.name}{self.RESET_SEQ}"
            message = f"{self.COLOR_SEQ % hl_color}{message}{self.RESET_SEQ}"
        else:
            # Dim logger name
            name_part = f"{self.DIM_SEQ}{record.name}{self.RESET_SEQ}"

        line = f"{asctime} {colored_level} {name_part}: {message}"

        # Include exception/stack if present
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)

        return line


class RepeatSuppressFilter(logging.Filter):
    """Suppress repeated identical messages for a time window on this handler.

    - Allows first message immediately.
    - Suppresses subsequent identical messages (same logger, level, text) for window_seconds.
    - On the next allowed emission, annotates the message with the number suppressed.
    """

    def __init__(self, window_seconds: float = 2.0) -> None:
        super().__init__()
        self.window = window_seconds
        self._last_time: float = 0.0
        self._last_key: Optional[tuple] = None
        self._suppressed: int = 0

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        import time

        key = (record.name, record.levelno, record.getMessage())
        now = time.monotonic()

        if self._last_key == key:
            # Same message as last allowed; check window
            if now - self._last_time < self.window:
                self._suppressed += 1
                return False
            # Window expired; annotate with suppressed count if any
            if self._suppressed > 0:
                # Clear args to avoid string formatting conflicts
                record.args = ()
                record.msg = f"{record.getMessage()} [suppressed {self._suppressed} repeats]"
                self._suppressed = 0
            self._last_time = now
            return True

        # Different message; reset suppression and allow
        self._last_key = key
        self._last_time = now
        self._suppressed = 0
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    This is a convenience function that creates and configures a logger with
    the standard application formatting and handlers.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handlers if this is the first time this logger is being configured
    if not logger.handlers:
        # Prefer centralized AppLogger handlers if available; otherwise fall back to a local console.
        app = None
        try:
            app = get_app_logger()
        except Exception:
            app = None

        if app is not None:
            # Attach only AppLogger's handlers (console/file). Do NOT create a per-logger console handler.
            for h in list(app._logger.handlers):  # reuse same handlers (e.g., rotating file)
                try:
                    # Avoid duplicate attachment of same handler instance
                    if all(h is not eh for eh in logger.handlers):
                        logger.addHandler(h)
                except Exception:
                    continue
            # Allow propagation so external frameworks (e.g., pytest caplog) can capture records
            logger.propagate = True
        else:
            # Centralized logger not configured yet: set up a local console handler with repeat suppression.
            console_handler = logging.StreamHandler()
            console_handler.addFilter(RepeatSuppressFilter(window_seconds=2.0))
            formatter = ColoredFormatter()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Set level to DEBUG by default (handlers will filter)
        logger.setLevel(logging.DEBUG)
    
    return logger


def configure_logging(
    *,
    name: str = "app",
    log_dir: Optional[Union[str, Path]] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    enable_exception_hook: bool = True,
    resource_manager: Optional[object] = None,
) -> "AppLogger":
    """Configure the application logger singleton.

    This is a centralized entry point for setting up console/file logging, with
    optional ResourceManager integration. Safe to call multiple times; the first
    call wins and subsequent calls return the existing logger.
    """
    global _configured, _app_logger
    if _configured and _app_logger is not None:
        return _app_logger

    # Instantiate the AppLogger (it will choose a default log_dir if None)
    _app_logger = AppLogger(
        name=name,
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        enable_exception_hook=enable_exception_hook,
        resource_manager=resource_manager,
    )
    _configured = True
    return _app_logger


def get_app_logger() -> Optional["AppLogger"]:
    """Return the configured AppLogger instance if available."""
    return _app_logger


class AppLogger(ILogger):
    """
    Centralized logging for the application.
    
    This class provides a unified interface for logging messages with different
    severity levels. It supports both console and file output with configurable
    formatting and log rotation.
    """
    
    def __init__(self, 
                name: str = "app",
                log_dir: Optional[Union[str, Path]] = None,
                console_level: int = logging.INFO,
                file_level: int = logging.DEBUG,
                max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                backup_count: int = 5,
                enable_exception_hook: bool = True,
                resource_manager: Optional[object] = None):
        """Initialize the logger.
        
        Args:
            name: Name of the logger
            log_dir: Directory to store log files. If None, a default location is used.
            console_level: Minimum log level for console output
            file_level: Minimum log level for file output
            max_bytes: Maximum size of a single log file before rotation
            backup_count: Number of backup log files to keep
        """
        self._logger = logging.getLogger(name)
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else self._get_default_log_dir()
        self._resource_manager = resource_manager
        self._json_ctx_warned = False
        self._logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers control actual level
        
        # Don't propagate to root logger
        self._logger.propagate = False
        
        # Set up log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up console handler
        self._setup_console_handler(console_level)
        
        # Set up file handler
        self._setup_file_handler(file_level, max_bytes, backup_count)
        if enable_exception_hook:
            self._install_exception_handler()

        # Best-effort resource registrations
        try:
            self._register_resources()
        except Exception:
            # Logging should not crash the app; ignore registration failures
            pass
        self.debug(f"Logger initialized. Log directory: {self.log_dir}")
    
    def _get_default_log_dir(self) -> Path:
        """Get the default log directory based on the operating system."""
        if os.name == 'nt':  # Windows
            app_data = os.getenv('LOCALAPPDATA')
            if app_data:
                return Path(app_data) / 'SPQModular' / 'logs'
        else:  # Unix-like
            cache_home = os.getenv('XDG_CACHE_HOME')
            if cache_home:
                return Path(cache_home) / 'spqmodular' / 'logs'
            home = os.getenv('HOME')
            if home:
                return Path(home) / '.cache' / 'spqmodular' / 'logs'
        
        # Fallback to current directory
        return Path.cwd() / 'logs'
    
    def _setup_console_handler(self, level: int) -> None:
        """Set up the console handler."""
        # Check if we already have a console handler
        for handler in self._logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                return  # Already have a console handler
        
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        # Global repeat suppression for console output
        console.addFilter(RepeatSuppressFilter(window_seconds=2.0))
        
        # Use colored formatter for console
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console.setFormatter(formatter)
        
        self._logger.addHandler(console)
    
    def _setup_file_handler(self, level: int, max_bytes: int, backup_count: int) -> None:
        """Set up the file handler with log rotation."""
        # Create log file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'app_{timestamp}.log'
        
        # Use RotatingFileHandler for log rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        file_handler.setLevel(level)
        
        # Apply repeat suppression for file output as well (prevents flooding)
        file_handler.addFilter(RepeatSuppressFilter(window_seconds=2.0))

        # Use a more detailed formatter for file output
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        self._logger.addHandler(file_handler)

    def _register_resources(self) -> None:
        """Register log resources with the ResourceManager if provided."""
        if not self._resource_manager:
            return
        try:
            # Local import to avoid mandatory dependency when not used
            from utils.resource_manager import ResourceType  # type: ignore
        except Exception:
            return

        # Register log directory
        try:
            # Directory handle registration (no-op cleanup)
            self._resource_manager.register(
                resource=self.log_dir,
                resource_type=ResourceType.DIRECTORY,
                description=f"Log directory for {self.name}",
                cleanup_handler=lambda p: None,
                tags={"logging", "directory"},
                path=str(self.log_dir),
            )
        except Exception:
            pass

        # Register handlers
        for h in list(self._logger.handlers):
            try:
                if isinstance(h, logging.handlers.RotatingFileHandler):
                    self._resource_manager.register(
                        resource=h,
                        resource_type=ResourceType.FILE_HANDLE,
                        description=f"RotatingFileHandler for {self.name}",
                        cleanup_handler=lambda handler: None,
                        file=getattr(h, "baseFilename", None),
                        tags={"logging", "file_handler"},
                    )
                elif isinstance(h, logging.StreamHandler):
                    self._resource_manager.register(
                        resource=h,
                        resource_type=ResourceType.CUSTOM,
                        description=f"Console handler for {self.name}",
                        cleanup_handler=lambda handler: None,
                        stream=str(getattr(h, "stream", "stderr")),
                        tags={"logging", "console_handler"},
                    )
            except Exception:
                continue
    
    def _install_exception_handler(self) -> None:
        """Install a global exception handler to log uncaught exceptions."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            # Don't log KeyboardInterrupt
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
                
            self._logger.critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
        
        # Set the global exception handler
        sys.excepthook = handle_exception
    
    def shutdown(self) -> None:
        """Safely shut down logging for this AppLogger instance.
        
        - Restores the default sys.excepthook
        - Flushes and closes all handlers, then removes them
        - Disables this logger to prevent further emissions
        - Clears module-level singleton flags so logging can be re-configured later
        - Never raises
        """
        try:
            # Best-effort restore of default exception hook
            try:
                sys.excepthook = sys.__excepthook__
            except Exception:
                pass

            # Close and remove handlers
            for h in list(self._logger.handlers):
                try:
                    try:
                        h.flush()
                    except Exception:
                        pass
                    h.close()
                except Exception:
                    pass
                try:
                    self._logger.removeHandler(h)
                except Exception:
                    pass

            # Disable this logger
            try:
                self._logger.disabled = True
            except Exception:
                pass

            # Clear module-level singleton so a fresh logger can be configured
            try:
                global _app_logger, _configured
                _app_logger = None
                _configured = False
            except Exception:
                pass
        except Exception:
            # Intentionally swallow any shutdown errors
            pass
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log an exception with stack trace."""
        self._log(logging.ERROR, message, exc_info=True, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal method to log a message with the specified level."""
        if not self._logger.isEnabledFor(level):
            return
            
        # Format the message with any additional context
        if kwargs:
            try:
                message = f"{message} - {json.dumps(kwargs, default=str)}"
            except (TypeError, ValueError):
                # One-time warning about JSON serialization failure; fall back to repr
                if not self._json_ctx_warned:
                    self._logger.warning("Logging context serialization failed; using repr() for context")
                    self._json_ctx_warned = True
                message = f"{message} - {repr(kwargs)}"
        
        # Log the message
        self._logger.log(level, message)


# --- Throttling & Dedup Helpers -------------------------------------------------

# Per-process simple throttle state
_throttle_state_lock = threading.Lock()
_throttle_last_emit: dict[str, float] = {}

# Per-process dedupe state: key -> (last_message, suppressed_count, last_time)
_dedupe_state_lock = threading.Lock()
_dedupe_state: dict[str, tuple[str, int, float]] = {}

def throttled(logger_fn, key: str, interval_ms: int):
    """
    Return a callable that logs messages using logger_fn at most once per interval_ms for the given key.

    Example:
        tdebug = throttled(logger.debug, "hotkeys:repeat", 500)
        tdebug("Repeat ignored for opacity_decrease")
    """
    interval_s = max(0.0, float(interval_ms) / 1000.0)
    k = str(key)

    def _emit(message: str) -> None:
        now = time.monotonic()
        with _throttle_state_lock:
            last = _throttle_last_emit.get(k, 0.0)
            if now - last < interval_s:
                return
            _throttle_last_emit[k] = now
        try:
            logger_fn(message)
        except Exception:
            # Never raise from logging helpers
            pass

    return _emit


def log_dedupe(logger_fn, key: str, window_ms: int):
    """
    Return a callable that suppresses identical consecutive messages within window_ms for the given key.
    When the window elapses or the message changes, emits a summary of suppressed repeats.

    Example:
        ddebug = log_dedupe(logger.debug, "qswitch:none", 2000)
        ddebug("No candidates for quickswitch")
    """
    window_s = max(0.0, float(window_ms) / 1000.0)
    k = str(key)

    def _emit(message: str) -> None:
        now = time.monotonic()
        with _dedupe_state_lock:
            last_msg, count, last_time = _dedupe_state.get(k, ("", 0, 0.0))
            if message == last_msg and (now - last_time) < window_s:
                _dedupe_state[k] = (last_msg, count + 1, last_time)
                return
            # If we had suppressed repeats for the previous message, emit a summary line first
            if count > 0 and last_msg:
                try:
                    logger_fn(f"{last_msg} [dedup suppressed {count} repeats]")
                except Exception:
                    pass
            # Emit current message and reset state
            _dedupe_state[k] = (message, 0, now)
        try:
            logger_fn(message)
        except Exception:
            pass

    return _emit
