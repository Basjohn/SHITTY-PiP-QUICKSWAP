"""
Debug Log Suppression System - Reduces excessive debug logging while maintaining important messages.

This module provides centralized debug log filtering and suppression to prevent spam
while preserving error and warning messages for diagnostics.
"""
import time
from typing import Dict, Set, Optional
from threading import Lock
from core.logging import get_logger


class LogSuppressor:
    """Centralized debug log suppression with repeat filtering and rate limiting."""
    
    def __init__(self, suppress_debug: bool = True, repeat_window: float = 5.0):
        """
        Initialize log suppressor.
        
        Args:
            suppress_debug: If True, suppress debug messages entirely
            repeat_window: Time window in seconds to suppress repeated messages
        """
        self._suppress_debug = suppress_debug
        self._repeat_window = repeat_window
        self._message_timestamps: Dict[str, float] = {}
        self._suppressed_messages: Set[str] = set()
        self._lock = Lock()
        self._logger = get_logger(__name__)
    
    def should_log_debug(self, message: str, source: str = "") -> bool:
        """
        Check if a debug message should be logged.
        
        Args:
            message: The debug message content
            source: Optional source identifier for the message
            
        Returns:
            True if message should be logged, False if suppressed
        """
        if not self._suppress_debug:
            return True
            
        # Always suppress debug messages in suppression mode
        return False
    
    def should_log_repeated(self, message: str, source: str = "") -> bool:
        """
        Check if a repeated message should be logged based on time window.
        
        Args:
            message: The message content
            source: Optional source identifier
            
        Returns:
            True if message should be logged, False if suppressed as repeat
        """
        message_key = f"{source}:{message}" if source else message
        current_time = time.time()
        
        with self._lock:
            last_time = self._message_timestamps.get(message_key, 0)
            
            if current_time - last_time >= self._repeat_window:
                # Allow message and update timestamp
                self._message_timestamps[message_key] = current_time
                if message_key in self._suppressed_messages:
                    self._suppressed_messages.remove(message_key)
                return True
            else:
                # Suppress repeated message
                if message_key not in self._suppressed_messages:
                    self._suppressed_messages.add(message_key)
                    # Log suppression notice once
                    self._logger.info(f"Suppressing repeated message from {source or 'unknown'}")
                return False
    
    def log_debug_if_allowed(self, logger, message: str, source: str = ""):
        """
        Log debug message only if not suppressed.
        
        Args:
            logger: Logger instance to use
            message: Debug message content
            source: Optional source identifier
        """
        if self.should_log_debug(message, source):
            logger.debug(message)
    
    def log_with_repeat_check(self, logger, level: str, message: str, source: str = ""):
        """
        Log message with repeat suppression check.
        
        Args:
            logger: Logger instance to use
            level: Log level (debug, info, warning, error)
            message: Message content
            source: Optional source identifier
        """
        if level == "debug" and not self.should_log_debug(message, source):
            return
            
        if not self.should_log_repeated(message, source):
            return
            
        # Log the message at appropriate level
        log_method = getattr(logger, level, logger.info)
        log_method(message)
    
    def clear_suppression_cache(self):
        """Clear the suppression cache to allow fresh logging."""
        with self._lock:
            self._message_timestamps.clear()
            self._suppressed_messages.clear()
    
    def set_debug_suppression(self, suppress: bool):
        """Enable or disable debug suppression."""
        self._suppress_debug = suppress
        if not suppress:
            self.clear_suppression_cache()


# Global log suppressor instance
_global_suppressor: Optional[LogSuppressor] = None


def get_log_suppressor() -> LogSuppressor:
    """Get the global log suppressor instance."""
    global _global_suppressor
    if _global_suppressor is None:
        _global_suppressor = LogSuppressor(suppress_debug=True, repeat_window=5.0)
    return _global_suppressor


def suppress_debug_log(logger, message: str, source: str = ""):
    """Convenience function to log debug message with suppression."""
    suppressor = get_log_suppressor()
    suppressor.log_debug_if_allowed(logger, message, source)


def log_with_suppression(logger, level: str, message: str, source: str = ""):
    """Convenience function to log with repeat suppression."""
    suppressor = get_log_suppressor()
    suppressor.log_with_repeat_check(logger, level, message, source)
