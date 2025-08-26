from __future__ import annotations

import threading
import time
from typing import Optional

from PySide6.QtCore import QObject

from core.logging import get_logger
from core.switching.mru_manager import get_mru_manager
from utils.window_validation import is_valid_window, get_window_title
import win32gui


class FocusTracker(QObject):
    """
    Universal focus tracking system that always records MRU changes.
    
    This runs independently of autoswitch and ensures MRU is always populated
    with recent focus changes for quickswitch functionality.
    
    - Polls foreground window every 200ms
    - Records valid focus changes to MRU immediately
    - Debounces rapid changes (100ms stability required)
    - Always active regardless of autoswitch settings
    """

    _instance: Optional["FocusTracker"] = None
    _lock = threading.Lock()

    POLL_INTERVAL_MS = 200
    STABLE_DEBOUNCE_MS = 100

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        super().__init__()
        self._logger = get_logger("FOCUS_TRACKER")
        
        # Focus tracking state
        self._last_seen_hwnd: Optional[int] = None
        self._candidate_hwnd: Optional[int] = None
        self._candidate_since: float = 0.0
        
        # Polling state
        self._polling_active = False
        
        # Start tracking immediately
        self._start_tracking()
        
        self._initialized = True
        self._logger.debug("Initialized FocusTracker")

    def _start_tracking(self) -> None:
        """Start focus tracking."""
        if not self._polling_active:
            self._polling_active = True
            from core.threading import ThreadManager
            ThreadManager.single_shot(self.POLL_INTERVAL_MS, self._poll_tick)
            self._logger.debug(f"Started focus tracking (poll interval: {self.POLL_INTERVAL_MS}ms)")

    def _stop_tracking(self) -> None:
        """Stop focus tracking."""
        if self._polling_active:
            self._polling_active = False
            self._logger.debug("Stopped focus tracking")

    def _poll_tick(self) -> None:
        """Self-rescheduling polling tick using ThreadManager.single_shot."""
        if not self._polling_active:
            return
            
        try:
            self._poll_foreground()
        except Exception as e:
            self._logger.error(f"Focus tracker poll tick failed: {e}", exc_info=True)
        finally:
            # Schedule next tick if still active
            if self._polling_active:
                from core.threading import ThreadManager
                ThreadManager.single_shot(self.POLL_INTERVAL_MS, self._poll_tick)

    def _poll_foreground(self) -> None:
        """Poll current foreground window and record stable changes."""
        try:
            # Get current foreground window with error handling
            try:
                hwnd = win32gui.GetForegroundWindow()
            except Exception:
                return
            
            if not hwnd:
                return
                
            # Skip if not a valid window for MRU tracking
            try:
                if not is_valid_window(hwnd):
                    return
            except Exception:
                return
                
            now = time.time() * 1000.0
            
            # Thread-safe state updates
            with self._lock:
                # Check if this is a new candidate
                if hwnd != self._candidate_hwnd:
                    self._candidate_hwnd = hwnd
                    self._candidate_since = now
                    return
                    
                # Check if candidate has been stable long enough
                if now - self._candidate_since < self.STABLE_DEBOUNCE_MS:
                    return
                    
                # Check if this is actually a change from last recorded
                if hwnd == self._last_seen_hwnd:
                    return
                    
                # Update last seen before releasing lock
                self._last_seen_hwnd = hwnd
                
            # Record the stable focus change (outside lock to avoid deadlock)
            try:
                get_mru_manager().record(hwnd)
                
                # Log focus change with reduced frequency
                if not hasattr(self, '_last_log_time'):
                    self._last_log_time = 0
                if now - self._last_log_time > 1000:  # Log at most once per second
                    try:
                        title = get_window_title(hwnd)
                    except Exception:
                        title = ""
                    self._logger.debug(f"Focus change recorded: hwnd={hwnd} title='{title}'")
                    self._last_log_time = now
                
            except Exception as e:
                self._logger.debug(f"Failed to record focus change: {e}")
                
        except Exception as e:
            self._logger.debug(f"Focus polling error: {e}")

    def get_current_focus(self) -> Optional[int]:
        """Get the current foreground window handle."""
        try:
            hwnd = win32gui.GetForegroundWindow()
            return hwnd if hwnd and is_valid_window(hwnd) else None
        except Exception:
            return None


# Convenience accessor
_def_instance: Optional[FocusTracker] = None

def get_focus_tracker() -> FocusTracker:
    global _def_instance
    if _def_instance is None:
        _def_instance = FocusTracker()
    return _def_instance
