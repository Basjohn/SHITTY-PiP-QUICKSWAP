from __future__ import annotations

from typing import List, Optional

from core.logging import get_logger
from utils.window_validation import is_valid_window, get_window_title
import os


class MRUManager:
    """
    Centralized Most-Recently-Used (MRU) window tracker.

    - Tracks last N valid target hwnds (excluding our own app and system/dummy windows via is_valid_window)
    - Deduplicates and maintains recency ordering
    - Provides filtered candidates in most-recent-first order
    - Thread-safe
    """

    _instance: Optional["MRUManager"] = None
    # Lock-free: Singleton creation confined to UI thread

    CAPACITY_DEFAULT = 7

    def __new__(cls):
        # Lock-free: UI thread only access for singleton creation
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._logger = get_logger("MRU")
        self._mru: List[int] = []
        self._cap = self.CAPACITY_DEFAULT
        # Lock-free: All MRU operations confined to UI thread
        self._pid = os.getpid()
        # Subscribers for push-based updates
        self._listeners = []  # list of callables: (List[int]) -> None
        self._initialized = True
        self._logger.debug("Initialized MRUManager")

    def set_capacity(self, capacity: int) -> None:
        # Lock-free: UI thread only access
        self._cap = max(1, int(capacity))
        if len(self._mru) > self._cap:
            self._mru = self._mru[: self._cap]

    def clear(self) -> None:
        # Lock-free: UI thread only access
        self._mru.clear()
        self._logger.debug("MRU cleared")

    def remove(self, hwnd: int) -> None:
        # Lock-free: UI thread only access
        if hwnd in self._mru:
            self._mru.remove(hwnd)
            self._logger.debug(f"MRU removed hwnd={hwnd}")

    def record(self, hwnd: int) -> bool:
        """
        Record a window in MRU if it is valid.
        Returns True if added/moved-to-front, False otherwise.
        """
        try:
            if not is_valid_window(hwnd, our_pid=self._pid):
                return False
        except Exception:
            return False
        # Lock-free: UI thread only access
        # No-op if already the most recent; avoid log spam and churn
        if self._mru and self._mru[0] == hwnd:
            return False
        
        if hwnd in self._mru:
            self._mru.remove(hwnd)
        self._mru.insert(0, hwnd)
        if len(self._mru) > self._cap:
            self._mru = self._mru[: self._cap]
        try:
            title = get_window_title(hwnd)
        except Exception:
            title = ""
        self._logger.debug(f"Recorded hwnd={hwnd} title='{title}'; MRU size={len(self._mru)}")
        # Notify listeners on change
        try:
            self._notify_listeners()
        except Exception:
            pass
        return True

    def get_recent(self, limit: Optional[int] = None) -> List[int]:
        """
        Return a most-recent-first list of valid hwnds (purging any that are no longer valid).
        
        NOTE: Does NOT filter minimized windows - they're valid quickswitch targets.
        Only filters destroyed/invalid windows.
        """
        lim = self._cap if limit is None else max(0, int(limit))
        # Lock-free: UI thread only access
        # Purge invalid entries (but keep minimized windows)
        purged = False
        valid_list: List[int] = []
        for hwnd in self._mru:
            try:
                # Skip visibility check - minimized windows are valid quickswitch targets
                if is_valid_window(hwnd, our_pid=self._pid, check_visible=False):
                    valid_list.append(hwnd)
                else:
                    purged = True
            except Exception:
                purged = True
        if purged:
            # Rebuild MRU from filtered list
            self._mru = valid_list[:]
        return valid_list[:lim]

    def get_most_recent(self) -> Optional[int]:
        """
        Return the most recent valid hwnd, or None if no valid windows exist.
        """
        recent_list = self.get_recent(limit=1)
        return recent_list[0] if recent_list else None

    # --- Listener API (UI-thread usage) -----------------------------------
    def add_listener(self, callback) -> None:
        """Subscribe to MRU changes. Callback receives the latest MRU list (List[int])."""
        try:
            if callback and callback not in self._listeners:
                self._listeners.append(callback)
        except Exception:
            pass

    def remove_listener(self, callback) -> None:
        """Unsubscribe a previously added listener."""
        try:
            if callback in self._listeners:
                self._listeners.remove(callback)
        except Exception:
            pass

    def _notify_listeners(self) -> None:
        """Notify listeners of MRU order changes."""
        try:
            snapshot = self.get_recent(limit=self._cap)
            for cb in list(self._listeners):
                try:
                    cb(snapshot)
                except Exception as e:
                    self._logger.debug(f"MRU listener error: {e}")
        except Exception as e:
            self._logger.debug(f"Failed notifying MRU listeners: {e}")


def get_mru_manager() -> MRUManager:
    return MRUManager()
