from __future__ import annotations

import threading
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
    _lock = threading.Lock()

    CAPACITY_DEFAULT = 7

    def __new__(cls):
        with cls._lock:
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
        self._mtx = threading.RLock()
        self._pid = os.getpid()
        self._initialized = True
        self._logger.debug("Initialized MRUManager")

    def set_capacity(self, capacity: int) -> None:
        with self._mtx:
            self._cap = max(1, int(capacity))
            if len(self._mru) > self._cap:
                self._mru = self._mru[: self._cap]

    def clear(self) -> None:
        with self._mtx:
            self._mru.clear()
            self._logger.debug("MRU cleared")

    def remove(self, hwnd: int) -> None:
        with self._mtx:
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
        with self._mtx:
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
            return True

    def get_recent(self, limit: Optional[int] = None) -> List[int]:
        """
        Return a most-recent-first list of valid hwnds (purging any that are no longer valid).
        """
        lim = self._cap if limit is None else max(0, int(limit))
        with self._mtx:
            # Purge invalid entries
            purged = False
            valid_list: List[int] = []
            for hwnd in self._mru:
                try:
                    if is_valid_window(hwnd, our_pid=self._pid):
                        valid_list.append(hwnd)
                    else:
                        purged = True
                except Exception:
                    purged = True
            if purged:
                # Rebuild MRU from filtered list
                self._mru = valid_list[:]
            return valid_list[:lim]


def get_mru_manager() -> MRUManager:
    return MRUManager()
