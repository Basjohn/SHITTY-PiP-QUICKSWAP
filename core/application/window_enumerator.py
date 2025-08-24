"""
Window enumeration functionality for the application.

This module provides the WindowEnumerator class which is responsible for
enumerating and managing windows in the system.
"""
from __future__ import annotations

import os
import time
import win32gui
import win32con
import win32process
import win32api
from typing import List, Optional, Tuple

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtCore import Qt

from utils.window_validation import is_valid_window, SYSTEM_WINDOW_TITLES, SYSTEM_WINDOW_CLASSES
from core.logging import get_logger
from core.window.icons import WindowIconManager

logger = get_logger(__name__)

class WindowEnumerator:
    """Handles window enumeration and management."""
    
    # Class-level caches for icons
    _icon_cache = {}
    _process_icon_cache = {}
    _blank_icon = None
    
    def __init__(self):
        """Initialize the WindowEnumerator."""
        self.app_instance = QApplication.instance()
        self.last_window_list = []
        self.last_refresh_time = 0
        self._icon_manager = WindowIconManager()
        
        # Initialize the blank icon only if a QApplication exists; otherwise defer
        if WindowEnumerator._blank_icon is None:
            if self.app_instance is not None:
                self._init_blank_icon()
            else:
                logger.debug("WindowEnumerator: QApplication not ready; deferring blank icon init")
    
    @classmethod
    def _init_blank_icon(cls) -> None:
        """Initialize a blank icon using resources/Blank.ico when possible.

        Requires a QApplication instance for reliable QIcon/QPixmap usage. If the
        application is not yet initialized, defer by setting an empty QIcon and
        let the caller re-attempt after QApplication exists.
        """
        try:
            if QApplication.instance() is None:
                logger.debug("Blank icon init requested before QApplication; deferring (empty QIcon)")
                cls._blank_icon = QIcon()
                return

            # Attempt to load explicit Blank.ico from the resources folder
            # Project structure: <root>/resources/Blank.ico
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            ico_path = os.path.join(base_dir, 'resources', 'Blank.ico')

            icon_loaded = False
            try:
                if os.path.exists(ico_path):
                    icon_candidate = QIcon(ico_path)
                    # Ensure we have a non-null icon
                    if not icon_candidate.isNull():
                        cls._blank_icon = icon_candidate
                        icon_loaded = True
                        logger.debug(f"Loaded Blank.ico from resources: {ico_path}")
            except Exception as e:
                logger.debug(f"Attempt to load Blank.ico failed: {e}")

            if not icon_loaded:
                # Fallback: generate a small transparent pixmap to avoid null
                pixmap = QPixmap(16, 16)
                pixmap.fill(Qt.transparent)
                cls._blank_icon = QIcon(pixmap)
                logger.warning("Blank.ico not found or invalid; using generated transparent icon")
        except Exception as e:
            logger.error(f"Failed to initialize blank icon: {e}")
            cls._blank_icon = QIcon()
    
    def refresh_window_list(self, force: bool = False) -> List[Tuple[int, str, QIcon]]:
        """Refresh the list of available windows.
        
        Args:
            force: If True, force a refresh even if the cache is still valid
            
        Returns:
            List of tuples containing (hwnd, title, icon) for each window
        """
        current_time = time.time()
        if not force and (current_time - self.last_refresh_time) < 1.0:  # 1 second cache
            return self.last_window_list
            
        try:
            windows = []
            for hwnd in self.enum_windows():
                try:
                    if not is_valid_window(hwnd):
                        continue
                        
                    title = win32gui.GetWindowText(hwnd)
                    window_class = win32gui.GetClassName(hwnd)
                    
                    # Skip system windows
                    if title in SYSTEM_WINDOW_TITLES or window_class in SYSTEM_WINDOW_CLASSES:
                        continue
                        
                    # Get window icon
                    icon = self._get_cached_window_icon(hwnd, window_class)
                    
                    windows.append((hwnd, title, icon))
                    
                except Exception as e:
                    logger.debug(f"Error processing window {hwnd}: {e}")
                    continue
            
            self.last_window_list = windows
            self.last_refresh_time = current_time
            return windows
            
        except Exception as e:
            logger.error(f"Error refreshing window list: {e}")
            return []
    
    @classmethod
    def enum_windows(cls) -> List[int]:
        """Enumerate all top-level windows.
        
        Returns:
            List of window handles
        """
        windows = []
        
        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                windows.append(hwnd)
            return True
            
        win32gui.EnumWindows(callback, None)
        return windows
    
    def get_capturable_windows_with_icons(self) -> List[Tuple[int, str, QIcon]]:
        """Get a list of capturable windows with their icons.
        
        Returns:
            List of tuples (hwnd, title, icon) for each window
        """
        return self.refresh_window_list()
    
    def _get_cached_window_icon(self, hwnd: int, window_class: str) -> QIcon:
        """Get a window icon from cache or load it if not cached.
        
        Args:
            hwnd: Window handle
            window_class: Window class name
            
        Returns:
            QIcon: The window icon or a blank icon if not available
        """
        # Try to get from cache first
        if hwnd in self._icon_cache:
            return self._icon_cache[hwnd]

        # Try to get from process cache
        process_key = self._get_process_icon_key(hwnd)
        if process_key in self._process_icon_cache:
            icon = self._process_icon_cache[process_key]
            # Cache per-hwnd for faster next lookups
            if icon is not None and (not hasattr(icon, 'isNull') or not icon.isNull()):
                self._icon_cache[hwnd] = icon
            return icon

        # Use centralized manager to fetch icon with robust fallbacks
        icon = self._icon_manager.get_window_icon(hwnd)

        # Ensure a non-null icon is returned for UI, but DO NOT cache blanks
        if icon is None or (hasattr(icon, 'isNull') and icon.isNull()):
            if WindowEnumerator._blank_icon is None:
                # Lazy init; handles both with and without QApplication
                self._init_blank_icon()
            return WindowEnumerator._blank_icon

        # Valid icon: cache by hwnd and process
        self._icon_cache[hwnd] = icon
        if process_key:
            self._process_icon_cache[process_key] = icon

        return icon
    
    def _get_process_icon_key(self, hwnd: int) -> Optional[str]:
        """Get a cache key based on the window's process.
        
        Args:
            hwnd: Window handle
            
        Returns:
            Process key or None if not available
        """
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
            if process:
                exe_path = win32process.GetModuleFileNameEx(process, 0)
                return exe_path.lower()
        except Exception as e:
            logger.debug(f"Error getting process key: {e}")
        return None
    
    def _get_window_icon(self, hwnd: int) -> Optional[QIcon]:
        """Get window icon using standard methods.
        
        Args:
            hwnd: Window handle
            
        Returns:
            QIcon or None if no icon found
        """
        try:
            # Try to get the icon using standard Windows APIs
            ICON_SMALL2 = getattr(win32con, "ICON_SMALL2", 2)
            hicon = win32gui.SendMessage(hwnd, win32con.WM_GETICON, ICON_SMALL2, 0)
            if not hicon:
                hicon = win32gui.SendMessage(hwnd, win32con.WM_GETICON, win32con.ICON_SMALL, 0)
            if not hicon:
                hicon = win32gui.SendMessage(hwnd, win32con.WM_GETICON, win32con.ICON_BIG, 0)
                
            if hicon:
                # Convert HICON to QIcon
                pixmap = QPixmap.fromImage(QImage.fromHICON(hicon))
                win32gui.DestroyIcon(hicon)
                return QIcon(pixmap)
                
            # Fall back to alternative methods if standard methods fail
            return self._get_alternative_window_icon(hwnd)
            
        except Exception as e:
            logger.debug(f"Error getting window icon: {e}")
            return None
    
    def _get_alternative_window_icon(self, hwnd: int) -> Optional[QIcon]:
        """Try alternative methods to get window icon.
        
        Args:
            hwnd: Window handle
            
        Returns:
            QIcon or None if no icon found
        """
        try:
            # Try to get the icon from the window class
            hicon = win32gui.GetClassLong(hwnd, win32con.GCL_HICONSM)
            if not hicon:
                hicon = win32gui.GetClassLong(hwnd, win32con.GCL_HICON)
                
            if hicon:
                # Convert HICON to QIcon
                pixmap = QPixmap.fromImage(QImage.fromHICON(hicon))
                return QIcon(pixmap)
                
            # As a last resort, try to get the icon from the process
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
            if process:
                exe_path = win32process.GetModuleFileNameEx(process, 0)
                if os.path.exists(exe_path):
                    return QIcon(exe_path)
                    
        except Exception as e:
            logger.debug(f"Error in alternative icon method: {e}")
            
        return None

    def get_window_icon(self, hwnd: int) -> Optional[QIcon]:
        """Public helper to get the best available icon for a window.
        
        This wraps the internal cached icon retrieval so other modules do not
        need to rely on private methods. Returns None if no icon is available.
        Gracefully handles errors and logs at debug level.
        
        Args:
            hwnd: Window handle
        Returns:
            QIcon or None
        """
        try:
            class_name = win32gui.GetClassName(hwnd)
            icon = self._get_cached_window_icon(hwnd, class_name)
            # Ensure a non-null icon is returned (use Blank.ico or generated)
            if icon is None or (hasattr(icon, 'isNull') and icon.isNull()):
                if WindowEnumerator._blank_icon is None:
                    self._init_blank_icon()
                return WindowEnumerator._blank_icon
            return icon
        except Exception as e:
            logger.debug(f"get_window_icon error for hwnd {hwnd}: {e}")
            # On error, prefer a safe blank icon to keep UI consistent
            try:
                if WindowEnumerator._blank_icon is None:
                    self._init_blank_icon()
                return WindowEnumerator._blank_icon
            except Exception:
                return None
