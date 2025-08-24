"""
Window icon handling utilities.

This module provides functions for retrieving and managing window icons
in a platform-agnostic way.
"""

from typing import Optional, Dict
import os
import win32con
import win32gui
import win32process
import win32api
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtCore import QObject

# Local imports
from core.logging import get_logger

logger = get_logger(__name__)

class WindowIconManager(QObject):
    """Manages window icons including caching and retrieval."""
    
    # Class-level caches
    _icon_cache: Dict[str, QIcon] = {}
    _process_icon_cache: Dict[str, QIcon] = {}
    _blank_icon: Optional[QIcon] = None
    
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._init_blank_icon()
    
    @classmethod
    def _init_blank_icon(cls) -> None:
        """Initialize the blank fallback icon if not already done."""
        if cls._blank_icon is None:
            cls._blank_icon = QIcon()
    
    def get_window_icon(self, hwnd: int) -> QIcon:
        """
        Get the icon for a window handle.
        
        Args:
            hwnd: The window handle to get the icon for
            
        Returns:
            QIcon: The window icon or a blank icon if none found
        """
        # First try to get the window class for caching
        try:
            window_class = win32gui.GetClassName(hwnd)
        except Exception:
            window_class = str(hwnd)
        
        # Check cache first
        cache_key = f"{hwnd}:{window_class}"
        if cache_key in self._icon_cache:
            return self._icon_cache[cache_key]
        
        # Try standard methods first
        icon = self._get_window_icon(hwnd)
        if icon and not icon.isNull():
            self._icon_cache[cache_key] = icon
            return icon
        
        # Try alternative methods
        icon = self._get_alternative_window_icon(hwnd)
        if icon and not icon.isNull():
            self._icon_cache[cache_key] = icon
            return icon
        
        # Do NOT cache blanks; allow future refreshes to retry fetching
        logger.warning("Falling back to blank icon (not cached) for hwnd=%s (class=%s)", hwnd, window_class)
        return self._blank_icon or QIcon()
    
    def _get_window_icon(self, hwnd: int) -> QIcon:
        """Get window icon using safe methods with timeouts and class fallbacks.

        Important: Do not destroy HICONs returned by WM_GETICON or GetClassLong;
        ownership remains with the window/class.
        """
        try:
            def _get_icon_via_msg(icon_type: int, timeout_ms: int = 200) -> int:
                try:
                    # Use SendMessageTimeout to avoid hangs on hung/minimized windows
                    res = win32gui.SendMessageTimeout(
                        hwnd,
                        win32con.WM_GETICON,
                        icon_type,
                        0,
                        win32con.SMTO_ABORTIFHUNG | win32con.SMTO_NORMAL,
                        timeout_ms,
                    )
                    # res is a tuple (success, lresult)
                    if res and res[0]:
                        return int(res[1])
                except Exception as e:
                    logger.debug("SendMessageTimeout WM_GETICON failed: %s", e)
                return 0

            ICON_SMALL2 = getattr(win32con, "ICON_SMALL2", 2)
            icon_handle = _get_icon_via_msg(ICON_SMALL2)
            if not icon_handle:
                icon_handle = _get_icon_via_msg(win32con.ICON_SMALL)
            if not icon_handle:
                icon_handle = _get_icon_via_msg(win32con.ICON_BIG)

            # Fallback to class icons; try both 32/64-bit APIs if available
            if not icon_handle:
                try:
                    icon_handle = win32gui.GetClassLong(hwnd, win32con.GCL_HICONSM)
                except Exception:
                    icon_handle = 0
            if not icon_handle:
                try:
                    icon_handle = win32gui.GetClassLong(hwnd, win32con.GCL_HICON)
                except Exception:
                    icon_handle = 0
            # Try pointer-sized variant if available (on 64-bit)
            if not icon_handle:
                try:
                    get_ptr = getattr(win32gui, 'GetClassLongPtr', None)
                    if get_ptr is not None:
                        icon_handle = get_ptr(hwnd, win32con.GCLP_HICONSM)
                        if not icon_handle:
                            icon_handle = get_ptr(hwnd, win32con.GCLP_HICON)
                except Exception:
                    icon_handle = 0

            if icon_handle:
                try:
                    img = QImage.fromHICON(icon_handle)
                    if not img.isNull():
                        return QIcon(QPixmap.fromImage(img))
                except Exception as e:
                    logger.debug("Failed to create icon from handle: %s", e)
        except Exception as e:
            logger.debug("Error in _get_window_icon: %s", e)
        
        return QIcon()
    
    def _get_alternative_window_icon(self, hwnd: int) -> QIcon:
        """Try alternative methods to get window icon.

        - Extract from process executable using QIcon(exe) first, then ExtractIconEx.
        - Cache by exe path to avoid repeated extraction.
        """
        try:
            # Get process ID
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if not pid:
                return QIcon()
            
            process = None
            try:
                process = win32api.OpenProcess(
                    win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                    False,
                    pid,
                )
                if not process:
                    return QIcon()
                exe_path = win32process.GetModuleFileNameEx(process, 0)
                if not exe_path or not os.path.exists(exe_path):
                    return QIcon()

                # Process cache
                if exe_path in self._process_icon_cache:
                    return self._process_icon_cache[exe_path]

                # First try QIcon(exe_path) which asks the shell for the file's icon
                icon = QIcon(exe_path)
                if icon and not icon.isNull():
                    self._process_icon_cache[exe_path] = icon
                    return icon

                # Fallback: Extract icon handles from EXE resources
                try:
                    # Extract large then small
                    large_icons, small_icons = win32gui.ExtractIconEx(exe_path, 0)
                    hicon = 0
                    if large_icons and len(large_icons) > 0:
                        hicon = large_icons[0]
                    elif small_icons and len(small_icons) > 0:
                        hicon = small_icons[0]
                    if hicon:
                        img = QImage.fromHICON(hicon)
                        # We created the HICON; destroy it after conversion
                        try:
                            win32gui.DestroyIcon(hicon)
                        except Exception:
                            pass
                        if not img.isNull():
                            icon = QIcon(QPixmap.fromImage(img))
                            self._process_icon_cache[exe_path] = icon
                            return icon
                except Exception as e:
                    logger.debug("ExtractIconEx failed: %s", e)
            except Exception as e:
                logger.debug("Error getting alternative icon: %s", e)
                return QIcon()
            finally:
                if process:
                    try:
                        win32api.CloseHandle(process)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("Error in _get_alternative_window_icon: %s", e)
        
        return QIcon()
    
    def clear_cache(self) -> None:
        """Clear all icon caches."""
        self._icon_cache.clear()
        self._process_icon_cache.clear()
