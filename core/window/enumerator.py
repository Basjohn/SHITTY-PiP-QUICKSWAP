"""
Window enumeration and management.

This module provides the WindowEnumerator class which handles the enumeration
of windows, icon retrieval, and window validation for the application.
"""
from __future__ import annotations

import ctypes
import logging
import os
import time
from typing import Any, List, Optional, Tuple

import win32api
import win32con
import win32gui
import win32process
from PySide6.QtCore import QObject
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtWidgets import QApplication

from core.logging import get_logger
from core.window.icons import WindowIconManager

logger = get_logger(__name__)

class WindowEnumerator(QObject):
    """A class to enumerate windows and manage window-related operations."""
    
    _blank_icon: Optional[QIcon] = None
    
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.app_instance = QApplication.instance()
        self.last_window_list: List[Tuple[int, str]] = []
        self.last_refresh_time: float = 0.0
        self.icon_manager = WindowIconManager(self)
        
        if WindowEnumerator._blank_icon is None:
            self._init_blank_icon()
    
    @classmethod
    def _init_blank_icon(cls) -> None:
        """Initialize the blank fallback icon."""
        cls._blank_icon = QIcon()
        # Resource-only to avoid filesystem fallbacks and warnings
        path = ":/icons/Blank.ico"
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            cls._blank_icon = QIcon(pixmap)
            logger.debug("Loaded fallback icon from: %s", path)
        else:
            logger.warning("Failed to load Blank.ico from Qt resources at %s", path)
    
    def _is_valid_window(self, hwnd: int, check_visible: bool = True) -> bool:
        """Check if a window is valid for inclusion in the window list."""
        from core.window.validation import is_valid_window as validate_window
        
        try:
            if not hwnd or hwnd == 0:
                return False
                
            result = validate_window(hwnd, os.getpid(), check_visible=check_visible)
            
            if not result and logger.isEnabledFor(logging.DEBUG):  # type: ignore
                try:
                    title = win32gui.GetWindowText(hwnd)
                    class_name = win32gui.GetClassName(hwnd)
                    logger.debug("Window %d (Title: %s, Class: %s): Excluded by validation checks", 
                                hwnd, title, class_name)
                except Exception as e:
                    logger.debug("Window %d: Excluded by validation checks (error: %s)", 
                                hwnd, e)
            
            return result
            
        except Exception as e:
            logger.error("Error validating window %d: %s", hwnd, e, exc_info=True)
            return False

    def refresh_window_list(self, force: bool = False) -> List[Tuple[int, str]]:
        """Refresh the cached list of windows."""
        current_time = time.time()
        if force or current_time - self.last_refresh_time > 2:
            self.last_window_list = self.enum_windows()
            self.last_refresh_time = current_time
        return self.last_window_list
    
    @classmethod
    def enum_windows(cls) -> List[Tuple[int, str]]:
        """Enumerate all top-level windows."""
        windows = []
        our_pid = os.getpid()
        our_titles = {
            'Shitty PiP QuickSwap', 'Settings', 'Sub-settings',
            'Overlay', 'Monitor Overlay', 'Window Overlay'
        }
        
        # Per-pass cache to avoid repeated process queries for the same PID
        pid_cache: dict[int, Optional[str]] = {}
        
        # Use limited rights and QueryFullProcessImageNameW to avoid VM_READ
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        
        def _query_image_path(pid: int) -> Optional[str]:
            """Best-effort query for a process image path without VM_READ."""
            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return None
            try:
                buf_len = ctypes.wintypes.DWORD(32768)
                buf = ctypes.create_unicode_buffer(buf_len.value)
                if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(buf_len)):
                    return buf.value.lower()
                return None
            finally:
                kernel32.CloseHandle(h)
        
        video_extensions = ['.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        media_player_processes = ['mpv', 'vlc', 'mpc', 'potplayer', 'wmplayer', 'kodi']
        media_player_classes = ['Qt5QWindowIcon', 'QWidget', 'mpv', 'VLC', 'MediaPlayerClassicW']
        
        def enum_windows_callback(hwnd: int, _: Any) -> bool:
            try:
                if not ctypes.windll.user32.IsWindowVisible(hwnd):
                    return True
                    
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
                title = ctypes.create_unicode_buffer(length)
                ctypes.windll.user32.GetWindowTextW(hwnd, title, length)
                title = title.value.strip()
                
                if not title or any(our_title in title for our_title in our_titles):
                    return True
                    
                class_name = win32gui.GetClassName(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                
                if pid == our_pid:
                    return True
                    
                # Skip system windows
                system_windows = [
                    'windows input experience', 'text input application',
                    'searchui', 'shellexperiencehost', 'applicationframehost'
                ]
                # Early title-based skip to avoid protected/OpenProcess calls when unnecessary
                if any(sys_win in title.lower() for sys_win in system_windows):
                    return True

                # Best-effort process image path using limited rights; cached per pass
                process_name = pid_cache.get(pid)
                if process_name is None:
                    img = _query_image_path(pid)
                    process_name = img or ""
                    pid_cache[pid] = process_name
                    if not img:
                        logger.debug("Process image unavailable for hwnd=%s pid=%s (likely protected/elevated).", hwnd, pid)

                if any(sys_win in process_name or sys_win in title.lower() 
                      for sys_win in system_windows):
                    return True
                
                # Media player detection
                is_media_player = any(mp in process_name for mp in media_player_processes)
                is_media_player = is_media_player or any(
                    mp_class.lower() in class_name.lower() 
                    for mp_class in media_player_classes
                )
                is_media_player = is_media_player or any(
                    ext in title.lower() 
                    for ext in video_extensions
                )
                
                if (class_name in ['Qt5QWindowIcon', 'QWidget'] and 
                    not is_media_player and class_name != 'Progman'):
                    return True
                
                rect = (ctypes.c_int * 4)()
                if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    if width > 0 and height > 0:
                        windows.append((hwnd, title))
                
            except Exception as e:
                logger.debug("Error in enum_windows_callback: %s", e)
                
            return True
            
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        ctypes.windll.user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)
        logger.debug("Enumerated %d windows.", len(windows))
        return windows
    
    def get_capturable_windows_with_icons(self) -> List[Tuple[int, str, QIcon]]:
        """Get a list of capturable windows with their icons."""
        windows = []
        window_list = self.refresh_window_list(True)
        
        progman_hwnd = self._get_desktop_window()
        if progman_hwnd:
            windows.append((progman_hwnd, "Desktop", self.icon_manager.get_window_icon(progman_hwnd)))
        
        for i, (hwnd, title) in enumerate(window_list):
            try:
                if not self._is_valid_window(hwnd) or hwnd == progman_hwnd:
                    continue
                
                window_class = win32gui.GetClassName(hwnd)
                if window_class == "Progman" and title == "Program Manager":
                    continue
                
                icon = self.icon_manager.get_window_icon(hwnd)
                windows.append((hwnd, title, icon))
                
                if i > 0 and i % 10 == 0:
                    QApplication.processEvents()
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error("Error processing window %d: %s", hwnd, e)
                windows.append((hwnd, title, self.icon_manager.get_window_icon(hwnd)))
        
        return self.sort_windows(windows)
    
    def _get_desktop_window(self) -> Optional[int]:
        """Get the handle for the desktop window."""
        try:
            hwnd = win32gui.FindWindow("Progman", "Program Manager")
            return hwnd or win32gui.GetDesktopWindow()
        except Exception as e:
            logger.error("Error getting desktop window: %s", e)
            return None
    
    def _get_desktop_icon(self) -> QIcon:
        """Get the icon for the desktop."""
        try:
            shell32 = ctypes.windll.shell32
            hicon = shell32.ExtractIconW(0, "shell32.dll", 15)
            if hicon:
                try:
                    pixmap = QPixmap.fromImage(QImage.fromHICON(hicon))
                    return QIcon(pixmap)
                finally:
                    ctypes.windll.user32.DestroyIcon(hicon)
        except Exception as e:
            logger.error("Error getting desktop icon: %s", e)
        return self.icon_manager.get_window_icon(0)

    def sort_windows(self, windows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
        """Sort windows according to the current sort order."""
        if not windows:
            return []
            
        desktop_window = None
        filtered_windows = []
        
        for window in windows:
            if len(window) >= 2 and window[1] == "Desktop":
                desktop_window = window
            else:
                filtered_windows.append(window)
        
        if not hasattr(self, 'app_instance') or not hasattr(self.app_instance, 'window_sort_order'):
            sorted_windows = sorted(filtered_windows, key=lambda w: w[1].lower())
        else:
            sort_order = self.app_instance.window_sort_order
            if sort_order == "Alphabetical":
                sorted_windows = sorted(filtered_windows, key=lambda w: w[1].lower())
            elif sort_order == "Most Recently Active" and hasattr(self.app_instance, 'mru_hwnds'):
                mru_hwnds_list = self.app_instance.mru_hwnds
                if mru_hwnds_list:
                    mru_set = set(mru_hwnds_list)
                    mru_present = []
                    other_windows = []
                    for w_info in filtered_windows:
                        if w_info[0] in mru_set:
                            mru_present.append(w_info)
                        else:
                            other_windows.append(w_info)
                    mru_order = {hwnd: i for i, hwnd in enumerate(mru_hwnds_list)}
                    mru_present.sort(key=lambda w: mru_order.get(w[0], float('inf')))
                    sorted_windows = mru_present + other_windows
                else:
                    sorted_windows = filtered_windows
            else:
                sorted_windows = sorted(filtered_windows, key=lambda w: w[1].lower())
        
        if desktop_window is not None:
            sorted_windows.append(desktop_window)
            
        return sorted_windows
