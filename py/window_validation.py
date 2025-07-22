"""
Window validation utilities for the SPQ application.

This module provides centralized window validation logic to determine
which windows should be included in window switching and overlay operations.
"""

import os
import ctypes
import ctypes.wintypes
from typing import Optional, Dict, Any

# Windows API constants and types
GWL_STYLE = -16
WS_VISIBLE = 0x10000000
WS_MINIMIZE = 0x20000000

# System window titles to exclude
SYSTEM_WINDOW_TITLES = frozenset({
    'Program Manager', 'Windows Input Experience', 'Text Input Application',
    'Search', 'Start', 'Settings', 'Action Center', 'Notification Center',
    'Cortana', 'Task View', 'Taskbar', 'Windows Shell Experience Host',
    'Windows Default Lock Screen', 'Windows Shell', 'Windows Shell Common',
    'Windows Shell Experience', 'Windows Shell Common DXL', 'Windows Shell Common DXL Helper',
    'Windows Shell Common DXL Render Window', 'Windows Shell Common DXL Thumbnail Window',
    'Windows Shell Common DXL Thumbnail Host Window', 'Windows Shell Common DXL Thumbnail Provider Window',
    'Windows Shell Common DXL Thumbnail Provider Host Window', 'Windows Shell Common DXL Thumbnail Provider Render Window',
    'Windows Shell Common DXL Thumbnail Provider Thumbnail Window', 'Windows Shell Common DXL Thumbnail Provider Thumbnail Host Window',
    'Windows Shell Common DXL Thumbnail Provider Thumbnail Render Window'
})

# System window classes to exclude
SYSTEM_WINDOW_CLASSES = frozenset({
    'Shell_TrayWnd',          # Main taskbar
    'Shell_SecondaryTrayWnd',  # Secondary monitor taskbars
    'Shell_CharmWindow',
    'Windows.UI.Core.CoreWindow',
    'ApplicationFrameWindow',
    'Windows.UI.Input.InputSite.WindowClass',
    'IME',
    'MSCTFIME UI',
    'SysShadow',
    'SysPager',
    'ToolbarWindow32',
    'tooltips_class32',
    'DummyDWMListenerWindow',
    'CiceroUIWndFrame',
    'NarratorHelperWindow',
    'CCSTARTScreen',
    'SearchPane',
    'WorkerW',
    'Button',
    'Static',
    'SysListView32'
})

# System processes to exclude
SYSTEM_PROCESSES = frozenset({
    'searchui.exe', 'shell32.dll', 'shellexperiencehost.exe',
    'startmenuexperiencehost.exe', 'applicationframehost.exe', 'searchapp.exe',
    'textinputhost.exe', 'windowsinternal.composableshell.experiences.textinput.inputapp.exe',
    'sihost.exe', 'taskhostw.exe', 'dwm.exe', 'ctfmon.exe', 'runtimebroker.exe',
    'systemsettings.exe', 'lockapp.exe', 'taskbar.exe'
})

# Video file extensions for media player detection
VIDEO_EXTENSIONS = frozenset({
    '.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg',
    '.m2ts', '.ts', '.mts', '.m2t', '.m2v', '.m4v', '.mpv', '.mpg2', '.mpg4', '.ogv',
    '.qt', '.rm', '.rmvb', '.vob', '.asf', '.divx', '.m4p', '.m4v', '.mxf', '.ogm',
    '.ogx', '.vp8', '.vp9', '.webm', '.yuv'
})

# Known media player processes and classes
MEDIA_PLAYER_PROCESSES = frozenset({
    'mpv', 'vlc', 'mpc', 'potplayer', 'gmp', 'wmplayer', 'kodi', 'jellyfin', 'plex'
})

MEDIA_PLAYER_CLASSES = frozenset({
    'Qt5QWindowIcon', 'QWidget', 'mpv', 'VLC', 'MediaPlayerClassicW', 'PotPlayer', 'WMPlayerApp', 'Kodi'
})

# Cached window information for performance
_window_info_cache: Dict[int, Dict[str, Any]] = {}


def is_window_visible(hwnd: int) -> bool:
    """Check if a window is visible.
    
    Args:
        hwnd: The window handle to check
        
    Returns:
        bool: True if the window is visible, False otherwise
    """
    try:
        return bool(ctypes.windll.user32.IsWindowVisible(hwnd))
    except Exception:
        return False


def get_window_rect(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """Get the rectangle of a window.
    
    Args:
        hwnd: The window handle
        
    Returns:
        Optional[tuple]: (left, top, right, bottom) coordinates, or None if failed
    """
    try:
        rect = ctypes.wintypes.RECT()
        if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        pass
    return None


def get_window_style(hwnd: int) -> int:
    """Get the window style flags.
    
    Args:
        hwnd: The window handle
        
    Returns:
        int: The window style flags, or 0 on error
    """
    try:
        return ctypes.windll.user32.GetWindowLongW(hwnd, GWL_STYLE)
    except Exception:
        return 0


def get_window_title(hwnd: int) -> str:
    """Get the title of a window.
    
    Args:
        hwnd: The window handle
        
    Returns:
        str: The window title, or an empty string if failed
    """
    try:
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
        buffer = ctypes.create_unicode_buffer(length)
        if ctypes.windll.user32.GetWindowTextW(hwnd, buffer, length):
            return buffer.value
    except Exception:
        pass
    return ""


def get_window_class(hwnd: int) -> str:
    """Get the window class name.
    
    Args:
        hwnd: The window handle
        
    Returns:
        str: The window class name, or an empty string if failed
    """
    try:
        buffer = ctypes.create_unicode_buffer(256)
        if ctypes.windll.user32.GetClassNameW(hwnd, buffer, 256):
            return buffer.value
    except Exception:
        pass
    return ""


def get_window_process_info(hwnd: int) -> tuple[int, str]:
    """Get the process ID and executable name for a window.
    
    Args:
        hwnd: The window handle
        
    Returns:
        tuple: (process_id, executable_name) or (0, "") on error
    """
    try:
        process_id = ctypes.wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        
        if process_id.value:
            process_handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, process_id.value)  # PROCESS_QUERY_LIMITED_INFORMATION
            if process_handle:
                try:
                    exe_path = ctypes.create_unicode_buffer(260)
                    size = ctypes.c_uint32(260)
                    if ctypes.windll.psapi.GetProcessImageFileNameW(process_handle, exe_path, size):
                        exe_name = os.path.basename(exe_path.value).lower()
                        return (process_id.value, exe_name)
                finally:
                    ctypes.windll.kernel32.CloseHandle(process_handle)
    except Exception:
        pass
    return (0, "")


def is_system_window(hwnd: int, our_pid: int) -> bool:
    """Check if a window is a system window that should be excluded.
    
    Args:
        hwnd: The window handle to check
        our_pid: The process ID of our application
        
    Returns:
        bool: True if it's a system window, False otherwise
    """
    # Skip invalid windows
    if not hwnd or hwnd == 0:
        return True
    
    # Get window information
    title = get_window_title(hwnd)
    window_class = get_window_class(hwnd)
    process_id, exe_name = get_window_process_info(hwnd)
    
    # Skip our own process windows
    if process_id == our_pid:
        return True
    
    # Skip windows with no title (usually system windows)
    if not title.strip():
        return True
    
    # Check against system window titles and classes
    if (title in SYSTEM_WINDOW_TITLES or 
        window_class in SYSTEM_WINDOW_CLASSES or
        exe_name in SYSTEM_PROCESSES):
        return True
    
    # Skip tool windows and other special windows
    style = get_window_style(hwnd)
    if style & 0x80:  # WS_DISABLED
        return True
    
    return False


def is_valid_window(hwnd: int, our_pid: Optional[int] = None, check_visible: bool = True) -> bool:
    """Check if a window is valid for capture or switching.
    
    Args:
        hwnd: The window handle to check
        our_pid: Optional process ID of our application (for filtering)
        check_visible: Whether to check if the window is visible
        
    Returns:
        bool: True if the window is valid, False otherwise
    """
    # Skip invalid windows
    if not hwnd or hwnd == 0:
        return False
    
    # Get our PID if not provided
    if our_pid is None:
        our_pid = os.getpid()
    
    # Check if it's a system window
    if is_system_window(hwnd, our_pid):
        return False
    
    # Check if the window is visible if requested
    if check_visible and not is_window_visible(hwnd):
        return False
    
    # Check window size
    rect = get_window_rect(hwnd)
    if not rect:
        return False
    
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    
    # Skip windows that are too small
    if width <= 0 or height <= 0:
        return False
    
    return True


def is_media_player(hwnd: int) -> bool:
    """Check if a window is likely a media player.
    
    Args:
        hwnd: The window handle to check
        
    Returns:
        bool: True if it's likely a media player, False otherwise
    """
    # Get window information
    title = get_window_title(hwnd)
    window_class = get_window_class(hwnd)
    _, exe_name = get_window_process_info(hwnd)
    
    # Check against known media player processes and classes
    if (any(media_exe in exe_name.lower() for media_exe in MEDIA_PLAYER_PROCESSES) or
        window_class in MEDIA_PLAYER_CLASSES):
        return True
    
    # Check if the window title contains a video file extension
    if any(ext in title.lower() for ext in VIDEO_EXTENSIONS):
        return True
    
    return False
