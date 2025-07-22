"""Enhanced window filtering for SPQ application.

This module provides improved window filtering to exclude system and dummy windows
from window switching and other operations.
"""

import os
import win32gui
import win32con
import win32process
import win32api
from dataclasses import dataclass
from typing import Optional, List
from debug_utils import get_logger

logger = get_logger(__name__)

@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    process_name: Optional[str]
    pid: int
    rect: Optional[tuple]
    visible: bool

class WindowFilter:
    """Provides methods to filter out system and dummy windows."""
    
    # Known system window titles (case-insensitive)
    SYSTEM_WINDOW_TITLES = [
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
    ]
    
    # Known system window class names (case-sensitive)
    SYSTEM_WINDOW_CLASSES = [
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
        'SysListView32',
        'Progman',  # Program Manager
        'EdgeUiInputTopWndClass',
        'EdgeUiInputWndClass',
        'ForegroundStaging',
        'ApplicationManager_ImmersiveShellWindow',
        'ImmersiveLauncher',
        'ImmersiveGutter',
        'MultitaskingViewFrame',
        'XamlExplorerHostIslandWindow',
        'Windows.UI.Composition.DesktopWindowContentBridge'
    ]
    
    # Known system processes (case-insensitive)
    # Note: explorer.exe is NOT in this list because File Explorer windows are legitimate
    # Note: cmd.exe and powershell.exe are NOT in this list - they should always be allowed
    SYSTEM_PROCESSES = [
        'searchui.exe', 'shell32.dll', 'shellexperiencehost.exe',
        'startmenuexperiencehost.exe', 'applicationframehost.exe', 'searchapp.exe',
        'textinputhost.exe', 'windowsinternal.composableshell.experiences.textinput.inputapp.exe',
        'sihost.exe', 'taskhostw.exe', 'dwm.exe', 'ctfmon.exe', 'runtimebroker.exe',
        'systemsettings.exe', 'lockapp.exe', 'searchapp.exe', 'taskbar.exe',
        'winlogon.exe', 'csrss.exe', 'smss.exe', 'lsass.exe', 'wininit.exe',
        'spoolsv.exe'
    ]
    
    # Explorer.exe window classes that should be allowed (File Explorer windows)
    EXPLORER_ALLOWED_CLASSES = [
        'CabinetWClass',      # File Explorer windows
        'ExploreWClass',      # Explorer windows (older style)
        'ExplorerTaskbarClass'  # Sometimes used for legitimate explorer windows
    ]
    
    # Firefox-specific class names that should be excluded
    FIREFOX_EXCLUDED_CLASSES = [
        'MozillaDialogClass',
        'MozillaDropShadowWindowClass',
        'MozillaTaskbarPreviewClass',
        'MozillaHiddenWindowClass'
    ]
    
    # Important applications that should NEVER be filtered out
    NEVER_FILTER_PROCESSES = [
        'cmd.exe',           # Command Prompt
        'powershell.exe',    # Windows PowerShell
        'pwsh.exe',          # PowerShell Core
        'conhost.exe',       # Console Host (when it has visible windows)
        'wt.exe',            # Windows Terminal
        'windowsterminal.exe'  # Windows Terminal (alternate name)
    ]
    
    # Common applications that should be allowed (whitelist approach)
    COMMON_APPLICATIONS = [
        'notepad.exe', 'mspaint.exe', 'write.exe', 'calc.exe',
        'firefox.exe', 'chrome.exe', 'msedge.exe', 'opera.exe', 'brave.exe',
        'code.exe', 'notepad++.exe', 'sublime_text.exe', 'atom.exe',
        'steam.exe', 'discord.exe', 'spotify.exe', 'vlc.exe',
        'winword.exe', 'excel.exe', 'powerpnt.exe', 'outlook.exe',
        'photoshop.exe', 'illustrator.exe', 'premiere.exe',
        'devenv.exe', 'idea64.exe', 'pycharm64.exe',
        'cmd.exe', 'powershell.exe', 'pwsh.exe', 'wt.exe'
    ]
    
    @classmethod
    def is_main_window(cls, hwnd):
        """Check if a window is a main application window (not a child or popup).
        
        Args:
            hwnd: The window handle to check
            
        Returns:
            bool: True if this is a main window, False if it's a child/popup
        """
        try:
            # Check if this window has a parent (if so, it's likely a child window)
            parent = win32gui.GetParent(hwnd)
            if parent != 0:
                return False
                
            # Check if this is an owner window (popup/dialog)
            owner = win32gui.GetWindow(hwnd, win32con.GW_OWNER)
            if owner != 0:
                # Some legitimate windows have owners (like modal dialogs)
                # But we want to exclude things like splash screens and tooltips
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                
                # If it's a tool window or has no taskbar button, it's probably not main
                if (ex_style & win32con.WS_EX_TOOLWINDOW or 
                    not (ex_style & win32con.WS_EX_APPWINDOW) and not (style & win32con.WS_OVERLAPPEDWINDOW)):
                    return False
                    
            return True
            
        except Exception as e:
            logger.debug(f"Error checking if window {hwnd} is main window: {e}")
            return False
    
    @classmethod
    def get_process_name(cls, pid):
        """Get the process name for a given PID.
        
        Args:
            pid: Process ID
            
        Returns:
            str: Process name (lowercase) or None if unable to determine
        """
        try:
            process_handle = win32api.OpenProcess(
                win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, 
                False, 
                pid
            )
            
            if process_handle:
                try:
                    process_name = win32process.GetModuleFileNameEx(process_handle, 0).lower()
                    win32api.CloseHandle(process_handle)
                    return os.path.basename(process_name)
                except Exception:
                    win32api.CloseHandle(process_handle)
                    return None
            return None
            
        except Exception:
            return None
    
    @classmethod
    def is_valid_window(cls, hwnd: int, our_pid: Optional[int] = None, min_width: int = 50, min_height: int = 30, cache: Optional[dict] = None) -> bool:
        """Check if a window should be included in window operations, with caching and configurable min size.
        Args:
            hwnd: The window handle to check
            our_pid: The process ID of our application (optional)
            min_width: Minimum allowed window width
            min_height: Minimum allowed window height
            cache: Optional dict to cache WindowInfo objects
        Returns:
            bool: True if the window should be included, False otherwise
        """
        if not hwnd or hwnd == 0:
            logger.debug(f"Window {hwnd}: Invalid window handle")
            return False
        try:
            if cache is not None and hwnd in cache:
                info = cache[hwnd]
            else:
                if our_pid is None:
                    our_pid = os.getpid()
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                visible = win32gui.IsWindowVisible(hwnd)
                title = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                process_name = cls.get_process_name(pid)
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                except Exception:
                    rect = None
                info = WindowInfo(hwnd, title, class_name, process_name, pid, rect, visible)
                if cache is not None:
                    cache[hwnd] = info
            # Skip our own process windows
            if info.pid == our_pid:
                logger.debug(f"Window {hwnd}: Excluding our own process window")
                return False
            if not info.visible:
                logger.debug(f"Window {hwnd}: Excluding invisible window")
                return False
            # CRITICAL: NEVER filter out important terminal/console applications
            if info.process_name and info.process_name in cls.NEVER_FILTER_PROCESSES:
                logger.debug(f"Window {hwnd}: NEVER filtering critical process: {info.process_name}")
                return True
            if info.class_name in ['Shell_TrayWnd', 'Shell_SecondaryTrayWnd']:
                logger.debug(f"Window {hwnd}: BLOCKED taskbar window")
                return False
            if info.class_name in cls.SYSTEM_WINDOW_CLASSES:
                logger.debug(f"Window {hwnd} (Class: {info.class_name}): Excluded by system window class")
                return False
            if info.title:
                title_lower = info.title.lower()
                for sys_win in cls.SYSTEM_WINDOW_TITLES:
                    if sys_win.lower() in title_lower:
                        logger.debug(f"Window {hwnd} (Title: {info.title}): Excluded by system window title: {sys_win}")
                        return False
            if not cls.is_main_window(hwnd):
                logger.debug(f"Window {hwnd}: Excluded as child/popup window")
                return False
            # Skip windows that are too small
            if info.rect:
                left, top, right, bottom = info.rect
                width, height = right - left, bottom - top
                if width <= 0 or height <= 0 or width < min_width or height < min_height:
                    logger.debug(f"Window {hwnd}: Excluded due to small size: {width}x{height}")
                    return False
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            if ex_style & win32con.WS_EX_TOOLWINDOW and not (ex_style & win32con.WS_EX_APPWINDOW):
                if info.process_name and info.process_name in cls.NEVER_FILTER_PROCESSES:
                    logger.debug(f"Window {hwnd}: Allowing tool window from critical process: {info.process_name}")
                    return True
                logger.debug(f"Window {hwnd}: Excluded tool window without app window flag")
                return False
            if style & win32con.WS_DISABLED:
                logger.debug(f"Window {hwnd}: Excluded disabled window")
                return False
            if 'firefox' in info.class_name.lower() or 'mozilla' in info.class_name.lower():
                if info.class_name in cls.FIREFOX_EXCLUDED_CLASSES:
                    logger.debug(f"Window {hwnd}: Excluded Firefox-specific window class: {info.class_name}")
                    return False
            if info.process_name:
                if info.process_name == 'explorer.exe':
                    if info.class_name in cls.EXPLORER_ALLOWED_CLASSES:
                        logger.debug(f"Window {hwnd}: Allowed File Explorer window (Class: {info.class_name})")
                        return True
                    elif info.class_name in cls.SYSTEM_WINDOW_CLASSES:
                        logger.debug(f"Window {hwnd}: Blocked explorer.exe system window (Class: {info.class_name})")
                        return False
                    elif info.title and len(info.title.strip()) > 0:
                        logger.debug(f"Window {hwnd}: Allowed explorer.exe window with title: '{info.title}'")
                        return True
                    else:
                        logger.debug(f"Window {hwnd}: Blocked explorer.exe window without title (Class: {info.class_name})")
                        return False
                if info.process_name in cls.COMMON_APPLICATIONS:
                    logger.debug(f"Window {hwnd}: Allowed by common application whitelist: {info.process_name}")
                    return True
                if info.process_name not in cls.NEVER_FILTER_PROCESSES:
                    for sys_proc in cls.SYSTEM_PROCESSES:
                        if sys_proc.lower() == info.process_name:
                            logger.debug(f"Window {hwnd}: Excluded by system process: {sys_proc}")
                            return False
            if not info.title and not info.class_name:
                logger.debug(f"Window {hwnd}: Excluded - no title or class name")
                return False
            logger.debug(f"Window {hwnd} (Title: '{info.title}', Class: '{info.class_name}', Process: '{info.process_name}'): INCLUDED")
            return True
        except Exception as e:
            logger.debug(f"Error checking window validity for hwnd {hwnd}: {e}", exc_info=True)
            return False

    
    @classmethod
    def get_filtered_windows(cls, our_pid: Optional[int] = None, min_width: int = 50, min_height: int = 30) -> List[int]:
        """Get a list of all valid windows that should be included in window operations.
        Args:
            our_pid: The process ID of our application (optional)
            min_width: Minimum allowed window width
            min_height: Minimum allowed window height
        Returns:
            List of window handles that passed the filter
        """
        valid_windows = []
        cache = {}
        def enum_window_callback(hwnd, lparam):
            if cls.is_valid_window(hwnd, our_pid, min_width, min_height, cache):
                valid_windows.append(hwnd)
            return True
        try:
            win32gui.EnumWindows(enum_window_callback, 0)
        except Exception as e:
            logger.error(f"Error enumerating windows: {e}")
        return valid_windows

    @staticmethod
    def run_filter_tests():
        """Simple unit test for filter rules (prints summary)."""
        print("Running WindowFilter unit test...")
        windows = WindowFilter.get_filtered_windows()
        print(f"Filtered windows found: {len(windows)}")
        for hwnd in windows[:5]:
            try:
                title = win32gui.GetWindowText(hwnd)
                print(f"  HWND: {hwnd}, Title: {title}")
            except Exception as e:
                print(f"  HWND: {hwnd}, Error: {e}")