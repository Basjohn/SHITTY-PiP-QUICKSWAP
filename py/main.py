# Standard library imports
import sys
import os
import ctypes
import ctypes.wintypes
import time
import traceback
import logging
import gc
import argparse
from pathlib import Path

# Centralized logging and debugging
from debug_utils import (get_logger, setup_logging, debug_enabled, 
                        set_debug_mode, log_perf, debug_print, DebugTimer)

def global_exception_hook(exctype, value, tb):
    """Global handler for all unhandled exceptions."""
    logger = get_logger("global")
    from debug_utils import log_exception
    log_exception("Unhandled exception occurred", exc=value, logger_name="global")
    # Optionally, call the default excepthook to print to stderr
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_exception_hook

# Windows API imports
import win32gui
import win32con
import win32process
import win32api

# Third-party imports
import keyboard
import monitor_utils
from constants import (
    DEFAULT_POSITION_PRESET, DEFAULT_WINDOW_OVERLAY_WIDTH, DEFAULT_WINDOW_OVERLAY_HEIGHT,
    DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR, DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR,
    Win32Constants
)
from PySide6.QtCore import (
    QEasingCurve, QEvent, QMargins, QMetaObject, QObject, QPoint, QProcess,
    QPropertyAnimation, QRect, QRegularExpression, QSettings, QSize, QSizeF,
    QStandardPaths, QTimer, Qt, QUrl, Signal, Slot, qInstallMessageHandler, 
    QtMsgType, QFile, QTextStream, QIODevice, Q_ARG
)
from PySide6.QtGui import (
    QAction, QColor, QCursor, QFont, QGuiApplication, QIcon, QImage,
    QKeySequence, QMouseEvent, QPaintEvent, QPainter, QPen, QPixmap,
    QScreen, QShortcut, QWheelEvent, QWindow
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpacerItem, QStyle, QSystemTrayIcon,
    QVBoxLayout, QWidget
)
from shiboken6 import isValid, delete

# Application imports
import resources_rc  # This will register all resources from the .qrc file
from window_overlay import BorderWidget, DWM_THUMBNAIL_PROPERTIES
import monitor_utils
from monitor_overlay import MonitorOverlay
from settings_panel import SettingsPanel
from subsettings_dialog import SubSettingsDialog
from window_switcher import set_window_switcher, WindowSwitcher
from window_management import WindowManager

try:
    pass
except Exception as e:
    print(f"CASCADE_DEBUG: Error importing AboutDialog: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Define resource paths using QRC
THEMES_DIR = ":/themes"
RESOURCES_DIR = ":/Resources"
SETTINGS_DIR = "Settings"  # Will be created in the user's app data directory
LOGS_DIR = "Logs"  # Will be created in the user's app data directory

# Add the Py directory to the Python path when running from source
if not getattr(sys, 'frozen', False):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ensure settings and logs directories exist in the user's app data directory
app_data_dir = os.path.join(os.getenv('APPDATA'), 'ShittyPiP')
os.makedirs(os.path.join(app_data_dir, SETTINGS_DIR), exist_ok=True)
os.makedirs(os.path.join(app_data_dir, LOGS_DIR), exist_ok=True)

# Set up full paths for settings and logs
SETTINGS_PATH = os.path.join(app_data_dir, SETTINGS_DIR)
LOGS_PATH = os.path.join(app_data_dir, LOGS_DIR)
SETTINGS_FILE = os.path.join(SETTINGS_PATH, "settings.ini")

# Initialize logger first
logger = get_logger(__name__)

# Import required Windows API modules
try:
    import win32gui
    import win32con
    import win32process
    import win32api
    import win32event
    import winerror
    import win32com.client
    import win32security
    import win32ts
    import win32ui
    import win32gui_struct
    from ctypes import wintypes, windll, WINFUNCTYPE
    WIN32_AVAILABLE = True
except ImportError as e:
    WIN32_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning(f"win32 modules not available. Some functionality may be limited. Error: {e}")

# Define Windows API types and functions
if WIN32_AVAILABLE:
    WinEventProcType = WINFUNCTYPE(
        None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND,
        wintypes.LONG, wintypes.LONG, wintypes.DWORD, wintypes.DWORD
    )
else:
    # Dummy implementation if win32 modules are not available
    WinEventProcType = None

# Import application modules

WinEventProcType = WINFUNCTYPE(
    None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND,
    wintypes.LONG, wintypes.LONG, wintypes.DWORD, wintypes.DWORD
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "Resources"
SETTINGS_DIR = PROJECT_ROOT / "Settings"
LOGS_DIR = PROJECT_ROOT / "Logs"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Shitty PiP QuickSwap - Window overlay application')
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
parser.add_argument('--switch', action='store_true', help='Launch app, test window switching, and exit automatically')
args = parser.parse_args()

# Initialize centralized logging with debug mode if requested
if args.debug:
    set_debug_mode(True)
    os.environ['SPQ_DEBUG'] = '1'

# Set up logging with proper configuration
setup_logging()
logger = get_logger(__name__)
logger.info("Application starting")

# These imports have been moved to the top of the file

MUTEX_NAME = "Global\\ShittyPiP_SingleInstance_Mutex"

def is_already_running():
    mutex = None
    try:
        mutex = win32event.CreateMutex(None, False, MUTEX_NAME)
        last_error = win32api.GetLastError()
        return last_error == winerror.ERROR_ALREADY_EXISTS
    except Exception as e:
        logger.error(f"Error in is_already_running: {e}")
        return False
    finally:
        if mutex is not None:
            win32api.CloseHandle(mutex)

def close_existing_instance():
    try:
        hwnd = win32gui.FindWindow("Qt5QWindowIcon", None)
        while hwnd:
            window_title = win32gui.GetWindowText(hwnd)
            if "PiP Overlay" in window_title:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                timeout = 0
                while win32gui.IsWindow(hwnd) and timeout < 50:
                    time.sleep(0.1)
                    timeout += 1
                return True
            hwnd = win32gui.GetWindow(hwnd, win32con.GW_HWNDNEXT)
        return False
    except Exception as e:
        logger.error(f"Error in close_existing_instance: {e}")
        return False

if is_already_running():
    if not close_existing_instance():
        logger.critical("Failed to close existing instance. Exiting.")
        sys.exit(1)

DWM_AVAILABLE = hasattr(ctypes.windll, 'dwmapi')

class MediaPlayerKeepAlive(QObject):
    """
    A class to keep media players active by periodically sending fake input events.
    This is a best-effort implementation and will fail gracefully if anything goes wrong.
    """
    def __init__(self, hwnd, parent=None):
        super().__init__(parent)
        self.hwnd = hwnd
        self.timer = QTimer()
        self.timer.timeout.connect(self._keep_alive_tick)
        self.is_active = False
        self.last_keepalive_time = 0
        self.keepalive_interval = 30  # seconds between keep-alive attempts
        
        # Media player detection
        self.window_class = None
        self.window_title = None
        self._update_window_info()
        
    def _update_window_info(self):
        """Update window class and title information."""
        try:
            if win32gui.IsWindow(self.hwnd):
                self.window_class = win32gui.GetClassName(self.hwnd)
                self.window_title = win32gui.GetWindowText(self.hwnd)
                return True
        except Exception as e:
            logger.debug(f"Error updating window info: {e}")
        return False
        
    def _is_media_player(self):
        """Check if the window is likely a media player."""
        if not self.window_class or not self.window_title:
            if not self._update_window_info():
                return False
                
        media_player_classes = ['mpv', 'Qt5QWindowIcon', 'VLC', 'vlc', 'MediaPlayerClassicW', 'WMP', 'WMPlayerApp']
        media_player_keywords = ['vlc', 'mpv', 'media player', 'kodi', 'jellyfin', 'plex', 'potplayer']
        
        class_matches = any(mp_class.lower() in self.window_class.lower() for mp_class in media_player_classes)
        title_matches = any(keyword in self.window_title.lower() for keyword in media_player_keywords)
        
        return class_matches or title_matches
        
    def _send_keepalive(self):
        """Send a keep-alive event to the window."""
        try:
            if not self._update_window_info():
                logger.debug("Window no longer exists, stopping keep-alive")
                self.stop()
                return False
                
            if not self._is_media_player():
                logger.debug("Window is not a media player, stopping keep-alive")
                self.stop()
                return False
                
            # Method 1: Send a harmless message
            try:
                win32gui.SendMessageTimeout(
                    self.hwnd, 
                    win32con.WM_APP + 1,  # A harmless application-defined message
                    0, 0, 
                    win32con.SMTO_ABORTIFHUNG, 
                    100  # 100ms timeout
                )
                logger.debug("Sent keep-alive message to window")
                return True
            except Exception as e:
                logger.debug(f"Error sending keep-alive message: {e}")
                
            # Method 2: If the first method fails, try a different approach
            try:
                # Bring window to foreground briefly
                current_foreground = win32gui.GetForegroundWindow()
                win32gui.SetForegroundWindow(self.hwnd)
                time.sleep(0.1)  # Very brief delay
                if current_foreground and win32gui.IsWindow(current_foreground):
                    win32gui.SetForegroundWindow(current_foreground)
                return True
            except Exception as e:
                logger.debug(f"Error with foreground switch keep-alive: {e}")
                
            return False
            
        except Exception as e:
            logger.debug(f"Error in keep-alive: {e}")
            return False
    
    def _keep_alive_tick(self):
        """Timer callback for keep-alive."""
        current_time = time.time()
        if current_time - self.last_keepalive_time >= self.keepalive_interval:
            if self._send_keepalive():
                self.last_keepalive_time = current_time
    
    def start(self):
        """Start the keep-alive timer."""
        if not self.is_active:
            if not self._is_media_player():
                logger.debug("Not starting keep-alive - not a media player")
                return False
                
            # Use keepalive_interval (in seconds) for the timer
            self.timer.start(self.keepalive_interval * 1000)  # Convert to milliseconds
            self.is_active = True
            self.last_keepalive_time = time.time()
            logger.debug(f"Started keep-alive for window {self.hwnd} (checking every {self.keepalive_interval} seconds)")
            return True
        return False
    
    def stop(self):
        """Stop the keep-alive timer."""
        if self.is_active:
            self.timer.stop()
            self.is_active = False
            logger.debug(f"Stopped keep-alive for window {self.hwnd}")
            
    def __del__(self):
        self.stop()


class WindowEnumerator:
    # Class-level icon cache to store loaded icons with weak references
    _icon_cache = {}
    _process_icon_cache = {}
    _blank_icon = None
    
    def __init__(self):
        self.app_instance = QApplication.instance()
        self.last_window_list = []
        self.last_refresh_time = 0
        
        # Initialize the blank icon if not already done
        if WindowEnumerator._blank_icon is None:
            self._init_blank_icon()
    
    @classmethod
    def _init_blank_icon(cls):
        """Initialize the blank fallback icon."""
        cls._blank_icon = QIcon()
        # Try both possible paths to ensure compatibility
        icon_paths = [":/Resources/Blank.ico", ":/Blank.ico"]
        
        for path in icon_paths:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                cls._blank_icon = QIcon(pixmap)
                logger.debug(f"Successfully loaded fallback icon from: {path}")
                return
        
        logger.warning("Failed to load Blank.ico from any resource path")
    
    def _is_valid_window(self, hwnd, check_visible=True):
        """Check if a window is valid for inclusion in the window list.
        
        This is a thin wrapper around window_validation.is_valid_window that
        adds additional logging for debugging purposes.
        
        Args:
            hwnd: The window handle to check
            check_visible: Whether to check if the window is visible (default: True)
            
        Returns:
            bool: True if the window should be included, False otherwise
        """
        from window_validation import is_valid_window as validate_window
        
        try:
            # Skip invalid window handles
            if not hwnd or hwnd == 0:
                return False
                
            # Call the validation function with all required parameters
            result = validate_window(hwnd, os.getpid(), check_visible=check_visible)
            
            # Only log debug info if debug logging is enabled
            if not result and logger.isEnabledFor(logging.DEBUG):
                try:
                    title = win32gui.GetWindowText(hwnd)
                    class_name = win32gui.GetClassName(hwnd)
                    logger.debug(f"Window {hwnd} (Title: {title}, Class: {class_name}): Excluded by validation checks")
                except Exception as e:
                    logger.debug(f"Window {hwnd}: Excluded by validation checks (error getting window info: {e})")
            
            return result
            
        except Exception as e:
            # Log the error but don't let it propagate
            logger.error(f"Error validating window {hwnd}: {e}", exc_info=True)
            return False

    def refresh_window_list(self, force=False):
        current_time = time.time()
        if force or current_time - self.last_refresh_time > 2:
            self.last_window_list = self.enum_windows()
            self.last_refresh_time = current_time
        return self.last_window_list
    
    @classmethod
    def enum_windows(cls):
        windows = []
        our_pid = os.getpid()
        our_titles = {
            'Shitty PiP QuickSwap',
            'Settings',
            'Sub-settings',
            'Overlay',
            'Monitor Overlay',
            'Window Overlay'
        }
        
        # Common video file extensions to detect media players by window title
        video_extensions = [
            '.mkv', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg',
            '.m2ts', '.ts', '.mts', '.m2t', '.m2v', '.m4v', '.mpv', '.mpg2', '.mpg4', '.ogv',
            '.qt', '.rm', '.rmvb', '.vob', '.asf', '.divx', '.m4p', '.m4v', '.mxf', '.ogm',
            '.ogx', '.vp8', '.vp9', '.webm', '.yuv'
        ]
        
        # Known media player processes and classes
        media_player_processes = ['mpv', 'vlc', 'mpc', 'potplayer', 'gmp', 'wmplayer', 'kodi', 'jellyfin', 'plex']
        media_player_classes = ['Qt5QWindowIcon', 'QWidget', 'mpv', 'VLC', 'MediaPlayerClassicW', 'PotPlayer', 'WMPlayerApp', 'Kodi']
        
        def enum_windows_callback(hwnd, _):
            try:
                # Skip if window is not visible
                if not ctypes.windll.user32.IsWindowVisible(hwnd):
                    return True
                    
                # Get window title
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
                title = ctypes.create_unicode_buffer(length)
                ctypes.windll.user32.GetWindowTextW(hwnd, title, length)
                title = title.value.strip()
                
                if not title:
                    return True
                    
                # Skip our own windows by title
                if any(our_title in title for our_title in our_titles):
                    return True
                    
                # Get window class name and process info
                class_name = win32gui.GetClassName(hwnd)
                
                # Get process name for additional filtering
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                process_name = ""
                try:
                    process = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
                    if process:
                        try:
                            process_name = win32process.GetModuleFileNameEx(process, 0)
                            process_name = os.path.basename(process_name).lower()
                        finally:
                            win32api.CloseHandle(process)
                except Exception as e:
                    logger.debug(f"Error getting process name for PID {pid}: {e}")
                
                # Skip our own process windows and system processes
                if pid == our_pid:
                    return True
                    
                # Skip known system windows
                system_windows = [
                    'windows input experience',  # Touch keyboard and related
                    'text input application',    # Windows 10+ touch keyboard
                    'searchui',                  # Windows Search UI
                    'shellexperiencehost',       # Shell Experience Host
                    'applicationframehost',      # UWP app host
                    'startmenuexperiencehost',   # Start Menu
                    'searchapp'                  # Windows Search
                ]
                
                # Check if this is a system window by process name or window title
                is_system_window = any(
                    sys_win.lower() in process_name.lower() or 
                    sys_win.lower() in title.lower()
                    for sys_win in system_windows
                )
                
                if is_system_window:
                    return True
                    
                # Check for media player by process name
                is_media_player = any(
                    mp_process in process_name.lower() 
                    for mp_process in media_player_processes
                )
                
                # Check for media player by window class
                is_media_player = is_media_player or any(
                    mp_class.lower() in class_name.lower() 
                    for mp_class in media_player_classes
                )
                
                # Check for media player by window title (video file extensions)
                title_lower = title.lower()
                is_media_player = is_media_player or any(
                    ext in title_lower 
                    for ext in video_extensions
                )
                
                # Skip non-media player windows with these classes, but always allow Program Manager
                if class_name in ['Qt5QWindowIcon', 'QWidget'] and not is_media_player and class_name != 'Progman':
                    return True
                
                # Check window size (don't check for minimized state)
                rect = (ctypes.c_int * 4)()
                if windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    if width > 0 and height > 0:
                        windows.append((hwnd, title))
                
            except Exception as e:
                logger.debug(f"Error in enum_windows_callback: {e}")
                
            return True
            
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        ctypes.windll.user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)
        logger.debug(f"Enumerated {len(windows)} windows after filtering.")
        return windows
    
    def get_capturable_windows_with_icons(self):
        """Get a list of capturable windows with their icons.
        
        Returns:
            list: List of tuples (hwnd, title, icon) for each window
        """
        windows = []
        window_list = self.refresh_window_list(True)
        
        # Cache the desktop window info first
        progman_hwnd = self._get_desktop_window()
        if progman_hwnd:
            windows.append((progman_hwnd, "Desktop", self._get_desktop_icon()))
        
        # Process windows in batches to prevent UI freezes
        for i, (hwnd, title) in enumerate(window_list):
            try:
                # Skip invalid windows and desktop window
                if not self._is_valid_window(hwnd) or hwnd == progman_hwnd:
                    continue
                
                # Get window class for filtering
                window_class = win32gui.GetClassName(hwnd)
                
                # Skip Program Manager window since we're handling it as Desktop
                if window_class == "Progman" and title == "Program Manager":
                    continue
                
                # Get or create window icon
                icon = self._get_cached_window_icon(hwnd, window_class)
                
                # Add window info to the list
                windows.append((hwnd, title, icon))
                
                # Process events every 10 windows to keep the UI responsive
                if i > 0 and i % 10 == 0:
                    QApplication.processEvents()
                    time.sleep(0.01)  # Small delay to prevent UI freezing
                    
            except Exception as e:
                logger.error(f"Error processing window {hwnd} ({title}): {e}")
                windows.append((hwnd, title, self._blank_icon))
        
        return self.sort_windows(windows)
    
    def _get_desktop_window(self):
        """Get the handle for the desktop window."""
        try:
            # Try to find the Program Manager window first
            progman_hwnd = win32gui.FindWindow("Progman", "Program Manager")
            if not progman_hwnd:
                logger.debug("Could not find Program Manager window, falling back to desktop window")
                progman_hwnd = win32gui.GetDesktopWindow()
            return progman_hwnd
        except Exception as e:
            logger.error(f"Error getting desktop window: {e}")
            return None
    
    def _get_desktop_icon(self):
        """Get the icon for the desktop."""
        try:
            # Try to get the system folder icon
            shell32 = ctypes.windll.shell32
            hicon = shell32.ExtractIconW(0, "shell32.dll", 15)  # 15 is the folder icon in shell32.dll
            if hicon:
                try:
                    pixmap = QPixmap.fromImage(QImage.fromHICON(hicon))
                    return QIcon(pixmap)
                except Exception as e:
                    logger.debug(f"Failed to create icon from handle: {e}")
                finally:
                    ctypes.windll.user32.DestroyIcon(hicon)
            return self._blank_icon
        except Exception as e:
            logger.error(f"Error getting desktop icon: {e}")
            return self._blank_icon
    
    def _get_cached_window_icon(self, hwnd, window_class):
        """Get a window icon from cache or load it if not cached."""
        # Create a cache key based on window handle and class
        cache_key = f"{hwnd}:{window_class}"
        
        # Try to get from cache first
        if cache_key in self._icon_cache:
            return self._icon_cache[cache_key]
        
        # Try to get process-based icon if available
        process_key = self._get_process_icon_key(hwnd)
        if process_key and process_key in self._process_icon_cache:
            icon = self._process_icon_cache[process_key]
            self._icon_cache[cache_key] = icon
            return icon
        
        # Get icon using standard methods
        try:
            icon = self._get_window_icon(hwnd)
            
            # If no icon found, try alternative methods
            if icon.isNull() or (hasattr(icon, 'name') and icon.name() == "Blank.ico"):
                alt_icon = self._get_alternative_window_icon(hwnd)
                if not alt_icon.isNull():
                    icon = alt_icon
            
            # Cache the icon if valid
            if not icon.isNull():
                self._icon_cache[cache_key] = icon
                
                # Also cache by process if possible
                if process_key:
                    self._process_icon_cache[process_key] = icon
            else:
                icon = self._blank_icon
                
            return icon
            
        except Exception as e:
            logger.debug(f"Error getting icon for window {hwnd}: {e}")
            return self._blank_icon
    
    def _get_process_icon_key(self, hwnd):
        """Get a cache key based on the window's process."""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
            if process:
                try:
                    exe_path = win32process.GetModuleFileNameEx(process, 0)
                    if exe_path:
                        # Use lowercase path for case-insensitive matching
                        return exe_path.lower()
                finally:
                    win32api.CloseHandle(process)
        except Exception as e:
            logger.debug(f"Error getting process key for window {hwnd}: {e}")
        return None
    
    def _get_window_icon(self, hwnd):
        """Get window icon using standard methods."""
        try:
            # Try WM_GETICON first (standard way)
            icon_handle = win32gui.SendMessage(hwnd, win32con.WM_GETICON, win32con.ICON_SMALL, 0)
            if not icon_handle:
                icon_handle = win32gui.SendMessage(hwnd, win32con.WM_GETICON, win32con.ICON_BIG, 0)
            if not icon_handle:
                icon_handle = win32gui.GetClassLong(hwnd, win32con.GCL_HICONSM)
            if not icon_handle:
                icon_handle = win32gui.GetClassLong(hwnd, win32con.GCL_HICON)
            
            if icon_handle:
                try:
                    return QIcon(QPixmap.fromImage(QImage.fromHICON(icon_handle)))
                except Exception as e:
                    logger.debug(f"Failed to create icon from handle: {e}")
        except Exception as e:
            logger.debug(f"Error in _get_window_icon: {e}")
        
        return self._blank_icon
    
    def _get_alternative_window_icon(self, hwnd):
        """Try alternative methods to get window icon."""
        try:
            # Get the window's process ID
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            # Get the process handle
            process = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
            if process:
                try:
                    # Get the executable path
                    exe_path = win32process.GetModuleFileNameEx(process, 0)
                    if os.path.exists(exe_path):
                        # Extract icon from the executable
                        icon = QIcon(exe_path)
                        if not icon.isNull():
                            return icon
                finally:
                    win32api.CloseHandle(process)
            
            # Try to get the application icon as last resort
            hicon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
            if hicon:
                try:
                    return QIcon(QPixmap.fromImage(QImage.fromHICON(hicon)))
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Error in _get_alternative_window_icon: {e}")
        
        return self._blank_icon
    
    def sort_windows(self, windows):
        if not windows:
            return []
            
        # Find and remove the Desktop window from the list if it exists
        desktop_window = None
        filtered_windows = []
        
        for window in windows:
            if len(window) >= 2 and window[1] == "Desktop":
                desktop_window = window
            else:
                filtered_windows.append(window)
        
        # Sort the remaining windows
        if not self.app_instance or not hasattr(self.app_instance, 'window_sort_order'):
            sorted_windows = sorted(filtered_windows, key=lambda w: w[1].lower())
        else:
            sort_order = self.app_instance.window_sort_order
            if sort_order == "Alphabetical":
                sorted_windows = sorted(filtered_windows, key=lambda w: w[1].lower())
            elif sort_order == "Most Recently Active" and hasattr(self.app_instance, 'mru_hwnds'):
                if self.app_instance.mru_hwnds:
                    mru_hwnds_list = self.app_instance.mru_hwnds
                    mru_set = set(mru_hwnds_list)
                    mru_present_windows = []
                    other_windows = []
                    for w_info in filtered_windows:
                        if w_info[0] in mru_set:
                            mru_present_windows.append(w_info)
                        else:
                            other_windows.append(w_info)
                    mru_order_map = {hwnd: i for i, hwnd in enumerate(mru_hwnds_list)}
                    mru_present_windows.sort(key=lambda w: mru_order_map.get(w[0], float('inf')))
                    sorted_windows = mru_present_windows + other_windows
                else:
                    sorted_windows = filtered_windows
            else:
                sorted_windows = sorted(filtered_windows, key=lambda w: w[1].lower())
        
        # Add the Desktop window back at the end if it exists
        if desktop_window is not None:
            sorted_windows.append(desktop_window)
            
        return sorted_windows

def set_high_dpi_settings():
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Round)
    elif hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

class PiPApplication(QApplication):
    foregroundWindowChanged = Signal(int)
    
    # Signal for key passthrough setting changes
    key_passthrough_setting_changed = Signal(bool, bool)  # enabled, aggressive
    
    def __init__(self, argv):
        """Initialize the application.
        """
        super(PiPApplication, self).__init__(argv)
        
        # These will be initialized later
        self._tray_icon = None
        self._exit_code = 0
        self._overlay_configs = []
        self._capture_fps = 30  # Default capture rate
        self._last_quick_switch_time = 0.0
        self.active_overlays = {}  # {hwnd: overlay_widget}
        self.cli_automation = None  # Will be initialized if CLI args request it
        
        # Set up application paths and directories
        self._setup_paths()
        
        # Initialize settings
        self._init_settings()
        
        # Set up application properties and metadata
        self._setup_application_properties()
        
        # Initialize window management
        self._init_window_management()
        
        # Set up UI components
        self._setup_ui()
        
        # Connect signals and load settings
        self._connect_signals()
        self.load_initial_settings()
        
        # Apply the loaded theme globally to all UI components
        if hasattr(self, 'current_theme') and self.current_theme:
            logger.debug(f"Applying initial theme globally: {self.current_theme}")
            self.apply_theme_globally(self.current_theme)
        
        # Register hotkeys and complete initialization
        self._register_or_unregister_switch_hotkey()
    
    def _setup_paths(self):
        """Set up application paths and directories."""
        if getattr(sys, 'frozen', False):
            self.application_path = os.path.dirname(sys.executable)
        else:
            self.application_path = os.path.dirname(os.path.abspath(__file__))
        
        self.settings_path = str(SETTINGS_DIR / "settings.ini")
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def _init_settings(self):
        """Initialize application settings."""
        self.settings = QSettings(self.settings_path, QSettings.Format.IniFormat)
        logger.info(f"Using settings file: {self.settings_path}")
        
        # Load settings with defaults
        self.current_theme = self.settings.value("theme", "Dark", type=str)
        self.current_opacity_int = 100
        self.window_sort_order = "Most Recently Active"
        self.auto_switch_enabled = self.settings.value("auto_switch_enabled", False, type=bool)
        # Set the application name for settings
        self.setApplicationName("ShittyPiP")
        self.setApplicationDisplayName("Shitty PiP QuickSwap")
        self.setOrganizationName("ShittyPiP")
        self.setOrganizationDomain("shittypip.com")
        
        # Schedule CLI automation to run after startup if requested
        if self.cli_automation:
            QTimer.singleShot(1000, self._run_cli_automation)
        self.setQuitOnLastWindowClosed(False)
        
        # UI styling
        self.setStyle("Fusion")
        self._setup_application_font()
        self.setWindowIcon(QIcon(":/Resources/ShittyPIP.ico"))
    
    def _setup_application_properties(self):
        """Set up application properties and metadata."""
        # Application info
        self.setApplicationName("Shitty PiP QuickSwap")
        self.setApplicationDisplayName("Shitty PiP QuickSwap")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("PiPOverlay")
        self.setQuitOnLastWindowClosed(False)
        
        # UI styling
        self.setStyle("Fusion")
        self._setup_application_font()
        self.setWindowIcon(QIcon(":/Resources/ShittyPIP.ico"))
    
    def _setup_application_font(self):
        """Set up the application font with fallback."""
        try:
            font = QFont("Segoe UI")
            font.setPointSize(10)
            logger.debug(f"Setting application font: {font.family()}, "
                        f"Point Size: {font.pointSize()}, "
                        f"Pixel Size: {font.pixelSize() if font.pixelSize() > 0 else 'default'}")
            self.setFont(font)
        except Exception as e:
            logger.error(f"Error setting application font: {e}")
            # Fallback to default font if there's an issue
            self.setFont(QFont())
    
    def _init_window_management(self):
        """Initialize window management related properties."""
        # MRU (Most Recently Used) list for window ordering
        self.mru_hwnds = []
        # Track last time each hwnd gained foreground focus so we can age-out stale candidates
        self._hwnd_last_focus_ts = {}
        self.MAX_MRU_ITEMS = 50  # Maximum number of MRU items to keep
        self._overlays_locked = False  # Track if overlays are locked
        
        # Initialize WindowManager for centralized window management
        self.window_manager = WindowManager(max_mru_items=self.MAX_MRU_ITEMS)
        logger.info("Initialized WindowManager in PiPApplication._init_window_management()")
        
        # Store the Progman window handle for desktop overlay
        self.progman_hwnd = win32gui.FindWindow("Progman", "Program Manager")
        if not self.progman_hwnd:
            self.progman_hwnd = win32gui.GetDesktopWindow()
        logger.debug(f"Stored Progman HWND: {self.progman_hwnd}")
        
        # Initialize window enumerator
        self.window_enumerator = WindowEnumerator()
        self.window_enumerator.app_instance = self
        
        # Initialize window switching
        self.window_switcher = set_window_switcher(self)
        logger.debug("Initialized window_switcher")
        
        # Initialize overlay tracking
        self.active_overlays = {}
        self.keep_alive_handlers = {}  # hwnd -> MediaPlayerKeepAlive
        self._win_event_hook = None
        self._win_event_proc_callback_ptr = None
    
    def _setup_ui(self):
        """Set up UI components."""
        self._tray_icon = None
        self._settings_panel = None
        self._sub_settings_dialog = None
        self._last_background_geometry = None
        self._current_switch_hotkey_id = None
        
        # Set up system tray
        self._setup_tray()
    
    def _connect_signals(self):
        """Connect application signals to their handlers."""
        self.focusChanged.connect(self.handle_focus_change)
        self.applicationStateChanged.connect(self.handle_application_state_change)
        self._setup_foreground_event_hook()
        
        # Show settings panel on startup
        self._show_settings()

    def _show_sub_settings(self):
        """Show the sub-settings dialog."""
        try:
            from subsettings_dialog import SubSettingsDialog
            
            # Check if dialog is already open
            if hasattr(self, '_sub_settings_dialog') and self._sub_settings_dialog is not None:
                try:
                    # If dialog exists, raise it and close it
                    if self._sub_settings_dialog.isVisible():
                        self._sub_settings_dialog.close()
                        self._sub_settings_dialog = None
                        return
                    else:
                        # Dialog exists but not visible, clean it up
                        self._sub_settings_dialog.deleteLater()
                        self._sub_settings_dialog = None
                except Exception as e:
                    logger.error(f"Error managing existing dialog: {e}")
                    self._sub_settings_dialog = None
            
            # Get the active window or primary screen
            active_window = QApplication.activeWindow()
            parent = active_window if active_window else None
            
            # Create the dialog with the correct parent and app instance
            dialog = SubSettingsDialog(parent=parent, app_instance=self)
            self._sub_settings_dialog = dialog  # Store reference
            
            # Ensure the dialog uses the current theme
            if hasattr(self, 'current_theme') and self.current_theme:
                logger.debug(f"Applying theme to subsettings dialog: {self.current_theme}")
                self.apply_theme_globally(self.current_theme, from_global=True)
            
            # Set window attributes
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.finished.connect(lambda: setattr(self, '_sub_settings_dialog', None))  # Clear reference on close
            
            # Set the dialog size (330x420)
            dialog.resize(330, 420)
            dialog.setModal(False)
            
            # Get the screen where the mouse is currently located
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()
            
            # Center the dialog on the screen
            x = screen_geometry.x() + (screen_geometry.width() - dialog.width()) // 2
            y = screen_geometry.y() + (screen_geometry.height() - dialog.height()) // 2
            
            # Ensure the dialog stays within screen bounds
            x = max(screen_geometry.left(), min(x, screen_geometry.right() - dialog.width()))
            y = max(screen_geometry.top(), min(y, screen_geometry.bottom() - dialog.height()))
            
            dialog.move(x, y)
            
            # Connect the hotkey settings changed signal
            dialog.hotkey_settings_changed.connect(self.update_switch_hotkey)
            
            # Show and activate the dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            
            # Ensure the window is not minimized and is active
            dialog.setWindowState(dialog.windowState() & ~Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            
            logger.info("Opened sub-settings dialog")
            return dialog
            
        except ImportError as e:
            logger.error(f"Failed to import SubSettingsDialog: {e}")
            QMessageBox.critical(
                None, 
                "Error", 
                f"Failed to load settings dialog: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Error in _show_sub_settings: {e}", exc_info=True)
            QMessageBox.critical(
                None, 
                "Error", 
                f"Failed to open settings: {str(e)}"
            )
            return None
    
    def _show_settings(self):
        try:
            # Clean up any existing settings panel if it's invalid
            if self._settings_panel is not None:
                try:
                    # Check if the C++ object is still valid using shiboken6
                    if isValid(self._settings_panel):
                        if not self._settings_panel.isVisible():
                            # If panel exists but is hidden, just show it
                            self._settings_panel.show()
                            self._settings_panel.activateWindow()
                            self._settings_panel.raise_()
                            return
                        else:
                            # If panel is already visible, just bring it to front
                            self._settings_panel.activateWindow()
                            self._settings_panel.raise_()
                            return
                    else:
                        logger.debug("Existing SettingsPanel is invalid, creating a new one")
                        self._settings_panel = None
                except Exception as e:
                    logger.error(f"Error checking existing SettingsPanel: {e}", exc_info=True)
                    self._settings_panel = None
            
            # Create a new settings panel if we don't have a valid one
            logger.debug("Creating new SettingsPanel")
            self._settings_panel = SettingsPanel(app_instance=self)
            
            # Connect the destroyed signal to clean up the reference
            self._settings_panel.destroyed.connect(self._on_settings_panel_destroyed)
            
            # Ensure the panel uses the current theme
            if hasattr(self, 'current_theme') and self.current_theme:
                logger.debug(f"Applying theme to settings panel: {self.current_theme}")
                self.apply_theme_globally(self.current_theme, from_global=True)
            
            # Show and activate the panel with proper window flags
            self._settings_panel.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
            self._settings_panel.setAttribute(Qt.WA_TranslucentBackground)
            self._settings_panel.show()
            self._settings_panel.activateWindow()
            self._settings_panel.raise_()
            logger.info("Created and showed new SettingsPanel")
            
        except Exception as e:
            logger.error(f"Error in _show_settings: {e}", exc_info=True)
            self._settings_panel = None
    
    def _on_settings_panel_destroyed(self):
        """Handle the settings panel being destroyed."""
        logger.debug("SettingsPanel destroyed, cleaning up reference")
        if self._settings_panel:
            try:
                if isValid(self._settings_panel):
                    self._settings_panel.deleteLater()
            except Exception as e:
                logger.error(f"Error in _on_settings_panel_destroyed: {e}", exc_info=True)
            finally:
                self._settings_panel = None

    def load_initial_settings(self):
        # Only load theme if it hasn't been loaded yet
        if not hasattr(self, 'current_theme') or not self.current_theme:
            self.current_theme = self.settings.value("theme", "Dark", type=str)
            
        self.current_opacity_int = self.settings.value("opacity", 100, type=int)
        # Load border opacity value (100% by default)
        border_opacity_int = self.settings.value("border_opacity", 100, type=int)
        self.current_border_opacity_float = border_opacity_int / 100.0
        self.window_sort_order = self.settings.value("windowSortOrder", "Most Recently Active", type=str)
        self.switch_hotkey_enabled = self.settings.value("hotkey_enabled", False, type=bool)
        self.switch_hotkey_sequence = self.settings.value("hotkey_sequence", "`", type=str)
        
        # Initialize opacity hotkey settings
        self.opacity_hotkeys_enabled = self.settings.value("opacity_hotkeys_enabled", False, type=bool)
        self.increase_opacity_hotkey = self.settings.value("increase_opacity_hotkey", "+", type=str)
        self.decrease_opacity_hotkey = self.settings.value("decrease_opacity_hotkey", "-", type=str)
        self.opacity_step = 10  # Percentage points to change opacity per hotkey press
        self._last_opacity_change = 0  # Timestamp of last opacity change
        self._opacity_change_delay = 0.1  # Minimum seconds between opacity changes
        
        logger.debug(f"Initial settings loaded: Theme='{self.current_theme}', Opacity={self.current_opacity_int}, "
                     f"BorderOpacity={border_opacity_int}%, WindowSort='{self.window_sort_order}'")

    def _unregister_switch_hotkey(self):
        """Unregister the current switch hotkey if one is registered.
        
        Returns:
            bool: True if unregistration was successful or no hotkey was registered,
                  False if an error occurred during unregistration.
        """
        if not hasattr(self, '_current_switch_hotkey_id') or self._current_switch_hotkey_id is None:
            if debug_enabled():
                logger.debug("No switch hotkey registered to unregister")
            return True
            
        try:
            import keyboard
            
            # Clear all hotkeys - this is more reliable than trying to remove specific ones
            keyboard.unhook_all()
            
            if debug_enabled():
                logger.debug("Successfully unregistered all hotkeys")
                
            return True
            
        except ImportError:
            logger.error("keyboard module not available for hotkey unregistration")
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering switch hotkey: {e}", exc_info=debug_enabled())
            return False
            
        finally:
            # Always clear the hotkey ID to prevent reference to invalid hotkeys
            self._current_switch_hotkey_id = None

    def _register_switch_hotkey(self, sequence=None):
        """Register the switch hotkey with the given sequence. Suppress if single-char, do not suppress if combo."""
        if sequence is None:
            sequence = self.switch_hotkey_sequence
        if not sequence:
            logger.warning("No hotkey sequence provided to register")
            return False
            
        try:
            # First unregister any existing hotkey
            self._unregister_switch_hotkey()
            
            # Determine if the hotkey is a single character (suppress) or a combo (do not suppress)
            is_combo = ('+' in sequence) or (len(sequence.strip()) > 1)
            suppress = not is_combo
            
            # Create a wrapper function that we can track
            def hotkey_handler():
                try:
                    self._handle_switch_hotkey_pressed()
                except Exception as e:
                    logger.error(f"Error in hotkey handler: {e}", exc_info=True)
            
            # Store the sequence for reference
            self._current_hotkey_sequence = sequence
            
            # Register the hotkey and store the remove function
            self._current_switch_hotkey_id = hotkey_handler
            keyboard.add_hotkey(
                sequence,
                hotkey_handler,
                suppress=suppress
            )
            
            logger.info(f"Registered switch hotkey: '{sequence}' (suppress={suppress})")
            return True
            
        except ValueError as ve:
            logger.error(f"Invalid hotkey sequence '{sequence}': {ve}")
            self._current_switch_hotkey_id = None
            return False
            
        except Exception as e:
            logger.error(f"Failed to register switch hotkey '{sequence}': {e}", exc_info=True)
            self._current_switch_hotkey_id = None
            return False

    def _register_or_unregister_switch_hotkey(self):
        """Register or unregister the switch hotkey and opacity hotkeys based on current settings."""
        # Handle switch hotkey
        self._unregister_switch_hotkey()
        if self.switch_hotkey_enabled and self.switch_hotkey_sequence:
            self._register_switch_hotkey()
            
        # Handle opacity hotkeys
        self._unregister_opacity_hotkeys()
        if self.opacity_hotkeys_enabled and self.increase_opacity_hotkey and self.decrease_opacity_hotkey:
            self._register_opacity_hotkeys()
            
    def _register_opacity_hotkeys(self):
        """Register the opacity hotkeys if they are enabled."""
        self._unregister_opacity_hotkeys()
        if not self.opacity_hotkeys_enabled:
            return
            
        try:
            import keyboard
            
            # Register increase opacity hotkey
            if self.increase_opacity_hotkey:
                self._increase_opacity_hotkey_id = keyboard.add_hotkey(
                    self.increase_opacity_hotkey,
                    self._handle_increase_opacity_hotkey,
                    suppress=True
                )
                
            # Register decrease opacity hotkey
            if self.decrease_opacity_hotkey:
                self._decrease_opacity_hotkey_id = keyboard.add_hotkey(
                    self.decrease_opacity_hotkey,
                    self._handle_decrease_opacity_hotkey,
                    suppress=True
                )
                
            logger.info(f"Registered opacity hotkeys: +={self.increase_opacity_hotkey}, -={self.decrease_opacity_hotkey}")
            
        except Exception as e:
            logger.error(f"Failed to register opacity hotkeys: {e}")
            # Ensure we don't have stale IDs if registration fails
            if hasattr(self, '_increase_opacity_hotkey_id'):
                delattr(self, '_increase_opacity_hotkey_id')
            if hasattr(self, '_decrease_opacity_hotkey_id'):
                delattr(self, '_decrease_opacity_hotkey_id')
            
    def _unregister_opacity_hotkeys(self):
        """Unregister the opacity hotkeys safely."""
        try:
            import keyboard
            
            # Define a helper function to safely remove a hotkey
            def safe_remove_hotkey(hotkey_id_attr):
                if not hasattr(self, hotkey_id_attr):
                    return
                    
                hotkey_id = getattr(self, hotkey_id_attr)
                if hotkey_id is None:
                    return
                    
                try:
                    keyboard.remove_hotkey(hotkey_id)
                    if debug_enabled():
                        logger.debug(f"Successfully unregistered hotkey: {hotkey_id_attr}")
                except Exception as e:
                    if debug_enabled():
                        logger.debug(f"Could not unregister hotkey {hotkey_id_attr} (may already be unregistered): {e}")
                finally:
                    setattr(self, hotkey_id_attr, None)
            
            # Remove both hotkeys safely
            safe_remove_hotkey('_increase_opacity_hotkey_id')
            safe_remove_hotkey('_decrease_opacity_hotkey_id')
            
            if debug_enabled():
                logger.debug("Finished unregistering opacity hotkeys")
                
        except ImportError:
            logger.error("keyboard module not available for hotkey unregistration")
        except Exception as e:
            logger.error(f"Unexpected error in _unregister_opacity_hotkeys: {e}", exc_info=debug_enabled())
            
    def _can_change_opacity(self):
        """Check if we can change opacity (rate limiting)."""
        current_time = time.time()
        if current_time - self._last_opacity_change < self._opacity_change_delay:
            return False
        self._last_opacity_change = current_time
        return True
        
    def _update_opacity_setting(self, opacity):
        """Update opacity setting with rate-limited disk writes."""
        try:
            self.settings.setValue("opacity", opacity)
            # Only sync to disk occasionally to reduce I/O
            if not hasattr(self, '_last_opacity_sync') or time.time() - self._last_opacity_sync > 1.0:
                self.settings.sync()
                self._last_opacity_sync = time.time()
        except Exception as e:
            logger.error(f"Error updating opacity setting: {e}")
    
    def _update_opacity_main_thread(self, new_opacity):
        """Update opacity on the main thread."""
        if not hasattr(self, 'active_overlays') or not self.active_overlays:
            return
            
        try:
            self.update_opacity(new_opacity)
            self._update_opacity_setting(new_opacity)
            logger.debug(f"Updated opacity to {new_opacity}%")
        except Exception as e:
            logger.error(f"Error updating opacity: {e}")
    
    def _handle_increase_opacity_hotkey(self):
        """Handle the increase opacity hotkey press."""
        if not self.opacity_hotkeys_enabled or not hasattr(self, 'active_overlays') or not self.active_overlays:
            return
            
        if not self._can_change_opacity():
            return
            
        try:
            current_opacity = self.current_opacity_int
            new_opacity = min(100, current_opacity + self.opacity_step)
            
            if new_opacity != current_opacity:
                # Use invokeMethod to ensure we're on the main thread
                QMetaObject.invokeMethod(
                    self,
                    "_update_opacity_main_thread",
                    Qt.QueuedConnection,
                    Q_ARG(int, new_opacity)
                )
        except Exception as e:
            logger.error(f"Error in _handle_increase_opacity_hotkey: {e}")
            
    def _handle_decrease_opacity_hotkey(self):
        """Handle the decrease opacity hotkey press."""
        if not self.opacity_hotkeys_enabled or not hasattr(self, 'active_overlays') or not self.active_overlays:
            return
            
        if not self._can_change_opacity():
            return
            
        try:
            current_opacity = self.current_opacity_int
            new_opacity = max(10, current_opacity - self.opacity_step)  # Minimum 10% opacity
            
            if new_opacity != current_opacity:
                # Use invokeMethod to ensure we're on the main thread
                QMetaObject.invokeMethod(
                    self,
                    "_update_opacity_main_thread",
                    Qt.QueuedConnection,
                    Q_ARG(int, new_opacity)
                )
        except Exception as e:
            logger.error(f"Error in _handle_decrease_opacity_hotkey: {e}")

    def update_switch_hotkey(self, enabled, sequence):
        """Update the switch hotkey with new settings and apply dynamically."""
        logger.info(f"Updating switch hotkey - Enabled: {enabled}, Sequence: '{sequence}'")
        self.switch_hotkey_enabled = enabled
        self.switch_hotkey_sequence = sequence
        self.settings.setValue("hotkey_enabled", enabled)
        self.settings.setValue("hotkey_sequence", sequence)
        self.settings.sync()
        self._register_or_unregister_switch_hotkey()
        if hasattr(self, '_settings_panel') and self._settings_panel:
            try:
                if hasattr(self._settings_panel, 'update_ui_from_settings'):
                    self._settings_panel.update_ui_from_settings()
                else:
                    logger.debug("Settings panel doesn't have update_ui_from_settings method")
            except Exception as e:
                logger.error(f"Error updating settings panel UI: {e}")
        return True

    def _handle_switch_hotkey_pressed(self):
        logger.info(f"Switch hotkey '{self.switch_hotkey_sequence}' pressed.")
        if not self.active_overlays:
            logger.info("No active overlays to switch focus to.")
            return
        overlay_widgets = list(self.active_overlays.values())
        
        # Use WindowSwitcher for validation, if available
        valid_overlays = []
        if hasattr(self, 'window_switcher') and hasattr(self.window_switcher, 'validate_window'):
            for widget in overlay_widgets:
                if hasattr(widget, 'hwnd') and widget.isVisible():
                    is_valid, reason = self.window_switcher.validate_window(widget.hwnd, for_auto_switch=True)
                    if is_valid:
                        valid_overlays.append(widget)
                    else:
                        # Only log rejected overlays at debug level
                        logger.debug(f"Overlay HWND {getattr(widget, 'hwnd', None)} rejected by window filter: {reason}")
        else:
            # Fallback: basic validation
            valid_overlays = [w for w in overlay_widgets if hasattr(w, 'hwnd') and w.isVisible()]
        
        logger.info(f"Switch hotkey: Found {len(valid_overlays)} valid overlays")
        
        if not valid_overlays:
            logger.warning("No valid overlays found for quick switch hotkey.")
            return
            
        try:
            # MRU logic: if window manager has mru_hwnds, use that order
            target_widget = None
            mru_hwnds = getattr(getattr(self, 'window_manager', None), 'mru_hwnds', None)
            
            if mru_hwnds:
                # Try to find the first overlay in MRU order
                for hwnd in mru_hwnds:
                    for widget in valid_overlays:
                        if getattr(widget, 'hwnd', None) == hwnd:
                            target_widget = widget
                            break
                    if target_widget:
                        break
            
            # If no MRU match, pick first valid overlay
            if not target_widget:
                target_widget = valid_overlays[0]
            
            # Use the correct method name: quick_swap_overlay
            if target_widget and hasattr(target_widget, 'quick_swap_overlay'):
                logger.info(f"Quick switch hotkey: triggering quick_swap_overlay on overlay for HWND: {getattr(target_widget, 'hwnd', None)}")
                target_widget.quick_swap_overlay()
            else:
                logger.warning("No suitable overlay found for quick switch hotkey.")
        except Exception as e:
            logger.error(f"Error handling switch hotkey: {e}")

    def _start_keep_alive(self, hwnd):
        """Start keep-alive for a media player window."""
        if not hasattr(self, 'keep_alive_handlers'):
            self.keep_alive_handlers = {}
            
        if hwnd not in self.keep_alive_handlers:
            try:
                handler = MediaPlayerKeepAlive(hwnd)
                if handler._is_media_player():  # Use the correct method name with underscore
                    self.keep_alive_handlers[hwnd] = handler
                    if handler.start():
                        logger.info(f"Started keep-alive for HWND: {hwnd}")
                        return True
            except Exception as e:
                logger.error(f"Error starting keep-alive for HWND {hwnd}: {e}", exc_info=True)
        return False
        
    def _stop_keep_alive(self, hwnd):
        """Stop keep-alive for a media player window."""
        if hasattr(self, 'keep_alive_handlers') and hwnd in self.keep_alive_handlers:
            try:
                self.keep_alive_handlers[hwnd].stop()
                del self.keep_alive_handlers[hwnd]
                logger.info(f"Stopped keep-alive for HWND: {hwnd}")
            except Exception as e:
                logger.error(f"Error stopping keep-alive for HWND {hwnd}: {e}", exc_info=True)
    
    def is_overlay_locked(self):
        """Check if overlays are currently locked.
        
        Returns:
            bool: True if overlays are locked, False otherwise
        """
        return getattr(self, '_overlays_locked', False)
        
    def set_overlay_lock(self, locked):
        """Set the overlay lock state.
        
        Args:
            locked (bool): Whether to lock the overlays
        """
        self._overlays_locked = bool(locked)
        logger.debug(f"Overlay lock {'enabled' if locked else 'disabled'}")
        
        # Update all active overlays with the new lock state
        if hasattr(self, 'active_overlays'):
            for overlay in self.active_overlays.values():
                if hasattr(overlay, 'update_lock_state'):
                    overlay.update_lock_state(locked)
    
    def toggle_overlay_lock(self):
        """Toggle the overlay lock state."""
        self.set_overlay_lock(not self.is_overlay_locked())
        return self._overlays_locked
    
    def _update_mru(self, hwnd):
        """Update the Most Recently Used (MRU) list with the given window handle.
        
        Args:
            hwnd: The window handle to update in the MRU list.
        """
        # Don't update MRU if overlays are locked
        if self.is_overlay_locked():
            return
            
        if not hwnd or not win32gui.IsWindow(hwnd):
            return
            
        # Don't add our own windows to MRU
        if hasattr(self, 'active_overlays'):
            for overlay in self.active_overlays.values():
                if hasattr(overlay, 'winId') and overlay.winId() == hwnd:
                    return
                    
        # Clean the MRU list before adding new items
        self.clean_mru_list()
                    
        # Remove any existing instances of this HWND
        if hwnd in self.mru_hwnds:
            self.mru_hwnds.remove(hwnd)
            # Remove old timestamp (will be re-added below)
            self._hwnd_last_focus_ts.pop(hwnd, None)
            
        # Add to front of list
        self.mru_hwnds.insert(0, hwnd)
        # Record the time this window was last focused for age-based logic
        self._hwnd_last_focus_ts[hwnd] = time.time()
        
        # Trim list if it gets too long
        if len(self.mru_hwnds) > self.MAX_MRU_ITEMS:
            self.mru_hwnds = self.mru_hwnds[:self.MAX_MRU_ITEMS]
            
        logger.debug(f"Updated MRU list. New order: {[f'HWND={h}' for h in self.mru_hwnds]}")
            
    def clean_mru_list(self):
        """Clean the MRU list by removing invalid and application windows.
        
        This ensures that our MRU list only contains valid, external windows.
        """
        # Don't clean MRU if overlays are locked
        if self.is_overlay_locked():
            return
            
        if not hasattr(self, 'mru_hwnds'):
            self.mru_hwnds = []
            return
            
        # Get a fresh list of our application's window handles
        our_windows = set()
        if hasattr(self, 'active_overlays'):
            for overlay in self.active_overlays.values():
                if hasattr(overlay, 'winId'):
                    our_windows.add(overlay.winId())
                    
        # Filter the MRU list
        cleaned_mru = []
        for hwnd in self.mru_hwnds:
            # Skip invalid windows
            if not hwnd or not win32gui.IsWindow(hwnd):
                continue
                
            # Skip our application's windows
            if hwnd in our_windows:
                continue
                
            # Skip invisible windows (but keep minimized windows in the list)
            # We keep minimized windows to allow quick switching to them
            if not win32gui.IsWindowVisible(hwnd):
                continue
                
            # Skip windows without titles
            if not win32gui.GetWindowText(hwnd):
                continue
                
            # If we got here, it's a valid window
            cleaned_mru.append(hwnd)
            
        # Update the MRU list with cleaned entries
        self.mru_hwnds = cleaned_mru

    def _on_overlay_opacity_changed(self, overlay, opacity):
        """Handle when an overlay's opacity is changed.
        
        Args:
            overlay: The overlay that changed
            opacity (float): New opacity value (0.0-1.0)
        """
        if debug_enabled():
            logger.debug(f"Overlay {overlay.hwnd} opacity changed to {opacity:.3f}")
            
        # Update the global opacity setting if this is the active overlay
        # Only update if the change is significant to avoid floating point comparison issues
        current_global_opacity = self.current_opacity_int / 100.0
        if abs(opacity - current_global_opacity) > 0.01:  # 1% threshold
            new_opacity_int = int(round(opacity * 100))
            if 0 <= new_opacity_int <= 100:
                self.current_opacity_int = new_opacity_int
                self.settings.setValue("opacity", new_opacity_int)
                if debug_enabled():
                    logger.debug(f"Updated global opacity to {new_opacity_int}% based on overlay change")

    def _on_border_opacity_changed(self, overlay, opacity):
        """Handle when an overlay's border opacity is changed.
        
        Args:
            overlay: The overlay that changed
            opacity (float): New border opacity value (0.0-1.0)
        """
        if debug_enabled():
            logger.debug(f"Overlay {overlay.hwnd} border opacity changed to {opacity:.3f}")
            
        # Update the global border opacity setting if this is the active overlay
        # Only update if the change is significant to avoid floating point comparison issues
        current_global_border_opacity = self.current_border_opacity_float
        if abs(opacity - current_global_border_opacity) > 0.01:  # 1% threshold
            self.current_border_opacity_float = opacity
            border_opacity_int = int(round(opacity * 100))
            self.settings.setValue("border_opacity", border_opacity_int)
            if debug_enabled():
                logger.debug(f"Updated global border opacity to {border_opacity_int}% based on overlay change")

    def add_overlay(self, overlay_widget):
        """Add an overlay to the application and start tracking it.
        
        Args:
            overlay_widget: The overlay widget to add
        """
        if not overlay_widget:
            logger.warning("Attempted to add None overlay widget")
            return
                
        hwnd = getattr(overlay_widget, 'hwnd', None)
        if not hwnd and hasattr(overlay_widget, 'winId'):
            try:
                hwnd = int(overlay_widget.winId())
                overlay_widget._hwnd = hwnd  # Cache the hwnd for future use
            except (AttributeError, TypeError):
                logger.warning("Could not get window ID for overlay widget")
                hwnd = id(overlay_widget)  # Fallback to object ID
                    
        if not hwnd:
            hwnd = id(overlay_widget)
            logger.warning(f"Overlay has no hwnd, using object id as fallback: {hwnd}")
                
        # Store the overlay
        self.active_overlays[hwnd] = overlay_widget
        logger.info(f"Added overlay with ID: {hwnd}, Type: {type(overlay_widget).__name__}, "
                   f"Total overlays: {len(self.active_overlays)}")
            
        # Connect opacity change signals
        if hasattr(overlay_widget, 'overlay_opacity_changed'):
            overlay_widget.overlay_opacity_changed.connect(
                lambda opacity, ov=overlay_widget: self._on_overlay_opacity_changed(ov, opacity))
                    
        if hasattr(overlay_widget, 'border_opacity_changed'):
            overlay_widget.border_opacity_changed.connect(
                lambda opacity, ov=overlay_widget: self._on_border_opacity_changed(ov, opacity))
            
        # Apply current settings to the new overlay
        if hasattr(overlay_widget, 'set_overlay_opacity'):
            # Use emit_signal=False to prevent signal loops
            overlay_widget.set_overlay_opacity(self.current_opacity_int / 100.0, emit_signal=False)
            if debug_enabled():
                logger.debug(f"Set initial overlay opacity to {self.current_opacity_int/100.0:.3f} for new overlay {hwnd}")
            
        # Apply theme if supported
        if hasattr(overlay_widget, 'apply_theme') and hasattr(self, 'current_theme'):
            overlay_widget.apply_theme(self.current_theme.lower())
                
        # Start keep-alive if this is a window overlay with a valid hwnd
        if hasattr(overlay_widget, 'hwnd') and overlay_widget.hwnd and overlay_widget.hwnd != 0:
            self._start_keep_alive(overlay_widget.hwnd)
                
        # Update MRU list
        self._update_mru(hwnd)
    
    def remove_overlay(self, overlay_widget):
        """Remove an overlay from the application.
        
        Args:
            overlay_widget: The overlay widget to remove
        """
        if not overlay_widget:
            logger.warning("Attempted to remove None overlay widget")
            return
                
        # Try to get hwnd, but don't fail if it's not available
        hwnd = None
        try:
            hwnd = getattr(overlay_widget, 'hwnd', None)
            if not hwnd and hasattr(overlay_widget, 'winId'):
                try:
                    hwnd = int(overlay_widget.winId())
                except (AttributeError, TypeError):
                    pass
                
            # If we still don't have an hwnd, try to find the overlay by object reference
            if not hwnd:
                for k, v in list(self.active_overlays.items()):
                    if v == overlay_widget:
                        hwnd = k
                        break
                            
            if hwnd in self.active_overlays:
                # Disconnect any signals first
                if hasattr(overlay_widget, 'overlay_opacity_changed'):
                    try:
                        overlay_widget.overlay_opacity_changed.disconnect()
                    except (TypeError, RuntimeError) as e:
                        if debug_enabled():
                            logger.debug(f"Error disconnecting opacity signal: {e}")
                                
                if hasattr(overlay_widget, 'border_opacity_changed'):
                    try:
                        overlay_widget.border_opacity_changed.disconnect()
                    except (TypeError, RuntimeError) as e:
                        if debug_enabled():
                            logger.debug(f"Error disconnecting border opacity signal: {e}")
                
                # Remove from active overlays
                del self.active_overlays[hwnd]
                logger.info(f"Removed overlay for HWND: {hwnd}, Remaining overlays: {len(self.active_overlays)}")
                
                # Stop keep-alive for this window
                self._stop_keep_alive(hwnd)
                
                # Clean up any associated resources
                if hasattr(overlay_widget, 'cleanup'):
                    overlay_widget.cleanup()
                        
                try:
                    overlay_widget.close()
                    overlay_widget.deleteLater()
                except Exception as e:
                    logger.error(f"Error cleaning up overlay widget: {e}")
                        
                # Force garbage collection to ensure resources are freed
                import gc
                gc.collect()
                    
            elif debug_enabled():
                logger.debug(f"Overlay HWND {hwnd} not found in active overlays")
                    
        except Exception as e:
            logger.error(f"Error in remove_overlay: {e}", exc_info=True)
    
    def update_opacity(self, opacity_int):
        """Update the overlay opacity for all active overlays.
        
        Args:
            opacity_int (int): Opacity value between 0-100
        """
        opacity_int = max(0, min(100, opacity_int))
        self.current_opacity_int = opacity_int
        opacity_float = opacity_int / 100.0
        self.set_all_overlays_opacity(opacity_float)
        self.settings.setValue("opacity", opacity_int)
        logger.debug(f"Updated overlay opacity to {opacity_int}%")
    
    def set_all_overlays_opacity(self, opacity_float):
        """Set the opacity for all active overlays.
        
        Args:
            opacity_float (float): Opacity value between 0.0-1.0
        """
        opacity_float = max(0.0, min(1.0, opacity_float))  # Clamp to valid range
        logger.debug(f"Setting opacity for all {len(self.active_overlays)} overlays to {opacity_float:.3f}")
            
        for overlay in self.active_overlays.values():
            try:
                if hasattr(overlay, 'set_overlay_opacity'):
                    # Use emit_signal=False to prevent signal loops
                    overlay.set_overlay_opacity(opacity_float, emit_signal=False)
                else:
                    overlay.setWindowOpacity(opacity_float)
            except Exception as e:
                logger.error(f"Error setting opacity on overlay {overlay}: {e}")
                if debug_enabled():
                    logger.debug(f"Overlay type: {type(overlay)}")
        
        # Save the opacity setting without triggering theme reapplication
        self.settings.setValue("overlay_opacity", opacity_float)
        # Only sync settings if we haven't done so recently
        current_time = time.time()
        if not hasattr(self, '_last_opacity_sync') or current_time - self._last_opacity_sync > 1.0:
            self.settings.sync()
            self._last_opacity_sync = current_time

    def _update_toggle_overlay_action_text(self):
        if not hasattr(self, '_toggle_toggle_action') or not self._toggle_toggle_action:
            return
        if not self.active_overlays:
            self._toggle_toggle_action.setText("No Overlay Active")
            self._toggle_toggle_action.setEnabled(False)
        else:
            any_visible = any(ov.isVisible() for ov in self.active_overlays.values())
            self._toggle_toggle_action.setText("Hide Overlay(s)" if any_visible else "Show Overlay(s)")
            self._toggle_toggle_action.setEnabled(True)

    def update_window_sort_order(self, sort_order):
        self.window_sort_order = sort_order
        self.settings.setValue("windowSortOrder", self.window_sort_order)
        logger.debug(f"Window sort order set to '{self.window_sort_order}'")
        if self._settings_panel and self._settings_panel.isVisible():
            self._settings_panel.load_windows()
        for overlay_widget in self.active_overlays.values():
            if hasattr(overlay_widget, 'update_sort_order_and_refresh_menu'):
                overlay_widget.update_sort_order_and_refresh_menu(self.window_sort_order)

    def _setup_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("No system tray available")
            return
            
        self._tray_icon = QSystemTrayIcon(self)
        
        # Use QRC resource system for the tray icon
        tray_icon = QIcon(":/Resources/ShittyPIP.ico")
        if not tray_icon.isNull():
            self._tray_icon.setIcon(tray_icon)
        else:
            logger.warning("Failed to load tray icon from resources, using fallback icon")
            self._tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
            
        # Create the tray menu
        self._tray_menu = QMenu()
        
        # Apply theme to the tray menu
        self._apply_tray_theme()
        
        # Add Main Window action (renamed from Settings)
        self._main_window_action = QAction("Main Window", self)
        self._main_window_action.triggered.connect(self._show_settings)
        self._tray_menu.addAction(self._main_window_action)
        
        # Add Subsettings action
        self._subsettings_action = QAction("Subsettings", self)
        self._subsettings_action.triggered.connect(self._show_sub_settings)
        self._tray_menu.addAction(self._subsettings_action)
        
        # Add About action
        self._about_action = QAction("About", self)
        self._about_action.triggered.connect(self._show_about_dialog)
        self._tray_menu.addAction(self._about_action)
        
        self._tray_menu.addSeparator()
        
        # Add Toggle Click-through action
        self._click_through_action = QAction("Toggle Click-through", self)
        self._click_through_action.setCheckable(True)
        # Load the current state from settings
        click_through_enabled = self.settings.value("click_through_enabled", False, type=bool)
        self._click_through_action.setChecked(click_through_enabled)
        self._click_through_action.triggered.connect(self.toggle_click_through_mode)
        self._tray_menu.addAction(self._click_through_action)
        
        self._tray_menu.addSeparator()
        
        # Add Reset action
        self._reset_action = QAction("Reset Active Overlay(s)", self)
        self._reset_action.triggered.connect(self._reset_active_overlays)
        self._tray_menu.addAction(self._reset_action)
        
        # Add Quit action
        self._quit_action = QAction("Quit", self)
        self._quit_action.triggered.connect(self.cleanup_and_quit)
        self._tray_menu.addAction(self._quit_action)
        
        # Set up the tray icon
        self._tray_icon.setContextMenu(self._tray_menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.setToolTip("PiP Overlay")
        self._tray_icon.show()
    
    def set_click_through_mode(self, enabled):
        """Set click-through mode for all active overlays.
        
        Args:
            enabled (bool): Whether to enable click-through mode
        """
        logger.info(f"Setting click-through mode to {enabled}")
        for overlay in self.active_overlays.values():
            try:
                if hasattr(overlay, 'set_click_through'):
                    overlay.set_click_through(enabled)
            except Exception as e:
                logger.error(f"Error setting click-through mode on overlay: {e}")
    
    def set_auto_switch(self, enabled):
        """Enable or disable auto-switch functionality for all active overlays.
        
        Args:
            enabled (bool): Whether to enable auto-switch
        """
        logger.info(f"Setting auto-switch to {enabled}")
        self.auto_switch_enabled = enabled
        self.settings.setValue("auto_switch_enabled", enabled)
        self.settings.sync()
        
        # Update all existing overlays
        for overlay in self.active_overlays.values():
            try:
                if hasattr(overlay, 'set_auto_switch'):
                    overlay.set_auto_switch(enabled)
                    # Reapply window flags to make changes take effect
                    overlay.show()
            except Exception as e:
                logger.error(f"Error setting auto-switch on overlay: {e}")
    
    def _apply_tray_theme(self):
        """Apply the current theme to the system tray menu with a 1.5px border."""
        if not hasattr(self, '_tray_menu') or not self._tray_menu:
            return
            
        try:
            # Get the current theme from settings or default to 'dark'
            theme = getattr(self, 'current_theme', 'dark').lower()
            
            # Define colors based on theme - matching the application's style manager
            if theme == 'light':
                # Light theme colors - matching the application's light theme
                border_color = '#000000'  # Black border
                background_color = '#f0f0f0'  # Light gray background
                text_color = '#000000'  # Black text
                highlight_color = '#7a7a7a'  # 40% darker than original (was 20%)
                highlight_text = '#ffffff'  # White text on hover (inverted)
                separator_color = '#c0c0c0'  # Light gray separator
            else:  # Dark theme
                # Dark theme colors - matching the application's dark theme
                border_color = '#ffffff'  # White border
                background_color = '#2a2a2a'  # Dark background (10% brighter than #1a1a1a)
                text_color = '#ffffff'  # White text
                highlight_color = '#3a3a3a'  # Slightly lighter than background for better visibility
                highlight_text = '#ffffff'  # White text on highlight
                separator_color = '#404040'  # Dark gray separator
            
            # Apply the theme stylesheet with border
            stylesheet = f"""
                /* Base menu styling */
                QMenu {{
                    background-color: {background_color};
                    color: {text_color};
                    border: 1px solid {border_color};
                    padding: 1px;
                    min-width: 120px;
                    font-weight: 500;
                }}
                
                /* Menu item styling */
                QMenu::item {{
                    padding: 4px 13px 4px 8px;  /* Added 1px more left, removed 2px right */
                    border: none;
                    min-width: 100px;
                    max-width: 200px;
                    text-align: left;
                    font-weight: 500;
                    margin: 0;
                    spacing: 3px;  /* Reduced from 6px to 3px (50% less) */
                }}
                
                /* Checkmark circle - Qt compatible */
                QMenu::indicator {{
                    width: 6px;  /* 40% smaller than previous 8px */
                    height: 6px;  /* 40% smaller than previous 8px */
                    border: 1px solid {text_color};
                    border-radius: 3px;  /* Slightly less rounded for the smaller size */
                    background: transparent;
                    margin-left: 2px;  /* Added 1px more left margin (total 2px) */
                    margin-right: -1px;  /* Pull checkmark 1px to the left */
                }}
                
                QMenu::indicator:checked {{
                    background: {text_color};
                }}
                
                QMenu::indicator:checked {{
                    background: {text_color};
                }}
                
                /* Ensure checkmark is visible in hover state */
                QMenu::item:selected QMenu::indicator {{
                    border-color: {highlight_text};
                }}
                
                QMenu::item:selected QMenu::indicator:checked {{
                    background: {highlight_text};
                }}
                
                /* Hover state */
                QMenu::item:selected {{
                    background-color: {highlight_color};
                    color: {highlight_text};
                }}
                
                /* Ensure checkmark is visible in hover state */
                QMenu::item:selected QMenu::indicator {{
                    border-color: {highlight_text};
                }}
                
                QMenu::item:selected QMenu::indicator:checked {{
                    background: {highlight_text};
                }}
                
                /* Disabled items */
                QMenu::item:disabled {{
                    color: #888888;
                }}
                
                /* Separator */
                QMenu::separator {{
                    height: 1px;
                    background: {separator_color};
                    margin: 2px 4px;
                }}
            """
            self._tray_menu.setStyleSheet(stylesheet)
            logger.debug(f"Applied {theme} theme to system tray menu")
        except Exception as e:
            logger.error(f"Error applying tray theme: {e}", exc_info=True)
            
    def toggle_click_through_mode(self):
        """Toggle click-through mode for all overlays and update the UI."""
        if not hasattr(self, 'settings'):
            return
            
        try:
            # Toggle the current state
            current = self.settings.value("click_through_enabled", False, type=bool)
            new_state = not current
            
            # Update the setting
            self.settings.setValue("click_through_enabled", new_state)
            
            # Block signals while updating UI elements to prevent feedback loops
            if hasattr(self, '_click_through_action'):
                self._click_through_action.blockSignals(True)
                self._click_through_action.setChecked(new_state)
                self._click_through_action.blockSignals(False)
                
            # Update the sub-settings dialog if it's open
            if hasattr(self, '_sub_settings_dialog') and self._sub_settings_dialog:
                if hasattr(self._sub_settings_dialog, 'click_through_checkbox'):
                    checkbox = self._sub_settings_dialog.click_through_checkbox
                    if checkbox:
                        checkbox.blockSignals(True)
                        checkbox.setChecked(new_state)
                        checkbox.blockSignals(False)
            
            # Apply the new click-through state to all overlays
            self.set_click_through_mode(new_state)
            
            logger.info(f"Click-through mode {'enabled' if new_state else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Error in toggle_click_through_mode: {e}", exc_info=True)
    
    def _on_settings_changed(self, key):
        """Handle settings changes to update the UI accordingly."""
        if key == "UI/theme":
            # Update the current theme from settings
            theme = self.settings.value("UI/theme", "dark", type=str).lower()
            if hasattr(self, 'current_theme') and self.current_theme == theme:
                return  # No change
                
            logger.debug(f"Theme changed to: {theme}")
            self.current_theme = theme
            
            # Apply the theme globally (which will also update the tray menu)
            self.apply_theme_globally(theme)
            
            # Force update the tray menu if it exists
            if hasattr(self, '_tray_menu') and self._tray_menu:
                self._tray_menu.hide()  # Force refresh by hiding and showing
                self._tray_menu.show()
            
    def set_setting(self, key, value):
        """Helper method to set a setting and update the UI if needed."""
        if hasattr(self, 'settings') and self.settings:
            self.settings.setValue(key, value)
            self._on_settings_changed(key)

    def _toggle_overlay_visibility(self):
        if not self.active_overlays:
            logger.debug("No active overlays to toggle visibility.")
            return
        any_visible = any(overlay.isVisible() for overlay in self.active_overlays.values())
        if any_visible:
            logger.debug(f"Hiding all {len(self.active_overlays)} active overlays.")
            self._hide_all_overlays()
        else:
            logger.debug(f"Showing all {len(self.active_overlays)} active overlays.")
            self._show_all_overlays()
        self._update_toggle_overlay_action_text()

    def _hide_all_overlays(self):
        try:
            for ov in self.active_overlays.values():
                if ov and not ov.isHidden():
                    ov.hide()
        except Exception as e:
            logger.error(f"Error hiding overlays: {str(e)}")

    def _show_all_overlays(self):
        try:
            for ov in self.active_overlays.values():
                if ov and ov.isHidden():
                    ov.show()
        except Exception as e:
            logger.error(f"Error showing overlays: {str(e)}")

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_settings()
            
    def _reset_active_overlays(self):
        """
        Reset the position and size of all active overlays to their default positions
        and ensure they are brought to the front of the window stack.
        """
        if not hasattr(self, 'active_overlays') or not self.active_overlays:
            logger.warning("No active overlays to reset")
            return
            
        logger.info("Resetting all active overlays")
        
        # Get the current click-through setting
        click_through_enabled = self.settings.value("click_through_enabled", False, type=bool)
        
        for hwnd, overlay in list(self.active_overlays.items()):
            try:
                if not overlay or not overlay.isVisible():
                    continue
                    
                # Save the current window state
                was_maximized = overlay.isMaximized()
                was_minimized = overlay.isMinimized()
                
                # Reset position and size using the correct method
                if hasattr(overlay, '_handle_reset_position'):
                    overlay._handle_reset_position()
                
                # Re-apply the window flags to ensure proper layering
                flags = (Qt.WindowStaysOnTopHint | 
                        Qt.FramelessWindowHint | 
                        (Qt.WindowTransparentForInput if click_through_enabled else 0))
                overlay.setWindowFlags(flags)
                
                # Restore window state
                if was_maximized:
                    overlay.showMaximized()
                elif was_minimized:
                    overlay.showMinimized()
                else:
                    overlay.showNormal()
                    
                # Ensure the window is on top and visible
                overlay.raise_()
                overlay.activateWindow()
                overlay.show()
                
                logger.debug(f"Reset overlay for window {hwnd}")
                
            except Exception as e:
                logger.error(f"Error resetting overlay for window {hwnd}: {e}", exc_info=True)
                # If the overlay is in a bad state, remove it
                self.active_overlays.pop(hwnd, None)
                try:
                    overlay.close()
                    overlay.deleteLater()
                except (RuntimeError, AttributeError) as e:
                    # Handle cases where the overlay is already deleted or in an invalid state
                    logger.debug(f"Error cleaning up overlay: {e}")
                except Exception as e:
                    # Catch any other exceptions to prevent crashes during cleanup
                    logger.warning(f"Unexpected error during overlay cleanup: {e}")

    def cleanup_active_overlays(self):
        """Clean up all active overlays before creating a new one."""
        try:
            logger.debug("Cleaning up active overlays...")
            # Get a list of all overlay widgets to avoid modifying the list during iteration
            overlays = list(getattr(self, 'active_overlays', {}).values())
            for overlay in overlays:
                try:
                    if hasattr(overlay, 'close'):
                        overlay.close()
                    if hasattr(overlay, 'deleteLater'):
                        overlay.deleteLater()
                    logger.debug(f"Scheduled overlay for deletion: {overlay}")
                except Exception as e:
                    logger.error(f"Error cleaning up overlay {overlay}: {e}", exc_info=True)
            
            # Clear the active overlays dictionary if it exists
            if hasattr(self, 'active_overlays'):
                self.active_overlays.clear()
                
            # Force garbage collection to ensure resources are freed
            import gc
            gc.collect()
            
            # Update UI if needed
            if hasattr(self, '_update_toggle_overlay_action_text'):
                self._update_toggle_overlay_action_text()
                
            logger.debug("Finished cleaning up active overlays")
            
        except Exception as e:
            logger.error(f"Error in cleanup_active_overlays: {e}", exc_info=True)

    def _cleanup_all_active_overlays(self):
        """Legacy method - use cleanup_active_overlays instead."""
        logger.warning("Using deprecated _cleanup_all_active_overlays - use cleanup_active_overlays instead")
        self.cleanup_active_overlays()

    def prepare_to_create_window_overlay(self, hwnd, initial_geometry=None):
        logger.info(f"Preparing to create window overlay for HWND: {hwnd}")
        if not self.window_enumerator._is_valid_window(hwnd):
            logger.error(f"Target window {hwnd} is invalid.")
            return
            
        # Check if we already have an active overlay for this window
        for overlay_hwnd, overlay in list(self.active_overlays.items()):
            if hasattr(overlay, 'target_hwnd') and overlay.target_hwnd == hwnd:
                logger.info(f"Found existing overlay for HWND {hwnd}, updating content")
                overlay.update_content(hwnd)
                return
        
        # Clean up any existing overlays before creating a new one
        self._cleanup_all_active_overlays()
        
        # Create a new overlay after a short delay
        gc.collect()
        QTimer.singleShot(50, lambda: self.actually_create_window_overlay(hwnd, initial_geometry))

    def actually_create_window_overlay(self, hwnd, initial_geometry=None):
        if not self.window_enumerator._is_valid_window(hwnd):
            logger.error(f"Target window {hwnd} is invalid.")
            return
            
        logger.info(f"Creating window overlay for HWND: {hwnd}")
        try:
            # Get the cursor position to determine which screen has focus
            cursor_pos = QCursor.pos()
            target_screen = QGuiApplication.screenAt(cursor_pos)
            
            # Fallback to primary screen if no screen at cursor position
            if not target_screen:
                target_screen = QGuiApplication.primaryScreen()
                logger.warning(f"Could not determine screen at cursor position {cursor_pos}, using primary screen")
            
            # Get window info
            window_title = win32gui.GetWindowText(hwnd)
            logger.info(f"Creating overlay for window: {window_title} on screen: {target_screen.name()}")
            
            # Get the monitor index for the target screen
            screens = QGuiApplication.screens()
            monitor_idx = screens.index(target_screen) if target_screen in screens else 0
            
            # Load the saved preset for this monitor
            preset_key = f"MonitorPresets/Monitor_{monitor_idx}_Preset"
            position_preset = self.settings.value(preset_key, DEFAULT_POSITION_PRESET)
            logger.info(f"Using position preset '{position_preset}' for monitor {monitor_idx}")
            
            # Calculate the initial geometry using the saved preset
            final_initial_geometry = self.calculate_position_geometry(target_screen, position_preset, "window")
            logger.info(f"Calculated initial window overlay geometry: {final_initial_geometry}")
            
            # Create the overlay widget with the current opacity setting
            new_overlay = BorderWidget(
                hwnd=hwnd, 
                opacity=self.current_opacity_int, 
                theme=self.current_theme,
                app_instance=self, 
                initial_geometry=final_initial_geometry
            )
            
            # Initialize the overlay with the current auto-switch setting
            new_overlay.set_auto_switch(self.auto_switch_enabled)
            logger.debug(f"Initialized overlay auto-switch to {self.auto_switch_enabled}")
            
            # Check if this is the desktop overlay
            try:
                if hwnd == self.progman_hwnd:
                    new_overlay.is_desktop_overlay = True
                    logger.info("Created desktop overlay with special handling")
            except Exception as e:
                logger.error(f"Error checking for desktop overlay: {e}", exc_info=True)
            
            if new_overlay.register_thumbnail():
                # Explicitly set the opacity before showing the window
                new_overlay.set_overlay_opacity(new_overlay.opacity, emit_signal=False)
                
                # Add to active overlays before showing to ensure proper signal connections
                self.add_overlay(new_overlay)
                
                # Show the window
                new_overlay.show()
                
                # Ensure the window is properly positioned on the target screen
                new_overlay.ensure_in_monitor_bounds(target_screen)
                
                logger.info(f"Created window overlay for '{window_title}' (HWND: {hwnd}) on screen {target_screen.name()}")
                
                # Force garbage collection to clean up any unused resources
                gc.collect()
                
                # Bring the overlay to the front
                new_overlay.raise_()
                new_overlay.activateWindow()
                
                # Emit opacity signal to ensure UI updates
                logger.debug(f"Emitting opacity signal for new overlay (HWND: {hwnd})")
                new_overlay.overlay_opacity_changed.emit(new_overlay.opacity)
                logger.debug(f"Emitted signal - Overlay Opacity: {new_overlay.opacity}")
                
                # Force an immediate repaint to ensure the opacity is applied
                new_overlay.update()
                
                return new_overlay
            else:
                logger.error(f"Failed to register DWM background for '{window_title}' (HWND: {hwnd}).")
                new_overlay.deleteLater()
                return None
                
        except Exception as e:
            logger.error(f"Error creating window overlay for HWND {hwnd}: {e}", exc_info=True)
            return None

    def get_menu_ready_windows(self):
        if not self.window_enumerator:
            logger.error("Window enumerator not available.")
            return []
        try:
            # Get all windows with icons
            windows = self.window_enumerator.get_capturable_windows_with_icons()
            logger.debug(f"Raw windows from enumerator: {[w[0] for w in windows]}")
            
            # Get our application's window titles and class names to exclude
            our_titles = {
                'Shitty PiP QuickSwap',
                'Settings',
                'Sub-settings',
                'Overlay',
                'Monitor Overlay',
                'Window Overlay'
            }
            our_classes = {
                'Qt5QWindowIcon',  # Common Qt window class
                'QWidget'          # Another common Qt window class
            }
            
            # Get our process ID to filter out our own windows
            our_pid = os.getpid()
            filtered_windows = []
            
            for hwnd, title, icon in windows:
                try:
                    # Skip if window title matches any of our known titles
                    if any(our_title in title for our_title in our_titles):
                        continue
                        
                    # Get window class name
                    class_name = win32gui.GetClassName(hwnd)
                    if class_name in our_classes:
                        continue
                        
                    # Get window process ID and skip if it's our own process
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    if pid == our_pid:
                        continue
                        
                    # If we got here, include the window
                    filtered_windows.append((hwnd, title, icon))
                    
                except Exception as e:
                    logger.debug(f"Error checking window {hwnd}: {e}")
                    continue
            
            logger.debug(f"Filtered windows: {len(filtered_windows)} out of {len(windows)}")
            return filtered_windows
            
        except Exception as e:
            logger.error(f"Error getting menu-ready windows: {e}", exc_info=True)
            return []

    def prepare_to_create_monitor_overlay(self, screen):
        try:
            if not screen:
                logger.error("No valid screen provided to prepare_to_create_monitor_overlay")
                return
            
            # Get detailed monitor info using monitor_utils
            monitor_info = monitor_utils.get_physical_monitor_info(screen)
            
            # Get screen name and index while we still have a valid QScreen reference
            screens = QGuiApplication.screens()
            screen_idx = screens.index(screen) if screen in screens else -1
            screen_name = monitor_info.get('device_name', screen.name() if hasattr(screen, 'name') else f"Display {screen_idx + 1}")
            logger.info(f"Preparing to create overlay for screen: {screen_name}")
            
            # Get screen geometry from monitor_info if available, otherwise use Qt's geometry
            if 'geometry' in monitor_info and monitor_info['geometry']:
                screen_geo = monitor_info['geometry']
            else:
                screen_geo = screen.geometry()
                
            logger.info(f"  Geometry: {screen_geo.width()}x{screen_geo.height()} @ ({screen_geo.x()},{screen_geo.y()})")
            
            # Get monitor index using monitor_utils
            monitor_idx = monitor_info.get('monitor_index', screen_idx)
            
            # Prepare screen info without storing the QScreen object
            screen_info = {
                # Store screen identification instead of the QScreen object
                'screen_name': screen_name,
                'screen_idx': screen_idx,
                'monitor_idx': monitor_idx,
                'geometry': screen_geo,  # Store geometry as a QRect, not a reference to screen.geometry()
                'is_primary': monitor_info.get('primary', False),
                'dpi': monitor_info.get('dpi', QSizeF(96, 96)),  # Store as QSizeF for consistency
                'scale_factor': monitor_info.get('scale_factor', 1.0),
                'physical_width': monitor_info.get('physical_width', screen_geo.width()),
                'physical_height': monitor_info.get('physical_height', screen_geo.height())
            }
            
            # Clean up any existing overlays for this screen
            self.cleanup_active_overlays()
            
            # Store the screen info as an instance variable to prevent garbage collection
            self._pending_screen_info = screen_info
            
            # Use a weak reference to self to avoid circular references
            from weakref import ref
            weak_self = ref(self)
            
            def create_overlay():
                strong_self = weak_self()
                if strong_self and hasattr(strong_self, '_pending_screen_info'):
                    screen_info = strong_self._pending_screen_info
                    try:
                        strong_self.actually_create_monitor_overlay(screen_info)
                    except Exception as e:
                        logger.error(f"Error creating monitor overlay: {e}", exc_info=True)
                        QMessageBox.critical(None, "Error", f"Failed to create monitor overlay: {str(e)}")
                    finally:
                        # Clean up the reference after use
                        if hasattr(strong_self, '_pending_screen_info'):
                            del strong_self._pending_screen_info
            
            # Schedule the actual creation with a small delay to ensure cleanup is complete
            QTimer.singleShot(50, create_overlay)
            
        except Exception as e:
            logger.error(f"Error in prepare_to_create_monitor_overlay: {e}", exc_info=True)
            QMessageBox.critical(None, "Error", f"Failed to prepare monitor overlay: {str(e)}")
    
    def actually_create_monitor_overlay(self, screen_info):
        new_overlay = None
        try:
            if not screen_info:
                logger.error("No screen info provided to actually_create_monitor_overlay")
                return
            
            # Get all screens to ensure we have the latest references
            screens = QGuiApplication.screens()
            if not screens:
                logger.error("No screens found!")
                return
            
            # Get screen by index if available, otherwise use primary screen
            screen_idx = screen_info.get('screen_idx', -1)
            screen_name = screen_info.get('name', 'Unknown')
            
            # Try to find the screen by index first
            if 0 <= screen_idx < len(screens):
                target_screen = screens[screen_idx]
                logger.info(f"Using screen at index {screen_idx}: {screen_name}")
            else:
                # Fall back to primary screen if index is invalid
                target_screen = QGuiApplication.primaryScreen() or screens[0]
                logger.warning(f"Invalid screen index {screen_idx}, using primary screen: {target_screen.name() if hasattr(target_screen, 'name') else 'unnamed'}")
            
            # Get the screen geometry from screen_info if available, otherwise use the current screen geometry
            screen_geo = screen_info.get('geometry', target_screen.geometry())
            
            # Log screen information
            logger.info(f"Creating overlay for screen: {screen_name}")
            logger.info(f"  Geometry: {screen_geo.width()}x{screen_geo.height()} @ ({screen_geo.x()},{screen_geo.y()})")
            
            # Log DPI and scaling information
            dpi = target_screen.logicalDotsPerInch()
            device_pixel_ratio = target_screen.devicePixelRatio()
            logger.info(f"Screen DPI: {dpi:.1f}, Device Pixel Ratio: {device_pixel_ratio:.2f}")
            
            # Get the monitor index for the target screen
            monitor_idx = screens.index(target_screen) if target_screen in screens else 0
            
            # Load the saved preset for this monitor
            preset_key = f"MonitorPresets/Monitor_{monitor_idx}_Preset"
            position_preset = self.settings.value(preset_key, DEFAULT_POSITION_PRESET)
            logger.info(f"Using position preset '{position_preset}' for monitor {monitor_idx}")
            
            # Calculate the initial geometry using the saved preset
            final_initial_geometry = self.calculate_position_geometry(
                target_screen, position_preset, "monitor"
            )
            
            # Log the calculated geometry
            if final_initial_geometry:
                logger.info(f"Calculated initial monitor overlay geometry: {final_initial_geometry.width()}x{final_initial_geometry.height()} @ ({final_initial_geometry.x()},{final_initial_geometry.y()})")
            
            # Fallback to default geometry if still not set
            if not final_initial_geometry:
                logger.warning("Failed to calculate initial geometry, using default")
                screen_geo = target_screen.availableGeometry()
                final_initial_geometry = QRect(
                    screen_geo.x() + screen_geo.width() // 4,
                    screen_geo.y() + screen_geo.height() // 4,
                    screen_geo.width() // 2,
                    screen_geo.height() // 2
                )
            
            # Create the new overlay
            screens = QGuiApplication.screens()
            monitor_idx = screens.index(target_screen) if target_screen in screens else 0
            
            logger.info(f"Creating new MonitorOverlay for monitor {monitor_idx} ({target_screen.name()})")
            
            # Create the overlay with the app instance and settings
            new_overlay = MonitorOverlay(
                screen=target_screen,  # Pass the target screen directly
                app_instance=self,
                initial_geometry=final_initial_geometry,
                opacity=1.0,  # Default opacity
                theme=self.current_theme.lower(),
                snap_distance=8,  # Default snap distance
                monitor_idx=monitor_idx  # Pass the monitor index
            )
            
            # Set initial theme and opacity
            new_overlay.apply_theme(self.current_theme.lower())
            new_overlay.set_overlay_opacity(self.current_opacity_int / 100.0)
            
            # Initialize the overlay with the current auto-switch setting
            if hasattr(new_overlay, 'set_auto_switch'):
                new_overlay.set_auto_switch(self.auto_switch_enabled)
                logger.debug(f"Initialized monitor overlay auto-switch to {self.auto_switch_enabled}")
            
            # Add to active overlays and show
            self.add_overlay(new_overlay)
            new_overlay.show()
            
            # Ensure the overlay is within the screen bounds
            new_overlay.ensure_in_monitor_bounds()
            logger.info(f"Successfully created monitor overlay for screen: {target_screen.name()}")
            
        except Exception as e:
            logger.error(f"Error creating monitor overlay: {e}", exc_info=True)
            if new_overlay:
                try:
                    new_overlay.deleteLater()
                except Exception as del_error:
                    logger.error(f"Error cleaning up failed overlay: {del_error}")
            return
        finally:
            gc.collect()

    def calculate_position_geometry(self, screen, position_preset, overlay_type):
        try:
            # Default to top-left corner if position_preset is not provided
            if not position_preset:
                position_preset = "TopLeft"
            
            # Get monitor info using monitor_utils
            monitor_info = monitor_utils.get_physical_monitor_info(screen)
            
            # Get the screen geometry in logical pixels
            screen_geo = screen.geometry()
            
            if monitor_info and 'physical_width' in monitor_info and 'physical_height' in monitor_info:
                # Use physical dimensions from monitor_utils
                monitor_width = monitor_info['physical_width']
                monitor_height = monitor_info['physical_height']
                
                # Get scale factor from monitor_utils or calculate from geometry
                scale_factor = monitor_info.get('scale_factor', 1.0)
                
                # Calculate overlay size based on monitor dimensions
                if overlay_type == "monitor":
                    # For monitor overlays, use a percentage of the physical size
                    w = int(monitor_width * DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR / scale_factor)
                    h = int(monitor_height * DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR / scale_factor)
                else:  # window overlay
                    # For window overlays, use fixed size with scaling
                    w = int(min(DEFAULT_WINDOW_OVERLAY_WIDTH / scale_factor, screen_geo.width() * 0.9))
                    h = int(min(DEFAULT_WINDOW_OVERLAY_HEIGHT / scale_factor, screen_geo.height() * 0.9))
                
                # Calculate position based on preset
                x, y = screen_geo.left(), screen_geo.top()
                if "Right" in position_preset:
                    x = screen_geo.right() - w
                if "Bottom" in position_preset:
                    y = screen_geo.bottom() - h
                if position_preset == "Centered":
                    x = screen_geo.left() + (screen_geo.width() - w) // 2
                    y = screen_geo.top() + (screen_geo.height() - h) // 2
                
                logger.debug(f"Using physical monitor dimensions: {monitor_width}x{monitor_height}, scale: {scale_factor:.2f}")
                return QRect(x, y, w, h)
            
            # Fallback to using Qt screen geometry if monitor_utils fails
            logger.debug("Falling back to Qt screen geometry for overlay positioning")
            geo = screen_geo
            if overlay_type == "monitor":
                w = int(geo.width() * DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR)
                h = int(geo.height() * DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR)
            else:  # window overlay
                w = min(DEFAULT_WINDOW_OVERLAY_WIDTH, int(geo.width() * 0.9))
                h = min(DEFAULT_WINDOW_OVERLAY_HEIGHT, int(geo.height() * 0.9))
            
            x, y = geo.left(), geo.top()
            if "Right" in position_preset:
                x = geo.right() - w
            if "Bottom" in position_preset:
                y = geo.bottom() - h
            if position_preset == "Centered":
                x = geo.left() + (geo.width() - w) // 2
                y = geo.top() + (geo.height() - h) // 2
            
            return QRect(x, y, w, h)
            
        except Exception as e:
            logger.error(f"Error in calculate_position_geometry: {e}", exc_info=True)
            return QRect(100, 100, 800, 600)

    def _show_about_dialog(self):
        """Show the about dialog."""
        try:
            from about_dialog import AboutDialog
            dialog = AboutDialog(self)
            dialog.exec()
        except Exception as e:
            logger.error(f"Error showing about dialog: {e}", exc_info=True)
            
    def _setup_foreground_event_hook(self):
        try:
            EVENT_SYSTEM_FOREGROUND = 0x0003
            WINEVENT_OUTOFCONTEXT = 0x0000
            @WinEventProcType
            def win_event_proc(hWinEventHook, event, hwnd, idObject, idChild, dwEventThread, dwmsEventTime):
                if event == EVENT_SYSTEM_FOREGROUND and hwnd:
                    try:
                        self.foregroundWindowChanged.emit(hwnd)
                        logger.debug(f"Foreground window changed: HWND={hwnd}")
                    except Exception as e:
                        logger.error(f"Error in win_event_proc for HWND {hwnd}: {e}")
            self._win_event_proc_callback_ptr = win_event_proc
            self._win_event_hook = windll.user32.SetWinEventHook(
                EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, 0,
                self._win_event_proc_callback_ptr, 0, 0, WINEVENT_OUTOFCONTEXT
            )
            if self._win_event_hook:
                logger.info("Set up foreground window event hook.")
                self.foregroundWindowChanged.connect(self._handle_foreground_window_change)
            else:
                logger.error("Failed to set up foreground window event hook.")
        except Exception as e:
            logger.error(f"Error setting up foreground event hook: {e}", exc_info=True)

    def _handle_foreground_window_change(self, hwnd):
        """Handle window foreground changes and implement auto-switch functionality.
        
        When a window that's being captured by an overlay is focused, this will
        automatically switch to the previous active window from the MRU list.
        """
        if not hwnd:
            return
            
        try:
            # Skip if it's our own window
            pid = ctypes.wintypes.DWORD()
            windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == os.getpid():
                logger.debug(f"Foreground window is our own process (HWND={hwnd}).")
                return
                
            # Skip if this is one of our application's windows
            our_pid = os.getpid()
            window_pid = ctypes.wintypes.DWORD()
            windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
            
            if window_pid.value == our_pid:
                logger.debug(f"Skipping our own window (PID: {our_pid}) from MRU list: HWND={hwnd}")
                return
                
            # Update MRU list for external windows only, but only keep entries WindowFilter accepts
            from window_filter import WindowFilter
            if not WindowFilter.is_valid_window(hwnd, our_pid=os.getpid()):
                logger.debug(f"Skipping adding invalid window to MRU: HWND={hwnd}")
            else:
                if hwnd in self.mru_hwnds:
                    self.mru_hwnds.remove(hwnd)
                    logger.debug(f"Moved external window to top of MRU list: HWND={hwnd}")
                else:
                    logger.debug(f"Added external window to MRU list: HWND={hwnd}")
                self.mru_hwnds.insert(0, hwnd)
                # Record focus timestamp for age-based logic
                self._hwnd_last_focus_ts[hwnd] = time.time()
                # Record last focus timestamp
                self._hwnd_last_focus_ts[hwnd] = time.time()
            if len(self.mru_hwnds) > self.MAX_MRU_ITEMS:
                self.mru_hwnds = self.mru_hwnds[:self.MAX_MRU_ITEMS]
                
            self.last_external_focused_hwnd = hwnd
            
                # --- Removed experimental branch that attempted to auto-switch when focused window was not captured ---
            if hasattr(self, 'active_overlays') and hwnd not in self.active_overlays:
                if self.auto_switch_enabled and self.active_overlays:
                    # Iterate over active overlays and switch the first one that supports auto-switch
                    for overlay_widget in self.active_overlays.values():
                        if getattr(overlay_widget, 'auto_switch', False):
                            if getattr(overlay_widget, 'hwnd', None) == hwnd:
                                # Overlay already shows this window
                                break
                            logger.debug(f"Auto-switching overlay to newly focused window {hwnd}")
                            try:
                                switched = False
                                if hasattr(self, 'window_switcher'):
                                    switched = self.window_switcher.switch_to_window(hwnd, current_overlay=overlay_widget)
                                else:
                                    win32gui.SetForegroundWindow(hwnd)
                                    switched = True
                                if switched:
                                    return  # Successful switch, no further processing needed
                            except Exception as e:
                                logger.error(f"Error auto-switching overlay to hwnd {hwnd}: {e}")
                            break  # Only attempt on one overlay to avoid multiple switches

            # Check if the focused window is being captured by any overlay
            if hasattr(self, 'active_overlays') and hwnd in self.active_overlays:
                if not self.auto_switch_enabled:
                    logger.debug(f"Auto-switch is disabled, not switching from captured window {hwnd}")
                    return
                    
                logger.debug(f"Focused window {hwnd} is being captured, finding previous window to switch to")
                
                # Iterate through MRU list and attempt to switch to up to 4 *valid* candidates
                MAX_AUTO_SWITCH_ATTEMPTS = 8  # Try up to 8 valid MRU candidates before giving up
                attempts = 0
                for idx in range(1, len(self.mru_hwnds)):
                    if attempts >= MAX_AUTO_SWITCH_ATTEMPTS:
                        break  # Reached attempt limit

                    target_hwnd = self.mru_hwnds[idx]

                    # Basic sanity checks and WindowFilter validation
                    from window_filter import WindowFilter

                    # Skip windows that WindowFilter deems invalid (e.g., taskbar, popup)
                    if not WindowFilter.is_valid_window(target_hwnd, our_pid=os.getpid()):
                        logger.debug(f"Skipping window HWND {target_hwnd} rejected by WindowFilter during auto-switch")
                        # Optional: prune invalid entry to keep MRU clean
                        try:
                            self.mru_hwnds.remove(target_hwnd)
                            self._hwnd_last_focus_ts.pop(target_hwnd, None)
                        except ValueError:
                            pass
                        continue

                # Age-out check: skip candidates not focused within last 8 seconds
                    last_ts = self._hwnd_last_focus_ts.get(target_hwnd)
                    is_stale = False
                    if last_ts:
                        is_stale = (time.time() - last_ts) > 8
                    # Skip stale candidates only if fresher alternatives remain ahead in the list
                    if is_stale and idx < len(self.mru_hwnds) - 1:
                        logger.debug(f"Deferring stale MRU window HWND {target_hwnd} (last focused {time.time()-last_ts:.1f}s ago)")
                        continue
                    # Remaining sanity checks
                    if not target_hwnd or target_hwnd == hwnd:
                        continue  # Same as current focused window
                    if not win32gui.IsWindow(target_hwnd):
                        logger.debug(f"Skipping invalid window HWND {target_hwnd}")
                        continue

                    # Skip our own application windows
                    windll.user32.GetWindowThreadProcessId(target_hwnd, ctypes.byref(pid))
                    if pid.value == os.getpid():
                        logger.debug(f"Skipping our own window HWND {target_hwnd}")
                        continue

                    # Skip minimized windows
                    if win32gui.IsIconic(target_hwnd):
                        logger.debug(f"Skipping minimized window HWND {target_hwnd}")
                        continue

                    # Skip if overlay already shows this HWND (would appear as no-op)
                    overlay_widget = self.active_overlays.get(hwnd)
                    if overlay_widget is not None and getattr(overlay_widget, 'hwnd', None) == target_hwnd:
                        logger.debug(f"Skipping window HWND {target_hwnd} because overlay already displays it")
                        continue

                    logger.debug(f"Auto-switching to previous window {target_hwnd}")

                    try:
                        if hasattr(self, 'window_switcher'):
                            attempts += 1  # Count this as an actual switch attempt
                            switched = self.window_switcher.switch_to_window(target_hwnd, current_overlay=overlay_widget)
                            if switched:
                                # Update active_overlays dictionary key to new target HWND for consistency
                                try:
                                    if hwnd in self.active_overlays:
                                        self.active_overlays[target_hwnd] = self.active_overlays.pop(hwnd)
                                except Exception as map_err:
                                    logger.error(f"Failed to remap active_overlays key: {map_err}")
                                return  # Success – stop searching further MRUs
                            logger.debug(f"Switch to HWND {target_hwnd} failed, trying next MRU candidate")
                        else:
                            win32gui.SetForegroundWindow(target_hwnd)
                            return  # Assume success if no exception
                    except Exception as e:
                        logger.error(f"Error switching to window {target_hwnd}: {e}")
                        attempts += 1  # Count failed attempt
                        # Continue to next MRU candidate
                        continue
                
                logger.debug("No valid previous window found to switch to")
                
        except Exception as e:
            logger.error(f"Error handling foreground window change for HWND {hwnd}: {e}", exc_info=True)

    def handle_focus_change(self, old, new):
        logger.debug(f"Qt focus changed: Old={old}, New={new}")

    def handle_application_state_change(self, state):
        logger.debug(f"Application state changed: {state}")

    def cleanup_and_quit(self):
        """Clean up resources before quitting the application."""
        try:
            # Unregister hotkeys
            self._unregister_switch_hotkey()
            self._unregister_opacity_hotkeys()
            if hasattr(self, 'keep_alive_handlers'):
                for hwnd in list(self.keep_alive_handlers.keys()):
                    self._stop_keep_alive(hwnd)
            
            self._cleanup_all_active_overlays()
            
            # Clean up the thumbnail preview if it exists
            if hasattr(self, 'thumbnail_preview') and self.thumbnail_preview:
                try:
                    self.thumbnail_preview.close()
                    self.thumbnail_preview.deleteLater()
                except Exception as e:
                    logger.error(f"Error cleaning up thumbnail preview: {e}", exc_info=True)
            
            # Remove the foreground window hook
            if hasattr(self, '_win_event_hook') and self._win_event_hook is not None:
                try:
                    # Use windll.user32.UnhookWinEvent instead of win32gui
                    result = ctypes.windll.user32.UnhookWinEvent(self._win_event_hook)
                    if result == 0:  # Function failed
                        error_code = ctypes.windll.kernel32.GetLastError()
                        if error_code != 0:  # 0 means success, non-zero is an error
                            logger.error(f"Failed to unhook win event: Windows error {error_code}")
                    self._win_event_hook = None
                    logger.debug("Successfully unregistered Windows event hook")
                except Exception as e:
                    logger.error(f"Error removing win event hook: {e}", exc_info=True)
                finally:
                    self._win_event_hook = None  # Ensure it's always cleared
            
            # Clean up the keyboard hook if it exists
            if hasattr(self, '_keyboard_hook') and self._keyboard_hook is not None:
                try:
                    self._keyboard_hook.unhook_all()
                except Exception as e:
                    logger.error(f"Error removing keyboard hook: {e}", exc_info=True)
            
            # Save settings
            try:
                if hasattr(self, 'settings') and self.settings is not None:
                    self.settings.sync()
                    logger.info("Settings saved successfully")
            except Exception as e:
                logger.error(f"Error saving settings: {e}", exc_info=True)
            
            logger.info("Cleanup complete. Exiting application.")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
        finally:
            # Ensure we always quit using QApplication.instance()
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def get_preset_geometry(self, overlay_type, screen_name):
        """
        Get the saved preset geometry for a given overlay type and screen name.
        
        Args:
            overlay_type (str): Type of overlay ("window" or "monitor")
            screen_name (str): Name of the screen to get the preset for
            
        Returns:
            QRect: The calculated geometry for the overlay, or None if not found
        """
        try:
            # Find the screen by name
            target_screen = None
            for screen in QGuiApplication.screens():
                if screen.name() == screen_name:
                    target_screen = screen
                    break
                    
            if not target_screen:
                logger.warning(f"Screen '{screen_name}' not found, using primary screen")
                target_screen = QGuiApplication.primaryScreen()
                if not target_screen:
                    logger.error("No screens available")
                    return None
            
            # Get the monitor index
            screens = QGuiApplication.screens()
            monitor_idx = screens.index(target_screen) if target_screen in screens else 0
            
            # Load the saved preset for this monitor
            preset_key = f"MonitorPresets/Monitor_{monitor_idx}_Preset"
            position_preset = self.settings.value(preset_key, DEFAULT_POSITION_PRESET)
            logger.info(f"Using position preset '{position_preset}' for {overlay_type} overlay on monitor {monitor_idx} ({screen_name})")
            
            # Calculate the geometry using the saved preset
            geometry = self.calculate_position_geometry(target_screen, position_preset, overlay_type)
            
            if geometry:
                logger.debug(f"Calculated {overlay_type} overlay geometry: {geometry.width()}x{geometry.height()} @ ({geometry.x()},{geometry.y()})")
            else:
                logger.warning(f"Failed to calculate geometry for {overlay_type} overlay on {screen_name}")
                
            return geometry
            
        except Exception as e:
            logger.error(f"Error getting preset geometry for {overlay_type} overlay on {screen_name}: {e}", exc_info=True)
            return None
            
    def _create_initial_overlays(self):
        # Do not create any overlays by default on startup
        # Overlays will be created when explicitly requested by the user
        logger.debug("Skipping automatic overlay creation on startup")

    def update_switch_hotkey_config(self, enabled, sequence):
        self.switch_hotkey_enabled = enabled
        self.switch_hotkey_sequence = sequence
        self._register_or_unregister_switch_hotkey()
        self.settings.setValue("SwitchHotkeyEnabled", enabled)
        self.settings.setValue("SwitchHotkeySequence", sequence)
        logger.info(f"Updated hotkey config: Enabled={enabled}, Sequence={sequence}")

    def set_capture_fps(self, fps):
        """Update the FPS for all active monitor overlays."""
        logger.debug(f"Setting capture FPS to {fps} for all monitor overlays")
        for overlay in self.active_overlays.values():
            if hasattr(overlay, 'set_fps'):
                overlay.set_fps(fps)

    def apply_theme_globally(self, theme_name, from_global=False):
        """
        Apply the specified theme to the entire application.
        
        Args:
            theme_name (str): Name of the theme to apply ('light' or 'dark')
            from_global (bool): If True, indicates this is a recursive call to prevent loops
        """
        try:
            # Normalize theme name
            theme_name = str(theme_name).lower().strip()
            if theme_name not in ['light', 'dark']:
                logger.warning(f"Invalid theme name: {theme_name}. Defaulting to 'dark'.")
                theme_name = 'dark'
                
            # Skip if theme isn't changing
            if hasattr(self, 'current_theme') and self.current_theme == theme_name and not from_global:
                logger.debug(f"Theme already set to {theme_name}, skipping reapplication")
                return True
                
            logger.debug(f"Applying theme globally: {theme_name} (from_global={from_global})")
            
            # Get the style manager instance
            from style_manager import style_manager
            
            # Apply theme to the application
            style_manager.apply_theme(theme_name)
            
            # Save the theme preference if not from a global change (to prevent loops)
            if not from_global:
                self.settings.setValue('theme', theme_name)
                self.current_theme = theme_name
            
            # Apply theme to all existing overlays
            for overlay in self.active_overlays.values():
                if hasattr(overlay, 'apply_theme'):
                    try:
                        overlay.apply_theme(theme_name, from_global=True)
                    except Exception as e:
                        logger.error(f"Error applying theme to overlay {overlay}: {e}", exc_info=True)
                    
            # Apply theme to settings panel if it exists
            if hasattr(self, '_settings_panel') and self._settings_panel is not None:
                try:
                    if hasattr(self._settings_panel, 'apply_theme'):
                        self._settings_panel.apply_theme(theme_name, from_global=True)
                except Exception as e:
                    logger.error(f"Error applying theme to settings panel: {e}", exc_info=True)
            
            # Apply theme to sub-settings dialog if it exists
            if hasattr(self, '_sub_settings_dialog') and self._sub_settings_dialog is not None:
                try:
                    if hasattr(self._sub_settings_dialog, 'apply_theme'):
                        self._sub_settings_dialog.apply_theme(theme_name, from_global=True)
                except Exception as e:
                    logger.error(f"Error applying theme to sub-settings dialog: {e}", exc_info=True)
            
            logger.info(f"Successfully applied theme: {theme_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying theme '{theme_name}': {e}", exc_info=True)
            return False

def qt_message_handler(mode, context, message):
    """
    Custom message handler for Qt messages.
    Maps Qt message types to Python logging levels.
    """
    # Map Qt message types to Python logging levels
    level_map = {
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
        QtMsgType.QtDebugMsg: logging.DEBUG
    }
    
    # Default to debug level for unknown message types
    level = level_map.get(mode, logging.DEBUG)
    
    # Format the message and strip any extra whitespace
    msg = message.strip()
    
    # Special handling for common Qt warnings we want to filter or handle specially
    if mode == QtMsgType.QtWarningMsg:
        # Filter out common but not useful Qt warnings
        if any(warning in msg for warning in [
            "QFont::setPixelSize: Pixel size <= 0",
            "QWindowsWindow::setGeometry: Unable to set geometry"
        ]):
            # Only log these in debug mode
            if debug_enabled():
                logger.log(level, f"Qt Warning: {msg}")
            return
    
    # Only process messages at or above our log level threshold
    if level >= logging.getLogger().getEffectiveLevel():
        # Get the logger for the current module or use root
        module_logger = get_logger(context.category or 'qt')
        module_logger.log(level, msg)

@log_perf(level=logging.INFO, threshold_ms=100.0)
def main():
    """
    Main entry point for the application.
    
    Returns:
        int: Application exit code (0 for success, non-zero for errors)
    """
    # Configure logging first
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        # Set up Qt message handling early to catch all Qt messages
        qInstallMessageHandler(qt_message_handler)
        
        # Set high DPI settings before creating the application
        set_high_dpi_settings()
        
        # Create application instance with timing
        with DebugTimer("Application initialization"):
            app = PiPApplication(sys.argv)
            
        # Ensure the resource system is initialized
        try:
            logger.debug("Resource system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize resource system: {e}")
            # Continue anyway, as we have fallback mechanisms
            
        # Create initial overlays if configured
        if app.settings.value("auto_create_overlays", True, type=bool):
            with DebugTimer("Creating initial overlays"):
                app._create_initial_overlays()
        
        logger.info("Application started successfully")
        return app.exec()
        
    except Exception:
        logger.critical("Fatal error in application", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger = get_logger(__name__)
        logger.critical("Unhandled exception in main", exc_info=True)
        sys.exit(1)