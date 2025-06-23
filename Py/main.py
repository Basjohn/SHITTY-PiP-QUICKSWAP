# Standard library imports
import sys
import os
import ctypes
import ctypes.wintypes
import time
import traceback
import logging
import gc
from pathlib import Path
import keyboard

# Windows API imports
import win32gui
import win32con
import win32process
import win32api

# Third-party imports
from PySide6.QtCore import (
    QEasingCurve, QEvent, QMargins, QObject, QPoint, QProcess,
    QPropertyAnimation, QRect, QRegularExpression, QSettings, QSize,
    QStandardPaths, QTimer, Qt, QUrl, Signal, Slot, qInstallMessageHandler, 
    QtMsgType, QFile, QTextStream, QIODevice
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
from monitor_overlay import MonitorOverlay
from settings_panel import SettingsPanel
from subsettings_dialog import SubSettingsDialog

try:
    from about_dialog import AboutDialog
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

# Import required Windows API modules
try:
    import win32gui
    import win32con
    import win32process
    import win32api
    import win32com.client
    import win32event
    import winerror
    import win32security
    import win32ts
    import win32ui
    import win32gui_struct
    from ctypes import wintypes, windll, WINFUNCTYPE
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logger.warning("win32 modules not available. Some functionality may be limited.")

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
from constants import (
    DEFAULT_POSITION_PRESET,
    DEFAULT_WINDOW_OVERLAY_WIDTH,
    DEFAULT_WINDOW_OVERLAY_HEIGHT,
    DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR,
    DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR
)

WinEventProcType = WINFUNCTYPE(
    None, wintypes.HANDLE, wintypes.DWORD, wintypes.HWND,
    wintypes.LONG, wintypes.LONG, wintypes.DWORD, wintypes.DWORD
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "Resources"
SETTINGS_DIR = PROJECT_ROOT / "Settings"
LOGS_DIR = PROJECT_ROOT / "Logs"

# Configure root logger to only show WARNING and above by default
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.WARNING  # Default log level for all modules

# Create logs directory if it doesn't exist
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Set up root logger
log_file_path = LOGS_DIR / "app.log"
root_logger = logging.getLogger()
root_logger.setLevel(LOG_LEVEL)

# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
    handler.close()

# Set up file handler with WARNING level
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.WARNING)  # Only log WARNING and above to file
file_formatter = logging.Formatter(LOG_FORMAT)
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Set up console handler with warning level by default
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)  # Changed from LOG_LEVEL to WARNING
console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Configure specific loggers for our modules
modules_to_log = [
    'monitor_overlay',
    'snap_utils',
    'window_overlay',
    'key_passthrough',
    '__main__'
]

# Set log level - only enable DEBUG if --debug flag is explicitly set
log_level = logging.DEBUG if '--debug' in sys.argv else logging.WARNING

for module in modules_to_log:
    logger = logging.getLogger(module)
    logger.setLevel(log_level)
    logger.propagate = True  # Ensure logs propagate to root logger
    
    # Configure key_passthrough logger
    if module == 'key_passthrough':
        # Only show WARNING and above for key_passthrough
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.WARNING)  # Changed from log_level to WARNING
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplicate logs from root logger

# Suppress noisy loggers
noisy_loggers = [
    'comtypes',
    'comtypes._post_coinit',
    'comtypes.client',
    'PIL',
    'matplotlib',
    'urllib3',
    'asyncio'
]

for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Get logger for main module
logger = logging.getLogger(__name__)
logger.info("Application logging configured")
logger.debug(f"Logging configured. Log file: {log_file_path}")

# These imports have been moved to the top of the file

MUTEX_NAME = "Global\\ShittyPiP_SingleInstance_Mutex"

def is_already_running():
    try:
        mutex = win32event.CreateMutex(None, False, MUTEX_NAME)
        last_error = win32api.GetLastError()
        return last_error == winerror.ERROR_ALREADY_EXISTS
    except Exception as e:
        logger.error(f"Error in is_already_running: {e}")
        return False

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
    # Class-level icon cache to store loaded icons
    _icon_cache = {}
    
    def __init__(self):
        self.app_instance = QApplication.instance()
        self.last_window_list = []
        self.last_refresh_time = 0
        
        # Load fallback icon from resources
        self._blank_icon = QIcon()
        # Try both possible paths to ensure compatibility
        icon_paths = [":/Resources/Blank.ico", ":/Blank.ico"]
        
        for path in icon_paths:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self._blank_icon = QIcon(pixmap)
                logger.debug(f"Successfully loaded fallback icon from: {path}")
                break
        else:
            logger.warning("Failed to load Blank.ico from any resource path")
    
    def _is_valid_window(self, hwnd):
        if not hwnd or hwnd == 0:
            return False
        try:
            rect = (ctypes.c_int * 4)()
            success = windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if not success:
                return False
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            return width > 0 and height > 0
        except Exception as e:
            logger.debug(f"Error checking window validity: {e}")
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
        windows = []
        window_list = self.refresh_window_list(True)
        
        # Add Desktop/Program Manager as a special case
        try:
            # Find the actual Program Manager window
            progman_hwnd = win32gui.FindWindow("Progman", "Program Manager")
            if not progman_hwnd:
                logger.warning("Could not find Program Manager window")
                progman_hwnd = win32gui.GetDesktopWindow()
                
            desktop_title = "Desktop"
            desktop_class = "Progman"
            
            # Get the system folder icon for the desktop
            shell32 = ctypes.windll.shell32
            SHGFI_ICON = 0x100
            SHGFI_SMALLICON = 0x1
            SHGFI_LARGEICON = 0x0
            
            # Get the system folder icon
            hicon = shell32.ExtractIconW(0, "shell32.dll", 15)  # 15 is the folder icon in shell32.dll
            if hicon:
                try:
                    pixmap = QPixmap.fromImage(QImage.fromHICON(hicon))
                    desktop_icon = QIcon(pixmap)
                except Exception as e:
                    logger.debug(f"Failed to create icon from handle: {e}")
                finally:
                    ctypes.windll.user32.DestroyIcon(hicon)
            else:
                desktop_icon = self._blank_icon
            
            # Add desktop to the list
            windows.append((progman_hwnd, desktop_title, desktop_icon))
            logger.info(f"Added Desktop (Progman) with HWND: {progman_hwnd}")
        except Exception as e:
            logger.error(f"Error adding desktop to window list: {e}", exc_info=True)
        
        # Process windows in batches to prevent UI freezes
        for i, (hwnd, title) in enumerate(window_list):
            try:
                # Skip invalid windows
                if not self._is_valid_window(hwnd):
                    continue
                
                # Get window class for filtering
                window_class = win32gui.GetClassName(hwnd)
                
                # Skip the actual Program Manager window since we're handling it specially as Desktop
                if window_class == "Progman" and title == "Program Manager":
                    continue
                
                # Skip any other windows that might be the desktop
                if hwnd == progman_hwnd:
                    continue
                
                cache_key = f"{hwnd}:{window_class}"
                
                # Try to get icon from cache first
                if cache_key in self._icon_cache:
                    icon = self._icon_cache[cache_key]
                else:
                    # Get icon using multiple methods
                    try:
                        icon = self._get_window_icon(hwnd)
                        if icon.isNull() or (hasattr(icon, 'name') and icon.name() == "Blank.ico"):
                            alt_icon = self._get_alternative_window_icon(hwnd)
                            if not alt_icon.isNull():
                                icon = alt_icon
                        
                        # Only cache valid icons
                        if not icon.isNull():
                            self._icon_cache[cache_key] = icon
                        else:
                            icon = self._blank_icon
                    except Exception as e:
                        logger.debug(f"Error getting icon for window {hwnd}: {e}")
                        icon = self._blank_icon
                
                # Add a small delay every 5 windows to prevent system slowdown
                if i > 0 and i % 5 == 0:
                    QApplication.processEvents()
                    time.sleep(0.01)  # Small delay to prevent UI freezing
                
                window_info = (hwnd, title, icon)
                windows.append(window_info)
                
            except Exception as e:
                logger.error(f"Error getting window info for {hwnd}: {e}")
                windows.append((hwnd, title, self._blank_icon))
        
        return self.sort_windows(windows)
    
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
        super().__init__(argv)
        if getattr(sys, 'frozen', False):
            self.application_path = os.path.dirname(sys.executable)
        else:
            self.application_path = os.path.dirname(os.path.abspath(__file__))
        self.settings_path = str(SETTINGS_DIR / "settings.ini")
        self.settings = QSettings(self.settings_path, QSettings.Format.IniFormat)
        logger.info(f"Using settings file: {self.settings_path}")
        self.current_theme = "Dark"
        self.current_opacity_int = 100
        self.window_sort_order = "Most Recently Active"
        self.setApplicationName("Shitty PiP QuickSwap")
        self.setApplicationDisplayName("Shitty PiP QuickSwap")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("PiPOverlay")
        self.setQuitOnLastWindowClosed(False)
        self.setStyle("Fusion")
        
        # Initialize MRU (Most Recently Used) list for window ordering
        self.mru_hwnds = []
        self.MAX_MRU_ITEMS = 50  # Maximum number of MRU items to keep
        
        # Store the Progman window handle for desktop overlay
        self.progman_hwnd = win32gui.FindWindow("Progman", "Program Manager")
        if not self.progman_hwnd:
            self.progman_hwnd = win32gui.GetDesktopWindow()
        logger.debug(f"Stored Progman HWND: {self.progman_hwnd}")
        
        # Keep track of keep-alive handlers for active overlays
        self.keep_alive_handlers = {}  # hwnd -> MediaPlayerKeepAlive
        # Initialize font with debug logging
        try:
            font = QFont("Segoe UI")
            font.setPointSize(10)
            logger.debug(f"Setting application font: {font.family()}, Point Size: {font.pointSize()}, Pixel Size: {font.pixelSize() if font.pixelSize() > 0 else 'default'}")
            self.setFont(font)
        except Exception as e:
            logger.error(f"Error setting application font: {e}")
            # Fallback to default font if there's an issue
            self.setFont(QFont())
        self.window_enumerator = WindowEnumerator()
        self.window_enumerator.app_instance = self
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        # Use QRC resource system to load the icon
        self.setWindowIcon(QIcon(":/Resources/ShittyPIP.ico"))
        self.active_overlays = {}
        self._tray_icon = None
        self._settings_panel = None
        self.last_external_focused_hwnd = None
        self._sub_settings_dialog = None
        self._last_background_geometry = None
        self._current_switch_hotkey_id = None
        self.mru_hwnds = []
        self.MAX_MRU_ITEMS = 50
        self._win_event_hook = None
        self._win_event_proc_callback_ptr = None
        self.focusChanged.connect(self.handle_focus_change)
        self.applicationStateChanged.connect(self.handle_application_state_change)
        self._setup_tray()
        self.load_initial_settings()
        self._register_or_unregister_switch_hotkey()
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
            
            # Set window attributes - let SubSettingsDialog handle its own theming
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
        self.current_theme = self.settings.value("theme", "Dark", type=str)
        self.current_opacity_int = self.settings.value("opacity", 100, type=int)
        self.window_sort_order = self.settings.value("windowSortOrder", "Most Recently Active", type=str)
        self.switch_hotkey_enabled = self.settings.value("SwitchHotkeyEnabled", False, type=bool)
        self.switch_hotkey_sequence = self.settings.value("SwitchHotkeySequence", "F9", type=str)
        logger.debug(f"Initial settings loaded: Theme='{self.current_theme}', Opacity={self.current_opacity_int}, WindowSort='{self.window_sort_order}'")

    def _unregister_switch_hotkey(self):
        """Unregister the current switch hotkey if one is registered."""
        if self._current_switch_hotkey_id is not None:
            try:
                # First try the normal way
                try:
                    keyboard.remove_hotkey(self._current_switch_hotkey_id)
                    logger.debug(f"Removed hotkey ID: {self._current_switch_hotkey_id}")
                    return True
                except KeyError:
                    # If the hotkey ID is not found, try to clear all hotkeys
                    logger.warning(f"Hotkey ID {self._current_switch_hotkey_id} not found, clearing all hotkeys")
                    keyboard.unhook_all_hotkeys()
                    return True
                except Exception as e:
                    logger.warning(f"Error removing hotkey ID {self._current_switch_hotkey_id}: {e}")
                    return False
                finally:
                    self._current_switch_hotkey_id = None
            except Exception as e:
                logger.error(f"Unexpected error in _unregister_switch_hotkey: {e}")
                return False
        return True
    
    def _register_switch_hotkey(self, sequence=None):
        """Register the switch hotkey with the given sequence.
        
        Args:
            sequence (str, optional): The hotkey sequence to register. If None, uses the current setting.
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        if sequence is None:
            sequence = self.switch_hotkey_sequence
            
        if not sequence:
            logger.warning("No hotkey sequence provided to register")
            return False
            
        try:
            # First unregister any existing hotkey to avoid duplicates
            self._unregister_switch_hotkey()
            
            # Register the new hotkey
            self._current_switch_hotkey_id = keyboard.add_hotkey(
                sequence,
                self._handle_switch_hotkey_pressed
            )
            
            if self._current_switch_hotkey_id is None:
                logger.error(f"Failed to register switch hotkey: add_hotkey returned None")
                return False
                
            logger.info(f"Registered switch hotkey: '{sequence}' with ID: {self._current_switch_hotkey_id}")
            return True
            
        except ValueError as ve:
            logger.error(f"Invalid hotkey sequence '{sequence}': {ve}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to register switch hotkey '{sequence}': {e}", exc_info=True)
            self._current_switch_hotkey_id = None
            return False
    
    def _register_or_unregister_switch_hotkey(self):
        """Register or unregister the switch hotkey based on current settings."""
        # First unregister any existing hotkey
        self._unregister_switch_hotkey()
        
        # Register new hotkey if enabled and sequence is set
        if self.switch_hotkey_enabled and self.switch_hotkey_sequence:
            self._register_switch_hotkey()
    
    def update_switch_hotkey(self, enabled, sequence):
        """Update the switch hotkey with new settings.
        
        Args:
            enabled (bool): Whether the hotkey should be enabled
            sequence (str): The new hotkey sequence
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        logger.info(f"Updating switch hotkey - Enabled: {enabled}, Sequence: '{sequence}'")
        
        # Update settings
        self.switch_hotkey_enabled = enabled
        self.switch_hotkey_sequence = sequence
        
        # Save to settings
        self.settings.setValue("Hotkeys/switch_hotkey_enabled", enabled)
        self.settings.setValue("Hotkeys/switch_hotkey_sequence", sequence)
        
        # Update hotkey registration
        self._register_or_unregister_switch_hotkey()
        
        # Update the settings panel if it exists and has the update_ui_from_settings method
        if hasattr(self, '_settings_panel') and self._settings_panel:
            try:
                # Check if the method exists before calling it
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
        overlay_hwnds = [widget.winId() for widget in overlay_widgets if hasattr(widget, 'winId') and widget.isVisible()]
        if not overlay_hwnds:
            logger.info("No visible overlay windows to switch focus to.")
            return
        try:
            current_foreground_hwnd = win32gui.GetForegroundWindow()
            logger.debug(f"Current foreground HWND: {current_foreground_hwnd}")
            current_index = overlay_hwnds.index(current_foreground_hwnd) if current_foreground_hwnd in overlay_hwnds else -1
            next_index = (current_index + 1) % len(overlay_hwnds) if current_index != -1 else 0
            target_hwnd = overlay_hwnds[next_index]
            logger.info(f"Switching focus to overlay HWND: {target_hwnd}")
            for widget in overlay_widgets:
                if widget.winId() == target_hwnd and hasattr(widget, 'quick_switch_overlay'):
                    widget.quick_switch_overlay()
                    break
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
    
    def _update_mru(self, hwnd):
        """Update the Most Recently Used (MRU) list with the given window handle.
        
        Args:
            hwnd: The window handle to update in the MRU list.
        """
        try:
            # Remove the hwnd if it already exists in the list
            if hwnd in self.mru_hwnds:
                self.mru_hwnds.remove(hwnd)
                
            # Add to the front of the list
            self.mru_hwnds.insert(0, hwnd)
            
            # Trim the list if it's too long
            if len(self.mru_hwnds) > self.MAX_MRU_ITEMS:
                self.mru_hwnds = self.mru_hwnds[:self.MAX_MRU_ITEMS]
                
            logger.debug(f"Updated MRU list. New order: {self.mru_hwnds}")
            
        except Exception as e:
            logger.error(f"Error updating MRU list: {e}")

    def add_overlay(self, overlay_widget):
        """Add an overlay to the application and start tracking it."""
        if not overlay_widget:
            return
            
        try:
            # For monitor overlays, hwnd might be None initially
            hwnd = getattr(overlay_widget, 'hwnd', None)
            if not hwnd and hasattr(overlay_widget, 'winId'):
                try:
                    hwnd = int(overlay_widget.winId())
                    overlay_widget._hwnd = hwnd  # Cache the hwnd for future use
                except (AttributeError, TypeError):
                    pass
                    
            if not hwnd:
                # Use object id as a fallback for overlays without hwnd
                hwnd = id(overlay_widget)
                logger.warning(f"Overlay has no hwnd, using object id as fallback: {hwnd}")
                
            self.active_overlays[hwnd] = overlay_widget
            logger.info(f"Added overlay with ID: {hwnd}, Type: {type(overlay_widget).__name__}, Total overlays: {len(self.active_overlays)}")
            
            # Start keep-alive if this is a window overlay with a valid hwnd
            if hasattr(overlay_widget, 'hwnd') and overlay_widget.hwnd and overlay_widget.hwnd != 0:
                self._start_keep_alive(overlay_widget.hwnd)
        except Exception as e:
            logger.error(f"Error adding overlay: {e}", exc_info=True)
        
        # Update MRU list
        self._update_mru(hwnd)
        
        # Start keep-alive for media players
        self._start_keep_alive(hwnd)

    def remove_overlay(self, overlay_widget):
        """Remove an overlay from the application."""
        if not overlay_widget:
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
        except Exception as e:
            logger.error(f"Error removing overlay: {e}", exc_info=True)
        if hwnd in self.active_overlays:
            del self.active_overlays[hwnd]
            logger.info(f"Removed overlay for HWND: {hwnd}, Remaining overlays: {len(self.active_overlays)}")
            
            # Stop keep-alive for this window
            self._stop_keep_alive(hwnd)
            
            # Clean up any associated resources
            overlay_widget.cleanup()
            overlay_widget.deleteLater()
            
            # Force garbage collection to ensure resources are freed
            gc.collect()

    def update_opacity(self, opacity_int):
        opacity_int = max(0, min(100, opacity_int))
        self.current_opacity_int = opacity_int
        self.settings.setValue("opacity", self.current_opacity_int)
        logger.debug(f"Opacity set to {self.current_opacity_int}%")
        self.set_all_overlays_opacity(self.current_opacity_int / 100.0)

    def set_all_overlays_opacity(self, opacity_float):
        self.current_opacity_int = int(opacity_float * 100)
        logger.debug(f"Setting opacity for all {len(self.active_overlays)} overlays to {opacity_float:.2f}")
        for overlay in self.active_overlays.values():
            if hasattr(overlay, 'set_overlay_opacity'):
                overlay.set_overlay_opacity(opacity_float)
            elif hasattr(overlay, 'setWindowOpacity'):
                overlay.setWindowOpacity(opacity_float)

    def apply_theme_globally(self, theme_name, from_global=True):
        self.current_theme = theme_name
        logger.debug(f"Applying theme globally: '{theme_name}'")
        for overlay in self.active_overlays.values():
            if hasattr(overlay, 'apply_theme'):
                overlay.apply_theme(theme_name, from_global)
        if self._settings_panel and hasattr(self._settings_panel, 'apply_theme'):
            self._settings_panel.apply_theme(theme_name, from_global)
        if self._sub_settings_dialog:
            self._sub_settings_dialog.apply_theme(theme_name, from_global)
        self.settings.setValue("theme", self.current_theme)

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
        
        # Add Settings action
        self._settings_action = QAction("Settings", self)
        self._settings_action.triggered.connect(self._show_settings)
        self._tray_menu.addAction(self._settings_action)
        
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
        if not hasattr(self, 'active_overlays') or not self.active_overlays:
            return
            
        for overlay in self.active_overlays.values():
            if not overlay:
                continue
                
            try:
                if enabled:
                    # Enable click-through
                    overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                    overlay.setWindowFlags(overlay.windowFlags() | 
                                        Qt.WindowTransparentForInput | 
                                        Qt.WindowDoesNotAcceptFocus)
                else:
                    # Disable click-through
                    overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
                    overlay.setWindowFlags((overlay.windowFlags() & 
                                          ~Qt.WindowTransparentForInput) | 
                                          Qt.WindowStaysOnTopHint)
                
                # Reapply window flags to make changes take effect
                overlay.show()
                
            except Exception as e:
                logger.error(f"Error setting click-through mode on overlay: {e}")
    
    def _apply_tray_theme(self):
        """Apply the current theme to the system tray menu."""
        if not hasattr(self, '_tray_menu') or not self._tray_menu:
            return
            
        try:
            # Get the current theme from settings or default to 'dark'
            theme = 'dark'
            if hasattr(self, 'settings') and self.settings:
                theme = self.settings.value("UI/theme", "dark", type=str).lower()
                
            # Apply the theme stylesheet if available
            from constants import ThemeColors
            if hasattr(ThemeColors, 'get_theme_stylesheet'):
                stylesheet = ThemeColors.get_theme_stylesheet(theme)
                if stylesheet:
                    self._tray_menu.setStyleSheet(stylesheet)
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
        if key == "UI/theme" and hasattr(self, '_tray_menu') and self._tray_menu:
            self._apply_tray_theme()
            
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
                except:
                    pass

    def _cleanup_all_active_overlays(self):
        logger.info("Cleaning up all active overlays")
        # Stop all keep-alive handlers first
        for hwnd in list(self.keep_alive_handlers.keys()):
            self._stop_keep_alive(hwnd)
            
        # Clean up overlays
        for hwnd, overlay in list(self.active_overlays.items()):
            try:
                overlay.close()
                overlay.deleteLater()
            except Exception as e:
                logger.error(f"Error cleaning up overlay for HWND {hwnd}: {e}", exc_info=True)
        self.active_overlays.clear()
        logger.debug(f"Cleared active_overlays dictionary.")
        self._update_toggle_overlay_action_text()

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
            
            # Create the overlay widget
            new_overlay = BorderWidget(
                hwnd=hwnd, 
                opacity=self.current_opacity_int, 
                theme=self.current_theme,
                app_instance=self, 
                initial_geometry=final_initial_geometry
            )
            
            # Check if this is the desktop overlay
            try:
                if hwnd == self.progman_hwnd:
                    new_overlay.is_desktop_overlay = True
                    logger.info("Created desktop overlay with special handling")
            except Exception as e:
                logger.error(f"Error checking for desktop overlay: {e}", exc_info=True)
            
            if new_overlay.register_thumbnail():
                self.add_overlay(new_overlay)
                new_overlay.show()
                new_overlay.ensure_in_monitor_bounds(target_screen)
                logger.info(f"Created window overlay for '{window_title}' (HWND: {hwnd}) on screen {target_screen.name()}")
                gc.collect()
                
                # Bring the overlay to the front
                new_overlay.raise_()
                new_overlay.activateWindow()
            else:
                logger.error(f"Failed to register DWM background for '{window_title}' (HWND: {hwnd}).")
                new_overlay.deleteLater()
                
        except Exception as e:
            logger.error(f"Error creating window overlay for HWND {hwnd}: {e}", exc_info=True)

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
            
            # Log the screen we're trying to create an overlay for
            logger.info(f"Preparing to create overlay for screen: {screen.name() if hasattr(screen, 'name') else 'unnamed'}")
            logger.info(f"  Geometry: {screen.geometry().width()}x{screen.geometry().height()} @ ({screen.geometry().x()},{screen.geometry().y()})")
            
            # Get screen info before cleaning up overlays
            screen_info = {
                'name': screen.name(),
                'geometry': {
                    'x': screen.geometry().x(),
                    'y': screen.geometry().y(),
                    'width': screen.geometry().width(),
                    'height': screen.geometry().height()
                },
                'screen_object': screen  # Store the actual screen object directly
            }
            
            # Clean up existing overlays
            self._cleanup_all_active_overlays()
            
            # Log all available screens before creating the overlay
            logger.info("Available screens before creating overlay:")
            for i, s in enumerate(QGuiApplication.screens()):
                g = s.geometry()
                logger.info(f"  Screen {i}: {s.name() if hasattr(s, 'name') else 'unnamed'} - {g.width()}x{g.height()} @ ({g.x()},{g.y()})")
            
            # Capture all necessary screen info before the delay
            screen_data = {
                'name': screen_info.get('name') if isinstance(screen_info, dict) else None,
                'geometry': screen_info.get('geometry') if isinstance(screen_info, dict) else None,
                'monitor_idx': screen_info.get('monitor_idx') if isinstance(screen_info, dict) else None
            }
            QTimer.singleShot(50, lambda data=screen_data: self.actually_create_monitor_overlay(data))
            
        except Exception as e:
            logger.error(f"Error in prepare_to_create_monitor_overlay: {e}", exc_info=True)
            QMessageBox.critical(None, "Error", f"Failed to prepare monitor overlay: {str(e)}")
    
    def actually_create_monitor_overlay(self, screen_info):
        new_overlay = None
        try:
            if not screen_info:
                logger.error("No screen info provided to actually_create_monitor_overlay")
                return
                
            # Check if we have the screen object directly
            if 'screen_object' in screen_info and isinstance(screen_info['screen_object'], QScreen):
                target_screen = screen_info['screen_object']
                logger.info(f"Using directly passed screen object: {target_screen.name() if hasattr(target_screen, 'name') else 'unnamed'}")
                
                # Log physical monitor info if available
                try:
                    from snap_utils import get_physical_monitor_for_screen
                    monitor_info = get_physical_monitor_for_screen(target_screen)
                    if monitor_info and 'physical_width' in monitor_info and 'physical_height' in monitor_info:
                        logger.info(f"Physical monitor info: {monitor_info['physical_width']}x{monitor_info['physical_height']} @ ({monitor_info.get('x', 0)},{monitor_info.get('y', 0)}) DPI: {monitor_info.get('dpi_x', 0)}x{monitor_info.get('dpi_y', 0)}")
                except Exception as e:
                    logger.warning(f"Could not get physical monitor info: {e}")
                    
            else:
                # Fall back to the old method of finding the screen
                logger.info("Screen object not found in screen_info, falling back to screen lookup")
                screens = QGuiApplication.screens()
                if not screens:
                    logger.error("No screens found!")
                    return
                    
                target_screen = None
                screen_name = screen_info.get('name') if isinstance(screen_info, dict) else None
                
                if screen_name:
                    logger.info(f"Looking up screen from info: {screen_name}")
                    
                    # Try to find the screen by name and geometry
                    for screen in screens:
                        if screen.name() == screen_name:
                            # If we have geometry info, verify it matches
                            if 'geometry' in screen_info:
                                geo = screen.geometry()
                                info_geo = screen_info['geometry']
                                if (geo.x() == info_geo['x'] and geo.y() == info_geo['y'] and
                                    geo.width() == info_geo['width'] and geo.height() == info_geo['height']):
                                    target_screen = screen
                                    logger.debug(f"Found matching screen by name and geometry: {screen_name}")
                                    
                                    # Log physical monitor info for the found screen
                                    try:
                                        from snap_utils import get_physical_monitor_info
                                        monitor_info = get_physical_monitor_info(screen)
                                        if monitor_info:
                                            logger.info(f"Physical monitor info: {monitor_info['width']}x{monitor_info['height']} @ ({monitor_info.get('x', 0)},{monitor_info.get('y', 0)}) DPI: {monitor_info.get('dpi_x', 0)}x{monitor_info.get('dpi_y', 0)}")
                                    except Exception as e:
                                        logger.warning(f"Could not get physical monitor info: {e}")
                                        
                                    break
                            else:
                                target_screen = screen
                                logger.debug(f"Found screen by name only: {screen_name}")
                                break
                
                # If we still don't have a screen, use the primary screen as fallback
                if not target_screen:
                    logger.warning(f"Could not find screen '{screen_name}', using primary screen")
                    target_screen = QGuiApplication.primaryScreen()
                    if not target_screen and screens:
                        target_screen = screens[0]
            
            if not target_screen:
                logger.error("No valid screen found for overlay")
                return
                
            logger.info(f"Using screen: {target_screen.name()} at {target_screen.geometry()}")
            
            # Log DPI and scaling information
            dpi = target_screen.logicalDotsPerInch()
            device_pixel_ratio = target_screen.devicePixelRatio()
            logger.info(f"Screen DPI: {dpi:.1f}, Device Pixel Ratio: {device_pixel_ratio:.2f}")
            
            # Get the monitor index for the target screen
            screens = QGuiApplication.screens()
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
                
            # First try to get physical monitor info
            try:
                from snap_utils import get_physical_monitor_for_screen
                monitor_info = get_physical_monitor_for_screen(screen)
                
                if monitor_info and 'width' in monitor_info and 'height' in monitor_info:
                    # Use physical dimensions for calculation
                    phys_width = monitor_info['width']
                    phys_height = monitor_info['height']
                    phys_x = monitor_info.get('x', 0)
                    phys_y = monitor_info.get('y', 0)
                    
                    # Calculate size based on physical dimensions
                    if overlay_type == "window":
                        w, h = DEFAULT_WINDOW_OVERLAY_WIDTH, DEFAULT_WINDOW_OVERLAY_HEIGHT
                    else:
                        w = int(phys_width * DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR)
                        h = int(phys_height * DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR)
                    
                    # Calculate position in physical coordinates - default to top-left (0,0)
                    x, y = 0, 0
                    if position_preset == "TopRight" or position_preset == "Right":
                        x = phys_width - w
                    elif position_preset == "BottomLeft" or position_preset == "Bottom":
                        y = phys_height - h
                    elif position_preset == "BottomRight":
                        x = phys_width - w
                        y = phys_height - h
                    elif position_preset == "Centered":
                        x = (phys_width - w) // 2
                        y = (phys_height - h) // 2
                    # Default is TopLeft (0,0)
                    
                    # Convert back to logical coordinates for Qt
                    screen_geo = screen.availableGeometry()
                    scale_x = screen_geo.width() / max(phys_width, 1)  # Avoid division by zero
                    scale_y = screen_geo.height() / max(phys_height, 1)  # Avoid division by zero
                    
                    logical_x = int(x * scale_x) + screen_geo.x()
                    logical_y = int(y * scale_y) + screen_geo.y()
                    logical_w = int(w * scale_x)
                    logical_h = int(h * scale_y)
                    
                    logger.debug(f"Using physical monitor info for positioning: {phys_width}x{phys_height} @ ({phys_x},{phys_y})")
                    return QRect(logical_x, logical_y, logical_w, logical_h)
                
            except Exception as e:
                logger.warning(f"Could not get physical monitor info, falling back to logical coordinates: {e}")
            
            # Fallback to logical coordinates if physical info is not available
            geo = screen.availableGeometry()
            if overlay_type == "window":
                w, h = DEFAULT_WINDOW_OVERLAY_WIDTH, DEFAULT_WINDOW_OVERLAY_HEIGHT
            else:
                w = int(geo.width() * DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR)
                h = int(geo.height() * DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR)
                
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
        if not hwnd:
            return
        try:
            pid = ctypes.wintypes.DWORD()
            windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == os.getpid():
                logger.debug(f"Foreground window is our own process (HWND={hwnd}).")
                return
            if hwnd not in self.mru_hwnds:
                self.mru_hwnds.insert(0, hwnd)
                if len(self.mru_hwnds) > self.MAX_MRU_ITEMS:
                    self.mru_hwnds = self.mru_hwnds[:self.MAX_MRU_ITEMS]
                logger.debug(f"Added HWND {hwnd} to MRU list.")
            else:
                self.mru_hwnds.remove(hwnd)
                self.mru_hwnds.insert(0, hwnd)
                logger.debug(f"Moved HWND {hwnd} to top of MRU list.")
            self.last_external_focused_hwnd = hwnd
        except Exception as e:
            logger.error(f"Error handling foreground window change for HWND {hwnd}: {e}")

    def handle_focus_change(self, old, new):
        logger.debug(f"Qt focus changed: Old={old}, New={new}")

    def handle_application_state_change(self, state):
        logger.debug(f"Application state changed: {state}")

    def cleanup_and_quit(self):
        logger.info("Cleaning up before quitting...")
        try:
            # Clean up all keep-alive handlers
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

def main():
    # Install the custom message handler first
    qInstallMessageHandler(qt_message_handler)
    
    # Set high DPI settings before creating the application
    set_high_dpi_settings()
    
    # Create the application instance
    app = PiPApplication(sys.argv)
    
    # Ensure the resource system is initialized
    try:
        # This will raise an exception if resources can't be loaded
        from . import resources_rc
        logger.debug("Resource system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize resource system: {e}")
        # Continue anyway, as we have fallback mechanisms
    
    try:
        logger.info("Application started successfully.")
        app._show_settings()
        exit_code = app.exec()
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)
        sys.exit(1)

def qt_message_handler(mode, context, message):
    """
    Custom message handler for Qt messages.
    """
    if mode == QtMsgType.QtWarningMsg and "QFont::setPixelSize: Pixel size <= 0" in message:
        # Log the full stack trace for font warnings
        import traceback
        stack = '\n'.join(traceback.format_stack())
        logger.warning(f"Qt Font Warning: {message}\nStack Trace:\n{stack}")
    elif mode == QtMsgType.QtWarningMsg:
        logger.warning(f"Qt Warning: {message}")
    elif mode == QtMsgType.QtCriticalMsg:
        logger.error(f"Qt Critical: {message}")
    elif mode == QtMsgType.QtFatalMsg:
        logger.critical(f"Qt Fatal: {message}")
    elif mode == QtMsgType.QtInfoMsg:
        logger.info(f"Qt Info: {message}")
    else:
        logger.debug(f"Qt Message ({mode}): {message}")

def main():
    # Install the custom message handler
    qInstallMessageHandler(qt_message_handler)
    
    # Set high DPI settings before creating the application
    set_high_dpi_settings()
    
    # Create and run the application
    app = PiPApplication(sys.argv)
    app._create_initial_overlays()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()