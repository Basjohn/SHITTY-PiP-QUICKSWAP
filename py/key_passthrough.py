import win32gui
import win32api
import win32con
import win32process
import ctypes
from ctypes import wintypes
import time
import threading
import json
import os
from typing import Optional, List, Tuple, Dict, TypedDict, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import psutil

# Debugging and logging utilities
from debug_utils import get_logger

# Import the UniversalMediaController
from universal_media_controller import UniversalMediaController

# Get logger instance
logger = get_logger(__name__)

# Import constants from constants.py
from constants import (
    InputType, AppType, KEY_NAMES
)

# Use win32con for virtual key codes
VK_SPACE = win32con.VK_SPACE
VK_RETURN = win32con.VK_RETURN
VK_LEFT = win32con.VK_LEFT
VK_RIGHT = win32con.VK_RIGHT
VK_UP = win32con.VK_UP
VK_DOWN = win32con.VK_DOWN

# Windows message constants
WM_APPCOMMAND = 0x0319
WM_MOUSEWHEEL = 0x020A

# AppCommand constants
APPCOMMAND_MEDIA_PLAY_PAUSE = 0xE0000

# Mouse wheel constant
WHEEL_DELTA = 120

# Type Aliases
HWND = int
VKCode = int

# Type Definitions
@dataclass
class AppDetectionRule:
    """Rule for detecting application type based on executable and window class."""
    exe_matches: List[str] = field(default_factory=list)
    class_matches: List[str] = field(default_factory=list)
    title_matches: List[str] = field(default_factory=list)


class AppInfo(TypedDict):
    """Information about the target application window."""
    title: str
    class_name: str
    exe: str
    type: AppType

# SendInput structures
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG))
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG))
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.wintypes.DWORD),
        ("wParamL", ctypes.wintypes.WORD),
        ("wParamH", ctypes.wintypes.WORD)
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [
            ("ki", KEYBDINPUT),
            ("mi", MOUSEINPUT),
            ("hi", HARDWAREINPUT)
        ]
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("_input", _INPUT)
    ]

# Import InputType for input-related constants

# Create aliases for backward compatibility
INPUT_KEYBOARD = InputType.INPUT_KEYBOARD
KEYEVENTF_KEYUP = InputType.KEYEVENTF_KEYUP
KEYEVENTF_UNICODE = InputType.KEYEVENTF_UNICODE
KEYEVENTF_SCANCODE = 0x0008

class KeyPassthrough:
    """Handles key passthrough functionality to send keystrokes to target windows.
    
    This class provides methods to send keystrokes to a target window, with various
    fallback mechanisms to ensure compatibility with different applications.
    """
    
    # Default timeouts in seconds
    MESSAGE_DELAY = 0.01  # Small delay between key down and key up
    MIN_KEY_INTERVAL = 0.05  # 50ms between keys for rate limiting
    
    # Default application detection rules
    DEFAULT_APP_RULES = {
        AppType.FIREFOX: {
            'exe_matches': ['firefox', 'mozilla'],
            'class_matches': ['mozilla']
        },
        AppType.CHROMIUM: {
            'exe_matches': ['chrome', 'msedge', 'opera', 'brave'],
            'class_matches': ['chrome', 'chromium']
        },
        AppType.MEDIA: {
            'exe_matches': ['spotify', 'vlc', 'mpv', 'wmplayer', 'musicbee', 'itunes', 'winamp'],
            'class_matches': []
        },
        AppType.GAME: {
            'exe_matches': ['game', 'launcher', 'steam'],
            'class_matches': ['unity']
        }
    }
    
    def __init__(self, state_change_callback=None, config_path: Optional[str] = None):
        """Initialize the KeyPassthrough instance with default settings.
        
        Args:
            state_change_callback: Optional callback function that will be called when the
                                 passthrough state changes. The callback should accept two
                                 boolean parameters: (enabled, aggressive_mode)
            config_path: Optional path to a JSON configuration file for app detection rules
        """
        self._enabled = False
        self._aggressive_mode = False
        self._target_hwnd: Optional[HWND] = None
        self._app_strategies: Dict[str, List[Tuple[str, Callable[[HWND, VKCode, AppInfo], bool]]]] = {}
        self._state_change_callback = state_change_callback
        self._thread_lock = threading.RLock()
        self._last_key_time = 0.0
        self._app_detection_rules = self._load_app_detection_rules(config_path)
        self._init_default_strategies()
    
    def _load_app_detection_rules(self, config_path: Optional[str] = None) -> Dict[AppType, AppDetectionRule]:
        """Load application detection rules from config file or use defaults.
        
        Args:
            config_path: Optional path to a JSON configuration file
            
        Returns:
            Dictionary mapping AppType to AppDetectionRule
        """
        rules = {}
        
        # First load default rules
        for app_type, rule_data in self.DEFAULT_APP_RULES.items():
            rules[app_type] = AppDetectionRule(
                exe_matches=rule_data.get('exe_matches', []),
                class_matches=rule_data.get('class_matches', []),
                title_matches=rule_data.get('title_matches', [])
            )
        
        # Override with rules from config file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                for app_type_str, rule_data in config.get('app_detection', {}).items():
                    try:
                        app_type = AppType[app_type_str.upper()]
                        rules[app_type] = AppDetectionRule(
                            exe_matches=rule_data.get('exe_matches', []),
                            class_matches=rule_data.get('class_matches', []),
                            title_matches=rule_data.get('title_matches', [])
                        )
                    except (KeyError, AttributeError):
                        logger.warning(f"Invalid app type in config: {app_type_str}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}", exc_info=True)
        
        return rules
    
    @contextmanager
    def _thread_attachment(self, hwnd: HWND):
        """Context manager for attaching to a window's thread.
        
        Args:
            hwnd: Window handle to attach to
            
        Yields:
            bool: True if attached to the thread, False otherwise
        """
        attached = False
        thread_id = None
        current_thread = None
        
        try:
            # Get the thread ID of the target window
            thread_id = win32process.GetWindowThreadProcessId(hwnd)[0]
            current_thread = win32api.GetCurrentThreadId()
            
            # Try to attach to the target window's thread if needed
            if thread_id != current_thread:
                try:
                    win32process.AttachThreadInput(current_thread, thread_id, True)
                    attached = True
                    logger.debug("Attached to target window's thread")
                except Exception as e:
                    logger.warning("Failed to attach to target window's thread: %s", str(e))
            
            yield attached
            
        except Exception as e:
            logger.error("Error in thread attachment: %s", str(e), exc_info=True)
            raise
            
        finally:
            # Always detach from the thread if we attached to it
            if attached and thread_id is not None and current_thread is not None:
                try:
                    win32process.AttachThreadInput(current_thread, thread_id, False)
                    logger.debug("Detached from target window's thread")
                except Exception as e:
                    logger.warning("Failed to detach from target window's thread: %s", str(e))
    
    def _rate_limit(self):
        """Ensure a minimum time between key events to prevent flooding."""
        with self._thread_lock:
            now = time.time()
            elapsed = now - self._last_key_time
            if elapsed < self.MIN_KEY_INTERVAL:
                sleep_time = self.MIN_KEY_INTERVAL - elapsed
                time.sleep(sleep_time)
            self._last_key_time = time.time()
    
    def _init_default_strategies(self) -> None:
        """Initialize the default key sending strategies."""
        # Define the default strategy that doesn't steal focus
        default_strategies = [
            ("Direct PostMessage", self._method_direct_postmessage),
        ]
        
        # All application types use the same strategy
        self._app_strategies = {app_type: default_strategies 
                              for app_type in [AppType.OTHER, AppType.FIREFOX, 
                                             AppType.CHROMIUM, AppType.MEDIA, AppType.GAME]}

    def set_target_window(self, hwnd: int):
        """Set the target window for key passthrough."""
        self._target_hwnd = hwnd
        logger.debug(f"Key passthrough target window set to: {hwnd}")

    def set_enabled(self, enabled: bool):
        """Enable or disable key passthrough."""
        if self._enabled != enabled:
            self._enabled = enabled
            logger.info(f"Key passthrough {'enabled' if enabled else 'disabled'}")
            if self._state_change_callback:
                self._state_change_callback(self._enabled, self._aggressive_mode)
        return enabled

    def is_enabled(self) -> bool:
        """Check if key passthrough is enabled."""
        return self._enabled

    def set_aggressive_mode(self, enabled: bool):
        """Enable or disable aggressive key passthrough mode."""
        if self._aggressive_mode != enabled:
            self._aggressive_mode = enabled
            logger.info(f"Aggressive key passthrough {'enabled' if enabled else 'disabled'}")
            if self._state_change_callback:
                self._state_change_callback(self._enabled, self._aggressive_mode)
        return enabled

    def is_aggressive_mode(self) -> bool:
        """Check if aggressive mode is enabled."""
        return self._aggressive_mode

    def _get_window_info_str(self, hwnd: int) -> str:
        """Get detailed window information string for logging."""
        if not hwnd or not win32gui.IsWindow(hwnd):
            return "INVALID_WINDOW"
            
        try:
            # Get window text and class first - these are most reliable
            try:
                window_text = win32gui.GetWindowText(hwnd)
                window_class = win32gui.GetClassName(hwnd)
            except Exception as e:
                window_text = f"<error: {str(e)[:50]}>"
                window_class = "<error>"
            
            # Get process ID
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                pid_str = str(pid)
            except:
                pid_str = "<error>"
            
            # Get process info if psutil is available
            process_info = ""
            try:
                if 'psutil' in globals():
                    try:
                        process = psutil.Process(pid)
                        process_name = process.name()
                        process_cmd = ' '.join(process.cmdline()[:3]) + ('...' if len(process.cmdline()) > 3 else '')
                        process_info = f", Process: {process_name}, Cmd: {process_cmd}"
                    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                        pass
            except Exception:
                # Silently ignore psutil errors
                pass
            
            # Get window rect
            try:
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                rect_str = f"({left}, {top}, {right}, {bottom}) [{right-left}x{bottom-top}]"
            except:
                rect_str = "N/A"
                
            return (
                f"HWND: 0x{hwnd:X}, "
                f"Title: '{window_text}', "
                f"Class: {window_class}, "
                f"PID: {pid_str}"
                f"{process_info}, "
                f"Rect: {rect_str}"
            )
        except Exception as e:
            return f"Error getting window info: {str(e)}"
    
    def _get_vk_name(self, vk_code: int) -> str:
        """Convert VK code to human-readable name."""
        vk_map = {
            win32con.VK_LEFT: "LEFT",
            win32con.VK_RIGHT: "RIGHT",
            win32con.VK_UP: "UP",
            win32con.VK_DOWN: "DOWN",
            win32con.VK_SPACE: "SPACE",
            win32con.VK_RETURN: "ENTER",
            win32con.VK_ESCAPE: "ESC",
            win32con.VK_TAB: "TAB",
            win32con.VK_BACK: "BACKSPACE",
            win32con.VK_DELETE: "DELETE",
            win32con.VK_HOME: "HOME",
            win32con.VK_END: "END",
            win32con.VK_PRIOR: "PAGE UP",
            win32con.VK_NEXT: "PAGE DOWN",
            win32con.VK_INSERT: "INSERT",
            win32con.VK_PAUSE: "PAUSE",
            win32con.VK_CAPITAL: "CAPS LOCK",
            win32con.VK_NUMLOCK: "NUM LOCK",
            win32con.VK_SCROLL: "SCROLL LOCK",
            0xB0: "MEDIA NEXT",
            0xB1: "MEDIA PREV",
            0xB2: "MEDIA STOP",
            0xB3: "MEDIA PLAY/PAUSE",
            0xAD: "VOLUME MUTE",
            0xAE: "VOLUME DOWN",
            0xAF: "VOLUME UP"
        }
        return vk_map.get(vk_code, f"0x{vk_code:02X}")
    
    def _log_key_action(self, action: str, vk_code: int, success: bool = True, extra: str = "") -> None:
        """Log key action with consistent formatting."""
        vk_name = self._get_vk_name(vk_code)
        target_info = self._get_window_info_str(self._target_hwnd) if self._target_hwnd else "NO_TARGET"
        status = "SUCCESS" if success else "FAILED"
        logger.info(
            f"[KEY_{action}] {status}: {vk_name} (0x{vk_code:02X}) | "
            f"Target: {target_info} | "
            f"Mode: {'AGGRESSIVE' if self._aggressive_mode else 'STANDARD'}{' | ' + extra if extra else ''}"
        )
    
    def send_key(self, vk_code: VKCode) -> bool:
        """Send a key to the target window if enabled.
        
        Args:
            vk_code: The virtual key code to send.
            
        Returns:
            bool: True if the key was sent successfully, False otherwise.
        """
        if not self._enabled:
            self._log_key_action("SEND", vk_code, False, "Key passthrough disabled")
            return False
            
        if not self._target_hwnd or not win32gui.IsWindow(self._target_hwnd):
            self._log_key_action("SEND", vk_code, False, "No valid target window")
            return False
            
        try:
            # Handle special keys with dedicated methods
            if vk_code == win32con.VK_SPACE:
                result = self._send_media_play_pause()
                self._log_key_action("MEDIA_PLAY_PAUSE", vk_code, result)
                return result
            elif vk_code in (win32con.VK_UP, win32con.VK_DOWN):
                result = self._send_scroll_command(vk_code)
                self._log_key_action("SCROLL", vk_code, result, f"Direction: {'UP' if vk_code == win32con.VK_UP else 'DOWN'}")
                return result
                
            # For all other keys, use the universal handler
            self._log_key_action("SEND", vk_code, True, "Initiating key send")
            result = self._send_key_universal(self._target_hwnd, vk_code)
            self._log_key_action("SEND_COMPLETE", vk_code, result, f"Result: {'Success' if result else 'Failed'}")
            return result
                
        except Exception as e:
            error_msg = f"Error in send_key: {str(e)}"
            self._log_key_action("SEND_ERROR", vk_code, False, error_msg)
            logger.error(error_msg, exc_info=True)
            return False

    def send_media_play_pause(self) -> bool:
        """Send space key for play/pause.
        
        Returns:
            bool: True if the key was sent successfully, False otherwise.
        """
        return self.send_key(win32con.VK_SPACE)
        
    def send_media_next_track(self) -> bool:
        """Send right arrow for next track.
        
        Returns:
            bool: True if the key was sent successfully, False otherwise.
        """
        return self.send_key(win32con.VK_RIGHT)
        
    def send_media_previous_track(self) -> bool:
        """Send left arrow for previous track.
        
        Returns:
            bool: True if the key was sent successfully, False otherwise.
        """
        return self.send_key(win32con.VK_LEFT)
        
    def send_space(self) -> bool:
        """Send space key.
        
        Returns:
            bool: True if the key was sent successfully, False otherwise.
        """
        return self.send_key(win32con.VK_SPACE)
        
    def send_enter(self) -> bool:
        """Send enter key.
        
        Returns:
            bool: True if the key was sent successfully, False otherwise.
        """
        return self.send_key(win32con.VK_RETURN)

    def _send_media_play_pause(self) -> bool:
        """Send media play/pause command to the target window.
        
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        hwnd = self._target_hwnd
        if not hwnd or not win32gui.IsWindow(hwnd):
            logger.error("Invalid target window for media command")
            return False
        
        with self._thread_lock:
            try:
                logger.debug("Using UniversalMediaController for media play/pause")
                # Try to identify the application
                app_info = self._get_app_info(hwnd)
                app_type = app_info.get('type', AppType.OTHER)
                
                # Use the media controller for media and browser apps
                if app_type in [AppType.MEDIA, AppType.FIREFOX, AppType.CHROMIUM]:
                    controller = UniversalMediaController()
                    success, _ = controller.play_pause()
                    if success:
                        logger.debug("Media play/pause command sent via UniversalMediaController")
                        return True
                    logger.debug("UniversalMediaController failed, falling back to direct method")
                
                # Fallback to direct method for other apps
                with self._thread_attachment(hwnd) as attached:
                    logger.debug("Trying WM_APPCOMMAND for media play/pause")
                    result = win32api.SendMessage(
                        hwnd,
                        WM_APPCOMMAND,
                        hwnd,
                        APPCOMMAND_MEDIA_PLAY_PAUSE << 16
                    )
                    
                    if not result:
                        logger.debug("WM_APPCOMMAND failed, trying direct SPACE key")
                        if not self._send_key_sequence(hwnd, win32con.VK_SPACE):
                            return self._send_input(win32con.VK_SPACE)
                        return True
                    
                    logger.debug("Media play/pause command sent, result: %s", result)
                    return result == 1
                
            except Exception as e:
                logger.error("Failed to send media command: %s", str(e), exc_info=True)
                return False
            
    def _send_scroll_command(self, vk_code: VKCode) -> bool:
        """Send scroll command to the target window.
        
        Args:
            vk_code: Either VK_UP or VK_DOWN
            
        Returns:
            bool: True if the scroll command was sent successfully, False otherwise.
        """
        if not self._target_hwnd or not win32gui.IsWindow(self._target_hwnd):
            logger.error("Invalid target window for scroll command")
            return False
            
        # Get window class to determine the best scrolling method
        window_class = win32gui.GetClassName(self._target_hwnd).lower()
        is_firefox = 'mozilla' in window_class or 'firefox' in window_class
        
        # Determine scroll direction and parameters
        if vk_code == win32con.VK_UP:
            scroll_cmd = win32con.SB_LINEUP if is_firefox else WHEEL_DELTA
            direction = "up"
            vk_code_to_send = win32con.VK_UP
        elif vk_code == win32con.VK_DOWN:
            scroll_cmd = win32con.SB_LINEDOWN if is_firefox else -WHEEL_DELTA
            direction = "down"
            vk_code_to_send = win32con.VK_DOWN
        else:
            logger.error("Invalid key code for scroll command: %d", vk_code)
            return False
            
        try:
            logger.debug("Sending scroll %s to window 0x%X (class: %s)", direction, self._target_hwnd, window_class)
            
            if is_firefox:
                # Method 1: Try WM_VSCROLL first (works better for Firefox)
                try:
                    result = win32api.SendMessage(
                        self._target_hwnd,
                        win32con.WM_VSCROLL,
                        scroll_cmd,
                        0
                    )
                    if result == 0:  # Success
                        logger.debug("Firefox scroll %s command sent via WM_VSCROLL", direction)
                        return True
                except Exception as e:
                    logger.debug("WM_VSCROLL failed, trying alternative methods: %s", str(e))
                
                # Method 2: Try sending actual arrow key presses
                try:
                    # Send key down
                    win32api.PostMessage(
                        self._target_hwnd,
                        win32con.WM_KEYDOWN,
                        vk_code_to_send,
                        0x00000001  # Repeat count = 1
                    )
                    # Send key up
                    win32api.PostMessage(
                        self._target_hwnd,
                        win32con.WM_KEYUP,
                        vk_code_to_send,
                        0xC0000001  # Key up flag
                    )
                    logger.debug("Firefox scroll %s command sent via key events", direction)
                    return True
                except Exception as e:
                    logger.debug("Key event method failed: %s", str(e))
                
                # Method 3: Fall back to mouse wheel as last resort
                logger.debug("Falling back to mouse wheel for Firefox scroll")
                
            # Default method for non-Firefox windows or if Firefox methods failed
            rect = win32gui.GetWindowRect(self._target_hwnd)
            x = (rect[0] + rect[2]) // 2
            y = (rect[1] + rect[3]) // 2
            
            # Create lParam with coordinates relative to window
            lparam = y << 16 | (x & 0xFFFF)
            
            # Send WM_MOUSEWHEEL message
            result = win32api.SendMessage(
                self._target_hwnd,
                WM_MOUSEWHEEL,
                (scroll_cmd if not is_firefox else (WHEEL_DELTA if vk_code == win32con.VK_UP else -WHEEL_DELTA)) << 16,
                lparam
            )
            logger.debug("Scroll %s command sent via WM_MOUSEWHEEL, result: %s", direction, result)
            return True
            
        except Exception as e:
            logger.error("Failed to send scroll command: %s", str(e), exc_info=True)
            return False
    
    def _send_key_universal(self, hwnd: HWND, vk_code: VKCode) -> bool:
        """Universal key sending that works with modern applications.
        
        Args:
            hwnd: The window handle to send the key to.
            vk_code: The virtual key code to send.
            
        Returns:
            bool: True if the key was sent successfully, False otherwise.
        """
        if not win32gui.IsWindow(hwnd):
            logger.error("Invalid window handle: 0x%X", hwnd)
            return False

        app_info = self._get_app_info(hwnd)
        key_name = KEY_NAMES.get(vk_code, f'VK_{vk_code}')
        
        # Get app type name safely
        app_type = app_info.get('type', 0)
        app_type_name = app_type.name if hasattr(app_type, 'name') else str(app_type)
        
        logger.debug("=== Sending %s to %s ===", key_name, app_type_name)
        logger.debug("Window: '%s' (Class: %s)", app_info.get('title', 'Unknown'), 
                   app_info.get('class_name', 'Unknown'))
        logger.debug("Process: %s", app_info.get('exe', 'Unknown'))

        # Get the appropriate strategy based on application type
        strategies = self._app_strategies.get(app_info['type'], [])

        # Try each strategy until one succeeds
        for method_name, method_func in strategies:
            try:
                logger.debug("Trying method: %s", method_name)
                if method_func(hwnd, vk_code, app_info):
                    logger.debug("✓ SUCCESS: %s worked for %s", method_name, key_name)
                    return True
                logger.debug("✗ FAILED: %s", method_name)
            except Exception as e:
                logger.debug("✗ ERROR in %s: %s", method_name, str(e), exc_info=True)

        logger.warning("All methods failed for %s", key_name)
        return False

    def _get_app_info(self, hwnd: HWND) -> AppInfo:
        """Get comprehensive application information for the given window.
        
        Args:
            hwnd: The window handle to get information for.
            
        Returns:
            AppInfo: A dictionary containing application information.
        """
        default_info = AppInfo(
            title="",
            class_name="",
            exe="",
            type=AppType.OTHER
        )
        
        with self._thread_lock:
            try:
                # Get window title and class
                title = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                exe_name = "unknown"
                
                # Get process information
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    if pid:
                        handle = win32api.OpenProcess(
                            win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, 
                            False, 
                            pid
                        )
                        if handle:
                            try:
                                exe_path = win32process.GetModuleFileNameEx(handle, 0)
                                exe_name = exe_path.split('\\')[-1].lower()
                            finally:
                                win32api.CloseHandle(handle)
                except Exception as e:
                    logger.debug("Failed to get process info: %s", str(e), exc_info=True)
                
                # Determine application type using detection rules
                app_type = self._detect_application_type(exe_name, class_name, title)
                
                return AppInfo(
                    title=title,
                    class_name=class_name,
                    exe=exe_name,
                    type=app_type
                )
                
            except Exception as e:
                logger.error("Failed to get application info: %s", str(e), exc_info=True)
                return default_info
    
    def _detect_application_type(self, exe_name: str, class_name: str, title: str) -> AppType:
        """Detect application type based on detection rules.
        
        Args:
            exe_name: Executable name (lowercase)
            class_name: Window class name (lowercase)
            title: Window title
            
        Returns:
            AppType: Detected application type
        """
        try:
            lower_class = class_name.lower() if class_name else ""
            lower_title = title.lower() if title else ""
            
            # Special case for Spotify since it's common
            if 'spotify' in exe_name.lower():
                logger.debug("Detected Spotify by executable name")
                return AppType.MEDIA
                
            # Check against detection rules
            for app_type, rule in self._app_detection_rules.items():
                # Check exe matches
                exe_match = any(match.lower() in exe_name.lower() for match in rule.exe_matches)
                
                # Check class matches
                class_match = any(match.lower() in lower_class for match in rule.class_matches)
                
                # Check title matches if any
                title_match = any(match.lower() in lower_title for match in rule.title_matches)
                
                if exe_match or class_match or title_match:
                    # Ensure we have a proper AppType enum, not an int
                    if not hasattr(app_type, 'name') or not isinstance(app_type, AppType):
                        logger.warning(f"Invalid app_type detected: {app_type}, defaulting to AppType.OTHER")
                        return AppType.OTHER
                        
                    logger.debug(
                        "Detected app type %s (exe: %s, class: %s, title: %s)",
                        app_type.name, exe_name, class_name, 
                        (title[:30] + "...") if title and len(title) > 30 else (title or "")
                    )
                    return app_type
            
            # Check for browser types
            if any(browser in exe_name.lower() for browser in ['chrome', 'msedge', 'firefox']):
                if 'chrome' in exe_name.lower() or 'msedge' in exe_name.lower():
                    return AppType.CHROMIUM
                return AppType.FIREFOX
            
            # Final check with UniversalMediaController
            try:
                controller = UniversalMediaController()
                running_apps = controller.list_running_media_apps()
                if running_apps and any(app in exe_name.lower() for app in running_apps):
                    return AppType.MEDIA
            except Exception as e:
                logger.debug(f"Error checking running media apps: {e}")
            
            return AppType.OTHER
            
        except Exception as e:
            logger.error(f"Error in _detect_application_type: {e}")
            return AppType.OTHER

    def _get_keyboard_state(self) -> Tuple[bool, bool, bool]:
        """Get the current state of modifier keys.
        
        Returns:
            Tuple[bool, bool, bool]: A tuple of (shift_pressed, ctrl_pressed, alt_pressed)
        """
        try:
            # Get the current keyboard state
            state = win32api.GetKeyState
            shift_pressed = bool(state(win32con.VK_SHIFT) & 0x8000)
            ctrl_pressed = bool(state(win32con.VK_CONTROL) & 0x8000)
            alt_pressed = bool(state(win32con.VK_MENU) & 0x8000)
            return shift_pressed, ctrl_pressed, alt_pressed
        except Exception as e:
            logger.warning(f"Error getting keyboard state: {e}")
            return False, False, False
    
    def _send_key_sequence(self, hwnd: HWND, vk_code: VKCode) -> bool:
        """Send a key sequence to the specified window.
        
        Args:
            hwnd: Target window handle
            vk_code: Virtual key code to send
            
        Returns:
            bool: True if the key was sent successfully, False otherwise
        """
        if not win32gui.IsWindow(hwnd):
            logger.error(f"Invalid window handle: 0x{hwnd:X}")
            return False
            
        try:
            # Get the scan code for the key
            scan_code = win32api.MapVirtualKey(vk_code, 0)
            
            # Build the lParam for key down
            lparam_down = 0x00000001  # repeat count = 1
            lparam_down |= (scan_code & 0xFF) << 16  # scan code
            
            # Build the lParam for key up
            lparam_up = lparam_down | (1 << 30) | (1 << 31)
            
            # Send key down
            win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, vk_code, lparam_down)
            time.sleep(0.01)  # Small delay between down and up
            
            # Send key up
            win32api.PostMessage(hwnd, win32con.WM_KEYUP, vk_code, lparam_up)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in _send_key_sequence: {e}", exc_info=True)
            return False
    def _method_direct_postmessage(self, hwnd: HWND, vk_code: VKCode, app_info: AppInfo) -> bool:
        """Send a key directly using PostMessage without stealing focus.
        
        This method sends the key press and release messages directly to the window
        without changing window focus or activation state.
        
        Args:
            hwnd: Target window handle
            vk_code: Virtual key code to send
            app_info: Application information
            
        Returns:
            bool: True if the key was sent successfully, False otherwise.
        """
        app_type = app_info.get('type', AppType.OTHER)
        
        # Only process if we have a valid window
        if not win32gui.IsWindow(hwnd):
            logger.error(f"Invalid window handle: 0x{hwnd:X}")
            return False
        
        # Get window title for logging
        window_title = win32gui.GetWindowText(hwnd)
        
        # Handle media controls for media players and browsers
        if app_type in [AppType.MEDIA, AppType.FIREFOX, AppType.CHROMIUM]:
            try:
                controller = UniversalMediaController()
                
                # Map VK codes to media controller methods
                if vk_code == win32con.VK_MEDIA_PLAY_PAUSE:
                    success, _ = controller.play_pause()
                    if success:
                        logger.debug(f"Successfully sent play/pause to {window_title}")
                        return True
                elif vk_code == win32con.VK_MEDIA_NEXT_TRACK or vk_code == win32con.VK_RIGHT:
                    success, _ = controller.next_track()
                    if success:
                        logger.debug(f"Successfully sent next track to {window_title}")
                        return True
                elif vk_code == win32con.VK_MEDIA_PREV_TRACK or vk_code == win32con.VK_LEFT:
                    success, _ = controller.previous_track()
                    if success:
                        logger.debug(f"Successfully sent previous track to {window_title}")
                        return True
                elif vk_code == win32con.VK_MEDIA_STOP:
                    success, _ = controller.stop()
                    if success:
                        logger.debug(f"Successfully sent stop to {window_title}")
                        return True
                # Handle volume controls for media players only (not browsers)
                elif vk_code == win32con.VK_UP and app_type == AppType.MEDIA:
                    success, _ = controller.volume_up()
                    if success:
                        logger.debug(f"Successfully increased volume for {window_title}")
                        return True
                elif vk_code == win32con.VK_DOWN and app_type == AppType.MEDIA:
                    success, _ = controller.volume_down()
                    if success:
                        logger.debug(f"Successfully decreased volume for {window_title}")
                        return True
                # Handle browser scrolling
                elif vk_code in [win32con.VK_UP, win32con.VK_DOWN] and app_type in [AppType.FIREFOX, AppType.CHROMIUM]:
                    # For browsers, we'll let the default handling continue to send the arrow keys
                    # which will naturally scroll the page
                    pass
                    
            except Exception as e:
                logger.warning(f"Error using UniversalMediaController for {window_title}: {e}")
                # Fall through to standard handling
        
        # Standard key sending for non-media keys or if media controller failed
        key_name = KEY_NAMES.get(vk_code, f'0x{vk_code:02X}')
        logger.debug(f"Sending key {key_name} (0x{vk_code:02X}) to window 0x{hwnd:X}")
        
        if not win32gui.IsWindow(hwnd):
            logger.error(f"Invalid window handle: 0x{hwnd:X}")
            return False
            
        with self._thread_lock:
            try:
                # Apply rate limiting
                self._rate_limit()
                
                # Get the scan code for the key
                scan_code = win32api.MapVirtualKey(vk_code, 0)
                
                # Check if this is an extended key (right alt/ctrl, numpad enter, etc.)
                is_extended = vk_code in {
                    win32con.VK_RMENU, win32con.VK_RCONTROL,
                    win32con.VK_RSHIFT, win32con.VK_RETURN,
                    win32con.VK_INSERT, win32con.VK_DELETE,
                    win32con.VK_HOME, win32con.VK_END,
                    win32con.VK_PRIOR, win32con.VK_NEXT,
                    win32con.VK_LEFT, win32con.VK_RIGHT,
                    win32con.VK_UP, win32con.VK_DOWN,
                    win32con.VK_NUMLOCK, win32con.VK_DIVIDE
                }
                
                # Build the lParam for key down
                lparam_down = 0x00000001  # repeat count = 1
                lparam_down |= (scan_code & 0xFF) << 16  # scan code
                if is_extended:
                    lparam_down |= 0x01000000  # set extended key flag
                
                # Get the current keyboard state for modifier keys
                shift_pressed, ctrl_pressed, alt_pressed = self._get_keyboard_state()
                
                # Set the context code (Alt key is down)
                if alt_pressed:
                    lparam_down |= (1 << 29)
                
                # Use thread attachment context manager
                with self._thread_attachment(hwnd) as attached:
                    if not attached:
                        logger.warning("Failed to attach to window thread, trying anyway...")
                    
                    # Send WM_KEYDOWN
                    win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, vk_code, lparam_down)
                    
                    # For character keys, send WM_CHAR
                    if 32 <= vk_code <= 126:  # ASCII printable characters
                        char = chr(vk_code)
                        if shift_pressed and char.isalpha():
                            char = char.upper()
                        char_code = ord(char)
                        win32api.PostMessage(hwnd, win32con.WM_CHAR, char_code, lparam_down)
                    
                    # Small delay between key down and key up
                    time.sleep(self.MESSAGE_DELAY)
                    
                    # Build the lParam for key up
                    lparam_up = lparam_down | (1 << 30) | (1 << 31)
                    
                    # Send WM_KEYUP
                    win32api.PostMessage(hwnd, win32con.WM_KEYUP, vk_code, lparam_up)
                
                return True
                
            except Exception as e:
                logger.error(f"Error in _method_direct_postmessage: {e}", exc_info=True)
                return False
        
        # Fall back to standard key sending for non-media keys or if media controller failed
        key_name = KEY_NAMES.get(vk_code, f'0x{vk_code:02X}')
        logger.info(f"Attempting to send key: {key_name} (VK: 0x{vk_code:02X}) to window 0x{hwnd:X}")
        
        try:
            # Skip if this is the ESC key (should be handled by the overlay)
            if vk_code == win32con.VK_ESCAPE:
                logger.debug("Skipping ESC key as it should be handled by the overlay")
                return False
                
            if not win32gui.IsWindow(hwnd):
                logger.error(f"Target window 0x{hwnd:X} is not valid")
                return False
            
            # Get window title for logging
            try:
                window_title = win32gui.GetWindowText(hwnd)
                logger.debug(f"Target window title: '{window_title}'")
                
                # Get the thread ID of the target window
                thread_id = win32process.GetWindowThreadProcessId(hwnd)[0]
                logger.debug(f"Target window thread ID: {thread_id}")
                
                # Get the current thread ID
                current_thread = win32api.GetCurrentThreadId()
                logger.debug(f"Current thread ID: {current_thread}")
                
                # Try to attach to the target window's thread if needed
                attached = False
                if thread_id != current_thread:
                    try:
                        win32process.AttachThreadInput(current_thread, thread_id, True)
                        attached = True
                        logger.debug("Attached to target window's thread")
                    except Exception as e:
                        logger.warning(f"Failed to attach to target window's thread: {e}")
                
            except Exception as e:
                logger.warning(f"Could not get window information: {e}")
            
            try:
                # Get the current keyboard state to handle modifiers
                shift_pressed, ctrl_pressed, alt_pressed = self._get_keyboard_state()
                logger.debug(f"Modifier keys - Shift: {shift_pressed}, Ctrl: {ctrl_pressed}, Alt: {alt_pressed}")
                
                # Send the key with current modifier state
                result = self._send_key_sequence(hwnd, vk_code)
                
                if result:
                    logger.info(f"Successfully sent key: {key_name} (0x{vk_code:02X})")
                else:
                    logger.warning(f"Failed to send key: {key_name} (0x{vk_code:02X})")
                
                return result
                
            finally:
                # Always detach from the thread if we attached to it
                if attached and 'thread_id' in locals() and 'current_thread' in locals():
                    try:
                        win32process.AttachThreadInput(current_thread, thread_id, False)
                        logger.debug("Detached from target window's thread")
                    except Exception as e:
                        logger.warning(f"Failed to detach from target window's thread: {e}")
            
        except Exception as e:
            logger.error(f"Error in _method_direct_postmessage for VK 0x{vk_code:02X}: {e}", exc_info=True)
            return False









    def _send_input(self, vk_code: VKCode) -> bool:
        """Low-level key press/release using SendInput.
        
        This is a fallback method that may be used in some cases.
        
        Args:
            vk_code: Virtual key code to send
            
        Returns:
            bool: True if the key was sent successfully
        """
        try:
            # Create keyboard input for key down
            keyboard_down = KEYBDINPUT()
            keyboard_down.wVk = vk_code
            keyboard_down.dwFlags = 0  # Key down
            
            # Create keyboard input for key up
            keyboard_up = KEYBDINPUT()
            keyboard_up.wVk = vk_code
            keyboard_up.dwFlags = InputType.KEYEVENTF_KEYUP
            
            # Create input structures
            input_down = INPUT()
            input_down.type = InputType.INPUT_KEYBOARD
            input_down.ii.ki = keyboard_down
            
            input_up = INPUT()
            input_up.type = InputType.INPUT_KEYBOARD
            input_up.ii.ki = keyboard_up
            
            # Send the inputs
            inputs = (INPUT * 2)(input_down, input_up)
            result = ctypes.windll.user32.SendInput(
                2,  # Number of inputs
                ctypes.byref(inputs),
                ctypes.sizeof(INPUT)
            )
            
            if result != 2:
                error = ctypes.windll.kernel32.GetLastError()
                logger.debug("SendInput failed. Result: %d, Error: 0x%X", result, error)
                return False
                
            return True
            
        except Exception as e:
            logger.debug("SendInput failed: %s", str(e), exc_info=True)
            return False

# Add missing HRESULT definition
if not hasattr(wintypes, 'HRESULT'):
    wintypes.HRESULT = ctypes.c_long