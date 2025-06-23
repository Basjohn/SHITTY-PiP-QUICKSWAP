import logging
import win32gui
import win32api
import win32con
import win32process
import ctypes
import ctypes.wintypes
import time
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)

# Additional Windows API constants
WM_ACTIVATE = 0x0006
WA_CLICKACTIVE = 2
WM_SETFOCUS = 0x0007
WM_KILLFOCUS = 0x0008
WM_SYSCOMMAND = 0x0112
SC_HOTKEY = 0xF150

# Hook constants
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

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

# Constants
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

class KeyPassthrough:
    def __init__(self):
        self._enabled = False
        self._aggressive_mode = False
        self._target_hwnd = None
        self._app_strategies = {}  # Cache strategies per application

    def set_target_window(self, hwnd: int):
        """Set the target window for key passthrough."""
        self._target_hwnd = hwnd
        logger.debug(f"Key passthrough target window set to: {hwnd}")

    def set_enabled(self, enabled: bool):
        """Enable or disable key passthrough."""
        self._enabled = enabled
        logger.info(f"Key passthrough {'enabled' if enabled else 'disabled'}")
        return enabled

    def is_enabled(self) -> bool:
        """Check if key passthrough is enabled."""
        return self._enabled

    def set_aggressive_mode(self, enabled: bool):
        """Enable or disable aggressive key passthrough mode."""
        self._aggressive_mode = enabled
        logger.info(f"Aggressive key passthrough {'enabled' if enabled else 'disabled'}")
        return enabled

    def is_aggressive_mode(self) -> bool:
        """Check if aggressive mode is enabled."""
        return self._aggressive_mode

    def send_key(self, vk_code: int) -> bool:
        """Send a key to the target window if enabled."""
        if not self._enabled or not self._target_hwnd:
            return False
            
        key_name = {
            win32con.VK_SPACE: 'SPACE',
            win32con.VK_RETURN: 'ENTER',
            win32con.VK_LEFT: 'LEFT',
            win32con.VK_RIGHT: 'RIGHT'
        }.get(vk_code, f'VK_{vk_code}')
        
        logger.debug(f"Sending {key_name} to window 0x{self._target_hwnd:X}")
        
        return self._send_key_universal(self._target_hwnd, vk_code)

    def send_media_play_pause(self):
        """Send space key for play/pause."""
        return self.send_key(win32con.VK_SPACE)
        
    def send_media_next_track(self):
        """Send right arrow for next track."""
        return self.send_key(win32con.VK_RIGHT)
        
    def send_media_previous_track(self):
        """Send left arrow for previous track."""
        return self.send_key(win32con.VK_LEFT)
        
    def send_space(self):
        """Send space key."""
        return self.send_key(win32con.VK_SPACE)
        
    def send_enter(self):
        """Send enter key."""
        return self.send_key(win32con.VK_RETURN)

    def _send_key_universal(self, hwnd: int, vk_code: int) -> bool:
        """Universal key sending that works with modern applications."""
        if not win32gui.IsWindow(hwnd):
            logger.error(f"Invalid window handle: 0x{hwnd:X}")
            return False

        app_info = self._get_app_info(hwnd)
        key_name = {
            win32con.VK_SPACE: 'SPACE',
            win32con.VK_RETURN: 'ENTER', 
            win32con.VK_LEFT: 'LEFT',
            win32con.VK_RIGHT: 'RIGHT'
        }.get(vk_code, f'VK_{vk_code}')
        
        logger.debug(f"=== Sending {key_name} to {app_info['type']} ===")
        logger.debug(f"Window: '{app_info['title']}' (Class: {app_info['class']})")
        logger.debug(f"Process: {app_info['exe']}")

        # Try methods in order of effectiveness
        methods = [
            ("Quick Focus + SendInput", self._method_quick_focus_sendinput),
            ("Activation Trick", self._method_activation_trick),
            ("Thread Attach + SendInput", self._method_thread_attach),
            ("Cursor + Click + SendInput", self._method_cursor_click),
        ]
        
        if self._aggressive_mode:
            methods.insert(0, ("Full Focus Steal", self._method_full_focus_steal))

        for method_name, method_func in methods:
            try:
                logger.debug(f"Trying method: {method_name}")
                if method_func(hwnd, vk_code, app_info):
                    logger.debug(f"✓ SUCCESS: {method_name} worked for {key_name}")
                    return True
                else:
                    logger.debug(f"✗ FAILED: {method_name}")
            except Exception as e:
                logger.debug(f"✗ ERROR in {method_name}: {e}")

        logger.warning(f"All methods failed for {key_name}")
        return False

    def _get_app_info(self, hwnd: int) -> Dict[str, str]:
        """Get comprehensive application information."""
        try:
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION, False, pid)
                exe_path = win32process.GetModuleFileNameEx(handle, 0)
                win32api.CloseHandle(handle)
                exe_name = exe_path.split('\\')[-1].lower()
            except:
                exe_name = "unknown"

            # Determine application type
            app_type = "other"
            if "firefox" in exe_name or "mozilla" in class_name.lower():
                app_type = "firefox"
            elif any(name in exe_name for name in ["chrome", "msedge", "opera"]) or "chrome" in class_name.lower():
                app_type = "chromium"
            elif any(name in exe_name for name in ["spotify", "vlc", "mpv", "wmplayer", "musicbee"]):
                app_type = "media"
            elif "unity" in class_name.lower() or exe_name.endswith("impact.exe"):
                app_type = "game"

            return {
                "title": title,
                "class": class_name,
                "exe": exe_name,
                "type": app_type
            }
        except Exception as e:
            logger.debug(f"Failed to get app info: {e}")
            return {"title": "", "class": "", "exe": "", "type": "other"}

    def _method_quick_focus_sendinput(self, hwnd: int, vk_code: int, app_info: Dict) -> bool:
        """Quick focus method - minimal disruption."""
        try:
            original_fg = win32gui.GetForegroundWindow()
            
            # Quick focus
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.02)  # Very brief pause
            
            # Send key
            success = self._send_input_key(vk_code)
            
            # Quick restore (don't wait)
            if original_fg and win32gui.IsWindow(original_fg):
                win32gui.SetForegroundWindow(original_fg)
            
            return success
        except:
            return False

    def _method_activation_trick(self, hwnd: int, vk_code: int, app_info: Dict) -> bool:
        """Use window activation messages to 'wake up' the window."""
        try:
            # Send activation message
            win32api.SendMessage(hwnd, WM_ACTIVATE, WA_CLICKACTIVE, 0)
            
            # Brief pause for processing
            time.sleep(0.01)
            
            # Try to focus and send key very quickly
            original_fg = win32gui.GetForegroundWindow()
            win32gui.SetForegroundWindow(hwnd)
            
            # Send immediately
            success = self._send_input_key(vk_code)
            
            # Restore immediately 
            if original_fg and win32gui.IsWindow(original_fg):
                win32gui.SetForegroundWindow(original_fg)
                
            return success
        except:
            return False

    def _method_thread_attach(self, hwnd: int, vk_code: int, app_info: Dict) -> bool:
        """Attach to the target window's thread for better input delivery."""
        try:
            current_thread = win32api.GetCurrentThreadId()
            target_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
            
            if target_thread == current_thread:
                return False
                
            # Attach to target thread
            win32process.AttachThreadInput(current_thread, target_thread, True)
            
            try:
                # Now we're attached, try to focus and send
                win32gui.SetForegroundWindow(hwnd)
                win32gui.SetFocus(hwnd)
                time.sleep(0.01)
                
                success = self._send_input_key(vk_code)
                
                return success
            finally:
                # Always detach
                try:
                    win32process.AttachThreadInput(current_thread, target_thread, False)
                except:
                    pass
                    
        except:
            return False

    def _method_cursor_click(self, hwnd: int, vk_code: int, app_info: Dict) -> bool:
        """Move cursor to window and simulate a click to activate, then send key."""
        try:
            # Get window rect
            rect = win32gui.GetWindowRect(hwnd)
            center_x = (rect[0] + rect[2]) // 2
            center_y = (rect[1] + rect[3]) // 2
            
            # Save cursor position
            old_pos = win32gui.GetCursorPos()
            
            try:
                # Move cursor to window center
                win32api.SetCursorPos(center_x, center_y)
                
                # Simulate a very light click to activate (down and up immediately)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                
                # Brief pause
                time.sleep(0.02)
                
                # Send key
                success = self._send_input_key(vk_code)
                
                return success
                
            finally:
                # Restore cursor
                win32api.SetCursorPos(old_pos[0], old_pos[1])
                
        except:
            return False

    def _method_full_focus_steal(self, hwnd: int, vk_code: int, app_info: Dict) -> bool:
        """Aggressive method - fully steal focus temporarily."""
        try:
            original_fg = win32gui.GetForegroundWindow()
            original_active = win32gui.GetActiveWindow()
            
            # Restore window if minimized
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(0.1)
            
            # Force foreground
            win32gui.SetForegroundWindow(hwnd)
            win32gui.SetActiveWindow(hwnd)
            win32gui.SetFocus(hwnd)
            
            # Give it time to properly focus
            time.sleep(0.1)
            
            # Send key
            success = self._send_input_key(vk_code)
            
            # Wait a bit for key to be processed
            time.sleep(0.05)
            
            # Restore original windows
            if original_fg and win32gui.IsWindow(original_fg):
                win32gui.SetForegroundWindow(original_fg)
                if original_active and win32gui.IsWindow(original_active):
                    win32gui.SetActiveWindow(original_active)
            
            return success
            
        except:
            return False

    def _send_input_key(self, vk_code: int) -> bool:
        """Send key using SendInput - the most reliable system-level method."""
        try:
            # Create input array for key down and key up
            inputs = (INPUT * 2)()
            
            # Key down event
            inputs[0].type = INPUT_KEYBOARD
            inputs[0]._input.ki.wVk = vk_code
            inputs[0]._input.ki.wScan = win32api.MapVirtualKey(vk_code, 0)
            inputs[0]._input.ki.dwFlags = 0
            inputs[0]._input.ki.time = 0
            inputs[0]._input.ki.dwExtraInfo = None
            
            # Key up event
            inputs[1].type = INPUT_KEYBOARD
            inputs[1]._input.ki.wVk = vk_code
            inputs[1]._input.ki.wScan = win32api.MapVirtualKey(vk_code, 0)
            inputs[1]._input.ki.dwFlags = KEYEVENTF_KEYUP
            inputs[1]._input.ki.time = 0
            inputs[1]._input.ki.dwExtraInfo = None
            
            # Send the input
            result = ctypes.windll.user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
            
            if result != 2:
                logger.debug(f"SendInput failed, returned {result}")
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"SendInput exception: {e}")
            return False