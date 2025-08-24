"""
Hotkey Manager Module

Provides centralized hotkey management for the application, including registration,
unregistration, and handling of global hotkeys.
"""

from core.logging import get_logger
from typing import Callable, Dict, Tuple, Any

from PySide6.QtCore import QObject, Signal, QMutex, QMutexLocker

import win32api
import win32con
import win32gui
import threading
import ctypes
from ctypes import wintypes
from typing import Optional, Any

# Centralized threading and resources
from core.threading import get_thread_manager, ThreadPoolType
from utils.resource_manager import get_resource_manager, ResourceType

logger = get_logger(__name__)

class HotkeyManager(QObject):
    """
    Manages global hotkeys for the application.
    
    This class handles registration, unregistration, and callback dispatching
    for global hotkeys. It's implemented as a singleton to ensure only one
    instance manages hotkeys system-wide.
    """
    
    # Signal emitted when a hotkey is pressed
    hotkey_triggered = Signal(str)  # hotkey_id
    
    _instance = None
    _lock = QMutex()
    
    def __new__(cls):
        """Implement singleton pattern with thread safety."""
        with QMutexLocker(cls._lock):
            if cls._instance is None:
                cls._instance = super(HotkeyManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the hotkey manager."""
        if self._initialized:
            return
            
        super().__init__()
        self._hotkeys: Dict[str, Tuple[Callable[..., None], Tuple[Any, ...]]] = {}
        self._system_hotkeys: Dict[str, int] = {}  # hotkey_id -> win32 hotkey id
        self._hotkey_id_counter = 1  # For generating unique win32 hotkey IDs
        self._hotkey_lock = threading.Lock()  # Thread safety for hotkey operations
        # Threading via ThreadManager
        self._message_thread = None  # legacy field retained for compatibility
        self._message_thread_running = False
        self._message_task_id: Optional[str] = None
        self._message_thread_id: Optional[int] = None
        # Suppression via low-level keyboard hook
        self._suppressed_hotkeys: Dict[str, Tuple[int, int]] = {}  # hotkey_id -> (modifiers, vk)
        self._keyboard_hook = None
        self._keyboard_hook_proc = None  # keep ref to avoid GC
        self._hook_lock = threading.Lock()
        # Keyboard library backend for single-key suppression
        self._kb_available = False
        self._kb_handlers: Dict[str, Any] = {}  # hotkey_id -> handler
        try:
            import keyboard  # type: ignore
            self._kb_available = True
            self._keyboard_lib = keyboard
            logger.debug("Keyboard library backend available for single-key suppression")
        except Exception as e:
            self._keyboard_lib = None
            logger.warning(f"Keyboard library backend not available: {e}")
        # ResourceManager registration for deterministic cleanup
        self._resource_id: Optional[Any] = None
        try:
            rm = get_resource_manager()
            self._resource_id = rm.register(
                self,
                ResourceType.CUSTOM,
                "HotkeyManager singleton",
                cleanup_handler=lambda obj: obj.shutdown(),
                tags={"hotkeys", "manager"},
            )
        except Exception as e:
            logger.debug(f"HotkeyManager ResourceManager registration skipped: {e}")
        self._initialized = True
        logger.debug("HotkeyManager initialized")
    
    def register_hotkey(self, hotkey_id: str, callback: Callable[..., None], *args: Any, sequence: str = None, suppress: bool = True, global_hotkey: bool = False) -> bool:
        """
        Register a hotkey with the given ID and callback.
        
        Args:
            hotkey_id: Unique identifier for the hotkey
            callback: Function to call when the hotkey is triggered
            *args: Arguments to pass to the callback
            sequence: The hotkey sequence (e.g., "ctrl+alt+s") - for compatibility
            suppress: Whether to suppress the hotkey from normal processing - for compatibility
            global_hotkey: Whether to register as a system-wide hotkey
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        logger.debug(f"register_hotkey called id={hotkey_id} sequence='{sequence}' global={global_hotkey} suppress={suppress}")
        if not hotkey_id:
            logger.error("Cannot register hotkey with empty ID")
            return False
            
        if hotkey_id in self._hotkeys:
            logger.warning(f"Hotkey {hotkey_id} is already registered")
            return False
            
        self._hotkeys[hotkey_id] = (callback, args)
        
        # Register as system-wide hotkey if requested
        if global_hotkey and sequence:
            modifiers, vk_code = self._parse_hotkey_sequence(sequence)
            # Policy: suppress only for single-key (no modifiers); combos -> RegisterHotKey (not suppressed)
            single_key, key_token = self._is_single_key(sequence)
            if suppress and single_key and modifiers == 0:
                if not self._kb_available:
                    if hotkey_id in self._hotkeys:
                        del self._hotkeys[hotkey_id]
                    logger.error("Keyboard backend not available for single-key suppression; aborting registration")
                    return False
                # Register single-key via keyboard lib with suppression
                ok = self._register_keyboard_single(hotkey_id, key_token)
                if not ok:
                    if hotkey_id in self._hotkeys:
                        del self._hotkeys[hotkey_id]
                    logger.error(f"Keyboard backend registration failed for {hotkey_id} '{sequence}'")
                    return False
                logger.debug(f"Registered suppressed single-key via keyboard backend: {hotkey_id} key='{key_token}'")
            else:
                # Non-suppressed (combos or suppress=False): use RegisterHotKey
                ok_sys = self._register_system_hotkey(hotkey_id, sequence)
                if not ok_sys:
                    logger.error(f"System hotkey registration failed for {hotkey_id} with sequence '{sequence}'")
                    if hotkey_id in self._hotkeys:
                        del self._hotkeys[hotkey_id]
                    return False
        
        logger.debug(f"Registered hotkey: {hotkey_id}")
        return True
    
    def unregister_hotkey(self, hotkey_id: str) -> bool:
        """
        Unregister a hotkey.
        
        Args:
            hotkey_id: ID of the hotkey to unregister
            
        Returns:
            bool: True if hotkey was unregistered, False if it wasn't found
        """
        # Unregister system-wide hotkey if it exists
        if hotkey_id in self._system_hotkeys:
            self._unregister_system_hotkey(hotkey_id)
        # Unregister keyboard backend handler if exists
        if hotkey_id in self._kb_handlers:
            try:
                self._unregister_keyboard_single(hotkey_id)
            except Exception as e:
                logger.error(f"Error unregistering keyboard backend for {hotkey_id}: {e}")
        # Remove from suppressed map if present
        removed_suppressed = False
        with self._hook_lock:
            if hotkey_id in self._suppressed_hotkeys:
                del self._suppressed_hotkeys[hotkey_id]
                removed_suppressed = True
            
        if hotkey_id in self._hotkeys:
            del self._hotkeys[hotkey_id]
            logger.debug(f"Unregistered hotkey: {hotkey_id}")
            # If we removed a suppressed hotkey and no more remain, release hook
            if removed_suppressed:
                with self._hook_lock:
                    if not self._suppressed_hotkeys:
                        self._uninstall_keyboard_hook()
            # If no more system/suppressed/kb handlers remain, stop the message loop
            if not self._system_hotkeys and not self._suppressed_hotkeys and not self._kb_handlers:
                self._stop_message_loop()
            return True
        return False
    
    def trigger_hotkey(self, hotkey_id: str) -> bool:
        """
        Trigger a hotkey manually.
        
        Args:
            hotkey_id: ID of the hotkey to trigger
            
        Returns:
            bool: True if hotkey was found and triggered, False otherwise
        """
        if hotkey_id not in self._hotkeys:
            return False
            
        callback, args = self._hotkeys[hotkey_id]
        try:
            callback(*args)
            logger.debug(f"Triggered hotkey: {hotkey_id}")
            self.hotkey_triggered.emit(hotkey_id)
            return True
        except Exception as e:
            logger.error(f"Error in hotkey callback for {hotkey_id}: {str(e)}")
            return False
    
    def clear_hotkeys(self) -> None:
        """Unregister all hotkeys."""
        # Unregister all system-wide hotkeys
        system_hotkey_ids = list(self._system_hotkeys.keys())
        for hotkey_id in system_hotkey_ids:
            self._unregister_system_hotkey(hotkey_id)
        
        self._hotkeys.clear()
        # Clear keyboard backend handlers
        for hk in list(self._kb_handlers.keys()):
            try:
                self._unregister_keyboard_single(hk)
            except Exception:
                pass
        self._kb_handlers.clear()
        with self._hook_lock:
            self._suppressed_hotkeys.clear()
        self._uninstall_keyboard_hook()
        logger.debug("Cleared all hotkeys")
        # Stop message loop after clearing
        self._stop_message_loop()
    
    def _parse_hotkey_sequence(self, sequence: str) -> Tuple[int, int]:
        """
        Parse a hotkey sequence string into win32 modifiers and virtual key code.
        
        Args:
            sequence: Hotkey sequence (e.g., "ctrl+alt+shift+key" or "-" or "=")
            
        Returns:
            Tuple of (modifiers, vk_code)
        """
        modifiers = 0
        vk_code = 0
        
        parts = [part.strip().lower() for part in sequence.split('+')]
        
        # Parse modifiers
        if 'ctrl' in parts:
            modifiers |= win32con.MOD_CONTROL
        if 'alt' in parts:
            modifiers |= win32con.MOD_ALT
        if 'shift' in parts:
            modifiers |= win32con.MOD_SHIFT
        
        # Parse key
        key_part = [part for part in parts if part not in ['ctrl', 'alt', 'shift']]
        if key_part:
            key = key_part[0]
            # Handle special keys
            if key == '-':
                vk_code = win32con.VK_OEM_MINUS
            elif key == '=':
                vk_code = win32con.VK_OEM_PLUS
            elif key in {'`', '~', 'tilde', 'grave', 'oem3', 'oem_3'}:
                # Backtick/tilde key: prefer VK_OEM_3 if available, otherwise 0xC0
                vk_code = getattr(win32con, 'VK_OEM_3', 0)
                if vk_code == 0:
                    # Use standard US layout code 0xC0; log for diagnostics
                    vk_code = 0xC0
                    logger.debug("VK_OEM_3 not found in win32con; using 0xC0 for backtick/tilde")
            elif key == 'space':
                vk_code = win32con.VK_SPACE
            elif key == 'enter':
                vk_code = win32con.VK_RETURN
            elif key == 'esc' or key == 'escape':
                vk_code = win32con.VK_ESCAPE
            elif key == 'tab':
                vk_code = win32con.VK_TAB
            elif len(key) == 1 and key.isalpha():
                # Single letter
                vk_code = ord(key.upper())
            elif len(key) == 1 and key.isdigit():
                # Single digit
                vk_code = ord(key)
            else:
                # Try to get virtual key code from win32con
                vk_name = f"VK_{key.upper()}"
                if hasattr(win32con, vk_name):
                    vk_code = getattr(win32con, vk_name)
        
        return modifiers, vk_code
    
    def _register_system_hotkey(self, hotkey_id: str, sequence: str) -> bool:
        """
        Register a system-wide hotkey using win32 API.
        
        Args:
            hotkey_id: Unique identifier for the hotkey
            sequence: The hotkey sequence (e.g., "ctrl+alt+s")
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        try:
            with self._hotkey_lock:
                logger.debug(f"Attempting to register system hotkey {hotkey_id} with sequence '{sequence}'")
                
                # Parse the hotkey sequence
                modifiers, vk_code = self._parse_hotkey_sequence(sequence)
                
                logger.debug(f"Parsed sequence '{sequence}' -> modifiers: {modifiers}, vk_code: {vk_code}")
                
                if vk_code == 0:
                    logger.error(f"Failed to parse hotkey sequence: {sequence}")
                    return False
                
                # Generate a unique win32 hotkey ID
                win32_hotkey_id = self._hotkey_id_counter
                self._hotkey_id_counter += 1
                
                logger.debug(f"Generated win32 hotkey ID: {win32_hotkey_id}")
                
                # Register the hotkey with win32
                result = win32api.RegisterHotKey(None, win32_hotkey_id, modifiers, vk_code)
                
                logger.debug(f"win32api.RegisterHotKey result: {result}")
                
                if result:
                    self._system_hotkeys[hotkey_id] = win32_hotkey_id
                    logger.debug(f"Registered system hotkey {hotkey_id} with sequence {sequence}")
                    
                    # Start message loop if not already running
                    if not self._message_thread_running:
                        logger.debug("Starting message loop thread")
                        self._start_message_loop()
                    
                    return True
                else:
                    logger.error(f"Failed to register system hotkey {hotkey_id} with sequence {sequence}")
                    return False
        except Exception as e:
            logger.error(f"Exception while registering system hotkey {hotkey_id}: {str(e)}")
            return False
    
    def _unregister_system_hotkey(self, hotkey_id: str) -> bool:
        """
        Unregister a system-wide hotkey.
        
        Args:
            hotkey_id: ID of the hotkey to unregister
            
        Returns:
            bool: True if hotkey was unregistered, False if it wasn't found
        """
        try:
            with self._hotkey_lock:
                if hotkey_id in self._system_hotkeys:
                    win32_hotkey_id = self._system_hotkeys[hotkey_id]
                    result = win32api.UnregisterHotKey(None, win32_hotkey_id)
                    
                    if result:
                        del self._system_hotkeys[hotkey_id]
                        logger.debug(f"Unregistered system hotkey {hotkey_id}")
                        return True
                    else:
                        logger.error(f"Failed to unregister system hotkey {hotkey_id}")
                        return False
                else:
                    logger.warning(f"System hotkey {hotkey_id} not found for unregistration")
                    return False
        except Exception as e:
            logger.error(f"Exception while unregistering system hotkey {hotkey_id}: {str(e)}")
            return False
    
    def _start_message_loop(self) -> None:
        """Start the message loop thread for processing hotkey events."""
        logger.debug("_start_message_loop called")
        if self._message_thread_running:
            logger.debug("Message loop already running")
            return
        # Submit long-running loop to ThreadManager (IO pool)
        try:
            tm = get_thread_manager()
            self._message_thread_running = True
            self._message_task_id = tm.submit_task(
                ThreadPoolType.IO,
                self._message_loop,
                task_id="hotkeys_message_loop",
                resource_tags={"hotkeys", "message_loop"},
            )
            logger.debug(f"Started hotkey message loop on IO pool via ThreadManager task_id={self._message_task_id}")
        except Exception as e:
            self._message_thread_running = False
            logger.error(f"Failed to start message loop via ThreadManager: {e}")
    
    def _stop_message_loop(self) -> None:
        """Stop the message loop thread."""
        running_before = self._message_thread_running
        self._message_thread_running = False
        # Post WM_QUIT to the loop thread to wake GetMessage
        try:
            if self._message_thread_id is not None:
                win32api.PostThreadMessage(int(self._message_thread_id), win32con.WM_QUIT, 0, 0)
        except Exception as e:
            if running_before:
                logger.debug(f"PostThreadMessage WM_QUIT failed: {e}")
    
    def _message_loop(self) -> None:
        """Message loop for processing hotkey events."""
        logger.debug("Hotkey message loop started")
        # Record this thread's ID for shutdown signaling
        try:
            self._message_thread_id = int(win32api.GetCurrentThreadId())
            logger.debug(f"Hotkey message loop thread id={self._message_thread_id}")
        except Exception:
            self._message_thread_id = None
        
        # Retain LL hook support only if explicitly populated (not used for single-key keyboard backend)
        self._install_keyboard_hook()
        
        try:
            while self._message_thread_running:
                # Use GetMessage to wait for messages
                try:
                    msg = win32gui.GetMessage(None, 0, 0)
                    if msg and msg[1] == win32con.WM_HOTKEY:
                        # Process hotkey message
                        hotkey_id = None
                        win32_hotkey_id = msg[2]
                        
                        # Find our hotkey ID
                        for hk_id, win32_id in self._system_hotkeys.items():
                            if win32_id == win32_hotkey_id:
                                hotkey_id = hk_id
                                break
                        
                        if hotkey_id and hotkey_id in self._hotkeys:
                            logger.debug(f"WM_HOTKEY received for {hotkey_id} (win32_id={win32_hotkey_id}); dispatching")
                            # Trigger the hotkey callback
                            self.trigger_hotkey(hotkey_id)
                    
                    # Dispatch other messages
                    if msg:
                        win32gui.TranslateMessage(msg)
                        win32gui.DispatchMessage(msg)
                except Exception as e:
                    if self._message_thread_running:
                        logger.error(f"Error in hotkey message loop: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Exception in hotkey message loop: {str(e)}")
        finally:
            # Always uninstall hook on loop exit
            self._uninstall_keyboard_hook()
            logger.debug("Hotkey message loop stopped")
            # Clear recorded thread id
            self._message_thread_id = None

    # -------------------- Lifecycle --------------------
    def shutdown(self) -> None:
        """Gracefully stop the message loop and unregister resources."""
        try:
            self.clear_hotkeys()
        except Exception:
            pass
        try:
            self._stop_message_loop()
        except Exception:
            pass
        # Best-effort unregister from ResourceManager
        try:
            if self._resource_id is not None:
                rm = get_resource_manager()
                rm.unregister(self._resource_id, force=True)
                self._resource_id = None
        except Exception:
            pass

    # -------------------- Low-level keyboard hook (suppression) --------------------
    def _install_keyboard_hook(self) -> bool:
        """Install WH_KEYBOARD_LL hook if suppressed hotkeys are present and hook not set.
        Returns True on success, False otherwise."""
        with self._hook_lock:
            if self._keyboard_hook is not None:
                return True
            if not self._suppressed_hotkeys:
                return True

            # ctypes user32 APIs
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            # Structures and constants
            class KBDLLHOOKSTRUCT(ctypes.Structure):
                _fields_ = [
                    ("vkCode", wintypes.DWORD),
                    ("scanCode", wintypes.DWORD),
                    ("flags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    # Use pointer-sized field; wintypes.ULONG_PTR may be missing on some Python builds
                    ("dwExtraInfo", ctypes.c_void_p),
                ]

            # Define LowLevelKeyboardProc: LRESULT CALLBACK(int, WPARAM, LPARAM)
            LLKPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

            def low_level_proc(nCode, wParam, lParam):
                try:
                    if nCode < 0:
                        return user32.CallNextHookEx(self._keyboard_hook, nCode, wParam, lParam)

                    if wParam in (win32con.WM_KEYDOWN, win32con.WM_SYSKEYDOWN):
                        kbd_struct = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                        vk = kbd_struct.vkCode
                        mods = self._read_modifiers()
                        target_id = None
                        with self._hook_lock:
                            for hk_id, (req_mods, req_vk) in self._suppressed_hotkeys.items():
                                if vk == req_vk and mods == req_mods:
                                    target_id = hk_id
                                    break
                        if target_id and target_id in self._hotkeys:
                            logger.debug(f"LLHOOK match for {target_id} (vk={vk}, mods={mods}); suppressing and dispatching")
                            self.trigger_hotkey(target_id)
                            return 1  # swallow

                    return user32.CallNextHookEx(self._keyboard_hook, nCode, wParam, lParam)
                except Exception as e:
                    logger.error(f"Error in keyboard hook: {e}")
                    return user32.CallNextHookEx(self._keyboard_hook, nCode, wParam, lParam)

            self._keyboard_hook_proc = LLKPROC(low_level_proc)
            try:
                # Strategy 1: use module handle
                h_mod = kernel32.GetModuleHandleW(None)
                self._keyboard_hook = user32.SetWindowsHookExW(
                    win32con.WH_KEYBOARD_LL, self._keyboard_hook_proc, h_mod, 0
                )
                if not self._keyboard_hook:
                    # Strategy 2: try with hMod = 0 (commonly used for LL hooks)
                    self._keyboard_hook = user32.SetWindowsHookExW(
                        win32con.WH_KEYBOARD_LL, self._keyboard_hook_proc, 0, 0
                    )
                if not self._keyboard_hook:
                    logger.error("Failed to install low-level keyboard hook for suppressed hotkeys")
                    self._keyboard_hook_proc = None
                    return False
                else:
                    logger.debug("Installed low-level keyboard hook for suppressed hotkeys")
                    return True
            except Exception as e:
                logger.error(f"Exception installing keyboard hook: {e}")
                self._keyboard_hook = None
                self._keyboard_hook_proc = None
                return False

    def _uninstall_keyboard_hook(self) -> None:
        with self._hook_lock:
            if self._keyboard_hook is not None:
                try:
                    user32 = ctypes.windll.user32
                    user32.UnhookWindowsHookEx(self._keyboard_hook)
                    logger.debug("Uninstalled low-level keyboard hook")
                except Exception as e:
                    logger.error(f"Exception uninstalling keyboard hook: {e}")
                finally:
                    self._keyboard_hook = None
                    self._keyboard_hook_proc = None

    def _read_modifiers(self) -> int:
        """Return current modifier mask using GetKeyState."""
        mods = 0
        try:
            if win32api.GetKeyState(win32con.VK_CONTROL) & 0x8000:
                mods |= win32con.MOD_CONTROL
            if win32api.GetKeyState(win32con.VK_MENU) & 0x8000:
                mods |= win32con.MOD_ALT
            if win32api.GetKeyState(win32con.VK_SHIFT) & 0x8000:
                mods |= win32con.MOD_SHIFT
        except Exception:
            pass
        return mods

    # -------------------- Keyboard library backend (single-key suppression) --------------------
    def _is_single_key(self, sequence: str) -> Tuple[bool, Optional[str]]:
        parts = [p.strip() for p in sequence.replace('+', ' + ').split('+') if p.strip()]
        if len(parts) == 1:
            return True, parts[0]
        return False, None

    def _register_keyboard_single(self, hotkey_id: str, key_token: str) -> bool:
        try:
            if hotkey_id in self._kb_handlers:
                self._unregister_keyboard_single(hotkey_id)
            kb = self._keyboard_lib
            # Use on_press_key with suppress=True
            def _cb(e):
                try:
                    # Strict single-key: do not dispatch if modifiers are pressed
                    if kb.is_pressed('ctrl') or kb.is_pressed('alt') or kb.is_pressed('shift'):
                        return
                    logger.debug(f"KB backend match for {hotkey_id} key='{key_token}'; suppressing and dispatching")
                    self.trigger_hotkey(hotkey_id)
                except Exception as ex:
                    logger.error(f"Error in keyboard backend callback for {hotkey_id}: {ex}")
            h = kb.on_press_key(key_token, _cb, suppress=True)
            self._kb_handlers[hotkey_id] = h
            return True
        except Exception as e:
            logger.error(f"Keyboard backend registration error for {hotkey_id} '{key_token}': {e}")
            return False

    def _unregister_keyboard_single(self, hotkey_id: str) -> None:
        kb = self._keyboard_lib
        h = self._kb_handlers.pop(hotkey_id, None)
        if h is not None:
            try:
                kb.unhook(h)
            except Exception:
                pass
    
    def is_hotkey_registered(self, hotkey_id: str) -> bool:
        """
        Check if a hotkey is registered.
        
        Args:
            hotkey_id: ID of the hotkey to check
            
        Returns:
            bool: True if the hotkey is registered, False otherwise
        """
        return hotkey_id in self._hotkeys
    
    def get_registered_hotkeys(self) -> Dict[str, Tuple[Callable[..., None], Tuple[Any, ...]]]:
        """
        Get all registered hotkeys.
        
        Returns:
            Dict mapping hotkey IDs to (callback, args) tuples
        """
        return self._hotkeys.copy()