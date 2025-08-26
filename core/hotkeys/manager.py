"""
Hotkey Manager Module

Provides centralized hotkey management for the application, including registration,
unregistration, and handling of global hotkeys.
"""

from core.logging import get_logger, throttled, log_dedupe
from typing import Callable, Dict, Tuple, Any

from PySide6.QtCore import QObject, Signal, QMutex, QMutexLocker

import win32api
import win32con
import win32gui
import threading
from typing import Optional

# Centralized threading and resources
from core.threading import ThreadManager, get_thread_manager, ThreadPoolType
from utils.lockfree.spsc_queue import SPSCQueue
from core.settings import get_settings_manager
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
        # Optional release callbacks for keyboard-backend single-key registrations
        self._hotkeys_release: Dict[str, Tuple[Callable[..., None], Tuple[Any, ...]]] = {}
        self._system_hotkeys: Dict[str, int] = {}  # hotkey_id -> win32 hotkey id
        self._hotkey_id_counter = 1  # For generating unique win32 hotkey IDs
        # Legacy locks (to be removed after full confinement to hotkey thread)
        self._hotkey_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._kb_lock = threading.Lock()
        self._message_thread: Optional[threading.Thread] = None
        self._message_thread_id: Optional[int] = None
        self._message_thread_running: bool = False
        # Custom WM_APP messages for marshalled commands
        self._WM_APP_REGISTER = win32con.WM_APP + 1
        self._WM_APP_UNREGISTER = win32con.WM_APP + 2
        # ThreadManager integration and lock-free command queue (P1)
        try:
            self._tm = get_thread_manager()
            self._cmd_queue: SPSCQueue = self._tm.create_spsc_queue(128)
        except Exception:
            self._tm = None
            self._cmd_queue = None  # type: ignore
        # Cross-thread marshal for registration/unregistration
        self._pending_cmds: Dict[int, Any] = {}
        try:
            self._WM_APP_REGISTER = int(getattr(win32con, 'WM_APP', 0x8000)) + 1
            self._WM_APP_UNREGISTER = int(getattr(win32con, 'WM_APP', 0x8000)) + 2
        except Exception:
            self._WM_APP_REGISTER = 0x8001
            self._WM_APP_UNREGISTER = 0x8002
        # Keyboard library backend for single-key suppression
        self._kb_available = False
        self._kb_handlers: Dict[str, Any] = {}  # hotkey_id -> handler or (press, release)
        self._kb_press_state: Dict[str, bool] = {}  # hotkey_id -> is_pressed gating
        self._kb_gate_tokens: Dict[str, int] = {}  # hotkey_id -> watchdog token
        # Keyboard library combo handlers (fallback for system-hotkey failures)
        self._kb_combo_handlers: Dict[str, Any] = {}  # hotkey_id -> keyboard combo handle
        self._kb_lock = threading.Lock()  # guard keyboard backend handler/state maps
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
        # HWND for message-only window (created on message thread)
        self._hwnd: Optional[int] = None
        # Settings toggles (P2)
        try:
            sm = get_settings_manager()
            self._prefer_kb_fallback: bool = bool(sm.get('hotkeys.prefer_keyboard_fallback', False))
            self._allow_single_digits: bool = bool(sm.get('hotkeys.allow_single_digits', False))
        except Exception:
            self._prefer_kb_fallback = False
            self._allow_single_digits = False
        # Throttled debug emitters to reduce log spam during high-frequency events
        try:
            self._t_debug_modheld = throttled(logger.debug, "hotkeys:modheld", 1000)
            self._t_debug_repeat = throttled(logger.debug, "hotkeys:repeat", 500)
            self._t_debug_release = throttled(logger.debug, "hotkeys:release", 500)
            self._t_debug_watchdog = throttled(logger.debug, "hotkeys:watchdog", 1000)
            # Example dedupe channel (available if needed):
            self._d_debug_general = log_dedupe(logger.debug, "hotkeys:dedupe", 1500)
        except Exception:
            # If helpers are unavailable for any reason, continue without throttling
            self._t_debug_modheld = logger.debug
            self._t_debug_repeat = logger.debug
            self._t_debug_release = logger.debug
            self._t_debug_watchdog = logger.debug
            self._d_debug_general = logger.debug
        logger.debug("HotkeyManager initialized")
    
    def register_hotkey(self, hotkey_id: str, callback: Callable[..., None], *args: Any, sequence: str = None, suppress: bool = True, global_hotkey: bool = False, on_release: Optional[Callable[..., None]] = None, release_args: Tuple[Any, ...] = ()) -> bool:
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
        # Store optional release callback (keyboard-backend only; ignored for system hotkeys)
        if on_release is not None:
            try:
                self._hotkeys_release[hotkey_id] = (on_release, release_args)
            except Exception:
                pass
        
        # Registration logic
        if sequence:
            # Determine if this is a single token and whether modifiers are present
            single_key, key_token = self._is_single_key(sequence)
            # For strict single-key tokens, modifiers must be 0; avoid VK parsing
            if single_key:
                modifiers = 0
            else:
                modifiers, _ = self._parse_hotkey_sequence(sequence)

            # Suppressed single-key path (keyboard backend) — independent of global_hotkey
            if suppress and single_key and modifiers == 0:
                normalized = (key_token or '').strip().lower()
                if not self._kb_available:
                    if hotkey_id in self._hotkeys:
                        del self._hotkeys[hotkey_id]
                    logger.error("Keyboard backend not available for single-key suppression; aborting registration")
                    return False
                # Attempt keyboard single-key registration for any token (safe or not);
                # caller explicitly requested suppress=True which implies keyboard backend.
                aliases = self._aliases_for_single_key(normalized)
                ok = self._register_keyboard_single(hotkey_id, aliases)
                if not ok:
                    if hotkey_id in self._hotkeys:
                        del self._hotkeys[hotkey_id]
                    logger.error(f"Keyboard backend registration failed for {hotkey_id} '{sequence}' (single-key)")
                    return False
                logger.debug(f"Registered suppressed single-key via keyboard backend: {hotkey_id} keys={aliases}")
            # System-wide registration for combos or when explicitly requested
            elif global_hotkey:
                # If user prefers keyboard fallback, try that path first
                if self._prefer_kb_fallback:
                    kb_ok = self._register_keyboard_combo(hotkey_id, sequence)
                    if kb_ok:
                        logger.debug(f"Preferred keyboard combo registered for {hotkey_id}: '{sequence}'")
                        logger.debug(f"Registered hotkey: {hotkey_id}")
                        return True
                    # If keyboard registration fails, fall through to system attempt
                ok_sys = self._register_system_hotkey(hotkey_id, sequence)
                if not ok_sys:
                    logger.error(f"System hotkey registration failed for {hotkey_id} with sequence '{sequence}'")
                    # Fallback: keyboard library combo hotkey with suppression
                    kb_ok = self._register_keyboard_combo(hotkey_id, sequence)
                    if not kb_ok:
                        if hotkey_id in self._hotkeys:
                            del self._hotkeys[hotkey_id]
                        return False
                    logger.debug(f"Registered keyboard combo fallback for {hotkey_id}: '{sequence}'")
        
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
        # Unregister keyboard combo handler if exists
        if hotkey_id in self._kb_combo_handlers:
            try:
                self._unregister_keyboard_combo(hotkey_id)
            except Exception as e:
                logger.error(f"Error unregistering keyboard combo backend for {hotkey_id}: {e}")
        
        if hotkey_id in self._hotkeys:
            del self._hotkeys[hotkey_id]
            logger.debug(f"Unregistered hotkey: {hotkey_id}")
            # Clear optional release callback mapping
            try:
                if hotkey_id in self._hotkeys_release:
                    del self._hotkeys_release[hotkey_id]
            except Exception:
                pass
            # If no more system/kb handlers remain, stop the message loop
            if not self._system_hotkeys and not self._kb_handlers and not self._kb_combo_handlers:
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
        try:
            self._hotkeys_release.clear()
        except Exception:
            pass
        # Clear keyboard backend handlers (single-key suppress)
        for hk in list(self._kb_handlers.keys()):
            try:
                self._unregister_keyboard_single(hk)
            except Exception:
                pass
        self._kb_handlers.clear()
        # Clear keyboard combo handlers
        for hk in list(self._kb_combo_handlers.keys()):
            try:
                self._unregister_keyboard_combo(hk)
            except Exception:
                pass
        self._kb_combo_handlers.clear()
        logger.debug("Cleared all hotkeys")
        # Stop message loop after clearing
        self._stop_message_loop()
    
    def _parse_hotkey_sequence(self, sequence: str) -> Tuple[int, int]:
        """
        Parse a hotkey sequence string into win32 modifiers and virtual key code.
        
        Args:
            sequence: Hotkey sequence (e.g., "ctrl+alt+s" or "-" or "=")
            
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
                # Some pywin32 builds may not expose VK_OEM_MINUS; fallback to 0xBD
                try:
                    vk_code = getattr(win32con, 'VK_OEM_MINUS')
                except Exception:
                    vk_code = 0xBD
            elif key == '=':
                # Some pywin32 builds may not expose VK_OEM_PLUS; fallback to 0xBB
                try:
                    vk_code = getattr(win32con, 'VK_OEM_PLUS')
                except Exception:
                    vk_code = 0xBB
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
    
    def _register_system_hotkey(self, hotkey_id: str, sequence: str, add_norepeat: bool = False) -> bool:
        """
        Register a system-wide hotkey using win32 API on the dedicated message thread.
        """
        try:
            # Ensure message loop is running first
            if not self._message_thread_running:
                logger.debug("Starting message loop thread")
                self._start_message_loop()

            # Parse sequence to modifiers/vk
            modifiers, vk_code = self._parse_hotkey_sequence(sequence)
            if add_norepeat and modifiers != 0:
                try:
                    modifiers |= win32con.MOD_NOREPEAT
                except Exception:
                    pass
            logger.debug(f"Parsed sequence '{sequence}' -> modifiers: {modifiers}, vk_code: {vk_code}")
            if vk_code == 0:
                logger.error(f"Failed to parse hotkey sequence: {sequence}")
                return False

            # Prepare sync command
            evt = threading.Event()
            result: Dict[str, Any] = {"ok": False, "error": None}
            with self._hotkey_lock:
                win32_hotkey_id = self._hotkey_id_counter
                self._hotkey_id_counter += 1
            cmd_id = id(evt)
            with self._cmd_lock:
                self._pending_cmds[cmd_id] = ("register", hotkey_id, win32_hotkey_id, modifiers, vk_code, evt, result)

            # Post to message thread
            try:
                mtid = int(self._message_thread_id or 0)
                win32api.PostThreadMessage(mtid, self._WM_APP_REGISTER, cmd_id, 0)
            except Exception as e:
                with self._cmd_lock:
                    self._pending_cmds.pop(cmd_id, None)
                logger.error(f"Failed to post register message to hotkey thread: {e}")
                return False

            # Wait for result
            evt.wait(timeout=2.0)
            ok = bool(result.get("ok"))
            if ok:
                with self._hotkey_lock:
                    self._system_hotkeys[hotkey_id] = win32_hotkey_id
                logger.debug(f"Registered system hotkey {hotkey_id} with sequence {sequence}")
            else:
                if result.get("error"):
                    logger.error(f"Exception while registering system hotkey {hotkey_id}: {result['error']}")
            return ok
        except Exception as e:
            logger.error(f"Exception while registering system hotkey {hotkey_id}: {str(e)}")
            return False

    def _unregister_system_hotkey(self, hotkey_id: str) -> bool:
        """Unregister a system hotkey on the dedicated message thread."""
        try:
            with self._hotkey_lock:
                if hotkey_id not in self._system_hotkeys:
                    logger.warning(f"System hotkey {hotkey_id} not found for unregistration")
                    return False
                win32_hotkey_id = self._system_hotkeys[hotkey_id]

            evt = threading.Event()
            result: Dict[str, Any] = {"ok": False, "error": None}

            # Prefer SPSC queue
            if self._cmd_queue is not None:
                pushed = self._cmd_queue.try_push(("unregister", hotkey_id, win32_hotkey_id, evt, result))
                if not pushed:
                    logger.debug("Hotkey unregister queue full; falling back to WM_APP path")
                else:
                    evt.wait(timeout=2.0)
                    ok = bool(result.get("ok"))
                    # Remove mapping regardless to avoid staleness
                    with self._hotkey_lock:
                        self._system_hotkeys.pop(hotkey_id, None)
                    if ok:
                        logger.debug(f"Unregistered system hotkey {hotkey_id}")
                    else:
                        if result.get("error"):
                            logger.error(f"Exception while unregistering system hotkey {hotkey_id}: {result['error']}")
                    return ok

            # Legacy WM_APP path
            cmd_id = id(evt)
            with self._cmd_lock:
                self._pending_cmds[cmd_id] = ("unregister", hotkey_id, win32_hotkey_id, 0, 0, evt, result)
            try:
                if self._message_thread_id:
                    win32api.PostThreadMessage(int(self._message_thread_id), self._WM_APP_UNREGISTER, cmd_id, 0)
                else:
                    win32api.PostThreadMessage(int(win32api.GetCurrentThreadId()), self._WM_APP_UNREGISTER, cmd_id, 0)
            except Exception as e:
                logger.error(f"Failed to post unregister command: {e}")
                return False
            evt.wait(timeout=2.0)
            ok = bool(result.get("ok"))
            # Remove mapping regardless to avoid staleness
            with self._hotkey_lock:
                self._system_hotkeys.pop(hotkey_id, None)
            if ok:
                logger.debug(f"Unregistered system hotkey {hotkey_id}")
            else:
                if result.get("error"):
                    logger.error(f"Exception while unregistering system hotkey {hotkey_id}: {result['error']}")
            return ok

        except Exception as e:
            logger.error(f"Exception while unregistering system hotkey {hotkey_id}: {str(e)}")
            return False

    def _start_message_loop(self) -> None:
        """Start the hotkey message loop via ThreadManager (single long-running task)."""
        logger.debug("_start_message_loop called")
        if self._message_thread_running:
            logger.debug("Message loop already running")
            return
        try:
            self._message_thread_running = True
            def _runner():
                try:
                    self._message_loop()
                finally:
                    self._message_thread_running = False
            if self._tm is not None:
                # Submit long-lived task to IO pool
                def _wrap():
                    _runner()
                try:
                    self._tm.submit_task(ThreadPoolType.IO, _wrap, task_id="HotkeysMessageLoop")
                    logger.debug("Started hotkey message loop via ThreadManager IO pool")
                except Exception as e:
                    logger.error(f"ThreadManager submission failed, falling back to raw thread: {e}")
                    t = threading.Thread(target=_runner, name="HotkeysMessageLoop", daemon=True)
                    t.start()
                    self._message_thread = t
            else:
                t = threading.Thread(target=_runner, name="HotkeysMessageLoop", daemon=True)
                t.start()
                self._message_thread = t
        except Exception as e:
            self._message_thread_running = False
            logger.error(f"Failed to start message loop: {e}")

    def _stop_message_loop(self) -> None:
        """Signal the message loop to stop and post WM_QUIT."""
        try:
            if not self._message_thread_running:
                return
            self._message_thread_running = False
            try:
                if self._message_thread_id:
                    win32api.PostThreadMessage(int(self._message_thread_id), win32con.WM_QUIT, 0, 0)
            except Exception:
                pass
        except Exception:
            pass

    def _message_loop(self) -> None:
        """Message loop for processing hotkey events and marshalled commands."""
        logger.debug("Hotkey message loop started")
        try:
            try:
                self._message_thread_id = int(win32api.GetCurrentThreadId())
                logger.debug(f"Hotkey message loop thread id={self._message_thread_id}")
            except Exception:
                self._message_thread_id = None

            # Helper to drain SPSC commands (P1)
            def _drain_commands():
                try:
                    while self._cmd_queue is not None:
                        ok, cmd = self._cmd_queue.try_pop()
                        if not ok:
                            break
                        if not cmd:
                            continue
                        ctype = cmd[0]
                        if ctype == "register":
                            _tag, hotkey_id, win32_hotkey_id, modifiers, vk_code, evt, result = cmd
                            try:
                                prev_id = None
                                with self._hotkey_lock:
                                    prev_id = self._system_hotkeys.get(hotkey_id)
                                if prev_id is not None:
                                    try:
                                        win32gui.UnregisterHotKey(self._hwnd or None, int(prev_id))
                                    except Exception:
                                        pass
                                if not self._hwnd:
                                    try:
                                        hinst = win32api.GetModuleHandle(None)
                                        wc = win32gui.WNDCLASS()
                                        wc.hInstance = hinst
                                        wc.lpszClassName = "SPQ_HotkeyMsgWnd"
                                        wc.lpfnWndProc = win32gui.DefWindowProc
                                        try:
                                            win32gui.RegisterClass(wc)
                                        except Exception:
                                            pass
                                        parent = getattr(win32con, 'HWND_MESSAGE', 0)
                                        self._hwnd = win32gui.CreateWindowEx(0, wc.lpszClassName, "SPQ_HotkeyMsgWnd", 0, 0, 0, 0, 0, parent, 0, hinst, None)
                                    except Exception as wh_ex:
                                        logger.error(f"Failed to create message-only window for hotkeys: {wh_ex}")
                                        self._hwnd = None
                                win32gui.RegisterHotKey(self._hwnd or None, int(win32_hotkey_id), int(modifiers), int(vk_code))
                                result["ok"] = True
                            except Exception as rex:
                                result["ok"] = False
                                try:
                                    lasterr = win32api.GetLastError()
                                except Exception:
                                    lasterr = None
                                result["error"] = f"{repr(rex)} last_error={lasterr}"
                            finally:
                                try:
                                    evt.set()
                                except Exception:
                                    pass
                        elif ctype == "unregister":
                            _tag, hotkey_id, win32_hotkey_id, evt, result = cmd
                            try:
                                win32gui.UnregisterHotKey(self._hwnd or None, int(win32_hotkey_id))
                                result["ok"] = True
                            except Exception as uex:
                                result["ok"] = False
                                result["error"] = str(uex)
                            finally:
                                try:
                                    evt.set()
                                except Exception:
                                    pass
                        elif ctype == "clear":
                            try:
                                # Unregister all system hotkeys
                                ids = []
                                with self._hotkey_lock:
                                    ids = list(self._system_hotkeys.values())
                                    self._system_hotkeys.clear()
                                for hid in ids:
                                    try:
                                        win32gui.UnregisterHotKey(self._hwnd or None, int(hid))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        elif ctype == "shutdown":
                            # Post WM_QUIT to end GetMessage loop
                            try:
                                if self._message_thread_id:
                                    win32api.PostThreadMessage(int(self._message_thread_id), win32con.WM_QUIT, 0, 0)
                            except Exception:
                                pass
                            return
                except Exception:
                    pass

            while True:
                # block until a message arrives
                res = win32gui.GetMessage(None, 0, 0)
                if not self._message_thread_running:
                    break
                if not res:
                    continue
                # Unpack message in a version-tolerant way
                msg_id = None
                wparam = None
                try:
                    if isinstance(res, tuple):
                        # Common pywin32 shapes:
                        #  - (hwnd, message, wParam, lParam, time, pt, ...)
                        #  - (message, wParam, lParam)
                        if len(res) >= 4:
                            msg_id = int(res[1])
                            wparam = int(res[2])
                        elif len(res) == 3:
                            msg_id = int(res[0])
                            wparam = int(res[1])
                        else:
                            logger.debug(f"GetMessage returned tuple of unexpected length: {len(res)}; skipping")
                            continue
                    else:
                        # Fallback for MSG-like objects
                        try:
                            msg_id = int(getattr(res, 'message', 0))
                            wparam = int(getattr(res, 'wParam', 0))
                        except Exception:
                            # As a last-ditch attempt, try sequence access
                            msg_id = int(res[0])  # type: ignore[index]
                            wparam = int(res[1])  # type: ignore[index]
                except Exception as up_ex:
                    logger.error(f"Failed to unpack message from GetMessage: {up_ex}")
                    continue
                if not msg_id:
                    continue

                # First drain any pending lock-free commands (P1)
                _drain_commands()

                # Handle marshalled registration (legacy path)
                if msg_id == self._WM_APP_REGISTER:
                    cmd_id = int(wparam)
                    with self._cmd_lock:
                        cmd = self._pending_cmds.pop(cmd_id, None)
                    if cmd:
                        _, hotkey_id, win32_hotkey_id, modifiers, vk_code, evt, result = cmd
                        try:
                            # If a previous mapping existed, unregister it first
                            prev_id = None
                            with self._hotkey_lock:
                                prev_id = self._system_hotkeys.get(hotkey_id)
                            if prev_id is not None:
                                try:
                                    win32gui.UnregisterHotKey(self._hwnd or None, int(prev_id))
                                except Exception:
                                    pass
                            # Ensure we have a message-only window to bind to
                            if not self._hwnd:
                                try:
                                    hinst = win32api.GetModuleHandle(None)
                                    wc = win32gui.WNDCLASS()
                                    wc.hInstance = hinst
                                    wc.lpszClassName = "SPQ_HotkeyMsgWnd"
                                    wc.lpfnWndProc = win32gui.DefWindowProc
                                    try:
                                        win32gui.RegisterClass(wc)
                                    except Exception:
                                        pass
                                    parent = getattr(win32con, 'HWND_MESSAGE', 0)
                                    self._hwnd = win32gui.CreateWindowEx(
                                        0,
                                        wc.lpszClassName,
                                        "SPQ_HotkeyMsgWnd",
                                        0,
                                        0, 0, 0, 0,
                                        parent,
                                        0,
                                        hinst,
                                        None,
                                    )
                                except Exception as wh_ex:
                                    logger.error(f"Failed to create message-only window for hotkeys: {wh_ex}")
                                    self._hwnd = None
                            win32gui.RegisterHotKey(self._hwnd or None, int(win32_hotkey_id), int(modifiers), int(vk_code))
                            result["ok"] = True
                        except Exception as rex:
                            result["ok"] = False
                            try:
                                lasterr = win32api.GetLastError()
                            except Exception:
                                lasterr = None
                            result["error"] = f"{repr(rex)} last_error={lasterr}"
                        finally:
                            try:
                                evt.set()
                            except Exception:
                                pass
                    continue

                # Handle marshalled unregistration (legacy path)
                if msg_id == self._WM_APP_UNREGISTER:
                    cmd_id = int(wparam)
                    with self._cmd_lock:
                        cmd = self._pending_cmds.pop(cmd_id, None)
                    if cmd:
                        _, hotkey_id, win32_hotkey_id, _m, _v, evt, result = cmd
                        try:
                            win32gui.UnregisterHotKey(self._hwnd or None, int(win32_hotkey_id))
                            result["ok"] = True
                        except Exception as uex:
                            result["ok"] = False
                            result["error"] = str(uex)
                        finally:
                            try:
                                evt.set()
                            except Exception:
                                pass
                    continue

                # Handle WM_HOTKEY
                if msg_id == win32con.WM_HOTKEY:
                    win32_hotkey_id = wparam
                    hotkey_id: Optional[str] = None
                    with self._hotkey_lock:
                        for hk_id, win32_id in self._system_hotkeys.items():
                            if win32_id == win32_hotkey_id:
                                hotkey_id = hk_id
                                break
                    if hotkey_id and hotkey_id in self._hotkeys:
                        logger.debug(f"WM_HOTKEY received for {hotkey_id} (win32_id={win32_hotkey_id}); dispatching")
                        self.trigger_hotkey(hotkey_id)

                # Default processing
                try:
                    # Best-effort translation/dispatch; some pywin32 builds expect a MSG struct or 3-tuple
                    try:
                        win32gui.TranslateMessage(res)
                        win32gui.DispatchMessage(res)
                    except Exception:
                        # Fallback: no-op if dispatching isn't applicable for the given shape
                        pass
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Exception in hotkey message loop: {str(e)}")
        finally:
            logger.debug("Hotkey message loop stopped")
            # Destroy message-only window if created
            try:
                if self._hwnd:
                    try:
                        win32gui.DestroyWindow(self._hwnd)
                    except Exception:
                        pass
            finally:
                self._hwnd = None
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
        # Join message thread briefly to avoid pool shutdown races
        try:
            t = self._message_thread
            if t and t.is_alive():
                t.join(timeout=1.0)
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

    # -------------------- Keyboard library backend (single-key suppression) --------------------
    def _is_single_key(self, sequence: str) -> Tuple[bool, Optional[str]]:
        parts = [p.strip() for p in sequence.replace('+', ' + ').split('+') if p.strip()]
        if len(parts) == 1:
            return True, parts[0]
        return False, None

    def _is_safe_single_key(self, token: str) -> bool:
        """Return True if the token is considered safe for per-key suppression."""
        if not token:
            return False
        t = token.lower()
        if t in {"-", "=", "oem_minus", "oem_plus", "minus", "equal", "equals"}:
            return True
        # Allow space as a safe single for keyboard-backend suppression
        if t == "space":
            return True
        # Treat OEM_3 (backtick/tilde) as safe single; we will bind tight aliases only
        if t in {"`", "~", "tilde", "grave", "oem_3"}:
            return True
        # Optional: allow digits as safe singles if enabled in settings
        if self._allow_single_digits and len(t) == 1 and t.isdigit():
            return True
        if t.startswith("f") and t[1:].isdigit():
            try:
                fnum = int(t[1:])
                return 1 <= fnum <= 24
            except Exception:
                return False
        return False

    def _aliases_for_single_key(self, token: str) -> Tuple[str, ...]:
        """Return minimal aliases for a given safe single key token."""
        t = token.lower()
        if t in {"-", "oem_minus", "minus"}:
            return ("-", "oem_minus", "minus")
        if t in {"=", "oem_plus", "equal", "equals"}:
            return ("=", "oem_plus", "equal", "equals")
        if t in {"`", "~", "tilde", "grave", "oem_3"}:
            # Backtick/OEM_3: use only tokens known to the keyboard backend
            return ("`", "oem_3", "grave")
        # Function keys: no aliases needed
        return (t,)

    def _register_keyboard_single(self, hotkey_id: str, key_token: Any) -> bool:
        try:
            if hotkey_id in self._kb_handlers:
                self._unregister_keyboard_single(hotkey_id)
            kb = self._keyboard_lib
            # Normalize tokens list
            tokens: Tuple[str, ...]
            if isinstance(key_token, (list, tuple)):
                tokens = tuple(str(k).strip().lower() for k in key_token if str(k).strip())
            else:
                tokens = (str(key_token).strip().lower(),)
            # Use on_press_key with suppress=True
            # Gate: trigger only on the first physical press, ignore auto-repeat until release
            with self._kb_lock:
                self._kb_press_state[hotkey_id] = False
                self._kb_gate_tokens[hotkey_id] = 0

            def _on_press(e):
                try:
                    # Always suppress to avoid character leaking into foreground apps
                    try:
                        e.suppress = True
                    except Exception:
                        pass
                    # Strict single-key: ignore Ctrl/Shift held. Do not treat transient Alt
                    # (used by foregrounding tricks) as a blocker for dispatch.
                    try:
                        if kb.is_pressed('ctrl') or kb.is_pressed('shift'):
                            self._t_debug_modheld(f"[HOTKEY] Modifier held (ctrl/shift). Ignoring press for {hotkey_id} key='{key_token}'")
                            return
                    except Exception:
                        # If kb.is_pressed fails, fall back to handling as single-key
                        pass
                    
                    # Ignore repeats while key is held down
                    if self._kb_press_state.get(hotkey_id, False):
                        self._t_debug_repeat(f"[HOTKEY] Repeat press ignored for {hotkey_id} key='{key_token}'")
                        return
                    with self._kb_lock:
                        self._kb_press_state[hotkey_id] = True
                    # Bump watchdog token and capture for this press
                    with self._kb_lock:
                        self._kb_gate_tokens[hotkey_id] = self._kb_gate_tokens.get(hotkey_id, 0) + 1
                        token = self._kb_gate_tokens[hotkey_id]
                    
                    logger.debug(f"[HOTKEY] First-press gated for {hotkey_id} key='{key_token}'; suppressing and dispatching")
                    self.trigger_hotkey(hotkey_id)

                    # Watchdog: if release is missed by keyboard backend, poll physical state and reset gate
                    def _watchdog_check(expected_token: int) -> None:
                        try:
                            # If token changed, a newer press/release cycle occurred; abort
                            if self._kb_gate_tokens.get(hotkey_id, -1) != expected_token:
                                return
                            # If physically still pressed, reschedule another check
                            try:
                                still_down = kb.is_pressed(key_token)
                            except Exception:
                                # If detection fails, be conservative: reschedule once more
                                still_down = True
                            if still_down:
                                ThreadManager.single_shot(150, lambda: _watchdog_check(expected_token))
                                return
                            # Not pressed anymore -> ensure gate is reset
                            with self._kb_lock:
                                self._kb_press_state[hotkey_id] = False
                            self._t_debug_watchdog(f"[HOTKEY] Watchdog reset gate for {hotkey_id} key='{key_token}' (no release event)")
                        except Exception:
                            pass

                    # Start watchdog a short time after press to avoid immediate overhead
                    ThreadManager.single_shot(400, lambda: _watchdog_check(token))
                except Exception as ex:
                    logger.error(f"Error in keyboard backend callback for {hotkey_id}: {ex}")
            def _on_release(e):
                try:
                    try:
                        e.suppress = True
                    except Exception:
                        pass
                    # Reset gate on release
                    if self._kb_press_state.get(hotkey_id, False):
                        self._t_debug_release(f"[HOTKEY] Release detected. Gate reset for {hotkey_id} key='{key_token}'")
                    with self._kb_lock:
                        self._kb_press_state[hotkey_id] = False
                    # Invalidate any pending watchdog for the previous press
                    with self._kb_lock:
                        self._kb_gate_tokens[hotkey_id] = self._kb_gate_tokens.get(hotkey_id, 0) + 1
                    # Dispatch optional release callback if registered
                    try:
                        rc = self._hotkeys_release.get(hotkey_id)
                        if rc:
                            cb, rargs = rc
                            try:
                                cb(*rargs)
                            except Exception as cb_ex:
                                logger.error(f"Error in release callback for {hotkey_id}: {cb_ex}")
                    except Exception:
                        pass
                except Exception:
                    pass

            handler_list = []
            for tok in tokens:
                try:
                    h_press = kb.on_press_key(tok, _on_press, suppress=True)
                    h_release = kb.on_release_key(tok, _on_release, suppress=True)
                    handler_list.append(h_press)
                    handler_list.append(h_release)
                except Exception as ex:
                    logger.debug(f"Keyboard backend could not bind token '{tok}' for {hotkey_id}: {ex}")
                    # continue trying other aliases
                    continue
            if not handler_list:
                logger.error(f"Keyboard backend registration failed for {hotkey_id}: no valid tokens from {tokens}")
                return False
            with self._kb_lock:
                self._kb_handlers[hotkey_id] = tuple(handler_list)
            return True
        except Exception as e:
            logger.error(f"Keyboard backend registration error for {hotkey_id} '{key_token}': {e}")
            return False

    def _unregister_keyboard_single(self, hotkey_id: str) -> None:
        kb = self._keyboard_lib
        with self._kb_lock:
            h = self._kb_handlers.pop(hotkey_id, None)
        # Clear pressed-state gate
        try:
            with self._kb_lock:
                if hotkey_id in self._kb_press_state:
                    del self._kb_press_state[hotkey_id]
                if hotkey_id in self._kb_gate_tokens:
                    del self._kb_gate_tokens[hotkey_id]
        except Exception:
            pass
        if h is not None:
            try:
                # Support tuple of (press, release) or single handler
                if isinstance(h, (tuple, list)):
                    for hh in h:
                        try:
                            kb.unhook(hh)
                        except Exception:
                            pass
                else:
                    kb.unhook(h)
            except Exception:
                pass

    def _normalize_keyboard_sequence(self, sequence: str) -> str:
        """Normalize a human sequence for the keyboard library (e.g., map '`' to 'grave')."""
        try:
            parts = [p.strip().lower() for p in sequence.split('+') if p.strip()]
            def norm_key(k: str) -> str:
                if k in {'`', 'oem_3', 'tilde', 'grave'}:
                    return 'grave'
                if k == 'esc':
                    return 'escape'
                return k
            return '+'.join(norm_key(p) for p in parts)
        except Exception:
            return sequence

    def _register_keyboard_combo(self, hotkey_id: str, sequence: str) -> bool:
        """
        Register a combo hotkey using the keyboard library as a fallback when
        system RegisterHotKey fails (e.g., for 'alt+`' on some layouts).
        """
        try:
            if not self._kb_available:
                return False
            kb = self._keyboard_lib
            # Unregister any existing combo first
            if hotkey_id in self._kb_combo_handlers:
                self._unregister_keyboard_combo(hotkey_id)
            seq = self._normalize_keyboard_sequence(sequence)
            handle = kb.add_hotkey(seq, lambda: self.trigger_hotkey(hotkey_id), suppress=True)
            with self._kb_lock:
                self._kb_combo_handlers[hotkey_id] = handle
            return True
        except Exception as ex:
            logger.debug(f"Keyboard combo fallback failed for {hotkey_id} '{sequence}': {ex}")
            return False

    def _unregister_keyboard_combo(self, hotkey_id: str) -> None:
        try:
            if hotkey_id not in self._kb_combo_handlers:
                return
            kb = self._keyboard_lib
            handle = None
            with self._kb_lock:
                handle = self._kb_combo_handlers.pop(hotkey_id, None)
            if handle is not None:
                try:
                    kb.remove_hotkey(handle)
                except Exception:
                    try:
                        kb.unhook(handle)
                    except Exception:
                        pass
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