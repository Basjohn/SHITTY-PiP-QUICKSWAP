"""
Safe Windows messaging helpers.

Centralized wrappers for Win32 message posting/sending with safety guards
and minimal, robust helpers for key events. This module is Windows-only;
on non-Windows platforms, all functions safely no-op and return False.

Policy:
- Prefer PostMessage (non-blocking) for safety.
- Provide SendMessageTimeout for responsiveness checks when explicitly needed.
- Compose reasonable lParam for key messages (scan code + flags) without
  overreach; we avoid focus stealing.

Public API:
- is_window(hwnd: int) -> bool
- post_message(hwnd: int, msg: int, wparam: int = 0, lparam: int = 0) -> bool
- send_message_timeout(hwnd: int, msg: int, wparam: int = 0, lparam: int = 0,
                       timeout_ms: int = 500, flags: int | None = None) -> bool
- safe_send_message(hwnd: int, msg: int, wparam: int = 0, lparam: int = 0,
                    timeout_ms: int = 500, flags: int | None = None) -> bool
- pack_lparam_key(vk: int, is_keyup: bool = False, repeat: int = 1, extended: bool = False) -> int
- key_press(hwnd: int, vk: int, delay_ms: int = 10) -> bool

Notes:
- For lParam packing, we include: repeat count (bits 0..15), scan code (16..23),
  extended flag (24), context/menu (29, left 0), previous state (30), transition (31).
- Scan code is derived via MapVirtualKey. Extended flag is best-effort based on vk set.
"""
from __future__ import annotations

from typing import Optional

try:
    import win32api
    import win32gui
    import win32con
    WIN_AVAILABLE = True
except Exception:
    WIN_AVAILABLE = False

from core.logging import get_logger

_logger = get_logger("winmsg")

# Common VKs that typically set the extended key flag
_EXTENDED_VK = {
    0x21, 0x22, 0x23, 0x24,             # PageUp, PageDown, End, Home
    win32con.VK_INSERT if WIN_AVAILABLE else 0x2D,
    win32con.VK_DELETE if WIN_AVAILABLE else 0x2E,
    win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
    win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
    win32con.VK_UP if WIN_AVAILABLE else 0x26,
    win32con.VK_DOWN if WIN_AVAILABLE else 0x28,
    win32con.VK_NUMLOCK if WIN_AVAILABLE else 0x90,
    win32con.VK_SCROLL if WIN_AVAILABLE else 0x91,
    win32con.VK_RCONTROL if WIN_AVAILABLE else 0xA3,
    win32con.VK_RMENU if WIN_AVAILABLE else 0xA5,
}


def is_window(hwnd: int) -> bool:
    if not WIN_AVAILABLE:
        return False
    try:
        return bool(win32gui.IsWindow(hwnd))
    except Exception:
        return False


def post_message(hwnd: int, msg: int, wparam: int = 0, lparam: int = 0) -> bool:
    if not WIN_AVAILABLE:
        return False
    try:
        if not is_window(hwnd):
            return False
        win32api.PostMessage(hwnd, msg, wparam, lparam)
        return True
    except Exception as e:
        _logger.error("PostMessage failed: hwnd=%s msg=%s err=%s", hwnd, hex(msg), e)
        return False


def send_message_timeout(
    hwnd: int,
    msg: int,
    wparam: int = 0,
    lparam: int = 0,
    timeout_ms: int = 500,
    flags: Optional[int] = None,
) -> bool:
    if not WIN_AVAILABLE:
        return False
    try:
        if not is_window(hwnd):
            return False
        # Default flags: normal + abort if hung
        f = flags if flags is not None else (win32con.SMTO_NORMAL | win32con.SMTO_ABORTIFHUNG)
        result = win32gui.SendMessageTimeout(hwnd, msg, wparam, lparam, f, int(max(0, timeout_ms)))
        return bool(result and result[0])
    except Exception as e:
        _logger.error("SendMessageTimeout failed: hwnd=%s msg=%s err=%s", hwnd, hex(msg), e)
        return False


def _map_scan_code(vk: int) -> int:
    if not WIN_AVAILABLE:
        return 0
    try:
        # MAPVK_VK_TO_VSC = 0
        sc = win32api.MapVirtualKey(vk, 0)
        return int(sc) & 0xFF
    except Exception:
        return 0


def pack_lparam_key(vk: int, is_keyup: bool = False, repeat: int = 1, extended: bool = False) -> int:
    repeat = max(1, int(repeat)) & 0xFFFF
    sc = _map_scan_code(vk)
    ext = 1 if (extended or vk in _EXTENDED_VK) else 0
    prev = 1 if is_keyup else 0  # keyup implies previous state was down
    trans = 1 if is_keyup else 0
    lparam = (
        (repeat)
        | (sc << 16)
        | (ext << 24)
        | (prev << 30)
        | (trans << 31)
    )
    return lparam & 0xFFFFFFFF


def key_press(hwnd: int, vk: int, delay_ms: int = 10) -> bool:
    """Post a simple key press (down+up) to a window.

    Uses WM_KEYDOWN/WM_KEYUP via PostMessage only.
    """
    if not WIN_AVAILABLE:
        return False
    try:
        if not is_window(hwnd):
            return False
        lp_down = pack_lparam_key(vk, is_keyup=False, repeat=1)
        lp_up = pack_lparam_key(vk, is_keyup=True, repeat=1)
        ok1 = post_message(hwnd, win32con.WM_KEYDOWN, vk, lp_down)
        if not ok1:
            return False
        # Minimal inter-event delay. Callers should externalize robust rate limiting.
        import time as _t
        _t.sleep(max(0, int(delay_ms)) / 1000.0)
        ok2 = post_message(hwnd, win32con.WM_KEYUP, vk, lp_up)
        return ok2
    except Exception as e:
        _logger.error("key_press failed: hwnd=%s vk=%s err=%s", hwnd, hex(vk), e)
        return False


def key_press_with_char(hwnd: int, vk: int, char_code: int | None = None, delay_ms: int = 10) -> bool:
    """Post a printable key press (down + WM_CHAR + up) to a window.

    - Uses WM_KEYDOWN, WM_CHAR, WM_KEYUP via PostMessage only.
    - For space and other printable keys some apps (notably browsers) expect WM_CHAR.
    - char_code defaults to the VK value when within ASCII range.
    """
    if not WIN_AVAILABLE:
        return False
    try:
        if not is_window(hwnd):
            return False
        # Determine character code
        ch = char_code
        if ch is None:
            # If vk maps directly to ASCII range, use it (works for space, digits, letters, basic punctuation)
            ch = vk & 0xFF
        lp_down = pack_lparam_key(vk, is_keyup=False, repeat=1)
        lp_char = pack_lparam_key(vk, is_keyup=False, repeat=1)
        lp_up = pack_lparam_key(vk, is_keyup=True, repeat=1)
        ok1 = post_message(hwnd, win32con.WM_KEYDOWN, vk, lp_down)
        if not ok1:
            return False
        # Small delay before CHAR dispatch
        import time as _t
        _t.sleep(max(0, int(delay_ms)) / 1000.0)
        okc = post_message(hwnd, win32con.WM_CHAR, ch, lp_char)
        if not okc:
            return False
        _t.sleep(max(0, int(delay_ms)) / 1000.0)
        oku = post_message(hwnd, win32con.WM_KEYUP, vk, lp_up)
        return oku
    except Exception as e:
        _logger.error("key_press_with_char failed: hwnd=%s vk=%s err=%s", hwnd, hex(vk), e)
        return False


def is_process_responsive(hwnd: int, timeout_ms: int = 500) -> bool:
    """Check if a window/process is responsive using WM_NULL probe."""
    if not WIN_AVAILABLE:
        return False
    try:
        if not is_window(hwnd):
            return False
        # Use WM_NULL as a safe probe message
        result = send_message_timeout(hwnd, win32con.WM_NULL, 0, 0, timeout_ms)
        return result
    except Exception as e:
        _logger.debug("Responsiveness check failed: hwnd=%s err=%s", hwnd, e)
        return False


def safe_send_message(
    hwnd: int,
    msg: int,
    wparam: int = 0,
    lparam: int = 0,
    timeout_ms: int = 500,
    flags: Optional[int] = None,
) -> bool:
    """Safely send a message with responsiveness check and timeout.

    - Verifies hwnd is valid and responsive using a WM_NULL probe with half the timeout.
    - Uses SendMessageTimeout with SMTO_ABORTIFHUNG to avoid UI hangs.
    """
    if not WIN_AVAILABLE:
        return False
    try:
        if not is_window(hwnd):
            return False
        # Check responsiveness first (use half of provided timeout)
        half_timeout = max(1, int(timeout_ms) // 2)
        if not is_process_responsive(hwnd, half_timeout):
            _logger.debug("Window unresponsive, skipping message: hwnd=%s msg=%s", hwnd, hex(msg))
            return False
        return send_message_timeout(hwnd, msg, wparam, lparam, timeout_ms, flags)
    except Exception as e:
        _logger.error("safe_send_message failed: hwnd=%s msg=%s err=%s", hwnd, hex(msg), e)
        return False


def safe_send_appcommand(hwnd: int, command: int, timeout_ms: int = 1000) -> bool:
    """Safely send WM_APPCOMMAND with responsiveness check and timeout."""
    if not WIN_AVAILABLE:
        return False
    try:
        if not is_window(hwnd):
            return False
        
        # Check responsiveness first
        if not is_process_responsive(hwnd, timeout_ms // 2):
            _logger.debug("Window unresponsive, skipping appcommand: hwnd=%s", hwnd)
            return False
        
        # Send WM_APPCOMMAND with command in high word of lParam
        WM_APPCOMMAND = 0x319
        lparam = command << 16
        return send_message_timeout(hwnd, WM_APPCOMMAND, hwnd, lparam, timeout_ms)
    except Exception as e:
        _logger.error("safe_send_appcommand failed: hwnd=%s cmd=%s err=%s", hwnd, command, e)
        return False


def send_wm_command(hwnd: int, command_id: int, timeout_ms: int = 500) -> bool:
    """Send WM_COMMAND to a window with the given command ID.

    wParam: low-word = command ID; high-word (notification code) = 0.
    lParam: handle to control (0 for menu/accelerator commands).
    """
    if not WIN_AVAILABLE:
        return False
    try:
        if not is_window(hwnd):
            return False
        WM_COMMAND = 0x0111
        return send_message_timeout(hwnd, WM_COMMAND, int(command_id) & 0xFFFF, 0, timeout_ms)
    except Exception as e:
        _logger.error("send_wm_command failed: hwnd=%s id=%s err=%s", hwnd, command_id, e)
        return False
