"""Per-app audio session volume control using Windows Core Audio (PyCAW).

- Provides helpers to adjust volume for a specific process (PID) or window (HWND).
- Strictly local: does not change global system volume.
- Graceful failure when PyCAW is not available; callers should handle False.
"""
from __future__ import annotations

from typing import Optional

from core.logging import get_logger
from utils.win.winmsg import WIN_AVAILABLE

_logger = get_logger("AUDIO_SESSION")

PYCAW_AVAILABLE = False
if WIN_AVAILABLE:
    try:
        # Lazy import guards; module remains importable without PyCAW at runtime
        from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume  # type: ignore
        from comtypes import CLSCTX_ALL  # type: ignore
        import win32process  # type: ignore
        try:
            import psutil  # type: ignore
            _PSUTIL_AVAILABLE = True
        except Exception:
            _PSUTIL_AVAILABLE = False
        PYCAW_AVAILABLE = True
    except Exception as e:  # pragma: no cover - optional dependency
        _logger.debug(f"PyCAW not available: {e}")


def is_available() -> bool:
    """Return True if per-app session control is available on this platform."""
    return bool(WIN_AVAILABLE and PYCAW_AVAILABLE)


def _find_session_simple_volume_for_pid(pid: int):
    """Return an ISimpleAudioVolume for the first session matching pid, or None."""
    if not is_available():
        return None
    try:
        sessions = AudioUtilities.GetAllSessions()
        for s in sessions:
            try:
                proc = getattr(s, "Process", None)
                if proc is not None and getattr(proc, "pid", None) == pid:
                    vol = s._ctl.QueryInterface(ISimpleAudioVolume)  # noqa: SLF001
                    return vol
            except Exception:
                continue
    except Exception as e:
        _logger.debug(f"GetAllSessions failed: {e}")
    return None


def _find_session_simple_volume_for_pid_family(pid: int):
    """Return ISimpleAudioVolume for pid or any of its child processes.

    This helps for apps (e.g., browsers) where the audio session lives in a
    sandboxed/renderer child process rather than the top-level window process.
    """
    if not is_available():
        return None
    # First, try the direct PID fast path
    vol = _find_session_simple_volume_for_pid(pid)
    if vol is not None:
        return vol
    # If psutil is available, search child processes recursively
    if WIN_AVAILABLE:
        try:
            if not globals().get("_PSUTIL_AVAILABLE", False):
                return None
            try:
                parent = psutil.Process(pid)  # type: ignore[name-defined]
            except Exception:
                return None
            # Snapshot sessions once
            try:
                sessions = AudioUtilities.GetAllSessions()
            except Exception as e:
                _logger.debug(f"GetAllSessions failed: {e}")
                return None
            # Build candidate PIDs: all children (recursive)
            try:
                children = parent.children(recursive=True)
            except Exception:
                children = []
            candidate_pids = {c.pid for c in children if getattr(c, "pid", None)}
            if not candidate_pids:
                return None
            for s in sessions:
                try:
                    proc = getattr(s, "Process", None)
                    spid = getattr(proc, "pid", None) if proc is not None else None
                    if spid in candidate_pids:
                        return s._ctl.QueryInterface(ISimpleAudioVolume)  # noqa: SLF001
                except Exception:
                    continue
        except Exception:
            # Silent fallback; callers will log
            return None
    return None


def adjust_session_volume_by_pid(pid: int, delta: float) -> bool:
    """Adjust session volume for a process by delta in [−1.0, +1.0].

    Returns True on success. Volume is clamped to [0.0, 1.0].
    """
    if not is_available():
        _logger.debug("Per-app volume unavailable (PyCAW missing or platform unsupported)")
        return False
    try:
        vol = _find_session_simple_volume_for_pid(pid) or _find_session_simple_volume_for_pid_family(pid)
        if vol is None:
            _logger.debug(f"No audio session found for pid={pid}")
            return False
        current = float(vol.GetMasterVolume())  # 0.0..1.0
        new_val = max(0.0, min(1.0, current + float(delta)))
        if new_val == current:
            return True
        vol.SetMasterVolume(new_val, None)
        try:
            _logger.debug(f"Adjusted session volume pid={pid}: {current:.2f} -> {new_val:.2f}")
        except Exception:
            pass
        return True
    except Exception as e:
        _logger.debug(f"adjust_session_volume_by_pid failed for pid={pid}: {e}")
        return False


def adjust_session_volume_for_hwnd(hwnd: int, delta: float) -> bool:
    """Adjust session volume for the process that owns hwnd by delta.

    Returns True on success; False if session not found or unavailable.
    """
    if not is_available():
        _logger.debug("Per-app volume unavailable; skipping")
        return False
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return adjust_session_volume_by_pid(pid, delta)
    except Exception as e:
        _logger.debug(f"adjust_session_volume_for_hwnd failed hwnd={hwnd}: {e}")
        return False


def get_session_volume_by_pid(pid: int) -> Optional[float]:
    """Return the current session volume for a process in [0.0, 1.0].

    Returns None if per-app session control is unavailable or the session
    for the given pid cannot be found.
    """
    if not is_available():
        return None
    try:
        vol = _find_session_simple_volume_for_pid(pid)
        if vol is None:
            vol = _find_session_simple_volume_for_pid_family(pid)
        if vol is None:
            return None
        level = float(vol.GetMasterVolume())
        # Clamp defensively in case of quirky providers
        if level < 0.0:
            level = 0.0
        elif level > 1.0:
            level = 1.0
        return level
    except Exception as e:  # pragma: no cover - hardware/driver dependent
        _logger.debug(f"get_session_volume_by_pid failed for pid={pid}: {e}")
        return None


def get_session_volume_for_hwnd(hwnd: int) -> Optional[float]:
    """Return the current session volume for the process owning hwnd.

    Returns a float in [0.0, 1.0] when available, otherwise None.
    """
    if not is_available():
        return None
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return get_session_volume_by_pid(pid)
    except Exception as e:
        _logger.debug(f"get_session_volume_for_hwnd failed hwnd={hwnd}: {e}")
        return None
