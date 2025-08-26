from __future__ import annotations

from typing import Optional


class _FocusState:
    """
    Thread-safe overlay focus state accessible across the app.
    - UI thread should set focus via set_overlay_focused.
    - Background threads can read via is_overlay_focused.
    """

    def __init__(self) -> None:
        # Lock-free boolean state. Writes occur on UI thread via callers.
        # Reads may occur on any thread. Under CPython, simple bool writes/reads
        # are atomic under the GIL and sufficient for this flag.
        self._overlay_focused: bool = False

    def set_overlay_focused(self, focused: bool) -> None:
        # Caller policy: invoke on UI thread. No locks.
        self._overlay_focused = bool(focused)

    def is_overlay_focused(self) -> bool:
        # Readable from any thread.
        return self._overlay_focused


_instance: Optional[_FocusState] = _FocusState()


def get_focus_state() -> _FocusState:
    # Eager, lock-free singleton accessor
    assert _instance is not None
    return _instance
