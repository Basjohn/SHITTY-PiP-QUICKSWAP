from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QSize
from core.threading.manager import ThreadManager

from core.logging import get_logger
from core.settings.settings_manager import SettingsManager
from core.graphics.overlay_manager import OverlayManager
from core.hotkeys.manager import HotkeyManager
import win32gui

from utils.window_validation import is_valid_window, get_window_rect
from utils.window.monitors import find_monitor_for_window
from core.switching.mru_manager import get_mru_manager
from core.switching.selection import compute_next_selection
from core.switching.autoswitch_controller import get_autoswitch_controller


class QuickSwitchController(QObject):
    """
    Centralized QuickSwitch controller.

    - Loads hotkey from SettingsManager key 'hotkeys.opacity_quickswitch'
    - Registers/unregisters global hotkey via HotkeyManager (system/global)
    - Exposes quickswitch() to swap the active overlay's source to the current foreground window
    - Live updating via update_hotkeys()
    - Explicit logging (prefix QUICKSWITCH) and strict no-fallback behavior
    """

    _instance: Optional["QuickSwitchController"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        super().__init__()
        self._logger = get_logger("QUICKSWITCH")
        self._settings = SettingsManager()
        self._overlay_manager = OverlayManager()
        self._hotkey_manager = HotkeyManager()
        # Track last selected hwnd per active overlay to enable cyclic navigation
        self._cycle_last_by_overlay: dict[str, int] = {}
        # Re-entrancy gate to prevent concurrent/overlapping quickswitch operations
        self._invoke_lock = threading.Lock()

        self._hotkey: Optional[str] = None
        self._hotkey_id: str = "quickswitch"

        self._load_hotkey()
        self._register_hotkey()

        self._initialized = True
        self._logger.debug("Initialized QuickSwitchController")

    def _load_hotkey(self) -> None:
        try:
            key = self._settings.get("hotkeys.opacity_quickswitch", "`")
            if not key or not isinstance(key, str):
                key = "`"
                self._settings.set("hotkeys.opacity_quickswitch", key)
            self._hotkey = key
            self._logger.debug(f"Loaded quickswitch hotkey: '{self._hotkey}'")
        except Exception as e:
            self._logger.error(f"Failed to load quickswitch hotkey: {e}", exc_info=True)
            self._hotkey = "`"
            self._settings.set("hotkeys.opacity_quickswitch", "`")

    def _unregister_hotkey(self) -> None:
        try:
            # Unregister via HotkeyManager
            if self._hotkey_manager.is_hotkey_registered(self._hotkey_id):
                ok = self._hotkey_manager.unregister_hotkey(self._hotkey_id)
                if not ok:
                    self._logger.error("HotkeyManager failed to unregister 'quickswitch'")
        except Exception as e:
            self._logger.error(f"Error during quickswitch hotkey unregister: {e}", exc_info=True)

    def _register_hotkey(self) -> None:
        try:
            self._unregister_hotkey()
            if not self._hotkey:
                self._logger.error("Quickswitch hotkey is empty; not registering")
                return
            self._logger.debug(f"Registering quickswitch hotkey '{self._hotkey}' via HotkeyManager")
            ok = self._hotkey_manager.register_hotkey(
                self._hotkey_id,
                self.quickswitch,
                sequence=self._hotkey,
                suppress=True,
                global_hotkey=True,
            )
            if not ok:
                self._logger.error(f"Failed to register quickswitch hotkey '{self._hotkey}' via HotkeyManager")
        except Exception as e:
            self._logger.error(f"Failed to register quickswitch hotkey '{self._hotkey}': {e}", exc_info=True)

    def update_hotkeys(self) -> None:
        """Reload and re-register the quickswitch hotkey."""
        self._logger.debug("Updating quickswitch hotkey from settings")
        self._load_hotkey()
        self._register_hotkey()

    def quickswitch(self, source: Optional[str] = None) -> None:
        """Swap the active overlay's source to the current foreground window (if valid).

        Args:
            source: Optional trigger source string (e.g., 'overlay.double_click', 'hotkey').
        """
        try:
            # Re-entrancy gating: drop if an operation is already in progress
            if not self._invoke_lock.acquire(blocking=False):
                try:
                    from core.application.core import get_app_core
                    get_app_core().events.publish(
                        "switch.reentry_suppressed",
                        {"reason": "in_progress", "source": source or "quickswitch"},
                        source="QuickSwitchController",
                    )
                except Exception as e:
                    self._logger.debug(f"Event publish failed (reentry_suppressed): {e}")
                self._logger.debug("Quickswitch suppressed due to re-entrancy gate")
                return
            # Ensure we always release the gate
            _gate_released = False
            # Get active overlay
            overlay = self._get_active_overlay()
            if overlay is None:
                self._logger.error("No active overlay; QuickSwitch aborted")
                return
            # Overlay lock gating: if overlays are locked, publish suppression and abort
            try:
                is_locked = self._overlay_manager.is_overlay_locked()
                self._logger.debug(f"Overlay lock check: is_locked={is_locked}")
                if is_locked:
                    try:
                        oid = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                    except Exception:
                        oid = None
                    try:
                        from core.application.core import get_app_core
                        get_app_core().events.publish(
                            "switch.lock_suppressed",
                            {"overlay_id": oid, "source": source or "quickswitch"},
                            source="QuickSwitchController",
                        )
                    except Exception as e:
                        self._logger.debug(f"Event publish failed (lock_suppressed): {e}")
                    self._logger.debug("Quickswitch suppressed due to overlay lock")
                    return
            except Exception:
                # Never fail due to lock gating check
                pass
            try:
                active_id = getattr(self._overlay_manager, "_active_overlay_id", None)
                self._logger.debug(f"Active overlay: id={active_id}, class={overlay.__class__.__name__}")
            except Exception:
                pass

            # Capture current foreground; do not record now to avoid selecting it
            try:
                cur_hwnd = win32gui.GetForegroundWindow()
                try:
                    cur_title = win32gui.GetWindowText(cur_hwnd)
                except Exception:
                    cur_title = ""
                self._logger.debug(f"Foreground hwnd={cur_hwnd} title='{cur_title}'")
            except Exception as e:
                self._logger.debug(f"Foreground read failed: {e}")
                cur_hwnd = None

            # Ensure overlay supports safe swap handler
            if not hasattr(overlay, "_handle_swap_window"):
                self._logger.error("Active overlay does not support window swapping (_handle_swap_window missing)")
                return

            # Attempt MRU candidates in order (most-recent first)
            mru = get_mru_manager()
            candidates = mru.get_recent(limit=7)
            
            # If MRU is empty or has insufficient candidates, seed with z-order windows
            if len(candidates) < 2:
                self._logger.debug(f"MRU has {len(candidates)} candidates, seeding from z-order")
                seeded_count = self._seed_mru_from_zorder(cur_hwnd)
                self._logger.debug(f"Seeded {seeded_count} windows from z-order")
                candidates = mru.get_recent(limit=7)
                
            # Final fallback: minimal working set from current context
            if not candidates:
                try:
                    current_src = getattr(overlay, "_current_source_hwnd", None)
                    if current_src is None:
                        current_src = getattr(overlay, "_src_hwnd", None)
                except Exception:
                    current_src = None
                minimal = []
                if cur_hwnd:
                    minimal.append(cur_hwnd)
                if current_src and current_src != cur_hwnd:
                    minimal.append(current_src)
                candidates = [h for h in minimal if h]
                if not candidates:
                    self._logger.debug("No MRU candidates available; QuickSwitch aborted")
                    return

            # Read current overlay source; we will swap the overlay's source to the chosen target,
            # and then focus the PREVIOUS source to truly "switch" between the two.
            try:
                current_src = getattr(overlay, "_current_source_hwnd", None)
                if current_src is None:
                    current_src = getattr(overlay, "_src_hwnd", None)
            except Exception:
                current_src = None
            pre_swap_src = int(current_src) if current_src else None

            # Build candidate list; ensure both current foreground and current overlay source are present
            ordered: list[int] = []
            seen: set[int] = set()
            if cur_hwnd:
                ordered.append(cur_hwnd)
                seen.add(cur_hwnd)
            if current_src and current_src not in seen:
                ordered.append(current_src)
                seen.add(current_src)
            # Add the rest, preserving order and uniqueness
            for h in candidates:
                if h and h not in seen:
                    ordered.append(h)
                    seen.add(h)
            candidates = ordered

            # Optional: Display Locked Switching — restrict to windows on the same monitor as overlay's content
            try:
                if bool(self._settings.get("features.display_locked_switching", False)) and current_src:
                    # Determine monitor index of current overlay source
                    try:
                        r = get_window_rect(int(current_src))
                        src_mon = find_monitor_for_window(QPoint(r[0], r[1]), QSize(max(1, r[2]-r[0]), max(1, r[3]-r[1]))) if r else None
                    except Exception:
                        src_mon = None
                    if src_mon is not None:
                        def on_same_monitor(h: int) -> bool:
                            try:
                                rr = get_window_rect(int(h))
                                mon = find_monitor_for_window(QPoint(rr[0], rr[1]), QSize(max(1, rr[2]-rr[0]), max(1, rr[3]-rr[1]))) if rr else None
                                return mon == src_mon
                            except Exception:
                                return False
                        candidates = [h for h in candidates if h and on_same_monitor(h)]
            except Exception:
                pass
            # Exclude only the current foreground from focus targets so we can jump to the new overlay source after manual swaps
            filtered = [h for h in candidates if h and h != cur_hwnd]
            self._logger.debug(f"MRU candidates={len(candidates)} filtered={len(filtered)} cur={cur_hwnd} src={current_src}")
            if not filtered:
                # Degenerate path: cur==current_src and MRU too small. As a strict last-resort, seed one from Z-order.
                try:
                    z = self._pick_from_zorder(exclude_a=cur_hwnd, exclude_b=current_src)
                except Exception:
                    z = None
                if z and z not in seen:
                    try:
                        if is_valid_window(z):
                            candidates.append(z)
                            filtered.append(z)
                            self._logger.debug(f"Seeded from Z-order as last resort: hwnd={z}")
                    except Exception:
                        pass
            if not filtered:
                self._logger.debug("MRU has no target other than current foreground; QuickSwitch aborted")
                return

            # Determine if the overlay itself (host/border/dwm host) holds focus after a context menu interaction
            is_overlay_self = False
            forbidden_for_focus: set[int] = set()
            try:
                host = getattr(overlay, "_host", None)
                if host:
                    forbidden_for_focus.add(int(host.winId()))
            except Exception:
                pass
            try:
                border = getattr(overlay, "_border_overlay", None)
                if border:
                    forbidden_for_focus.add(int(border.winId()))
            except Exception:
                pass
            try:
                dwm_host = getattr(overlay, "_dwm_host", None)
                if dwm_host:
                    forbidden_for_focus.add(int(dwm_host.winId()))
            except Exception:
                pass
            did_swap = False
            swap_target: Optional[int] = None
            focus_target: Optional[int] = None

            if cur_hwnd and cur_hwnd in forbidden_for_focus:
                # Overlay is foreground (e.g., double-click on overlay). Select the NEXT candidate via cyclic logic
                # so we don't stick on current_src and we never include the overlay window itself.
                try:
                    active_id = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                except Exception:
                    active_id = None
                # Seed the cycle to advance past the current_src if available
                try:
                    if current_src:
                        key_active = str(active_id) if active_id is not None else "_global"
                        self._cycle_last_by_overlay[key_active] = int(current_src)
                        self._cycle_last_by_overlay["_global"] = int(current_src)
                except Exception:
                    pass
                chosen, display_idx, reason, ordered, start_idx, ref_last, ref_fore = compute_next_selection(
                    candidates=candidates,
                    filtered=filtered,
                    cur_hwnd=cur_hwnd,
                    current_src=current_src,
                    cycle_last_by_overlay=self._cycle_last_by_overlay,
                    overlay_id=active_id,
                    pick_from_zorder=lambda a, b: self._pick_from_zorder(exclude_a=a, exclude_b=b),
                    is_valid=is_valid_window,
                )
                self._logger.debug(
                    f"Overlay-foreground select: key={str(active_id) if active_id is not None else '_global'} reason={reason} ref_last={ref_last} ref_fore={ref_fore} "
                    f"start_idx={start_idx} list={ordered}"
                )
            else:
                # Centralized cyclic selection
                try:
                    active_id = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                except Exception:
                    active_id = None

                chosen, display_idx, reason, ordered, start_idx, ref_last, ref_fore = compute_next_selection(
                    candidates=candidates,
                    filtered=filtered,
                    cur_hwnd=cur_hwnd,
                    current_src=current_src,
                    cycle_last_by_overlay=self._cycle_last_by_overlay,
                    overlay_id=active_id,
                    pick_from_zorder=lambda a, b: self._pick_from_zorder(exclude_a=a, exclude_b=b),
                    is_valid=is_valid_window,
                )
                self._logger.debug(
                    f"Cycle select: key={str(active_id) if active_id is not None else '_global'} reason={reason} ref_last={ref_last} ref_fore={ref_fore} "
                    f"start_idx={start_idx} list={ordered}"
                )
            if not chosen:
                self._logger.debug("All MRU candidates failed; QuickSwitch aborted")
                return
            # Apply overlay swap. If our overlay host/border/DWM host holds the foreground
            # (e.g., after context menu interactions), swap to the chosen target instead.
            # Otherwise, swap to the window we are leaving (current foreground),
            # but never to our own overlay host/border/DWM host windows.
            try:
                if cur_hwnd and cur_hwnd != current_src:
                    forbidden: set[int] = set()
                    try:
                        host = getattr(overlay, "_host", None)
                        if host:
                            forbidden.add(int(host.winId()))
                    except Exception:
                        pass
                    try:
                        border = getattr(overlay, "_border_overlay", None)
                        if border:
                            forbidden.add(int(border.winId()))
                    except Exception:
                        pass
                    try:
                        dwm_host = getattr(overlay, "_dwm_host", None)
                        if dwm_host:
                            forbidden.add(int(dwm_host.winId()))
                    except Exception:
                        pass
                    if cur_hwnd in forbidden:
                        # Overlay holds focus: swap overlay to 'chosen' target (if valid) instead of skipping
                        try:
                            if chosen and chosen not in forbidden and chosen != current_src and is_valid_window(int(chosen)):
                                overlay._handle_swap_window(int(chosen))
                                did_swap = True
                                swap_target = int(chosen)
                                self._logger.debug(
                                    f"Requested source swap to chosen hwnd={chosen} (overlay foreground)"
                                )
                            else:
                                self._logger.debug(
                                    f"Skipping overlay swap: foreground hwnd={cur_hwnd} is overlay window and chosen is same as current_src/forbidden/invalid"
                                )
                        except Exception:
                            self._logger.debug(
                                f"Skipping overlay swap: validation error for chosen hwnd={chosen} while overlay in foreground"
                            )
                    else:
                        # Validate target to avoid swapping to our own application window
                        try:
                            if not is_valid_window(cur_hwnd):
                                self._logger.debug(
                                    f"Skipping overlay swap: foreground hwnd={cur_hwnd} is invalid or belongs to our process"
                                )
                            else:
                                overlay._handle_swap_window(cur_hwnd)
                                did_swap = True
                                swap_target = int(cur_hwnd)
                                self._logger.debug(f"Requested source swap to foreground hwnd={cur_hwnd}")
                        except Exception:
                            # If validation fails unexpectedly, be conservative and skip the swap
                            self._logger.debug(
                                f"Skipping overlay swap: validation error for foreground hwnd={cur_hwnd}"
                            )
            except Exception as e:
                self._logger.error(f"Swap overlay to foreground hwnd={cur_hwnd} failed: {e}")
                # Continue with focus change even if the overlay swap failed

            # Decide focus target: if a swap happened, focus the PRE-SWAP source to "switch".
            if did_swap and pre_swap_src:
                focus_target = int(pre_swap_src)
            else:
                # If no swap, fall back to chosen when valid, else keep pre-swap src
                try:
                    focus_target = int(chosen) if 'chosen' in locals() and chosen else (int(pre_swap_src) if pre_swap_src else None)
                except Exception:
                    focus_target = pre_swap_src

            # MRU: ensure 'cur' is tracked (excluding overlay windows), then put focus target on top for next cycle stability
            try:
                forbidden_mru: set[int] = set()
                try:
                    host = getattr(overlay, "_host", None)
                    if host:
                        forbidden_mru.add(int(host.winId()))
                except Exception:
                    pass
                try:
                    border = getattr(overlay, "_border_overlay", None)
                    if border:
                        forbidden_mru.add(int(border.winId()))
                except Exception:
                    pass
                try:
                    dwm_host = getattr(overlay, "_dwm_host", None)
                    if dwm_host:
                        forbidden_mru.add(int(dwm_host.winId()))
                except Exception:
                    pass
                if cur_hwnd and cur_hwnd not in forbidden_mru:
                    get_mru_manager().record(cur_hwnd)
                if focus_target:
                    get_mru_manager().record(focus_target)
            except Exception:
                pass

            # Defer focus to chosen on the UI thread to avoid foreground lock issues
            # Suppress Autoswitch while QuickSwitch manipulates focus to prevent conflict
            # Suppress autoswitch while we change focus
            try:
                if focus_target:
                    get_autoswitch_controller().suppress_for(900, last_seen_hwnd=focus_target)
            except Exception:
                pass
            self._logger.debug(f"Dispatching UI-thread focus change to hwnd={focus_target} (pre_swap={pre_swap_src}, swapped_to={swap_target}) in 25ms")
            try:
                if focus_target:
                    ThreadManager.single_shot(25, lambda: self._focus_window(focus_target))
            except Exception as fe:
                # If UI dispatch fails, try immediate focus as best-effort
                self._logger.debug(f"UI dispatch failed, focusing immediately: {fe}")
                try:
                    if focus_target:
                        self._focus_window(focus_target)
                except Exception:
                    pass
            return

            self._logger.debug("All MRU candidates failed; QuickSwitch aborted")
        except Exception as e:
            self._logger.error(f"Quickswitch failed: {e}", exc_info=True)
        finally:
            # Always release the re-entrancy lock
            try:
                if hasattr(self, '_invoke_lock'):
                    self._invoke_lock.release()
            except Exception:
                pass

    def _seed_mru_from_zorder(self, start_hwnd: Optional[int]) -> int:
        """Populate MRU by enumerating top-level windows in Z-order and recording valid ones.
        Preserves existing MRU order and only adds missing windows.
        Returns the number of windows added to MRU.
        """
        cap = 8
        targets = []
        try:
            # EnumWindows returns in Z-order (top-most to bottom)
            win32gui.EnumWindows(lambda h, p: p.append(int(h)) or True, targets)
        except Exception:
            targets = []
        added = 0
        if not targets:
            return added
            
        # Get existing MRU to avoid disrupting current order
        mru = get_mru_manager()
        existing_mru = set(mru.get_recent(limit=20))  # Get more to check against
        
        # Collect valid windows that aren't already in MRU
        valid_new_windows = []
        for hwnd in targets:
            if len(valid_new_windows) >= cap:
                break
            try:
                if hwnd not in existing_mru and is_valid_window(hwnd):
                    valid_new_windows.append(hwnd)
            except Exception:
                continue
                
        # Add new windows in reverse z-order (bottom-most first) to preserve
        # the principle that more recently focused windows should be higher in MRU
        for hwnd in reversed(valid_new_windows):
            try:
                mru.record(hwnd)
                added += 1
            except Exception:
                pass
                
        return added

    def _focus_window(self, hwnd: int) -> None:
        """Attempt to bring the given window to the foreground and restore if minimized."""
        try:
            import win32con
            import win32api
            import win32process
            # Restore if minimized
            try:
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            except Exception:
                pass
            # First simple attempt
            try:
                win32gui.SetForegroundWindow(hwnd)
                return
            except Exception:
                pass
            # Try SetWindowPos sequence
            try:
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
                win32gui.SetForegroundWindow(hwnd)
                return
            except Exception:
                pass
            # Attach thread input trick
            try:
                fg = win32gui.GetForegroundWindow()
                if fg:
                    fg_tid, _ = win32process.GetWindowThreadProcessId(fg)
                    tgt_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
                    cur_tid = win32api.GetCurrentThreadId()
                    # Attach our thread to both to allow SetForegroundWindow
                    win32api.AttachThreadInput(cur_tid, fg_tid, True)
                    win32api.AttachThreadInput(cur_tid, tgt_tid, True)
                    try:
                        win32gui.BringWindowToTop(hwnd)
                        win32gui.SetForegroundWindow(hwnd)
                    finally:
                        win32api.AttachThreadInput(cur_tid, fg_tid, False)
                        win32api.AttachThreadInput(cur_tid, tgt_tid, False)
            except Exception:
                pass
            # Alt keystroke trick to satisfy foreground lock
            try:
                VK_MENU = 0x12
                win32api.keybd_event(VK_MENU, 0, 0, 0)
                win32api.keybd_event(VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass
        except Exception:
            # Focus manipulations can fail due to OS foreground lock; ignore
            pass

    def _pick_from_zorder(self, exclude_a: Optional[int], exclude_b: Optional[int]) -> Optional[int]:
        """Return first valid top-level hwnd in Z-order excluding the given two."""
        targets = []
        try:
            win32gui.EnumWindows(lambda h, p: p.append(int(h)) or True, targets)
        except Exception:
            return None
        for h in targets:
            if not h or h == exclude_a or h == exclude_b:
                continue
            try:
                if is_valid_window(h):
                    return h
            except Exception:
                continue
        return None

    def _get_active_overlay(self):
        # Access the active overlay via OverlayManager internals in a controlled way
        # If OverlayManager exposes a public getter, prefer it; otherwise retrieve by internal id
        try:
            # Best-effort: try known private attribute; log if not present
            active_id = getattr(self._overlay_manager, "_active_overlay_id", None)
            if not active_id:
                return None
            overlays = getattr(self._overlay_manager, "_overlays", {})
            return overlays.get(active_id)
        except Exception:
            return None


# Convenience accessor
_def_instance: Optional[QuickSwitchController] = None

def get_quickswitch_controller() -> QuickSwitchController:
    global _def_instance
    if _def_instance is None:
        _def_instance = QuickSwitchController()
    return _def_instance
