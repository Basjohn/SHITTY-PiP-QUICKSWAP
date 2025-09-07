from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QSize, QCoreApplication, QThread
from core.threading import ThreadManager

from core.logging import get_logger, throttled, log_dedupe
from core.settings.settings_manager import SettingsManager
from core.graphics.overlay_manager import OverlayManager
import win32gui

from utils.window_validation import is_valid_window, get_window_rect
from utils.window.monitors import find_monitor_for_window
from core.switching.mru_manager import get_mru_manager
from core.switching.selection import compute_next_selection
from core.switching.autoswitch_controller import get_autoswitch_controller
from utils.resource_manager import get_resource_manager, ResourceType

try:
    import keyboard  # type: ignore
except Exception:  # pragma: no cover - import guarded; tests can monkeypatch
    keyboard = None  # will be validated at runtime


class QuickSwitchController(QObject):
    """
    Centralized QuickSwitch controller.

    - Registers/unregisters a global combo via the `keyboard` library directly
      (default: "shift+x"). Honors `hotkeys.opacity_quickswitch` for the combo.
      No fallback or backtick special-casing.
    - Exposes `quickswitch()` to swap the active overlay's source to the current
      foreground window; maintains MRU and dispatch behavior unchanged.
    - Integrates with `ResourceManager` for deterministic cleanup on shutdown.
    - Settings: respects `hotkeys.quickswitch_enabled` (bool) and
      `hotkeys.opacity_quickswitch` (string combo).
    - Lock-free: implements an 800ms cooldown using a monotonic timestamp and a
      simple in-flight flag; UI-thread dispatch via `ThreadManager`.

    """
    _instance: Optional["QuickSwitchController"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        super().__init__()
        self._logger = get_logger("QUICKSWITCH")
        # Throttled/deduped debug emitters to reduce spam from rapid invocations
        try:
            self._t_debug_reentry = throttled(self._logger.debug, "quickswitch:reentry", 250)
            self._t_debug_lock = throttled(self._logger.debug, "quickswitch:lock", 500)
            self._t_debug_fore = throttled(self._logger.debug, "quickswitch:foreground", 750)
            self._t_debug_mru = throttled(self._logger.debug, "quickswitch:mru", 400)
            self._t_debug_seed = throttled(self._logger.debug, "quickswitch:seed", 1000)
            self._t_debug_dispatch = throttled(self._logger.debug, "quickswitch:dispatch", 400)
            self._t_debug_cooldown = throttled(self._logger.debug, "quickswitch:cooldown", 500)
            self._d_debug_fail = log_dedupe(self._logger.debug, "quickswitch:fail", 1500)
        except Exception:
            # Fallbacks if helpers unavailable
            self._t_debug_reentry = self._logger.debug
            self._t_debug_lock = self._logger.debug
            self._t_debug_fore = self._logger.debug
            self._t_debug_mru = self._logger.debug
            self._t_debug_seed = self._logger.debug
            self._t_debug_dispatch = self._logger.debug
            self._t_debug_cooldown = self._logger.debug
            self._d_debug_fail = self._logger.debug
        self._settings = SettingsManager()
        self._overlay_manager = OverlayManager()
        # Track last selected hwnd per active overlay to enable cyclic navigation
        self._cycle_last_by_overlay: dict[str, int] = {}
        # Lock-free reentrancy flag
        self._inflight: bool = False

        # Hotkey registration (keyboard library) state
        self._hotkey_id: str = "quickswitch"
        self._kb_handle = None  # keyboard handler id
        self._kb_resource_id = None  # ResourceManager registration id
        self._rm = get_resource_manager()

        # Default combo; actual combo read from settings key 'hotkeys.opacity_quickswitch'
        self._default_combo: str = "shift+x"

        # Simple cooldown (monotonic timestamp). No locks; best-effort gating.
        self._cooldown_ms: int = 800
        self._cooldown_until: float = 0.0

        self._register_hotkey()

        self._initialized = True
        self._logger.debug("Initialized QuickSwitchController")

    class _KbHotkeyResource:
        """Weakref-able wrapper for a keyboard hotkey handle with cleanup."""
        def __init__(self, handle, sequence: str):
            self.handle = handle
            self.sequence = sequence

        def cleanup(self):
            try:
                if keyboard is not None and self.handle is not None:
                    keyboard.remove_hotkey(self.handle)
            except Exception:
                pass

    def _unregister_hotkey(self) -> None:
        """Unregister the keyboard hotkey via ResourceManager cleanup."""
        try:
            if self._kb_resource_id is not None:
                rid = self._kb_resource_id
                self._kb_resource_id = None
                try:
                    self._rm.unregister(rid)
                except Exception as e:
                    self._logger.debug(f"ResourceManager unregister failed for quickswitch hotkey: {e}")
            # Best-effort direct removal if resource path didn't run
            try:
                if keyboard is not None and self._kb_handle is not None:
                    keyboard.remove_hotkey(self._kb_handle)
            except Exception:
                pass
            finally:
                self._kb_handle = None
        except Exception as e:
            self._logger.error(f"Error during quickswitch hotkey unregister: {e}", exc_info=True)

    def _register_hotkey(self) -> None:
        """Register global combo via keyboard library (no fallback) and RM cleanup.

        Reads combo from SettingsManager key 'hotkeys.opacity_quickswitch',
        defaulting to 'shift+x'.
        """
        try:
            self._unregister_hotkey()
            # Respect feature toggle: only register when enabled
            if not self._is_enabled():
                self._logger.debug("Quickswitch hotkey disabled via settings; not registering")
                return
            if keyboard is None:
                self._logger.error("keyboard library not available; quickswitch hotkey not registered")
                return

            # Determine combo from settings (no fallback registration path)
            try:
                combo = str(self._settings.get("hotkeys.opacity_quickswitch", self._default_combo) or self._default_combo)
            except Exception:
                combo = self._default_combo

            sequence_used = None
            handle = None
            try:
                handle = keyboard.add_hotkey(combo, lambda: self.quickswitch("hotkey"), suppress=False)
                sequence_used = combo
                self._logger.debug(f"Registered quickswitch hotkey via keyboard: '{sequence_used}'")
            except Exception as e1:
                self._logger.error(f"Failed to register quickswitch hotkey via keyboard (combo '{combo}' failed): {e1}")
                return

            # Register resource for deterministic cleanup
            try:
                res = QuickSwitchController._KbHotkeyResource(handle, sequence_used)
                rid = self._rm.register(
                    res,
                    resource_type=ResourceType.CUSTOM,
                    description=f"keyboard hotkey: {self._hotkey_id} [{sequence_used}]",
                    cleanup_handler=lambda r: r.cleanup(),
                    cleanup_priority=10,
                )
                self._kb_handle = handle
                self._kb_resource_id = rid
            except Exception as e:
                # If RM registration fails, keep the hotkey but log it; cleanup will be best-effort
                self._logger.debug(f"ResourceManager register failed for quickswitch hotkey: {e}")
                self._kb_handle = handle
                self._kb_resource_id = None
        except Exception as e:
            self._logger.error(f"Failed to register quickswitch hotkey: {e}", exc_info=True)

    def update_hotkeys(self) -> None:
        """Re-register the quickswitch hotkey (enable/disable responsive)."""
        self._logger.debug("Updating quickswitch hotkey registration")
        self._register_hotkey()

    def _is_enabled(self) -> bool:
        """Return True if quickswitch hotkey is enabled in settings."""
        try:
            return bool(self._settings.get("hotkeys.quickswitch_enabled", False))
        except Exception:
            return False

    def quickswitch(self, source: Optional[str] = None) -> None:
        """Swap the active overlay's source to the current foreground window (if valid).

        Args:
            source: Optional trigger source string (e.g., 'overlay.double_click', 'hotkey').
        """
        try:
            # Cooldown early-drop: apply before dispatching to UI thread
            try:
                now = time.monotonic()
                if now < getattr(self, "_cooldown_until", 0.0):
                    try:
                        from core.application.core import get_app_core
                        get_app_core().events.publish(
                            "switch.cooldown_suppressed",
                            {"remaining_ms": int((self._cooldown_until - now) * 1000), "source": source or "quickswitch"},
                            source="QuickSwitchController",
                        )
                    except Exception as e:
                        self._logger.debug(f"Event publish failed (cooldown_suppressed): {e}")
                    self._t_debug_cooldown("Quickswitch suppressed due to cooldown (early)")
                    return
            except Exception:
                pass

            # If not on UI thread, schedule a single in-flight task to run there
            try:
                app = QCoreApplication.instance()
                on_ui = (app is not None and QThread.currentThread() is app.thread())
            except Exception:
                on_ui = False
            if not on_ui:
                if getattr(self, "_inflight", False):
                    try:
                        from core.application.core import get_app_core
                        get_app_core().events.publish(
                            "switch.reentry_suppressed",
                            {"reason": "in_flight", "source": source or "quickswitch"},
                            source="QuickSwitchController",
                        )
                    except Exception as e:
                        self._logger.debug(f"Event publish failed (reentry_suppressed): {e}")
                    self._t_debug_reentry("Quickswitch suppressed (in-flight)")
                    return
                # mark in-flight and bounce to UI thread
                self._inflight = True
                try:
                    ThreadManager.run_on_ui_thread(self._quickswitch_impl, source)
                except Exception as e:
                    # If dispatch fails, clear and log
                    self._inflight = False
                    self._logger.error(f"Failed to dispatch quickswitch to UI thread: {e}")
                return

            # Already on UI thread: run implementation directly
            self._quickswitch_impl(source)
        except Exception as e:
            self._logger.error(f"Quickswitch failed: {e}", exc_info=True)

    def _quickswitch_impl(self, source: Optional[str]) -> None:
        """UI-thread implementation of quickswitch (lock-free)."""
        try:
            # Ensure in-flight flag is set (may already be set by off-thread caller)
            if not getattr(self, "_inflight", False):
                self._inflight = True

            # Authoritative cooldown gate on UI thread
            try:
                now = time.monotonic()
                if now < self._cooldown_until:
                    try:
                        from core.application.core import get_app_core
                        get_app_core().events.publish(
                            "switch.cooldown_suppressed",
                            {"remaining_ms": int((self._cooldown_until - now) * 1000), "source": source or "quickswitch"},
                            source="QuickSwitchController",
                        )
                    except Exception as e:
                        self._logger.debug(f"Event publish failed (cooldown_suppressed): {e}")
                    self._t_debug_cooldown("Quickswitch suppressed due to cooldown (UI)")
                    return
                self._cooldown_until = now + (self._cooldown_ms / 1000.0)
            except Exception:
                pass

            # Get active overlay
            overlay = self._get_active_overlay()
            if overlay is None:
                self._logger.error("No active overlay; QuickSwitch aborted")
                return
            # Overlay lock gating
            try:
                is_globally_locked = self._overlay_manager.is_overlay_locked()
                is_individually_locked = getattr(overlay, "_is_window_locked", False)
                self._t_debug_lock(
                    f"Overlay lock check: global={is_globally_locked} individual={is_individually_locked}"
                )
                if is_globally_locked or is_individually_locked:
                    try:
                        oid = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                    except Exception:
                        oid = None
                    
                    # Focus the captured window instead of switching when locked
                    try:
                        captured_hwnd = getattr(overlay, '_captured_hwnd', None) or getattr(overlay, '_source_hwnd', None)
                        if captured_hwnd:
                            win32gui.SetForegroundWindow(captured_hwnd)
                            self._logger.info(f"Focused locked overlay's captured window: {captured_hwnd}")
                        else:
                            self._logger.debug("No captured window found to focus on locked overlay")
                    except Exception as focus_err:
                        self._logger.error(f"Failed to focus captured window on locked overlay: {focus_err}")
                    
                    try:
                        from core.application.core import get_app_core
                        lock_type = "global" if is_globally_locked else "individual"
                        get_app_core().events.publish(
                            "switch.lock_suppressed",
                            {"overlay_id": oid, "source": source or "quickswitch", "lock_type": lock_type},
                            source="QuickSwitchController",
                        )
                    except Exception as e:
                        self._logger.debug(f"Event publish failed (lock_suppressed): {e}")
                    self._t_debug_lock("Quickswitch suppressed due to overlay lock - focused captured window instead")
                    return
            except Exception:
                pass
            try:
                active_id = getattr(self._overlay_manager, "_active_overlay_id", None)
                self._logger.debug(f"Active overlay: id={active_id}, class={overlay.__class__.__name__}")
            except Exception:
                pass

            # Capture current foreground
            try:
                cur_hwnd = win32gui.GetForegroundWindow()
                try:
                    cur_title = win32gui.GetWindowText(cur_hwnd)
                except Exception:
                    cur_title = ""
                self._t_debug_fore(f"Foreground hwnd={cur_hwnd} title='{cur_title}'")
            except Exception as e:
                self._t_debug_fore(f"Foreground read failed: {e}")
                cur_hwnd = None

            # Ensure overlay supports safe swap handler
            if not hasattr(overlay, "_handle_swap_window"):
                self._logger.error("Active overlay does not support window swapping (_handle_swap_window missing)")
                return

            # Attempt MRU candidates
            mru = get_mru_manager()
            candidates = mru.get_recent(limit=7)
            if len(candidates) < 2:
                self._t_debug_seed(f"MRU has {len(candidates)} candidates, seeding from z-order")
                seeded_count = self._seed_mru_from_zorder(cur_hwnd)
                self._t_debug_seed(f"Seeded {seeded_count} windows from z-order")
                candidates = mru.get_recent(limit=7)

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
                    self._d_debug_fail("No MRU candidates available; QuickSwitch aborted")
                    return

            # Read current overlay source and build candidate order
            try:
                current_src = getattr(overlay, "_current_source_hwnd", None)
                if current_src is None:
                    current_src = getattr(overlay, "_src_hwnd", None)
            except Exception:
                current_src = None
            pre_swap_src = int(current_src) if current_src else None

            ordered: list[int] = []
            seen: set[int] = set()
            if cur_hwnd:
                ordered.append(cur_hwnd)
                seen.add(cur_hwnd)
            if current_src and current_src not in seen:
                ordered.append(current_src)
                seen.add(current_src)
            for h in candidates:
                if h and h not in seen:
                    ordered.append(h)
                    seen.add(h)
            candidates = ordered

            # Optional display-locked filtering
            # Persist src_mon and on_same_monitor for potential Z-order fallback checks
            src_mon = None
            def on_same_monitor(_: int) -> bool:  # default predicate accepts nothing when feature off
                return True
            try:
                if bool(self._settings.get("features.display_locked_switching", False)) and current_src:
                    try:
                        r = get_window_rect(int(current_src))
                        src_mon = find_monitor_for_window(QPoint(r[0], r[1]), QSize(max(1, r[2]-r[0]), max(1, r[3]-r[1]))) if r else None
                    except Exception:
                        src_mon = None
                    if src_mon is not None:
                        def _on_same_monitor(h: int) -> bool:
                            try:
                                rr = get_window_rect(int(h))
                                mon = find_monitor_for_window(QPoint(rr[0], rr[1]), QSize(max(1, rr[2]-rr[0]), max(1, rr[3]-rr[1]))) if rr else None
                                return mon == src_mon
                            except Exception:
                                return False
                        on_same_monitor = _on_same_monitor  # type: ignore
                        candidates = [h for h in candidates if h and on_same_monitor(h)]
            except Exception:
                pass

            filtered = [h for h in candidates if h and h != cur_hwnd]
            self._t_debug_mru(f"MRU candidates={len(candidates)} filtered={len(filtered)} cur={cur_hwnd} src={current_src}")
            if not filtered:
                try:
                    z = self._pick_from_zorder(exclude_a=cur_hwnd, exclude_b=current_src)
                except Exception:
                    z = None
                if z and z not in seen:
                    try:
                        if is_valid_window(z) and on_same_monitor(z):
                            candidates.append(z)
                            filtered.append(z)
                            self._t_debug_seed(f"Seeded from Z-order as last resort: hwnd={z}")
                    except Exception:
                        pass
            if not filtered:
                self._d_debug_fail("MRU has no target other than current foreground; QuickSwitch aborted")
                return

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
                try:
                    active_id = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                except Exception:
                    active_id = None
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
                self._t_debug_mru(
                    f"Overlay-foreground select: key={str(active_id) if active_id is not None else '_global'} reason={reason} ref_last={ref_last} ref_fore={ref_fore} "
                    f"start_idx={start_idx} list={ordered}"
                )
            else:
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
                self._t_debug_mru(
                    f"Cycle select: key={str(active_id) if active_id is not None else '_global'} reason={reason} ref_last={ref_last} ref_fore={ref_fore} "
                    f"start_idx={start_idx} list={ordered}"
                )
            if not chosen:
                self._d_debug_fail("All MRU candidates failed; QuickSwitch aborted")
                return

            # Always attempt to swap - either to foreground or to chosen candidate
            try:
                swap_hwnd = None
                
                # Determine swap target: prefer foreground if valid, otherwise use chosen
                if cur_hwnd and cur_hwnd != current_src and is_valid_window(cur_hwnd):
                    # Check if foreground is forbidden (overlay window)
                    if cur_hwnd not in forbidden_for_focus:
                        # Check display lock if enabled
                        if on_same_monitor(cur_hwnd):
                            swap_hwnd = cur_hwnd
                            self._logger.debug(f"Swapping to foreground hwnd={cur_hwnd}")
                        else:
                            self._logger.debug(f"Foreground hwnd={cur_hwnd} off-monitor, trying chosen")
                    else:
                        self._logger.debug(f"Foreground hwnd={cur_hwnd} is overlay window, trying chosen")
                
                # If foreground not suitable, try chosen candidate
                if not swap_hwnd and chosen and chosen != current_src:
                    if is_valid_window(int(chosen)) and int(chosen) not in forbidden_for_focus:
                        if on_same_monitor(int(chosen)):
                            swap_hwnd = int(chosen)
                            self._logger.debug(f"Swapping to chosen hwnd={chosen}")
                        else:
                            self._logger.debug(f"Chosen hwnd={chosen} off-monitor")
                
                # Perform the swap
                if swap_hwnd:
                    overlay._handle_swap_window(swap_hwnd)
                    did_swap = True
                    swap_target = swap_hwnd
                    self._logger.debug(f"Requested source swap to hwnd={swap_hwnd}")
                else:
                    self._logger.debug(f"No valid swap target found: cur_hwnd={cur_hwnd}, chosen={chosen}, current_src={current_src}")
                    
            except Exception as e:
                self._logger.error(f"Swap overlay failed: {e}", exc_info=True)

            if did_swap and pre_swap_src:
                focus_target = int(pre_swap_src)
            else:
                try:
                    focus_target = int(chosen) if 'chosen' in locals() and chosen else (int(pre_swap_src) if pre_swap_src else None)
                except Exception:
                    focus_target = pre_swap_src

            try:
                forbidden_mru: set[int] = set(forbidden_for_focus)
                if cur_hwnd and cur_hwnd not in forbidden_mru:
                    get_mru_manager().record(cur_hwnd)
                if focus_target:
                    get_mru_manager().record(focus_target)
            except Exception:
                pass

            try:
                if focus_target:
                    get_autoswitch_controller().suppress_for(900, last_seen_hwnd=focus_target)
            except Exception:
                pass
            self._t_debug_dispatch(f"Dispatching UI-thread focus change to hwnd={focus_target} (pre_swap={pre_swap_src}, swapped_to={swap_target}) in 25ms")
            try:
                if focus_target:
                    ThreadManager.single_shot(25, lambda: self._focus_window(focus_target))
            except Exception as fe:
                self._t_debug_dispatch(f"UI dispatch failed, focusing immediately: {fe}")
                try:
                    if focus_target:
                        self._focus_window(focus_target)
                except Exception:
                    pass
            return
        except Exception as e:
            self._logger.error(f"Quickswitch(UI) failed: {e}", exc_info=True)
        finally:
            try:
                self._inflight = False
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
        """
        Get the currently active overlay from the OverlayManager.
        
        Accesses the active overlay via OverlayManager internals in a controlled way.
        If OverlayManager exposes a public getter, prefer it; otherwise retrieve by internal id.
        
        Returns:
            Optional[Overlay]: The active overlay instance, or None if no overlay is active.
        """
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
