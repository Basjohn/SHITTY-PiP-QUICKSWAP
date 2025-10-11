from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QSize

from core.logging import get_logger
from core.settings import get_settings_manager
from core.graphics.overlay_manager import OverlayManager
from core.threading import ThreadManager

import win32gui

from utils.window_validation import is_valid_window, get_window_rect, get_window_title
from core.switching.mru_manager import get_mru_manager
from core.switching.selection import compute_next_selection
from utils.window.monitors import find_monitor_for_window
from utils.resource_manager import get_resource_manager, ResourceType


class ForegroundAutoswitchController(QObject):
    """
    Foreground-based autoswitch controller (focus change monitoring).

    Monitors foreground window focus changes and automatically swaps overlays
    when the user focuses a window that's currently displayed in an overlay.

    NOTE: This is NOT related to closed-window monitoring (see ClosedWindowSwitchManager).

    - Observes foreground window changes using a Qt timer (polling)
    - Applies strict filtering to candidate windows
    - Debounces rapid changes before triggering a swap
    - Safely delegates swapping to the overlay's _handle_swap_window (UI-thread safe)
    - Live settings via apply_settings(); reads 'features.autoswitch_enabled'

    Explicit logging with 'FOREGROUND_AUTOSWITCH' prefix. No silent fallbacks.
    """

    POLL_INTERVAL_MS = 250
    STABLE_DEBOUNCE_MS = 300

    def __init__(self) -> None:
        super().__init__()
        self._logger = get_logger("FOREGROUND_AUTOSWITCH")
        self._settings = get_settings_manager()
        self._overlay_manager = OverlayManager()
        self._rm = get_resource_manager()
        self._rm_id = None

        self._enabled: bool = False
        self._polling_active: bool = False

        # Debounce/state
        self._last_seen_hwnd: Optional[int] = None
        self._candidate_hwnd: Optional[int] = None
        self._candidate_since: Optional[float] = None
        self._last_applied_src: Optional[int] = None
        # Future: share cycle state with QuickSwitch if needed
        self._cycle_last_by_overlay: dict[str, int] = {}
        # Suppression gate to ignore foreground changes triggered by QuickSwitch
        self._suppress_until_ms: float = 0.0
        self._suppress_last_seen: Optional[int] = None

        # Register settings change handlers for live enable/disable and docking mode gating
        self._settings_key_enable = "features.autoswitch_enabled"
        self._settings_key_docking_mode = "docking.mode"
        self._settings_handler = self._on_setting_changed
        try:
            self._settings.register_change_handler(self._settings_key_enable, self._settings_handler)
            self._settings.register_change_handler(self._settings_key_docking_mode, self._settings_handler)
        except Exception as e:
            self._logger.error(f"Failed to register settings handler for {self._settings_key_enable}: {e}")

        # Register with ResourceManager for deterministic cleanup
        try:
            self._rm_id = self._rm.register(
                self,
                resource_type=ResourceType.CUSTOM,
                description="AutoswitchController",
                cleanup_handler=lambda r: r.shutdown(),
                cleanup_priority=20,
            )
        except Exception as e:
            self._logger.debug(f"ResourceManager register failed for AutoswitchController: {e}")

        self.apply_settings()

        self._logger.debug("Initialized ForegroundAutoswitchController")

    def shutdown(self) -> None:
        """Deterministically stop polling, unregister handlers, and cleanup resources."""
        try:
            # Stop observation and reset state
            self._enabled = False
            self._polling_active = False
            self._last_seen_hwnd = None
            self._candidate_hwnd = None
            self._candidate_since = None
            self._last_applied_src = None
            self._suppress_until_ms = 0.0
            self._suppress_last_seen = None
        except Exception:
            pass
        # Unregister settings handler
        try:
            if hasattr(self, "_settings_handler") and self._settings_handler is not None:
                self._settings.unregister_change_handler(self._settings_key_enable, self._settings_handler)
        except Exception as e:
            self._logger.debug(f"Failed to unregister settings handler: {e}")
        # Unregister from ResourceManager (idempotent)
        try:
            if self._rm_id is not None:
                rid = self._rm_id
                self._rm_id = None
                try:
                    self._rm.unregister(rid)
                except Exception as ue:
                    self._logger.debug(f"ResourceManager unregister failed for AutoswitchController: {ue}")
        except Exception:
            pass

    def _on_setting_changed(self, key: str, value) -> None:
        """React to live changes for autoswitch enable flag, serialized on UI thread."""
        try:
            if key not in (self._settings_key_enable, self._settings_key_docking_mode):
                return
            # Ensure apply_settings runs on UI thread
            ThreadManager.run_on_ui_thread(self.apply_settings)
        except Exception as e:
            self._logger.debug(f"Failed to process settings change for {key}: {e}")

    def _pick_from_zorder(self, exclude_a: Optional[int], exclude_b: Optional[int]) -> Optional[int]:
        """Return a reasonable next window from Z-order excluding up to two handles, honoring validity checks.
        This is a light helper for selection hints; returns None if none found.
        """
        try:
            targets = []
            win32gui.EnumWindows(lambda h, p: p.append(int(h)) or True, targets)
        except Exception:
            targets = []
        if not targets:
            return None
        for h in targets:
            if not h:
                continue
            if exclude_a and h == exclude_a:
                continue
            if exclude_b and h == exclude_b:
                continue
            try:
                if is_valid_window(h):
                    return h
            except Exception:
                continue
        return None

    # Public API: temporarily suppress autoswitch reacting to foreground changes.
    # Typically called by QuickSwitch around manual focus changes.
    def suppress_for(self, duration_ms: int = 800, last_seen_hwnd: Optional[int] = None) -> None:
        try:
            now_ms = time.time() * 1000.0
            self._suppress_until_ms = max(self._suppress_until_ms, now_ms + max(0, int(duration_ms)))
            self._suppress_last_seen = int(last_seen_hwnd) if last_seen_hwnd else None
            self._logger.debug(
                f"Suppression armed for {duration_ms}ms (until={int(self._suppress_until_ms)}) last_seen={self._suppress_last_seen}"
            )
        except Exception as e:
            self._logger.error(f"Failed to arm suppression: {e}")

    def _get_active_overlay(self):
        """Get the currently active overlay from the overlay manager."""
        try:
            return self._overlay_manager.get_active_overlay()
        except Exception as e:
            self._logger.error(f"Failed to get active overlay: {e}")
            return None

    def _get_active_docking_manager(self):
        """Get the active docking manager if docking mode is active."""
        try:
            from utils.resource_manager import find_resource_by_description
            docking_manager = find_resource_by_description("DockingOverlayManager")
            if docking_manager and getattr(docking_manager, '_is_active', False):
                return docking_manager
        except Exception as e:
            self._logger.debug(f"Failed to get docking manager: {e}")
        return None

    # --- Internal helpers -------------------------------------------------
    def _is_overlay_locked(self, overlay) -> bool:
        """Return True if the active overlay is individually locked.

        Supports both direct DWM overlays (IntegratedDWMOverlay with `_is_window_locked`)
        and docking overlays (DockingOverlay wrapper exposing `_dwm_overlay`).
        """
        try:
            # Direct backend overlay
            if hasattr(overlay, "_is_window_locked"):
                return bool(getattr(overlay, "_is_window_locked", False))
            # Docking wrapper → backend
            if hasattr(overlay, "_dwm_overlay") and getattr(overlay, "_dwm_overlay") is not None:
                return bool(getattr(overlay._dwm_overlay, "_is_window_locked", False))
        except Exception:
            return False
        return False

    def _poll_tick(self) -> None:
        """Self-rescheduling polling tick using ThreadManager.single_shot.

        This replaces the raw QTimer periodic timer with a centralized, UI-thread
        single-shot that reschedules itself while enabled.
        """
        if not self._enabled or not self._polling_active:
            # Ensure we stop rescheduling when disabled
            self._polling_active = False
            return
        try:
            self._poll_foreground()
        except Exception as e:
            self._logger.error(f"Autoswitch poll tick failed: {e}", exc_info=True)
        finally:
            # Schedule next tick if still enabled/active
            if self._enabled and self._polling_active:
                ThreadManager.single_shot(self.POLL_INTERVAL_MS, self._poll_tick)

    def apply_settings(self) -> None:
        """Apply current settings and start/stop observation accordingly."""
        try:
            enabled_base = bool(self._settings.get("features.autoswitch_enabled", False))
            # Gate autoswitch when docking cycle mode is active
            cycle_active = False
            try:
                docking_mode = str(self._settings.get("docking.mode", "normal") or "normal").lower()
                if docking_mode == "cycle":
                    # Only gate if docking manager is active
                    dm = self._get_active_docking_manager()
                    cycle_active = dm is not None and getattr(dm, "is_active", lambda: False)()
            except Exception:
                cycle_active = False

            enabled = enabled_base and not cycle_active
            if enabled != self._enabled:
                self._enabled = enabled
                if self._enabled:
                    if not self._polling_active:
                        self._logger.debug("Enabling foreground autoswitch observation")
                        self._polling_active = True
                        ThreadManager.single_shot(self.POLL_INTERVAL_MS, self._poll_tick)
                else:
                    if self._polling_active:
                        why = "docking cycle mode" if cycle_active else "setting disabled"
                        self._logger.debug(f"Disabling autoswitch observation ({why})")
                        self._polling_active = False
                # Reset state when toggled
                self._last_seen_hwnd = None
                self._candidate_hwnd = None
                self._candidate_since = None
            else:
                # Ensure timer state matches
                if self._enabled and not self._polling_active:
                    self._logger.debug("Enabling autoswitch observation (sync)")
                    self._polling_active = True
                    ThreadManager.single_shot(self.POLL_INTERVAL_MS, self._poll_tick)
        except Exception as e:
            self._logger.error(f"Failed to apply autoswitch settings: {e}", exc_info=True)

    def _poll_foreground(self) -> None:
        if not self._enabled:
            return
        try:
            hwnd = win32gui.GetForegroundWindow()
        except Exception as e:
            self._logger.error(f"GetForegroundWindow failed: {e}")
            return

        # Suppression: ignore polls during a temporary gate (e.g., after QuickSwitch)
        try:
            now_ms = time.time() * 1000.0
            if now_ms < self._suppress_until_ms:
                # Optionally set last_seen to stabilize and avoid back-to-back triggers
                if self._suppress_last_seen is not None:
                    self._last_seen_hwnd = int(self._suppress_last_seen)
                # Keep candidate stable but do not act
                self._candidate_hwnd = hwnd
                self._candidate_since = now_ms
                return
            else:
                # Clear suppression window
                self._suppress_until_ms = 0.0
                self._suppress_last_seen = None
        except Exception:
            # Never fail polling due to suppression logic
            pass

        # Filter invalid
        if not is_valid_window(hwnd):
            return

        # Do not record MRU on every poll; we'll record when we actually apply a swap or when candidate changes

        now = time.time() * 1000.0

        # Debounce logic: require stable candidate for STABLE_DEBOUNCE_MS
        if self._candidate_hwnd != hwnd:
            self._candidate_hwnd = hwnd
            self._candidate_since = now
            return

        if self._candidate_since is None:
            self._candidate_since = now
            return

        if (now - self._candidate_since) < self.STABLE_DEBOUNCE_MS:
            return

        # Stable foreground change detected
        self._candidate_since = now  # prevent repeated triggers too fast
        # Note: MRU recording is handled by FocusTracker, not here

        # If same as last applied, skip
        if self._last_seen_hwnd == hwnd:
            return

        # Check for active docking mode first
        docking_manager = self._get_active_docking_manager()
        if docking_manager:
            # Delegate to docking mode autoswitch logic with correct semantics
            try:
                # Get current overlay assignments (all overlays A/B/C/D/E)
                main_overlay_hwnd = None
                if hasattr(docking_manager, '_main_overlay') and docking_manager._main_overlay:
                    main_overlay_hwnd = getattr(docking_manager._main_overlay, '_target_hwnd', None)
                
                # Check secondary overlays too
                secondary_matches = []
                if hasattr(docking_manager, '_secondary_overlays'):
                    for i, overlay in enumerate(docking_manager._secondary_overlays):
                        if overlay:
                            target_hwnd = getattr(overlay, '_target_hwnd', None)
                            if target_hwnd == hwnd:
                                secondary_matches.append((f'secondary_{i}', overlay))
                
                # Check if focused window matches main overlay
                if main_overlay_hwnd == hwnd:
                    # Check if main overlay is locked - if so, skip autoswitch
                    try:
                        is_locked = False
                        if hasattr(docking_manager, 'is_overlay_locked'):
                            is_locked = docking_manager.is_overlay_locked('main')
                        self._logger.debug(f"Docking autoswitch: main overlay lock check - is_locked={is_locked}, has_method={hasattr(docking_manager, 'is_overlay_locked')}")
                        
                        if is_locked:
                            self._logger.info(f"FOREGROUND_AUTOSWITCH BLOCKED: Main overlay is locked - skipping cycle for hwnd {hwnd}")
                        else:
                            self._logger.debug(f"Docking autoswitch: focusing same window {hwnd} as main overlay - triggering cycle")
                            docking_manager.handle_autoswitch_event(hwnd)
                    except Exception as e:
                        self._logger.error(f"Lock check failed: {e}")
                        # Fall back to normal behavior
                        self._logger.debug(f"Docking autoswitch: focusing same window {hwnd} as main overlay - triggering cycle")
                        docking_manager.handle_autoswitch_event(hwnd)
                
                # Check if focused window matches any secondary overlay (B/C/D/E)
                elif secondary_matches:
                    for overlay_id, overlay in secondary_matches:
                        try:
                            is_locked = False
                            if hasattr(docking_manager, 'is_overlay_locked'):
                                is_locked = docking_manager.is_overlay_locked(overlay_id)
                            
                            if is_locked:
                                self._logger.info(f"FOREGROUND_AUTOSWITCH BLOCKED: Overlay {overlay_id} is locked - skipping swap for hwnd {hwnd}")
                            else:
                                self._logger.debug(f"Docking autoswitch: focusing window {hwnd} in {overlay_id} - triggering swap")
                                # Trigger Normal mode swap for this secondary overlay
                                if hasattr(docking_manager, '_normal_mode_swap_with_foreground'):
                                    docking_manager._normal_mode_swap_with_foreground(overlay_id)
                                else:
                                    self._logger.warning("Docking manager missing _normal_mode_swap_with_foreground method")
                        except Exception as e:
                            self._logger.error(f"Secondary overlay {overlay_id} autoswitch failed: {e}")
                
                else:
                    self._logger.debug(f"Docking autoswitch: focusing different window {hwnd} (main={main_overlay_hwnd}) - no action")
                    # Just update MRU for the focused window
                    try:
                        get_mru_manager().record(hwnd)
                    except Exception:
                        pass
                
                self._last_seen_hwnd = hwnd
                return
            except Exception as e:
                self._logger.error(f"Docking mode autoswitch failed: {e}")
                # Fall through to regular overlay handling
        
        overlay = self._get_active_overlay()
        if overlay is None:
            # No active overlay; nothing to do
            return

        # Overlay lock gating: check both global and individual overlay lock states
        try:
            # Check global overlay lock
            is_globally_locked = self._overlay_manager.is_overlay_locked()
            # Check individual overlay lock state (supports docking wrapper by dereferencing backend)
            is_individually_locked = self._is_overlay_locked(overlay)
            
            if is_globally_locked or is_individually_locked:
                try:
                    oid = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                except Exception:
                    oid = None
                # Publish suppression event (best-effort)
                try:
                    from core.application.core import get_app_core
                    lock_type = "global" if is_globally_locked else "individual"
                    get_app_core().events.publish(
                        "switch.lock_suppressed",
                        {"overlay_id": oid, "source": "autoswitch", "lock_type": lock_type},
                        source="AutoswitchController",
                    )
                except Exception as e:
                    self._logger.debug(f"Event publish failed (lock_suppressed): {e}")
                self._logger.debug(f"Autoswitch suppressed due to {'global' if is_globally_locked else 'individual'} overlay lock")
                self._last_seen_hwnd = hwnd
                return
        except Exception:
            # Never fail due to lock gating check
            pass

        if not hasattr(overlay, "_handle_swap_window"):
            self._logger.error("Active overlay missing _handle_swap_window; autoswitch aborted")
            return

        # Determine current overlay source if available
        try:
            current_src = getattr(overlay, "_current_source_hwnd", None)
            if current_src is None:
                current_src = getattr(overlay, "_src_hwnd", None)
        except Exception:
            current_src = None
        if current_src == hwnd:
            # Foreground equals current capture — choose next MRU candidate cyclically (skip invalid like taskbar)
            mru = get_mru_manager()
            base = mru.get_recent(limit=7)
            # Build ordered candidates: [foreground, current_src, MRU...], unique-preserving
            ordered: list[int] = []
            seen: set[int] = set()
            if hwnd:
                ordered.append(hwnd)
                seen.add(hwnd)
            if current_src and current_src not in seen:
                ordered.append(current_src)
                seen.add(current_src)
            for h in base:
                if h and h not in seen:
                    ordered.append(h)
                    seen.add(h)
            candidates = ordered

            # Optional: Display Locked Switching — restrict to windows on the same monitor as overlay's content
            try:
                if bool(self._settings.get("features.display_locked_switching", False)) and current_src:
                    try:
                        src_r = get_window_rect(int(current_src))
                        src_mon = find_monitor_for_window(QPoint(src_r[0], src_r[1]), QSize(max(1, src_r[2]-src_r[0]), max(1, src_r[3]-src_r[1]))) if src_r else None
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

            # Exclude only the current foreground from target focus list
            filtered = [h for h in candidates if h and h != hwnd]
            if not filtered:
                # Reset cycle state to avoid sticky behavior and abort
                key = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                try:
                    if key is not None and str(key) in self._cycle_last_by_overlay:
                        self._cycle_last_by_overlay.pop(str(key), None)
                except Exception:
                    pass
                self._logger.debug("Autoswitch: candidates filtered to empty (fg==current); aborted and cycle reset")
                self._last_seen_hwnd = hwnd
                return
            chosen, display_idx, reason, ordered, start_idx, ref_last, ref_fore = compute_next_selection(
                candidates=candidates,
                filtered=filtered,
                cur_hwnd=hwnd,
                current_src=current_src,
                cycle_last_by_overlay=self._cycle_last_by_overlay,
                overlay_id=getattr(overlay, "id", None) or getattr(overlay, "identifier", None),
                # Autoswitch must not use Z-order fallback; adhere to strict no-fallback policy
                pick_from_zorder=lambda a, b: None,
                is_valid=is_valid_window,
            )
            # Monitor awareness (logging only)
            try:
                fg_rect = get_window_rect(hwnd)
                src_rect = get_window_rect(current_src) if current_src else None
                fg_mon = find_monitor_for_window(QPoint(fg_rect[0], fg_rect[1]), QSize(fg_rect[2]-fg_rect[0], fg_rect[3]-fg_rect[1])) if fg_rect else None
                src_mon = None
                if src_rect:
                    src_mon = find_monitor_for_window(QPoint(src_rect[0], src_rect[1]), QSize(src_rect[2]-src_rect[0], src_rect[3]-src_rect[1]))
            except Exception:
                fg_mon = None
                src_mon = None
            self._logger.debug(
                f"Autoswitch cycle (fg==current): reason={reason} start_idx={start_idx} list={ordered} chosen={chosen} fg_mon={fg_mon} src_mon={src_mon}"
            )
            if not chosen:
                # Reset cycle state on failure to allow fresh start next time
                key = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                try:
                    if key is not None and str(key) in self._cycle_last_by_overlay:
                        self._cycle_last_by_overlay.pop(str(key), None)
                except Exception:
                    pass
                self._logger.debug("Autoswitch: no valid MRU alternative; aborted and cycle reset")
                self._last_seen_hwnd = hwnd
                return
            try:
                # Validate chosen target to avoid swapping to our own application window
                try:
                    if not is_valid_window(chosen):
                        self._logger.debug(
                            f"Autoswitch: chosen hwnd={chosen} invalid or belongs to our process; aborted"
                        )
                        self._last_seen_hwnd = hwnd
                        return
                except Exception:
                    self._logger.debug(
                        f"Autoswitch: validation error for chosen hwnd={chosen}; aborted"
                    )
                    self._last_seen_hwnd = hwnd
                    return
                overlay._handle_swap_window(chosen)
                try:
                    get_mru_manager().record(chosen)
                except Exception:
                    pass
                self._last_seen_hwnd = hwnd
                self._last_applied_src = chosen
                try:
                    title = get_window_title(chosen)
                except Exception:
                    title = ""
                self._logger.debug(f"Autoswitch (fg==current) → MRU[{display_idx}] hwnd={chosen} title='{title}'")
                return
            except Exception as e:
                self._logger.error(f"Autoswitch (fg==current) swap failed: {e}", exc_info=True)
                self._last_seen_hwnd = hwnd
                return

        # Foreground != current_src: normal case is to do nothing.
        # Only engage an emergency swap if the current source is missing/invalid.
        emergency = False
        if current_src is None:
            emergency = True
        else:
            try:
                if not is_valid_window(current_src):
                    emergency = True
            except Exception:
                emergency = True

        if not emergency:
            # No action required; update last seen and exit quietly
            self._last_seen_hwnd = hwnd
            # Keep MRU warm so emergency paths have candidates
            try:
                get_mru_manager().record(hwnd)
            except Exception:
                pass
            self._logger.debug(
                f"Autoswitch: foreground!=current_src; no action (fg={hwnd}, cur={current_src})"
            )
            return

        # Emergency path: try to recover to a valid alternative using cyclic MRU
        try:
            mru = get_mru_manager()
            base = mru.get_recent(limit=7)
            # Build ordered candidates: [foreground, current_src, MRU...], unique-preserving
            ordered: list[int] = []
            seen: set[int] = set()
            if hwnd:
                ordered.append(hwnd)
                seen.add(hwnd)
            if current_src and current_src not in seen:
                ordered.append(current_src)
                seen.add(current_src)
            for h in base:
                if h and h not in seen:
                    ordered.append(h)
                    seen.add(h)
            candidates = ordered

            # Optional: Display Locked Switching — restrict to windows on the same monitor as overlay's content
            try:
                if bool(self._settings.get("features.display_locked_switching", False)) and current_src:
                    try:
                        src_r = get_window_rect(int(current_src))
                        src_mon = find_monitor_for_window(QPoint(src_r[0], src_r[1]), QSize(max(1, src_r[2]-src_r[0]), max(1, src_r[3]-src_r[1]))) if src_r else None
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

            filtered = [h for h in candidates if h and h != hwnd]
            if not filtered:
                key = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                try:
                    if key is not None and str(key) in self._cycle_last_by_overlay:
                        self._cycle_last_by_overlay.pop(str(key), None)
                except Exception:
                    pass
                self._logger.debug("Autoswitch emergency: candidates filtered to empty; aborted and cycle reset")
                self._last_seen_hwnd = hwnd
                return
            chosen, display_idx, reason, ordered, start_idx, ref_last, ref_fore = compute_next_selection(
                candidates=candidates,
                filtered=filtered,
                cur_hwnd=hwnd,
                current_src=current_src,
                cycle_last_by_overlay=self._cycle_last_by_overlay,
                overlay_id=getattr(overlay, "id", None) or getattr(overlay, "identifier", None),
                # Autoswitch must not use Z-order fallback; adhere to strict no-fallback policy
                pick_from_zorder=lambda a, b: None,
                is_valid=is_valid_window,
            )
            self._logger.debug(
                f"Autoswitch emergency cycle: reason={reason} start_idx={start_idx} list={ordered} chosen={chosen}"
            )
            if not chosen:
                key = getattr(overlay, "id", None) or getattr(overlay, "identifier", None)
                try:
                    if key is not None and str(key) in self._cycle_last_by_overlay:
                        self._cycle_last_by_overlay.pop(str(key), None)
                except Exception:
                    pass
                self._logger.debug("Autoswitch emergency: no valid MRU alternative; aborted and cycle reset")
                self._last_seen_hwnd = hwnd
                return
            try:
                if not is_valid_window(chosen):
                    self._logger.debug(
                        f"Autoswitch emergency: chosen hwnd={chosen} invalid; aborted"
                    )
                    self._last_seen_hwnd = hwnd
                    return
            except Exception:
                self._logger.debug(
                    f"Autoswitch emergency: validation error for chosen hwnd={chosen}; aborted"
                )
                self._last_seen_hwnd = hwnd
                return
            overlay._handle_swap_window(chosen)
            try:
                get_mru_manager().record(chosen)
            except Exception:
                pass
            self._last_seen_hwnd = hwnd
            self._last_applied_src = chosen
            try:
                title = get_window_title(chosen)
            except Exception:
                title = ""
            self._logger.debug(f"Autoswitch emergency → MRU[{display_idx}] hwnd={chosen} title='{title}'")
        except Exception as e:
            self._logger.error(f"Autoswitch emergency swap failed: {e}", exc_info=True)

# Convenience accessor
_foreground_autoswitch_controller: Optional[ForegroundAutoswitchController] = None

def get_foreground_autoswitch_controller() -> ForegroundAutoswitchController:
    """Get the global foreground autoswitch controller instance."""
    global _foreground_autoswitch_controller
    if _foreground_autoswitch_controller is None:
        _foreground_autoswitch_controller = ForegroundAutoswitchController()
    return _foreground_autoswitch_controller
