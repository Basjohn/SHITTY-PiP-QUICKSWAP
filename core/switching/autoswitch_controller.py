from __future__ import annotations

import time
import threading
from typing import Optional

from PySide6.QtCore import QObject, QPoint, QSize

from core.logging import get_logger
from core.settings.settings_manager import SettingsManager
from core.graphics.overlay_manager import OverlayManager
from core.threading import ThreadManager

import win32gui

from utils.window_validation import is_valid_window, get_window_rect, get_window_title
from core.switching.mru_manager import get_mru_manager
from core.switching.selection import compute_next_selection
from utils.window.monitors import find_monitor_for_window


class AutoswitchController(QObject):
    """
    Centralized Autoswitch controller (scaffold).

    - Observes foreground window changes using a Qt timer (polling)
    - Applies strict filtering to candidate windows
    - Debounces rapid changes before triggering a swap
    - Safely delegates swapping to the overlay's _handle_swap_window (UI-thread safe)
    - Live settings via apply_settings(); reads 'features.autoswitch_enabled'

    Explicit logging with 'AUTOSWITCH' prefix. No silent fallbacks.
    """

    POLL_INTERVAL_MS = 250
    STABLE_DEBOUNCE_MS = 300

    _instance: Optional["AutoswitchController"] = None
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
        self._logger = get_logger("AUTOSWITCH")
        self._settings = SettingsManager()
        self._overlay_manager = OverlayManager()

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

        self.apply_settings()

        self._initialized = True
        self._logger.debug("Initialized AutoswitchController")

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
        try:
            active_id = getattr(self._overlay_manager, "_active_overlay_id", None)
            if not active_id:
                return None
            overlays = getattr(self._overlay_manager, "_overlays", {})
            return overlays.get(active_id)
        except Exception:
            return None

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
            enabled = bool(self._settings.get("features.autoswitch_enabled", False))
            if enabled != self._enabled:
                self._enabled = enabled
                if self._enabled:
                    if not self._polling_active:
                        self._logger.debug("Enabling autoswitch observation")
                        self._polling_active = True
                        ThreadManager.single_shot(self.POLL_INTERVAL_MS, self._poll_tick)
                else:
                    if self._polling_active:
                        self._logger.debug("Disabling autoswitch observation")
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

        overlay = self._get_active_overlay()
        if overlay is None:
            # No active overlay; nothing to do
            return

        # Overlay lock gating: check both global and individual overlay lock states
        try:
            # Check global overlay lock
            is_globally_locked = self._overlay_manager.is_overlay_locked()
            # Check individual overlay lock state
            is_individually_locked = getattr(overlay, "_is_window_locked", False)
            
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
_def_instance: Optional[AutoswitchController] = None

def get_autoswitch_controller() -> AutoswitchController:
    global _def_instance
    if _def_instance is None:
        _def_instance = AutoswitchController()
    return _def_instance
