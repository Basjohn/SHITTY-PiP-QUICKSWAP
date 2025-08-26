from __future__ import annotations

import threading
import time
from typing import Optional

from PySide6.QtCore import QObject, Signal

from core.logging import get_logger
from core.settings.settings_manager import SettingsManager
from core.threading import ThreadManager
from core.media import get_media_controller
from utils.state.focus_state import get_focus_state
from .keypassthrough_blocklist import get_blocklist

# Windows messaging helpers (safe wrappers)
from utils.win.winmsg import (
    is_window as _is_window,
    post_message as _post_message,
    pack_lparam_key as _pack_lparam_key,
)

try:
    import win32con as _w32c  # for WM_* constants and VKs if needed
    _WIN_AVAILABLE = True
except Exception:
    _WIN_AVAILABLE = False

# Fallback WM_* values if pywin32 not present (should be present on Windows)
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101

# Media key VK codes
_VK_MEDIA_PLAY_PAUSE = 0xB3
_VK_MEDIA_STOP = 0xB2
_VK_MEDIA_PREV_TRACK = 0xB1
_VK_MEDIA_NEXT_TRACK = 0xB0
_MEDIA_KEYS = {_VK_MEDIA_PLAY_PAUSE, _VK_MEDIA_STOP, _VK_MEDIA_PREV_TRACK, _VK_MEDIA_NEXT_TRACK}

# Local media fallback key(s)
_VK_SPACE = 0x20
_VK_LEFT = 0x25
_VK_UP = 0x26
_VK_RIGHT = 0x27
_VK_DOWN = 0x28

# System volume keys
_VK_VOLUME_MUTE = 0xAD
_VK_VOLUME_DOWN = 0xAE
_VK_VOLUME_UP = 0xAF


class KeyPassthroughController(QObject):
    """
    Centralized, rate-limited key passthrough controller.

    Phase 1 policy:
    - PostMessage-only (non-blocking), never steals focus.
    - Windows-only; safe no-op on non-Windows.
    - Settings integration via key 'features.keypassthrough_enabled'.
    - Rate limiting (min interval) using ThreadManager.single_shot.
    - Qt signals for state changes; EventSystem publishing for observability.
    - Media key routing to MediaController when enabled.
    - Browser child hotkey path is ONLY used when media control is disabled; otherwise media
      commands are routed via MediaController (no global media commands when no target).

    Public API:
    - set_target_hwnd(hwnd: Optional[int])
    - passthrough_key(vk: int) -> bool
    - enabled property reflects settings; changes via settings only.
    """

    enabled_changed = Signal(bool)
    target_changed = Signal(int)

    _instance: Optional["KeyPassthroughController"] = None
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
        self._logger = get_logger("KEYPASS")
        self._settings = SettingsManager()

        # State
        self._enabled: bool = bool(self._settings.get("features.keypassthrough_enabled", False))
        self._target_hwnd: Optional[int] = None
        self._media_routing_enabled: bool = bool(self._settings.get("features.media_control_enabled", False))
        # Verbose decision logging (non-media paths)
        self._verbose: bool = bool(self._settings.get("debug.keypassthrough_verbose", False))
        # Verbose logging specifically for volume hold timers
        self._vol_verbose: bool = bool(self._settings.get("debug.volume_hold_verbose", False))

        # Block feedback throttling/deduping state
        self._block_lock = threading.Lock()
        self._last_block_reason: Optional[str] = None
        self._last_block_flash_ms: float = 0.0
        try:
            self._block_flash_interval_ms: int = int(self._settings.get("ui.block_flash_min_interval_ms", 250))
        except Exception:
            self._block_flash_interval_ms = 250

        # Rate limiting
        self._min_interval_ms: int = 18  # ~55Hz cap
        self._keyup_delay_ms: int = 10   # minimal keyup separation
        self._last_sent_ts: float = 0.0
        self._pending_vk: Optional[int] = None
        self._send_scheduled: bool = False

        # Rapid-hold suppression for Up/Down (thread-safe)
        self._vol_suppress_ms: int = 35
        self._last_vol_up_ts: float = 0.0
        self._last_vol_down_ts: float = 0.0
        self._rate_lock = threading.Lock()

        # Volume hold (press-and-hold) state
        self._hold_lock = threading.RLock()
        self._hold_active_up: bool = False
        self._hold_active_down: bool = False
        self._hold_token_up: int = 0
        self._hold_token_down: int = 0
        # Configurable initial delay and repeat interval (ms)
        try:
            self._hold_initial_delay_ms: int = int(self._settings.get("input.volume_hold_initial_delay_ms", 200))
        except Exception:
            self._hold_initial_delay_ms = 200
        try:
            self._hold_interval_ms: int = int(self._settings.get("input.volume_hold_interval_ms", 75))
        except Exception:
            self._hold_interval_ms = 75

        # Settings change wiring
        try:
            self._settings.register_change_handler(
                "features.keypassthrough_enabled", self._on_setting_changed
            )
            self._settings.register_change_handler(
                "features.media_control_enabled", self._on_setting_changed
            )
        except Exception as e:
            self._logger.error(f"Failed to register settings handler: {e}")

        self._initialized = True
        self._logger.debug(
            f"Initialized KeyPassthroughController enabled={self._enabled} media_routing={self._media_routing_enabled} min_interval={self._min_interval_ms}ms"
        )
        # Emit initial state to ensure subscribers reflect startup setting
        try:
            ThreadManager.run_on_ui_thread(lambda: self._emit_enabled_changed(self._enabled))
        except Exception:
            self._emit_enabled_changed(self._enabled)

    # --- Public API -----------------------------------------------------
    def set_target_hwnd(self, hwnd: Optional[int]) -> None:
        """Set or clear the passthrough target window handle.

        If hwnd is not a valid top-level window (or platform not Windows), target is cleared.
        """
        new_hwnd: Optional[int] = None
        try:
            if hwnd and _WIN_AVAILABLE and _is_window(int(hwnd)):
                new_hwnd = int(hwnd)
        except Exception:
            new_hwnd = None

        changed = new_hwnd != self._target_hwnd
        self._target_hwnd = new_hwnd
        if changed:
            # Emit on UI thread
            try:
                ThreadManager.run_on_ui_thread(lambda: self._emit_target_changed(new_hwnd))
            except Exception:
                self._emit_target_changed(new_hwnd)
            self._publish("key.passthrough.target", {"hwnd": int(new_hwnd) if new_hwnd else 0})
            self._logger.debug(f"Target hwnd set to {new_hwnd}")
            
            # Prepare window for media commands if media routing is enabled
            if new_hwnd and self._media_routing_enabled:
                self._prepare_window_for_media(new_hwnd)

    def passthrough_key(self, vk: int) -> bool:
        """Request forwarding of a virtual-key to the target window.

        Media keys are routed to MediaController if media routing is enabled.
        Other keys are passed through to the target window if passthrough is enabled.
        
        Returns True if accepted for delivery (immediate or scheduled), False otherwise.
        """
        try:
            vk_int = int(vk)
        except Exception:
            return False
        
        # Route media keys to MediaController if enabled
        if self._media_routing_enabled and vk_int in _MEDIA_KEYS:
            # Suppress global media keys while overlay is focused to avoid duplicates
            try:
                if get_focus_state().is_overlay_focused():
                    return self._block("media-key-overlay-focused", extra={"vk": int(vk_int)})
            except Exception:
                # Be conservative on failure and block to prevent duplicate handling
                return self._block("media-key-focus-check-failed", extra={"vk": int(vk_int)})
            return self._route_media_key(vk_int)

        # Route system volume keys (hardware VKs) when enabled
        # IMPORTANT: perform overlay-focus gating BEFORE mapping arrows to volume
        if self._media_routing_enabled and vk_int in (_VK_VOLUME_UP, _VK_VOLUME_DOWN, _VK_VOLUME_MUTE):
            # If overlay is focused, ignore hardware volume keys to avoid duplicate changes
            try:
                if get_focus_state().is_overlay_focused():
                    return self._block("volume-key-overlay-focused", extra={"vk": int(vk_int)})
            except Exception:
                # Be conservative on failure and block to prevent duplicate handling
                return self._block("volume-key-focus-check-failed", extra={"vk": int(vk_int)})
        
        # When Media Control is ON, treat Arrow Up/Down as Volume Up/Down (local mapping)
        if self._media_routing_enabled and vk_int in (_VK_UP, _VK_DOWN):
            vk_int = _VK_VOLUME_UP if vk_int == _VK_UP else _VK_VOLUME_DOWN

        # Route system volume keys (correct mapping) when enabled
        if self._media_routing_enabled and vk_int in (_VK_VOLUME_UP, _VK_VOLUME_DOWN, _VK_VOLUME_MUTE):
            try:
                # Media Control path: do not depend on passthrough being enabled
                media_controller = get_media_controller()
                target_hwnd = self._target_hwnd if (_WIN_AVAILABLE and self._target_hwnd and _is_window(self._target_hwnd)) else None
                if not target_hwnd:
                    self._publish("key.passthrough.media_routed", {
                        "vk": int(vk_int),
                        "success": False,
                        "message": "No target hwnd; global media routing disabled",
                        "note": "volume-no-target"
                    })
                    self._logger.warning("Volume key ignored: no target hwnd; global media routing disabled by policy")
                    return self._block("volume-no-target")

                # Tiny suppression window for rapid holds on Volume Up/Down
                if vk_int in (_VK_VOLUME_UP, _VK_VOLUME_DOWN):
                    now_ms = time.monotonic() * 1000.0
                    with self._rate_lock:
                        if vk_int == _VK_VOLUME_UP:
                            if now_ms - self._last_vol_up_ts < float(self._vol_suppress_ms):
                                return True
                            self._last_vol_up_ts = now_ms
                        else:  # _VK_VOLUME_DOWN
                            if now_ms - self._last_vol_down_ts < float(self._vol_suppress_ms):
                                return True
                            self._last_vol_down_ts = now_ms

                if vk_int == _VK_VOLUME_UP:
                    success, msg = media_controller.volume_up_for_hwnd(int(target_hwnd))
                    note = "volume-up-local"
                elif vk_int == _VK_VOLUME_DOWN:
                    success, msg = media_controller.volume_down_for_hwnd(int(target_hwnd))
                    note = "volume-down-local"
                else:  # _VK_VOLUME_MUTE
                    # Map to APPCOMMAND mute through send_media_command path
                    success, msg = media_controller._send_command_for_hwnd(int(target_hwnd), media_controller.APPCOMMAND_VOLUME_MUTE)
                    note = "volume-mute-local"

                self._publish("key.passthrough.media_routed", {
                    "vk": int(vk_int),
                    "success": success,
                    "message": msg,
                    "note": note
                })
                if success:
                    if self._verbose:
                        self._logger.debug(f"Volume key routed: {note}: {msg}")
                else:
                    self._logger.warning(f"Volume key routing failed ({note}): {msg}")
                return success
            except Exception as e:
                self._logger.error(f"Failed to route volume key to media controller: {e}")
                return False

        # Note: Up/Down arrows are mapped to volume keys above when media control is enabled.

        # Route spacebar as local play/pause ONLY when there is a valid target hwnd
        if self._media_routing_enabled and vk_int == _VK_SPACE:
            try:
                media_controller = get_media_controller()
                target_hwnd = self._target_hwnd if (_WIN_AVAILABLE and self._target_hwnd and _is_window(self._target_hwnd)) else None
                if not target_hwnd:
                    self._publish("key.passthrough.media_routed", {
                        "vk": int(vk_int),
                        "success": False,
                        "message": "No target hwnd; global media routing disabled",
                        "note": "spacebar-no-target"
                    })
                    self._logger.warning("Spacebar ignored: no target hwnd; global media routing disabled by policy")
                    return self._block("spacebar-no-target")
                # Firefox special-case: behave like nonmedia passthrough for SPACE
                try:
                    app_name = media_controller.detect_app_for_hwnd(int(target_hwnd))
                except Exception:
                    app_name = None
                if app_name == 'firefox':
                    handled = self._enqueue_passthrough(int(vk_int))
                    self._publish("key.passthrough.media_routed", {
                        "vk": int(vk_int),
                        "success": handled,
                        "message": "Firefox bypass: SPACE passthrough",
                        "note": "spacebar-firefox-bypass"
                    })
                    return handled

                success, msg = media_controller.play_pause_for_hwnd(int(target_hwnd))
                self._publish("key.passthrough.media_routed", {
                    "vk": int(vk_int),
                    "success": success,
                    "message": msg,
                    "note": "spacebar-local"
                })
                if success:
                    if self._verbose:
                        self._logger.debug(f"Spacebar routed as play/pause: {msg}")
                else:
                    self._logger.warning(f"Spacebar routing failed: {msg}")
                return success
            except Exception as e:
                self._logger.error(f"Failed to route spacebar to media controller: {e}")
                return False

        # Route arrow keys ONLY when there is a valid target hwnd (no volume mapping here)
        if self._media_routing_enabled and vk_int in (_VK_LEFT, _VK_RIGHT):
            try:
                media_controller = get_media_controller()
                target_hwnd = self._target_hwnd if (_WIN_AVAILABLE and self._target_hwnd and _is_window(self._target_hwnd)) else None
                if not target_hwnd:
                    self._publish("key.passthrough.media_routed", {
                        "vk": int(vk_int),
                        "success": False,
                        "message": "No target hwnd; global media routing disabled",
                        "note": "arrow-no-target"
                    })
                    self._logger.warning("Arrow key ignored: no target hwnd; global media routing disabled by policy")
                    return self._block("arrow-no-target")
                # Firefox special-case: behave like nonmedia passthrough for ARROWS
                try:
                    app_name = media_controller.detect_app_for_hwnd(int(target_hwnd))
                except Exception:
                    app_name = None
                if app_name == 'firefox':
                    handled = self._enqueue_passthrough(int(vk_int))
                    self._publish("key.passthrough.media_routed", {
                        "vk": int(vk_int),
                        "success": handled,
                        "message": "Firefox bypass: ARROW passthrough",
                        "note": "arrow-firefox-bypass"
                    })
                    return handled

                if vk_int == _VK_LEFT:
                    success, msg = media_controller.previous_for_hwnd(int(target_hwnd))
                    note = "previous-local"
                elif vk_int == _VK_RIGHT:
                    success, msg = media_controller.next_for_hwnd(int(target_hwnd))
                    note = "next-local"
                self._publish("key.passthrough.media_routed", {
                    "vk": int(vk_int),
                    "success": success,
                    "message": msg,
                    "note": note
                })
                if success:
                    if self._verbose:
                        self._logger.debug(f"Arrow key routed: {note}: {msg}")
                else:
                    self._logger.warning(f"Arrow key routing failed ({note}): {msg}")
                return success
            except Exception as e:
                self._logger.error(f"Failed to route arrow key to media controller: {e}")
                return False
            
        # If this is a media key but media routing is disabled, do NOT forward via passthrough.
        # This enforces the policy: media functionality only when media control is enabled.
        if vk_int in _MEDIA_KEYS:
            return self._block("media-key-media-routing-disabled")

        # Standard key passthrough
        if not self._enabled:
            if self._verbose:
                self._logger.debug("Passthrough disabled via settings; ignoring key")
            return self._block("passthrough-disabled")
        if not _WIN_AVAILABLE:
            return self._block("platform-not-windows")
        hwnd = self._target_hwnd
        if not (hwnd and _is_window(hwnd)):
            if self._verbose:
                self._logger.debug("No valid target hwnd; ignoring key")
            # Distinguish between missing and invalid target for clearer telemetry
            if hwnd:
                return self._block("invalid-target", extra={"hwnd": int(hwnd)})
            return self._block("no-target-hwnd")

        # Blocklist pre-check: consult centralized loader before forwarding
        # Policy: when Media Control is enabled, do NOT block media-related keys via blocklist.
        # Media-related here includes hardware media keys and common mappings (SPACE/ARROWS)
        # which are otherwise handled earlier in media-routing branches.
        try:
            media_related = vk_int in (_VK_SPACE, _VK_LEFT, _VK_RIGHT) or vk_int in _MEDIA_KEYS
            if bool(self._settings.get("features.keypassthrough_blocklist_enabled", True)):
                if not (self._media_routing_enabled and media_related):
                    bl = get_blocklist()
                    match = bl.match_for_hwnd(int(hwnd))
                    if match:
                        if self._verbose:
                            self._logger.debug(f"Blocked by blocklist: {match}")
                        return self._block("blocklist", extra={"hwnd": int(hwnd), "match": match})
                else:
                    if self._verbose:
                        self._logger.debug("Blocklist bypass: media control enabled and media-related key")
        except Exception as e:
            # Non-fatal: log at debug and continue passthrough path
            if self._verbose:
                self._logger.debug(f"Blocklist check failed: {e}")

        # When media routing is disabled, try browser-aware child targeting for common media-related keys.
        # This avoids sending to the wrong child window in multi-child apps (e.g., browsers) while
        # maintaining the policy of not issuing global media commands.
        if not self._media_routing_enabled and vk_int in (_VK_SPACE, _VK_LEFT, _VK_RIGHT):
            try:
                if self._try_browser_child_hotkey(int(vk_int)):
                    return True
            except Exception:
                # Fall back to normal passthrough
                pass

        return self._enqueue_passthrough(vk_int)

    # --- Press/Release API (with timer-based volume holds) --------------
    def press_passthrough_key(self, vk: int) -> bool:
        """Press semantics for passthrough.

        - For Volume Up/Down when media routing is enabled: perform an immediate step
          and start a hold timer for smooth repeats.
        - For other keys, fall back to single-tap passthrough logic.
        """
        try:
            vk_int = int(vk)
        except Exception:
            return False
        
        # If this is a hardware volume key and overlay is focused, ignore to avoid duplicate changes
        if self._media_routing_enabled and vk_int in (_VK_VOLUME_UP, _VK_VOLUME_DOWN):
            try:
                if get_focus_state().is_overlay_focused():
                    return self._block("volume-press-overlay-focused", extra={"vk": int(vk_int)})
            except Exception:
                return self._block("volume-press-focus-check-failed", extra={"vk": int(vk_int)})

        # When Media Control is ON, treat Arrow Up/Down as Volume Up/Down (local mapping)
        if self._media_routing_enabled and vk_int in (_VK_UP, _VK_DOWN):
            vk_int = _VK_VOLUME_UP if vk_int == _VK_UP else _VK_VOLUME_DOWN
        # Volume keys when media routing is enabled
        if self._media_routing_enabled and vk_int in (_VK_VOLUME_UP, _VK_VOLUME_DOWN):
            # Gate by valid target hwnd
            target_hwnd = self._target_hwnd if (_WIN_AVAILABLE and self._target_hwnd and _is_window(self._target_hwnd)) else None
            if not target_hwnd:
                return self._block("volume-press-no-target")

            # Immediate step for responsiveness
            self.passthrough_key(_VK_VOLUME_UP if vk_int == _VK_VOLUME_UP else _VK_VOLUME_DOWN)
            # Start hold timer
            try:
                if self._vol_verbose:
                    self._logger.debug(
                        f"VOL_HOLD: press start is_up={vk_int == _VK_VOLUME_UP} initial_delay={self._hold_initial_delay_ms}ms interval={self._hold_interval_ms}ms hwnd={int(target_hwnd)}"
                    )
            except Exception:
                pass
            self._start_volume_hold(is_up=(vk_int == _VK_VOLUME_UP))
            return True

        return self.passthrough_key(vk_int)

    def release_passthrough_key(self, vk: int) -> None:
        """Release semantics for passthrough.

        Stops any active volume hold timers for Volume Up/Down.
        """
        try:
            vk_int = int(vk)
        except Exception:
            return
        if vk_int in (_VK_VOLUME_UP, _VK_UP):
            self._stop_volume_hold(is_up=True)
        elif vk_int in (_VK_VOLUME_DOWN, _VK_DOWN):
            self._stop_volume_hold(is_up=False)
        return

    # Volume hold helpers -------------------------------------------------
    def _start_volume_hold(self, is_up: bool) -> None:
        with self._hold_lock:
            # Ensure mutual exclusivity: cancel opposite direction if active
            if is_up and self._hold_active_down:
                self._hold_active_down = False
                self._hold_token_down += 1  # invalidate any pending DOWN ticks
                try:
                    if self._vol_verbose:
                        self._logger.debug("VOL_HOLD: cancelling opposite (down) before starting up")
                except Exception:
                    pass
            elif (not is_up) and self._hold_active_up:
                self._hold_active_up = False
                self._hold_token_up += 1  # invalidate any pending UP ticks
                try:
                    if self._vol_verbose:
                        self._logger.debug("VOL_HOLD: cancelling opposite (up) before starting down")
                except Exception:
                    pass

            if is_up:
                if self._hold_active_up:
                    try:
                        if self._vol_verbose:
                            self._logger.debug("VOL_HOLD: start ignored (already active up)")
                    except Exception:
                        pass
                    return
                self._hold_active_up = True
                self._hold_token_up += 1
                tok = int(self._hold_token_up)
            else:
                if self._hold_active_down:
                    try:
                        if self._vol_verbose:
                            self._logger.debug("VOL_HOLD: start ignored (already active down)")
                    except Exception:
                        pass
                    return
                self._hold_active_down = True
                self._hold_token_down += 1
                tok = int(self._hold_token_down)
        # Schedule first repeat after initial delay
        delay = max(0, int(self._hold_initial_delay_ms))
        try:
            try:
                self._publish("key.passthrough.volume.hold_start", {"direction": "up" if is_up else "down", "token": int(tok), "delay_ms": int(delay)})
                if self._vol_verbose:
                    self._logger.debug(f"VOL_HOLD: scheduled first tick token={tok} delay={delay}ms")
            except Exception:
                pass
            ThreadManager.single_shot(delay, lambda: self._volume_hold_tick(is_up, tok))
        except Exception:
            # Best-effort immediate tick
            self._volume_hold_tick(is_up, tok)

    def _stop_volume_hold(self, is_up: bool) -> None:
        with self._hold_lock:
            if is_up:
                self._hold_active_up = False
                self._hold_token_up += 1  # invalidate pending ticks
            else:
                self._hold_active_down = False
                self._hold_token_down += 1
        try:
            self._publish("key.passthrough.volume.hold_stop", {"direction": "up" if is_up else "down"})
            if self._vol_verbose:
                self._logger.debug(f"VOL_HOLD: stop direction={'up' if is_up else 'down'}")
        except Exception:
            pass

    def _volume_hold_tick(self, is_up: bool, token: int) -> None:
        # Validate still active and token matches
        with self._hold_lock:
            active = self._hold_active_up if is_up else self._hold_active_down
            cur_tok = self._hold_token_up if is_up else self._hold_token_down
        # Diagnostic snapshot of gating state for this tick
        try:
            try:
                overlay_focused = get_focus_state().is_overlay_focused()
            except Exception:
                overlay_focused = False
            try:
                tgt = self._target_hwnd if (_WIN_AVAILABLE and self._target_hwnd and _is_window(self._target_hwnd)) else None
            except Exception:
                tgt = None
            if self._vol_verbose:
                self._logger.debug(
                    f"VOL_HOLD: tick snapshot is_up={is_up} token={token} cur_tok={cur_tok} active={active} "
                    f"media_enabled={self._media_routing_enabled} overlay_focused={overlay_focused} "
                    f"target_valid={bool(tgt)} target_hwnd={int(tgt) if tgt else 0}"
                )
        except Exception:
            pass
        if not active or token != cur_tok:
            try:
                if self._vol_verbose:
                    self._logger.debug(
                        f"VOL_HOLD: tick ignored token_mismatch_or_inactive is_up={is_up} token={token} cur_tok={cur_tok} active={active}"
                    )
            except Exception:
                pass
            return

        # Helper: reschedule next repeat if still active and token matches
        def _reschedule_if_active() -> None:
            with self._hold_lock:
                a2 = self._hold_active_up if is_up else self._hold_active_down
                t2 = self._hold_token_up if is_up else self._hold_token_down
            if not a2 or token != t2:
                try:
                    if self._vol_verbose:
                        self._logger.debug("VOL_HOLD: not rescheduling (inactive or token changed)")
                except Exception:
                    pass
                return
            try:
                interval = max(0, int(self._hold_interval_ms))
                try:
                    if self._vol_verbose:
                        self._logger.debug(f"VOL_HOLD: reschedule token={token} interval={interval}ms")
                except Exception:
                    pass
                ThreadManager.single_shot(interval, lambda: self._volume_hold_tick(is_up, token))
            except Exception:
                # If scheduling fails, attempt one more immediate tick to avoid stall
                self._volume_hold_tick(is_up, token)

        # Validate environment: media routing enabled and valid target hwnd
        try:
            if not self._media_routing_enabled:
                try:
                    if self._vol_verbose:
                        self._logger.debug("VOL_HOLD: tick gated (media routing disabled) — rescheduling")
                except Exception:
                    pass
                _reschedule_if_active()
                return
        except Exception:
            try:
                if self._vol_verbose:
                    self._logger.debug("VOL_HOLD: tick gated (focus check failed) — rescheduling")
            except Exception:
                pass
            _reschedule_if_active()
            return
        target_hwnd = self._target_hwnd if (_WIN_AVAILABLE and self._target_hwnd and _is_window(self._target_hwnd)) else None
        if not target_hwnd:
            try:
                if self._vol_verbose:
                    self._logger.debug("VOL_HOLD: tick gated (no target hwnd) — rescheduling")
            except Exception:
                pass
            _reschedule_if_active()
            return

        # Perform a volume step via MediaController (local to hwnd)
        try:
            mc = get_media_controller()
            if mc:
                if is_up:
                    ok, msg = mc.volume_up_for_hwnd(int(target_hwnd))
                    try:
                        self._publish("key.passthrough.volume.hold_step", {"direction": "up", "ok": bool(ok), "msg": msg or ""})
                        if not ok and self._vol_verbose:
                            self._logger.debug(f"VOL_HOLD: step up failed: {msg}")
                    except Exception:
                        pass
                else:
                    ok, msg = mc.volume_down_for_hwnd(int(target_hwnd))
                    try:
                        self._publish("key.passthrough.volume.hold_step", {"direction": "down", "ok": bool(ok), "msg": msg or ""})
                        if not ok and self._vol_verbose:
                            self._logger.debug(f"VOL_HOLD: step down failed: {msg}")
                    except Exception:
                        pass
        except Exception:
            pass

        # Reschedule next repeat if still active
        _reschedule_if_active()

    def is_enabled(self) -> bool:
        """Return current passthrough enabled state (settings-backed)."""
        try:
            return bool(self._enabled)
        except Exception:
            return False

    # --- Internals ------------------------------------------------------
    def _emit_target_changed(self, hwnd: Optional[int]) -> None:
        try:
            if hwnd is not None:
                self.target_changed.emit(int(hwnd))
        except Exception:
            pass

    def _emit_enabled_changed(self, enabled: bool) -> None:
        try:
            self.enabled_changed.emit(bool(enabled))
        except Exception:
            pass

    def _on_setting_changed(self, key: str, value) -> None:
        if key == "features.keypassthrough_enabled":
            new_enabled = bool(value)
            if new_enabled == self._enabled:
                return
            self._enabled = new_enabled
            # Stop any scheduled send when disabling
            if not new_enabled:
                self._pending_vk = None
                self._send_scheduled = False
            # Emit on UI thread
            try:
                ThreadManager.run_on_ui_thread(lambda: self._emit_enabled_changed(new_enabled))
            except Exception:
                self._emit_enabled_changed(new_enabled)
            self._publish("key.passthrough.enabled", {"enabled": new_enabled})
            self._logger.debug(f"Passthrough enabled set to {new_enabled} via settings")
        elif key == "features.media_control_enabled":
            new_media_enabled = bool(value)
            if new_media_enabled == self._media_routing_enabled:
                return
            self._media_routing_enabled = new_media_enabled
            self._publish("key.passthrough.media_routing", {"enabled": new_media_enabled})
            self._logger.debug(f"Media routing enabled set to {new_media_enabled} via settings")

    def _flush_send(self) -> None:
        self._send_scheduled = False
        vk = self._pending_vk
        self._pending_vk = None
        if vk is None:
            return
        hwnd = self._target_hwnd
        if not (_WIN_AVAILABLE and hwnd and _is_window(hwnd)):
            return
        # Compute lparams once
        lp_down = _pack_lparam_key(vk, is_keyup=False, repeat=1)
        lp_up = _pack_lparam_key(vk, is_keyup=True, repeat=1)

        # Post keydown immediately, keyup after small delay
        ok_down = _post_message(hwnd, _w32c.WM_KEYDOWN if _WIN_AVAILABLE else _WM_KEYDOWN, vk, lp_down)
        if not ok_down:
            return
        self._last_sent_ts = time.monotonic() * 1000.0
        self._publish("key.passthrough.forwarded", {"vk": int(vk), "hwnd": int(hwnd), "phase": "down"})
        # Schedule keyup
        try:
            ThreadManager.single_shot(self._keyup_delay_ms, lambda: self._post_keyup(hwnd, vk, lp_up))
        except Exception:
            self._post_keyup(hwnd, vk, lp_up)

    def _post_keyup(self, hwnd: int, vk: int, lp_up: int) -> None:
        if not (_WIN_AVAILABLE and hwnd and _is_window(hwnd)):
            return
        _post_message(hwnd, _w32c.WM_KEYUP if _WIN_AVAILABLE else _WM_KEYUP, vk, lp_up)
        self._publish("key.passthrough.forwarded", {"vk": int(vk), "hwnd": int(hwnd), "phase": "up"})
        
    def _block(self, reason: str, extra: Optional[dict] = None) -> bool:
        """Centralized handler for blocked passthrough decisions.

        - Publishes an observability event with the reason
        - Triggers a brief black flash on the active overlay's focus indicator
          via OverlayHost, using ThreadManager to run on the UI thread
        - Always returns False for call-site convenience
        """
        try:
            # Publish observability event
            payload = {"reason": str(reason)}
            if extra is not None:
                try:
                    # Ensure serializable basics only
                    payload.update({"extra": extra})
                except Exception:
                    pass
            self._publish("key.passthrough.blocked", payload)
        except Exception:
            pass

        # Optional debug log for blocked decisions
        try:
            if self._verbose:
                self._logger.debug(f"KEYPASS BLOCK: reason={reason} extra={extra if extra is not None else {}}")
        except Exception:
            pass

        # Attempt to trigger the focus indicator flash on the active overlay, throttled/deduped
        try:
            now_ms = time.monotonic() * 1000.0
            should_flash = False
            with self._block_lock:
                # Dedupe: if same reason within interval, suppress
                if (now_ms - self._last_block_flash_ms) >= float(self._block_flash_interval_ms) or (self._last_block_reason != reason):
                    self._last_block_flash_ms = now_ms
                    self._last_block_reason = str(reason)
                    should_flash = True
            if should_flash:
                # Local import to avoid import-time cycles
                from core.graphics.overlay_manager import OverlayManager
                overlay = OverlayManager().get_active_overlay()
                if overlay is not None:
                    host = getattr(overlay, "_host", None)
                    if host is not None and hasattr(host, "flash_focus_indicator"):
                        try:
                            ThreadManager.run_on_ui_thread(lambda: host.flash_focus_indicator(300))
                        except Exception:
                            # Last-resort direct call; host should be a QWidget-owned object
                            try:
                                host.flash_focus_indicator(300)
                            except Exception:
                                pass
        except Exception:
            # Non-critical; ignore failures in UI feedback path
            pass

        return False

    def _route_media_key(self, vk: int) -> bool:
        """Route media key to MediaController with fallback to passthrough.
        
        Returns True if the key was handled, False otherwise.
        """
        try:
            media_controller = get_media_controller()
            if not media_controller:
                return self._fallback_passthrough_media_key(vk)
            
            target_hwnd = self._target_hwnd if (_WIN_AVAILABLE and self._target_hwnd and _is_window(self._target_hwnd)) else None
            if not target_hwnd:
                # Strict policy: never send global media commands
                self._publish("key.passthrough.media_routed", {
                    "vk": int(vk),
                    "success": False,
                    "message": "No target hwnd; global media routing disabled",
                    "note": "media-no-target"
                })
                self._logger.warning(f"Media key VK_{vk:02X} ignored: no target hwnd; global media routing disabled by policy")
                return False

            # Validate window responsiveness before routing
            if not self._ensure_window_responsive(target_hwnd):
                if self._verbose:
                    self._logger.debug(f"Target window {target_hwnd} unresponsive, using fallback passthrough")
                # Emit explicit telemetry for unresponsive target without UI flash
                try:
                    self._publish("key.passthrough.blocked", {"reason": "unresponsive-target", "extra": {"hwnd": int(target_hwnd)}})
                except Exception:
                    pass
                return self._fallback_passthrough_media_key(vk)

            # Map VK codes to MediaController actions (local-only)
            if vk == _VK_MEDIA_PLAY_PAUSE:
                success, msg = media_controller.play_pause_for_hwnd(int(target_hwnd))
            elif vk == _VK_MEDIA_STOP:
                success, msg = media_controller.stop_for_hwnd(int(target_hwnd))
            elif vk == _VK_MEDIA_PREV_TRACK:
                success, msg = media_controller.previous_for_hwnd(int(target_hwnd))
            elif vk == _VK_MEDIA_NEXT_TRACK:
                success, msg = media_controller.next_for_hwnd(int(target_hwnd))
            else:
                return False
                
            self._publish("key.passthrough.media_routed", {
                "vk": int(vk), 
                "success": success, 
                "message": msg
            })
            
            if success:
                if self._verbose:
                    self._logger.debug(f"Media key VK_{vk:02X} routed successfully: {msg}")
                return True
            else:
                if self._verbose:
                    self._logger.debug(f"Media key VK_{vk:02X} routing failed, trying passthrough: {msg}")
                # Fallback to passthrough for apps like Firefox that handle media keys directly
                return self._fallback_passthrough_media_key(vk)
                
        except Exception as e:
            self._logger.error(f"Failed to route media key VK_{vk:02X}: {e}")
            return self._fallback_passthrough_media_key(vk)
    
    def _fallback_passthrough_media_key(self, vk: int) -> bool:
        """Fallback passthrough for media keys when MediaController routing fails.
        
        This allows apps like Firefox to receive media keys directly when the
        MediaController can't handle them properly.
        """
        if not self._enabled:
            return False
        if not _WIN_AVAILABLE:
            return False
        hwnd = self._target_hwnd
        if not (hwnd and _is_window(hwnd)):
            return False
            
        handled = self._enqueue_passthrough(vk)
        if handled:
            if self._verbose:
                self._logger.debug(f"Media key VK_{vk:02X} fallback to passthrough")
        return handled

    def _try_browser_child_hotkey(self, vk: int) -> bool:
        """Attempt to send a key to browser child content windows first when media routing is OFF.

        - Only triggers when the current target hwnd belongs to a recognized browser app.
        - Uses MediaController's browser hotkey path to prefer child windows (e.g., embedded video players).
        - Never issues global media commands; it only posts key messages (and WM_CHAR for Firefox when needed).
        """
        try:
            if not _WIN_AVAILABLE:
                return False

            hwnd = self._target_hwnd
            if not (hwnd and _is_window(hwnd)):
                return False

            media_controller = get_media_controller()
            if not media_controller:
                return False

            app_name = media_controller.detect_app_for_hwnd(int(hwnd))
            if app_name not in ("chrome", "edge", "firefox", "discord"):
                return False

            include_char = False
            char_code: Optional[int] = None
            # For Firefox, include WM_CHAR for printable keys like SPACE for better compatibility
            if app_name == "firefox" and vk == _VK_SPACE:
                include_char = True
                char_code = 0x20

            # Use MediaController's browser-aware hotkey path to target child windows first
            ok = False
            try:
                ok = media_controller._send_browser_hotkey(int(hwnd), int(vk), include_char=include_char, char_code=char_code)
            except Exception:
                ok = False

            if ok:
                self._publish("key.passthrough.forwarded", {"vk": int(vk), "hwnd": int(hwnd), "phase": "single", "note": "browser-child"})
                if self._verbose:
                    self._logger.debug(f"Browser-aware child hotkey accepted: app={app_name} vk={vk:02X} hwnd={hwnd}")
                return True

            # Not accepted by children/top-level via browser path; fall back to normal passthrough
            if self._verbose:
                self._logger.debug(f"Browser-aware child hotkey not accepted; falling back to normal passthrough: app={app_name} vk={vk:02X}")
            return False
        except Exception as e:
            if self._verbose:
                self._logger.debug(f"Browser-aware child hotkey path failed: {e}")
            return False

    def _enqueue_passthrough(self, vk: int) -> bool:
        """Enqueue a key for passthrough with rate limiting and keyup scheduling."""
        try:
            # Use the same rate limiting as regular passthrough
            self._pending_vk = int(vk)
            now = time.monotonic() * 1000.0
            elapsed = now - self._last_sent_ts
            delay = max(0, int(self._min_interval_ms - elapsed))
            if not self._send_scheduled:
                self._send_scheduled = True
                try:
                    ThreadManager.single_shot(delay, self._flush_send)
                except Exception:
                    # If scheduling fails, attempt immediate send
                    self._flush_send()
            return True
        except Exception as e:
            self._logger.error(f"Failed to enqueue passthrough for VK_{vk:02X}: {e}")
            return False

    def _prepare_window_for_media(self, hwnd: int) -> None:
        """Proactively prepare window for media commands by sending a benign message.
        
        This helps wake up dormant windows (like Firefox) that may not respond to
        media commands until they've been interacted with.
        """
        try:
            if not _WIN_AVAILABLE:
                return
            
            # Send WM_NULL (0x0000) - a benign message that doesn't affect the window
            # but ensures it's responsive to subsequent messages
            from utils.win.winmsg import safe_send_message
            if hasattr(_w32c, 'WM_NULL'):
                wm_null = _w32c.WM_NULL
            else:
                wm_null = 0x0000  # WM_NULL constant
            
            # Use a short timeout to avoid blocking
            success = safe_send_message(hwnd, wm_null, 0, 0, timeout_ms=100)
            if success:
                if self._verbose:
                    self._logger.debug(f"Prepared window {hwnd} for media commands")
            else:
                if self._verbose:
                    self._logger.debug(f"Window preparation failed for {hwnd} (non-critical)")
        except Exception as e:
            # Non-critical failure - log at debug level
            if self._verbose:
                self._logger.debug(f"Window preparation error for {hwnd}: {e}")

    def _ensure_window_responsive(self, hwnd: int) -> bool:
        """Check if window is responsive before sending media commands.
        
        Returns True if responsive, False if unresponsive or check fails.
        Uses a short timeout to avoid blocking the UI.
        """
        try:
            if not _WIN_AVAILABLE:
                return True  # Assume responsive on non-Windows
            
            from utils.win.winmsg import is_process_responsive
            # Use a short timeout (500ms) to avoid UI blocking
            return is_process_responsive(hwnd, timeout_ms=500)
        except Exception as e:
            self._logger.debug(f"Responsiveness check failed for {hwnd}: {e}")
            return True  # Assume responsive if check fails

    def _publish(self, event_type: str, data: dict) -> None:
        # Publish via ApplicationCore's EventSystem when available
        try:
            from core.application.core import get_app_core
            core = get_app_core()
            if core and hasattr(core, "events") and core.events:
                core.events.publish(event_type, data=data, source=self)
        except Exception:
            # EventSystem might not be initialized yet; ignore
            pass


# Convenience accessor
_def_instance: Optional[KeyPassthroughController] = None

def get_key_passthrough_controller() -> KeyPassthroughController:
    global _def_instance
    if _def_instance is None:
        _def_instance = KeyPassthroughController()
    return _def_instance
