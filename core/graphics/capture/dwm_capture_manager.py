"""
DwmCaptureManager - DWM-based window capture pipeline with threading integration.

Runs a background capture loop on ThreadManager's CAPTURE pool that:
- Validates a source window (HWND)
- Publishes latest source content-rect metadata to a FrameExchange (TripleBuffer)
- Integrates with ThumbnailManager to register/update/unregister a DWM thumbnail
- Throttles UI signals via a UI coalescer and ThreadManager UI dispatch utilities
- Registers resources with ResourceManager for deterministic cleanup

This manager focuses on exchanging lightweight metadata (rects/timestamps) and
thumbnail lifecycle – no pixel copies. Renderers/overlays can consume the
FrameExchange to react to geometry changes and the ThumbnailManager drives DWM.
"""
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from PySide6.QtCore import QObject, Signal

from core.logging import get_logger
from core.threading import get_thread_manager
from utils.resource_manager import get_resource_manager, ResourceType
from utils.frame_exchange import get_exchange, FrameExchange
from utils.window.thumbnail_manager import ThumbnailManager
from core.window.validation import is_valid_window, get_window_rect

logger = get_logger(__name__)


@dataclass
class DwmContentRect:
    """Lightweight content-rect metadata for a source window."""
    hwnd_src: int
    rect: Tuple[int, int, int, int]  # (l, t, r, b)
    timestamp: float


class DwmCaptureManager(QObject):
    """
    Manages DWM thumbnail lifecycle and window-rect polling on the CAPTURE pool.

    Features:
    - Asynchronous capture loop using ThreadManager.capture_context
    - Latest-value exchange via FrameExchange (TripleBuffer)
    - DWM thumbnail registration/update/unregistration via ThumbnailManager
    - UI notifications with throttling and coalescing
    - Centralized ResourceManager registrations & cleanup
    """

    # Signals (emitted on UI thread via ThreadManager dispatch/coalescer)
    content_rect_updated = Signal(object)  # DwmContentRect
    capture_started = Signal()
    capture_stopped = Signal()
    capture_error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._tm = get_thread_manager()
        self._rm = get_resource_manager()

        # State
        self._is_running: bool = False
        self._stop_event = threading.Event()
        self._capture_task_id: Optional[str] = None
        self._hwnd_src: Optional[int] = None
        self._hwnd_dest: Optional[int] = None
        self._last_rect: Optional[Tuple[int, int, int, int]] = None
        self._thumb_registered: bool = False

        # UI emission throttling (ms)
        self._ui_emit_min_interval_ms: int = 16
        self._last_ui_emit_ts: float = 0.0

        # Thumbnail manager (centralized DWM calls)
        self._thumb_mgr = ThumbnailManager()

        # FrameExchange for latest content rect
        self._fx_name: str = "dwm_content_rects"
        self._fx: FrameExchange[DwmContentRect] = get_exchange(self._fx_name)

        # UI coalescer for batching UI work (thumbnail updates and signals)
        self._uiq = self._tm.create_ui_coalescer(
            name="dwm_capture_updates", capacity=128, mode="latest", window_ms=7
        )

        # Register for centralized cleanup
        self._rm.register(
            self,
            ResourceType.CUSTOM,
            "DwmCaptureManager instance",
            cleanup_handler=lambda obj: obj.cleanup(),
            tags={"manager", "capture", "dwm"},
        )

        logger.debug("DwmCaptureManager initialized")

    # Configuration -----------------------------------------------------------
    def set_source_window(self, hwnd_src: int) -> bool:
        if self._is_running:
            logger.warning("Cannot change source while running")
            return False
        try:
            if not isinstance(hwnd_src, int) or hwnd_src <= 0:
                return False
            if not is_valid_window(hwnd_src, our_pid=os.getpid(), check_visible=True):
                logger.error("Invalid source window: %s", hwnd_src)
                return False
            self._hwnd_src = hwnd_src
            # Use a per-source exchange name to avoid contention
            self._fx_name = f"dwm_content_rects_{hwnd_src}"
            self._fx = get_exchange(self._fx_name)
            logger.info("DWM source window set: hwnd=%s", hwnd_src)
            return True
        except Exception as e:
            logger.error("Error setting source window: %s", e, exc_info=True)
            return False

    def set_destination_window(self, hwnd_dest: int) -> bool:
        if self._is_running:
            logger.warning("Cannot change destination while running")
            return False
        try:
            if not isinstance(hwnd_dest, int) or hwnd_dest <= 0:
                return False
            self._hwnd_dest = hwnd_dest
            logger.info("DWM destination window set: hwnd=%s", hwnd_dest)
            return True
        except Exception as e:
            logger.error("Error setting destination window: %s", e, exc_info=True)
            return False

    def set_ui_emit_interval_ms(self, interval_ms: int) -> None:
        self._ui_emit_min_interval_ms = max(8, int(interval_ms))

    # Lifecycle ---------------------------------------------------------------
    def start(self) -> bool:
        if self._is_running:
            logger.warning("DWM capture already running")
            return True
        if not self._hwnd_src:
            msg = "No source window set"
            logger.error(msg)
            self.capture_error.emit(msg)
            return False

        try:
            # Register thumbnail if a destination is provided
            if self._hwnd_dest and not self._thumb_registered:
                th = self._thumb_mgr.register_thumbnail(self._hwnd_dest, self._hwnd_src)
                self._thumb_registered = th is not None
                if not self._thumb_registered:
                    logger.error("Failed to register DWM thumbnail (dest=%s src=%s)", self._hwnd_dest, self._hwnd_src)

            # Reset counters/state
            self._stop_event.clear()
            self._is_running = True
            self._last_rect = None
            self._last_ui_emit_ts = 0.0

            # Submit capture loop on CAPTURE pool
            def _run_loop():
                self._capture_loop()

            with self._tm.capture_context() as ctx:
                self._capture_task_id = ctx.submit_capture(
                    _run_loop,
                    task_id=f"dwm_capture_{id(self)}",
                    resource_tags={"capture", "dwm"},
                )

            logger.info("DWM capture started (CAPTURE pool)")
            self.capture_started.emit()
            return True
        except Exception as e:
            logger.error("Failed to start DWM capture: %s", e, exc_info=True)
            self._is_running = False
            self.capture_error.emit(f"Failed to start DWM capture: {e}")
            return False

    def stop(self) -> None:
        if not self._is_running:
            return
        logger.debug("Stopping DWM capture")
        self._is_running = False
        self._stop_event.set()
        if self._capture_task_id:
            try:
                self._tm.cancel_task(self._capture_task_id)
            except Exception:
                pass
            finally:
                self._capture_task_id = None

        # Unregister thumbnail if we registered one
        if self._hwnd_dest and self._thumb_registered:
            try:
                self._thumb_mgr.unregister_thumbnail(self._hwnd_dest)
            except Exception:
                logger.debug("Thumbnail unregister failed", exc_info=True)
            finally:
                self._thumb_registered = False

        logger.info("DWM capture stopped")
        self.capture_stopped.emit()

    # Loop --------------------------------------------------------------------
    def _capture_loop(self) -> None:
        """Runs on the CAPTURE pool; polls content rect, publishes, and coalesces UI updates."""
        try:
            while self._is_running and not self._stop_event.is_set():
                loop_start = time.perf_counter()

                # Validate and get rect
                try:
                    if not self._hwnd_src or not is_valid_window(self._hwnd_src, our_pid=os.getpid(), check_visible=True):
                        raise RuntimeError("Source window became invalid or invisible")

                    rect = get_window_rect(self._hwnd_src)
                    if not rect:
                        raise RuntimeError("Failed to get source window rect")

                    if rect != self._last_rect:
                        self._last_rect = rect
                        meta = DwmContentRect(hwnd_src=self._hwnd_src, rect=rect, timestamp=time.time())
                        try:
                            self._fx.publish(meta)
                        except Exception:
                            logger.debug("FrameExchange publish failed", exc_info=True)

                        # UI throttled emission and thumbnail update
                        now_ms = time.perf_counter() * 1000.0
                        if (now_ms - self._last_ui_emit_ts) >= self._ui_emit_min_interval_ms:
                            self._last_ui_emit_ts = now_ms
                            # Enqueue UI work via coalescer
                            self._uiq.submit(lambda m=meta: self._emit_rect_and_update_thumbnail(m))

                    # Pace the loop lightly to avoid CPU burn. Rect polling is cheap.
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = max(0.0, 0.005 - elapsed)  # ~200Hz cap
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    logger.error("DWM capture loop error: %s", e, exc_info=True)
                    # Dispatch error signal on UI
                    self._tm.run_on_ui_thread(self.capture_error.emit, f"DWM capture error: {e}")
                    break
        except Exception as e:
            logger.error("Fatal DWM capture error: %s", e, exc_info=True)
            self._tm.run_on_ui_thread(self.capture_error.emit, f"Fatal DWM capture error: {e}")
        finally:
            self._is_running = False
            logger.debug("DWM capture loop exited")

    # UI-thread work ----------------------------------------------------------
    def _emit_rect_and_update_thumbnail(self, meta: DwmContentRect) -> None:
        """Runs on UI thread: emit signal and update thumbnail properties if applicable."""
        try:
            # Emit content rect update for UI consumers
            try:
                self.content_rect_updated.emit(meta)
            except Exception:
                logger.debug("content_rect_updated emit failed", exc_info=True)

            # Update thumbnail destination/source rectangles if we have a thumb
            if self._hwnd_dest and self._thumb_registered:
                try:
                    l, t, r, b = meta.rect
                    # Use the src rect as full window bounds; dest rect defaults to same size
                    self._thumb_mgr.update_thumbnail(
                        self._hwnd_dest,
                        dest_rect=(l, t, r, b),
                        src_rect=(l, t, r, b),
                        visible=True,
                        source_client_area_only=False,
                    )
                except Exception:
                    logger.debug("Thumbnail update failed", exc_info=True)
        except Exception as e:
            logger.exception("_emit_rect_and_update_thumbnail raised: %s", e)

    # Introspection -----------------------------------------------------------
    def is_running(self) -> bool:
        return self._is_running

    def exchange_name(self) -> str:
        return self._fx_name

    def cleanup(self) -> None:
        logger.debug("Cleaning up DwmCaptureManager")
        try:
            self.stop()
        finally:
            try:
                # UI coalescer is registered with ResourceManager; request a flush
                self._uiq.flush()
            except Exception:
                pass


# Singleton instance
_dwm_capture_manager: Optional[DwmCaptureManager] = None


def get_dwm_capture_manager() -> DwmCaptureManager:
    global _dwm_capture_manager
    if _dwm_capture_manager is None:
        _dwm_capture_manager = DwmCaptureManager()
    return _dwm_capture_manager
