"""
PipelineManager - Centralized DXGI capture pipeline wrapper (DXGI-only).

Provides a unified interface and signals equivalent to `MonitorCaptureManager`
so it can be used as a drop-in replacement by UI code. This simplified manager
always selects the DXGI Desktop Duplication pipeline via `MonitorCaptureManager`.

Notes:
- Windows Graphics Capture (WGC) and D3D11 swapchain presentation paths have
  been removed from selection; DXGI-only for robustness and low overhead.
- Registration with ResourceManager ensures central cleanup.
- Settings such as `capture.fps` are honored by the backend. The
  `graphics.pipeline` setting is accepted but only `dxgi` is supported; any
  other value will be coerced to `dxgi` with a log.
"""
from __future__ import annotations

from typing import Optional, Callable, Dict, Any
import time

from PySide6.QtCore import QObject, Signal, QSize

from core.logging import get_logger
from core.threading import get_thread_manager
from utils.resource_manager import get_resource_manager, ResourceType
from core.settings.settings_manager import SettingsManager

# Reuse existing capture frame type
from core.graphics.capture.monitor_capture_manager import (
    CaptureFrame,
    get_monitor_capture_manager,
    MonitorCaptureManager,
)

logger = get_logger(__name__)

# WGC pipeline removed from selection in this simplified manager.


class PipelineManager(QObject):
    """
    Central manager that selects the capture pipeline and forwards
    lifecycle and frames to consumers.

    Signals:
    - frame_captured(CaptureFrame)
    - capture_started()
    - capture_stopped()
    - capture_error(str)
    """

    # Signals mirror MonitorCaptureManager for drop-in usage
    frame_captured = Signal(object)  # CaptureFrame
    capture_started = Signal()
    capture_stopped = Signal()
    capture_error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._thread_manager = get_thread_manager()
        self._resource_manager = get_resource_manager()
        self._settings = SettingsManager()

        # Active pipeline and backend manager
        self._pipeline: Optional[str] = None
        self._backend: Optional[QObject] = None

        # Config
        self._target_monitor: Optional[Dict[str, Any]] = None
        self._frame_callback: Optional[Callable[[CaptureFrame], None]] = None
        # Remember pending capture rate until backend is bound
        self._pending_capture_rate: Optional[float] = None
        # Pending swapchain HWND (None means clear). Use a flag to distinguish "unset" vs "set to None"
        self._pending_swapchain_hwnd_value: Optional[int] = None
        self._pending_swapchain_hwnd_set: bool = False
        # Pending swapchain size (logical QSize) to forward to backend when available
        self._pending_swapchain_size: Optional[QSize] = None
        self._pending_swapchain_size_set: bool = False

        # Register for centralized cleanup
        self._resource_manager.register(
            self,
            ResourceType.CUSTOM,
            'PipelineManager instance',
            cleanup_handler=lambda obj: obj.cleanup(),
            tags={"manager", "graphics", "pipeline"},
        )

        logger.debug("PipelineManager initialized")

    # --- Public API (drop-in compatible) ---------------------------------
    def set_target_monitor(self, monitor_info: Dict[str, Any]) -> bool:
        if self.is_capturing():
            logger.warning("Cannot change monitor while capturing")
            return False
        self._target_monitor = monitor_info
        # Bind backend now so errors surface early
        backend = self._ensure_backend(bound=True)
        if backend is None:
            return False
        try:
            return backend.set_target_monitor(monitor_info)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Backend set_target_monitor failed: {e}", exc_info=True)
            return False

    def set_frame_callback(self, callback: Optional[Callable[[CaptureFrame], None]]) -> None:
        self._frame_callback = callback
        backend = self._ensure_backend(bound=False)
        if backend is not None:
            try:
                backend.set_frame_callback(callback)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_swapchain_target_hwnd(self, hwnd: Optional[int]) -> None:
        """Set or clear the target HWND for DXGI swapchain presentation on the backend.

        - Stored and applied when the backend is bound.
        - If the current backend doesn't support this method, it's a no-op.
        """
        try:
            self._pending_swapchain_hwnd_value = int(hwnd) if hwnd is not None else None
        except Exception:
            self._pending_swapchain_hwnd_value = None
        self._pending_swapchain_hwnd_set = True

        backend = self._ensure_backend(bound=False)
        if backend is not None:
            try:
                # type: ignore[attr-defined]
                backend.set_swapchain_target_hwnd(self._pending_swapchain_hwnd_value)  
            except Exception:
                # Backend may not support this (e.g., DXGI blitter); ignore.
                pass

    def set_swapchain_size(self, size: QSize) -> None:
        """Forward the logical swapchain host size to the active backend (DPI-aware backends will convert).

        Stored pending and applied upon backend bind. Backends that do not support
        swapchain size updates will ignore the call.
        """
        try:
            # Store as-is (logical QSize); DPI conversion is backend-specific
            self._pending_swapchain_size = size
            self._pending_swapchain_size_set = True
        except Exception:
            # If something is wrong with the object, clear pending
            self._pending_swapchain_size = None
            self._pending_swapchain_size_set = True

        backend = self._ensure_backend(bound=False)
        if backend is not None and self._pending_swapchain_size is not None:
            try:
                # type: ignore[attr-defined]
                backend.set_swapchain_size(self._pending_swapchain_size)
            except Exception:
                # Backend may not support this; ignore.
                pass

    def set_capture_rate(self, fps: float) -> None:
        """Set capture frame rate on the active backend if supported.

        Stored as pending and applied upon backend bind for drop-in compatibility.
        """
        try:
            self._pending_capture_rate = float(fps)
        except Exception:
            self._pending_capture_rate = None
        backend = self._ensure_backend(bound=False)
        if backend is not None and self._pending_capture_rate is not None:
            try:
                backend.set_capture_rate(self._pending_capture_rate)  # type: ignore[attr-defined]
            except Exception:
                # Backend may not support this; ignore.
                pass

    def start_capture(self) -> bool:
        backend = self._ensure_backend(bound=True)
        if backend is None:
            return False
        # Propagate callback in case it was set before binding
        if self._frame_callback is not None:
            try:
                backend.set_frame_callback(self._frame_callback)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            ok = backend.start_capture()  # type: ignore[attr-defined]
            return bool(ok)
        except Exception as e:
            logger.error(f"Failed to start capture via pipeline '{self._pipeline}': {e}", exc_info=True)
            self.capture_error.emit(str(e))
            return False

    def stop_capture(self) -> None:
        backend = self._backend
        if backend is None:
            return
        logger.debug(f"PipelineManager.stop_capture called for pipeline '{self._pipeline}'")
        try:
            backend.stop_capture()  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Backend stop_capture raised: {e}", exc_info=True)
        
        # Attempt to synchronize with backend capture thread/task to avoid teardown races
        # 1) If backend tracks a capture task id, join it with a bounded timeout
        try:
            task_id = getattr(backend, '_capture_task_id', None)
            if task_id:
                try:
                    self._thread_manager.get_task_result(task_id, timeout=2.5)
                    logger.debug(f"Capture task {task_id} joined successfully (pipeline '{self._pipeline}')")
                except KeyError:
                    logger.debug(f"Capture task {task_id} not found (already completed)")
                except TimeoutError:
                    logger.warning(f"Capture task {task_id} did not finish within timeout; proceeding")
                except Exception as e:
                    logger.debug(f"Error while joining capture task {task_id}: {e}", exc_info=True)
        except Exception:
            # Attribute access or thread manager issues should not break shutdown
            logger.debug("No capture task id available on backend for join")

        # 2) Bounded poll on backend.is_capturing() to ensure full stop is observed
        try:
            deadline = time.monotonic() + 2.5
            while True:
                try:
                    still = bool(getattr(backend, 'is_capturing') and backend.is_capturing())  # type: ignore[attr-defined]
                except Exception:
                    still = False
                if not still or time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            if still:
                logger.warning(f"Backend still reports capturing after stop timeout for pipeline '{self._pipeline}'")
            else:
                logger.debug("Backend reports not capturing after stop")
        except Exception:
            logger.debug("Error while polling backend capturing state", exc_info=True)

    def is_capturing(self) -> bool:
        backend = self._backend
        try:
            return bool(backend and backend.is_capturing())  # type: ignore[attr-defined]
        except Exception:
            return False

    def get_capture_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            'pipeline_selected': self._pipeline,
        }
        backend = self._backend
        try:
            if backend is not None:
                bstats = backend.get_capture_stats()  # type: ignore[attr-defined]
                if isinstance(bstats, dict):
                    stats.update(bstats)
        except Exception:
            pass
        return stats

    def reset_capture_stats(self) -> None:
        backend = self._backend
        try:
            if backend is not None:
                backend.reset_capture_stats()  # type: ignore[attr-defined]
        except Exception:
            pass

    def cleanup(self) -> None:
        logger.debug("Cleaning up PipelineManager")
        try:
            self.stop_capture()
        finally:
            self._disconnect_backend()
            self._backend = None

    # --- Diagnostics -----------------------------------------------------
    def probe_pipeline_availability(self, prefer: Optional[str] = None) -> Dict[str, Any]:
        """Probe the pipeline selection without starting capture (DXGI-only)."""
        requested = (prefer or self._settings.get('graphics.pipeline', 'dxgi')).strip().lower()
        # Coerce non-dxgi requests to dxgi with reason
        if requested != 'dxgi':
            reason = f"graphics.pipeline='{requested}' unsupported; using dxgi"
            requested = 'dxgi'
        else:
            reason = ''
        info = get_monitor_capture_manager().probe_backend_availability('dxgi')
        effective = 'dxgi' if info.get('effective') == 'dxgi' else None
        return {
            'requested': 'dxgi',
            'effective': effective,
            'reason': reason or info.get('reason', ''),
        }

    # --- Internal helpers ------------------------------------------------
    def _ensure_backend(self, bound: bool) -> Optional[QObject]:
        if self._backend is not None:
            return self._backend

        selected = str(self._settings.get('graphics.pipeline', 'dxgi')).strip().lower()
        if selected != 'dxgi':
            logger.info(f"graphics.pipeline='{selected}' unsupported; defaulting to 'dxgi'")
        self._pipeline = 'dxgi'

        backend = get_monitor_capture_manager()
        self._bind_backend(backend)

        # Apply previously set monitor
        if bound and self._target_monitor is not None:
            try:
                backend.set_target_monitor(self._target_monitor)  # type: ignore[attr-defined]
            except Exception:
                pass

        return self._backend

    def _bind_backend(self, backend: QObject) -> None:
        # Disconnect any previous
        self._disconnect_backend()
        self._backend = backend

        # Wire signals from backend to us
        try:
            backend.frame_captured.connect(self.frame_captured.emit)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            backend.capture_started.connect(self.capture_started.emit)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            backend.capture_stopped.connect(self.capture_stopped.emit)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            backend.capture_error.connect(self.capture_error.emit)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Apply pending configuration
        try:
            if self._frame_callback is not None:
                backend.set_frame_callback(self._frame_callback)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if self._pending_capture_rate is not None:
                backend.set_capture_rate(self._pending_capture_rate)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if self._pending_swapchain_hwnd_set:
                # Only attempt if previously set; ignore if backend lacks the API
                # type: ignore[attr-defined]
                backend.set_swapchain_target_hwnd(self._pending_swapchain_hwnd_value)
        except Exception:
            pass
        try:
            if self._pending_swapchain_size_set and self._pending_swapchain_size is not None:
                # Forward any pending logical size (backend will handle DPI conversion)
                # type: ignore[attr-defined]
                backend.set_swapchain_size(self._pending_swapchain_size)
        except Exception:
            pass

        logger.info(f"PipelineManager bound to pipeline: {self._pipeline}")

    def _disconnect_backend(self) -> None:
        b = self._backend
        if not b:
            return
        # Best-effort disconnect; ignore if signals not present
        try:
            b.frame_captured.disconnect(self.frame_captured.emit)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            b.capture_started.disconnect(self.capture_started.emit)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            b.capture_stopped.disconnect(self.capture_stopped.emit)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            b.capture_error.disconnect(self.capture_error.emit)  # type: ignore[attr-defined]
        except Exception:
            pass


# Singleton access
_pipeline_manager: Optional[PipelineManager] = None

def get_pipeline_manager() -> PipelineManager:
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager
