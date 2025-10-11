"""
MonitorCaptureManager - Screen capture via DXGI Desktop Duplication (dxcam).

Provides continuous screen capture functionality with proper thread management
and integration with the centralized ThreadManager system.

Notes:
- Backend: DXGI (dxcam) only. MSS (GDI) has been removed.
- Selection: env var SPQ_CAPTURE_BACKEND supports 'dxgi' or 'auto' only; 'mss' is unsupported.
"""

import time
import os
import threading
from typing import Optional, Dict, Any, Callable
import weakref
from dataclasses import dataclass

# MSS (GDI) backend removed

# Optional DXGI Desktop Duplication via dxcam
try:
    import dxcam  # type: ignore
    DXCAM_AVAILABLE = True
except Exception:
    DXCAM_AVAILABLE = False
    dxcam = None  # type: ignore

from PySide6.QtCore import QObject, Signal, QRect, QSize

from core.logging import get_logger
from core.threading import get_thread_manager
from utils.resource_manager import get_resource_manager, ResourceType
from utils.monitor_utils import get_all_monitors
from utils.frame_exchange import get_exchange
from core.settings import get_settings_manager

logger = get_logger(__name__)


@dataclass
class CaptureFrame:
    """Represents a captured frame with metadata.

    Notes:
    - image_data contains BGRA8888 or RGB888 bytes for CPU blit paths.
    - d3d11_tex_ptr, when non-None, may carry an ID3D11Texture2D pointer value
      for zero-copy GPU presentation (Phase 2). Consumers must treat it as
      optional and fall back to image_data when absent.
    """
    image_data: bytes
    width: int
    height: int
    timestamp: float
    monitor_index: int
    d3d11_tex_ptr: int | None = None

class MonitorCaptureManager(QObject):
    """
    Manages DXGI-based screen capture with proper threading and resource management.
    
    Features:
    - Continuous capture loop on worker thread
    - Configurable capture rate based on monitor refresh rate
    - Proper resource cleanup via ResourceManager
    - Thread-safe frame delivery
    - Monitor selection and validation
    """
    
    # Signals
    frame_captured = Signal(object)  # CaptureFrame
    capture_started = Signal()
    capture_stopped = Signal()
    capture_error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._thread_manager = get_thread_manager()
        self._resource_manager = get_resource_manager()
        
        # Capture state
        self._is_capturing = False
        # Async producer (CAPTURE pool) task id
        self._capture_task_id: Optional[str] = None
        self._stop_event = threading.Event()
        
        # Backend instances
        self._dxcam_camera = None
        
        # Capture settings
        self._target_monitor = None
        self._capture_rate = 30.0  # FPS
        self._frame_callback: Optional[Callable[[CaptureFrame], None]] = None
        # UI emission throttle (ms). Derived from capture_rate; min 8ms (~125Hz)
        self._ui_emit_min_interval_ms: int = 33
        self._last_ui_emit_ts: float = 0.0

        # Backend selection
        self._backend_requested: str = os.getenv("SPQ_CAPTURE_BACKEND", "auto").strip().lower()
        self._backend_effective: Optional[str] = None
        # Central pipeline selection (settings: graphics.pipeline) is handled in PipelineManager;
        # this manager implements the DXGI-only path

        # Swapchain presentation (no-op in this DXGI capture manager; stored for compatibility)
        self._swapchain_target_hwnd: Optional[int] = None
        self._swapchain_size_logical: Optional[QSize] = None
        
        # Performance tracking
        self._last_capture_time = 0.0
        self._frame_count = 0
        # Leak/instrumentation counters
        self._frames_created: int = 0
        self._frames_finalized: int = 0
        self._bytes_captured_total: int = 0
        # Timing instrumentation throttle
        self._last_timing_log_ts: float = 0.0
        
        # Register for centralized cleanup via core.resources API
        self._resource_manager.register(
            self,
            ResourceType.CUSTOM,
            'MonitorCaptureManager instance',
            cleanup_handler=lambda obj: obj.cleanup(),
            tags={"manager", "capture"}
        )
        
        logger.debug("MonitorCaptureManager initialized")

        # FrameExchange (TripleBuffer) for latest-frame handoff (MON-1)
        self._fx_name: str = "monitor_frames"
        self._fx = get_exchange(self._fx_name)
        try:
            # Configure drop behavior to release large payloads immediately
            self._fx.set_on_drop(self._on_drop_frame)
            # Initialize coalescing window based on default FPS
            try:
                min_pub = max(0.0, (1.0 / float(self._capture_rate)) * 0.5)
                self._fx.set_min_publish_interval(min_pub)
            except Exception:
                pass
        except Exception:
            logger.debug("FrameExchange set_on_drop unsupported", exc_info=True)

        # Apply FPS from settings and watch for changes
        try:
            self._settings = get_settings_manager()
            fps_val = int(self._settings.get('capture.fps', 30))
            self.set_capture_rate(float(fps_val))

            def _on_setting_changed(key: str, value: Any) -> None:
                if key == 'capture.fps':
                    try:
                        self.set_capture_rate(float(int(value)))
                        logger.info(f"Applied capture FPS from settings: {value}")
                    except Exception as e:
                        logger.warning(f"Invalid capture.fps value '{value}': {e}")

            # Optional backend request from settings (DXGI only; 'mss' unsupported)
            try:
                be = str(self._settings.get('capture.backend', self._backend_requested or 'auto')).strip().lower()
                if be:
                    self._backend_requested = be
            except Exception:
                pass

            # Pipeline selection handled centrally (no-op here)

            # Lightweight handler via register_change_handler for specific key
            self._settings.register_change_handler('capture.fps', lambda k, v: _on_setting_changed(k, v))
        except Exception:
            logger.debug("SettingsManager unavailable or error reading capture.fps", exc_info=True)
    
    def set_target_monitor(self, monitor_info: Dict[str, Any]) -> bool:
        """
        Set the target monitor for capture.
        
        Args:
            monitor_info: Monitor information dict from monitor_utils
            
        Returns:
            bool: True if monitor was set successfully
        """
        if self._is_capturing:
            logger.warning("Cannot change monitor while capturing")
            return False
            
        try:
            # Validate monitor info
            if not monitor_info or 'rect' not in monitor_info:
                logger.error("Invalid monitor info provided")
                return False
                
            rect = monitor_info['rect']
            if not isinstance(rect, QRect) or rect.isEmpty():
                logger.error("Invalid monitor rect")
                return False
                
            self._target_monitor = monitor_info
            # Prefer physical dimensions in logs to avoid DPI confusion
            phys_w = int(monitor_info.get('physical_width', rect.width()))
            phys_h = int(monitor_info.get('physical_height', rect.height()))
            pos = monitor_info.get('position', rect.topLeft())
            logger.info(f"Target monitor set: {phys_w}x{phys_h} at {pos.x()},{pos.y()}")
            # Update exchange name per monitor to support multi-monitor scenarios
            mon_idx = monitor_info.get('qt_index', 0)
            self._fx_name = f"monitor_frames_{mon_idx}"
            self._fx = get_exchange(self._fx_name)
            try:
                self._fx.set_on_drop(self._on_drop_frame)
                # Re-apply coalescing policy on new exchange instance
                try:
                    min_pub = max(0.0, (1.0 / float(self._capture_rate)) * 0.5)
                    self._fx.set_min_publish_interval(min_pub)
                except Exception:
                    pass
            except Exception:
                logger.debug("FrameExchange set_on_drop unsupported (retarget)", exc_info=True)
            return True
            
        except Exception as e:
            logger.error(f"Error setting target monitor: {e}", exc_info=True)
            return False
    
    def set_capture_rate(self, fps: float) -> None:
        """
        Set the capture frame rate.
        
        Args:
            fps: Target frames per second (1.0 - 120.0)
        """
        fps = max(1.0, min(120.0, fps))
        self._capture_rate = fps
        # Update UI throttle window to approximately one frame at target FPS
        self._ui_emit_min_interval_ms = max(8, int(1000.0 / self._capture_rate))
        logger.debug(f"Capture rate set to {fps} FPS; ui_emit_min_interval_ms={self._ui_emit_min_interval_ms}")
        # Configure FrameExchange coalescing interval to reduce churn
        try:
            # Coalesce publishes closer than half a frame interval
            min_pub = max(0.0, (1.0 / float(self._capture_rate)) * 0.5)
            if self._fx is not None:
                try:
                    self._fx.set_min_publish_interval(min_pub)
                except Exception:
                    pass
        except Exception:
            logger.debug("Failed to set FrameExchange coalescing interval", exc_info=True)
    
    def set_frame_callback(self, callback: Optional[Callable[[CaptureFrame], None]]) -> None:
        """
        Set callback for captured frames.
        
        Args:
            callback: Function to call with each captured frame
        """
        self._frame_callback = callback

    def set_swapchain_target_hwnd(self, hwnd: Optional[int]) -> None:
        """No-op compatibility method for PipelineManager forwarding.

        Stored locally to support potential future integration, but this manager
        does not perform swapchain presentation directly.
        """
        try:
            self._swapchain_target_hwnd = int(hwnd) if hwnd is not None else None
        except Exception:
            self._swapchain_target_hwnd = None

    def set_swapchain_size(self, size: QSize) -> None:
        """No-op compatibility method for PipelineManager forwarding.

        Stores the logical size; DPI-aware translation, if needed, would be handled
        by a presenter backend. This capture manager ignores it.
        """
        try:
            self._swapchain_size_logical = size
        except Exception:
            self._swapchain_size_logical = None

    def probe_backend_availability(self, prefer: Optional[str] = None) -> Dict[str, Any]:
        """Probe which backend would be selected without starting capture.

        Args:
            prefer: Optional preferred backend string ("dxgi" or "auto").
                    If None, uses current settings/env (`self._backend_requested`).

        Returns:
            Dict with keys: 'requested', 'effective', 'reason'.
        """
        requested = (prefer or self._backend_requested or "auto").strip().lower()
        effective: Optional[str]
        reason = ""
        if requested in ("dxgi", "dxcam", "desktopdup"):
            if DXCAM_AVAILABLE:
                effective = "dxgi"
                reason = "dxcam present"
            else:
                effective = None
                reason = "dxcam not available"
        else:  # auto
            if DXCAM_AVAILABLE:
                effective = "dxgi"
                reason = "auto: dxcam preferred and present"
            else:
                effective = None
                reason = "auto: no backend available"

        return {
            'requested': requested,
            'effective': effective,
            'reason': reason,
        }
    
    def start_capture(self) -> bool:
        """
        Start continuous screen capture.
        
        Returns:
            bool: True if capture started successfully
        """
        if self._is_capturing:
            logger.warning("Capture already running")
            return True
            
        if not self._target_monitor:
            logger.error("No target monitor set")
            self.capture_error.emit("No target monitor set")
            return False
        
        # Pipeline handled by PipelineManager; this manager always runs DXGI
            
        # Resolve backend selection (DXGI only)
        requested = (self._backend_requested or "auto").lower()
        effective = None
        if requested in ("dxgi", "dxcam", "desktopdup"):
            if DXCAM_AVAILABLE:
                effective = "dxgi"
            else:
                msg = "dxcam (DXGI) not available; no capture backend present"
                logger.error(msg)
                self.capture_error.emit(msg)
                return False
        else:  # auto
            if DXCAM_AVAILABLE:
                effective = "dxgi"
            else:
                msg = "No capture backend available (dxcam missing)"
                logger.error(msg)
                self.capture_error.emit(msg)
                return False

        self._backend_effective = effective
        logger.info(f"Monitor capture starting with backend: {self._backend_effective.upper()} (requested='{requested}')")

        try:
            # Reset state
            self._stop_event.clear()
            self._is_capturing = True
            self._frame_count = 0
            self._last_ui_emit_ts = 0.0

            # Start capture on ThreadManager CAPTURE pool (async producer)
            # DXGI camera will be created on the worker thread to avoid thread-local issues
            def _run_loop():
                self._capture_loop()

            with self._thread_manager.capture_context() as ctx:
                self._capture_task_id = ctx.submit_capture(
                    _run_loop,
                    task_id=f"monitor_capture_{id(self)}",
                    resource_tags={"capture", "monitor"},
                )

            logger.info("Monitor capture started (CAPTURE pool)")
            self.capture_started.emit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to start capture: {e}", exc_info=True)
            self._is_capturing = False
            self.capture_error.emit(f"Failed to start capture: {e}")
            return False
    
    def stop_capture(self) -> None:
        """Stop screen capture."""
        if not self._is_capturing:
            return
            
        logger.debug("Stopping monitor capture (requesting thread exit)")
        self._is_capturing = False
        self._stop_event.set()
        # Explicitly wait for the capture task to finish to avoid teardown races
        if self._capture_task_id:
            task_id = self._capture_task_id
            try:
                self._thread_manager.get_task_result(task_id, timeout=2.0)
                logger.debug(f"Capture task {task_id} joined successfully")
            except KeyError:
                # Task already completed or not tracked
                logger.debug(f"Capture task {task_id} not found (already completed)")
            except TimeoutError:
                logger.warning(f"Capture task {task_id} did not finish within timeout; continuing cleanup")
            except Exception as e:
                logger.debug(f"Error while joining capture task {task_id}: {e}", exc_info=True)
            finally:
                self._capture_task_id = None
        
        # Cleanup dxcam
        if self._dxcam_camera is not None:
            try:
                self._dxcam_camera.stop()
            except Exception:
                pass
            finally:
                self._dxcam_camera = None
        
        logger.info("Monitor capture stopped")
        self.capture_stopped.emit()
        # Drop retained references in the exchange to assist GC
        try:
            self._fx.clear()
        except Exception:
            logger.debug("FrameExchange clear unsupported", exc_info=True)
    
    def _capture_loop(self) -> None:
        """Main capture loop running on CAPTURE pool worker thread."""
        frame_interval = 1.0 / self._capture_rate
        backend = self._backend_effective

        # DXGI Desktop Duplication path (dxcam)
        if backend == "dxgi":
            camera = None
            try:
                mon_idx = int(self._target_monitor.get('qt_index', 0)) if isinstance(self._target_monitor, dict) else 0
                # Create dxcam camera for the specified output. Prefer BGRA to match GL uploader expectations.
                camera = dxcam.create(output_idx=mon_idx, output_color='BGRA')  # type: ignore
                self._dxcam_camera = camera
                try:
                    camera.start(target_fps=int(self._capture_rate))
                except Exception:
                    # If start with target fps not supported, fall back to default start
                    camera.start()

                logger.debug(f"DXGI capture loop starting on output_idx={mon_idx}")
                last_log_ts = 0.0
                while self._is_capturing and not self._stop_event.is_set():
                    loop_start = time.perf_counter()
                    try:
                        t_cap0 = time.perf_counter()
                        frame_np = camera.get_latest_frame()
                        t_cap1 = time.perf_counter()
                        if frame_np is None:
                            # No frame yet; sleep briefly
                            time.sleep(min(0.002, frame_interval))
                            continue

                        h, w = int(frame_np.shape[0]), int(frame_np.shape[1])
                        # Copy to bytes for downstream consumers; ensure release of numpy buffer
                        t_copy0 = time.perf_counter()
                        try:
                            rgb_data = frame_np.tobytes(order='C')
                        finally:
                            # Promptly release numpy buffer to reduce memory pressure
                            try:
                                del frame_np
                            except Exception:
                                pass
                        t_copy1 = time.perf_counter()

                        frame = CaptureFrame(
                            image_data=rgb_data,
                            width=w,
                            height=h,
                            timestamp=time.time(),
                            monitor_index=self._target_monitor.get('qt_index', 0)
                        )
                        self._frames_created += 1
                        try:
                            self._bytes_captured_total += len(rgb_data)
                        except Exception:
                            pass
                        try:
                            weakref.finalize(frame, self._on_frame_finalized)
                        except Exception:
                            pass

                        try:
                            self._fx.publish(frame)
                        except Exception:
                            logger.debug("FrameExchange publish failed", exc_info=True)

                        # Deliver to UI (throttled) via signal and optional callback
                        try:
                            now_ms = time.perf_counter() * 1000.0
                            if (now_ms - self._last_ui_emit_ts) >= self._ui_emit_min_interval_ms:
                                self._last_ui_emit_ts = now_ms

                                # Emit signal (queued across threads)
                                try:
                                    self.frame_captured.emit(frame)
                                except Exception:
                                    logger.debug("frame_captured.emit failed", exc_info=True)

                                # Invoke callback if provided (compatibility)
                                cb = self._frame_callback
                                if cb is not None:
                                    try:
                                        cb(frame)
                                    except Exception as e:
                                        logger.debug(f"frame callback raised: {e}", exc_info=True)
                        except Exception:
                            logger.debug("UI emission block failed", exc_info=True)

                        # Throttled timing log (~1s)
                        now = time.perf_counter()
                        if (now - last_log_ts) >= 1.0:
                            cap_ms = (t_cap1 - t_cap0) * 1000.0
                            copy_ms = (t_copy1 - t_copy0) * 1000.0
                            logger.info(
                                f"DXGI timings: dup_ms={cap_ms:.2f} copy_ms={copy_ms:.2f} size={w}x{h} bytes={len(rgb_data)}"
                            )
                            last_log_ts = now

                        self._frame_count += 1
                        elapsed = time.perf_counter() - loop_start
                        sleep_time = frame_interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    except Exception as e:
                        logger.error(f"Error in DXGI capture loop: {e}", exc_info=True)
                        break
            except Exception as e:
                logger.error(f"Fatal error initializing DXGI (dxcam) capture: {e}", exc_info=True)
            finally:
                try:
                    if camera is not None:
                        camera.stop()
                except Exception:
                    pass
                self._dxcam_camera = None
                self._is_capturing = False
                logger.debug(f"DXGI capture loop ended. Total frames: {self._frame_count}")
            return

        # No capture backend path; DXGI handled above. If we arrive here, stop state and log.
        self._is_capturing = False
        logger.debug(f"Capture loop ended. Total frames: {self._frame_count}")
        
    def _on_drop_frame(self, frame: CaptureFrame) -> None:
        """Reduce memory of a frame that is about to be dropped/coalesced.

        This helps ensure large byte buffers are released promptly even if
        the object remains referenced transiently.
        """
        try:
            if hasattr(frame, 'image_data') and isinstance(frame.image_data, (bytes, bytearray, memoryview)):
                frame.image_data = b''
        except Exception:
            pass
    
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        return self._is_capturing
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        return {
            'is_capturing': self._is_capturing,
            'frame_count': self._frame_count,
            'capture_rate': self._capture_rate,
            'target_monitor': self._target_monitor.get('device_name', 'Unknown') if self._target_monitor else None,
            # Instrumentation
            'frames_created': self._frames_created,
            'frames_finalized': self._frames_finalized,
            'bytes_captured_total': self._bytes_captured_total,
            # Backend diagnostics
            'backend_requested': self._backend_requested,
            'backend_effective': self._backend_effective,
        }

    def reset_capture_stats(self) -> None:
        """Reset instrumentation counters to measure fresh after a fix."""
        self._last_capture_time = 0.0
        self._frame_count = 0
        self._frames_created = 0
        self._frames_finalized = 0
        self._bytes_captured_total = 0

    def _on_frame_finalized(self) -> None:
        """Weakref finalizer callback to count when frames are GC'd."""
        try:
            self._frames_finalized += 1
        except Exception:
            pass
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.debug("Cleaning up MonitorCaptureManager")
        self.stop_capture()

# Singleton instance
_monitor_capture_manager = None

def get_monitor_capture_manager() -> MonitorCaptureManager:
    """Get the singleton MonitorCaptureManager instance."""
    global _monitor_capture_manager
    if _monitor_capture_manager is None:
        _monitor_capture_manager = MonitorCaptureManager()
    return _monitor_capture_manager
