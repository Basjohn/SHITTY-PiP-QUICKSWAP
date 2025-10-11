"""
MonitorBackend - Backend implementation for monitor capture overlays.

Implements the `Overlay` abstract base with a Qt host widget that renders
monitor capture content via `ui/overlays/monitor/monitor_overlay.py`.
"""

from typing import Optional, Dict, Any

from core.logging import get_logger
from core.threading import get_thread_manager
from core.graphics.types import OverlayConfig
from core.graphics.overlay import Overlay as OverlayBase
from ui.overlays.monitor.monitor_overlay import MonitorOverlay
from utils.monitor_utils import get_all_monitors

logger = get_logger(__name__)


class MonitorBackend(OverlayBase):
    """
    Backend for monitor capture overlays.

    Lifecycle mirrors `SoftwareOverlay` by hosting a top-level QWidget and
    exposing it via `self._host` for OverlayManager z-order registration.
    """

    def __init__(self, config: OverlayConfig):
        super().__init__(config)
        self._thread_manager = get_thread_manager()

        # Host widget and state
        self._host: Optional[MonitorOverlay] = None
        self._target_monitor: Optional[Dict[str, Any]] = None

        logger.debug("MonitorBackend constructed")

    # ---- OverlayBase required implementation hooks -------------------------
    def _initialize_impl(self) -> None:
        # Validate monitor target from properties
        props = self._config.properties or {}
        monitor_target = props.get("monitor_target")
        if monitor_target is None:
            raise ValueError("MonitorBackend requires 'monitor_target' in properties")

        target_monitor = self._find_monitor_by_target(monitor_target)
        if not target_monitor:
            raise RuntimeError(f"Target monitor not found: {monitor_target}")

        # Create and configure host widget
        self._host = MonitorOverlay()
        # Store a back-reference for cleanup compatibility if needed by others
        self._host._parent_overlay = self  # noqa: SLF001
        
        # Propagate app_instance if available (set by OverlayManager)
        if hasattr(self, '_app_instance_for_host'):
            self._host.app_instance = self._app_instance_for_host
            logger.debug("Propagated app_instance to MonitorOverlay host widget")

        if not self._host.set_target_monitor(target_monitor):
            # Ensure deletion on failure
            try:
                self._host.deleteLater()
            except Exception:
                pass
            self._host = None
            raise RuntimeError("Failed to set target monitor on MonitorOverlay")

        self._target_monitor = target_monitor

        # Apply initial geometry and opacity
        try:
            self._host.setGeometry(self._config.position.x(), self._config.position.y(),
                                   self._config.size.width(), self._config.size.height())
        except Exception:
            # Fallback to resize/move if direct setGeometry fails
            try:
                self._host.move(self._config.position)
                self._host.resize(self._config.size)
            except Exception:
                pass

        try:
            self._host.setWindowOpacity(float(self._config.opacity))
        except Exception:
            pass

        logger.info("MonitorBackend initialized for monitor: %s",
                    target_monitor.get("device_name", "Unknown"))

    def _show_impl(self) -> None:
        if self._host:
            self._host.show()

    def _hide_impl(self) -> None:
        if self._host:
            # Stop capture when hidden is handled by the widget itself
            self._host.hide()

    def _close_impl(self) -> None:
        # Ensure host is torn down
        if self._host:
            try:
                self._host.close()
            except Exception:
                pass
            try:
                self._host.deleteLater()
            except Exception:
                pass
            self._host = None
        self._target_monitor = None

    def _render_impl(self) -> None:
        # Rendering/capture is managed internally by MonitorOverlay
        # when visible; nothing to do here.
        pass

    def _config_updated(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        if not self._host:
            return
        # Geometry updates
        try:
            pos_changed = old_config.get("position") != new_config.get("position")
            size_changed = old_config.get("size") != new_config.get("size")
            if pos_changed or size_changed:
                # Use OverlayBase helpers if needed, but here apply directly to host
                self._host.setGeometry(self._config.position.x(), self._config.position.y(),
                                       self._config.size.width(), self._config.size.height())
        except Exception:
            pass

        # Title
        try:
            if old_config.get("title") != new_config.get("title"):
                self._host.setWindowTitle(self._config.title)
        except Exception:
            pass

        # Opacity
        try:
            if old_config.get("opacity") != new_config.get("opacity"):
                self._host.setWindowOpacity(float(self._config.opacity))
        except Exception:
            pass

    # ---- Helpers ------------------------------------------------------------
    def _find_monitor_by_target(self, monitor_target: Any) -> Optional[Dict[str, Any]]:
        """
        Find monitor by target specification.
        
        Args:
            monitor_target: Monitor target (index, name, or monitor dict)
            
        Returns:
            Monitor info dict or None if not found
        """
        try:
            monitors = get_all_monitors()
            if not monitors:
                logger.warning("No monitors available")
                return None
            
            # If target is already a monitor dict, validate and return
            if isinstance(monitor_target, dict) and 'rect' in monitor_target:
                return monitor_target
            
            # If target is an index
            if isinstance(monitor_target, int):
                if 0 <= monitor_target < len(monitors):
                    return monitors[monitor_target]
                else:
                    logger.error(f"Monitor index {monitor_target} out of range (0-{len(monitors)-1})")
                    return None
            
            # If target is a device name
            if isinstance(monitor_target, str):
                for monitor in monitors:
                    if monitor.get('device_name', '') == monitor_target:
                        return monitor
                logger.error(f"Monitor with device name '{monitor_target}' not found")
                return None
            
            logger.error(f"Invalid monitor target type: {type(monitor_target)}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding monitor by target: {e}", exc_info=True)
            return None

    # Optional helper APIs used by callers/tests
    def update_source(self, new_source: Any) -> bool:  # type: ignore[override]
        if not self._host:
            return False
        try:
            target_monitor = self._find_monitor_by_target(new_source)
            if not target_monitor:
                logger.error("Failed to resolve new monitor target: %s", new_source)
                return False
            ok = self._host.set_target_monitor(target_monitor)
            if ok:
                self._target_monitor = target_monitor
            return ok
        except Exception as e:
            logger.error("Error updating monitor source: %s", e, exc_info=True)
            return False

    def get_capture_stats(self) -> Dict[str, Any]:  # type: ignore[override]
        if self._host and hasattr(self._host, "get_capture_stats"):
            return self._host.get_capture_stats()
        return {}

    def is_capturing(self) -> bool:  # type: ignore[override]
        if self._host and hasattr(self._host, "is_capturing"):
            return bool(self._host.is_capturing())
        return False
