"""
CaptureDisplayWidget - Display widget with hall-of-mirrors detection.

Provides the main display area for monitor capture with integrated
hall-of-mirrors detection and themed warning display, using a QWidget-based
renderer (no OpenGL).
"""

from typing import Optional, Dict, Any
from PySide6.QtCore import QPoint, QRect, QSize, Signal
from PySide6.QtGui import QFont, QFontMetrics, QPainter, QColor
from PySide6.QtWidgets import QWidget

from core.logging import get_logger
from core.graphics.capture import MonitorCaptureManager, D3D11MonitorRenderer, CaptureFrame
from utils.theme.theme_manager import get_theme_manager
from utils.monitor_utils import get_monitor_at

logger = get_logger(__name__)

class CaptureDisplayWidget(QWidget):
    """
    Widget that combines a QWidget-based renderer with hall-of-mirrors detection.
    
    Features:
    - Non-OpenGL monitor renderer integration
    - Real-time hall-of-mirrors detection
    - Themed warning display
    - Monitor change detection
    """
    
    # Signals
    hall_of_mirrors_detected = Signal(bool)
    capture_error = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._theme_manager = get_theme_manager()
        
        # Renderer (QWidget blitter)
        self._renderer: Optional[D3D11MonitorRenderer] = None
        
        # Monitor tracking
        self._target_monitor: Optional[Dict[str, Any]] = None
        self._current_overlay_monitor: Optional[Dict[str, Any]] = None
        self._hall_of_mirrors_active = False
        
        # Warning display
        self._warning_font: Optional[QFont] = None
        
        # Initialize renderer
        self._init_renderer()
        
        # Set up theming
        self._apply_theme()
        self._theme_manager.theme_changed.connect(self._apply_theme)
        
        logger.debug("CaptureDisplayWidget initialized")
    
    def _init_renderer(self) -> None:
        """Initialize the renderer (always QWidget blitter)."""
        try:
            self._renderer = D3D11MonitorRenderer(self)
            # Connect signals
            self._renderer.render_error.connect(self.capture_error.emit)

            # Set up layout to fill widget
            from PySide6.QtWidgets import QVBoxLayout
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._renderer)

            logger.debug("Renderer initialized (QWidget blitter)")

        except Exception as e:
            logger.error(f"Failed to initialize renderer: {e}", exc_info=True)
            self.capture_error.emit(f"Renderer initialization failed: {e}")

    def _on_swapchain_ready(self, hwnd: int) -> None:
        """Deprecated: swapchain path removed; method retained for API stability."""
        logger.debug("_on_swapchain_ready called but swapchain path is removed")

    def _on_swapchain_resized(self, size: QSize) -> None:
        """Deprecated: swapchain path removed; method retained for API stability."""
        try:
            logger.debug(f"_on_swapchain_resized called (ignored): {size.width()}x{size.height()}")
        except Exception:
            logger.debug("_on_swapchain_resized called (ignored)")
    
    def _apply_theme(self) -> None:
        """Apply current theme styling."""
        try:
            # Get theme colors for warning display
            colors = self._theme_manager.get_theme_colors()
            bg_color = colors.get('base', '#2b2b2b')
            text_color = colors.get('text', '#ffffff')
            
            # Set up warning font
            self._warning_font = QFont()
            self._warning_font.setFamily("Segoe UI")
            self._warning_font.setPointSize(14)
            self._warning_font.setBold(True)
            
            logger.debug("Theme applied to CaptureDisplayWidget")
            
        except Exception as e:
            logger.warning(f"Error applying theme: {e}")
    
    def set_target_monitor(self, monitor_info: Dict[str, Any]) -> bool:
        """
        Set the target monitor for capture.
        
        Args:
            monitor_info: Monitor information dict
            
        Returns:
            bool: True if monitor was set successfully
        """
        try:
            self._target_monitor = monitor_info
            
            # Check for hall-of-mirrors immediately
            self._check_hall_of_mirrors()
            
            # Ensure the renderer consumes frames from the correct exchange
            # MonitorCaptureManager publishes to "monitor_frames_{qt_index}"
            # so we must point the renderer to the same index.
            try:
                if self._renderer and isinstance(monitor_info, dict):
                    qt_index = int(monitor_info.get('qt_index', 0))
                    # Both renderer types expose set_monitor_index
                    self._renderer.set_monitor_index(qt_index)
                    logger.debug(f"Renderer FrameExchange set to monitor_frames_{qt_index}")
            except Exception as e:
                logger.warning(f"Failed to set renderer monitor index: {e}")
            
            logger.info(f"Target monitor set: {monitor_info.get('device_name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting target monitor: {e}", exc_info=True)
            return False
    
    def update_frame(self, frame: CaptureFrame) -> None:
        """
        Update display with new capture frame.
        
        Args:
            frame: New capture frame
        """
        if not self._renderer:
            return
            
        # Check hall-of-mirrors before updating
        self._check_hall_of_mirrors()
        
        if not self._hall_of_mirrors_active:
            # Normal frame update
            self._renderer.update_frame(frame)
        # If hall-of-mirrors is active, renderer will show warning
    
    def _check_hall_of_mirrors(self) -> None:
        """Check if overlay is on the same monitor as capture target."""
        try:
            if not self._target_monitor:
                return
                
            # Get current overlay monitor
            overlay_monitor = self._get_overlay_monitor()
            
            # Compare monitors
            is_same_monitor = self._are_same_monitor(self._target_monitor, overlay_monitor)
            
            if is_same_monitor != self._hall_of_mirrors_active:
                self._hall_of_mirrors_active = is_same_monitor
                
                if self._renderer:
                    if is_same_monitor:
                        # Show warning
                        self._renderer.set_hall_of_mirrors_warning(
                            True,
                            "CAPTURE IS CURRENT DISPLAY\nMOVE TO OTHER DISPLAY FOR OUTPUT"
                        )
                    else:
                        # Hide warning
                        self._renderer.set_hall_of_mirrors_warning(False)
                
                # Emit signal
                self.hall_of_mirrors_detected.emit(is_same_monitor)
                
                logger.debug(f"Hall-of-mirrors detection: {is_same_monitor}")
                
        except Exception as e:
            logger.error(f"Error in hall-of-mirrors detection: {e}", exc_info=True)
    
    def _get_overlay_monitor(self) -> Optional[Dict[str, Any]]:
        """Get the monitor that contains this overlay widget."""
        try:
            # Get widget global position
            widget_pos = self.mapToGlobal(QPoint(0, 0))
            widget_center = QPoint(
                widget_pos.x() + self.width() // 2,
                widget_pos.y() + self.height() // 2
            )
            
            # Use centralized helper to find monitor at point (cached, non-spammy)
            return get_monitor_at(widget_center)
            
        except Exception as e:
            logger.error(f"Error getting overlay monitor: {e}", exc_info=True)
            return None
    
    def _are_same_monitor(self, monitor1: Dict[str, Any], monitor2: Optional[Dict[str, Any]]) -> bool:
        """
        Check if two monitors are the same.
        
        Args:
            monitor1: First monitor info
            monitor2: Second monitor info (can be None)
            
        Returns:
            bool: True if monitors are the same
        """
        if not monitor2:
            return False
            
        try:
            # Compare by device name first
            name1 = monitor1.get('device_name', '')
            name2 = monitor2.get('device_name', '')
            if name1 and name2 and name1 == name2:
                return True
                
            # Fallback to position comparison
            rect1 = monitor1.get('rect', QRect())
            rect2 = monitor2.get('rect', QRect())
            
            return (rect1.x() == rect2.x() and 
                    rect1.y() == rect2.y() and
                    rect1.width() == rect2.width() and
                    rect1.height() == rect2.height())
                    
        except Exception as e:
            logger.error(f"Error comparing monitors: {e}", exc_info=True)
            return False
    
    def moveEvent(self, event) -> None:
        """Handle widget move - check for hall-of-mirrors."""
        super().moveEvent(event)
        self._check_hall_of_mirrors()
    
    def resizeEvent(self, event) -> None:
        """Handle widget resize."""
        super().resizeEvent(event)
        # Renderer will handle its own resize
    
    def get_renderer(self) -> Optional[D3D11MonitorRenderer]:
        """Get the renderer instance."""
        return self._renderer
    
    def is_hall_of_mirrors_active(self) -> bool:
        """Check if hall-of-mirrors warning is currently active."""
        return self._hall_of_mirrors_active
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.debug("Cleaning up CaptureDisplayWidget")

        if self._renderer:
            self._renderer.cleanup()
            self._renderer = None
        # Swapchain path removed; no HWND cleanup required
