"""
MonitorOverlay - Main overlay widget for monitor capture display.

Provides a complete overlay widget with the same styling and behavior as
integrated_dwm_overlay but for monitor capture instead of window capture.
"""

from typing import Optional, Dict, Any
from PySide6.QtCore import Qt, QRect, QRectF, Signal, QEventLoop, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtGui import QMouseEvent, QWheelEvent, QResizeEvent, QMoveEvent, QPainterPath, QRegion

from core.logging import get_logger
from core.threading import get_thread_manager
from core.graphics.pipeline_manager import get_pipeline_manager
from utils.window.behavior import WindowBehaviorManager, get_resize_edge_for_pos, get_cursor_for_edge
from utils.theme.theme_manager import get_theme_manager
from utils.overlay_context_menu import OverlayContextMenu
from utils.monitor_utils import get_all_monitors
from utils.window.overlay_constants import OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT, DEFAULT_ASPECT
from ui.overlays.integrated_border_canvas import IntegratedBorderCanvas
from .capture_display_widget import CaptureDisplayWidget
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QGuiApplication

logger = get_logger(__name__)

class MonitorOverlay(QWidget):
    """
    Monitor capture overlay with integrated border rendering and styling.
    
    Features:
    - Same styling as integrated_dwm_overlay
    - Monitor capture display via OpenGL
    - Hall-of-mirrors detection
    - Window behavior management (drag, resize, snap)
    - Integrated border rendering
    - Theme integration
    """
    
    # Signals
    overlay_closed = Signal()
    capture_started = Signal()
    capture_stopped = Signal()
    capture_error = Signal(str)
    hall_of_mirrors_changed = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._thread_manager = get_thread_manager()
        self._theme_manager = get_theme_manager()
        self._capture_manager = get_pipeline_manager()
        # Allow child canvas to locate its owning overlay (parity with OverlayHost pattern)
        self._parent_overlay = self
        
        # Overlay state
        self._target_monitor: Optional[Dict[str, Any]] = None
        self._is_capturing = False
        
        # Cached aspect ratio for consistent scaling during manual resize (from DWM overlay)
        self._source_aspect: Optional[float] = None
        
        # Window behavior manager for drag/resize/snap
        # Expose as `_behavior` for IntegratedBorderCanvas delegation compatibility
        self._behavior: Optional[WindowBehaviorManager] = None
        # Back-compat alias used by some legacy overlays
        self._window_behavior: Optional[WindowBehaviorManager] = None
        
        # UI components
        self._border_canvas: Optional[IntegratedBorderCanvas] = None
        self._capture_display: Optional[CaptureDisplayWidget] = None
        
        # Initialize overlay
        self._init_overlay()
        self._setup_ui()
        self._connect_signals()

        # Ensure default cursor state (Arrow) with no stale managed overrides
        try:
            from utils.cursor_manager import unset_managed_cursor
            unset_managed_cursor("window_behavior_drag", self)
            unset_managed_cursor("window_behavior_resize", self)
        except Exception:
            pass
        try:
            self.setCursor(Qt.ArrowCursor)
        except Exception:
            pass

        logger.debug("MonitorOverlay initialized")
    
    def _init_overlay(self) -> None:
        """Initialize overlay window properties."""
        # Set window properties to match integrated_dwm_overlay
        self.setWindowFlags(
            Qt.Tool | 
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint
        )
        
        # Window attributes
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        
        # Set object name for QSS styling
        self.setObjectName("monitorOverlay")
        
        # Set minimum size
        self.setMinimumSize(OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT)
        
        # Initialize window behavior manager
        self._behavior = WindowBehaviorManager(
            widget=self,
            min_width=OVERLAY_MIN_WIDTH,
            min_height=OVERLAY_MIN_HEIGHT
        )
        # Back-compat alias
        self._window_behavior = self._behavior
    
    def _setup_ui(self) -> None:
        """Set up the overlay UI components."""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create integrated border canvas (same as DWM overlay)
        self._border_canvas = IntegratedBorderCanvas(self)
        layout.addWidget(self._border_canvas)
        
        # Create capture display widget inside the border canvas
        self._capture_display = CaptureDisplayWidget(self._border_canvas)
        
        # IMPORTANT: Do not use a layout on the border canvas. Instead, bind the
        # capture widget's geometry to the canvas content rect so that the
        # integrated border and inner accent remain visible.
        try:
            # Set initial geometry to current content rect
            initial_rect = self._border_canvas.content_rect()
            self._capture_display.setGeometry(initial_rect)
            # Keep geometry in sync with content rect changes
            self._border_canvas.contentRectChanged.connect(self._capture_display.setGeometry)
        except Exception as e:
            logger.warning(f"Failed to bind capture display geometry to content rect: {e}")
        
        # Ensure mouse tracking and event filters on self and children to keep cursor fresh
        try:
            self.setMouseTracking(True)
            if self._border_canvas:
                self._border_canvas.setMouseTracking(True)
                self._border_canvas.installEventFilter(self)
            if self._capture_display:
                self._capture_display.setMouseTracking(True)
                self._capture_display.installEventFilter(self)
            # Also filter self to catch enter/leave without child propagation issues
            self.installEventFilter(self)
        except Exception:
            pass

        # Apply theme
        self._apply_theme()

        # Attach unified context menu
        try:
            self._ctx_menu = OverlayContextMenu(self, overlay_type='monitor')
            self._ctx_menu.attach_to_overlay(self)
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to attach overlay context menu: {e}")
        
        # Ensure initial window-level masking is applied once UI is ready
        try:
            self._apply_window_masking()
        except Exception as e:
            logger = get_logger(__name__)
            logger.debug(f"Initial window masking skipped: {e}")
    
    def _connect_signals(self) -> None:
        """Connect all signals and slots."""
        # Theme changes
        self._theme_manager.theme_changed.connect(self._apply_theme)
        
        # Capture manager signals
        self._capture_manager.frame_captured.connect(self._on_frame_captured)
        self._capture_manager.capture_started.connect(self._on_capture_started)
        self._capture_manager.capture_stopped.connect(self._on_capture_stopped)
        self._capture_manager.capture_error.connect(self._on_capture_error)
        
        # Capture display signals
        if self._capture_display:
            self._capture_display.hall_of_mirrors_detected.connect(self._on_hall_of_mirrors_changed)
            self._capture_display.capture_error.connect(self.capture_error.emit)
    
    def _apply_theme(self) -> None:
        """Apply current theme to overlay components."""
        try:
            # Apply theme to self (for QSS styling)
            self._theme_manager.apply_theme_to_widget(self)
            
            logger.debug("Theme applied to MonitorOverlay")
            
        except Exception as e:
            logger.warning(f"Error applying theme: {e}")
    
    def set_target_monitor(self, monitor_info: Dict[str, Any]) -> bool:
        """
        Set the target monitor for capture.
        
        Args:
            monitor_info: Monitor information dict from monitor_utils
            
        Returns:
            bool: True if monitor was set successfully
        """
        try:
            # Stop current capture if running
            if self._is_capturing:
                self.stop_capture()
            
            # Set target monitor
            self._target_monitor = monitor_info
            
            # Configure capture manager
            if not self._capture_manager.set_target_monitor(monitor_info):
                logger.error("Failed to set target monitor in capture manager")
                return False
            
            # Configure capture display
            if self._capture_display:
                if not self._capture_display.set_target_monitor(monitor_info):
                    logger.error("Failed to set target monitor in capture display")
                    return False

            # Cache monitor aspect ratio for consistent scaling during resize (from DWM overlay logic)
            self._cache_monitor_aspect(monitor_info)
            
            # Propagate content aspect to integrated border canvas for proper letterbox/pillarbox
            try:
                if self._border_canvas is not None:
                    rect = monitor_info.get('rect') if isinstance(monitor_info, dict) else None
                    if isinstance(rect, QRect):
                        w, h = rect.width(), rect.height()
                        if w > 0 and h > 0:
                            self._border_canvas.set_content_aspect(w, h)
                        else:
                            self._border_canvas.set_content_aspect(0, 0)
            except Exception as e:
                logger.warning(f"Failed to set content aspect on border canvas: {e}")
            
            logger.info(f"Target monitor set: {monitor_info.get('device_name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting target monitor: {e}", exc_info=True)
            return False
    
    def start_capture(self) -> bool:
        """
        Start monitor capture.
        
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
            
        try:
            # Start capture
            if self._capture_manager.start_capture():
                self._is_capturing = True
                logger.info("Monitor capture started")
                return True
            else:
                logger.error("Failed to start capture")
                return False
                
        except Exception as e:
            logger.error(f"Error starting capture: {e}", exc_info=True)
            self.capture_error.emit(f"Failed to start capture: {e}")
            return False
    
    def stop_capture(self) -> None:
        """Stop monitor capture."""
        if not self._is_capturing:
            return
            
        try:
            self._capture_manager.stop_capture()
            self._is_capturing = False
            logger.info("Monitor capture stopped")
            
        except Exception as e:
            logger.error(f"Error stopping capture: {e}", exc_info=True)
    
    def _on_frame_captured(self, frame) -> None:
        """Handle captured frame from capture manager."""
        if self._capture_display:
            self._capture_display.update_frame(frame)
    
    def _on_capture_started(self) -> None:
        """Handle capture started signal."""
        self._is_capturing = True
        self.capture_started.emit()
    
    def _on_capture_stopped(self) -> None:
        """Handle capture stopped signal."""
        self._is_capturing = False
        self.capture_stopped.emit()
    
    def _on_capture_error(self, error: str) -> None:
        """Handle capture error signal."""
        logger.error(f"Capture error: {error}")
        self.capture_error.emit(error)
    
    def _cache_monitor_aspect(self, monitor_info: Dict[str, Any]) -> None:
        """Cache the monitor aspect ratio for consistent scaling during resize.
        
        Uses the same robust logic as DWM overlay for aspect ratio handling.
        """
        try:
            if not monitor_info:
                self._source_aspect = None
                return
            
            # Get monitor dimensions
            rect = monitor_info.get('rect')
            if isinstance(rect, QRect):
                src_w = rect.width()
                src_h = rect.height()
            else:
                # Fallback to width/height keys
                src_w = monitor_info.get('width', 0)
                src_h = monitor_info.get('height', 0)
            
            if src_w > 0 and src_h > 0:
                aspect = src_w / src_h
                # Apply same bounds checking as DWM overlay
                if 0.2 <= aspect <= 5.0:  # Between 1:5 and 5:1
                    self._source_aspect = aspect
                    logger.debug(
                        f"Cached monitor aspect ratio: {self._source_aspect:.3f} ({src_w}x{src_h})"
                    )
                    return
                else:
                    logger.debug(
                        f"Monitor aspect ratio out of bounds: {aspect:.3f} ({src_w}x{src_h}), using default"
                    )
            
            # Fallback to default aspect ratio
            self._source_aspect = DEFAULT_ASPECT[0] / DEFAULT_ASPECT[1]
            logger.debug(f"Using default aspect ratio: {self._source_aspect:.3f}")
            
        except Exception as e:
            self._source_aspect = DEFAULT_ASPECT[0] / DEFAULT_ASPECT[1]
            logger.debug(f"Failed to cache monitor aspect, using default: {e}")

    def _on_hall_of_mirrors_changed(self, detected: bool) -> None:
        """Handle hall of mirrors detection change."""
        self.hall_of_mirrors_changed.emit(detected)
    
    # Mouse event forwarding to WindowBehaviorManager (same as integrated_dwm_overlay)
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Forward mouse press events to window behavior manager."""
        if self._behavior:
            # Allow dragging anywhere on the overlay surface
            self._behavior.handle_mouse_press(event, is_draggable_region=lambda p: True)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Forward mouse move events to window behavior manager."""
        if self._behavior:
            self._behavior.handle_mouse_move(event)
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Forward mouse release events to window behavior manager."""
        if self._behavior:
            self._behavior.handle_mouse_release(event)
        super().mouseReleaseEvent(event)
    
    def leaveEvent(self, event) -> None:
        """Forward leave events to window behavior manager."""
        if self._window_behavior:
            self._window_behavior.leaveEvent(event)
        super().leaveEvent(event)
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Forward wheel events to window behavior manager for resize."""
        if self._window_behavior:
            self._window_behavior.wheelEvent(event)
        super().wheelEvent(event)
    
    def moveEvent(self, event: QMoveEvent) -> None:
        """Handle overlay move - trigger hall-of-mirrors check."""
        super().moveEvent(event)
        
        # Trigger hall-of-mirrors check on move
        if self._capture_display:
            # Use single shot to avoid excessive checks during drag
            self._thread_manager.single_shot(100, self._capture_display._check_hall_of_mirrors)
    
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle overlay resize."""
        super().resizeEvent(event)
        
        # Border canvas will handle its own resize
        # Capture display will handle its own resize
        # Force geometry sync to close any transient border gaps
        try:
            if self._border_canvas and self._capture_display:
                self._capture_display.setGeometry(self._border_canvas.content_rect())
                # Post a zero-delay confirmation sync after layout settles
                self._thread_manager.single_shot(0, lambda: self._capture_display.setGeometry(self._border_canvas.content_rect()))
        except Exception:
            pass

    def _apply_window_masking(self) -> None:
        """Apply window-level masking for rounded corners while preserving mouse events.

        Mirrors the behavior in `core/graphics/backends/dwm/integrated_dwm_backend.py`
        for visual parity. Applies a rounded region mask to this overlay window
        when rounded borders are enabled and the radius > 0; otherwise clears
        the mask. Uses a slightly smaller radius than the canvas for crisp edges.
        """
        try:
            canvas = self._border_canvas
            if not canvas:
                return

            if (hasattr(canvas, '_border_metrics') and canvas._border_metrics and
                hasattr(canvas, '_rounded_enabled') and canvas._rounded_enabled):
                metrics = canvas._border_metrics
                radius = getattr(metrics, 'corner_radius', 0) or 0
                if radius > 0:
                    # Build rounded rect path using slightly smaller radius (legacy parity)
                    rectf = QRectF(self.rect())
                    path = QPainterPath()
                    mask_radius = max(1.0, float(radius) - 0.5)
                    path.addRoundedRect(rectf, mask_radius, mask_radius)
                    region = QRegion(path.toFillPolygon().toPolygon())
                    self.setMask(region)
                    logger.debug(f"Applied window mask with radius {mask_radius}")
                    return

            # Clear mask if rounded borders are disabled or radius is zero
            self.clearMask()

        except Exception as e:
            logger.error(f"Window masking failed: {e}")
            # Ensure mask is cleared on failure to prevent broken state
            try:
                self.clearMask()
            except Exception:
                pass
    
    def closeEvent(self, event) -> None:
        """Handle overlay close."""
        logger.debug("MonitorOverlay closing")
        
        # Stop capture
        self.stop_capture()

        # Wait briefly for capture shutdown to complete to avoid teardown races
        try:
            if self._capture_manager and self._capture_manager.is_capturing():
                logger.debug("Waiting up to 400ms for capture backend to report stopped")
                loop = QEventLoop(self)

                def _quit_loop():
                    if loop.isRunning():
                        loop.quit()

                try:
                    self._capture_manager.capture_stopped.connect(_quit_loop)
                    self._capture_manager.capture_error.connect(_quit_loop)
                except Exception:
                    pass

                # Timeout guard via ThreadManager (no raw QTimer)
                self._thread_manager.single_shot(400, _quit_loop)
                loop.exec()
        except Exception:
            pass
        
        # Clean up components
        if self._capture_display:
            # Disconnect content rect binding
            try:
                if self._border_canvas and hasattr(self._border_canvas, 'contentRectChanged'):
                    self._border_canvas.contentRectChanged.disconnect(self._capture_display.setGeometry)  # type: ignore[arg-type]
            except Exception:
                pass
            # Cleanup renderer path
            try:
                self._capture_display.cleanup()
            except Exception:
                pass
            try:
                self._capture_display.hide()
            except Exception:
                pass
            try:
                self._capture_display.setParent(None)
                self._capture_display.deleteLater()
            except Exception:
                pass
            self._capture_display = None

        # Detach context menu filters
        try:
            if hasattr(self, '_ctx_menu') and self._ctx_menu:
                self._ctx_menu.detach_from_overlay(self)
        except Exception:
            pass

        # Disconnect theme/capture signals
        try:
            if self._theme_manager:
                self._theme_manager.theme_changed.disconnect(self._apply_theme)
        except Exception:
            pass
        try:
            if self._capture_manager:
                self._capture_manager.frame_captured.disconnect(self._on_frame_captured)
                self._capture_manager.capture_started.disconnect(self._on_capture_started)
                self._capture_manager.capture_stopped.disconnect(self._on_capture_stopped)
                self._capture_manager.capture_error.disconnect(self._on_capture_error)
        except Exception:
            pass

        # Delete border canvas
        if self._border_canvas:
            try:
                self._border_canvas.hide()
            except Exception:
                pass
            try:
                self._border_canvas.setParent(None)
                self._border_canvas.deleteLater()
            except Exception:
                pass
            self._border_canvas = None
        
        # Emit closed signal
        self.overlay_closed.emit()
        
        super().closeEvent(event)
    
    def show(self) -> None:
        """Show the overlay and start capture."""
        super().show()

        # Reassert default cursor on show (guard against prior resize/drag states)
        try:
            from utils.cursor_manager import unset_managed_cursor
            unset_managed_cursor("window_behavior_drag", self)
            unset_managed_cursor("window_behavior_resize", self)
        except Exception:
            pass
        try:
            self.setCursor(Qt.ArrowCursor)
        except Exception:
            pass

        # Auto-start capture when shown
        if self._target_monitor and not self._is_capturing:
            self.start_capture()

    def eventFilter(self, obj, event):
        """Intercept child mouse/hover/leave to keep cursor state correct.

        Ensures edge-resize cursors are cleared immediately when moving away
        from edges, even when child widgets consume events.
        """
        try:
            et = event.type()
            if et in (QEvent.MouseMove, QEvent.HoverMove):
                # Translate position to overlay coords
                try:
                    # Qt6 API
                    gpos = event.globalPosition().toPoint() if hasattr(event, 'globalPosition') else event.globalPos()
                    pos = self.mapFromGlobal(gpos)
                except Exception:
                    try:
                        # Fallback via local pos mapping
                        lpos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
                        pos = obj.mapTo(self, lpos)
                    except Exception:
                        pos = None

                if pos is not None and self._behavior and not (self._behavior.state.is_dragging or self._behavior.state.is_resizing):
                    edge = get_resize_edge_for_pos(pos, self)
                    cursor = get_cursor_for_edge(edge)
                    if cursor is not None:
                        self.setCursor(cursor)
                    else:
                        # Clear any managed cursor overrides and revert to Arrow
                        try:
                            from utils.cursor_manager import unset_managed_cursor
                            unset_managed_cursor("window_behavior_drag", self)
                            unset_managed_cursor("window_behavior_resize", self)
                        except Exception:
                            pass
                        self.setCursor(Qt.ArrowCursor)
                return False

            if et == QEvent.Leave:
                # When the mouse leaves any child/self, ensure arrow unless actively dragging/resizing
                if self._behavior and not (self._behavior.state.is_dragging or self._behavior.state.is_resizing):
                    try:
                        from utils.cursor_manager import unset_managed_cursor
                        unset_managed_cursor("window_behavior_drag", self)
                        unset_managed_cursor("window_behavior_resize", self)
                    except Exception:
                        pass
                    self.setCursor(Qt.ArrowCursor)
                return False

        except Exception:
            # Never break event flow due to cursor maintenance
            pass
        return super().eventFilter(obj, event)
    
    def hide(self) -> None:
        """Hide the overlay and stop capture."""
        # Stop capture when hidden
        if self._is_capturing:
            self.stop_capture()
        
        super().hide()
    
    def get_target_monitor(self) -> Optional[Dict[str, Any]]:
        """Get the current target monitor."""
        return self._target_monitor
    
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        return self._is_capturing
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        stats = self._capture_manager.get_capture_stats()
        stats['hall_of_mirrors'] = (
            self._capture_display.is_hall_of_mirrors_active() 
            if self._capture_display else False
        )
        return stats

    # --- Context menu handlers -------------------------------------------------
    def _handle_reset_position(self) -> None:
        """Reset overlay geometry to a default centered size on the current screen."""
        try:
            # Determine target screen based on current window center
            center = self.frameGeometry().center()
            screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
            if screen is None:
                logger.error("Reset position: no screen available")
                return
            sgeo = screen.availableGeometry()

            # Compute a sensible default size with 16:9 aspect, respecting minimums
            min_w, min_h = OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT
            # Base size: 40% of screen width, maintain 16:9
            target_w = max(min_w, int(sgeo.width() * 0.4))
            # Constrain by height as well
            max_h_by_screen = int(sgeo.height() * 0.4)
            # Apply 16:9 aspect ratio
            target_h = max(min_h, int(target_w * DEFAULT_ASPECT[1] / DEFAULT_ASPECT[0]))
            if target_h > max_h_by_screen:
                target_h = max(min_h, max_h_by_screen)
                target_w = max(min_w, int(target_h * DEFAULT_ASPECT[0] / DEFAULT_ASPECT[1]))

            # Center within available geometry
            x = sgeo.x() + (sgeo.width() - target_w) // 2
            y = sgeo.y() + (sgeo.height() - target_h) // 2
            self.setGeometry(QRect(x, y, target_w, target_h))

            # Ensure masking and inner geometry sync after reset
            try:
                self._apply_window_masking()
            except Exception:
                pass
            try:
                if self._border_canvas and self._capture_display:
                    self._capture_display.setGeometry(self._border_canvas.content_rect())
            except Exception:
                pass
            logger.debug(f"Overlay reset to centered {target_w}x{target_h} on {screen.name() if hasattr(screen,'name') else 'screen'}")
        except Exception as e:
            logger.error(f"Error resetting overlay position: {e}", exc_info=True)

    def _handle_quit_application(self) -> None:
        """Quit the application cleanly."""
        try:
            app = QApplication.instance()
            if app is not None:
                app.quit()
            else:
                logger.error("Quit requested but no QApplication instance available")
        except Exception as e:
            logger.error(f"Error quitting application: {e}")
    def _handle_switch_monitor(self, screen_obj) -> None:
        """Switch capture to the selected QScreen from the context menu."""
        try:
            monitors = get_all_monitors()
            target = None

            # Prefer direct object identity match
            for m in monitors:
                if m.get('screen_object') is screen_obj:
                    target = m
                    break

            # Fallback: name match
            if target is None:
                try:
                    sname = screen_obj.name() if hasattr(screen_obj, 'name') else None
                    if sname:
                        for m in monitors:
                            sobj = m.get('screen_object')
                            if sobj is not None and hasattr(sobj, 'name') and sobj.name() == sname:
                                target = m
                                break
                except Exception:
                    pass

            # Fallback: geometry match
            if target is None:
                try:
                    sgeo = screen_obj.geometry() if hasattr(screen_obj, 'geometry') else None
                    if sgeo is not None:
                        for m in monitors:
                            rect = m.get('rect')
                            if isinstance(rect, QRect) and rect == sgeo:
                                target = m
                                break
                except Exception:
                    pass

            if target is None:
                logger.error("Context menu switch failed: selected screen not found in monitor list")
                return

            was_running = self._is_capturing
            if was_running:
                self.stop_capture()

            if self.set_target_monitor(target) and was_running:
                # Restart capture on new monitor
                self.start_capture()

        except Exception as e:
            logger.error(f"Error handling switch monitor: {e}", exc_info=True)
