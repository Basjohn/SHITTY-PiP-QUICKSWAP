import logging
import math
import os
import sys
from debug_utils import debug_enabled
import time
import numpy as np
import mss
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QObject, Signal, QThread, QMutex, QMutexLocker
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPixmap, QScreen, QPainter, QPen, QFont, QMouseEvent, QAction, QCursor
from PySide6.QtWidgets import QMainWindow, QMenu, QApplication
from monitor_utils import get_physical_monitor_info, get_all_monitors
import snap_utils

def edge_to_cursor(edge):
    """
    Convert an edge name to the corresponding Qt cursor shape.
    
    Args:
        edge (str): Edge name (e.g., 'top', 'bottom', 'left', 'right', etc.)
        
    Returns:
        Qt.CursorShape: The corresponding cursor shape
    """
    if not edge:
        return Qt.ArrowCursor
        
    edge = edge.lower()
    if edge in ['top', 'bottom']:
        return Qt.SizeVerCursor
    elif edge in ['left', 'right']:
        return Qt.SizeHorCursor
    elif edge in ['top-left', 'bottom-right']:
        return Qt.SizeFDiagCursor
    elif edge in ['top-right', 'bottom-left']:
        return Qt.SizeBDiagCursor
    return Qt.ArrowCursor

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.getLogger('mss').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)

class CaptureWorker(QObject):
    frame_ready = Signal(object, int, int, float)
    finished = Signal()
    
    def __init__(self, screen=None, parent=None):
        super().__init__(parent)
        self._mss_instance = None
        self._mutex = QMutex()
        self._running = False
        self._capture_params = None
        self._last_frame = None
        self._fps = 60
        self._screen = screen
        self._monitors = None
        
    def set_fps(self, fps):
        with QMutexLocker(self._mutex):
            try:
                self._fps = max(1, min(240, int(fps)))
                logger.debug(f"Capture FPS set to {self._fps}")
            except (ValueError, TypeError):
                logger.warning(f"Invalid FPS value: {fps}, using default 60")
                self._fps = 60
        
    def set_capture_params(self, monitor_idx, width, height, screen=None):
        with QMutexLocker(self._mutex):
            self._capture_params = (monitor_idx, width, height)
            if screen is not None:
                self._screen = screen
    
    def stop(self):
        self._running = False
    
    def _monitor_cpu_usage(self):
        pass
    
    def run(self):
        try:
            if os.name == 'nt':
                try:
                    import win32api
                    import win32process
                    thread_handle = win32api.GetCurrentThread()
                    win32process.SetThreadPriority(thread_handle, win32process.THREAD_PRIORITY_LOWEST)
                except Exception as e:
                    logger.warning(f"Failed to set thread priority: {e}")
            
            self._running = True
            mss_instance = mss.mss()
            monitors = mss_instance.monitors
            logger.debug(f"CaptureWorker started at {self._fps} FPS")
            
            if not monitors or len(monitors) < 2:
                logger.warning("No valid monitors found in MSS")
                return
            
            while self._running:
                try:
                    with QMutexLocker(self._mutex):
                        if not self._capture_params:
                            time.sleep(0.01)
                            continue
                        monitor_idx, width, height = self._capture_params
                    
                    if monitor_idx < 0 or monitor_idx >= len(monitors):
                        monitor_idx = 1  # Fallback to first physical monitor
                        logger.debug("Invalid monitor index, falling back to first physical monitor (1)")
                        
                    monitor = monitors[monitor_idx]
                    try:
                        screenshot = mss_instance.grab(monitor)
                        if not screenshot or not screenshot.raw:
                            time.sleep(0.05)
                            continue
                    except Exception as e:
                        logger.debug(f"Error capturing monitor {monitor_idx}: {e}", extra={"suppress": True})
                        time.sleep(0.1)
                        continue
                        
                    img_array = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 4))
                    
                    if self._last_frame is not None and np.array_equal(img_array, self._last_frame):
                        time.sleep(0.001)
                        continue
                        
                    self._last_frame = img_array
                    rgb_array = np.ascontiguousarray(img_array[..., [2, 1, 0]])
                    
                    dpr = self._screen.devicePixelRatio() if self._screen else 1.0
                    self.frame_ready.emit(rgb_array, screenshot.width, screenshot.height, dpr)
                    
                    frame_start = time.time()
                    sleep_time = max(0.001, 1.0 / self._fps - (time.time() - frame_start))
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in capture loop: {e}", exc_info=True)
                    time.sleep(0.1)
                    
        finally:
            if mss_instance:
                mss_instance.close()
            self._running = False
            self.finished.emit()

class MonitorOverlay(QMainWindow):
    closed = Signal()

    def __init__(self, screen: 'QScreen' = None, opacity: float = None, theme: str = "dark", snap_distance: int = 8, 
                     app_instance=None, initial_geometry=None, monitor_idx: int = None):
        # Initialize base class with minimal flags - we'll set the rest after basic initialization
        super().__init__(None)
        
        # Initialize instance variables first
        self.app_instance = app_instance
        self.theme = theme.lower()
        self.snap_distance = snap_distance
        self.mss_instance = mss.mss()
        self._mss_monitor_idx = None
        self.capture_target_screen = screen or QGuiApplication.primaryScreen()
        self._display_screen = None
        self._pixmap = None
        self._edge_margin = 8
        self._drag_state = {
            'is_resizing': False,
            'resize_edge': None,
            'drag_start_global': None,
            'initial_geometry': None,
            'drag_offset': None
        }
        self._drag_initial_mouse_pos = None
        self._drag_initial_window_pos = None
        self._is_resizing = False
        self._resize_edge = None
        self._initial_geometry = None
        self._worker_thread = None
        self._capture_worker = None
        self._cached_screen_geometry = None
        self._cached_dpr = 1.0
        self._is_snapped = False
        self._snap_deadzone = 5
        self.border_pen = None
        self.border_opacity = 1.0  # Default to fully opaque
        
        # Initialize opacity tracking
        self.opacity = 1.0  # Will be overridden by settings
        self._opacity_initialized = False
        
        # Set window attributes and flags before showing
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Load opacity from settings or use default
        initial_opacity = None
        opacity_source = "not set yet"
        
        if self.app_instance and hasattr(self.app_instance, 'settings'):
            try:
                settings_opacity = self.app_instance.settings.value("overlay_opacity", None, type=int)
                if settings_opacity is not None:
                    initial_opacity = max(0.1, min(1.0, float(settings_opacity) / 100.0))
                    opacity_source = f"saved settings ({settings_opacity}%)"
                    logger.info(f"[MonitorOverlay] Loading overlay opacity from {opacity_source}")
            except Exception as e:
                logger.error(f"[MonitorOverlay] Error loading overlay opacity from settings: {e}")
        
        if initial_opacity is None:
            if opacity is not None:
                initial_opacity = max(0.1, min(1.0, float(opacity)))
                opacity_source = f"constructor parameter ({opacity*100:.0f}%)"
            else:
                initial_opacity = 0.8  # Default value
                opacity_source = "default (80%)"
            logger.info(f"[MonitorOverlay] Using {opacity_source} for initial opacity")
        
        # Set initial opacity
        logger.info(f"[MonitorOverlay] Setting initial opacity: {initial_opacity*100:.0f}% (source: {opacity_source})")
        self.opacity = initial_opacity
        self._opacity_initialized = True
        
        # Initialize UI components
        self.apply_theme(self.theme)
        self._setup_mss_monitor_mapping()
        self._init_ui(initial_geometry)
        
        # Position the window
        self._position_on_non_captured_monitor()
        
        # Show the window first
        self.show()
        
        # Initialize the capture worker
        self._init_capture_worker()
        
        # Apply opacity after a small delay to ensure window is fully initialized
        def apply_final_opacity():
            try:
                # Re-apply the window flags and attributes
                self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
                self.setAttribute(Qt.WA_TranslucentBackground)
                self.setAttribute(Qt.WA_NoSystemBackground)
                self.show()
                
                # Apply the opacity
                self.setWindowOpacity(initial_opacity)
                logger.info(f"[MonitorOverlay] Applied final opacity to window: {initial_opacity*100:.0f}%")
                
                # Force a theme reload to ensure everything is applied correctly
                self._force_theme_reload()
                
                # Force an update to ensure everything is rendered correctly
                self.update()
                
                # Raise and activate the window to ensure it's on top
                self.raise_()
                self.activateWindow()
                
            except Exception as e:
                logger.error(f"[MonitorOverlay] Error in final opacity application: {e}")
        
        # Apply the final opacity after a short delay
        QTimer.singleShot(100, apply_final_opacity)
    
    def _force_theme_reload(self):
        """Force a theme reload using the exact sequence from the logs."""
        if not hasattr(self, 'opacity'):
            return
            
        logger.info("[MonitorOverlay] Applying theme reload sequence...")
        
        try:
            # First apply theme with from_global=False
            logger.info("[MonitorOverlay] Applying theme (from_global=False)")
            self.apply_theme(self.theme, from_global=False)
            
            # Apply theme styles
            logger.info("[MonitorOverlay] Applying theme styles")
            self.apply_theme_styles()
            
            # Apply theme again with from_global=False
            logger.info("[MonitorOverlay] Re-applying theme (from_global=False)")
            self.apply_theme(self.theme, from_global=False)
            
            # Apply theme with from_global=True
            logger.info("[MonitorOverlay] Applying theme (from_global=True)")
            self.apply_theme(self.theme, from_global=True)
            
            # Re-apply the opacity
            self.setWindowOpacity(self.opacity)
            
            # Force updates and repaint
            self.update()
            
            logger.info(f"[MonitorOverlay] Theme reload complete, opacity: {self.opacity*100:.0f}%")
            
        except Exception as e:
            logger.error(f"[MonitorOverlay] Error during theme reload: {e}")
        
        # Final repaint
        self.update()
        
    def set_overlay_opacity(self, opacity, emit_signal: bool = True):
        """Set the overlay window's opacity.
        
        Note: This method only applies the opacity to the window and does NOT save it to settings.
        The monitor overlay should only read from settings, not write to them.
        
        Args:
            opacity: Opacity value, which can be:
                   - float: 0.0 (transparent) to 1.0 (opaque)
                   - int: 0 (transparent) to 100 (opaque), will be normalized to 0.0-1.0
            emit_signal: Whether to emit the opacity_changed signal (default: True)
        """
        try:
            # Get current opacity, defaulting to 1.0 if not set
            old_opacity = getattr(self, 'opacity', 1.0)
            
            # Normalize input to 0.0-1.0 range
            try:
                # Convert to float first
                opacity_float = float(opacity)
                
                # If value > 1.0, assume it's in 0-100 range and normalize
                if opacity_float > 1.0:
                    new_opacity = max(0.1, min(1.0, opacity_float / 100.0))  # Minimum 0.1 (10%) opacity
                    if debug_enabled():
                        logger.debug(f"[MonitorOverlay] Normalized opacity {opacity_float} to {new_opacity:.3f}")
                else:
                    new_opacity = max(0.1, min(1.0, opacity_float))  # Minimum 0.1 (10%) opacity
                    
            except (TypeError, ValueError):
                logger.error(f"[MonitorOverlay] Invalid opacity value: {opacity} (type: {type(opacity)}). Using previous value: {old_opacity:.3f}")
                return
                
            # Only proceed if the opacity has meaningfully changed or this is the initial set
            if self._opacity_initialized and math.isclose(new_opacity, old_opacity, abs_tol=0.001):
                if debug_enabled():
                    logger.debug(f"[MonitorOverlay] Opacity unchanged: {new_opacity:.3f}")
                return
                
            if debug_enabled():
                if not self._opacity_initialized:
                    logger.debug(f"[MonitorOverlay] Setting initial opacity to {new_opacity:.3f}")
                else:
                    logger.debug(f"[MonitorOverlay] Updating opacity from {old_opacity:.3f} to {new_opacity:.3f}")
            
            # Mark as initialized and update the stored opacity
            self._opacity_initialized = True
            self.opacity = new_opacity
            
            # Apply the new opacity to the window
            try:
                # Ensure window is properly initialized before setting opacity
                if not self.testAttribute(Qt.WA_WState_Created):
                    self.show()
                    self.hide()
                
                # Set window opacity
                self.setWindowOpacity(new_opacity)
                
                # Force a repaint to ensure the change is visible
                self.update()
                
                # Sometimes we need to force a window update
                if self.isVisible():
                    self.raise_()
                    self.activateWindow()
                
                if debug_enabled():
                    logger.debug(f"[MonitorOverlay] Successfully updated opacity to {new_opacity:.3f}")
                    
            except Exception as e:
                logger.error(f"[MonitorOverlay] Failed to set window opacity: {e}", exc_info=debug_enabled())
                return
            
            # Emit signal if requested
            if emit_signal and hasattr(self, 'overlay_opacity_changed'):
                try:
                    if debug_enabled():
                        logger.debug("[MonitorOverlay] Emitting overlay_opacity_changed signal")
                    # Emit the value in 0-100 range for UI consistency
                    self.overlay_opacity_changed.emit(int(new_opacity * 100))
                except Exception as e:
                    logger.error(f"[MonitorOverlay] Error emitting opacity_changed signal: {e}")
                
        except Exception as e:
            logger.error(f"[MonitorOverlay] Critical error in set_overlay_opacity: {e}", exc_info=debug_enabled())
        
    def _position_on_non_captured_monitor(self):
        try:
            screens = QGuiApplication.screens()
            if not screens:
                logger.error("No screens available for positioning overlay")
                self.move(0, 0)
                return

            if not self.capture_target_screen or len(screens) == 1:
                target_screen = QGuiApplication.primaryScreen()
                if target_screen:
                    self._position_top_left_on_screen(target_screen)
                    logger.info(f"Positioned overlay on primary screen: {target_screen.name() if hasattr(target_screen, 'name') else 'unnamed'}")
                else:
                    logger.warning("No primary screen found, using default position")
                    self.move(0, 0)
                return

            for screen in screens:
                if screen != self.capture_target_screen:
                    self._position_top_left_on_screen(screen)
                    logger.info(f"Positioned overlay on screen: {screen.name() if hasattr(screen, 'name') else 'unnamed'}")
                    return

            primary_screen = QGuiApplication.primaryScreen()
            if primary_screen:
                self._position_top_left_on_screen(primary_screen)
                logger.info(f"Fallback: Positioned overlay on primary screen: {primary_screen.name() if hasattr(primary_screen, 'name') else 'unnamed'}")
            else:
                logger.warning("No suitable screen found, using default position")
                self.move(0, 0)
        except Exception as e:
            logger.error(f"Error positioning overlay: {str(e)}", exc_info=True)
            self.move(0, 0)

    def _position_top_left_on_screen(self, screen: QScreen):
        screen_geo = screen.availableGeometry()
        margin = 10
        x = screen_geo.x() + margin
        y = screen_geo.y() + margin
        self.move(x, y)

    def _center_on_screen(self, screen: QScreen):
        screen_geo = screen.availableGeometry()
        window_size = self.size()
        x = screen_geo.x() + (screen_geo.width() - window_size.width()) // 2
        y = screen_geo.y() + (screen_geo.height() - window_size.height()) // 2
        self.move(x, y)

    def _select_display_screen(self):
        """Select a display screen for the overlay, preferring one different from capture_target_screen."""
        try:
            screens = QGuiApplication.screens()
            if not screens:
                logger.error("No screens available for display selection")
                return QGuiApplication.primaryScreen()

            if len(screens) == 1:
                return screens[0]

            for screen in screens:
                if screen != self.capture_target_screen:
                    return screen

            return QGuiApplication.primaryScreen()
        except Exception as e:
            logger.error(f"Error selecting display screen: {e}", exc_info=True)
            return QGuiApplication.primaryScreen()

    def set_fps(self, fps: int):
        if self._capture_worker:
            self._capture_worker.set_fps(fps)
            logger.debug(f"Set FPS to {fps} for monitor overlay")
        else:
            logger.warning("Cannot set FPS: No capture worker available")
            
    def set_overlay_opacity(self, opacity, emit_signal: bool = True):
        """Set the overlay window's opacity.
        
        Args:
            opacity: Opacity value, which can be:
                   - float: 0.0 (transparent) to 1.0 (opaque)
                   - int: 0 (transparent) to 100 (opaque), will be normalized to 0.0-1.0
            emit_signal: Whether to emit the opacity_changed signal (default: True)
            
        Note:
            All internal opacity values are stored in the 0.0-1.0 range.
            Input values outside this range will be clamped.
        """
        try:
            # Get current opacity, defaulting to 1.0 if not set
            old_opacity = getattr(self, 'opacity', 1.0)
            
            # Normalize input to 0.0-1.0 range
            try:
                # Convert to float first
                opacity_float = float(opacity)
                
                # If value > 1.0, assume it's in 0-100 range and normalize
                if opacity_float > 1.0:
                    new_opacity = max(0.0, min(1.0, opacity_float / 100.0))
                    if debug_enabled():
                        logger.debug(f"[MonitorOverlay] Normalized opacity {opacity_float} to {new_opacity:.3f}")
                else:
                    new_opacity = max(0.0, min(1.0, opacity_float))
                    
            except (TypeError, ValueError):
                logger.error(f"[MonitorOverlay] Invalid opacity value: {opacity} (type: {type(opacity)}). Using previous value: {old_opacity:.3f}")
                return
                
            # Only proceed if the opacity has meaningfully changed or this is the initial set
            if self._opacity_initialized and math.isclose(new_opacity, old_opacity, abs_tol=0.001):
                if debug_enabled():
                    logger.debug(f"[MonitorOverlay] Opacity unchanged: {new_opacity:.3f}")
                return
                
            if debug_enabled():
                if not self._opacity_initialized:
                    logger.debug(f"[MonitorOverlay] Setting initial opacity to {new_opacity:.3f}")
                else:
                    logger.debug(f"[MonitorOverlay] Updating opacity from {old_opacity:.3f} to {new_opacity:.3f}")
            
            # Mark as initialized and update the stored opacity
            self._opacity_initialized = True
            self.opacity = new_opacity
            
            # Apply the new opacity to the window
            try:
                self.setWindowOpacity(new_opacity)
            except Exception as e:
                logger.error(f"[MonitorOverlay] Failed to set window opacity: {e}")
                return
            
            # Force a repaint to ensure the change is visible
            self.update()
            
            # Emit signal if requested
            if emit_signal and hasattr(self, 'overlay_opacity_changed'):
                try:
                    if debug_enabled():
                        logger.debug("[MonitorOverlay] Emitting overlay_opacity_changed signal")
                    # Emit the value in 0-100 range for UI consistency
                    self.overlay_opacity_changed.emit(int(new_opacity * 100))
                except Exception as e:
                    logger.error(f"[MonitorOverlay] Error emitting opacity_changed signal: {e}")
            
            if debug_enabled():
                logger.debug(f"[MonitorOverlay] Successfully updated opacity to {new_opacity:.3f}")
                
        except Exception as e:
            logger.error(f"[MonitorOverlay] Critical error in set_overlay_opacity: {e}", exc_info=debug_enabled())
    
    def set_border_opacity(self, opacity, emit_signal: bool = True):
        """Set the border opacity.
        
        Args:
            opacity: Opacity value, which can be:
                   - float: 0.0 (transparent) to 1.0 (opaque)
                   - int: 0 (transparent) to 100 (opaque), will be normalized to 0.0-1.0
            emit_signal: Whether to emit the border_opacity_changed signal (default: True)
            
        Note:
            All internal opacity values are stored in the 0.0-1.0 range.
            Input values outside this range will be clamped.
        """
        try:
            # Get current border opacity, defaulting to 1.0 if not set
            old_border_opacity = getattr(self, 'border_opacity', 1.0)
            
            # Normalize input to 0.0-1.0 range
            try:
                # Convert to float first
                opacity_float = float(opacity)
                
                # If value > 1.0, assume it's in 0-100 range and normalize
                if opacity_float > 1.0:
                    new_opacity = max(0.0, min(1.0, opacity_float / 100.0))
                    if debug_enabled():
                        logger.debug(f"[MonitorOverlay] Normalized border opacity {opacity_float} to {new_opacity:.3f}")
                else:
                    new_opacity = max(0.0, min(1.0, opacity_float))
                    
            except (TypeError, ValueError):
                logger.error(f"[MonitorOverlay] Invalid border opacity value: {opacity} (type: {type(opacity)}). Using previous value: {old_border_opacity:.3f}")
                return
                
            # Only proceed if the opacity has meaningfully changed
            if math.isclose(new_opacity, old_border_opacity, abs_tol=0.001):
                if debug_enabled():
                    logger.debug(f"[MonitorOverlay] Border opacity unchanged: {new_opacity:.3f}")
                return
                
            if debug_enabled():
                logger.debug(f"[MonitorOverlay] Updating border opacity from {old_border_opacity:.3f} to {new_opacity:.3f}")
            
            # Update the stored border opacity
            self.border_opacity = new_opacity
            
            # Initialize border_pen if it doesn't exist
            if not hasattr(self, 'border_pen') or self.border_pen is None:
                self.border_pen = QPen(Qt.white if self.theme == 'dark' else Qt.black, 2)
            
            # Update the border color with the new opacity
            try:
                color = self.border_pen.color()
                # Preserve the RGB values, only update the alpha
                color.setAlphaF(new_opacity)
                self.border_pen.setColor(color)
                if debug_enabled():
                    logger.debug(f"[MonitorOverlay] Updated border pen color with opacity {new_opacity:.3f}")
            except Exception as e:
                logger.error(f"[MonitorOverlay] Failed to update border pen color: {e}")
            
            # Force a repaint to ensure the change is visible
            self.update()
            
            # Emit signal if requested
            if emit_signal and hasattr(self, 'border_opacity_changed'):
                try:
                    if debug_enabled():
                        logger.debug("[MonitorOverlay] Emitting border_opacity_changed signal")
                    # Emit the value in 0-100 range for UI consistency
                    self.border_opacity_changed.emit(int(new_opacity * 100))
                except Exception as e:
                    logger.error(f"[MonitorOverlay] Error emitting border_opacity_changed signal: {e}")
            
            if debug_enabled():
                logger.debug(f"[MonitorOverlay] Successfully updated border opacity to {new_opacity:.3f}")
                
        except Exception as e:
            logger.error(f"[MonitorOverlay] Critical error in set_border_opacity: {e}", exc_info=debug_enabled())
        
    def apply_theme(self, theme, from_global=False):
        """Apply the specified theme to the widget.
        
        Args:
            theme (str): Name of the theme to apply (e.g., 'dark', 'light')
            from_global (bool): Whether this is being called from a global theme change
        """
        try:
            if debug_enabled():
                logger.debug(f"[MonitorOverlay] Applying theme: {theme}, from_global: {from_global}")
            
            self.theme = theme.lower()
            
            # Store current opacity before applying theme
            current_border_opacity = getattr(self, 'border_opacity', 1.0)
            
            # Set up colors based on theme
            if self.theme == "dark":
                border_color = QColor(255, 255, 255)  # White for dark theme
                stylesheet = """
                    QCheckBox {
                        color: white;
                        padding: 4px;
                    }
                """
            else:
                border_color = QColor(0, 0, 0)  # Black for light theme
                stylesheet = """
                    QCheckBox {
                        color: black;
                        padding: 4px;
                    }
                """
            
            # Apply the border color with preserved opacity
            self.border_pen = QPen(border_color, 2)
            border_color.setAlphaF(current_border_opacity)
            self.border_pen.setColor(border_color)
            
            # Apply stylesheet
            self.setStyleSheet(stylesheet)
            self.update()
            
            if debug_enabled():
                logger.debug(f"[MonitorOverlay] Theme applied with border opacity: {current_border_opacity:.3f}")
                
        except Exception as e:
            logger.error(f"[MonitorOverlay] Error applying theme: {e}", exc_info=debug_enabled())

    def _setup_mss_monitor_mapping(self):
        try:
            mss_instance = mss.mss()
            mss_monitors = mss_instance.monitors
            current_screen = self.capture_target_screen or QGuiApplication.primaryScreen()
            if not current_screen:
                self._mss_monitor_idx = 1
                logger.warning("No current screen, defaulting to MSS monitor 1")
                return False
            
            current_geo = current_screen.geometry()
            center_x = current_geo.x() + current_geo.width() // 2
            center_y = current_geo.y() + current_geo.height() // 2
            logger.debug(f"Current screen {current_screen.name()} center: ({center_x}, {center_y})")
            
            for idx, monitor in enumerate(mss_monitors[1:], start=1):
                logger.debug(f"MSS monitor {idx}: left={monitor['left']}, top={monitor['top']}, width={monitor['width']}, height={monitor['height']}")
                if (monitor['left'] <= center_x < monitor['left'] + monitor['width'] and
                    monitor['top'] <= center_y < monitor['top'] + monitor['height']):
                    self._mss_monitor_idx = idx
                    logger.debug(f"Mapped screen {current_screen.name()} to MSS monitor {idx} by point containment")
                    return True
            
            primary_screen = QGuiApplication.primaryScreen()
            if primary_screen:
                primary_geo = primary_screen.geometry()
                primary_center_x = primary_geo.x() + primary_geo.width() // 2
                primary_center_y = primary_geo.y() + primary_geo.height() // 2
                for idx, monitor in enumerate(mss_monitors[1:], start=1):
                    if (monitor['left'] <= primary_center_x < monitor['left'] + monitor['width'] and
                        monitor['top'] <= primary_center_y < monitor['top'] + monitor['height']):
                        self._mss_monitor_idx = idx
                        logger.debug(f"Fallback to primary monitor MSS index {idx}")
                        return True
            
            self._mss_monitor_idx = 1
            logger.warning("No matching MSS monitor found, using first physical monitor")
            return False
        except Exception as e:
            logger.error(f"Error in monitor mapping: {e}", exc_info=True)
            self._mss_monitor_idx = 1
            return False

    def _init_context_menu(self):
        """Initialize the context menu using the unified OverlayContextMenu."""
        try:
            from overlay_context_menu import OverlayContextMenu
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self.show_context_menu)
            
            # Initialize the menus first
            self.switch_to_window_menu = QMenu("Switch to Window", self)
            self.switch_to_monitor_menu = QMenu("Switch to Monitor", self)
            
            # Create the context menu builder with the initialized menus
            self._context_menu_builder = OverlayContextMenu(
                self, 
                overlay_type='monitor',
                config={
                    'show_switch_to_window': True,
                    'show_switch_to_monitor': True,
                    'switch_to_window_menu': self.switch_to_window_menu,
                    'switch_to_monitor_menu': self.switch_to_monitor_menu
                }
            )
            
            # Connect menu about to show signals
            self.switch_to_window_menu.aboutToShow.connect(self._populate_switch_window_menu)
            self.switch_to_monitor_menu.aboutToShow.connect(self._populate_switch_monitor_menu)
            
        except Exception as e:
            logger.error(f"Error initializing context menu: {e}", exc_info=True)
    
    def _apply_context_menu_theme(self):
        """Apply theme to the context menu (handled by OverlayContextMenu)."""
        if hasattr(self, '_context_menu_builder'):
            self._context_menu_builder.apply_theme(self.theme, from_global=True)
    
    def _handle_show_settings(self):
        if self.app_instance:
            self.app_instance.show_settings()
    
    def _handle_show_sub_settings(self):
        if self.app_instance and hasattr(self.app_instance, '_show_sub_settings'):
            dialog = self.app_instance._show_sub_settings()
            # Connect to the border opacity changed signal if available
            if hasattr(dialog, 'border_opacity_changed'):
                dialog.border_opacity_changed.connect(self.set_border_opacity)
            # Load current border opacity if available
            if hasattr(dialog, 'border_opacity_slider'):
                self.set_border_opacity(dialog.border_opacity_slider.value())
    
    def _init_ui(self, initial_geometry=None):
        try:
            if debug_enabled():
                logger.debug("[MonitorOverlay] Initializing UI with theme: %s", self.theme)
            
            # Set up window flags
            flags = Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
            click_through_enabled = False
            
            # Get click-through setting from app settings if available
            if self.app_instance and hasattr(self.app_instance, 'settings'):
                click_through_enabled = self.app_instance.settings.value("click_through_enabled", False, type=bool)
                
                # Opacity is already set in __init__, ensure it's marked as initialized
                if not hasattr(self, '_opacity_initialized') or not self._opacity_initialized:
                    self._opacity_initialized = True
                    logger.debug(f"Using pre-initialized opacities - Overlay: {self.opacity*100:.0f}%")
            
            # Set window attributes
            self.setWindowFlags(flags)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_NoSystemBackground)
            self.setAttribute(Qt.WA_AcceptTouchEvents, False)
            self.setAttribute(Qt.WA_NoMousePropagation, not click_through_enabled)
            self.setAttribute(Qt.WA_MouseNoMask, not click_through_enabled)
            self.setMouseTracking(True)
            self.setCursor(Qt.ArrowCursor)
            
            # Apply the current theme (which will set up border pen with correct opacity)
            self.apply_theme(self.theme)
            
            # Set window opacity and focus policy
            self.setWindowOpacity(self.opacity)
            self.setFocusPolicy(Qt.StrongFocus)
            self.setAttribute(Qt.WA_ShowWithoutActivating, True)
            
            # Set initial size and position
            if initial_geometry and isinstance(initial_geometry, QRect):
                self.setGeometry(initial_geometry)
            else:
                self.resize(800, 450)
                
            # Initialize context menu and ensure window is on screen
            self._init_context_menu()
            self.ensure_in_monitor_bounds(self._display_screen)
            
            if debug_enabled():
                logger.debug(
                    f"[MonitorOverlay] UI initialized - Opacity: {self.opacity:.3f}, "
                    f"Border Opacity: {getattr(self, 'border_opacity', 1.0):.3f}, "
                    f"Click-through: {click_through_enabled}"
                )
                
        except Exception as e:
            logger.error(f"[MonitorOverlay] Error initializing UI: {e}", exc_info=debug_enabled())
            # Ensure basic functionality even if there's an error
            self.resize(800, 450)
            self.setWindowOpacity(1.0)
        
    def _init_capture_worker(self):
        if not self.capture_target_screen:
            logger.error("No target screen set for capture worker")
            return
        self._capture_worker = CaptureWorker(screen=self.capture_target_screen)
        self._worker_thread = QThread()
        self._capture_worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._capture_worker.run)
        self._capture_worker.frame_ready.connect(self._on_frame_ready)
        self._capture_worker.finished.connect(self._worker_thread.quit)
        self._capture_worker.finished.connect(self._capture_worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()
        self._update_capture_params()
        self.show()
        
    def _on_frame_ready(self, frame_data, width, height, dpr):
        try:
            if frame_data is None or frame_data.size == 0:
                logger.warning("Received empty frame data")
                return
            frame_height, frame_width, channels = frame_data.shape
            if channels != 3:
                logger.warning(f"Unexpected channel count: {channels}")
                return
            bytes_per_line = 3 * frame_width
            qimage = QImage(frame_data.data, frame_width, frame_height, bytes_per_line, QImage.Format_RGB888)
            if qimage.isNull():
                logger.warning("Failed to create QImage from frame data")
                return
            self._pixmap = QPixmap.fromImage(qimage)
            if self.capture_target_screen:
                self._pixmap.setDevicePixelRatio(dpr)
            self.update()
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def _update_capture_params(self):
        if not self._capture_worker or not self.capture_target_screen:
            return
        geometry = self.geometry()
        self._capture_worker.set_capture_params(
            monitor_idx=self._mss_monitor_idx,
            width=geometry.width(),
            height=geometry.height(),
            screen=self.capture_target_screen
        )
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        current_physical_screen = QGuiApplication.screenAt(self.geometry().center()) or QGuiApplication.primaryScreen()
        widget_rect = self.rect()
        bg_color = QColor(30, 30, 30) if self.theme == "dark" else QColor(240, 240, 240)
        painter.fillRect(widget_rect, bg_color)
        
        if current_physical_screen == self.capture_target_screen:
            painter.setPen(QColor("white"))
            font = QFont()
            font.setBold(True)
            font.setPixelSize(16)
            painter.setFont(font)
            text = "CURRENT MONITOR"
            text_rect = QRect(0, 0, widget_rect.width(), widget_rect.height() // 2)
            painter.drawText(text_rect, Qt.AlignCenter, text)
            
            font.setBold(False)
            font.setPixelSize(10)
            painter.setFont(font)
            subtext = "DRAG TO A DIFFERENT DISPLAY"
            subtext_rect = QRect(0, widget_rect.height() // 2, widget_rect.width(), widget_rect.height() // 2)
            painter.drawText(subtext_rect, Qt.AlignCenter, subtext)
        else:
            if self._pixmap and not self._pixmap.isNull():
                pixmap_size = self._pixmap.size()
                pixmap_ratio = pixmap_size.width() / pixmap_size.height()
                widget_ratio = widget_rect.width() / widget_rect.height()
                if pixmap_ratio > widget_ratio:
                    new_width = widget_rect.width()
                    new_height = int(new_width / pixmap_ratio)
                    new_x = 0
                    new_y = (widget_rect.height() - new_height) // 2
                else:
                    new_height = widget_rect.height()
                    new_width = int(new_height * pixmap_ratio)
                    new_x = (widget_rect.width() - new_width) // 2
                    new_y = 0
                target_rect = QRect(new_x, new_y, new_width, new_height)
                painter.drawPixmap(target_rect, self._pixmap, self._pixmap.rect())
                border_color = QColor(255, 255, 255) if self.theme == "dark" else QColor(0, 0, 0)
                border_pen = QPen(border_color)
                border_pen.setWidth(1)
                painter.setPen(border_pen)
                painter.drawRect(target_rect.adjusted(0, 0, -1, -1))
            else:
                error_bg = QColor(40, 40, 40) if self.theme == "dark" else QColor(220, 220, 220)
                error_text = QColor(255, 255, 255) if self.theme == "dark" else QColor(0, 0, 0)
                painter.fillRect(widget_rect, error_bg)
                painter.setPen(QPen(error_text, 1))
                painter.drawText(widget_rect, Qt.AlignCenter, "No Content / Error")
        
        # Draw the border with the current border opacity
        if self.border_pen:
            # The border pen's color already has the correct alpha from set_border_opacity
            painter.setPen(self.border_pen)
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        logger.debug(f"Mouse press at {event.position().toPoint()}, global: {event.globalPosition().toPoint()}")
        if event.button() == Qt.RightButton:
            self.show_context_menu(event.position().toPoint())
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._drag_state = snap_utils.handle_overlay_mouse_press(event, self)
            logger.debug(f"Drag state after press: {self._drag_state}")
            if not self._drag_state['is_resizing']:
                self._is_snapped = False
                logger.info(f"Starting drag with offset: {self._drag_state.get('drag_offset')}")
            event.accept()
        else:
            logger.debug("Non-left/right button press, passing to parent")
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if hasattr(self, '_drag_state'):
                if snap_utils.handle_overlay_mouse_release(event, self, self._drag_state):
                    self.ensure_in_monitor_bounds()
                    event.accept()
                    return
            self._is_snapped = False
            self.ensure_in_monitor_bounds()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        scale = 1.1 if delta > 0 else 0.9
        new_w = max(100, int(self.width() * scale))
        new_h = max(100, int(self.height() * scale))
        self.resize(new_w, new_h)
        self.ensure_in_monitor_bounds()
        self._update_capture_params()
        event.accept()

    def _get_resize_edge(self, pos):
        return snap_utils.get_resize_edge_for_pos(pos, self, self._edge_margin)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position().toPoint()
        global_pos = event.globalPosition().toPoint()
        if not hasattr(self, '_drag_state') or not any([
            self._drag_state.get('is_resizing'),
            self._drag_state.get('drag_offset') is not None
        ]):
            edge = snap_utils.get_resize_edge_for_pos(pos, self, self._edge_margin)
            logger.debug(f"Mouse move - pos: {pos}, edge: {edge}")
            if edge:
                if ('top' in edge and 'left' in edge) or ('bottom' in edge and 'right' in edge):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif ('top' in edge and 'right' in edge) or ('bottom' in edge and 'left' in edge):
                    self.setCursor(Qt.SizeBDiagCursor)
                elif 'left' in edge or 'right' in edge:
                    self.setCursor(Qt.SizeHorCursor)
                elif 'top' in edge or 'bottom' in edge:
                    self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        if hasattr(self, '_drag_state'):
            logger.debug(f"Before handle_overlay_mouse_move - drag_state: {self._drag_state}")
            handled = snap_utils.handle_overlay_mouse_move(event, self, self._drag_state, self.snap_distance)
            logger.debug(f"After handle_overlay_mouse_move - handled: {handled}, drag_state: {self._drag_state}")
            if handled:
                if hasattr(self, '_capture_worker'):
                    self._update_capture_params()
                event.accept()
                return
        else:
            logger.debug("No _drag_state attribute found")
        super().mouseMoveEvent(event)

    def _fallback_ensure_in_monitor_bounds(self, screen):
        if not screen:
            return
        try:
            screen_geo = screen.availableGeometry()
            window_geo = self.geometry()
            x = window_geo.x()
            y = window_geo.y()
            w = window_geo.width()
            h = window_geo.height()
            new_x, new_y = x, y
            if x < screen_geo.left():
                new_x = screen_geo.left()
            elif x + w > screen_geo.right():
                new_x = screen_geo.right() - w
            if y < screen_geo.top():
                new_y = screen_geo.top()
            elif y + h > screen_geo.bottom():
                new_y = screen_geo.bottom() - h
            if new_x != x or new_y != y:
                logger.debug(f"Fallback: Adjusted position from ({x},{y}) to ({new_x},{new_y}) logical")
                self.move(new_x, new_y)
        except Exception as e:
            logger.error(f"Error in fallback_ensure_in_monitor_bounds: {e}")
            
    def ensure_in_monitor_bounds(self, target_screen=None):
        if self._is_resizing:
            return
        screen_to_use = target_screen or self.screen() or QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        if not screen_to_use:
            logger.warning("ensure_in_monitor_bounds: Could not determine screen.")
            return
        try:
            monitor_info = get_physical_monitor_info(screen_to_use)
            if not monitor_info:
                logger.error("Could not get monitor info, using fallback method")
                self._fallback_ensure_in_monitor_bounds(screen_to_use)
                return
            phys_width = monitor_info.get('width', 0)
            phys_height = monitor_info.get('height', 0)
            phys_pos = monitor_info.get('position', QPoint(0, 0))
            phys_x = phys_pos.x()
            phys_y = phys_pos.y()
            if phys_width <= 0 or phys_height <= 0:
                logger.error(f"Invalid physical dimensions: {phys_width}x{phys_height}, using fallback")
                self._fallback_ensure_in_monitor_bounds(screen_to_use)
                return
            screen_geo = screen_to_use.availableGeometry()
            window_geo = self.geometry()
            x, y = window_geo.x(), window_geo.y()
            w, h = window_geo.width(), window_geo.height()
            scale_x = phys_width / screen_geo.width() if screen_geo.width() > 0 else 1.0
            scale_y = phys_height / screen_geo.height() if screen_geo.height() > 0 else 1.0
            x_phys = int((x - screen_geo.x()) * scale_x)
            y_phys = int((y - screen_geo.y()) * scale_y)
            w_phys = int(w * scale_x)
            h_phys = int(h * scale_y)
            logger.debug(f"Bounds check - Screen: {screen_to_use.name() if hasattr(screen_to_use, 'name') else 'Unknown'}, "
                       f"Physical: {phys_width}x{phys_height}, "
                       f"Current: {x_phys},{y_phys} {w_phys}x{h_phys}")
            new_x_phys, new_y_phys = x_phys, y_phys
            if x_phys < 0:
                new_x_phys = 0
            if y_phys < 0:
                new_y_phys = 0
            if x_phys + w_phys > phys_width:
                new_x_phys = max(0, phys_width - w_phys)
            if y_phys + h_phys > phys_height:
                new_y_phys = max(0, phys_height - h_phys)
            if new_x_phys != x_phys or new_y_phys != y_phys:
                new_x = int(new_x_phys / scale_x) + screen_geo.x()
                new_y = int(new_y_phys / scale_y) + screen_geo.y()
                logger.debug(f"Adjusted position from ({x},{y}) to ({new_x},{new_y}) logical")
                self.move(new_x, new_y)
        except Exception as e:
            logger.error(f"Error in ensure_in_monitor_bounds: {e}", exc_info=True)
            # Fall back to the simpler method if there's an error
            self._fallback_ensure_in_monitor_bounds(screen_to_use)
                
    def resizeEvent(self, event):
        """Handle window resize events to ensure opacity persistence.
        
        This method is called whenever the window is resized. It ensures that the 
        window and border opacity settings are reapplied after the resize operation
        is complete, which prevents opacity loss during resizing.
        
        Args:
            event: The resize event
        """
        # Call parent's resizeEvent first
        super().resizeEvent(event)
        
        try:
            # Get current opacity values
            current_opacity = getattr(self, 'opacity', 1.0)
            current_border_opacity = getattr(self, 'border_opacity', 1.0)
            
            if debug_enabled():
                logger.debug(f"[MonitorOverlay] Resize event - Reapplying opacity: {current_opacity:.3f}, "
                           f"border: {current_border_opacity:.3f}")
            
            # Reapply the current opacity to ensure it persists after resize
            # Use emit_signal=False to prevent signal loops
            self.set_overlay_opacity(current_opacity, emit_signal=False)
            self.set_border_opacity(current_border_opacity, emit_signal=False)
            
            # Update capture parameters after resize
            if hasattr(self, '_update_capture_params'):
                self._update_capture_params()
                
        except Exception as e:
            logger.error(f"[MonitorOverlay] Error in resizeEvent: {e}", exc_info=debug_enabled())
    
    def _fallback_ensure_in_monitor_bounds(self, screen):
        """Fallback method to ensure window stays within monitor bounds."""
        if not screen:
            return
        try:
            screen_geo = screen.availableGeometry()
            window_geo = self.geometry()
            x = window_geo.x()
            y = window_geo.y()
            w = window_geo.width()
            h = window_geo.height()
            new_x, new_y = x, y
            if x < screen_geo.left():
                new_x = screen_geo.left()
            elif x + w > screen_geo.right():
                new_x = screen_geo.right() - w
            if y < screen_geo.top():
                new_y = screen_geo.top()
            elif y + h > screen_geo.bottom():
                new_y = screen_geo.bottom() - h
            if new_x != x or new_y != y:
                logger.debug(f"Fallback: Adjusted position from ({x},{y}) to ({new_x},{new_y}) logical")
                self.move(new_x, new_y)
        except Exception as e:
            logger.error(f"Error in fallback_ensure_in_monitor_bounds: {e}")
    
    def ensure_in_monitor_bounds(self, target_screen=None):
        """Ensure the window stays within the bounds of the target screen."""
        if getattr(self, '_is_resizing', False):
            return
            
        screen_to_use = target_screen or self.screen() or QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        if not screen_to_use:
            logger.warning("ensure_in_monitor_bounds: Could not determine screen.")
            return
            
        try:
            monitor_info = get_physical_monitor_info(screen_to_use)
            if not monitor_info:
                logger.error("Could not get monitor info, using fallback method")
                self._fallback_ensure_in_monitor_bounds(screen_to_use)
                return
                
            phys_width = monitor_info.get('width', 0)
            phys_height = monitor_info.get('height', 0)
            phys_pos = monitor_info.get('position', QPoint(0, 0))
            
            if phys_width <= 0 or phys_height <= 0:
                logger.error(f"Invalid physical dimensions: {phys_width}x{phys_height}, using fallback")
                self._fallback_ensure_in_monitor_bounds(screen_to_use)
                return
                
            screen_geo = screen_to_use.availableGeometry()
            window_geo = self.geometry()
            x, y = window_geo.x(), window_geo.y()
            w, h = window_geo.width(), window_geo.height()
            
            scale_x = phys_width / screen_geo.width() if screen_geo.width() > 0 else 1.0
            scale_y = phys_height / screen_geo.height() if screen_geo.height() > 0 else 1.0
            
            x_phys = int((x - screen_geo.x()) * scale_x)
            y_phys = int((y - screen_geo.y()) * scale_y)
            w_phys = int(w * scale_x)
            h_phys = int(h * scale_y)
            
            logger.debug(
                f"Bounds check - Screen: {getattr(screen_to_use, 'name', 'Unknown')}, "
                f"Physical: {phys_width}x{phys_height}, Current: {x_phys},{y_phys} {w_phys}x{h_phys}"
            )
            
            new_x_phys, new_y_phys = x_phys, y_phys
            if x_phys < 0:
                new_x_phys = 0
            if y_phys < 0:
                new_y_phys = 0
            if x_phys + w_phys > phys_width:
                new_x_phys = max(0, phys_width - w_phys)
            if y_phys + h_phys > phys_height:
                new_y_phys = max(0, phys_height - h_phys)
                
            if new_x_phys != x_phys or new_y_phys != y_phys:
                new_x = int(new_x_phys / scale_x) + screen_geo.x()
                new_y = int(new_y_phys / scale_y) + screen_geo.y()
                logger.debug(f"Adjusted position from ({x},{y}) to ({new_x},{new_y}) logical")
                self.move(new_x, new_y)
                
        except Exception as e:
            logger.error(f"Error in ensure_in_monitor_bounds: {e}", exc_info=True)
            # Fall back to the simpler method if there's an error
            self._fallback_ensure_in_monitor_bounds(screen_to_use)
    
    def show_context_menu(self, position):
        """Show the context menu using the unified OverlayContextMenu.
        
        Args:
            position: The position where the context menu was requested (in widget coordinates)
        """
        try:
            # Ensure context menu is initialized
            if not hasattr(self, '_context_menu_builder'):
                logger.warning("Context menu builder not initialized, initializing now")
                self._init_context_menu()
                
            if hasattr(self, '_context_menu_builder'):
                # Update menu items that might have changed
                if hasattr(self, '_populate_switch_window_menu'):
                    self._populate_switch_window_menu()
                if hasattr(self, '_populate_switch_monitor_menu'):
                    self._populate_switch_monitor_menu()
                
                # Apply theme and show the menu
                self._apply_context_menu_theme()
                self._context_menu_builder.show_menu(position)
                logger.debug("Context menu shown successfully")
            else:
                logger.error("Failed to initialize context menu builder")
                
        except Exception as e:
            logger.error(f"Failed to show context menu: {e}", exc_info=True)
            try:
                # Fallback: Create a minimal context menu
                from PySide6.QtWidgets import QMenu
                menu = QMenu(self)
                menu.addAction("Close", self.close)
                menu.exec(self.mapToGlobal(position))
            except Exception as fallback_error:
                logger.error(f"Fallback context menu failed: {fallback_error}", exc_info=True)

    def _handle_reset_position(self):
        logger.debug("Resetting monitor overlay position and size")
        if not hasattr(self, 'app_instance') or not self.app_instance or not hasattr(self.app_instance, 'settings'):
            logger.warning("Cannot reset position: app_instance or settings not available")
            return
        current_screen = self._display_screen or self.capture_target_screen or QGuiApplication.primaryScreen()
        if not current_screen:
            logger.error("Could not determine current screen for reset")
            return
        try:
            screens = QGuiApplication.screens()
            monitor_idx = screens.index(current_screen) if current_screen in screens else 0
            preset_key = f"MonitorPresets/Monitor_{monitor_idx}_Preset"
            position_preset = self.app_instance.settings.value(preset_key, "Center")
            if hasattr(self.app_instance, 'calculate_position_geometry'):
                new_geometry = self.app_instance.calculate_position_geometry(
                    current_screen, position_preset, "monitor"
                )
                if new_geometry and new_geometry.isValid():
                    logger.info(f"Resetting monitor overlay to {position_preset} position: {new_geometry}")
                    self.setGeometry(new_geometry)
                    self._update_capture_params()
                    return
            logger.warning("Failed to calculate position from preset, using default position")
            screen_geo = current_screen.availableGeometry()
            default_geometry = QRect(
                screen_geo.x() + screen_geo.width() // 4,
                screen_geo.y() + screen_geo.height() // 4,
                screen_geo.width() // 2,
                screen_geo.height() // 2
            )
            self.setGeometry(default_geometry)
            self._update_capture_params()
        except Exception as e:
            logger.error(f"Error in _handle_reset_position: {e}", exc_info=True)
            if current_screen:
                screen_geo = current_screen.availableGeometry()
                self.move(screen_geo.topLeft())
                self.resize(800, 600)
                self._update_capture_params()

    def _get_display_name(self, screen, idx):
        try:
            name = screen.name().strip() or f"Display {idx + 1}"
            geom = screen.geometry()
            dpi = screen.logicalDotsPerInch()
            refresh_rate = screen.refreshRate()
            manufacturer = screen.manufacturer().strip()
            model = screen.model().strip()
            display_info = f"{name} - {geom.width()}x{geom.height()}"
            if dpi > 0:
                display_info += f" @ {int(dpi)} DPI"
            if refresh_rate > 0:
                display_info += f" ({refresh_rate:.0f}Hz)"
            if manufacturer or model:
                display_info += f"\n{manufacturer} {model}".strip()
            return display_info
        except Exception as e:
            logger.warning(f"Error getting display info: {e}")
            return f"Display {idx + 1}"

    def _populate_switch_monitor_menu(self):
        self.switch_to_monitor_menu.clear()
        try:
            monitors = get_all_monitors()
            if not monitors:
                no_screens_action = QAction("No screens available", self.switch_to_monitor_menu)
                no_screens_action.setEnabled(False)
                self.switch_to_monitor_menu.addAction(no_screens_action)
                return
            current_screen_name = self.capture_target_screen.name() if hasattr(self.capture_target_screen, 'name') else ""
            for monitor in monitors:
                screen = monitor.get('screen_object')
                if not screen or not hasattr(screen, 'name'):
                    continue
                if screen.name() == current_screen_name:
                    continue
                display_name = monitor.get('display_name', '')
                if not display_name:
                    display_name = f"{screen.name()} ({screen.geometry().width()}x{screen.geometry().height()})"
                if 'dpi' in monitor and isinstance(monitor['dpi'], (tuple, list)) and len(monitor['dpi']) >= 2:
                    dpi_x, dpi_y = monitor['dpi']
                    display_name += f" @ {int(dpi_x)} DPI"
                screen_action = QAction(display_name, self.switch_to_monitor_menu)
                screen_action.triggered.connect(lambda checked=False, s=screen: self._handle_swap_screen(s))
                self.switch_to_monitor_menu.addAction(screen_action)
        except Exception as e:
            logger.error(f"Error populating monitor menu: {e}", exc_info=True)
            error_action = QAction("Error loading monitors", self.switch_to_monitor_menu)
            error_action.setEnabled(False)
            self.switch_to_monitor_menu.addAction(error_action)

    def _populate_switch_window_menu(self):
        """Populate the Switch To Window submenu with available windows."""
        self.switch_to_window_menu.clear()
        windows_data = []
        
        # Save current opacity before any operations that might affect it
        current_opacity = getattr(self, 'opacity', 1.0)
        
        try:
            # Try overlay-provided method first
            if hasattr(self, 'app_instance') and hasattr(self.app_instance, 'get_menu_ready_windows'):
                windows_data = self.app_instance.get_menu_ready_windows()
            elif hasattr(self, 'get_menu_ready_windows'):
                windows_data = self.get_menu_ready_windows()
                
            # Fallback: empty
            if not windows_data:
                action = QAction("No other windows found", self.switch_to_window_menu)
                action.setEnabled(False)
                self.switch_to_window_menu.addAction(action)
                return
                
            for hwnd, title, icon in windows_data:
                display_title = title.strip() if len(title.strip()) < 60 else title[:57] + "..."
                if not display_title:
                    display_title = f"[No Title] ({hwnd})"
                action = QAction(display_title, self.switch_to_window_menu)
                if icon and hasattr(icon, 'isNull') and not icon.isNull():
                    action.setIcon(icon)
                action.setData(hwnd)
                
                # Create a closure to preserve the current opacity when the action is triggered
                def create_swap_handler(hwnd, opacity):
                    def handler():
                        # Store the current opacity before switching
                        if hasattr(self, 'app_instance') and hasattr(self.app_instance, 'settings'):
                            self.app_instance.settings.setValue("overlay_opacity", int(opacity * 100))
                        # Call the swap method
                        if hasattr(self, '_handle_swap_window'):
                            self._handle_swap_window(hwnd)
                        else:
                            self._handle_initiate_window_swap(hwnd)
                    return handler
                
                # Connect the action with the current opacity
                action.triggered.connect(create_swap_handler(hwnd, current_opacity))
                self.switch_to_window_menu.addAction(action)
                
        except Exception as e:
            logger.error(f"Error populating window menu: {e}")
            error_action = QAction("Error loading windows", self.switch_to_window_menu)
            error_action.setEnabled(False)
            self.switch_to_window_menu.addAction(error_action)

    def _validate_hwnd(self, hwnd):
        """Validate that the HWND is valid and the window is available for capture."""
        try:
            if not hwnd or not isinstance(hwnd, int) or hwnd == 0:
                logger.error(f"Invalid window handle: {hwnd}")
                return False
                
            # Skip validation if win32gui is not available
            try:
                import win32gui
                import win32con
                
                if not win32gui.IsWindow(hwnd):
                    logger.error(f"Window handle {hwnd} is not a valid window")
                    return False
                    
                if not win32gui.IsWindowVisible(hwnd):
                    logger.error(f"Window {hwnd} is not visible")
                    return False
                    
                # Check if window is minimized
                if win32gui.IsIconic(hwnd):
                    logger.error(f"Window {hwnd} is minimized")
                    return False
                    
                # Check if window is on a visible display
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    window_rect = QRect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
                    
                    # Check if any part of the window is on a visible display
                    window_visible = False
                    for screen in QGuiApplication.screens():
                        screen_geo = screen.geometry()
                        if window_rect.intersects(screen_geo):
                            window_visible = True
                            break
                            
                    if not window_visible:
                        logger.error(f"Window {hwnd} is not on a visible display")
                        return False
                        
                except Exception as e:
                    logger.warning(f"Could not verify window position: {e}")
                    
            except ImportError:
                logger.warning("win32gui not available, skipping advanced HWND validation")
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating HWND {hwnd}: {e}", exc_info=True)
            return False
    
    def _create_window_overlay_safely(self, hwnd, opacity, theme, geometry):
        """Safely create a window overlay with error handling and fallbacks."""
        try:
            # Try to create the window overlay (hidden initially)
            if hasattr(self.app_instance, 'create_window_overlay'):
                logger.debug("Using create_window_overlay method")
                return self.app_instance.create_window_overlay(
                    hwnd=hwnd,
                    opacity=opacity,
                    theme=theme,
                    initial_geometry=geometry,
                    visible=False  # Create hidden first
                )
                
            elif hasattr(self.app_instance, 'prepare_to_create_window_overlay'):
                logger.debug("Using prepare_to_create_window_overlay method")
                if hasattr(self.app_instance, 'window_geometry'):
                    self.app_instance.window_geometry = geometry
                if hasattr(self.app_instance, 'window_opacity'):
                    self.app_instance.window_opacity = opacity
                if hasattr(self.app_instance, 'window_theme'):
                    self.app_instance.window_theme = theme
                    
                return self.app_instance.prepare_to_create_window_overlay(hwnd)
                
            else:
                logger.error("No suitable method found to create window overlay")
                return False
                
        except Exception as e:
            logger.error(f"Error creating window overlay: {e}", exc_info=True)
            return False
    
    def _handle_swap_window(self, hwnd):
        """
        Handle window swap by delegating to the app instance.
        
        Args:
            hwnd: The window handle to switch to
        """
        if self.app_instance and hasattr(self.app_instance, 'prepare_to_create_window_overlay'):
            self.app_instance.prepare_to_create_window_overlay(hwnd)
        else:
            logger.error("app_instance or prepare_to_create_window_overlay method not found.")
    
    def _handle_initiate_window_swap(self, hwnd):
        """
        Handle the window swap from monitor overlay to window overlay.
        This is a legacy method that delegates to _handle_swap_window.
        
        Args:
            hwnd: The window handle to switch to
        """
        self._handle_swap_window(hwnd)

    def _handle_swap_screen(self, new_screen):
        if not new_screen or new_screen == self.capture_target_screen:
            return
        try:
            screen_name = new_screen.name() if hasattr(new_screen, 'name') else 'unknown'
            logger.info(f"Attempting to swap to screen '{screen_name}'")
            old_screen = self.capture_target_screen
            self.capture_target_screen = new_screen
            self._pixmap = None
            if self._setup_mss_monitor_mapping():
                self._display_screen = self._select_display_screen()
                logger.debug(f"Updated display screen to: {self._display_screen.name() if hasattr(self._display_screen, 'name') else 'unknown'}")
                self._position_top_left_on_screen(self._display_screen)
                self._update_capture_params()
                self.ensure_in_monitor_bounds(self._display_screen)
                self.update()
                QTimer.singleShot(100, self._reapply_mouse_settings)
                display_name = self._display_screen.name() if hasattr(self._display_screen, 'name') else 'unknown'
                logger.info(f"Successfully swapped to screen '{screen_name}', displayed on '{display_name}'")
            else:
                logger.error(f"Failed to map screen '{screen_name}'")
                self.capture_target_screen = old_screen or QGuiApplication.primaryScreen()
                self._display_screen = self._select_display_screen()
                self._setup_mss_monitor_mapping()
                self._position_top_left_on_screen(self._display_screen)
                self._update_capture_params()
                self.ensure_in_monitor_bounds(self._display_screen)
        except Exception as e:
            logger.error(f"Error in _handle_swap_screen: {e}", exc_info=True)
            self.capture_target_screen = QGuiApplication.primaryScreen()
            self._display_screen = self._select_display_screen()
            self._setup_mss_monitor_mapping()
            self._position_top_left_on_screen(self._display_screen)
            self._update_capture_params()
            self.ensure_in_monitor_bounds(self._display_screen)

    def _handle_show_settings(self):
        app = QApplication.instance()
        if app:
            if hasattr(app, '_settings_panel') and app._settings_panel:
                app._settings_panel.show()
                app._settings_panel.activateWindow()
                app._settings_panel.raise_()
            elif hasattr(app, '_show_settings'):
                app._show_settings()

    def _handle_show_sub_settings(self):
        logger.debug("Show sub-settings action triggered")
        if self.app_instance and hasattr(self.app_instance, '_show_sub_settings'):
            try:
                logger.debug("Using application's _show_sub_settings method")
                self.app_instance._show_sub_settings()
                return
            except Exception as e:
                logger.error(f"Error in app._show_sub_settings(): {e}")
        try:
            from subsettings_dialog import SubSettingsDialog
            logger.debug("Creating SubSettingsDialog directly")
            dialog = SubSettingsDialog(parent=self, app_instance=self.app_instance)
            dialog.setWindowFlags(
                Qt.Dialog | 
                Qt.WindowTitleHint | 
                Qt.WindowCloseButtonHint |
                Qt.WindowStaysOnTopHint |
                Qt.WindowSystemMenuHint |
                Qt.WindowMinMaxButtonsHint
            )
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.setModal(False)
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()
            dialog.resize(400, 300)
            x = screen_geometry.x() + (screen_geometry.width() - dialog.width()) // 2
            y = screen_geometry.y() + (screen_geometry.height() - dialog.height()) // 2
            x = max(screen_geometry.left(), min(x, screen_geometry.right() - dialog.width()))
            y = max(screen_geometry.top(), min(y, screen_geometry.bottom() - dialog.height()))
            dialog.move(x, y)
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            dialog.setWindowState(dialog.windowState() & ~Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            logger.debug(f"SubSettingsDialog shown at position: {dialog.pos()}, size: {dialog.size()}")
        except ImportError as e:
            logger.error(f"Failed to import SubSettingsDialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load settings dialog: {e}")
        except Exception as e:
            logger.error(f"Unexpected error showing settings dialog: {e}")
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def _reapply_mouse_settings(self):
        try:
            self.setMouseTracking(True)
            self.unsetCursor()
            if hasattr(self, '_drag_state'):
                self._drag_state.update({
                    'is_resizing': False,
                    'resize_edge': None,
                    'drag_start_global': None,
                    'initial_geometry': None,
                    'drag_offset': None
                })
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            edge = self._get_resize_edge(cursor_pos)
            if edge:
                self.setCursor(edge_to_cursor(edge))
            else:
                self.unsetCursor()
            logger.debug("Mouse settings reapplied after screen swap")
        except Exception as e:
            logger.error(f"Error reapplying mouse settings: {e}", exc_info=True)

    @property
    def hwnd(self):
        if not hasattr(self, '_hwnd') or not self._hwnd:
            self._hwnd = int(self.winId()) if hasattr(self, 'winId') else 0
        return self._hwnd

    def _handle_quit_application(self):
        try:
            self._stop_capture_worker()
            self.close()
            if hasattr(self, 'closed'):
                self.closed.emit()
            if hasattr(self, 'app_instance') and self.app_instance:
                QTimer.singleShot(100, self.app_instance.quit)
        except Exception as e:
            logger.error(f"Error during application quit: {e}", exc_info=True)
        finally:
            if hasattr(self, '_mss_instance'):
                try:
                    self._mss_instance.close()
                except Exception as e:
                    logger.warning(f"Error closing MSS instance: {e}")
            self.deleteLater()

    def closeEvent(self, event):
        logger.info("MonitorOverlay closeEvent triggered")
        try:
            self._stop_capture_worker()
            if hasattr(self, 'mss_instance') and self.mss_instance:
                self.mss_instance.close()
            event.accept()
        except Exception as e:
            logger.error(f"Error during close event: {e}", exc_info=True)
            event.accept()
        finally:
            super().closeEvent(event)

    def _stop_capture_worker(self):
        try:
            if not self._capture_worker:
                return
            self._capture_worker.stop()
            if self._worker_thread and self._worker_thread.isRunning():
                self._worker_thread.quit()
                if not self._worker_thread.wait(1000):
                    logger.warning("Capture worker thread did not stop gracefully, terminating...")
                    self._worker_thread.terminate()
                    if not self._worker_thread.wait(1000):
                        logger.error("Failed to terminate capture worker thread")
                self._worker_thread.deleteLater()
        except Exception as e:
            logger.error(f"Error stopping capture worker: {e}", exc_info=True)
        finally:
            self._capture_worker = None
            self._worker_thread = None
