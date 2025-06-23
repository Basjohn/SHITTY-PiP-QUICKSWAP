import ctypes
import logging
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import mss
import snap_utils
from typing import Optional, Dict, Tuple, Any
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QObject, Signal, QThread, QMutex, QMutexLocker
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPixmap, QScreen, QPainter, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import QMainWindow, QMenu, QApplication
from PySide6.QtGui import QMouseEvent, QAction
from snap_utils import apply_snap


# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    # Set to INFO level to reduce debug output
    logger.setLevel(logging.INFO)
    
    # Suppress mss debug output
    logging.getLogger('mss').setLevel(logging.WARNING)
    # Suppress numpy debug output
    logging.getLogger('numpy').setLevel(logging.WARNING)

class CaptureWorker(QObject):
    """Worker thread for performing screen captures and frame comparison."""
    
    frame_ready = Signal(object, int, int, float)
    finished = Signal()
    
    def __init__(self, screen=None, parent=None):
        super().__init__(parent)
        self._mss_instance = None
        self._mutex = QMutex()
        self._running = False
        self._capture_params = None
        self._last_frame = None
        self._fps = 60  # Default to 60 FPS
        self._screen = screen
        self._monitors = None
        
    def set_fps(self, fps):
        with QMutexLocker(self._mutex):
            try:
                self._fps = max(1, min(240, int(fps)))  # Clamp between 1 and 240
                logger.debug(f"Capture FPS set to {self._fps}")
            except (ValueError, TypeError) as e:
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
        """CPU monitoring is no longer used as it overrides user settings."""
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
                    
                    if monitor_idx < 0 or monitor_idx + 1 >= len(monitors):
                        monitor_idx = 0  # Fallback to primary monitor
                        logger.debug(f"Invalid monitor index, falling back to primary monitor (0)", extra={"suppress": True})
                        
                    monitor = monitors[monitor_idx + 1]
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
                    
                    # Calculate sleep time based on FPS, with a minimum of 1ms
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
    def __init__(self, screen: 'QScreen' = None, opacity: float = 1.0, theme: str = "dark", snap_distance: int = 8, 
                 app_instance=None, initial_geometry=None, monitor_idx: int = None):
        super().__init__()
        self.app_instance = app_instance
        self.opacity = opacity
        self.theme = theme.lower()
        self.snap_distance = snap_distance
        self.mss_instance = mss.mss()
        self._mss_monitor_idx = None
        self.capture_target_screen = screen or QGuiApplication.primaryScreen()
        self._display_screen = None
        self._pixmap = None
        self._edge_margin = 8
        # Initialize drag state for centralized mouse handling
        self._drag_state = {
            'is_resizing': False,
            'resize_edge': None,
            'drag_start_global': None,
            'initial_geometry': None,
            'drag_offset': None
        }
        # Keep legacy attributes for backward compatibility
        self._drag_initial_mouse_pos = None
        self._drag_initial_window_pos = None
        self._is_resizing = False
        self._resize_edge = None
        self._initial_geometry = None
        self._worker_thread = None
        self._capture_worker = None
        self._cached_screen_geometry = None
        self._cached_dpr = 1.0
        self._is_snapped = False  # Track snap state
        self._snap_deadzone = 5  # Pixels to move before unsnapping
        self.border_pen = None
        self.apply_theme(self.theme)
        
        self._setup_mss_monitor_mapping()
        self._display_screen = self._select_display_screen()
        logger.debug(f"Initialized with capture screen: {self.capture_target_screen.name()}, display screen: {self._display_screen.name()}")
        self._init_ui(initial_geometry)
        self._init_capture_worker()
        
    def set_opacity(self, opacity: float):
        """Set the window opacity."""
        self.opacity = opacity
        self.setWindowOpacity(opacity)
        
    def set_overlay_opacity(self, opacity: float):
        """Set the window opacity (backward compatibility).
        
        This is an alias for set_opacity to maintain compatibility with existing code.
        """
        self.set_opacity(opacity)
        
    def set_fps(self, fps: int):
        """Set the FPS for the capture worker."""
        if self._capture_worker:
            self._capture_worker.set_fps(fps)
            logger.debug(f"Set FPS to {fps} for monitor overlay")
        else:
            logger.warning("Cannot set FPS: No capture worker available")
        
    def apply_theme(self, theme, from_global=False):
        """Apply the specified theme to the overlay.
        
        Args:
            theme (str): Name of the theme to apply ('dark' or 'light')
            from_global (bool): Whether this is being called from a global theme change
        """

        
        self.theme = theme.lower()
        
        # Set border color based on theme
        if self.theme == "dark":
            self.border_pen = QPen(QColor(255, 255, 255), 2)  # White border for dark theme
            self.setStyleSheet("""
                QCheckBox {
                    color: white;
                    padding: 4px;
                }
                QLabel {
                    color: white;
                }
            """)
        else:
            self.border_pen = QPen(QColor(0, 0, 0), 2)  # Black border for light theme
            self.setStyleSheet("""
                QCheckBox {
                    color: black;
                    padding: 4px;
                }
                QLabel {
                    color: black;
                }
            """)
            
        self.update()

    def _setup_mss_monitor_mapping(self):
        """Simplified monitor mapping using fallback to primary monitor."""
        try:
            screens = QGuiApplication.screens()
            if not screens:
                logger.warning("No screens found")
                return False
                
            # Use the provided screen or primary screen
            screen = self.capture_target_screen or screens[0]
            
            # Get all available monitors from MSS (skip first one as it's the combined display)
            mss_monitors = self.mss_instance.monitors[1:] if len(self.mss_instance.monitors) > 1 else []
            
            if not mss_monitors:
                logger.warning("No MSS monitors found")
                return False
                
            # Simple mapping: use screen index if valid, otherwise use first monitor
            screen_idx = screens.index(screen) if screen in screens else 0
            self._mss_monitor_idx = min(screen_idx, len(mss_monitors) - 1)
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in monitor mapping: {e}")
            # Fallback to first monitor
            self._mss_monitor_idx = 0
            return True

    def _init_context_menu(self):
        """Initialize the context menu with actions and submenus."""
        try:
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self.show_context_menu)
            self.context_menu = QMenu(self)

            # Create all the menu items
            self.switch_to_window_action = QAction("Switch To Window", self)
            self.switch_to_window_menu = QMenu("Switch To Window", self.context_menu)
            self.switch_to_window_action.setMenu(self.switch_to_window_menu)
            self.context_menu.addAction(self.switch_to_window_action)

            self.switch_to_monitor_action = QAction("Switch To Monitor", self)
            self.switch_to_monitor_menu = QMenu("Switch To Monitor", self.context_menu)
            self.switch_to_monitor_action.setMenu(self.switch_to_monitor_menu)
            self.context_menu.addAction(self.switch_to_monitor_action)
            
            self.context_menu.addSeparator()

            # Store these actions as instance variables for later use
            self.show_settings_action = QAction("Main Window", self)
            self.show_settings_action.triggered.connect(self._handle_show_settings)
            self.context_menu.addAction(self.show_settings_action)

            self.show_sub_settings_action = QAction("Subsettings", self)
            self.show_sub_settings_action.triggered.connect(self._handle_show_sub_settings)
            self.context_menu.addAction(self.show_sub_settings_action)
            
            self.context_menu.addSeparator()

            hide_action = QAction("Hide", self)
            hide_action.triggered.connect(self.close)
            self.context_menu.addAction(hide_action)

            reset_action = QAction("Reset", self)
            reset_action.triggered.connect(self._handle_reset_position)
            self.context_menu.addAction(reset_action)
            
            self.context_menu.addSeparator()

            quit_app_action = QAction("Quit Application", self)
            quit_app_action.triggered.connect(self._handle_quit_application)
            self.context_menu.addAction(quit_app_action)
            
            # Apply the theme to the context menu and submenus
            self._apply_context_menu_theme()
            
        except Exception as e:
            logger.error(f"Error initializing context menu: {e}", exc_info=True)
    
    def _apply_context_menu_theme(self):
        """Apply the current theme to the context menu and its submenus."""
        if not hasattr(self, 'context_menu') or not self.context_menu:
            return
            
        # Get the current theme from instance variable or default to 'dark'
        theme = getattr(self, 'theme', 'dark')
        
        # Apply the stylesheet from our centralized theme
        try:
            from constants import ThemeColors
            stylesheet = ThemeColors.get_theme_stylesheet(theme)
            self.context_menu.setStyleSheet(stylesheet)
            # Also apply to submenus if they exist
            submenus = [
                getattr(self, 'switch_to_window_menu', None),
                getattr(self, 'switch_to_monitor_menu', None)
            ]
            for submenu in submenus:
                if submenu:
                    submenu.setStyleSheet(stylesheet)
        except Exception as e:
            logger.error(f"Error applying theme to context menu: {e}", exc_info=True)
    
    def _handle_show_settings(self):
        """Show the main settings window."""
        if self.app_instance:
            self.app_instance.show_settings()
    
    def _handle_show_sub_settings(self):
        """Show the subsettings dialog."""
        if self.app_instance and hasattr(self.app_instance, '_show_sub_settings'):
            self.app_instance._show_sub_settings()
    
    def _stop_capture_worker(self):
        """Safely stop the capture worker and clean up the thread."""
        try:
            # First, stop the worker if it exists
            if hasattr(self, '_capture_worker') and self._capture_worker is not None:
                try:
                    logger.debug("Stopping capture worker...")
                    self._capture_worker.stop()
                except RuntimeError as e:
                    if 'wrapped C/C++ object' in str(e):
                        logger.debug("Capture worker already deleted, skipping stop")
                    else:
                        logger.debug(f"Error stopping capture worker: {e}")
                except Exception as e:
                    logger.debug(f"Error stopping capture worker: {e}")
            
            # Then clean up the thread
            if hasattr(self, '_worker_thread') and self._worker_thread is not None:
                try:
                    logger.debug("Waiting for worker thread to finish...")
                    if self._worker_thread.isRunning():
                        self._worker_thread.quit()
                        if not self._worker_thread.wait(1000):  # Wait up to 1 second
                            logger.warning("Worker thread did not finish in time, terminating...")
                            self._worker_thread.terminate()
                            self._worker_thread.wait()
                    
                    # Schedule for deletion
                    self._worker_thread.deleteLater()
                except RuntimeError as e:
                    if 'wrapped C/C++ object' in str(e):
                        logger.debug("Worker thread already deleted, skipping cleanup")
                    else:
                        logger.error(f"Error stopping worker thread: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error stopping worker thread: {e}", exc_info=True)
                finally:
                    self._worker_thread = None
            
            # Clean up the worker
            if hasattr(self, '_capture_worker') and self._capture_worker is not None:
                try:
                    self._capture_worker.deleteLater()
                except RuntimeError as e:
                    if 'wrapped C/C++ object' not in str(e):
                        logger.debug(f"Error deleting capture worker: {e}")
                except Exception as e:
                    logger.debug(f"Error deleting capture worker: {e}")
                finally:
                    self._capture_worker = None
                
        except Exception as e:
            logger.error(f"Error in _stop_capture_worker: {e}", exc_info=True)
    
    def closeEvent(self, event):
        """Handle the window close event."""
        logger.debug("Close event received, cleaning up...")
        self._stop_capture_worker()
        event.accept()
    
    def _handle_quit_application(self):
        """Handle the quit application action from the context menu."""
        try:
            # Stop the capture worker and close the window
            self._stop_capture_worker()
            self.close()
            
            # If we have an app instance, try to quit the application
            if hasattr(self, 'app_instance') and self.app_instance:
                QTimer.singleShot(100, self.app_instance.quit)
                
        except Exception as e:
            logger.error(f"Error during application quit: {e}", exc_info=True)
    
    def _handle_reset_position(self):
        """Reset the window to its default position."""
        if self._display_screen:
            screen_geo = self._display_screen.availableGeometry()
            self.move(screen_geo.center() - self.rect().center())
            logger.debug(f"Centered overlay on {self._display_screen.name()} at {screen_geo.center()}")
    
    def _init_ui(self, initial_geometry=None):
        # Set base window flags
        flags = Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        
        # Check if click-through is enabled in settings
        click_through_enabled = False
        if self.app_instance and hasattr(self.app_instance, 'settings'):
            click_through_enabled = self.app_instance.settings.value("click_through_enabled", False, type=bool)
        
        # Add transparent for input if click-through is enabled
        if click_through_enabled:
            flags |= Qt.WindowTransparentForInput
        
        # Apply window flags and attributes
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_AcceptTouchEvents, False)
        self.setAttribute(Qt.WA_NoMousePropagation, not click_through_enabled)  # Allow mouse events to propagate only if not in click-through mode
        self.setAttribute(Qt.WA_MouseNoMask, not click_through_enabled)  # Ensure mouse events are properly received
        self.setMouseTracking(True)  # Enable mouse tracking for hover events
        self.setCursor(Qt.ArrowCursor)
        self.setWindowOpacity(self.opacity)
        
        # Ensure the window is focusable and accepts mouse events
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        
        # Log the initial click-through state for debugging
        logger.debug(f"MonitorOverlay created with click-through: {click_through_enabled}")
        
        logger.info(f"Initializing UI with display screen: {self._display_screen.name() if self._display_screen else 'None'}")
        
        if initial_geometry and isinstance(initial_geometry, QRect):
            self.setGeometry(initial_geometry)
        else:
            self.resize(800, 450)
            if self._display_screen:
                screen_geo = self._display_screen.availableGeometry()
                self.move(screen_geo.center() - self.rect().center())
                logger.debug(f"Centered overlay on {self._display_screen.name()} at {screen_geo.center()}")
        
        self._init_context_menu()
        self.ensure_in_monitor_bounds(self._display_screen)
        
        # Force focus and reapply mouse settings after a short delay
        self.activateWindow()
        QTimer.singleShot(100, self._reapply_mouse_settings)
        
    def _select_display_screen(self):
        """Select a screen different from the capturing screen."""
        screens = QGuiApplication.screens()
        if len(screens) <= 1:
            logger.info("Only one screen available, using it for display")
            return screens[0] if screens else None
        
        for screen in screens:
            if screen != self.capture_target_screen:
                logger.debug(f"Selected display screen: {screen.name()} (capture screen: {self.capture_target_screen.name()})")
                return screen
        logger.warning("No alternative screen found, using primary screen")
        return QGuiApplication.primaryScreen()

    def _reapply_mouse_settings(self):
        """Reapply mouse tracking and cursor to ensure interactivity."""
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)
        logger.info("Reapplied mouse settings")
        
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
        
        # Initialize variables to avoid UnboundLocalError
        new_x = 0
        new_y = 0
        new_width = 0
        new_height = 0
        
        # Use fully opaque colors since we're handling transparency with setWindowOpacity()
        if current_physical_screen == self.capture_target_screen:
            # Current monitor - show warning background
            bg_color = QColor(30, 30, 30) if self.theme == "dark" else QColor(240, 240, 240)
            painter.fillRect(widget_rect, bg_color)
            painter.setPen(QColor("#FF6B6B"))
            font = QFont()
            font.setBold(True)
            font.setPointSize(50)
            painter.setFont(font)
            painter.drawText(widget_rect, Qt.AlignCenter, "CURRENT MONITOR")
        else:
            # Normal background for other monitors
            bg_color = QColor(30, 30, 30) if self.theme == "dark" else QColor(240, 240, 240)
            painter.fillRect(widget_rect, bg_color)
            
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
                
                # Use fully opaque border with theme-appropriate color
                border_color = QColor(255, 255, 255) if self.theme == "dark" else QColor(0, 0, 0)
                border_pen = QPen(border_color)
                border_pen.setWidth(1)
                painter.setPen(border_pen)
                painter.drawRect(target_rect.adjusted(0, 0, -1, -1))
            else:
                # Error state - use theme-appropriate colors without alpha
                error_bg = QColor(40, 40, 40) if self.theme == "dark" else QColor(220, 220, 220)
                error_text = QColor(255, 255, 255) if self.theme == "dark" else QColor(0, 0, 0)
                painter.fillRect(widget_rect, error_bg)
                painter.setPen(QPen(error_text, 1))
                painter.drawText(widget_rect, Qt.AlignCenter, "No Content / Error")
        
        if self.border_pen:
            painter.setPen(self.border_pen)
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        logger.debug(f"Mouse press at {event.position().toPoint()}, global: {event.globalPosition().toPoint()}")
        
        # Handle right-click to show context menu
        if event.button() == Qt.RightButton:
            self.show_context_menu(event.position().toPoint())
            event.accept()
            return
            
        # Use centralized mouse press handler for left button
        if event.button() == Qt.LeftButton:
            self._drag_state = snap_utils.handle_overlay_mouse_press(event, self)
            logger.debug(f"Drag state after press: {self._drag_state}")
            
            if not self._drag_state['is_resizing']:
                self._is_snapped = False  # Reset snap state on new drag
                logger.info(f"Starting drag with offset: {self._drag_state.get('drag_offset')}")
            event.accept()
        else:
            logger.debug("Non-left/right button press, passing to parent")
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Use centralized mouse release handler
            if hasattr(self, '_drag_state'):
                if snap_utils.handle_overlay_mouse_release(event, self, self._drag_state):
                    self.ensure_in_monitor_bounds()
                    event.accept()
                    return
            
            # Fallback to default behavior if not handled
            self._is_snapped = False  # Reset snap state
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

    # _get_resize_edge has been moved to snap_utils.get_resize_edge_for_pos
    # This method is kept for backward compatibility with any existing code
    def _get_resize_edge(self, pos):
        return snap_utils.get_resize_edge_for_pos(pos, self, self._edge_margin)

    def mouseMoveEvent(self, event: QMouseEvent):
        # Log mouse position for debugging
        pos = event.position().toPoint()
        global_pos = event.globalPosition().toPoint()
        
        # First, handle cursor changes based on resize edge
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
        
        # Use centralized mouse move handler
        if hasattr(self, '_drag_state'):
            logger.debug(f"Before handle_overlay_mouse_move - drag_state: {self._drag_state}")
            handled = snap_utils.handle_overlay_mouse_move(event, self, self._drag_state, self.snap_distance)
            logger.debug(f"After handle_overlay_mouse_move - handled: {handled}, drag_state: {self._drag_state}")
            
            if handled:
                # Update capture parameters if window was moved/resized
                if hasattr(self, '_capture_worker'):
                    self._update_capture_params()
                event.accept()
                return
        else:
            logger.debug("No _drag_state attribute found")
        
        super().mouseMoveEvent(event)

    def _fallback_ensure_in_monitor_bounds(self, screen):
        """Fallback method to ensure window is within screen bounds using Qt's logical coordinates."""
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
            
            # Adjust position to be within screen bounds
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
            
    def get_windows_monitor_resolution(self, screen):
        """Get the true Windows monitor resolution using snap_utils.
        
        This method is deprecated and will be removed in a future version.
        Use snap_utils.get_physical_monitor_info() instead.
        """
        try:
            from snap_utils import get_physical_monitor_for_screen
            monitor_info = get_physical_monitor_for_screen(screen)
            if monitor_info and 'physical_width' in monitor_info and 'physical_height' in monitor_info:
                return monitor_info['physical_width'], monitor_info['physical_height']
            
            # Fallback to Qt's device pixel ratio if we couldn't get the monitor info
            logger.warning("Could not get monitor info from snap_utils, falling back to Qt's DPI scaling")
            device_ratio = screen.devicePixelRatio()
            geo = screen.geometry()
            return int(geo.width() * device_ratio), int(geo.height() * device_ratio)
            
        except Exception as e:
            logger.error(f"Error getting monitor resolution: {e}")
            # Final fallback to Qt's calculation (should rarely be needed)
            device_ratio = screen.devicePixelRatio()
            geo = screen.geometry()
            return int(geo.width() * device_ratio), int(geo.height() * device_ratio)

    def ensure_in_monitor_bounds(self, target_screen=None):
        if self._is_resizing:
            return  # Skip bounds checking during resizing
            
        screen_to_use = target_screen or self.screen() or QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        if not screen_to_use:
            logger.warning("ensure_in_monitor_bounds: Could not determine screen.")
            return
        
        try:
            from snap_utils import get_physical_monitor_for_screen
            
            # Get physical monitor information
            monitor_info = get_physical_monitor_for_screen(screen_to_use)
            if not monitor_info or 'physical_width' not in monitor_info or 'physical_height' not in monitor_info:
                logger.error("Could not get physical monitor info, using fallback method")
                self._fallback_ensure_in_monitor_bounds(screen_to_use)
                return
                
            # Get physical monitor dimensions and position
            phys_width = monitor_info.get('physical_width', 0)
            phys_height = monitor_info.get('physical_height', 0)
            phys_x = monitor_info.get('physical_position', QPoint(0, 0)).x()
            phys_y = monitor_info.get('physical_position', QPoint(0, 0)).y()
            
            if phys_width <= 0 or phys_height <= 0:
                logger.error(f"Invalid physical dimensions: {phys_width}x{phys_height}, using fallback")
                self._fallback_ensure_in_monitor_bounds(screen_to_use)
                return
            
            # Get the screen's geometry in logical coordinates
            screen_geo = screen_to_use.availableGeometry()
            
            # Get the current window geometry in logical coordinates
            window_geo = self.geometry()
            x, y = window_geo.x(), window_geo.y()
            w, h = window_geo.width(), window_geo.height()
            
            # Calculate scale factors between logical and physical coordinates
            scale_x = phys_width / screen_geo.width() if screen_geo.width() > 0 else 1.0
            scale_y = phys_height / screen_geo.height() if screen_geo.height() > 0 else 1.0
            
            # Convert window position to physical coordinates relative to the screen
            x_phys = int((x - screen_geo.x()) * scale_x)
            y_phys = int((y - screen_geo.y()) * scale_y)
            w_phys = int(w * scale_x)
            h_phys = int(h * scale_y)
            
            logger.debug(f"Bounds check - Screen: {screen_to_use.name() if screen_to_use else 'Unknown'}, "
                       f"Physical: {phys_width}x{phys_height}, "
                       f"Current: {x_phys},{y_phys} {w_phys}x{h_phys}")
            
            # Check and adjust position in physical coordinates
            new_x_phys, new_y_phys = x_phys, y_phys
            if x_phys < 0:
                new_x_phys = 0
            if y_phys < 0:
                new_y_phys = 0
            if x_phys + w_phys > phys_width:
                new_x_phys = max(0, phys_width - w_phys)
            if y_phys + h_phys > phys_height:
                new_y_phys = max(0, phys_height - h_phys)
            
            # Only move if position changed
            if new_x_phys != x_phys or new_y_phys != y_phys:
                # Convert back to logical coordinates
                new_x = int(new_x_phys / scale_x) + screen_geo.x()
                new_y = int(new_y_phys / scale_y) + screen_geo.y()
                
                logger.debug(f"Adjusted position from ({x},{y}) to ({new_x},{new_y}) logical")
                self.move(new_x, new_y)
            
        except Exception as e:
            logger.error(f"Error in ensure_in_monitor_bounds: {e}", exc_info=True)
            self._fallback_ensure_in_monitor_bounds(screen_to_use)
    
    def _apply_context_menu_theme(self):
        """Apply the current theme to the context menu and its submenus."""
        if not hasattr(self, 'context_menu') or not self.context_menu:
            return
            
        # Get the current theme from settings or instance variable
        theme = getattr(self, 'theme', 'dark')
        
        # Apply the stylesheet from our centralized theme
        try:
            from constants import ThemeColors
            stylesheet = ThemeColors.get_theme_stylesheet(theme)
            self.context_menu.setStyleSheet(stylesheet)
            # Also apply to submenus if they exist
            submenus = [
                getattr(self, 'switch_to_window_menu', None),
                getattr(self, 'switch_to_monitor_menu', None)
            ]
            for submenu in submenus:
                if submenu:
                    submenu.setStyleSheet(stylesheet)
        except Exception as e:
            logger.error(f"Error applying theme to context menu: {e}", exc_info=True)

    def show_context_menu(self, position):
        try:
            if not hasattr(self, 'context_menu') or not self.context_menu:
                logger.warning("Context menu not initialized, initializing now")
                self._init_context_menu()
                
            if not hasattr(self, '_populate_switch_window_menu') or not hasattr(self, '_populate_switch_monitor_menu'):
                logger.error("Required menu population methods not found")
                return
                
            try:
                self._populate_switch_window_menu()
                self._populate_switch_monitor_menu()
                
                # Ensure the menu has actions before showing it
                if not self.context_menu.actions():
                    logger.warning("Context menu has no actions, adding default actions")
                    self.context_menu.addAction("Close", self.close)
                    
                # Show the context menu at the cursor position
                self.context_menu.exec(self.mapToGlobal(position))
                logger.debug("Context menu shown successfully")
                
            except Exception as e:
                logger.error(f"Error populating context menu: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Failed to show context menu: {e}", exc_info=True)
            # Try to show a basic context menu as fallback
            try:
                menu = QMenu()
                menu.addAction("Close", self.close)
                menu.exec(self.mapToGlobal(position))
            except Exception as fallback_error:
                logger.error(f"Fallback context menu also failed: {fallback_error}")

    def _handle_reset_position(self):
        """Reset the overlay's position and size based on the saved preset for the current monitor."""
        logger.debug("Resetting monitor overlay position and size")
        
        if not hasattr(self, 'app_instance') or not self.app_instance or not hasattr(self.app_instance, 'settings'):
            logger.warning("Cannot reset position: app_instance or settings not available")
            return
            
        # Get the current screen (display screen takes precedence over capture screen)
        current_screen = self._display_screen or self.capture_target_screen or QGuiApplication.primaryScreen()
        if not current_screen:
            logger.error("Could not determine current screen for reset")
            return
            
        try:
            # Get the monitor index
            screens = QGuiApplication.screens()
            monitor_idx = screens.index(current_screen) if current_screen in screens else 0
            
            # Load the saved preset for this monitor
            preset_key = f"MonitorPresets/Monitor_{monitor_idx}_Preset"
            position_preset = self.app_instance.settings.value(preset_key, "Center")  # Default to "Center" if not found
            
            # Calculate the new geometry based on the preset
            if hasattr(self.app_instance, 'calculate_position_geometry'):
                new_geometry = self.app_instance.calculate_position_geometry(
                    current_screen, position_preset, "monitor"
                )
                
                if new_geometry and new_geometry.isValid():
                    logger.info(f"Resetting monitor overlay to {position_preset} position: {new_geometry}")
                    self.setGeometry(new_geometry)
                    self._update_capture_params()
                    return
            
            # Fallback to default behavior if preset calculation fails
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
            # Fallback to simple reset if there's an error
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
        available_screens = QGuiApplication.screens()
        
        if not available_screens:
            no_screens_action = QAction("No screens available", self.switch_to_monitor_menu)
            no_screens_action.setEnabled(False)
            self.switch_to_monitor_menu.addAction(no_screens_action)
            return

        for idx, screen_obj in enumerate(available_screens):
            if screen_obj == self.capture_target_screen:
                continue
            
            display_name = self._get_display_name(screen_obj, idx)
            screen_action = QAction(display_name, self.switch_to_monitor_menu)
            screen_action.triggered.connect(lambda checked=False, s=screen_obj: self._handle_swap_screen(s))
            self.switch_to_monitor_menu.addAction(screen_action)

    def _populate_switch_window_menu(self):
        self.switch_to_window_menu.clear()
        try:
            # Use get_menu_ready_windows which includes icons and central filtering
            if self.app_instance and hasattr(self.app_instance, 'get_menu_ready_windows'):
                windows_data = self.app_instance.get_menu_ready_windows()
                logger.debug(f"Found {len(windows_data)} windows for context menu")
                
                # Apply sorting if needed
                if hasattr(self.app_instance, 'window_sort_order'):
                    if self.app_instance.window_sort_order == "Alphabetical":
                        windows_data.sort(key=lambda item: item[1].lower() if item[1] else "")
                
                # Add actions with icons
                for hwnd, title, icon in windows_data:
                    display_title = title.strip() or "[Untitled Window]"
                    action = QAction(display_title, self.switch_to_window_menu)
                    if icon and not icon.isNull():
                        action.setIcon(icon)
                    action.triggered.connect(lambda checked=False, h=hwnd: self._handle_initiate_window_swap(h))
                    self.switch_to_window_menu.addAction(action)
                return
                
            # Fall back to window_enumerator.get_capturable_windows_with_icons() if available
            if hasattr(self.app_instance, 'window_enumerator') and hasattr(self.app_instance.window_enumerator, 'get_capturable_windows_with_icons'):
                try:
                    # Get windows with filtering already applied in get_capturable_windows_with_icons()
                    windows_data = self.app_instance.window_enumerator.get_capturable_windows_with_icons()
                    logger.debug(f"Found {len(windows_data)} windows via window_enumerator")
                    
                    # Apply sorting if needed
                    if hasattr(self.app_instance, 'window_sort_order'):
                        if self.app_instance.window_sort_order == "Alphabetical":
                            windows_data.sort(key=lambda item: item[1].lower() if item[1] else "")
                    
                    # Add actions with icons
                    for hwnd, title, icon in windows_data:
                        display_title = title.strip() or "[Untitled Window]"
                        action = QAction(display_title, self.switch_to_window_menu)
                        if icon and not icon.isNull():
                            action.setIcon(icon)
                        action.triggered.connect(lambda checked=False, h=hwnd: self._handle_initiate_window_swap(h))
                        self.switch_to_window_menu.addAction(action)
                    return
                except Exception as e:
                    logger.warning(f"Error using window_enumerator.get_capturable_windows_with_icons(): {e}")
            
            # Final fallback to basic window enumeration (no icons)
            from main import WindowEnumerator
            windows_data = WindowEnumerator.enum_windows()
            logger.debug(f"Found {len(windows_data)} windows via basic enum_windows")
            
            # Apply sorting if needed
            if hasattr(self.app_instance, 'window_sort_order'):
                if self.app_instance.window_sort_order == "Alphabetical":
                    windows_data.sort(key=lambda item: item[1].lower() if item[1] else "")
            
            # Add actions without icons
            for hwnd, title in windows_data:
                display_title = title.strip() or "[Untitled Window]"
                action = QAction(display_title, self.switch_to_window_menu)
                action.triggered.connect(lambda checked=False, h=hwnd: self._handle_initiate_window_swap(h))
                self.switch_to_window_menu.addAction(action)
                
        except ImportError:
            error_action = QAction("Error: WindowEnumerator not found", self.switch_to_window_menu)
            error_action.setEnabled(False)
            self.switch_to_window_menu.addAction(error_action)
        except Exception as e:
            logger.error(f"Error populating window menu: {e}", exc_info=True)
            error_action = QAction("Error loading windows", self.switch_to_window_menu)
            error_action.setEnabled(False)
            self.switch_to_window_menu.addAction(error_action)

    def _handle_initiate_window_swap(self, hwnd):
        if self.app_instance and hasattr(self.app_instance, 'prepare_to_create_window_overlay'):
            self.app_instance.prepare_to_create_window_overlay(hwnd)
        else:
            logger.error("app_instance or prepare_to_create_window_overlay method not found.")

    def _handle_swap_screen(self, new_screen):
        if new_screen and new_screen != self.capture_target_screen:
            logger.info(f"Attempting to swap to screen '{new_screen.name()}'")
            self.capture_target_screen = new_screen
            self._pixmap = None
            if self._setup_mss_monitor_mapping():
                self._display_screen = self._select_display_screen()
                logger.debug(f"Updated display screen to: {self._display_screen.name()}")
                self._update_capture_params()
                self.ensure_in_monitor_bounds(self._display_screen)
                self.update()
                QTimer.singleShot(100, self._reapply_mouse_settings)
                logger.info(f"Successfully swapped to screen '{new_screen.name()}', displayed on '{self._display_screen.name() if self._display_screen else 'None'}'")
            else:
                logger.error(f"Failed to map screen '{new_screen.name()}'")
                self.capture_target_screen = QGuiApplication.primaryScreen()
                self._display_screen = self._select_display_screen()
                self._setup_mss_monitor_mapping()
                self._update_capture_params()
                self.ensure_in_monitor_bounds(self._display_screen)
                self.update()

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
        """Handle the show sub-settings action from the context menu."""
        logger.debug("Show sub-settings action triggered")
        
        # Try to use the application's _show_sub_settings method if available
        if self.app_instance and hasattr(self.app_instance, '_show_sub_settings'):
            try:
                logger.debug("Using application's _show_sub_settings method")
                self.app_instance._show_sub_settings()
                return
            except Exception as e:
                logger.error(f"Error in app._show_sub_settings(): {e}")
        
        # Fallback to direct creation if app method is not available
        try:
            from subsettings_dialog import SubSettingsDialog
            logger.debug("Creating SubSettingsDialog directly")
            
            # Create the dialog with proper parent and flags
            dialog = SubSettingsDialog(parent=self, app_instance=self.app_instance)
            
            # Set window flags to match application style
            dialog.setWindowFlags(
                Qt.Dialog | 
                Qt.WindowTitleHint | 
                Qt.WindowCloseButtonHint |
                Qt.WindowStaysOnTopHint |
                Qt.WindowSystemMenuHint |
                Qt.WindowMinMaxButtonsHint
            )
            
            # Set window attributes
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            dialog.setModal(False)
            
            # Get the screen where the mouse is currently located
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()
            
            # Set a reasonable default size
            dialog.resize(400, 300)
            
            # Center the dialog on the screen
            x = screen_geometry.x() + (screen_geometry.width() - dialog.width()) // 2
            y = screen_geometry.y() + (screen_geometry.height() - dialog.height()) // 2
            
            # Ensure the dialog stays within screen bounds
            x = max(screen_geometry.left(), min(x, screen_geometry.right() - dialog.width()))
            y = max(screen_geometry.top(), min(y, screen_geometry.bottom() - dialog.height()))
            
            dialog.move(x, y)
            
            # Show and activate the dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            
            # Ensure the window is not minimized and is active
            dialog.setWindowState(dialog.windowState() & ~Qt.WindowMinimized)
            dialog.raise_()
            dialog.activateWindow()
            
            logger.debug(f"SubSettingsDialog shown at position: {dialog.pos()}, size: {dialog.size()}")
            
        except ImportError as e:
            logger.error(f"Failed to import SubSettingsDialog: {e}")
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to load settings dialog: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error showing settings dialog: {e}")
            QMessageBox.critical(
                self, 
                "Error", 
                f"An unexpected error occurred: {e}")

def _handle_initiate_window_swap(self, hwnd):
    if self.app_instance and hasattr(self.app_instance, 'prepare_to_create_window_overlay'):
        self.app_instance.prepare_to_create_window_overlay(hwnd)
    else:
        logger.error("app_instance or prepare_to_create_window_overlay method not found.")

def _handle_swap_screen(self, new_screen):
    if new_screen and new_screen != self.capture_target_screen:
        logger.info(f"Attempting to swap to screen '{new_screen.name()}'")
        self.capture_target_screen = new_screen
        self._pixmap = None
        if self._setup_mss_monitor_mapping():
            self._display_screen = self._select_display_screen()
            logger.debug(f"Updated display screen to: {self._display_screen.name()}")
            self._update_capture_params()
            self.ensure_in_monitor_bounds(self._display_screen)
            self.update()
            QTimer.singleShot(100, self._reapply_mouse_settings)
            logger.info(f"Successfully swapped to screen '{new_screen.name()}', displayed on '{self._display_screen.name() if self._display_screen else 'None'}'")
        else:
            logger.error(f"Failed to map screen '{new_screen.name()}'")
            self.capture_target_screen = QGuiApplication.primaryScreen()
            self._display_screen = self._select_display_screen()
            self._setup_mss_monitor_mapping()
            self._update_capture_params()
            self.ensure_in_monitor_bounds(self._display_screen)
            self.update()

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
    """Handle the show sub-settings action from the context menu."""
    logger.debug("Show sub-settings action triggered")
    
    # Try to use the application's _show_sub_settings method if available
    if self.app_instance and hasattr(self.app_instance, '_show_sub_settings'):
        try:
            logger.debug("Using application's _show_sub_settings method")
            self.app_instance._show_sub_settings()
            return
        except Exception as e:
            logger.error(f"Error in app._show_sub_settings(): {e}")
    
    # Fallback to direct creation if app method is not available
    try:
        from subsettings_dialog import SubSettingsDialog
        logger.debug("Creating SubSettingsDialog directly")
        
        # Create the dialog with proper parent and flags
        dialog = SubSettingsDialog(parent=self, app_instance=self.app_instance)
        
        # Set window flags to match application style
        dialog.setWindowFlags(
            Qt.Dialog | 
            Qt.WindowTitleHint | 
            Qt.WindowCloseButtonHint |
            Qt.WindowStaysOnTopHint |
            Qt.WindowSystemMenuHint |
            Qt.WindowMinMaxButtonsHint
        )
        
        # Set window attributes
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setModal(False)
        
        # Get the screen where the mouse is currently located
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # Set a reasonable default size
        dialog.resize(400, 300)
        
        # Center the dialog on the screen
        x = screen_geometry.x() + (screen_geometry.width() - dialog.width()) // 2
        y = screen_geometry.y() + (screen_geometry.height() - dialog.height()) // 2
        
        # Ensure the dialog stays within screen bounds
        x = max(screen_geometry.left(), min(x, screen_geometry.right() - dialog.width()))
        y = max(screen_geometry.top(), min(y, screen_geometry.bottom() - dialog.height()))
        
        dialog.move(x, y)
        
        # Show and activate the dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        
        # Ensure the window is not minimized and is active
        dialog.setWindowState(dialog.windowState() & ~Qt.WindowMinimized)
        dialog.raise_()
        dialog.activateWindow()
        
        logger.debug(f"SubSettingsDialog shown at position: {dialog.pos()}, size: {dialog.size()}")
        
    except ImportError as e:
        logger.error(f"Failed to import SubSettingsDialog: {e}")
        QMessageBox.critical(
            self, 
            "Error", 
            f"Failed to load settings dialog: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error showing settings dialog: {e}")
        QMessageBox.critical(
            self, 
            "Error", 
            f"An unexpected error occurred: {e}"
        )

    @property
    def hwnd(self):
        """Get the native window handle for this overlay."""
        if not hasattr(self, '_hwnd') or not self._hwnd:
            self._hwnd = int(self.winId()) if hasattr(self, 'winId') else 0
        return self._hwnd

    def _handle_quit_application(self):
        """Handle application quit request with proper cleanup."""
        try:
            # Stop capture worker first
            self._stop_capture_worker()
            
            # Close the window
            self.close()
            
            # Emit the closed signal if it exists
            if hasattr(self, 'closed'):
                self.closed.emit()
                
            # Request application exit if we have an app instance
            if hasattr(self, 'app_instance') and self.app_instance:
                QTimer.singleShot(100, self.app_instance.quit)
                
        except Exception as e:
            logger.error(f"Error during application quit: {e}", exc_info=True)
        finally:
            # Always ensure resources are cleaned up
            if hasattr(self, '_mss_instance'):
                try:
                    self._mss_instance.close()
                except Exception as e:
                    logger.warning(f"Error closing MSS instance: {e}")
            
            # Ensure the window is properly destroyed
            self.deleteLater()

    def closeEvent(self, event):
        """Clean up resources when the window is closed."""
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
            # Ensure we call the parent's closeEvent
            super().closeEvent(event)

    def _stop_capture_worker(self):
        """Safely stop the capture worker and clean up resources."""
        try:
            if not self._capture_worker:
                return
                
            # Stop the capture worker
            self._capture_worker.stop()
            
            # Clean up the worker thread if it exists
            if self._worker_thread and self._worker_thread.isRunning():
                # Request the thread to quit
                self._worker_thread.quit()
                
                # Wait for the thread to finish (with timeout)
                if not self._worker_thread.wait(1000):  # 1 second timeout
                    logger.warning("Capture worker thread did not stop gracefully, terminating...")
                    self._worker_thread.terminate()
                    if not self._worker_thread.wait(1000):  # Additional wait after terminate
                        logger.error("Failed to terminate capture worker thread")
                
                # Clean up the thread object
                self._worker_thread.deleteLater()
                
        except Exception as e:
            logger.error(f"Error stopping capture worker: {e}", exc_info=True)
        finally:
            # Always clean up references
            self._capture_worker = None
            self._worker_thread = None
