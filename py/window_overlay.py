import ctypes
import ctypes.wintypes
import logging
import math
import time
import win32process
import win32con  # For virtual key codes
import win32api  # For key simulation
try:
    import sip
except ImportError:
    sip = None
from debug_utils import get_logger, log_perf, debug_enabled
import win32gui
from key_passthrough import KeyPassthrough

# Centralized logging and debugging
from debug_utils import DebugTimer, log_exception
from PySide6.QtCore import Qt, QTimer, QSize, QRect, Signal, QEvent
from PySide6.QtGui import QPainter, QColor, QGuiApplication, QPaintEvent, QPixmap, QPen, QKeyEvent
from PySide6.QtWidgets import QMainWindow, QMenu, QWidget, QApplication

# Import snap utilities
import snap_utils

# Import constants from constants module
from constants import (
    DEFAULT_WINDOW_OVERLAY_WIDTH, 
    DEFAULT_WINDOW_OVERLAY_HEIGHT
)


# Initialize logger - debug_utils handles all configuration
logger = get_logger(__name__)

user32 = ctypes.windll.user32
dwmapi = ctypes.windll.dwmapi

def _simulate_alt_press():
    VK_MENU = 0x12  # ALT key
    user32.keybd_event(VK_MENU, 0, 0, 0)
    user32.keybd_event(VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)

# Simple RECT structure for DWM functions
class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

# DWM thumbnail properties structure
class DWM_THUMBNAIL_PROPERTIES(ctypes.Structure):
    _fields_ = [
        ("dwFlags", ctypes.wintypes.DWORD),
        ("rcDestination", RECT),
        ("rcSource", RECT),
        ("opacity", ctypes.c_byte),
        ("fVisible", ctypes.wintypes.BOOL),
        ("fSourceClientAreaOnly", ctypes.wintypes.BOOL),
    ]

class FocusIndicatorWidget(QWidget):
    """A widget that shows a focus indicator or lock icon in the bottom-right corner."""
    def __init__(self, parent_ref=None):
        super().__init__()  # No parent to make it top-level
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint |
            Qt.Tool |
            Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # Changed to handle mouse events
        self.setFocusPolicy(Qt.NoFocus)
        
        # Store parent reference
        self._parent_ref = parent_ref
        
        # Indicator properties
        self._size = 10  # 40% smaller circle (was 16)
        self._margin = 6
        self._opacity = 0.85
        self._is_locked = False  # Track lock state
        self._passthrough_enabled = False  # Track passthrough state
        
        # Set initial and minimum size, enable mouse tracking
        self.setMinimumSize(7, 7)  # 40% smaller minimum size (was 12)
        self.resize(self._size, self._size)
        self.setMouseTracking(True)
        self.hide()
        
        # Listen for parent move/resize events for immediate update
        if self._parent_ref:
            self._parent_ref.installEventFilter(self)
        
        if debug_enabled():
            logger.debug(f"FocusIndicatorWidget initialized with size {self._size}")

        
    def set_parent_reference(self, parent_ref):
        """Set the parent reference for position calculations."""
        self._parent_ref = parent_ref
        
    def update_position(self, rect=None):
        """Update the position based on parent widget's global position and size.
        
        Args:
            rect: Optional QRect to use for positioning. If None, uses parent's frame geometry.
        """
        try:
            if not self._parent_ref:
                return
            # Use provided rect or get parent's frame geometry
            if rect is None:
                frame_rect = self._parent_ref.frameGeometry()
                if not hasattr(self._parent_ref, 'mapToGlobal'):
                    frame_rect = QRect(
                        self._parent_ref.x(),
                        self._parent_ref.y(),
                        self._parent_ref.width(),
                        self._parent_ref.height()
                    )
            else:
                frame_rect = QRect(
                    self._parent_ref.mapToGlobal(rect.topLeft()),
                    rect.size()
                )
            # Calculate position and set geometry in one operation
            self.setGeometry(
                frame_rect.right() - self._size - self._margin,
                frame_rect.bottom() - self._size - self._margin,
                self._size,
                self._size
            )
            if debug_enabled():
                logger.debug(f"FocusIndicatorWidget moved to {self.pos()} (frame_rect={frame_rect})")
        except Exception as e:
            logger.debug("Error updating focus indicator position: %s", str(e), exc_info=True)

    def eventFilter(self, obj, event):
        # Immediately update position on parent move/resize
        if obj is self._parent_ref and event.type() in (QEvent.Move, QEvent.Resize):
            self.update_position()
        return super().eventFilter(obj, event)

    
    def set_locked(self, locked):
        """Set the lock state of the indicator.
        
        Args:
            locked (bool): Whether the indicator should be in the locked state
        """
        if self._is_locked != locked:
            self._is_locked = locked
            self._update_tooltip()
            self.update()  # Trigger a repaint
    
    def is_locked(self):
        """Return whether the indicator is in the locked state."""
        return self._is_locked
        
    def set_passthrough_enabled(self, enabled):
        """Set the passthrough state of the indicator.
        
        Args:
            enabled (bool): Whether key passthrough is enabled
        """
        if self._passthrough_enabled != enabled:
            self._passthrough_enabled = enabled
            if debug_enabled():
                logger.debug(f"FocusIndicatorWidget passthrough set to {enabled}")
            self._update_tooltip()
            self.update()  # Trigger a repaint

            
    def is_passthrough_enabled(self):
        """Return whether key passthrough is enabled."""
        return self._passthrough_enabled
        
    def _update_tooltip(self):
        """Update the tooltip based on current states."""
        if self._is_locked and self._passthrough_enabled:
            self.setToolTip("Locked | Key Passthrough ON")
        elif self._is_locked:
            self.setToolTip("Locked")
        elif self._passthrough_enabled:
            self.setToolTip("Key Passthrough ON")
        else:
            self.setToolTip("Unlocked")
    
    def mousePressEvent(self, event):
        """Handle mouse press events.
        
        Toggles the lock state on left click and forwards the event to the parent.
        """
        if event.button() == Qt.LeftButton:
            # Toggle the parent's lock state if available
            if self._parent_ref and hasattr(self._parent_ref, 'toggle_window_lock'):
                self._parent_ref.toggle_window_lock()
    
    def enterEvent(self, event):
        """Show hand cursor on hover."""
        self.setCursor(Qt.PointingHandCursor)
    
    def leaveEvent(self, event):
        """Restore default cursor when leaving."""
        self.unsetCursor()
    
    def paintEvent(self, event: QPaintEvent):
        """Draw the focus indicator with lock state visualization."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            # Only show red when passthrough is enabled
            show_red = self._passthrough_enabled
            if self._is_locked:
                # Choose the appropriate lock icon based on passthrough state
                lock_icon = ":/Resources/lockred.png" if show_red else ":/Resources/lock.png"
                lock_pixmap = QPixmap(lock_icon)
                if not lock_pixmap.isNull():
                    dpr = self.devicePixelRatioF() if hasattr(self, 'devicePixelRatioF') else 1.0
                    
                    # Calculate maximum possible size that fits within the widget with some margin
                    max_size = min(self.width(), self.height()) * 1.52 * dpr  # Reduced from 1.6x to 1.52x (5% smaller)
                    
                    # Calculate target size while maintaining aspect ratio
                    icon_ratio = lock_pixmap.width() / lock_pixmap.height()
                    if lock_pixmap.width() > lock_pixmap.height():
                        target_width = int(max_size)
                        target_height = int(target_width / icon_ratio)
                    else:
                        target_height = int(max_size)
                        target_width = int(target_height * icon_ratio)
                    
                    # Scale the pixmap
                    scaled_pixmap = lock_pixmap.scaled(
                        target_width,
                        target_height,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    scaled_pixmap.setDevicePixelRatio(dpr)
                    
                    # Calculate position to center the icon
                    lock_x = int((self.width() - (scaled_pixmap.width() / dpr)) // 2)
                    lock_y = int((self.height() - (scaled_pixmap.height() / dpr)) // 2)
                    
                    # Save painter state, set opacity, draw, then restore
                    painter.save()
                    painter.setOpacity(0.9)  # 10% less opacity than before
                    painter.drawPixmap(lock_x, lock_y, scaled_pixmap)
                    painter.restore()
                
                # Only show red background if not showing red lock
                if show_red and not self._is_locked:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(255, 50, 50, 192))  # Red with 75% opacity
                    painter.drawEllipse(self.rect())
            elif show_red:
                # Show red indicator when passthrough is enabled (even if window isn't focused)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 50, 50, 192))  # Red with 75% opacity
                painter.drawEllipse(self.rect())
            else:
                # Default white indicator
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 255, 255, 153))  # 60% opacity white
                painter.drawEllipse(self.rect())
        except Exception as e:
            if debug_enabled():
                logger.error(f"Error in FocusIndicatorWidget paint: {e}", exc_info=True)
        finally:
            painter.end()



class BorderWidget(QMainWindow):
    # Signals for opacity changes
    overlay_opacity_changed = Signal(float)  # 0.0-1.0 range
    border_opacity_changed = Signal(float)   # 0.0-1.0 range
    
    def handle_swap_window(self, new_hwnd):
        """Handle swapping the current window in the overlay with a new window.
        
        This method is called by the WindowSwitcher.quick_switch_windows method
        to perform the actual window swap in the overlay.
        
        Args:
            new_hwnd (int): Window handle to show in the overlay
            
        Returns:
            bool: True if the swap was successful, False otherwise
        """
        logger.info(f"BorderWidget.handle_swap_window called with HWND: {new_hwnd}")
        logger.info(f"Swapping window in overlay from {self.hwnd} to {new_hwnd}")
        
        # Check if the overlay is locked
        if getattr(self, '_is_window_locked', False):
            logger.info("Window swap prevented: Overlay is locked")
            return False
            
        try:
            # Store the original hwnd for cleanup
            original_hwnd = self.hwnd
            
            # Clean up existing thumbnail if there is one
            if hasattr(self, 'thumbnail') and self.thumbnail:
                try:
                    dwmapi.DwmUnregisterThumbnail(self.thumbnail)
                    logger.info(f"Unregistered thumbnail for HWND {original_hwnd}")
                except Exception as e:
                    logger.error(f"Error unregistering thumbnail: {e}")
                self.thumbnail = None
            
            # Update the window handle
            self.hwnd = new_hwnd
            
            # Update the key_passthrough target window
            if hasattr(self, 'key_passthrough') and self.key_passthrough is not None:
                logger.debug(f"Updating key_passthrough target window to HWND {new_hwnd}")
                self.key_passthrough.set_target_window(new_hwnd)
            
            # Register and update the new thumbnail
            if not self._register_thumbnail():
                logger.error(f"Failed to register thumbnail for new window {new_hwnd}")
                return False
                
            # Update window title and other metadata
            if hasattr(self, '_update_window_title'):
                self._update_window_title()
                
            # Update the UI to reflect the new window
            self.update()
            
            logger.info(f"BorderWidget.handle_swap_window succeeded for HWND: {new_hwnd}")
            return True
        except Exception as e:
            logger.exception(f"Error in handle_swap_window: {e}")
            return False
            
    def _add_swap_methods(self):
        """Explicitly add swap methods to this instance to ensure they're available at runtime.
        This is a diagnostic/fix function to address method resolution issues.
        """
        # Direct instance method assignment
        logger.info(f"Adding swap methods to BorderWidget instance {id(self)}")
        
        # Define the swap window implementation directly here to avoid class method lookup issues
        def _direct_handle_swap_window(instance_self, new_hwnd):
            logger.info(f"Instance method handle_swap_window called for hwnd {new_hwnd}")
            logger.info(f"Swapping window in overlay from {instance_self.hwnd} to {new_hwnd}")
            try:
                # Store the original hwnd for cleanup
                original_hwnd = instance_self.hwnd
                
                # Clean up existing thumbnail if there is one
                if hasattr(instance_self, 'thumbnail') and instance_self.thumbnail:
                    try:
                        dwmapi.DwmUnregisterThumbnail(instance_self.thumbnail)
                        logger.info(f"Unregistered thumbnail for HWND {original_hwnd}")
                    except Exception as e:
                        logger.error(f"Error unregistering thumbnail: {e}")
                    instance_self.thumbnail = None
                
                # Update the window handle and create new thumbnail
                instance_self.hwnd = new_hwnd
                
                # Update the key passthrough target window
                if hasattr(instance_self, 'key_passthrough') and instance_self.key_passthrough:
                    logger.info(f"Updating key passthrough target window to HWND: {new_hwnd}")
                    instance_self.key_passthrough.set_target_window(new_hwnd)
                
                # Register and update the new thumbnail
                if not instance_self.register_thumbnail():
                    logger.error(f"Failed to register thumbnail for new window {new_hwnd}")
                    return False
                    
                # Update window title and other metadata
                if hasattr(instance_self, '_update_window_title'):
                    instance_self._update_window_title()
                    
                # Update the UI to reflect the new window
                instance_self.update()
                
                logger.info(f"Direct handle_swap_window succeeded for HWND: {new_hwnd}")
                return True
            except Exception as e:
                logger.exception(f"Error in handle_swap_window: {e}")
                return False
            
        # Bind methods to this instance
        setattr(self, 'handle_swap_window', _direct_handle_swap_window.__get__(self, BorderWidget))
        setattr(self, '_handle_swap_window', _direct_handle_swap_window.__get__(self, BorderWidget))
        
        # Verify methods are now available
        logger.info(f"Method verification - handle_swap_window: {hasattr(self, 'handle_swap_window')}")
        logger.info(f"Method verification - _handle_swap_window: {hasattr(self, '_handle_swap_window')}")
        logger.info(f"BorderWidget methods: {[m for m in dir(self) if not m.startswith('__') and ('swap' in m or 'handle' in m)]}")
        
        # Return self to allow method chaining
        return self

    @log_perf(level=logging.DEBUG, threshold_ms=10.0)
    def __init__(self, hwnd=None, monitor_index=0, theme="auto", opacity=100, app_instance=None, initial_geometry=None):
        super().__init__()
        if debug_enabled():
            logger.debug(f"Creating BorderWidget: hwnd={hwnd}, monitor_index={monitor_index}, theme={theme}, opacity={opacity}, initial_geometry={initial_geometry}")
        
        # Store initialization parameters
        self.hwnd = hwnd
        self.monitor_index = None if hwnd is not None else monitor_index
        self.app_instance = app_instance
        self.thumbnail = None
        
        # Initialize opacity from settings if available, otherwise use the provided default
        if app_instance and hasattr(app_instance, 'settings'):
            settings_opacity = app_instance.settings.value("overlay_opacity", opacity, type=int)
            self.opacity = float(settings_opacity) / 100.0  # Convert from 0-100 to 0.0-1.0
            if debug_enabled():
                logger.debug(f"[Opacity] Loaded from settings: {self.opacity:.3f}")
        else:
            # Fallback to provided/default values
            self.opacity = float(opacity) / 100.0  # Convert from 0-100 to 0.0-1.0
            if debug_enabled():
                logger.debug(f"[Opacity] Using default value: {self.opacity:.3f}")
        
        self.theme = theme  # Store the theme parameter
        
        # Initialize key passthrough with a callback for state changes
        self.key_passthrough = KeyPassthrough(state_change_callback=self._handle_key_passthrough_change)
        self.key_passthrough.set_target_window(self.hwnd) if self.hwnd else None
        
        # Track desktop overlay state
        self.is_desktop_overlay = False
        self.minimized_windows = []  # Store handles of minimized windows
        self.last_click_time = 0  # For double-click detection
        self.double_click_interval = 300  # ms between clicks for double-click
        self._passed_initial_geometry = initial_geometry
        
        # Initialize passthrough state variables
        self._initial_passthrough_state = False
        self._initial_aggressive_mode = False
        
        # Get initial passthrough state from settings
        if app_instance:
            self._initial_passthrough_state = app_instance.settings.value("key_passthrough_enabled", True, type=bool)
            self._initial_aggressive_mode = app_instance.settings.value("aggressive_key_passthrough", False, type=bool)
            
            # Ensure only one mode is active
            if self._initial_aggressive_mode:
                self._initial_passthrough_state = True
        
        # Create focus indicator with initial state
        self._focus_indicator = FocusIndicatorWidget(self)
        self._focus_indicator.set_passthrough_enabled(self._initial_passthrough_state or self._initial_aggressive_mode)
        self._focus_indicator.hide()
        
        # Set up key passthrough after focus indicator is created
        if app_instance:
            # Set initial state
            if self._initial_aggressive_mode:
                self.key_passthrough.set_enabled(True)
                self.key_passthrough.set_aggressive_mode(True)
            else:
                self.key_passthrough.set_enabled(self._initial_passthrough_state)
            
            # Connect to the key passthrough setting changed signal
            if hasattr(app_instance, 'key_passthrough_setting_changed'):
                app_instance.key_passthrough_setting_changed.connect(self._handle_key_passthrough_change)
            else:
                if debug_enabled():
                    logger.debug("key_passthrough_setting_changed signal not found in app_instance")
        
        # Single timer for position updates to avoid multiple rapid calls
        self._position_update_timer = QTimer()
        self._position_update_timer.setSingleShot(True)
        self._position_update_timer.timeout.connect(self._update_focus_indicator_position)
        
        # Update indicator position after window is shown
        QTimer.singleShot(100, self._update_focus_indicator_position)
        
        # Enable focus tracking
        self.setFocusPolicy(Qt.StrongFocus)

        self.window_sort_order = "Most Recently Active"
        if self.app_instance and hasattr(self.app_instance, 'window_sort_order'):
            self.window_sort_order = self.app_instance.window_sort_order
        self.thumbnail = None
        self.hwnd_self = None
        self.source_size = None
        self.thumbnail_rect = None

        self.alt_view_edit_menu = None
        self.thumbnail_settings_menu = None
        self.position_preset_menu = None
        
        # Set initial opacity before setting up the window
        if debug_enabled():
            logger.debug(f"[Opacity] Setting initial opacity: {self.opacity:.3f}")
        
        # Apply initial opacity before setting up the window
        self.set_overlay_opacity(self.opacity, emit_signal=False)
        
        # Call the setup methods
        # Initialize drag state
        self._drag_state = {
            'is_resizing': False,
            'resize_edge': None,
            'drag_start_global': None,
            'initial_geometry': None,
            'drag_offset': None
        }
        
        # Mouse handler for drag/resize/move logic
        
        
        
        # Initialize window lock state
        self._is_window_locked = False
        
        # Initialize border width and pen
        self._border_width = 2  # 2px border width for better visibility
        
        # Initialize border pen with default color and opacity
        self.border_pen = QPen()
        self.border_pen.setWidth(self._border_width)
        self.border_pen.setStyle(Qt.SolidLine)
        self.border_pen.setCapStyle(Qt.SquareCap)
        self.border_pen.setJoinStyle(Qt.MiterJoin)
        
        # Update border pen with current theme colors
        self._update_border_pen()
        
        self._setup_window()
        self._init_context_menu()
        
        # Apply theme safely with a default fallback
        if hasattr(self, 'theme'):
            self.apply_theme(self.theme)
        
        # Ensure opacities are still set after theme application
        if debug_enabled():
            logger.debug(f"[Opacity] Verifying opacities after theme - overlay: {getattr(self, 'opacity', 'N/A')}, border: {getattr(self, 'border_opacity', 'N/A')}")
        
        # Force update to ensure changes are visible
        self.update()

        # Ensure swap methods are always available on the instance
        self._add_swap_methods()
        
    def _handle_show_settings(self):
        """Handle showing the settings dialog from context menu."""
        try:
            app = QApplication.instance()
            if app:
                if hasattr(app, '_settings_panel') and app._settings_panel:
                    app._settings_panel.show()
                    app._settings_panel.activateWindow()
                    app._settings_panel.raise_()
                elif hasattr(app, '_show_settings'):
                    app._show_settings()
                elif self.app_instance and hasattr(self.app_instance, 'show_settings'):
                    self.app_instance.show_settings()
                    if hasattr(self.app_instance, 'activateWindow'):
                        self.app_instance.activateWindow()  # Bring settings window to front
        except Exception:
            logger.exception("Error in _handle_show_settings")

    def _handle_show_sub_settings(self):
        """Handle showing the sub-settings dialog from the context menu."""
        logger.debug("Show sub-settings action triggered (window overlay)")
        if self.app_instance and hasattr(self.app_instance, '_show_sub_settings'):
            try:
                logger.debug("Using application's _show_sub_settings method")
                self.app_instance._show_sub_settings()
                return
            except Exception as e:
                logger.error(f"Error in app._show_sub_settings(): {e}")
        try:
            from subsettings_dialog import SubSettingsDialog
            logger.debug("Creating SubSettingsDialog directly for window overlay")
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
            screen = QGuiApplication.screenAt(self.geometry().center()) or QGuiApplication.primaryScreen()
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
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to load settings dialog: {e}")
        except Exception as e:
            logger.error(f"Unexpected error showing settings dialog: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.RightButton:
                self.show_context_menu(event.pos())
                event.accept()
                return
            if event.button() == Qt.LeftButton:
                # Use snap_utils for unified drag/resize state
                self._drag_state = snap_utils.handle_overlay_mouse_press(event, self)
                if debug_enabled():
                    logger.debug(f"Drag state after press: {self._drag_state}")
                if not self._drag_state['is_resizing']:
                    if hasattr(self, '_is_snapped'):
                        self._is_snapped = False
                    logger.info(f"Starting drag with offset: {self._drag_state.get('drag_offset')}")
                self.setCursor(snap_utils.get_cursor_for_edge(self._drag_state['resize_edge']))
                event.accept()
            else:
                logger.debug("Non-left/right button press, passing to parent")
                super().mousePressEvent(event)
        except Exception as e:
            logger.exception("Error in mousePressEvent", exc_info=e)

    def mouseMoveEvent(self, event):
        try:
            pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
            event.globalPos()
            
            # Allow all mouse moves when window is locked - lock only prevents window switching
            if hasattr(self, '_is_window_locked') and self._is_window_locked:
                logger.debug("Window is locked - allowing move/resize but preventing window switching")
            
            # If we're not dragging or resizing, check for resize edges and update cursor
            if not hasattr(self, '_drag_state') or not any([
                self._drag_state.get('is_resizing'),
                self._drag_state.get('drag_offset') is not None
            ]):
                edge = snap_utils.get_resize_edge_for_pos(pos, self, self._border_width)
                
                # Set appropriate cursor based on edge (standardized with MonitorOverlay)
                if edge:
                    cursor = snap_utils.get_cursor_for_edge(edge)
                    if cursor:
                        self.setCursor(cursor)
                    else:
                        self.setCursor(Qt.ArrowCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
                
            # Handle active dragging or resizing
            if hasattr(self, '_drag_state'):
                logger.debug(f"Mouse move with drag_state: {self._drag_state}")
                
                # Handle mouse move with snap_utils
                handled = snap_utils.handle_overlay_mouse_move(event, self, self._drag_state)
                if debug_enabled():
                    logger.debug(f"After handle_overlay_mouse_move - handled: {handled}, drag_state: {self._drag_state}")
                if handled:
                    if hasattr(self, '_capture_worker'):
                        self._update_capture_params()
                    event.accept()
                    return
            else:
                logger.debug("No _drag_state attribute found")
            
            # Pass to parent for standard event handling
            super().mouseMoveEvent(event)
        except Exception as e:
            logger.exception("Error in mouseMoveEvent", exc_info=e)
            
    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                if hasattr(self, '_drag_state'):
                    logger.debug(f"Mouse release with drag_state: {self._drag_state}")
                    if snap_utils.handle_overlay_mouse_release(event, self, self._drag_state):
                        logger.debug("Mouse release handled by snap_utils")
                        if hasattr(self, 'ensure_in_monitor_bounds'):
                            self.ensure_in_monitor_bounds()
                        event.accept()
                        return
                
                # Reset snapped state if applicable
                if hasattr(self, '_is_snapped'):
                    self._is_snapped = False
                    logger.debug("Reset snapped state on mouse release")
                    
                # Always ensure window is within monitor bounds after mouse release
                if hasattr(self, 'ensure_in_monitor_bounds'):
                    self.ensure_in_monitor_bounds()
                    
            # Pass to parent for standard event handling
            super().mouseReleaseEvent(event)
            event.accept()
        except Exception as e:
            logger.exception("Error in mouseReleaseEvent", exc_info=e)
            
    def mouseDoubleClickEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                self.quick_swap_overlay()
            event.accept()
        except Exception as e:
            logger.exception("Error in mouseDoubleClickEvent", exc_info=e)
            
    @log_perf(level=logging.DEBUG, threshold_ms=10.0)
    def quick_swap_overlay(self):
        """Handle quick window switching when double-clicked.
        
        Delegates to WindowSwitcher to perform MRU-based quick switch
        between the current overlay window and the next most recently used window.
        
        Returns:
            bool: True if the quick swap was successful or skipped due to lock,
                  False if an error occurred.
        """
        logger.debug("Quick swap triggered by double-click")
        
        # Check if the overlay is locked
        if getattr(self, '_is_window_locked', False):
            logger.debug("Quick swap skipped: Overlay is locked")
            return True  # Return True to indicate this wasn't an error, just skipped
        
        try:
            # Validate we have the required components
            if not hasattr(self, 'app_instance') or not self.app_instance:
                logger.error("Cannot quick swap: No app instance available")
                return False
                
            if not hasattr(self.app_instance, 'window_switcher'):
                logger.error("Cannot quick swap: WindowSwitcher not available")
                return False
                
            # Get the current window handle
            current_hwnd = getattr(self, 'hwnd', None)
            if not current_hwnd or not win32gui.IsWindow(current_hwnd):
                logger.error(f"Cannot quick swap: Invalid current window handle: {current_hwnd}")
                return False
                
            # Get the currently focused window (if any)
            try:
                focused_hwnd = win32gui.GetForegroundWindow()
                if focused_hwnd and self.app_instance.window_switcher._is_our_window(focused_hwnd):
                    # Don't use our own windows as the focused window
                    focused_hwnd = 0
            except Exception as e:
                logger.warning(f"Error getting foreground window: {e}")
                focused_hwnd = 0
                
            # Use window_switcher to perform the quick switch
            logger.debug(f"Performing quick switch from window {current_hwnd}")
            success = self.app_instance.window_switcher.quick_switch_windows(
                current_hwnd,  # overlay_hwnd - the window currently in the overlay
                focused_hwnd,  # focused_hwnd - the window currently having focus
                self           # overlay_widget - the overlay widget instance (self)
            )
            
            if success:
                logger.info(f"Quick switch by double-click completed successfully from window {current_hwnd}")
                # Update timestamp to prevent immediate auto-switching
                if hasattr(self.app_instance, '_last_quick_switch_time'):
                    self.app_instance._last_quick_switch_time = time.time()
                    logger.debug("Set quick switch timestamp to prevent auto-switching")
            else:
                logger.warning("Quick switch by double-click failed")
                
            return success
        except Exception as e:
            logger.exception(f"Error in quick_swap_overlay: {e}")
            return False

    def wheelEvent(self, event):
        try:
            # Zoom/resize overlay with mouse wheel
            delta = event.angleDelta().y()
            if delta == 0:
                return
            geom = self.geometry()
            scale = 1.05 if delta > 0 else 0.95
            min_size = 60
            new_w = max(int(geom.width() * scale), min_size)
            new_h = max(int(geom.height() * scale), min_size)
            self.setGeometry(geom.x(), geom.y(), new_w, new_h)
            event.accept()
        except Exception as e:
            logger.exception("Error in wheelEvent", exc_info=e)
    
    # Signal-slot standardization - add consistent connection management
    def _connect_signals(self):
        """Connect all signals to their slots in a standardized way.
        This centralizes signal-slot connections for better maintainability.
        """
        try:
            # Context menu signals
            self.customContextMenuRequested.connect(self.show_context_menu)
            
            # Window state signals
            if hasattr(self, 'window_lock_toggled'):
                self.window_lock_toggled.connect(self._update_window_lock_state)
                
            # Any other signals that need consistent connection
            logger.debug("All signals connected successfully")
        except Exception as e:
            logger.error(f"Error connecting signals: {e}")
    
    # State tracking consolidation
    def _reset_state(self):
        """Reset all state tracking variables to their default values.
        Called during initialization and when state needs to be reset.
        """
        try:
            self._is_snapped = False
            self._is_resizing = False
            if hasattr(self, '_drag_state'):
                self._drag_state = {
                    'is_dragging': False,
                    'is_resizing': False,
                    'drag_offset': None,
                    'resize_edge': None
                }
            logger.debug("State variables reset to defaults")
        except Exception as e:
            logger.error(f"Error resetting state: {e}")


    @log_perf(level=logging.DEBUG, threshold_ms=10.0)
    def toggle_window_lock(self):
        """Toggle the window lock state.
        
        When locked, the overlay will stay attached to the current HWND
        but will still allow moving and resizing the overlay.
        """
        # Toggle the lock state using the app instance if available
        if self.app_instance and hasattr(self.app_instance, 'toggle_overlay_lock'):
            is_locked = self.app_instance.toggle_overlay_lock()
            self._is_window_locked = is_locked
        else:
            self._is_window_locked = not self._is_window_locked
            
        logger.debug(f"Window lock {'enabled' if self._is_window_locked else 'disabled'}")
        
        # Update the focus indicator to show lock state
        if hasattr(self, '_focus_indicator'):
            self._focus_indicator.set_locked(self._is_window_locked)
            
        # Update the lock action in the context menu if it exists
        if hasattr(self, 'lock_action'):
            self.lock_action.setChecked(self._is_window_locked)
            
    def update_lock_state(self, locked):
        """Update the lock state of this overlay.
        
        Args:
            locked (bool): Whether the overlay should be locked
        """
        if self._is_window_locked != locked:
            self._is_window_locked = locked
            logger.debug(f"Window lock state updated: {'locked' if locked else 'unlocked'}")
            
            # Update the focus indicator
            if hasattr(self, '_focus_indicator'):
                self._focus_indicator.set_locked(locked)
                
            # Update the lock action in the context menu if it exists
            if hasattr(self, 'lock_action'):
                self.lock_action.setChecked(locked)
        logger.info(f"Window lock {'enabled' if self._is_window_locked else 'disabled'}")
        
        # Update the tooltip to be more descriptive
        if hasattr(self, 'lock_action'):
            if self._is_window_locked:
                self.lock_action.setToolTip("Unlock to allow changing the target window")
            else:
                self.lock_action.setToolTip("Lock to current window")
    
    def set_auto_switch(self, enabled):
        """Enable or disable auto-switch functionality for this overlay.
        
        Args:
            enabled (bool): Whether to enable auto-switch
        """
        try:
            logger.debug(f"Setting auto_switch to {enabled} for window overlay")
            self.auto_switch_enabled = enabled
            
            # Update the menu checkbox if it exists
            if hasattr(self, 'context_menu') and self.context_menu:
                for action in self.context_menu.actions():
                    if hasattr(action, 'text') and action.text() == "Auto Switch on Focus Change":
                        action.setChecked(enabled)
                        break
                        
            if self.app_instance:
                self.app_instance.settings.setValue("auto_switch_enabled", enabled)
        except Exception as e:
            log_exception("Error in set_auto_switch", e)
            raise
    
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
            
        Returns:
            bool: True if opacity was changed, False otherwise
        """
        try:
            # Get current opacity, defaulting to 1.0 if not set
            old_opacity = getattr(self, 'opacity', 1.0)
            
            if debug_enabled():
                logger.debug(f"[Overlay] Setting opacity - requested: {opacity} (type: {type(opacity)}), "
                           f"current: {old_opacity:.3f}, emit_signal: {emit_signal}")
            
            # Normalize input to 0.0-1.0 range
            try:
                # Convert to float first
                opacity_float = float(opacity)
                
                # If value > 1.0, assume it's in 0-100 range and normalize
                if opacity_float > 1.0:
                    new_opacity = max(0.0, min(1.0, opacity_float / 100.0))
                    if debug_enabled():
                        logger.debug(f"[Overlay] Normalized opacity {opacity_float} to {new_opacity:.3f}")
                else:
                    new_opacity = max(0.0, min(1.0, opacity_float))
                    
            except (TypeError, ValueError):
                logger.error(f"[Overlay] Invalid opacity value: {opacity} (type: {type(opacity)}). Using previous value: {old_opacity:.3f}")
                return False
                
            # Only proceed if the opacity has meaningfully changed
            if math.isclose(new_opacity, old_opacity, abs_tol=0.001):
                if debug_enabled():
                    logger.debug(f"[Overlay] Opacity unchanged: {new_opacity:.3f}")
                return False
                
            if debug_enabled():
                logger.debug(f"[Overlay] Updating opacity from {old_opacity:.3f} to {new_opacity:.3f}")
            
            # Update the stored opacity
            self.opacity = new_opacity
            
            # Apply opacity immediately if window is visible, otherwise store as pending
            try:
                # Always store the target opacity
                self._pending_opacity = new_opacity
                
                # If window is visible, apply the opacity immediately
                if self.isVisible() and self.windowHandle() and self.windowHandle().isVisible():
                    self.setWindowOpacity(new_opacity)
                    if debug_enabled():
                        logger.debug(f"[Overlay] Set window opacity to {new_opacity:.3f}")
                    
                    # Force a repaint to ensure the change is visible
                    self.update()
                    
                    # Clear pending opacity since we've applied it
                    if hasattr(self, '_pending_opacity'):
                        delattr(self, '_pending_opacity')
                else:
                    if debug_enabled():
                        logger.debug(f"[Overlay] Window not ready, storing as pending: {new_opacity:.3f}")
                    
                    # If we're in the process of showing, schedule a check
                    QTimer.singleShot(100, self._apply_pending_opacity)
                    return False
                    
            except Exception as e:
                logger.error(f"[Overlay] Error setting window opacity: {e}")
                # Still try to apply as pending if there was an error
                self._pending_opacity = new_opacity
                QTimer.singleShot(100, self._apply_pending_opacity)
                return False
            
            # Clear any pending opacity since we've applied it
            if hasattr(self, '_pending_opacity'):
                delattr(self, '_pending_opacity')
            
            # Emit signal if requested
            if emit_signal:
                try:
                    if debug_enabled():
                        logger.debug("[Overlay] Emitting overlay_opacity_changed signal")
                    self.overlay_opacity_changed.emit(new_opacity)
                except Exception as e:
                    logger.error(f"[Overlay] Error emitting opacity_changed signal: {e}")
            
            if debug_enabled():
                logger.debug(f"[Overlay] Successfully updated opacity to {new_opacity:.3f}")
                
            return True
                
        except Exception as e:
            logger.error(f"[Overlay] Critical error in set_overlay_opacity: {e}", exc_info=debug_enabled())
            return False
                
    def _update_border_pen(self):
        """Update the border pen with current theme colors."""
        try:
            # Initialize border pen if it doesn't exist
            if not hasattr(self, 'border_pen'):
                self.border_pen = QPen()
                self.border_pen.setWidth(self._border_width)
                self.border_pen.setStyle(Qt.SolidLine)
                self.border_pen.setCapStyle(Qt.SquareCap)
                self.border_pen.setJoinStyle(Qt.MiterJoin)
            
            # Get border color based on theme
            is_dark_theme = False
            
            # Check theme in multiple possible ways
            if hasattr(self, 'theme'):
                if hasattr(self.theme, 'name'):
                    # If theme has a name attribute (like QPalette)
                    is_dark_theme = 'dark' in str(self.theme.name()).lower()
                else:
                    # If theme is just a string
                    is_dark_theme = 'dark' in str(self.theme).lower()
            
            # Also check the current palette as a fallback
            if not is_dark_theme:
                palette = self.palette()
                window_color = palette.window().color()
                # If the window background is dark, it's likely a dark theme
                is_dark_theme = window_color.lightness() < 128
            
            # Set border color based on theme
            if is_dark_theme:
                border_color = QColor(Qt.white)  # White border for dark theme
                if debug_enabled():
                    logger.debug("[Border] Setting white border for dark theme")
            else:
                border_color = QColor(Qt.black)  # Black border for light theme
                if debug_enabled():
                    logger.debug("[Border] Setting black border for light theme")
            
            # Set the border color (fully opaque)
            border_color.setAlpha(255)
            self.border_pen.setColor(border_color)
            
            # Force a repaint to apply the new border color
            self.update()
            
            if debug_enabled():
                logger.debug(f"[Border] Updated border pen - Color: {border_color.name()}")
                           
        except Exception as e:
            logger.error(f"[Border] Error updating border pen: {e}", exc_info=debug_enabled())
    
    def refresh_opacity_from_settings(self):
        """Refresh opacity settings from application settings.
        
        This method is called during initialization and when settings change.
        It updates the overlay opacity from settings.
        """
        try:
            if not self.app_instance or not hasattr(self.app_instance, 'settings'):
                if debug_enabled():
                    logger.debug("[Opacity] No app instance or settings available, using current values")
                return
                
            # Get the current value before updating
            old_opacity = getattr(self, 'opacity', 1.0)
            
            # Load new value from settings
            settings_opacity = self.app_instance.settings.value("overlay_opacity", 100, type=int)
            
            # Convert to float 0.0-1.0 range
            new_opacity = float(settings_opacity) / 100.0
            
            # Update instance variable
            self.opacity = new_opacity
            
            if debug_enabled():
                logger.debug(f"[Opacity] Refreshed from settings - "
                           f"opacity: {old_opacity:.3f} -> {new_opacity:.3f}")
            
            # Apply the new opacity if it's changed
            if not math.isclose(old_opacity, new_opacity, rel_tol=1e-5):
                self.set_overlay_opacity(new_opacity, emit_signal=False)
                
        except Exception as e:
            logger.error(f"[Opacity] Error refreshing opacity from settings: {e}", exc_info=debug_enabled())
            # Try to recover by setting default value
            try:
                self.opacity = 1.0
                self.set_overlay_opacity(1.0, emit_signal=False)
            except Exception as recovery_error:
                logger.error(f"[Opacity] Recovery failed: {recovery_error}")
    
    def _handle_key_passthrough_change(self, enabled: bool, aggressive: bool):
        """Handle changes to key passthrough settings.
        
        Args:
            enabled: Whether key passthrough is enabled
            aggressive: Whether aggressive mode is enabled
        """
        if debug_enabled():
            logger.debug(f"Key passthrough state changed - enabled: {enabled}, aggressive: {aggressive}")
        
        # Determine the effective passthrough state
        passthrough_enabled = enabled or aggressive
        
        # Update the focus indicator to reflect the new state
        if hasattr(self, '_focus_indicator'):
            # Update the passthrough state
            self._focus_indicator.set_passthrough_enabled(passthrough_enabled)
            
            # Force immediate update of the indicator
            self._focus_indicator.show()
            self._focus_indicator.raise_()
            
            # Ensure the indicator is visible and on top
            QApplication.processEvents()
            
            if debug_enabled():
                logger.debug(f"Focus indicator updated - passthrough: {passthrough_enabled}")
        else:
            logger.warning("Focus indicator not available for update")
    
    def set_border_opacity(self, opacity: float, emit_signal: bool = True):
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
            
            if debug_enabled():
                logger.debug(f"[Border] Setting border opacity - requested: {opacity} (type: {type(opacity)}), "
                           f"current: {old_border_opacity:.3f}, emit_signal: {emit_signal}")
            
            # Normalize input to 0.0-1.0 range
            try:
                # Convert to float first
                opacity_float = float(opacity)
                
                # If value > 1.0, assume it's in 0-100 range and normalize
                if opacity_float > 1.0:
                    new_opacity = max(0.0, min(1.0, opacity_float / 100.0))
                    if debug_enabled():
                        logger.debug(f"[Border] Normalized opacity {opacity_float} to {new_opacity:.3f}")
                else:
                    new_opacity = max(0.0, min(1.0, opacity_float))
                    
            except (TypeError, ValueError):
                logger.error(f"[Border] Invalid opacity value: {opacity} (type: {type(opacity)}). Using previous value: {old_border_opacity:.3f}")
                return
                
            # Only proceed if the opacity has meaningfully changed
            if math.isclose(new_opacity, old_border_opacity, abs_tol=0.001):
                if debug_enabled():
                    logger.debug(f"[Border] Border opacity unchanged: {new_opacity:.3f}")
                return
                
            if debug_enabled():
                logger.debug(f"[Border] Updating border opacity from {old_border_opacity:.3f} to {new_opacity:.3f}")
            
            # Update the stored border opacity
            self.border_opacity = new_opacity
            
            # Apply the new opacity to the window
            try:
                self.setWindowOpacity(new_opacity)
                if debug_enabled():
                    logger.debug(f"[Border] Set window opacity to {new_opacity:.3f}")
            except Exception as e:
                logger.error(f"[Border] Failed to set window opacity: {e}")
                return
            
            # Force a repaint to ensure the change is visible
            self.update()
            
            # Emit signal if requested
            if emit_signal:
                try:
                    if debug_enabled():
                        logger.debug("[Border] Emitting border_opacity_changed signal")
                    self.border_opacity_changed.emit(new_opacity)
                except Exception as e:
                    logger.error(f"[Border] Error emitting border_opacity_changed signal: {e}")
            
            if debug_enabled():
                logger.debug(f"[Border] Successfully updated border opacity to {new_opacity:.3f}")
                
        except Exception as e:
            logger.error(f"[Border] Critical error in set_border_opacity: {e}", exc_info=debug_enabled())

            
    def _handle_key_passthrough_change(self, enabled: bool, aggressive: bool):
        """Handle changes to key passthrough settings."""
        logger.debug(f"Key passthrough settings changed - enabled: {enabled}, aggressive: {aggressive}")
        
        # Update the key passthrough state
        if aggressive:
            self.key_passthrough.set_enabled(True)
            self.key_passthrough.set_aggressive_mode(True)
        else:
            self.key_passthrough.set_aggressive_mode(False)
            self.key_passthrough.set_enabled(enabled)
        
        # Update the focus indicator's passthrough state
        self._safe_focus_indicator('set_passthrough_enabled', enabled or aggressive)
        
        # Force update the focus indicator to ensure it's visible if needed
        if self.isActiveWindow():
            self._safe_focus_indicator('update')
        
        # Save settings to persistent storage
        if self.app_instance and hasattr(self.app_instance, 'settings'):
            settings = self.app_instance.settings
            
            # Block signals to prevent recursive updates
            settings.blockSignals(True)
            
            try:
                # Save the new settings
                settings.setValue("key_passthrough_enabled", enabled)
                settings.setValue("aggressive_key_passthrough", aggressive)
                
                # Force immediate write to disk
                settings.sync()
                logger.debug(f"Settings saved - key_passthrough_enabled: {enabled}, aggressive_key_passthrough: {aggressive}")
                
            except Exception as e:
                log_exception("Error saving key passthrough settings", e)
            finally:
                # Always unblock signals in case of error
                settings.blockSignals(False)



    def _setup_window(self):
        try:
            # Store current state before making any changes
            current_geom = self.geometry() if self.isVisible() else None
            was_visible = self.isVisible()
            
            # Set up window flags
            flags = (
                Qt.Window |  # Makes it a proper window
                Qt.FramelessWindowHint |
                Qt.WindowStaysOnTopHint |
                Qt.Tool
            )
            
            # Check if click-through is enabled in settings
            click_through_enabled = False
            if self.app_instance and hasattr(self.app_instance, 'settings'):
                click_through_enabled = self.app_instance.settings.value("click_through_enabled", False, type=bool)
            
            # Add transparent for input if click-through is enabled
            if click_through_enabled:
                flags |= Qt.WindowTransparentForInput
            
            # Only set window flags if they've actually changed
            if self.windowFlags() != flags:
                if debug_enabled():
                    logger.debug("[Setup] Updating window flags")
                self.setWindowFlags(flags)
            
            # Set window attributes
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setMouseTracking(True)
            
            # Set minimum size first
            self.setMinimumSize(100, 75)
            
            # Apply initial geometry if provided, otherwise use default size and position
            if self._passed_initial_geometry:
                if debug_enabled():
                    logger.debug(f"[Setup] Applying initial geometry: {self._passed_initial_geometry}")
                self.setGeometry(self._passed_initial_geometry)
            elif not current_geom or current_geom.isNull() or not current_geom.isValid():
                # Default size from constants only if we don't have a valid current geometry
                default_size = QSize(DEFAULT_WINDOW_OVERLAY_WIDTH, DEFAULT_WINDOW_OVERLAY_HEIGHT)
                if debug_enabled():
                    logger.debug(f"[Setup] Setting default window size: {default_size.width()}x{default_size.height()}")
                self.resize(default_size)
                
                # Position at top-left of the screen
                screen = QGuiApplication.primaryScreen()
                screen_geo = screen.availableGeometry()
                self.move(screen_geo.topLeft())
            
            # Show the window first before applying opacity to avoid visual glitches
            if was_visible or not self.isVisible():
                if debug_enabled():
                    logger.debug("[Setup] Showing window before applying opacity")
                self.show()
            
            # Apply window opacity after showing to ensure it takes effect
            # Use a single-shot timer to ensure all window initialization is complete
            QTimer.singleShot(0, self._apply_final_opacity)
            
            # Log the final state for debugging
            if debug_enabled():
                logger.debug(f"[Setup] Window setup complete with:\n"
                           f"  - Click-through: {click_through_enabled}"
                           f"\n  - Geometry: {self.geometry()}"
                           f"\n  - Target opacity: {getattr(self, 'opacity', 'N/A')}"
                           f"\n  - Theme: {getattr(self, 'theme', 'default')}")
                    
        except Exception as e:
            logger.error(f"[Setup] Error in _setup_window: {e}", exc_info=debug_enabled())
            # Attempt to recover by forcing a full recreation
            try:
                self.show()
                QTimer.singleShot(0, self._apply_final_opacity)
            except Exception as recovery_error:
                logger.error(f"[Setup] Recovery failed: {recovery_error}")
                # If we can't recover, log the error and continue
                
    def _apply_pending_opacity(self):
        """Apply any pending opacity settings.
        
        This is called after the window is shown to ensure opacity is applied correctly.
        """
        if not hasattr(self, '_pending_opacity'):
            return
            
        try:
            pending_opacity = self._pending_opacity
            if debug_enabled():
                logger.debug(f"[Opacity] Applying pending opacity: {pending_opacity:.3f}")
            
            # Always store the current window opacity for comparison
            current_opacity = self.windowOpacity()
            
            # Only update if the opacity has actually changed
            if not math.isclose(current_opacity, pending_opacity, abs_tol=0.001):
                self.setWindowOpacity(pending_opacity)
                self.update()
                
                if debug_enabled():
                    logger.debug(f"[Opacity] Applied pending opacity: {pending_opacity:.3f}")
            
            # Clear the pending opacity after applying
            delattr(self, '_pending_opacity')
            
            # Force a repaint to ensure the change is visible
            self.repaint()
            
        except Exception as e:
            logger.error(f"[Opacity] Error applying pending opacity: {e}", exc_info=debug_enabled())
            # Schedule another attempt if we still have a pending opacity
            if hasattr(self, '_pending_opacity'):
                QTimer.singleShot(100, self._apply_pending_opacity)
    
    def _apply_final_opacity(self):
        """Apply the final opacity settings after window is fully initialized.
        
        This method is called after the window is shown to avoid visual glitches.
        """
        try:
            # Get current opacity value or use default
            current_opacity = getattr(self, 'opacity', 1.0)
            
            if debug_enabled():
                logger.debug(f"[Opacity] Applying final opacity: {current_opacity:.3f}")
            
            # Apply the opacity
            self.setWindowOpacity(current_opacity)
            
            # Force a repaint to ensure the change is visible
            self.update()
            
            # Log the final state
            if debug_enabled():
                logger.debug(f"[Opacity] Final opacity applied: {self.windowOpacity():.3f}")
                
        except Exception as e:
            logger.error(f"[Opacity] Error applying final opacity: {e}", exc_info=debug_enabled())
            # Store as pending and try again later
            self._pending_opacity = getattr(self, 'opacity', 1.0)
            QTimer.singleShot(100, self._apply_pending_opacity)
    
    def _init_context_menu(self):
        # Use OverlayContextMenu for unified context menu logic
        from overlay_context_menu import OverlayContextMenu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self._context_menu_builder = OverlayContextMenu(self, overlay_type='window')
        self.customContextMenuRequested.connect(self.show_context_menu)

    def get_menu_ready_windows(self):
        """Get a list of windows suitable for the Switch To Window menu.
        
        This method delegates to the app_instance's get_menu_ready_windows method.
        If app_instance is not available or doesn't have the method, returns an empty list.
        
        Returns:
            List of tuples: (hwnd, title, icon) for each available window
        """
        if self.app_instance and hasattr(self.app_instance, 'get_menu_ready_windows'):
            return self.app_instance.get_menu_ready_windows()
        return []

    # _apply_context_menu_theme is now handled by OverlayContextMenu; remove local implementation.
            
    def _handle_reset_position(self):
        """Reset the window to the saved preset position and size for the current monitor."""
        try:
            if not hasattr(self, 'app_instance') or not hasattr(self.app_instance, 'get_preset_geometry'):
                logger.warning("Cannot reset position: app_instance or get_preset_geometry not available")
                return
                
            # Get the current screen
            screen = QGuiApplication.screenAt(self.pos())
            if not screen:
                logger.warning("Cannot reset position: could not determine current screen")
                return
                
            # Get the preset geometry for this overlay type and screen
            overlay_type = "window"  # This is a window overlay
            geometry = self.app_instance.get_preset_geometry(overlay_type, screen.name())
            
            if geometry:
                logger.info(f"Resetting window overlay to preset geometry: {geometry}")
                self.setGeometry(geometry)
                self.update()
            else:
                logger.warning("No preset geometry found for current screen and overlay type")
                
        except Exception as e:
            log_exception("Error resetting window position", e)

    def update_sort_order_and_refresh_menu(self, sort_order):
        self.window_sort_order = sort_order
        logger.debug(f"WindowOverlayWidget sort order updated to: {self.window_sort_order}")

    # _populate_switch_window_menu is now handled by OverlayContextMenu; remove local implementation.

    def _get_display_name(self, screen, idx):
        """Generate a display name for the screen with additional information.
        
        Args:
            screen: The QScreen object
            idx: The index of the screen
            
        Returns:
            str: A formatted display name for the screen
        """
        name = screen.name() or f"Screen {idx + 1}"
        manufacturer = screen.manufacturer()
        model = screen.model()
        
        if manufacturer or model:
            name = f"{name} ({manufacturer} {model})".strip()
            
        # Get the screen resolution
        size = screen.size()
        name = f"{name} - {size.width()}x{size.height()}"
        
        return name
        
    # _populate_switch_monitor_menu is now handled by OverlayContextMenu; remove local implementation.
            
    def _handle_swap_screen(self, new_screen):
        if new_screen and hasattr(self, 'capture_target_screen') and new_screen != self.capture_target_screen:
            logger.info(f"Attempting to swap to screen '{new_screen.name()}'")
            self.capture_target_screen = new_screen
            if hasattr(self, '_pixmap'):
                self._pixmap = None
            
            if hasattr(self, '_setup_mss_monitor_mapping') and callable(self._setup_mss_monitor_mapping):
                if self._setup_mss_monitor_mapping():
                    if hasattr(self, '_select_display_screen') and callable(self._select_display_screen):
                        self._display_screen = self._select_display_screen()
                        logger.debug(f"Updated display screen to: {self._display_screen.name() if hasattr(self._display_screen, 'name') else 'unknown'}")
                    
                    if hasattr(self, '_update_capture_params') and callable(self._update_capture_params):
                        self._update_capture_params()
                    
                    self.ensure_in_monitor_bounds(getattr(self, '_display_screen', new_screen))
                    self.update()
                    
                    if hasattr(self, '_reapply_mouse_settings') and callable(self._reapply_mouse_settings):
                        QTimer.singleShot(100, self._reapply_mouse_settings)
                    
                    logger.info(f"Successfully swapped to screen '{new_screen.name()}', displayed on '{self._display_screen.name() if hasattr(self._display_screen, 'name') else 'None'}'")
                else:
                    logger.error(f"Failed to map screen '{new_screen.name()}'")
                    self.capture_target_screen = QGuiApplication.primaryScreen()
                    if hasattr(self, '_select_display_screen') and callable(self._select_display_screen):
                        self._display_screen = self._select_display_screen()
                    
                    if hasattr(self, '_setup_mss_monitor_mapping') and callable(self._setup_mss_monitor_mapping):
                        self._setup_mss_monitor_mapping()
                    
                    if hasattr(self, '_update_capture_params') and callable(self._update_capture_params):
                        self._update_capture_params()
                    
                    self.ensure_in_monitor_bounds(getattr(self, '_display_screen', self.capture_target_screen))
                    self.update()
            else:
                # Fallback for simpler behavior if required methods don't exist
                screen_geometry = new_screen.availableGeometry()
                self.move(screen_geometry.topLeft())
                self.resize(screen_geometry.size())
                self.ensure_in_monitor_bounds(new_screen)
                self.update()
                logger.info(f"Swapped to screen '{new_screen.name()}' using fallback method")
                
    def _handle_switch_to_monitor_overlay(self):
        """Switch from window overlay to monitor overlay for the current display"""
        try:
            logger.info("Switching from window overlay to monitor overlay")
            
            # Get current screen for this overlay
            current_screen = QGuiApplication.screenAt(self.geometry().center())
            if not current_screen:
                current_screen = QGuiApplication.primaryScreen()
                if not current_screen:
                    logger.error("No valid screen found for monitor overlay")
                    return False
                
            logger.debug(f"Current screen: {current_screen.name() if hasattr(current_screen, 'name') else 'unnamed'}, geometry: {current_screen.geometry()}")
            
            # Get current geometry to preserve size and position
            current_geometry = self.geometry()
            logger.debug(f"Current window geometry: {current_geometry}")
            
            # Close this window overlay
            self.close()
            
            # Create a new monitor overlay for the current screen if app_instance is available
            if not self.app_instance:
                logger.error("Cannot switch to monitor overlay: app_instance not available")
                return False
                
            # Try both methods for backward compatibility
            if hasattr(self.app_instance, 'prepare_to_create_monitor_overlay'):
                # Newer method with QScreen object
                logger.debug("Using prepare_to_create_monitor_overlay method")
                try:
                    self.app_instance.prepare_to_create_monitor_overlay(current_screen)
                    logger.info(f"Successfully initiated monitor overlay creation for screen {current_screen.name() if hasattr(current_screen, 'name') else 'unnamed'}")
                    return True
                except Exception as e:
                    logger.error(f"Error in prepare_to_create_monitor_overlay: {e}")
                    # Fall through to try the older method
            
            # Fall back to older method if available
            if hasattr(self.app_instance, 'create_monitor_overlay'):
                logger.debug("Falling back to create_monitor_overlay method")
                try:
                    screens = QGuiApplication.screens()
                    screen_idx = screens.index(current_screen) if current_screen in screens else 0
                    logger.debug(f"Creating monitor overlay for screen index {screen_idx}")
                    self.app_instance.create_monitor_overlay(screen_idx, initial_geometry=current_geometry)
                    logger.info(f"Successfully created monitor overlay for screen index {screen_idx}")
                    return True
                except Exception as e:
                    logger.error(f"Error in create_monitor_overlay: {e}")
            
            logger.error("No valid method found to create monitor overlay")
            return False
                
        except Exception as e:
            logger.exception(f"Error switching to monitor overlay: {e}")
            return False
                
    def show_context_menu(self, pos):
        """Show the unified context menu at the given position (delegates to OverlayContextMenu)."""
        if hasattr(self, '_context_menu_builder'):
            self._context_menu_builder.show_menu(pos)
        else:
            # Fallback: show a minimal menu
            menu = QMenu()
            menu.addAction("Close", self.close)
            menu.exec(self.mapToGlobal(pos))
            
    def _handle_quit_application(self):
        """Handle the Quit Application action from the context menu.
        
        This method attempts to clean up resources and quit the application
        by delegating to the app_instance's cleanup_and_quit or quit methods.
        Falls back to QApplication.quit() if app_instance is not available.
        """
        logger.debug("Quitting application via window overlay context menu")
        try:
            if self.app_instance and hasattr(self.app_instance, 'cleanup_and_quit'):
                self.app_instance.cleanup_and_quit()
            elif self.app_instance and hasattr(self.app_instance, 'quit'):
                self.app_instance.quit()
            else:
                logger.warning("cleanup_and_quit method not found on app_instance. Falling back to QApplication.quit()")
                from PySide6.QtWidgets import QApplication
                QApplication.quit()
        except Exception as e:
            logger.error(f"Exception in _handle_quit_application: {e}")
            # Final fallback: try to quit directly
            try:
                from PySide6.QtWidgets import QApplication
                QApplication.quit()
            except Exception as final_error:
                logger.error(f"Failed to quit application: {final_error}")

    
    def register_thumbnail(self):
        """Register a DWM thumbnail for the window."""
        logger.debug(f"[register_thumbnail] Called for hwnd={self.hwnd}")
        if not self.hwnd:
            logger.warning("No target hwnd for DWM thumbnail registration.")
            return False
        self._cleanup_thumbnail()
        try:
            self.hwnd_self = int(self.winId())
            logger.debug(f"[register_thumbnail] Overlay window HWND (hwnd_self): {self.hwnd_self}")
            logger.debug(f"[register_thumbnail] Target capture HWND: {self.hwnd}")
            self.thumbnail = ctypes.wintypes.HANDLE()
            result = dwmapi.DwmRegisterThumbnail(self.hwnd_self, self.hwnd, ctypes.byref(self.thumbnail))
            logger.debug(f"[register_thumbnail] DwmRegisterThumbnail result: {result}, thumbnail handle: {getattr(self.thumbnail, 'value', None)}")
            if result != 0:
                logger.error(f"DwmRegisterThumbnail failed with error: {result} (hwnd_self={self.hwnd_self}, hwnd={self.hwnd})")
                self.thumbnail = None
                return False
            logger.info(f"DWM thumbnail registered (Handle: {self.thumbnail.value}) for HWND {self.hwnd} on self HWND {self.hwnd_self}")
            self.source_size = QSize(self._query_thumbnail_source_size().cx, self._query_thumbnail_source_size().cy)
            self.update_thumbnail()
            return True
        except Exception as e:
            log_exception("Exception during thumbnail registration", e)
            self._cleanup_thumbnail()
            return False


    def _query_thumbnail_source_size(self):
        size = ctypes.wintypes.SIZE()
        if self.thumbnail and self.thumbnail.value:
            hr = dwmapi.DwmQueryThumbnailSourceSize(self.thumbnail, ctypes.byref(size))
            if hr != 0:
                logger.error(f"DwmQueryThumbnailSourceSize failed: HRESULT 0x{hr:08X}")
        return size

    def _cleanup_thumbnail(self):
        if hasattr(self, "thumbnail") and self.thumbnail and self.thumbnail.value:
            logger.debug(f"Unregistering DWM thumbnail (Handle: {self.thumbnail.value})")
            try:
                result = dwmapi.DwmUnregisterThumbnail(self.thumbnail)
                if result != 0:
                    logger.warning(f"Failed to unregister DWM thumbnail: HRESULT 0x{result:08X}")
            except Exception as e:
                log_exception("Exception during DWM thumbnail cleanup", e)
            finally:
                self.thumbnail = None
                self.source_size = None
                self.thumbnail_rect = None

    def update_thumbnail(self):
        if not self.thumbnail or not self.thumbnail.value:
            logger.debug(f"[update_thumbnail] No valid DWM thumbnail. thumbnail={self.thumbnail}, value={getattr(self.thumbnail, 'value', None)}")
            return False
        current_source_dims_physical = self._query_thumbnail_source_size()
        logger.debug(f"[update_thumbnail] Source dims: cx={current_source_dims_physical.cx}, cy={current_source_dims_physical.cy}")
        if current_source_dims_physical.cx <= 0 or current_source_dims_physical.cy <= 0:
            logger.warning(f"update_thumbnail: Invalid physical source {current_source_dims_physical.cx}x{current_source_dims_physical.cy}. Not updating.")
            return False
        self.source_size = QSize(current_source_dims_physical.cx, current_source_dims_physical.cy)
        logical_overlay_rect = self.rect()
        dpr = self.devicePixelRatioF()
        bw_logical = self._border_width
        # Make content area 1 pixel smaller on each side to prevent overlapping the border
        content_area_logical = logical_overlay_rect.adjusted(bw_logical + 1, bw_logical + 1, -bw_logical - 1, -bw_logical - 1)
        logical_source_width = self.source_size.width() / dpr if dpr > 0 else self.source_size.width()
        logical_source_height = self.source_size.height() / dpr if dpr > 0 else self.source_size.height()
        logical_source_qsize = QSize(int(round(logical_source_width)), int(round(logical_source_height)))
        thumbnail_fit_in_content_area = self.calculate_aspect_ratio_rect(logical_source_qsize, QRect(0, 0, content_area_logical.width(), content_area_logical.height()))
        # Position the thumbnail with the adjusted offset to account for the 1px inset
        # Center the DWM thumbnail as precisely as possible (no manual offset)
        display_rect_qrect_logical = thumbnail_fit_in_content_area.translated(bw_logical + 1, bw_logical + 1)
        self.thumbnail_rect = display_rect_qrect_logical
        phys_dest_left = int(round(display_rect_qrect_logical.left() * dpr))
        phys_dest_top = int(round(display_rect_qrect_logical.top() * dpr))
        phys_dest_right = int(round((display_rect_qrect_logical.left() + display_rect_qrect_logical.width()) * dpr))
        phys_dest_bottom = int(round((display_rect_qrect_logical.top() + display_rect_qrect_logical.height()) * dpr))
        dest_rect_dwm = RECT(phys_dest_left, phys_dest_top, phys_dest_right, phys_dest_bottom)
        logger.debug(f"[update_thumbnail] dest_rect_dwm: {dest_rect_dwm.left}, {dest_rect_dwm.top}, {dest_rect_dwm.right}, {dest_rect_dwm.bottom}")
        props = DWM_THUMBNAIL_PROPERTIES()
        props.dwFlags = (0x00000001 | 0x00000002 | 0x00000004 | 0x00000008 | 0x00000010)
        props.rcDestination = dest_rect_dwm
        props.rcSource = RECT(0, 0, self.source_size.width(), self.source_size.height())
        # Set DWM thumbnail opacity to match overlay window opacity
        overlay_opacity = max(0, min(255, int(self.windowOpacity() * 255)))
        props.opacity = ctypes.c_byte(overlay_opacity)
        props.fVisible = True
        props.fSourceClientAreaOnly = False
        hr = dwmapi.DwmUpdateThumbnailProperties(self.thumbnail, ctypes.byref(props))
        logger.debug(f"[update_thumbnail] DwmUpdateThumbnailProperties hr={hr}")
        if hr != 0:
            logger.error(f"DwmUpdateThumbnailProperties failed: HRESULT 0x{hr:08X}")
            return False
        self.update()
        return True


    def calculate_aspect_ratio_rect(self, source_size, target_qrect):
        if not source_size or source_size.width() <= 0 or source_size.height() <= 0:
            return QRect(target_qrect.topLeft(), QSize(0, 0))
        source_w, source_h = source_size.width(), source_size.height()
        target_w, target_h = target_qrect.width(), target_qrect.height()
        if target_w <= 0 or target_h <= 0:
            return QRect(target_qrect.topLeft(), QSize(0, 0))
        source_ar, target_ar = source_w / source_h, target_w / target_h
        if source_ar > target_ar:
            render_w, render_h = target_w, int(target_w / source_ar)
        else:
            render_h, render_w = target_h, int(target_h * source_ar)
        pos_x = target_qrect.left() + (target_w - render_w) // 2
        pos_y = target_qrect.top() + (target_h - render_h) // 2
        return QRect(pos_x, pos_y, render_w, render_h)

    def get_theme_colors(self):
        """Get colors based on the current theme.
        
        Returns:
            dict: A dictionary of color values for various UI elements
        """
        theme_map = {
            "dark": {
                "border": QColor(255, 255, 255, 255),
                "background": QColor(30, 30, 30, 150),
                "text": QColor(255, 255, 255),
                "control": QColor(100, 100, 100),
                "accent": QColor(0, 122, 204)
            },
            "light": {
                "border": QColor(0, 0, 0, 255),
                "background": QColor(240, 240, 240, 150),
                "text": QColor(0, 0, 0),
                "control": QColor(180, 180, 180),
                "accent": QColor(0, 122, 204)
            }
        }
        
        # Ensure we have a theme attribute and use a safe default if not
        if not hasattr(self, 'theme') or not self.theme:
            logger.warning("Theme attribute missing in get_theme_colors, using default 'dark' theme")
            self.theme = 'dark'
        
        # Get the base theme colors
        colors = theme_map.get(self.theme.lower(), theme_map["dark"])
        
        # Add derived keys needed by other methods
        bg_color = colors["background"]
        colors["background_rgb"] = f"({bg_color.red()}, {bg_color.green()}, {bg_color.blue()})"
        colors["fill"] = colors["background"]  # Use background color as fill
        
        return colors

    def apply_theme(self, theme=None, from_global=False):
        """Apply the specified theme to the widget while preserving opacity settings.
        
        Args:
            theme (str, optional): Name of the theme to apply. If None, uses current theme.
            from_global (bool): Whether this is being called from a global theme change.
        """
        try:
            # Store current opacity settings before applying theme
            current_opacity = getattr(self, 'opacity', 1.0)
            current_border_opacity = getattr(self, 'border_opacity', 1.0)
            
            if debug_enabled():
                logger.debug(f"[Theme] Applying theme: {theme or 'current'}, "
                           f"current_opacity: {current_opacity:.3f}, "
                           f"current_border_opacity: {current_border_opacity:.3f}")
                logger.debug(f"[Theme] Call stack: {''.join(traceback.format_stack(limit=5))}")
            
            # Update theme if provided
            theme_changed = False
            if theme is not None: 
                old_theme = getattr(self, 'theme', 'unknown')
                self.theme = theme.lower()
                if old_theme != self.theme:
                    theme_changed = True
                    if debug_enabled():
                        logger.debug(f"[Theme] Theme changed from '{old_theme}' to '{self.theme}'")
            
            # Get the appropriate styles for the theme
            colors = self.get_theme_colors()
            text_rgb = f"{colors['text'].red()},{colors['text'].green()},{colors['text'].blue()}"
            border_rgb = f"{colors['border'].red()},{colors['border'].green()},{colors['border'].blue()}"
            accent_rgb = f"{colors['accent'].red()},{colors['accent'].green()},{colors['accent'].blue()}"
            
            if debug_enabled():
                logger.debug(f"[Theme] Applying styles with colors - text: rgb({text_rgb}), "
                           f"border: rgb({border_rgb}), accent: rgb({accent_rgb})")
            
            # Apply stylesheet with theme colors
            self.setStyleSheet(f"""
                QMenu {{ 
                    background-color: rgb{colors['background_rgb']}; 
                    color: rgb({text_rgb}); 
                    border: 1px solid rgb({border_rgb}); 
                    padding: 5px; 
                }}
                QMenu::item:selected {{ 
                    background-color: rgb({accent_rgb}); 
                }}
                QCheckBox {{
                    color: rgb({text_rgb});
                    padding: 4px;
                }}
                QLabel {{
                    color: rgb({text_rgb});
                }}
            """)
            
            # Update border pen with new theme colors if theme changed
            if theme_changed:
                self._update_border_pen()
                if debug_enabled():
                    logger.debug("[Theme] Updated border pen with new theme colors")
            
            # Re-apply current opacity settings after theme change
            if hasattr(self, 'opacity'):
                if debug_enabled():
                    logger.debug(f"[Theme] Restoring overlay opacity to: {current_opacity:.3f}")
                self.set_overlay_opacity(current_opacity, emit_signal=False)
                
            if hasattr(self, 'border_opacity'):
                if debug_enabled():
                    logger.debug(f"[Theme] Restoring border opacity to: {current_border_opacity:.3f}")
                self.set_border_opacity(current_border_opacity, emit_signal=False)
            
            # Force update to ensure changes are applied
            self.update()
            
            if debug_enabled():
                logger.debug("[Theme] Theme applied successfully with preserved opacity settings")
                logger.debug(f"[Theme] Final state - opacity: {getattr(self, 'opacity', 'N/A'):.3f}, "
                           f"border_opacity: {getattr(self, 'border_opacity', 'N/A'):.3f}")
                
        except Exception as e:
            logger.error(f"[Theme] Error applying theme: {e}", exc_info=debug_enabled())
            if debug_enabled():
                logger.debug(f"[Theme] Error state - theme: {getattr(self, 'theme', 'unknown')}, "
                           f"opacity: {getattr(self, 'opacity', 'N/A')}, "
                           f"border_opacity: {getattr(self, 'border_opacity', 'N/A')}")
            # Attempt to restore previous opacity values on error
            try:
                if hasattr(self, 'opacity') and 'current_opacity' in locals():
                    self.set_overlay_opacity(current_opacity, emit_signal=True)
                if hasattr(self, 'border_opacity') and 'current_border_opacity' in locals():
                    self.set_border_opacity(current_border_opacity, emit_signal=True)
            except Exception as restore_error:
                logger.error(f"[Theme] Failed to restore opacity after error: {restore_error}")

    def _safe_focus_indicator(self, action=None, *args, **kwargs):
        """Safely access the focus indicator, performing an action if provided.
        
        Args:
            action (str, optional): Name of the method to call on the focus indicator
            *args: Arguments to pass to the action method
            **kwargs: Keyword arguments to pass to the action method
            
        Returns:
            The result of the action, or None if the indicator is not available
        """
        try:
            # Check if focus indicator exists and is valid
            if not hasattr(self, '_focus_indicator') or not self._focus_indicator:
                return None
                
            # Check if the C++ object is still valid
            if hasattr(sip, 'isdeleted') and sip.isdeleted(self._focus_indicator):
                self._focus_indicator = None
                return None
                
            # If no action requested, just return the indicator
            if action is None:
                return self._focus_indicator
                
            # Perform the requested action
            method = getattr(self._focus_indicator, action, None)
            if method and callable(method):
                return method(*args, **kwargs)
                
        except RuntimeError as e:
            if "already deleted" in str(e):
                self._focus_indicator = None
        except Exception as e:
            logger.error(f"Error in safe_focus_indicator: {e}")
            
        return None

    def focusInEvent(self, event):
        """Handle focus in event to show the focus indicator."""
        super().focusInEvent(event)
        self._safe_focus_indicator('show')
        self._update_focus_indicator_position()
            
    def focusOutEvent(self, event):
        """Handle focus out event to hide the focus indicator."""
        super().focusOutEvent(event)
        self._safe_focus_indicator('hide')
            
    def _fallback_ensure_in_monitor_bounds(self, screen):
        """Fallback method to ensure the window stays within monitor bounds.
        
        This method uses a simpler approach when the primary bounds-checking fails.
        
        Args:
            screen: The screen to use for bounds checking
        """
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
        """Ensure the window is within the bounds of the target monitor.
        
        Uses a robust approach with physical monitor dimensions and falls back
        to a simpler method if needed.
        
        Args:
            target_screen: The target screen to ensure the window is within.
                          If None, uses the current screen.
        """
        try:
            # Skip if currently resizing
            if hasattr(self, '_is_resizing') and self._is_resizing:
                return
                
            if target_screen is None:
                target_screen = QGuiApplication.screenAt(self.pos())
                if not target_screen:
                    return
            
            # Try to use physical monitor info first for more accurate positioning
            from snap_utils import get_physical_monitor_info
            monitor_info = get_physical_monitor_info(target_screen)
            if not monitor_info:
                logger.warning("Could not get physical monitor info, using fallback method")
                self._fallback_ensure_in_monitor_bounds(target_screen)
                return
                
            phys_width = monitor_info.get('width', 0)
            phys_height = monitor_info.get('height', 0)
            
            if phys_width <= 0 or phys_height <= 0:
                logger.error(f"Invalid physical dimensions: {phys_width}x{phys_height}, using fallback")
                self._fallback_ensure_in_monitor_bounds(target_screen)
                return
                
            screen_geo = target_screen.availableGeometry()
            window_geo = self.geometry()
            x, y = window_geo.x(), window_geo.y()
            w, h = window_geo.width(), window_geo.height()
            
            # Calculate scaling factors
            scale_x = phys_width / screen_geo.width() if screen_geo.width() > 0 else 1.0
            scale_y = phys_height / screen_geo.height() if screen_geo.height() > 0 else 1.0
            
            # Convert logical to physical coordinates
            x_phys = int((x - screen_geo.x()) * scale_x)
            y_phys = int((y - screen_geo.y()) * scale_y)
            w_phys = int(w * scale_x)
            h_phys = int(h * scale_y)
            
            logger.debug(f"Bounds check - Screen: {target_screen.name() if hasattr(target_screen, 'name') else 'Unknown'}, "
                       f"Physical: {phys_width}x{phys_height}, "
                       f"Current: {x_phys},{y_phys} {w_phys}x{h_phys}")
            
            # Adjust physical coordinates if out of bounds
            new_x_phys, new_y_phys = x_phys, y_phys
            if x_phys < 0:
                new_x_phys = 0
            if y_phys < 0:
                new_y_phys = 0
            if x_phys + w_phys > phys_width:
                new_x_phys = max(0, phys_width - w_phys)
            if y_phys + h_phys > phys_height:
                new_y_phys = max(0, phys_height - h_phys)
            
            # Convert back to logical coordinates and move if needed
            if new_x_phys != x_phys or new_y_phys != y_phys:
                new_x = int(new_x_phys / scale_x) + screen_geo.x()
                new_y = int(new_y_phys / scale_y) + screen_geo.y()
                logger.debug(f"Adjusted position from ({x},{y}) to ({new_x},{new_y}) logical")
                self.move(new_x, new_y)
        except Exception as e:
            logger.error(f"Error in ensure_in_monitor_bounds: {e}", exc_info=True)
            self._fallback_ensure_in_monitor_bounds(target_screen)

    def _update_focus_indicator_position(self):
        """Update the focus indicator's position."""
        self._safe_focus_indicator('update_position')

    
    def _schedule_position_update(self, delay=10):
        """Schedule a position update with debouncing."""
        if hasattr(self, '_position_update_timer'):
            self._position_update_timer.stop()
            self._position_update_timer.start(delay)
            
    def moveEvent(self, event):
        """Handle window move events."""
        super().moveEvent(event)
        self._schedule_position_update()
        
    def showEvent(self, event):
        """Handle show event to ensure proper positioning."""
        super().showEvent(event)
        # Apply any pending opacity when the window is shown
        if hasattr(self, '_pending_opacity'):
            QTimer.singleShot(0, self._apply_pending_opacity)
        else:
            # Ensure opacity is applied even if there's no pending opacity
            QTimer.singleShot(0, lambda: self.set_overlay_opacity(getattr(self, 'opacity', 1.0), False))
        
        # Update focus indicator after a short delay
        QTimer.singleShot(50, self._update_focus_indicator_position)
        
    def closeEvent(self, event):
        """Clean up resources when closing the window."""
        try:
            # Clean up focus indicator using safe method
            focus_indicator = self._safe_focus_indicator()
            if focus_indicator:
                try:
                    focus_indicator.hide()
                    focus_indicator.setParent(None)
                    focus_indicator.deleteLater()
                except Exception as e:
                    logger.error(f"Error cleaning up focus indicator in closeEvent: {e}")
                finally:
                    self._focus_indicator = None
            
            # Call the cleanup method to ensure all resources are properly released
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Error in closeEvent: {e}")
        finally:
            # Always call the parent's closeEvent
            super().closeEvent(event)

    def resizeEvent(self, event):
        """Handle window resize events.
        
        Ensures opacity settings are maintained during and after resize operations.
        """
        # Refresh opacity from settings first
        try:
            self.refresh_opacity_from_settings()
        except Exception as e:
            logger.error(f"[Resize] Error refreshing opacity from settings: {e}")
        
        # Get current opacity values
        current_opacity = getattr(self, 'opacity', 1.0)
        current_border_opacity = getattr(self, 'border_opacity', 1.0)
        
        if debug_enabled():
            old_size = event.oldSize()
            new_size = event.size()
            logger.debug(f"[Resize] Starting resizeEvent - Old size: {old_size.width()}x{old_size.height()}, "
                       f"New size: {new_size.width()}x{new_size.height()}")
            logger.debug(f"[Resize] Current opacities - overlay: {current_opacity:.3f}, "
                       f"border: {current_border_opacity:.3f}")
        
        try:
            # Call parent class implementation first
            super().resizeEvent(event)
            
            # Update position and thumbnail
            self._schedule_position_update()
            
            if debug_enabled():
                logger.debug("[Resize] Updating thumbnail")
            self.update_thumbnail()
            
            # Ensure window is still valid after resize
            if not self.testAttribute(Qt.WA_WState_Created):
                logger.warning("[Resize] Window not properly created after resize, recreating...")
                self._setup_window()
                return
            
            # Apply opacities using the proper methods
            if debug_enabled():
                logger.debug(f"[Resize] Re-applying opacities - overlay: {current_opacity:.3f}, "
                           f"border: {current_border_opacity:.3f}")
            
            # Use setWindowOpacity directly to avoid recursion
            try:
                self.setWindowOpacity(current_opacity)
                if debug_enabled():
                    logger.debug(f"[Resize] Set window opacity to {current_opacity:.3f}")
            except Exception as e:
                logger.error(f"[Resize] Failed to set window opacity: {e}")
            
            # Ensure the window is still visible
            if not self.isVisible():
                if debug_enabled():
                    logger.debug("[Resize] Window not visible after resize, showing...")
                self.show()
            
            # Force a complete update of the window
            self.update()
            
            # Update the border pen with current opacity
            if hasattr(self, '_update_border_pen'):
                self._update_border_pen()
            
            # Force a repaint of the window
            self.repaint()
            
            if debug_enabled():
                logger.debug("[Resize] Resize event completed successfully")
                
        except Exception as e:
            logger.error(f"[Resize] Error during resize event: {e}", exc_info=debug_enabled())
            
            # Attempt to recover by recreating the window with current settings
            try:
                logger.warning("[Resize] Attempting to recover by recreating window...")
                self._setup_window()
                
                # Re-apply opacities after recovery
                self.set_overlay_opacity(current_opacity, emit_signal=False)
                self.set_border_opacity(current_border_opacity, emit_signal=False)
                
            except Exception as recovery_error:
                logger.critical(f"[Resize] Failed to recover from resize error: {recovery_error}", 
                              exc_info=debug_enabled())

    def paintEvent(self, event):
        # Refresh opacity from settings at the start of each paint
        try:
            self.refresh_opacity_from_settings()
        except Exception as e:
            logger.error(f"[Paint] Error refreshing opacity from settings: {e}")
        
        # Get current opacity values
        current_opacity = getattr(self, 'opacity', 1.0)
        current_border_opacity = getattr(self, 'border_opacity', 1.0)
        
        # Log paint event with current state
        if debug_enabled():
            logger.debug(f"[Paint] Starting paintEvent - "
                       f"opacity: {current_opacity:.3f}, "
                       f"border_opacity: {current_border_opacity:.3f}, "
                       f"windowOpacity: {self.windowOpacity():.3f}")
        
        # Ensure window opacity is correct
        try:
            if not math.isclose(self.windowOpacity(), current_opacity, abs_tol=0.01):
                if debug_enabled():
                    logger.warning(f"[Paint] Window opacity mismatch! "
                                 f"Current: {self.windowOpacity():.3f}, "
                                 f"Expected: {current_opacity:.3f}. "
                                 f"Correcting...")
                self.setWindowOpacity(current_opacity)
        except Exception as e:
            logger.error(f"[Paint] Error checking window opacity: {e}")
        
        painter = QPainter(self)
        
        try:
            # Draw the main content
            painter.setRenderHint(QPainter.Antialiasing)
            colors = self.get_theme_colors()

            # Draw DWM thumbnail first (if present)
            # (Assuming thumbnail is rendered by Windows after DwmUpdateThumbnailProperties, so just ensure border is drawn last)

            # Draw background with current fill color
            try:
                fill_color = colors['fill']
                # Don't apply opacity here - it's already handled by windowOpacity
                painter.fillRect(self.rect(), fill_color)
                
                if debug_enabled():
                    logger.debug(f"[Paint] Filled with color {fill_color.name()}")
                    
            except Exception as e:
                logger.error(f"[Paint] Error filling background: {e}")
            
            # Ensure border pen is up to date with current opacity
            try:
                self._update_border_pen()
            except Exception as e:
                logger.error(f"[Paint] Error updating border pen: {e}")
                
        except Exception as e:
            logger.error(f"[Paint] Error in paintEvent: {e}", exc_info=debug_enabled())
            # Re-raise to ensure proper error handling
            raise

        # Draw borders with opacity
        bw_logical = self._border_width
        
        # Use the border pen that was initialized with the correct opacity
        painter.setPen(self.border_pen)
        painter.setBrush(Qt.NoBrush)
        
        if debug_enabled():
            border_color = self.border_pen.color()
            logger.debug(f"[Paint] Drawing borders with color: RGB({border_color.red()},{border_color.green()},{border_color.blue()}), "
                       f"Alpha: {border_color.alpha()}")

        overlay_w = self.rect().width()
        overlay_h = self.rect().height()
        
        if debug_enabled():
            logger.debug(f"[Paint] Drawing borders - width: {overlay_w}, height: {overlay_h}, border_width: {bw_logical}")

        # Draw the four border edges
        painter.drawRect(0, 0, overlay_w, bw_logical)  # Top
        painter.drawRect(0, overlay_h - bw_logical, overlay_w, bw_logical)  # Bottom
        painter.drawRect(0, bw_logical, bw_logical, overlay_h - 2 * bw_logical)  # Left
        painter.drawRect(overlay_w - bw_logical, bw_logical, bw_logical, overlay_h - 2 * bw_logical)  # Right

        painter.end()
        
        # Update focus indicator position
        if hasattr(self, '_focus_indicator') and self._focus_indicator and self._focus_indicator.isVisible():
            self._focus_indicator.update_position(self.rect())
            
        if debug_enabled():
            logger.debug("[Paint] paintEvent completed")


    @log_perf(level=logging.DEBUG)
    def cleanup(self):
        """Clean up resources used by the BorderWidget.
        
        This method is idempotent and can be safely called multiple times.
        It ensures all resources are properly released and prevents access to deleted C++ objects.
        """
        try:
            logger.debug("Cleaning up BorderWidget resources")
            
            # Skip if already cleaned up
            if getattr(self, '_cleaned_up', False):
                logger.debug("BorderWidget already cleaned up, skipping")
                return
                
            # Disconnect all signals first to prevent any callbacks during cleanup
            try:
                self.blockSignals(True)
                
                # Clean up focus indicator safely
                if hasattr(self, '_focus_indicator'):
                    with DebugTimer("Cleaning up focus indicator"):
                        try:
                            # Check if the C++ object still exists
                            if sip.isdeleted(self._focus_indicator) if hasattr(sip, 'isdeleted') else False:
                                logger.debug("Focus indicator C++ object already deleted")
                            else:
                                self._focus_indicator.hide()
                                self._focus_indicator.setParent(None)
                                self._focus_indicator.deleteLater()
                        except RuntimeError as e:
                            if "wrapped C/C++ object" in str(e) or "already deleted" in str(e):
                                logger.debug("Focus indicator already deleted")
                            else:
                                logger.error(f"RuntimeError cleaning up focus indicator: {e}")
                        except Exception as e:
                            logger.error(f"Error cleaning up focus indicator: {e}")
                        finally:
                            self._focus_indicator = None
            
                # Clean up thumbnail if it exists
                if hasattr(self, '_cleanup_thumbnail'):
                    with DebugTimer("Cleaning up thumbnail"):
                        try:
                            self._cleanup_thumbnail()
                        except Exception as e:
                            log_exception("Error cleaning up thumbnail", e)
            
                # Clean up key passthrough if it exists
                if hasattr(self, 'key_passthrough'):
                    with DebugTimer("Cleaning up key passthrough"):
                        try:
                            # Clear any callbacks to prevent reference cycles
                            if hasattr(self.key_passthrough, 'set_state_change_callback'):
                                self.key_passthrough.set_state_change_callback(None)
                        except Exception as e:
                            log_exception("Error during key passthrough cleanup", e)
                        finally:
                            self.key_passthrough = None
                
                # Clean up timers
                if hasattr(self, '_position_update_timer'):
                    try:
                        self._position_update_timer.stop()
                        self._position_update_timer.deleteLater()
                    except Exception as e:
                        logger.error(f"Error cleaning up position update timer: {e}")
                    finally:
                        self._position_update_timer = None
                
                # Disconnect from app instance signals
                if hasattr(self, 'app_instance') and self.app_instance:
                    try:
                        if hasattr(self.app_instance, 'key_passthrough_setting_changed'):
                            self.app_instance.key_passthrough_setting_changed.disconnect(self._handle_key_passthrough_change)
                    except Exception as e:
                        logger.error(f"Error disconnecting app instance signals: {e}")
                
            finally:
                self.blockSignals(False)
                
        except Exception as e:
            log_exception("Unexpected error during cleanup", e)
            
        finally:
            # Mark as cleaned up to prevent duplicate cleanup
            self._cleaned_up = True
            
            # Clear other references
            self.app_instance = None
            
            # Schedule for deletion in the next event loop iteration
            try:
                if not sip.isdeleted(self) if hasattr(sip, 'isdeleted') else False:
                    self.deleteLater()
            except RuntimeError as e:
                if "wrapped C/C++ object" not in str(e) and "already deleted" not in str(e):
                    logger.error(f"Error in deleteLater: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in deleteLater: {e}")

    def __del__(self):
        """Destructor to ensure all resources are properly cleaned up."""
        try:
            logger.debug("BorderWidget __del__ called")
            # Cleanup will be handled by closeEvent and Qt's parent-child hierarchy
            pass
        except Exception as e:
            # Prevent exceptions in __del__ from being raised
            try:
                log_exception("Error during BorderWidget destructor", e)
            except Exception:
                pass
    


    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle key release events to prevent them from propagating to other windows."""
        event.accept()
        logger.debug(f"Key release event - key: {event.key()} (0x{event.key():X})")
        
    def _get_key_name(self, key_code: int) -> str:
        """Convert key code to human-readable name."""
        # Map of Qt key codes to names
        key_names = {
            Qt.Key_Left: "LEFT",
            Qt.Key_Right: "RIGHT",
            Qt.Key_Up: "UP",
            Qt.Key_Down: "DOWN",
            Qt.Key_Space: "SPACE",
            Qt.Key_Return: "ENTER",
            Qt.Key_Enter: "ENTER",
            Qt.Key_Escape: "ESC",
            Qt.Key_Tab: "TAB",
            Qt.Key_Backspace: "BACKSPACE",
            Qt.Key_Delete: "DELETE",
            Qt.Key_Home: "HOME",
            Qt.Key_End: "END",
            Qt.Key_PageUp: "PAGE UP",
            Qt.Key_PageDown: "PAGE DOWN",
            Qt.Key_Insert: "INSERT",
            Qt.Key_Print: "PRINT",
            Qt.Key_Pause: "PAUSE",
            Qt.Key_CapsLock: "CAPS LOCK",
            Qt.Key_NumLock: "NUM LOCK",
            Qt.Key_ScrollLock: "SCROLL LOCK",
            Qt.Key_MediaPlay: "MEDIA PLAY",
            Qt.Key_MediaPause: "MEDIA PAUSE",
            Qt.Key_MediaStop: "MEDIA STOP",
            Qt.Key_MediaNext: "MEDIA NEXT",
            Qt.Key_MediaPrevious: "MEDIA PREVIOUS",
            Qt.Key_MediaTogglePlayPause: "MEDIA TOGGLE PLAY/PAUSE",
            Qt.Key_VolumeUp: "VOLUME UP",
            Qt.Key_VolumeDown: "VOLUME DOWN",
            Qt.Key_VolumeMute: "VOLUME MUTE"
        }
        
        # Check if it's a function key (F1-F12)
        if Qt.Key_F1 <= key_code <= Qt.Key_F12:
            return f"F{key_code - Qt.Key_F1 + 1}"
            
        # Check if it's a letter (A-Z)
        if Qt.Key_A <= key_code <= Qt.Key_Z:
            return chr(ord('A') + key_code - Qt.Key_A)
            
        # Check if it's a number (0-9)
        if Qt.Key_0 <= key_code <= Qt.Key_9:
            return str(key_code - Qt.Key_0)
            
        # Check if it's a keypad number (0-9)
        if Qt.Key_0 <= key_code <= Qt.Key_9:
            return f"NUMPAD {key_code - Qt.Key_0}"
            
        # Check if it's in our name mapping
        if key_code in key_names:
            return key_names[key_code]
            
        # Return hex code as string if no name found
        return f"0x{key_code:X}"
        
    def _get_modifiers_string(self, modifiers: int) -> str:
        """Convert Qt modifiers to string representation."""
        mods = []
        if modifiers & Qt.ShiftModifier:
            mods.append("Shift")
        if modifiers & Qt.ControlModifier:
            mods.append("Ctrl")
        if modifiers & Qt.AltModifier:
            mods.append("Alt")
        if modifiers & Qt.MetaModifier:
            mods.append("Meta")
        return "+".join(mods) if mods else "None"
        
    def _get_window_info(self, hwnd: int) -> str:
        """Get window information string for logging."""
        if not hwnd or not win32gui.IsWindow(hwnd):
            return "INVALID_WINDOW"
            
        try:
            # Get window text
            window_text = win32gui.GetWindowText(hwnd)
            
            # Get window class
            window_class = win32gui.GetClassName(hwnd)
            
            # Get process name
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process_name = "Unknown"
            try:
                process = psutil.Process(pid)
                process_name = process.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
            return f"'{window_text}' (Class: {window_class}, PID: {pid}, Process: {process_name})"
            
        except Exception as e:
            return f"Error getting window info: {str(e)}"
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events for key passthrough."""
        # Always accept the event first to prevent it from propagating to other windows
        event.accept()
        
        # Initialize key_name and vk_code at the start
        key_name = None
        vk_code = None
        
        try:
            key = event.key()
            key_text = event.text()
            
            # Get key and modifier information
            key_name = self._get_key_name(key)
            modifiers_str = self._get_modifiers_string(event.modifiers())
            
            # Log the key press with detailed information
            logger.info(f"[KEY_PRESS] Key: {key_name} (0x{key:X}), Text: '{key_text}', Modifiers: {modifiers_str}")
            
            # Apply initial geometry if needed
            if not hasattr(self, '_initial_geometry_applied') and hasattr(self, '_passed_initial_geometry') and self._passed_initial_geometry:
                logger.debug(f"Applying passed initial_geometry: {self._passed_initial_geometry}")
                self.setGeometry(self._passed_initial_geometry)
                self._initial_geometry_applied = True
            
            # Check key passthrough state
            passthrough_enabled = hasattr(self, 'key_passthrough') and self.key_passthrough and self.key_passthrough.is_enabled()
            logger.debug(f"Key passthrough state - enabled: {passthrough_enabled}")
            
            # Get and log target window info
            target_hwnd = getattr(self, 'hwnd', None)
            is_window_valid = bool(target_hwnd and win32gui.IsWindow(target_hwnd))
            logger.debug(f"Target HWND: {target_hwnd}, Valid: {is_window_valid}")
            
            # Skip if key passthrough is not properly set up
            if not passthrough_enabled or not is_window_valid:
                logger.debug("Skipping key passthrough - not enabled or invalid target window")
                return
        
            # Handle special keys first
            if key == Qt.Key_Escape:
                logger.debug("Escape key pressed, closing overlay")
                self.close()
                return
                
            # Define key mappings from Qt to Windows VK codes
            key_mappings = {
                Qt.Key_Enter: ("ENTER", win32con.VK_RETURN),
                Qt.Key_Return: ("ENTER", win32con.VK_RETURN),
                Qt.Key_Shift: ("SHIFT", win32con.VK_SHIFT),
                Qt.Key_Control: ("CTRL", win32con.VK_CONTROL),
                Qt.Key_Alt: ("ALT", win32con.VK_MENU),
                Qt.Key_Left: ("LEFT", win32con.VK_LEFT),
                Qt.Key_Right: ("RIGHT", win32con.VK_RIGHT),
                Qt.Key_Up: ("UP", win32con.VK_UP),
                Qt.Key_Down: ("DOWN", win32con.VK_DOWN),
                Qt.Key_Tab: ("TAB", win32con.VK_TAB),
                Qt.Key_Backspace: ("BACKSPACE", win32con.VK_BACK),
                Qt.Key_Delete: ("DELETE", win32con.VK_DELETE),
                Qt.Key_Home: ("HOME", win32con.VK_HOME),
                Qt.Key_End: ("END", win32con.VK_END),
                Qt.Key_PageUp: ("PAGEUP", win32con.VK_PRIOR),
                Qt.Key_PageDown: ("PAGEDOWN", win32con.VK_NEXT),
                Qt.Key_Insert: ("INSERT", win32con.VK_INSERT),
                Qt.Key_Print: ("PRINT", win32con.VK_PRINT),
                Qt.Key_Pause: ("PAUSE", win32con.VK_PAUSE),
                Qt.Key_CapsLock: ("CAPSLOCK", win32con.VK_CAPITAL),
                Qt.Key_NumLock: ("NUMLOCK", win32con.VK_NUMLOCK),
                Qt.Key_ScrollLock: ("SCROLLLOCK", win32con.VK_SCROLL),
                Qt.Key_Space: ("SPACE", win32con.VK_SPACE),
                Qt.Key_0: ("0", 0x30),
                Qt.Key_1: ("1", 0x31),
                Qt.Key_2: ("2", 0x32),
                Qt.Key_3: ("3", 0x33),
                Qt.Key_4: ("4", 0x34),
                Qt.Key_5: ("5", 0x35),
                Qt.Key_6: ("6", 0x36),
                Qt.Key_7: ("7", 0x37),
                Qt.Key_8: ("8", 0x38),
                Qt.Key_9: ("9", 0x39),
                Qt.Key_A: ("A", 0x41),
                Qt.Key_B: ("B", 0x42),
                Qt.Key_C: ("C", 0x43),
                Qt.Key_D: ("D", 0x44),
                Qt.Key_E: ("E", 0x45),
                Qt.Key_F: ("F", 0x46),
                Qt.Key_G: ("G", 0x47),
                Qt.Key_H: ("H", 0x48),
                Qt.Key_I: ("I", 0x49),
                Qt.Key_J: ("J", 0x4A),
                Qt.Key_K: ("K", 0x4B),
                Qt.Key_L: ("L", 0x4C),
                Qt.Key_M: ("M", 0x4D),
                Qt.Key_N: ("N", 0x4E),
                Qt.Key_O: ("O", 0x4F),
                Qt.Key_P: ("P", 0x50),
                Qt.Key_Q: ("Q", 0x51),
                Qt.Key_R: ("R", 0x52),
                Qt.Key_S: ("S", 0x53),
                Qt.Key_T: ("T", 0x54),
                Qt.Key_U: ("U", 0x55),
                Qt.Key_V: ("V", 0x56),
                Qt.Key_W: ("W", 0x57),
                Qt.Key_X: ("X", 0x58),
                Qt.Key_Y: ("Y", 0x59),
                Qt.Key_Z: ("Z", 0x5A),
                Qt.Key_F1: ("F1", win32con.VK_F1),
                Qt.Key_F2: ("F2", win32con.VK_F2),
                Qt.Key_F3: ("F3", win32con.VK_F3),
                Qt.Key_F4: ("F4", win32con.VK_F4),
                Qt.Key_F5: ("F5", win32con.VK_F5),
                Qt.Key_F6: ("F6", win32con.VK_F6),
                Qt.Key_F7: ("F7", win32con.VK_F7),
                Qt.Key_F8: ("F8", win32con.VK_F8),
                Qt.Key_F9: ("F9", win32con.VK_F9),
                Qt.Key_F10: ("F10", win32con.VK_F10),
                Qt.Key_F11: ("F11", win32con.VK_F11),
                Qt.Key_F12: ("F12", win32con.VK_F12),
            }
            
            # Handle keys from the mapping first
            if key in key_mappings:
                key_name, vk_code = key_mappings[key]
                logger.debug(f"Mapped key {key} to {key_name} (VK: 0x{vk_code:X})")
            # Fallback for keys not in the mapping
            elif Qt.Key_0 <= key <= Qt.Key_9:
                key_name = f"NUM_{key - Qt.Key_0}"
                vk_code = 0x30 + (key - Qt.Key_0)  # 0x30 is VK_0
                logger.debug(f"Mapped numeric key {key} to {key_name} (VK: 0x{vk_code:X})")
            elif Qt.Key_A <= key <= Qt.Key_Z:
                key_name = f"KEY_{chr(ord('A') + key - Qt.Key_A)}"
                vk_code = 0x41 + (key - Qt.Key_A)  # 0x41 is VK_A
                logger.debug(f"Mapped alpha key {key} to {key_name} (VK: 0x{vk_code:X})")
            else:
                key_name = f"UNKNOWN_KEY_{key}"
                vk_code = key
                logger.debug(f"Unmapped key {key} using raw code 0x{key:X}")
            
            if key_name and vk_code is not None:
                if debug_enabled():
                    logger.debug(f"Key press detected: {key_name} (VK: 0x{vk_code:02X})")
                
                # Forward the key to the target window if we have a valid VK code
                try:
                    logger.debug(f"Sending key {key_name} (VK: 0x{vk_code:02X}) to window {target_hwnd}")
                    
                    # Use the key_passthrough system
                    if hasattr(self, 'key_passthrough') and self.key_passthrough and self.key_passthrough.is_enabled():
                        try:
                            logger.debug(f"Attempting to send key {key_name} (VK: 0x{vk_code:02X}) via key_passthrough")
                            result = self.key_passthrough.send_key(vk_code)
                            logger.debug(f"Key passthrough result: {result}")
                            if result:
                                event.accept()
                                return
                            else:
                                logger.warning(f"Key passthrough failed for {key_name} (VK: 0x{vk_code:02X})")
                        except Exception as e:
                            logger.error(f"Error in key_passthrough.send_key: {e}", exc_info=True)
                    else:
                        logger.debug("Key passthrough not available or not enabled")
                    
                    # Fallback direct PostMessage without stealing focus
                    if target_hwnd and win32gui.IsWindow(target_hwnd):
                        # Initialize attached flag outside try block
                        attached = False
                        thread_id = None
                        current_thread = None
                        
                        try:
                            thread_id = win32process.GetWindowThreadProcessId(target_hwnd)[0]
                            current_thread = win32api.GetCurrentThreadId()
                            
                            # Try to attach to the target window's input queue
                            if thread_id != current_thread:
                                try:
                                    win32process.AttachThreadInput(current_thread, thread_id, True)
                                    attached = True
                                    logger.debug(f"Attached to target thread {thread_id}")
                                except Exception as e:
                                    logger.debug(f"Failed to attach to thread: {e}")
                            
                            # Prepare the lParam for key events
                            # bits 0-15: repeat count (1)
                            # bits 16-23: scan code (0 = use default)
                            # bit 24: extended key flag (0 = not extended)
                            # bit 29: context code (0 = key was pressed while the key is up)
                            # bit 30: previous key state (0 = key was up)
                            # bit 31: transition state (0 = key press, 1 = key release)
                            lparam_press = 0x00000001 | (0 << 30)
                            lparam_release = lparam_press | (1 << 30) | (1 << 31)
                            
                            # Get the scan code for the virtual key
                            scan_code = win32api.MapVirtualKey(vk_code, 0)
                            lparam_press = (scan_code << 16) | 1
                            lparam_release = lparam_press | (1 << 30) | (1 << 31)
                            
                            # Get window info for logging
                            window_info = self._get_window_info(target_hwnd)
                            
                            # Log the key action with window info
                            logger.info(f"[SEND_KEY] Sending {key_name} (0x{vk_code:X}) to window: {window_info}")
                            
                            # Send the key press
                            result_down = win32api.PostMessage(target_hwnd, win32con.WM_KEYDOWN, vk_code, lparam_press)
                            time.sleep(0.01)  # Small delay between key down and up
                            
                            # Also send WM_CHAR for proper character input
                            result_char = win32api.PostMessage(target_hwnd, win32con.WM_CHAR, vk_code, lparam_press)
                            
                            # Send the key release
                            result_up = win32api.PostMessage(target_hwnd, win32con.WM_KEYUP, vk_code, lparam_release)
                            
                            # Log the results
                            logger.info(
                                f"[SEND_KEY_RESULT] Key: {key_name}, "
                                f"Results - DOWN: {result_down!=0}, CHAR: {result_char!=0}, UP: {result_up!=0}, "
                                f"Target: {window_info}"
                            )
                            return True
                            
                        except Exception as e:
                            logger.error(f"Error in direct PostMessage: {e}", exc_info=True)
                            return False
                            
                        finally:
                            # Always detach if we attached
                            if attached and thread_id is not None and current_thread is not None:
                                try:
                                    win32process.AttachThreadInput(current_thread, thread_id, False)
                                    logger.debug(f"Detached from thread {thread_id}")
                                except Exception as e:
                                    logger.debug(f"Failed to detach from thread: {e}")
                    else:
                        logger.warning(f"Target window {target_hwnd} is not valid for key forwarding")
                    
                except Exception as e:
                    logger.error(f"Error forwarding key press to window: {e}", exc_info=True)
                
                # Mark the event as handled to prevent duplicate processing
                event.accept()
            else:
                # Unhandled key
                if debug_enabled():
                    logger.debug(f"Unhandled key press: {key} (0x{key:X})")
                event.ignore()
            
        except Exception as e:
            logger.error(f"Error in keyPressEvent: {e}", exc_info=True)
            event.ignore()
            
            try:
                # Log the key press
                logger.debug(f"Sending key {key_name} (VK: {vk_code}) to window {getattr(self, 'hwnd', 'N/A')}")
                
                # Safely get the target window handle
                target_hwnd = getattr(self, 'hwnd', None)
                if not target_hwnd:
                    logger.warning(f"No target window handle available for key {key_name}")
                    return
                
                # Validate the window
                if hasattr(self, 'window_validation') and hasattr(self.window_validation, 'is_valid_window'):
                    if not self.window_validation.is_valid_window(target_hwnd):
                        logger.error(f"Invalid window handle: {target_hwnd} for key {key_name}")
                        return
                
                # Log key passthrough state
                logger.debug(f"Key passthrough enabled: {self.key_passthrough.is_enabled()}, "
                           f"Aggressive mode: {self.key_passthrough.is_aggressive_mode()}")
                
                # Use the key passthrough module to send the key
                if hasattr(self.key_passthrough, 'send_key'):
                    try:
                        # Convert key code to VKCode if needed
                        from key_passthrough import VKCode
                        if not isinstance(vk_code, VKCode):
                            vk_code = VKCode(vk_code)
                            
                        result = self.key_passthrough.send_key(vk_code)
                        logger.debug(f"Key {key_name} send result: {result}")
                        
                        if not result:
                            logger.warning(f"Failed to send key {key_name} to window {target_hwnd}")
                    except Exception as e:
                        logger.error(f"Error in key passthrough: {e}")
                else:
                    logger.error("Key passthrough module is missing send_key method")
                
            except Exception as e:
                logger.error(f"Error forwarding key {key_name}: {str(e)}", exc_info=True)
            
            return  # Always consume the event to prevent duplicate processing
        
        # For debugging other keys
        # logger.debug(f"Unhandled key press: {event.key()}")
        event.ignore()  # Let the event propagate if not handled
        return
        
        # Clean up context menus
        menus_to_clean = [
            ('context_menu', getattr(self, 'context_menu', None)),
            ('switch_to_window_menu', getattr(self, 'switch_to_window_menu', None)),
            ('switch_to_monitor_menu', getattr(self, 'switch_to_monitor_menu', None))
        ]
        
        for name, menu in menus_to_clean:
            if menu:
                try:
                    menu.clear()
                    menu.deleteLater()
                    logger.debug(f"Cleaned up {name}")
                except Exception as e:
                    log_exception(f"Error cleaning up {name}", e)
                finally:
                    setattr(self, name, None)
        
        # Clean up focus indicator
        if hasattr(self, '_focus_indicator') and self._focus_indicator:
            try:
                self._focus_indicator.hide()
                self._focus_indicator.setParent(None)
                self._focus_indicator.deleteLater()
                logger.debug("Cleaned up focus indicator")
            except Exception as e:
                log_exception("Error cleaning up focus indicator", e)
        
        # Clean up key passthrough
        if hasattr(self, 'key_passthrough'):
            try:
                self.key_passthrough = None
            except Exception as e:
                log_exception("Error cleaning up key_passthrough", e)
        
        # Clear references
        self.hwnd = None
        if hasattr(self, 'app_instance'):
            self.app_instance = None
        
        logger.debug("Close event handling complete")
        

def keyPressEvent(self, event: QKeyEvent):
    """Handle key press events for key passthrough."""
    if not event:
        return
    
    # Apply initial geometry before processing key events
    if not hasattr(self, '_initial_geometry_applied') and hasattr(self, '_passed_initial_geometry') and self._passed_initial_geometry:
        logger.debug(f"Applying passed initial_geometry: {self._passed_initial_geometry}")
        self.setGeometry(self._passed_initial_geometry)
        self._initial_geometry_applied = True
    
    key = event.key()
    
    # Handle Escape key to close the window
    if key == Qt.Key_Escape:
        self.close()
        event.accept()
        return
    
    # Check if key passthrough is enabled and we have a valid target window
    if not hasattr(self, 'key_passthrough') or not self.key_passthrough or \
       not hasattr(self.key_passthrough, 'is_enabled') or not self.key_passthrough.is_enabled() or \
       not hasattr(self, 'hwnd') or not self.hwnd:
        event.ignore()
        return
    
    # Define key mappings from Qt to Windows VK codes
    key_mappings = {
        Qt.Key_Space: ("SPACE", win32con.VK_SPACE),
        Qt.Key_Return: ("ENTER", win32con.VK_RETURN),
        Qt.Key_Enter: ("ENTER", win32con.VK_RETURN),
        Qt.Key_Shift: ("SHIFT", win32con.VK_SHIFT),
        Qt.Key_Control: ("CTRL", win32con.VK_CONTROL),
        Qt.Key_Alt: ("ALT", win32con.VK_MENU),
        Qt.Key_Left: ("LEFT", win32con.VK_LEFT),
        Qt.Key_Right: ("RIGHT", win32con.VK_RIGHT),
        Qt.Key_Up: ("UP", win32con.VK_UP),
        Qt.Key_Down: ("DOWN", win32con.VK_DOWN),
        Qt.Key_Tab: ("TAB", win32con.VK_TAB),
        Qt.Key_Backspace: ("BACKSPACE", win32con.VK_BACK),
        Qt.Key_Delete: ("DELETE", win32con.VK_DELETE),
        Qt.Key_Home: ("HOME", win32con.VK_HOME),
        Qt.Key_End: ("END", win32con.VK_END),
        Qt.Key_PageUp: ("PAGEUP", win32con.VK_PRIOR),
        Qt.Key_PageDown: ("PAGEDOWN", win32con.VK_NEXT),
        Qt.Key_Insert: ("INSERT", win32con.VK_INSERT),
        Qt.Key_Print: ("PRINT", win32con.VK_PRINT),
        Qt.Key_Pause: ("PAUSE", win32con.VK_PAUSE),
        Qt.Key_CapsLock: ("CAPSLOCK", win32con.VK_CAPITAL),
        Qt.Key_NumLock: ("NUMLOCK", win32con.VK_NUMLOCK),
        Qt.Key_ScrollLock: ("SCROLLLOCK", win32con.VK_SCROLL),
    }
    
    # Handle alphanumeric keys
    if Qt.Key_0 <= key <= Qt.Key_9:
        key_name = f"NUM_{key - Qt.Key_0}"
        vk_code = win32con.VK_0 + (key - Qt.Key_0)
    elif Qt.Key_A <= key <= Qt.Key_Z:
        key_name = f"KEY_{chr(ord('A') + key - Qt.Key_A)}"
        vk_code = ord(chr(key).upper())  # Convert to ASCII and then to VK code
    elif key in key_mappings:
        key_name, vk_code = key_mappings[key]
    else:
        # Unhandled key
        event.ignore()
        return
    
    logger.debug(f"Key press detected: {key_name} (VK: {vk_code})")
    
    # Mark the event as handled to prevent duplicate processing
    event.accept()
    
    try:
        # Log the key press with target window info
        target_hwnd = getattr(self.key_passthrough, '_target_hwnd', None) or getattr(self, 'hwnd', None)
        logger.debug(f"Sending key {key_name} (VK: {vk_code}) to window {target_hwnd}")
        
        if not target_hwnd:
            logger.warning(f"No target window handle available for key {key_name}")
            return
            
        # Use the key passthrough module to send the key
        if hasattr(self.key_passthrough, 'send_key'):
            result = self.key_passthrough.send_key(vk_code)
            logger.debug(f"Key {key_name} send result: {result}")
            
            if not result:
                logger.warning(f"Failed to send key {key_name} to window {target_hwnd}")
        else:
            logger.error("Key passthrough module is missing send_key method")
        
    except Exception as e:
        logger.error(f"Error forwarding key {key_name}: {str(e)}", exc_info=True)
    
    # Prevent quick-switch if the overlay is locked
    if hasattr(self, '_is_window_locked') and self._is_window_locked:
        logger.debug("Quick switch prevented: Overlay is locked")
        return
            
        if not self.hwnd:
            logger.info("Quick switch ignored: Overlay is not targeting a specific window")
            return
            
        if not self.app_instance or not hasattr(self.app_instance, 'window_switcher'):
            logger.warning("Quick switch aborted: No app instance or window_switcher available")
            return

        try:
            # Get the currently focused window
            focused_hwnd = win32gui.GetForegroundWindow()
            
            # Delegate the quick switch to WindowSwitcher
            success = self.app_instance.window_switcher.quick_switch_windows(
                self.hwnd, 
                focused_hwnd,
                self
            )
            
            if success:
                logger.info(f"Quick switch completed between overlay {self.hwnd} and focused window {focused_hwnd}")
                # Set the timestamp to prevent immediate auto-switching
                if hasattr(self.app_instance, '_last_quick_switch_time'):
                    self.app_instance._last_quick_switch_time = time.time()
                    logger.debug("Set quick switch cooldown timestamp")
            else:
                logger.warning("Quick switch operation failed")
                
        except Exception as e:
            log_exception("Error during quick switch", e)
            
    @log_perf(level=logging.DEBUG, threshold_ms=10.0)
    def _handle_swap_window(self, target_hwnd: int):
        import time  # Import time module at the start of the method
        """Handle window switching by delegating to WindowSwitcher.
        
        This method is called by the WindowSwitcher to perform the actual window swap
        after the quick switch logic has determined the target window.
        
        Args:
            target_hwnd: The window handle to switch to
            
        Returns:
            bool: True if the swap was successful, False otherwise
        """
        logger.debug("=== SWAP WINDOW DEBUG ===")
        logger.debug(f"Target HWND: {target_hwnd}, Current HWND: {getattr(self, 'hwnd', None)}")
        
        try:
            # Basic validation
            if not target_hwnd or not win32gui.IsWindow(target_hwnd):
                logger.error(f"Invalid target window handle: {target_hwnd}")
                return False
                
            if not hasattr(self, 'app_instance') or not self.app_instance:
                logger.error("Cannot swap window: No app instance available")
                return False
                
            if not hasattr(self.app_instance, 'window_switcher'):
                logger.error("Cannot swap window: WindowSwitcher not available")
                return False
            
            # Store the current window before switching
            previous_hwnd = getattr(self, 'hwnd', None)
            logger.debug(f"Previous HWND: {previous_hwnd}")
            
            # If we're already showing this window, nothing to do
            if target_hwnd == previous_hwnd:
                logger.debug("Target window is already displayed in overlay")
                return True
            
            try:
                # Clean up the existing thumbnail before changing the HWND
                if hasattr(self, '_cleanup_thumbnail'):
                    logger.debug("Cleaning up existing thumbnail")
                    self._cleanup_thumbnail()
                
                # Store the previous window state for restoration if needed
                was_minimized = False
                if previous_hwnd and win32gui.IsWindow(previous_hwnd):
                    was_minimized = win32gui.IsIconic(previous_hwnd)
                
                # Update the overlay to show the new window
                self.hwnd = target_hwnd
                logger.debug(f"Set new HWND: {self.hwnd}")
                
                # If the window is minimized, restore it (but don't focus it)
                if win32gui.IsIconic(target_hwnd):
                    logger.debug(f"Restoring minimized window: {target_hwnd}")
                    try:
                        win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                        logger.debug("Window restored successfully")
                        # Small delay to let the window restore
                        import time
                        time.sleep(0.1)
                    except Exception as restore_error:
                        logger.error(f"Failed to restore window: {restore_error}")
                        # Continue even if restore fails - the window might still be usable
                
                # Update the thumbnail with retry logic
                thumbnail_registered = False
                if hasattr(self, 'register_thumbnail'):
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            logger.debug(f"Registering new thumbnail (attempt {attempt + 1}/{max_retries})")
                            # Ensure any existing thumbnail is cleaned up
                            self._cleanup_thumbnail()
                            
                            # Skip if trying to register our own window as a thumbnail
                            if target_hwnd == self.winId():
                                logger.error("Cannot register overlay window as its own thumbnail")
                                break
                                
                            # Register the new thumbnail
                            success = self.register_thumbnail()
                            
                            # Verify thumbnail was registered
                            if success and hasattr(self, 'thumbnail') and self.thumbnail and self.thumbnail.value:
                                logger.debug("Thumbnail registered successfully")
                                thumbnail_registered = True
                                break
                            else:
                                logger.warning(f"Thumbnail registration attempt {attempt + 1} failed")
                                time.sleep(0.1)  # Small delay before retry
                        except Exception as thumb_error:
                            logger.error(f"Attempt {attempt + 1} failed to register thumbnail: {thumb_error}")
                            time.sleep(0.1)  # Small delay before retry
                
                if not thumbnail_registered:
                    logger.error("Failed to register thumbnail after all attempts")
                    # Only try final time if we're not trying to register our own window
                    if target_hwnd != self.winId():
                        time.sleep(0.2)
                        self._cleanup_thumbnail()
                        if self.register_thumbnail():
                            thumbnail_registered = True
                
                # If we still don't have a thumbnail, log it but continue
                if not thumbnail_registered:
                    logger.warning("Proceeding without thumbnail registration")
                
                # Update key passthrough target if available
                if hasattr(self, 'key_passthrough') and self.key_passthrough:
                    try:
                        self.key_passthrough.set_target_window(target_hwnd)
                        logger.debug("Updated key passthrough target window")
                    except Exception as e:
                        logger.error(f"Failed to update key passthrough target: {e}")
                
                # Update MRU list if available
                if hasattr(self.app_instance, 'update_mru_list'):
                    try:
                        logger.debug("Updating MRU list")
                        self.app_instance.update_mru_list(target_hwnd)
                        if previous_hwnd:
                            self.app_instance.update_mru_list(previous_hwnd)
                    except Exception as mru_error:
                        logger.error(f"Failed to update MRU list: {mru_error}")
                
                # Force a full repaint and update
                self.update()
                self.repaint()
                QApplication.processEvents()
                
                logger.info(f"Successfully switched overlay to window {target_hwnd}")
                return True
                
            except Exception as e:
                # Restore the previous window if something went wrong
                log_exception(f"Error switching to window {target_hwnd}", e)
                
                # Only attempt to restore if we have a valid previous window
                if previous_hwnd and win32gui.IsWindow(previous_hwnd):
                    try:
                        logger.debug("Attempting to restore previous window")
                        # Restore the previous HWND
                        self.hwnd = previous_hwnd
                        
                        # Clean up any existing thumbnail
                        if hasattr(self, '_cleanup_thumbnail'):
                            self._cleanup_thumbnail()
                        
                        # Re-register the thumbnail for the previous window
                        if hasattr(self, 'register_thumbnail'):
                            self.register_thumbnail()
                        
                        # Restore minimized state if needed
                        if was_minimized:
                            win32gui.ShowWindow(previous_hwnd, win32con.SW_MINIMIZE)
                        
                        logger.debug("Successfully restored previous window")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore previous window: {restore_error}")
                
                return False
            
        except Exception:
            log_exception(f"Unexpected error in _handle_swap_window for HWND {target_hwnd}")
            return False

    def _handle_show_settings(self):
        """Handle showing the settings dialog."""
        try:
            app = QApplication.instance()
            if app:
                if hasattr(app, '_settings_panel') and app._settings_panel:
                    app._settings_panel.show()
                    app._settings_panel.activateWindow()
                    app._settings_panel.raise_()
                elif hasattr(app, '_show_settings'):
                    app._show_settings()
                elif self.app_instance and hasattr(self.app_instance, 'show_settings'):
                    self.app_instance.show_settings()
                    if hasattr(self.app_instance, 'activateWindow'):
                        self.app_instance.activateWindow()  # Bring settings window to front
        except Exception:
            log_exception("Error in _handle_show_settings")
            raise
            
    def _handle_show_sub_settings(self):
        """Handle showing the sub-settings dialog."""
        try:
            if self.app_instance and hasattr(self.app_instance, 'show_sub_settings'):
                self.app_instance.show_sub_settings()
                if hasattr(self.app_instance, 'activateWindow'):
                    self.app_instance.activateWindow()  # Bring settings window to front
            elif self.app_instance and hasattr(self.app_instance, '_show_sub_settings'):
                # Fallback to old method name if it exists
                self.app_instance._show_sub_settings()
                if hasattr(self.app_instance, 'activateWindow'):
                    self.app_instance.activateWindow()
        except Exception:
            log_exception("Error in _handle_show_sub_settings")
            raise

    def _handle_reset_position(self):
        """Reset the overlay's position and size based on the saved preset for the current monitor."""
        logger.debug("Resetting overlay position and size")
        
        if not self.app_instance or not hasattr(self.app_instance, 'settings'):
            logger.warning("Cannot reset position: app_instance or settings not available")
            return
            
        # Get the current screen
        screen = QGuiApplication.screenAt(self.geometry().center())
        if not screen:
            screen = QGuiApplication.primaryScreen()
            
        # Get the monitor index
        screens = QGuiApplication.screens()
        monitor_idx = screens.index(screen) if screen in screens else 0
        
        # Load the saved preset for this monitor
        preset_key = f"MonitorPresets/Monitor_{monitor_idx}_Preset"
        position_preset = self.app_instance.settings.value(preset_key, "Center")  # Default to "Center" if not found
        
        # Calculate the new geometry based on the preset
        if hasattr(self.app_instance, 'calculate_position_geometry'):
            overlay_type = "monitor" if self.monitor_index is not None else "window"
            new_geometry = self.app_instance.calculate_position_geometry(
                screen, position_preset, overlay_type
            )
            
            if new_geometry and new_geometry.isValid():
                logger.info(f"Resetting overlay to {position_preset} position: {new_geometry}")
                self.setGeometry(new_geometry)
                return
        
        # Fallback to default behavior if preset calculation fails
        logger.warning("Failed to calculate position from preset, using default position")
        screen_geo = screen.availableGeometry()
        default_geometry = QRect(
            screen_geo.x() + screen_geo.width() // 4,
            screen_geo.y() + screen_geo.height() // 4,
            screen_geo.width() // 2,
            screen_geo.height() // 2
        )
        self.setGeometry(default_geometry)

    def _handle_quit_application(self):
        logger.debug("Quitting application via overlay context menu")
        try:
            if self.app_instance and hasattr(self.app_instance, 'cleanup_and_quit'):
                self.app_instance.cleanup_and_quit()
            elif self.app_instance and hasattr(self.app_instance, 'quit'):
                self.app_instance.quit()
            else:
                logger.warning("cleanup_and_quit method not found on app_instance. Falling back to QApplication.quit()")
                from PySide6.QtWidgets import QApplication
                QApplication.quit()
        except Exception as e:
            logger.error(f"Exception in _handle_quit_application: {e}")

            logger.info(f"Swapping window in overlay from {self.hwnd} to {new_hwnd}")
        
        # Check if the overlay is locked
        if getattr(self, '_is_window_locked', False):
            logger.info("Window swap prevented: Overlay is locked")
            return False
            
        try:
            # Store the original hwnd for cleanup
            original_hwnd = self.hwnd
            
            # Clean up existing thumbnail if there is one
            if hasattr(self, 'thumbnail') and self.thumbnail:
                try:
                    dwmapi.DwmUnregisterThumbnail(self.thumbnail)
                    logger.info(f"Unregistered thumbnail for HWND {original_hwnd}")
                except Exception as e:
                    logger.error(f"Error unregistering thumbnail: {e}")
                self.thumbnail = None
            
            # Update the window handle and create new thumbnail
            self.hwnd = new_hwnd
            
            # Register and update the new thumbnail\n\n            # Update the key_passthrough target window\n            if hasattr(self, 'key_passthrough') and self.key_passthrough is not None:\n                logger.debug(f"Updating key_passthrough target window to HWND {new_hwnd}")\n                self.key_passthrough.set_target_window(new_hwnd)
            if not self._register_thumbnail():
                logger.error(f"Failed to register thumbnail for new window {new_hwnd}")
                return False
                
            # Update window title and other metadata
            if hasattr(self, '_update_window_title'):
                self._update_window_title()
                
            # Update the UI to reflect the new window
            self.update()
            
            logger.info(f"BorderWidget.handle_swap_window succeeded for HWND: {new_hwnd}")
            return True
        except Exception as e:
            logger.exception(f"Error in handle_swap_window: {e}")
            return False
    def _add_swap_methods(self):
        """Explicitly add swap methods to this instance to ensure they're available at runtime.
        
        This is a diagnostic/fix function to address method resolution issues.
        """
        # Direct instance method assignment
        logger.info(f"Adding swap methods to BorderWidget instance {id(self)}")
        
        # Define the method locally to ensure proper binding to self
        def _instance_handle_swap_window(instance_self, new_hwnd):
            logger.info(f"Instance method handle_swap_window called for hwnd {new_hwnd}")
            return BorderWidget.handle_swap_window(instance_self, new_hwnd)
            
        # Bind the methods directly to this instance
        setattr(self, 'handle_swap_window', _instance_handle_swap_window.__get__(self, BorderWidget))
        setattr(self, '_handle_swap_window', _instance_handle_swap_window.__get__(self, BorderWidget))
        
        # Verify methods are now available
        logger.info(f"Method verification - handle_swap_window: {hasattr(self, 'handle_swap_window')}")
        logger.info(f"Method verification - _handle_swap_window: {hasattr(self, '_handle_swap_window')}")
        logger.info(f"BorderWidget methods: {[m for m in dir(self) if not m.startswith('__') and ('swap' in m or 'handle' in m)]}")
        
        # Return self to allow method chaining
        return self

    def moveEvent(self, event):
        super().moveEvent(event)
        # Update thumbnail position when window is moved
        if hasattr(self, 'thumbnail') and self.thumbnail:
            self.update_thumbnail()
        # Update focus indicator position
        if hasattr(self, '_focus_indicator') and self._focus_indicator.isVisible():
            self._focus_indicator.update_position(self.rect())
