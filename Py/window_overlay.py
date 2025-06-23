import ctypes
import ctypes.wintypes
import logging
import sys
import time
import win32gui
import win32con
import win32process
import win32api
from key_passthrough import KeyPassthrough
from ctypes import windll, Structure, c_ulong, c_ushort, c_long, byref, sizeof
from PySide6.QtCore import Qt, QPoint, QPointF, QTimer, QSize, QRect
from PySide6.QtGui import QPainter, QPen, QColor, QGuiApplication, QAction, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import QMainWindow, QMenu, QWidget, QVBoxLayout, QLabel, QCheckBox, QSizePolicy, QApplication, QMenuBar, QStyleFactory

# Import snap utilities
import snap_utils
from snap_utils import ensure_within_available_desktop, get_resize_edge_for_pos

# Import constants from constants module
from constants import (
    DEFAULT_WINDOW_OVERLAY_WIDTH, 
    DEFAULT_WINDOW_OVERLAY_HEIGHT,
    DEFAULT_POSITION_PRESET
)


# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)

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
    """A widget that shows a focus indicator in the bottom-right corner."""
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
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.NoFocus)
        
        # Store parent reference
        self._parent_ref = parent_ref
        
        # Indicator properties
        self._size = 20
        self._margin = 10
        self._opacity = 0.75
        self._size_ratio = 0.05
        
        # Set initial size
        self.resize(self._size, self._size)
        self.hide()
        
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
                # Convert to global coordinates if needed
                if not hasattr(self._parent_ref, 'mapToGlobal'):
                    frame_rect = QRect(
                        self._parent_ref.x(),
                        self._parent_ref.y(),
                        self._parent_ref.width(),
                        self._parent_ref.height()
                    )
            else:
                # Convert local rect to global coordinates
                frame_rect = QRect(
                    self._parent_ref.mapToGlobal(rect.topLeft()),
                    rect.size()
                )
            
            # Calculate size based on parent size
            min_dimension = min(frame_rect.width(), frame_rect.height())
            new_size = max(10, int(min_dimension * self._size_ratio))
            
            # Update size if changed
            if new_size != self._size:
                self._size = new_size
            
            # Calculate position and set geometry in one operation
            self.setGeometry(
                frame_rect.right() - self._size - self._margin,
                frame_rect.bottom() - self._size - self._margin,
                self._size,
                self._size
            )
        except Exception as e:
            print(f"Error updating focus indicator position: {e}")
    
    def paintEvent(self, event: QPaintEvent):
        """Draw the focus indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Draw a subtle shadow
        shadow_color = QColor(0, 0, 0, 100)
        painter.setPen(Qt.NoPen)
        painter.setBrush(shadow_color)
        painter.drawEllipse(1, 1, self._size-1, self._size-1)
        
        # Check if we should use red indicator (if either passthrough option is enabled)
        use_red = False
        if self._parent_ref and hasattr(self._parent_ref, 'key_passthrough'):
            key_passthrough = self._parent_ref.key_passthrough
            if hasattr(key_passthrough, 'is_enabled') and key_passthrough.is_enabled():
                use_red = True
                if hasattr(key_passthrough, 'is_aggressive_mode') and key_passthrough.is_aggressive_mode():
                    use_red = True
        
        # Draw the indicator with 20% more transparency (multiply alpha by 0.8)
        base_opacity = self._opacity * 0.8
        if use_red:
            indicator_color = QColor(255, 80, 80, int(255 * base_opacity))  # Red color
        else:
            indicator_color = QColor(255, 255, 255, int(255 * base_opacity))  # White color
            
        painter.setBrush(indicator_color)
        painter.drawEllipse(0, 0, self._size-1, self._size-1)
        
        painter.end()


class BorderWidget(QMainWindow):
    def __init__(self, hwnd=None, monitor_index=0, theme="auto", opacity=100, app_instance=None, initial_geometry=None):
        super().__init__()
        logger.info(f"Initializing overlay with hwnd={hwnd}, monitor={monitor_index}, opacity={opacity}")
        
        self.hwnd = hwnd
        self.monitor_index = monitor_index
        current_theme_str = theme.lower() if theme else "auto"
        self.theme = "dark" if current_theme_str == "auto" else current_theme_str
        self.opacity = opacity / 100.0
        self.app_instance = app_instance
        
        # Initialize key passthrough
        self.key_passthrough = KeyPassthrough()
        self.key_passthrough.set_target_window(self.hwnd) if self.hwnd else None
        
        # Connect to settings changes
        if app_instance:
            # Set initial state from settings
            initial_state = app_instance.settings.value("key_passthrough_enabled", True, type=bool)
            aggressive_mode = app_instance.settings.value("aggressive_key_passthrough", False, type=bool)
            
            # Ensure only one mode is active
            if aggressive_mode:
                self.key_passthrough.set_enabled(True)
                self.key_passthrough.set_aggressive_mode(True)
            else:
                self.key_passthrough.set_enabled(initial_state)
            
            # Connect to the key passthrough setting changed signal
            if hasattr(app_instance, 'key_passthrough_setting_changed'):
                app_instance.key_passthrough_setting_changed.connect(self._handle_key_passthrough_change)
            else:
                logger.warning("key_passthrough_setting_changed signal not found in app_instance")
        
        # Track desktop overlay state
        self.is_desktop_overlay = False
        self.minimized_windows = []  # Store handles of minimized windows
        self.last_click_time = 0  # For double-click detection
        self.double_click_interval = 300  # ms between clicks for double-click
        self._passed_initial_geometry = initial_geometry
        
        # Focus indicator - Create as a separate top-level widget
        self._focus_indicator = FocusIndicatorWidget(self)
        self._focus_indicator.hide()
        
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

        self.context_menu = None
        self.switch_to_window_menu = None
        self.switch_to_monitor_menu = None
        
        self._border_width = 2
        self._drag_pos_global = None
        self._drag_start_global = None
        self._initial_geometry = None
        self._drag_offset_to_window_topleft = None
        self._edge_margin = 10
        self._is_resizing = False
        self._resize_edge = None
        self._resize_timer = None
        
        # Initialize drag state
        self._drag_state = {
            'is_resizing': False,
            'resize_edge': None,
            'drag_start_global': None,
            'initial_geometry': None,
            'drag_offset': None
        }
        
        self._setup_window()
        self._init_context_menu()
        self.apply_theme(self.theme)
        
        # Enable key events for spacebar passthrough
        self.setFocusPolicy(Qt.StrongFocus)
        
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
                logger.error(f"Error saving key passthrough settings: {e}", exc_info=True)
            finally:
                # Always unblock signals in case of error
                settings.blockSignals(False)

    def keyPressEvent(self, event):
        """Handle key press events for key passthrough."""
        # Apply initial geometry before processing key events
        if not hasattr(self, '_initial_geometry_applied') and self._passed_initial_geometry:
            logger.debug(f"Applying passed initial_geometry: {self._passed_initial_geometry}")
            self.setGeometry(self._passed_initial_geometry)
            self._initial_geometry_applied = True
        
        # Handle spacebar or enter key press - forward to the captured window
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            key_name = "SPACE" if event.key() == Qt.Key_Space else "ENTER"
            logger.debug(f"Key press detected: {key_name}")
            
            if not self.hwnd:
                logger.warning(f"No target window handle available for key {key_name}")
                event.ignore()
                return
                
            if not win32gui.IsWindow(self.hwnd):
                logger.error(f"Invalid window handle: {self.hwnd} for key {key_name}")
                event.ignore()
                return
                
            try:
                # Mark the event as handled to prevent duplicate processing
                event.accept()
                logger.debug(f"Key {key_name} event accepted, preparing to forward...")
                
                # Get the virtual key code
                vk_code = win32con.VK_SPACE if event.key() == Qt.Key_Space else win32con.VK_RETURN
                logger.debug(f"Sending key {key_name} (VK: {vk_code}) to window {self.hwnd}")
                
                # Log key passthrough state
                logger.debug(f"Key passthrough enabled: {self.key_passthrough.is_enabled()}, "
                            f"Aggressive mode: {self.key_passthrough.is_aggressive_mode()}")
                
                # Use the key passthrough module to send the key
                result = self.key_passthrough.send_key(vk_code)
                logger.debug(f"Key {key_name} send result: {result}")
                
                if not result:
                    logger.warning(f"Failed to send key {key_name} to window {self.hwnd}")
                
                return  # Consume the event to prevent duplicate processing
                
            except Exception as e:
                logger.error(f"Error forwarding {key_name} to window {self.hwnd}: {e}", exc_info=True)
                event.ignore()
                return
        
        # For debugging other keys
        # logger.debug(f"Unhandled key press: {event.key()}")
        super().keyPressEvent(event)

    def _setup_window(self):
        # Initial window flags - start with basic flags
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
        
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        
        # Set minimum size first
        self.setMinimumSize(100, 75)
        
        # Apply initial geometry if provided, otherwise use default size and position
        if self._passed_initial_geometry:
            logger.debug(f"Applying initial geometry: {self._passed_initial_geometry}")
            self.setGeometry(self._passed_initial_geometry)
        else:
            # Default size from constants
            default_size = QSize(DEFAULT_WINDOW_OVERLAY_WIDTH, DEFAULT_WINDOW_OVERLAY_HEIGHT)
            logger.debug(f"Setting default window size: {default_size.width()}x{default_size.height()}")
            self.resize(default_size)
            
            # Position at top-left of the screen
            screen = QGuiApplication.primaryScreen()
            screen_geo = screen.availableGeometry()
            self.move(screen_geo.topLeft())
        
        # Apply opacity
        self.setWindowOpacity(self.opacity)
        
        # Ensure the window is shown with the new flags and geometry
        self.show()
        
        # Log the initial state for debugging
        logger.debug(f"Window created with click-through: {click_through_enabled}, geometry: {self.geometry()}")

    def _init_context_menu(self):
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

            main_window_action = QAction("Main Window", self)
            main_window_action.triggered.connect(self._handle_show_settings)
            self.context_menu.addAction(main_window_action)

            sub_settings_action = QAction("Subsettings", self)
            sub_settings_action.triggered.connect(self._handle_show_sub_settings)
            self.context_menu.addAction(sub_settings_action)
            
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
            logger.error(f"Error resetting window position: {e}", exc_info=True)

    def update_sort_order_and_refresh_menu(self, sort_order):
        self.window_sort_order = sort_order
        logger.debug(f"WindowOverlayWidget sort order updated to: {self.window_sort_order}")

    def _populate_switch_window_menu(self):
        try:
            if not self.switch_to_window_menu:
                logger.warning("_populate_switch_window_menu called but switch_to_window_menu is not initialized")
                return

            self.switch_to_window_menu.clear()
            logger.debug("Populating window list in window overlay context menu")

            if not self.app_instance:
                logger.error("Cannot populate window menu: app_instance is None")
                error_action = QAction("Error: App instance not available", self.switch_to_window_menu)
                error_action.setEnabled(False)
                self.switch_to_window_menu.addAction(error_action)
                return

            if not hasattr(self.app_instance, 'get_menu_ready_windows'):
                logger.error("Cannot populate window menu: get_menu_ready_windows method not found in app_instance")
                error_action = QAction("Error: Window listing unavailable", self.switch_to_window_menu)
                error_action.setEnabled(False)
                self.switch_to_window_menu.addAction(error_action)
                return

            # Get windows with filtering already applied in get_menu_ready_windows()
            windows_data = self.app_instance.get_menu_ready_windows()
            logger.debug(f"Found {len(windows_data)} windows for context menu")

            if not windows_data:
                no_windows_action = QAction("No other windows found", self.switch_to_window_menu)
                no_windows_action.setEnabled(False)
                self.switch_to_window_menu.addAction(no_windows_action)
                return

            # Apply sorting if needed
            if hasattr(self.app_instance, 'window_sort_order'):
                if self.app_instance.window_sort_order == "Alphabetical":
                    windows_data.sort(key=lambda item: item[1].lower() if item[1] else "")

            for hwnd, title, icon in windows_data:
                display_title = title.strip() if len(title.strip()) < 60 else title[:57] + "..."
                if not display_title:
                    class_name = win32gui.GetClassName(hwnd) or "Untitled"
                    display_title = f"[No Title] ({class_name})"

                action = QAction(display_title, self.switch_to_window_menu)
                if icon and not icon.isNull():
                    action.setIcon(icon)
                action.setData(hwnd)
                # Use a helper function to avoid lambda closure issues
                def create_trigger(hwnd):
                    return lambda checked: self._handle_swap_window(hwnd)
                action.triggered.connect(create_trigger(hwnd))
                self.switch_to_window_menu.addAction(action)

        except Exception as e:
            logger.error(f"Error populating switch window menu: {e}", exc_info=True)
            error_action = QAction("Error loading windows", self.switch_to_window_menu)
            error_action.setEnabled(False)
            self.switch_to_window_menu.addAction(error_action)

    def _populate_switch_monitor_menu(self):
        if not self.switch_to_monitor_menu:
            logger.warning("_populate_switch_monitor_menu called but switch_to_monitor_menu is not initialized")
            return

        self.switch_to_monitor_menu.clear()
        available_screens = QGuiApplication.screens()
        if not available_screens:
            no_screens_action = QAction("No screens available", self.switch_to_monitor_menu)
            no_screens_action.setEnabled(False)
            self.switch_to_monitor_menu.addAction(no_screens_action)
        else:
            for idx, screen_obj in enumerate(available_screens):
                action_text = screen_obj.name() or f"Screen {idx + 1}"
                manufacturer = screen_obj.manufacturer()
                model = screen_obj.model()
                if manufacturer or model:
                    action_text += f" ({manufacturer} {model})".strip()
                monitor_action = QAction(action_text, self.switch_to_monitor_menu)
                monitor_action.triggered.connect(lambda checked=False, s=screen_obj: self._handle_initiate_monitor_swap(s))
                self.switch_to_monitor_menu.addAction(monitor_action)

    def show_context_menu(self, pos):
        if self.switch_to_window_action and self.hwnd:
            self._populate_switch_window_menu()
            self.switch_to_window_action.setVisible(True)
        else:
            self.switch_to_window_action.setVisible(False)

        if self.switch_to_monitor_action:
            self._populate_switch_monitor_menu()
            self.switch_to_monitor_action.setVisible(True)

        if self.context_menu:
            self.context_menu.exec(self.mapToGlobal(pos))
        else:
            logger.error("Context menu not initialized before showing in BorderWidget.")

    def register_thumbnail(self):
        if not self.hwnd:
            logger.warning("No target hwnd for DWM thumbnail registration.")
            return False
            
        self._cleanup_thumbnail()
        
        try:
            self.hwnd_self = int(self.winId())
            logger.debug(f"Overlay window HWND: {self.hwnd_self}")
            
            self.thumbnail = ctypes.wintypes.HANDLE()
            result = dwmapi.DwmRegisterThumbnail(self.hwnd_self, self.hwnd, ctypes.byref(self.thumbnail))
            
            if result != 0:
                logger.error(f"DwmRegisterThumbnail failed with error: {result}")
                self.thumbnail = None
                return False
                
            logger.info(f"DWM thumbnail registered (Handle: {self.thumbnail.value}) for HWND {self.hwnd} on self HWND {self.hwnd_self}")
            self.source_size = QSize(self._query_thumbnail_source_size().cx, self._query_thumbnail_source_size().cy)
            self.update_thumbnail()
            return True
            
        except Exception as e:
            logger.error(f"Exception during thumbnail registration: {e}")
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
                logger.error(f"Exception during DWM thumbnail cleanup: {e}")
            finally:
                self.thumbnail = None
                self.source_size = None
                self.thumbnail_rect = None

    def update_thumbnail(self):
        if not self.thumbnail or not self.thumbnail.value:
            logger.debug("update_thumbnail: No valid DWM thumbnail.")
            return False
        current_source_dims_physical = self._query_thumbnail_source_size()
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
        display_rect_qrect_logical = thumbnail_fit_in_content_area.translated(bw_logical + 1, bw_logical + 1)
        self.thumbnail_rect = display_rect_qrect_logical

        phys_dest_left = int(round(display_rect_qrect_logical.left() * dpr))
        phys_dest_top = int(round(display_rect_qrect_logical.top() * dpr))
        phys_dest_right = int(round((display_rect_qrect_logical.left() + display_rect_qrect_logical.width()) * dpr))
        phys_dest_bottom = int(round((display_rect_qrect_logical.top() + display_rect_qrect_logical.height()) * dpr))
        dest_rect_dwm = RECT(phys_dest_left, phys_dest_top, phys_dest_right, phys_dest_bottom)

        props = DWM_THUMBNAIL_PROPERTIES()
        props.dwFlags = (0x00000001 | 0x00000002 | 0x00000004 | 0x00000008 | 0x00000010)
        props.rcDestination = dest_rect_dwm
        props.rcSource = RECT(0, 0, self.source_size.width(), self.source_size.height())
        props.opacity = ctypes.c_byte(255)
        props.fVisible = True
        props.fSourceClientAreaOnly = False
        hr = dwmapi.DwmUpdateThumbnailProperties(self.thumbnail, ctypes.byref(props))
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
        if target_w <= 0 or target_h <= 0: return QRect(target_qrect.topLeft(), QSize(0, 0))
        source_ar, target_ar = source_w / source_h, target_w / target_h
        if source_ar > target_ar:
            render_w, render_h = target_w, int(target_w / source_ar)
        else:
            render_h, render_w = target_h, int(target_h * source_ar)
        pos_x = target_qrect.left() + (target_w - render_w) // 2
        pos_y = target_qrect.top() + (target_h - render_h) // 2
        return QRect(pos_x, pos_y, render_w, render_h)

    def get_theme_colors(self):
        theme_map = {
            "dark": {"border": QColor(40,40,40), "background_rgb": "(30,30,30)", "text": QColor(240,240,240), "accent": QColor(0,120,215), "fill": QColor(20,20,20)},
            "light": {"border": QColor(180,180,180), "background_rgb": "(240,240,240)", "text": QColor(30,30,30), "accent": QColor(0,102,204), "fill": QColor(200,200,200)}
        }
        return theme_map.get(self.theme.lower(), theme_map["dark"])

    def apply_theme(self, theme=None, from_global=False):
        """Apply the specified theme to the widget.
        
        Args:
            theme (str, optional): Name of the theme to apply. If None, uses current theme.
            from_global (bool): Whether this is being called from a global theme change.
        """
        if theme is not None: 
            self.theme = theme.lower()
            
        # Get the appropriate styles for the theme
        colors = self.get_theme_colors()
        text_rgb = f"{colors['text'].red()},{colors['text'].green()},{colors['text'].blue()}"
        border_rgb = f"{colors['border'].red()},{colors['border'].green()},{colors['border'].blue()}"
        accent_rgb = f"{colors['accent'].red()},{colors['accent'].green()},{colors['accent'].blue()}"
        
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
        self.update()

    def focusInEvent(self, event):
        """Handle focus in event to show the focus indicator."""
        super().focusInEvent(event)
        if hasattr(self, '_focus_indicator') and self._focus_indicator:
            self._focus_indicator.show()
            self._update_focus_indicator_position()
            
    def focusOutEvent(self, event):
        """Handle focus out event to hide the focus indicator."""
        super().focusOutEvent(event)
        if hasattr(self, '_focus_indicator') and self._focus_indicator:
            self._focus_indicator.hide()
            
    def _update_focus_indicator_position(self):
        """Update the focus indicator's position."""
        if hasattr(self, '_focus_indicator') and self._focus_indicator:
            self._focus_indicator.update_position()
    
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
        # Ensure position is updated when window becomes visible
        QTimer.singleShot(50, self._update_focus_indicator_position)
        
    def closeEvent(self, event):
        """Clean up focus indicator when closing."""
        if hasattr(self, '_focus_indicator') and self._focus_indicator:
            self._focus_indicator.close()
        if hasattr(self, '_position_update_timer'):
            self._position_update_timer.stop()
        super().closeEvent(event)
        
    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        self._schedule_position_update()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw the main content
        painter.setRenderHint(QPainter.Antialiasing)
        colors = self.get_theme_colors()
        
        # Draw background
        painter.fillRect(self.rect(), colors['fill'])
        
        # Draw borders
        bw_logical = self._border_width
        if self.theme.lower() == "dark":
            border_color = QColor(Qt.white)
        else:
            border_color = QColor(Qt.black)
        painter.setPen(Qt.NoPen)
        painter.setBrush(border_color)

        overlay_w = self.rect().width()
        overlay_h = self.rect().height()

        painter.drawRect(0, 0, overlay_w, bw_logical)
        painter.drawRect(0, overlay_h - bw_logical, overlay_w, bw_logical)
        painter.drawRect(0, bw_logical, bw_logical, overlay_h - 2 * bw_logical)
        painter.drawRect(overlay_w - bw_logical, bw_logical, bw_logical, overlay_h - 2 * bw_logical)
        
        painter.end()
        
        # Update focus indicator position
        if self._focus_indicator.isVisible():
            self._focus_indicator.update_position(self.rect())

    def cleanup(self):
        """Clean up resources used by the BorderWidget."""
        try:
            logger.debug("BorderWidget cleanup called")
            
            # Clean up focus indicator
            if hasattr(self, '_focus_indicator'):
                try:
                    self._focus_indicator.hide()
                    self._focus_indicator.deleteLater()
                except Exception as e:
                    logger.warning(f"Error cleaning up focus indicator: {e}")
            
            # Clean up thumbnail if it exists
            if hasattr(self, 'thumbnail') and self.thumbnail and self.thumbnail.value:
                try:
                    self._cleanup_thumbnail()
                except Exception as e:
                    logger.warning(f"Error cleaning up thumbnail: {e}")
            
            # Clean up window if it exists and is still valid
            if hasattr(self, 'hwnd_self') and self.hwnd_self:
                try:
                    # Check if window still exists and is valid
                    if win32gui.IsWindow(self.hwnd_self):
                        win32gui.DestroyWindow(self.hwnd_self)
                        logger.debug("Successfully destroyed window handle")
                    else:
                        logger.debug("Window handle was already destroyed")
                except Exception as e:
                    logger.warning(f"Error destroying window handle: {e}")
                finally:
                    # Ensure we clear the handle to prevent double-free
                    self.hwnd_self = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error during BorderWidget cleanup: {e}", exc_info=True)
            return False
            
    def __del__(self):
        """Destructor to ensure all resources are properly cleaned up."""
        try:
            logger.debug("BorderWidget __del__ called")
            # Cleanup will be handled by closeEvent and Qt's parent-child hierarchy
            pass
        except Exception as e:
            # Prevent exceptions in __del__ from being raised
            try:
                logger.error(f"Error during BorderWidget destructor: {e}", exc_info=True)
            except:
                pass
    
    def closeEvent(self, event):
        """Clean up resources when the window is closed."""
        logger.debug(f"BorderWidget closeEvent: cleaning up resources for hwnd={self.hwnd}")
        
        try:
            # Clean up any pending resize timers
            if hasattr(self, '_resize_timer') and self._resize_timer:
                try:
                    self._resize_timer.stop()
                    self._resize_timer.deleteLater()
                except Exception as e:
                    logger.error(f"Error stopping resize timer: {e}")
                finally:
                    self._resize_timer = None
            
            # Clean up DWM thumbnail
            if hasattr(self, 'thumbnail') and self.thumbnail:
                try:
                    self._cleanup_thumbnail()
                except Exception as e:
                    logger.error(f"Error cleaning up thumbnail: {e}", exc_info=True)
            
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
                        logger.error(f"Error cleaning up {name}: {e}", exc_info=True)
                    finally:
                        setattr(self, name, None)
            
            # Clean up focus indicator
            if hasattr(self, '_focus_indicator') and self._focus_indicator:
                try:
                    self._focus_indicator.hide()
                    self._focus_indicator.setParent(None)
                    self._focus_indicator.deleteLater()
                    logger.debug("Cleaned up focus indicator")
                except RuntimeError as e:
                    if 'wrapped C/C++ object' not in str(e):
                        logger.error(f"Error cleaning up focus indicator: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error cleaning up focus indicator: {e}", exc_info=True)
                finally:
                    if hasattr(self, '_focus_indicator'):
                        del self._focus_indicator
            
            # Clean up position update timer
            if hasattr(self, '_position_update_timer'):
                try:
                    self._position_update_timer.stop()
                    self._position_update_timer.deleteLater()
                    logger.debug("Cleaned up position update timer")
                except Exception as e:
                    logger.error(f"Error cleaning up position update timer: {e}", exc_info=True)
            
            # Ensure thread input is detached
            if hasattr(self, '_restore_original_focus'):
                try:
                    self._restore_original_focus()
                except Exception as e:
                    logger.error(f"Error restoring original focus: {e}", exc_info=True)
            
            # Clean up key passthrough
            if hasattr(self, 'key_passthrough'):
                try:
                    self.key_passthrough = None
                except Exception as e:
                    logger.error(f"Error cleaning up key_passthrough: {e}", exc_info=True)
            
            # Clear references
            self.hwnd = None
            if hasattr(self, 'app_instance'):
                self.app_instance = None
            
            logger.debug("Close event handling complete")
            
        except Exception as e:
            logger.error(f"Error in closeEvent: {e}", exc_info=True)
        finally:
            # Always call the parent's closeEvent
            try:
                super().closeEvent(event)
            except Exception as e:
                logger.error(f"Error in parent closeEvent: {e}", exc_info=True)
    
    def keyPressEvent(self, event):
        """Handle key press events for the overlay.
        
        This method captures key presses and forwards them to the target window
        without stealing focus from it.
        """
        key = event.key()
        
        # Handle key passthrough for media and navigation keys
        if hasattr(self, 'key_passthrough') and self.key_passthrough and hasattr(self.key_passthrough, 'is_enabled') and self.key_passthrough.is_enabled():
            try:
                # Handle media keys
                if key == Qt.Key.Key_MediaPlay:
                    if hasattr(self.key_passthrough, 'send_media_play_pause'):
                        self.key_passthrough.send_media_play_pause()
                        event.accept()
                        return
                elif key == Qt.Key.Key_MediaNext:
                    if hasattr(self.key_passthrough, 'send_media_next_track'):
                        self.key_passthrough.send_media_next_track()
                        event.accept()
                        return
                elif key == Qt.Key.Key_MediaPrevious:
                    if hasattr(self.key_passthrough, 'send_media_previous_track'):
                        self.key_passthrough.send_media_previous_track()
                        event.accept()
                        return
                
                # Handle spacebar and enter key
                if key in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    # Clear focus from any focused widget in the overlay
                    focused = self.focusWidget()
                    if focused:
                        focused.clearFocus()
                    
                    # Send the key to the target window
                    if key == Qt.Key.Key_Space and hasattr(self.key_passthrough, 'send_space'):
                        self.key_passthrough.send_space()
                        event.accept()
                        return
                    elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and hasattr(self.key_passthrough, 'send_enter'):
                        self.key_passthrough.send_enter()
                        event.accept()
                        return
            except Exception as e:
                logger.error(f"Error in key event handling: {e}")
        
        # Let the base class handle other keys or if passthrough is disabled
        super().keyPressEvent(event)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        
        # Ensure window stays within virtual desktop bounds after resize
        try:
            current_pos = self.pos()
            # Only try to ensure bounds if we have the function available
            if 'ensure_within_available_desktop' in globals():
                new_pos = ensure_within_available_desktop(current_pos, self.size())
                if new_pos != current_pos:
                    self.move(new_pos)
        except Exception as e:
            logger.error(f"Error ensuring window bounds after resize: {e}")
        
        # Set up or reset the resize timer
        if not hasattr(self, '_resize_timer') or not self._resize_timer:
            self._resize_timer = QTimer(self)
            self._resize_timer.timeout.connect(self._handle_resize_timeout)
            self._resize_timer.setSingleShot(True)
        else:
            self._resize_timer.stop()
        self._resize_timer.start(50)

    def _handle_resize_timeout(self):
        if self.thumbnail and self.thumbnail.value:
            logger.debug("Resize timeout: Updating thumbnail.")
            self.update_thumbnail()

    def _send_keep_alive_signal(self, hwnd):
        """Send a keep-alive signal to the window by simulating a maximize/restore."""
        try:
            if not hwnd or not win32gui.IsWindow(hwnd):
                return False
                
            # Get current window state
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            was_maximized = bool(style & win32con.WS_MAXIMIZE)
            
            # Send a maximize/restore message to keep the window active
            if was_maximized:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                
            logger.debug(f"Sent keep-alive signal to window {hwnd}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to send keep-alive signal to window {hwnd}: {e}")
            return False
    
    def _is_media_application(self, hwnd):
        """Check if the window is a media application."""
        try:
            class_name = win32gui.GetClassName(hwnd)
            media_classes = ['WMPlayerApp', 'VLC', 'WMP', 'PotPlayer', 'MPC-HC', 'MPC-BE',
                           'ApplicationFrameWindow', 'WMPlayerApp', 'VLC', 'WMPApp', 'PotPlayer64',
                           'Qt5QWindowIcon', 'Qt5152QWindowIcon', 'Qt6QWindowIcon', 'QWidget']
            return any(media in class_name for media in media_classes)
        except Exception as e:
            logger.debug(f"Error checking media application: {e}")
            return False
            
    def _is_browser_window(self, hwnd):
        """Check if the window belongs to a browser."""
        try:
            class_name = win32gui.GetClassName(hwnd)
            window_text = win32gui.GetWindowText(hwnd)
            
            browser_indicators = [
                'Mozilla', 'Chrome', 'Edge', 'Safari', 'Opera',
                'Firefox', 'Chromium', 'Brave'
            ]
            
            return any(indicator in class_name or indicator in window_text 
                      for indicator in browser_indicators)
        except:
            return False
    
    def test_simple_key(self, hwnd):
        """Simple test - just send spacebar directly"""
        try:
            logger.setLevel(logging.DEBUG)  # Set debug level
            logger.debug(f"Sending spacebar to window {hwnd}")
            win32api.PostMessage(hwnd, win32con.WM_KEYDOWN, win32con.VK_SPACE, 0)
            win32api.PostMessage(hwnd, win32con.WM_KEYUP, win32con.VK_SPACE, 0)
            logger.debug(f"Sent spacebar to {hwnd}")
            return True
        except Exception as e:
            logger.error(f"Failed to send spacebar to {hwnd}: {e}")
            return False
        """Check if the window belongs to a browser."""
        try:
            class_name = win32gui.GetClassName(hwnd)
            window_text = win32gui.GetWindowText(hwnd)
            
            browser_indicators = [
                'Mozilla', 'Chrome', 'Edge', 'Safari', 'Opera',
                'Firefox', 'Chromium', 'Brave'
            ]
            
            return any(indicator in class_name or indicator in window_text 
                      for indicator in browser_indicators)
        except:
            return False

    def _is_media_playing_in_browser(self, hwnd):
        """Check if the window appears to be a browser with media playing."""
        # For now, assume if it's a browser window, media might be playing
        # since we can't reliably detect media state from window properties
        return self._is_browser_window(hwnd)
        
    def _activate_window_silently(self, hwnd):
        """
        Activate a window without bringing it to the foreground.
        
        Args:
            hwnd: The window handle to activate
            
        Returns:
            bool: True if activation was successful, False otherwise
        """
        if not hwnd or not win32gui.IsWindow(hwnd):
            logger.warning(f"Invalid window handle: {hwnd}")
            return False
            
        try:
            # Save the current foreground window if we haven't already
            if not hasattr(self, '_original_foreground') or not self._original_foreground:
                self._original_foreground = win32gui.GetForegroundWindow()
            
            # Get thread IDs
            current_thread = win32api.GetCurrentThreadId()
            target_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
            
            # Method 1: Try to activate the window directly first
            try:
                win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
            except Exception as e:
                logger.debug(f"WM_ACTIVATE failed for window {hwnd}: {e}")
            
            # Method 2: Attach to the window's thread to set focus
            attached = False
            if target_thread != current_thread:
                try:
                    if win32process.AttachThreadInput(current_thread, target_thread, True):
                        attached = True
                        logger.debug(f"Attached to thread for window {hwnd}")
                    else:
                        logger.warning(f"Failed to attach to thread for window {hwnd}")
                except Exception as e:
                    logger.error(f"Error attaching to thread for window {hwnd}: {e}")
            
            try:
                # Set focus to the window
                if win32gui.SetFocus(hwnd):
                    logger.debug(f"Set focus to window {hwnd}")
                else:
                    logger.warning(f"Failed to set focus to window {hwnd}")
                
                # Method 3: Send WM_SETFOCUS message
                try:
                    win32gui.SendMessage(hwnd, win32con.WM_SETFOCUS, 0, 0)
                except Exception as e:
                    logger.debug(f"WM_SETFOCUS failed for window {hwnd}: {e}")
                
                return True
                
            finally:
                # Always detach if we attached
                if attached:
                    try:
                        if not win32process.AttachThreadInput(current_thread, target_thread, False):
                            logger.warning(f"Failed to detach from thread for window {hwnd}")
                    except Exception as e:
                        logger.error(f"Error detaching from thread for window {hwnd}: {e}")
            
        except Exception as e:
            logger.error(f"Error in _activate_window_silently for window {hwnd}: {e}", exc_info=True)
            return False

    def _restore_original_focus(self):
        """
        Restore the original window focus and clean up thread attachments.
        This ensures we don't leave any thread input attached.
        """
        try:
            # Get current thread and foreground window
            current_thread = win32api.GetCurrentThreadId()
            foreground_hwnd = win32gui.GetForegroundWindow()
            
            # If we have an original foreground window and it's still valid
            if hasattr(self, '_original_foreground') and self._original_foreground:
                try:
                    # Get the thread that owns the original foreground window
                    original_thread = win32process.GetWindowThreadProcessId(self._original_foreground)[0]
                    
                    # If we're attached to this thread, detach
                    if win32process.GetCurrentThreadId() != original_thread:
                        if win32process.AttachThreadInput(win32process.GetCurrentThreadId(), original_thread, False):
                            logger.debug(f"Detached thread input from window {self._original_foreground}")
                        else:
                            logger.warning(f"Failed to detach thread input from window {self._original_foreground}")
                except Exception as e:
                    logger.debug(f"Error detaching thread input: {e}")
                
                # Clear the reference
                self._original_foreground = None
            
            # Extra safety: If we're still attached to the current foreground window, detach
            if foreground_hwnd:
                try:
                    foreground_thread = win32process.GetWindowThreadProcessId(foreground_hwnd)[0]
                    if win32process.AttachThreadInput(current_thread, foreground_thread, False):
                        logger.debug("Detached thread input from current foreground window")
                except Exception as e:
                    logger.debug(f"Error in thread cleanup: {e}")
                    
        except Exception as e:
            logger.error(f"Error in _restore_original_focus: {e}", exc_info=True)
        finally:
            # Ensure we clear any thread attachments even if an error occurred
            if hasattr(self, '_original_foreground'):
                self._original_foreground = None

    def _get_focused_window(self):
        """Get the window that currently has keyboard focus."""
        try:
            # Get the foreground window
            foreground_hwnd = win32gui.GetForegroundWindow()
            if not foreground_hwnd:
                return None
                
            # Get the thread that owns the foreground window
            foreground_thread = win32process.GetWindowThreadProcessId(foreground_hwnd)[1]
            current_thread = win32api.GetCurrentThreadId()
            
            # Attach to the foreground thread to get its focused window
            if foreground_thread != current_thread:
                win32process.AttachThreadInput(current_thread, foreground_thread, True)
                try:
                    focused_hwnd = win32gui.GetFocus()
                finally:
                    win32process.AttachThreadInput(current_thread, foreground_thread, False)
            else:
                focused_hwnd = win32gui.GetFocus()
                
            return focused_hwnd if focused_hwnd else foreground_hwnd
            
        except Exception as e:
            logger.debug(f"Could not get focused window: {e}")
            return win32gui.GetForegroundWindow()
            
    def _identify_application(self, hwnd):
        """Simple app identification for spacebar handling."""
        try:
            class_name = win32gui.GetClassName(hwnd)
            exe_name = self._get_process_name(hwnd).lower()
            
            # Firefox
            if 'mozilla' in class_name.lower() or 'firefox' in exe_name:
                return 'firefox'
                
            # Chrome/Edge
            if any(browser in class_name.lower() for browser in ['chrome', 'edge']):
                return 'chromium'
                
            # Media players (Spotify, MPV, etc.)
            if any(player in exe_name for player in ['spotify', 'mpv', 'vlc']):
                return 'media_player'
                
            # Everything else (games, Steam, etc.)
            return 'other'
            
        except Exception as e:
            logger.debug(f"Failed to identify application: {e}")
            return 'other'
            
    def _get_process_name(self, hwnd):
        """Get the process name for a window."""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION, False, pid)
            exe_path = win32process.GetModuleFileNameEx(handle, 0)
            win32api.CloseHandle(handle)
            return exe_path.split('\\')[-1]
        except:
            return ""
            
    def _get_main_window(self, hwnd):
        """Get the main window (root parent) of a given window."""
        current = hwnd
        while True:
            parent = win32gui.GetParent(current)
            if not parent:
                break
            current = parent
        return current
        
    def _send_raw_key(self, hwnd, vk_code):
        """Send raw key press to window."""
        try:
            # Debug logging
            window_text = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            logger.debug(f"Sending key {vk_code} to: {window_text} ({class_name})")
            
            # Send the key messages
            win32api.SendMessage(hwnd, win32con.WM_KEYDOWN, vk_code, 0)
            win32api.SendMessage(hwnd, win32con.WM_KEYUP, vk_code, 0)
            
            # For Enter key, also send WM_CHAR
            if vk_code == win32con.VK_RETURN:
                win32api.SendMessage(hwnd, win32con.WM_CHAR, 13, 0)  # ASCII 13 = Enter
            # For Space key, also send WM_CHAR  
            elif vk_code == win32con.VK_SPACE:
                win32api.SendMessage(hwnd, win32con.WM_CHAR, 32, 0)  # ASCII 32 = Space
                
            # For games and apps that might need child window targeting
            if self._identify_application(hwnd) == 'other':
                self._send_to_child_windows(hwnd, vk_code)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to send raw key {vk_code} to hwnd {hwnd}: {e}")
            return False
            
    def _send_media_key(self, hwnd, vk_code):
        """Send media key using multiple methods for better compatibility."""
        success = False
        
        # Method 1: WM_APPCOMMAND to the main window
        try:
            main_hwnd = self._get_main_window(hwnd)
            if vk_code == win32con.VK_SPACE:  # Spacebar for play/pause
                win32api.SendMessage(main_hwnd, win32con.WM_APPCOMMAND, main_hwnd, 
                                   (win32con.APPCOMMAND_MEDIA_PLAY_PAUSE << 16))
                success = True
        except Exception as e:
            logger.debug(f"WM_APPCOMMAND failed: {e}")
        
        # Method 2: Direct key messages to focused window
        try:
            self._send_key_messages(hwnd, vk_code)
            success = True
        except Exception as e:
            logger.debug(f"Direct key messages failed: {e}")
        
        # Method 3: For browsers, try to find and target the video element
        if self._is_browser_window(hwnd):
            try:
                self._send_browser_media_key(hwnd, vk_code)
                success = True
            except Exception as e:
                logger.debug(f"Browser media key failed: {e}")
        
        return success
        
    def _send_regular_key(self, hwnd, vk_code):
        """Send regular key press."""
        try:
            # Always send to the focused window first
            self._send_key_messages(hwnd, vk_code)
            
            # For browsers, also try to send to content areas
            if self._is_browser_window(hwnd):
                self._send_browser_media_key(hwnd, vk_code)
                
            return True
            
        except Exception as e:
            logger.debug(f"Regular key send failed: {e}")
            return False

    def _send_key_to_window(self, hwnd, vk_code, is_media_key=False):
        """Send a key press to a window without bringing it to the foreground."""
        try:
            # Get the window that currently has keyboard focus
            focused_hwnd = self._get_focused_window()
            
            # If we have a focused window, use that instead of the passed hwnd
            target_hwnd = focused_hwnd if focused_hwnd else hwnd
            
            if is_media_key:
                return self._send_media_key(target_hwnd, vk_code)
            else:
                return self._send_regular_key(target_hwnd, vk_code)
                
        except Exception as e:
            logger.warning(f"Failed to send key to window {hwnd}: {e}")
            return False

    def keyPressEvent(self, event):
        """Handle all key press events for the overlay.
        
        This method processes key events and forwards them to the target window
        when appropriate, without stealing focus from it.
        """
        if not hasattr(self, 'key_passthrough') or not self.key_passthrough or not self.hwnd or not win32gui.IsWindow(self.hwnd):
            super().keyPressEvent(event)
            return
            
        key = event.key()
        handled = False
        
        try:
            # Handle media keys first
            if key == Qt.Key.Key_MediaPlay:
                if hasattr(self.key_passthrough, 'send_media_play_pause'):
                    self.key_passthrough.send_media_play_pause()
                    handled = True
            elif key == Qt.Key.Key_MediaNext:
                if hasattr(self.key_passthrough, 'send_media_next_track'):
                    self.key_passthrough.send_media_next_track()
                    handled = True
            elif key == Qt.Key.Key_MediaPrevious:
                if hasattr(self.key_passthrough, 'send_media_previous_track'):
                    self.key_passthrough.send_media_previous_track()
                    handled = True
            # Handle spacebar and enter key
            elif key in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
                # Clear focus from any focused widget in the overlay
                focused = self.focusWidget()
                if focused:
                    focused.clearFocus()
                
                # Send the key to the target window
                try:
                    if key == Qt.Key.Key_Space and hasattr(self.key_passthrough, 'send_space'):
                        self.key_passthrough.send_space()
                    elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and hasattr(self.key_passthrough, 'send_enter'):
                        self.key_passthrough.send_enter()
                    handled = True
                except Exception as e:
                    key_name = "spacebar" if key == Qt.Key.Key_Space else "Enter"
                    logger.warning(f"Error forwarding {key_name} to window {self.hwnd}: {e}", exc_info=True)
            
            # Mark the event as handled if we processed it
            if handled:
                event.accept()
                # Ensure we maintain focus after handling the key
                self.setFocus()
                return
                
        except Exception as e:
            logger.error(f"Error in keyPressEvent: {e}", exc_info=True)
        
        # Let the base class handle other keys or if passthrough is disabled
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            current_time = event.timestamp()
            
            # Check for double-click (within double_click_interval ms)
            if current_time - self.last_click_time < self.double_click_interval:
                if self.is_desktop_overlay:
                    self._handle_desktop_double_click()
                    event.accept()
                    return
            
            self.last_click_time = current_time
            
            # Check if we're starting a resize operation
            resize_edge = get_resize_edge_for_pos(event.pos(), self, margin=8)
            if resize_edge:
                self._drag_state.update({
                    'is_resizing': True,
                    'resize_edge': resize_edge,
                    'drag_start_global': event.globalPos(),
                    'initial_geometry': self.geometry()
                })
                event.accept()
                return
                
            # Otherwise, it's a drag operation
            self._drag_state.update({
                'is_resizing': False,
                'resize_edge': None,
                'drag_start_global': event.globalPos(),
                'initial_geometry': self.geometry(),
                'drag_offset': event.pos()
            })
            
            # Set focus to this widget on click to enable keyboard input
            self.setFocus()
            event.accept()
            
        super().mousePressEvent(event)

    def _handle_desktop_double_click(self):
        """Handle double-click on desktop overlay."""
        if not self.is_desktop_overlay:
            return
            
        import win32gui
        import win32con
        import win32api
        
        try:
            # Get primary monitor info
            primary_monitor = win32api.GetMonitorInfo(0)
            primary_rect = primary_monitor['Monitor']
            
            # If we have minimized windows, restore them
            if self.minimized_windows:
                logger.info("Restoring windows on primary monitor")
                for hwnd in self.minimized_windows:
                    try:
                        if win32gui.IsWindow(hwnd):
                            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                            win32gui.BringWindowToTop(hwnd)
                    except Exception as e:
                        logger.error(f"Error restoring window {hwnd}: {e}")
                self.minimized_windows = []
                return
            
            # Otherwise, minimize all windows on primary monitor
            logger.info("Minimizing windows on primary monitor")
            windows_to_minimize = []
            
            def enum_windows_callback(hwnd, _):
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                    
                if win32gui.IsIconic(hwnd):
                    return True
                    
                try:
                    # Get window rect
                    window_rect = win32gui.GetWindowRect(hwnd)
                    
                    # Check if window is on primary monitor using center point
                    center_x = (window_rect[0] + window_rect[2]) // 2
                    center_y = (window_rect[1] + window_rect[3]) // 2
                    
                    if (primary_rect[0] <= center_x <= primary_rect[2] and 
                        primary_rect[1] <= center_y <= primary_rect[3]):
                        windows_to_minimize.append(hwnd)
                except Exception as e:
                    logger.error(f"Error processing window {hwnd}: {e}")
                return True
            
            # Find all windows on primary monitor
            win32gui.EnumWindows(enum_windows_callback, None)
            
            # Minimize all found windows
            for hwnd in windows_to_minimize:
                try:
                    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                except Exception as e:
                    logger.error(f"Error minimizing window {hwnd}: {e}")
            
            # Store the list of minimized windows
            self.minimized_windows = windows_to_minimize
            
        except Exception as e:
            logger.error(f"Error in desktop double-click handler: {e}")
    
    def _get_windows_on_primary(self):
        """Get all non-minimized windows on the primary monitor."""
        import win32gui
        import win32api
        
        # Get the primary monitor info
        primary_monitor = win32api.GetMonitorInfo(0)
        primary_rect = primary_monitor['Monitor']
        windows = []
        
        def enum_windows_callback(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return True
                
            # Skip minimized windows
            if win32gui.IsIconic(hwnd):
                return True
                
            try:
                # Get window rect and check if it's on the primary monitor
                window_rect = win32gui.GetWindowRect(hwnd)
                if (window_rect[0] < primary_rect[2] and 
                    window_rect[2] > primary_rect[0] and
                    window_rect[1] < primary_rect[3] and 
                    window_rect[3] > primary_rect[1]):
                    windows.append(hwnd)
            except Exception as e:
                logger.error(f"Error processing window {hwnd}: {e}")
            return True
            
        win32gui.EnumWindows(enum_windows_callback, None)
        return windows

    def mouseMoveEvent(self, event):
        # First, handle cursor changes based on resize edge
        if not hasattr(self, '_drag_state') or not any([
            self._drag_state.get('is_resizing'),
            self._drag_state.get('drag_offset') is not None
        ]):
            edge = snap_utils.get_resize_edge_for_pos(event.pos(), self)
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
            handled = snap_utils.handle_overlay_mouse_move(event, self, self._drag_state)
            if handled:
                event.accept()
                return
        
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Use centralized mouse release handler
            if hasattr(self, '_drag_state'):
                if snap_utils.handle_overlay_mouse_release(event, self, self._drag_state):
                    # Handle any cleanup specific to window overlay
                    if self.thumbnail and self.thumbnail.value:
                        self._handle_resize_timeout()
                    event.accept()
                    return
            
            # Fallback to default behavior if not handled
            self.setCursor(Qt.ArrowCursor)
            
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        scale_factor = 1.1 if delta > 0 else (1 / 1.1)
        new_width = max(self.minimumSizeHint().width(), int(self.width() * scale_factor))
        new_height = max(self.minimumSizeHint().height(), int(self.height() * scale_factor))
        self.resize(new_width, new_height)
        self.ensure_in_monitor_bounds()
        event.accept()

    def quick_switch_overlay(self):
        # Handle desktop overlay case first
        if hasattr(self, 'is_desktop_overlay') and self.is_desktop_overlay:
            logger.info("Quick switch on desktop overlay - triggering desktop double-click action")
            self._handle_desktop_double_click()
            return
            
        if not self.hwnd:
            logger.info("Quick switch ignored: Overlay is not targeting a specific window (self.hwnd is None).")
            return
            
        hwnd_to_activate = self.hwnd
        overlay_hwnd = int(self.winId())
        
        current_foreground_hwnd = self.app_instance.last_external_focused_hwnd if self.app_instance else None
        swap_target_hwnd = None
        settings_class_names = ["QMainWindow", "SettingsPanel", "CrosshairPicker"]
        
        if current_foreground_hwnd and win32gui.IsWindow(current_foreground_hwnd):
            window_class = win32gui.GetClassName(current_foreground_hwnd)
            window_title = win32gui.GetWindowText(current_foreground_hwnd)
            if window_class and window_title and not any(class_name in window_class for class_name in settings_class_names) and "Shitty PiP" not in window_title and current_foreground_hwnd != overlay_hwnd:
                swap_target_hwnd = current_foreground_hwnd

        if not swap_target_hwnd and self.app_instance and hasattr(self.app_instance, 'settings'):
            last_saved_hwnd = self.app_instance.settings.value("LastUsedWindowHwnd", 0, int)
            if last_saved_hwnd and win32gui.IsWindow(last_saved_hwnd) and last_saved_hwnd != overlay_hwnd:
                window_class = win32gui.GetClassName(last_saved_hwnd)
                window_title = win32gui.GetWindowText(last_saved_hwnd)
                if window_class and window_title and not any(class_name in window_class for class_name in settings_class_names) and "Shitty PiP" not in window_title:
                    swap_target_hwnd = last_saved_hwnd

        if not swap_target_hwnd:
            current_active_hwnd = win32gui.GetForegroundWindow()
            if current_active_hwnd and win32gui.IsWindow(current_active_hwnd) and current_active_hwnd != overlay_hwnd:
                window_class = win32gui.GetClassName(current_active_hwnd)
                window_title = win32gui.GetWindowText(current_active_hwnd)
                if window_class and window_title and not any(class_name in window_class for class_name in settings_class_names) and "Shitty PiP" not in window_title:
                    swap_target_hwnd = current_active_hwnd

        logger.debug(f"[QUICK SWITCH] Overlay HWND: {overlay_hwnd}, Target HWND to activate: {hwnd_to_activate}, Swap target HWND: {swap_target_hwnd}")

        # Log specific reasons for potential failure
        if not self.app_instance:
            logger.warning("Quick switch aborted: No app instance")
            return
            
        if not hwnd_to_activate:
            logger.warning("Quick switch aborted: No target window to activate")
            return
            
        if not swap_target_hwnd:
            # Instead of failing, use the current foreground window as a fallback
            swap_target_hwnd = win32gui.GetForegroundWindow()
            if swap_target_hwnd and swap_target_hwnd != overlay_hwnd:
                logger.debug(f"Using current foreground window as fallback: {swap_target_hwnd}")
            else:
                logger.warning("Quick switch aborted: No valid window to swap with")
                return
                
        if hwnd_to_activate == swap_target_hwnd:
            logger.debug("Skipping quick switch - target and source windows are the same")
            return

        try:
            # Get window class and title for special handling
            window_class = win32gui.GetClassName(hwnd_to_activate).lower()
            window_title = win32gui.GetWindowText(hwnd_to_activate).lower()
            
            # Special handling for video players and browsers
            is_video_player = any(name in window_class or name in window_title 
                                for name in ['mpc', 'mpv', 'vlc', 'potplayer', 'kodi', 'plex', 'jellyfin'])
            is_browser = any(name in window_class 
                            for name in ['chrome', 'firefox', 'msedge', 'iexplore', 'safari'])
            
            # Restore window if minimized
            if win32gui.IsIconic(hwnd_to_activate):
                # For video players, first restore, then activate
                if is_video_player:
                    # First restore the window
                    win32gui.ShowWindow(hwnd_to_activate, win32con.SW_RESTORE)
                    # Small delay to let the window restore
                    time.sleep(0.1)
                    # Force a repaint
                    win32gui.RedrawWindow(hwnd_to_activate, None, None, 
                                        win32con.RDW_INVALIDATE | 
                                        win32con.RDW_ERASE | 
                                        win32con.RDW_ALLCHILDREN)
                else:
                    # Standard restore for other windows
                    win32gui.ShowWindow(hwnd_to_activate, win32con.SW_RESTORE)
                    
            # Ensure window is visible
            win32gui.ShowWindow(hwnd_to_activate, win32con.SW_SHOW)
            
            # For browsers, we need to be more aggressive with activation
            if is_browser:
                _simulate_alt_press()
                win32gui.SetForegroundWindow(hwnd_to_activate)
                # Second attempt after a small delay
                time.sleep(0.1)
                win32gui.SetForegroundWindow(hwnd_to_activate)
            else:
                _simulate_alt_press()
                win32gui.SetForegroundWindow(hwnd_to_activate)
            
            logger.info(f"Switching overlay to preview window: {swap_target_hwnd}")
            self._handle_swap_window(swap_target_hwnd)
            
            if self.app_instance and hasattr(self.app_instance, 'settings'):
                self.app_instance.settings.setValue("LastUsedWindowHwnd", swap_target_hwnd)
        except win32gui.error as e:
            logger.error(f"Error during quick switch (win32 call failed): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Generic error during quick switch: {e}", exc_info=True)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            logger.debug("Left double-click detected on BorderWidget.")
            # Handle desktop overlay case first
            if hasattr(self, 'is_desktop_overlay') and self.is_desktop_overlay:
                logger.debug("Left double-click on desktop overlay")
                self._handle_desktop_double_click()
            else:
                self.quick_switch_overlay()
        else:
            super().mouseDoubleClickEvent(event)

    def _get_resize_edge(self, pos):
        return snap_utils.get_resize_edge_for_pos(pos, self, self._edge_margin)

    def ensure_in_monitor_bounds(self, target_screen=None):
        screen_to_use = target_screen or self.screen() or QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        if not screen_to_use:
            logger.warning("ensure_in_monitor_bounds: Could not determine screen. Aborting bounds check.")
            return
            
        bounds = screen_to_use.availableGeometry()
        x, y = self.x(), self.y()
        w, h = self.width(), self.height()
        
        if x < bounds.left(): 
            x = bounds.left()
        if y < bounds.top(): 
            y = bounds.top()
        if x + w > bounds.right(): x = bounds.right() - w
        if y + h > bounds.bottom(): y = bounds.bottom() - h
        if x < bounds.left(): x = bounds.left()
        if y < bounds.top(): y = bounds.top()
        
        self.move(x, y)

    def _handle_reset_position(self):
        logger.debug("Resetting overlay size and position (Window Overlay)")
        current_screen = self.screen() or QGuiApplication.primaryScreen()
        if current_screen and self.app_instance:
            position_preset = self.app_instance.settings.value(f"MonitorPresets/{current_screen.name()}/position_preset", DEFAULT_POSITION_PRESET, type=str)
            reset_geometry = self.app_instance.calculate_position_geometry(current_screen, position_preset, "window")
            if reset_geometry:
                self.setGeometry(reset_geometry)
                logger.info(f"Reset window overlay to geometry {reset_geometry} on screen '{current_screen.name()}' using preset '{position_preset}'.")
            else:
                default_width = self.app_instance.settings.value("WindowOverlay/default_width", DEFAULT_WINDOW_OVERLAY_WIDTH, type=int)
                default_height = self.app_instance.settings.value("WindowOverlay/default_height", DEFAULT_WINDOW_OVERLAY_HEIGHT, type=int)
                self.move(current_screen.availableGeometry().topLeft())
                self.resize(default_width, default_height)
        elif not self.app_instance:
            logger.error("app_instance not available for reset logic. Falling back to basic reset.")
            self.move(0, 0)
        else:
            logger.error("Could not determine any screen to reset position to. Falling back to (0,0).")
            self.move(0, 0)
        self.ensure_in_monitor_bounds()

    def _handle_show_settings(self):
        logger.debug("Showing main settings panel")
        if self.app_instance and hasattr(self.app_instance, '_show_settings'):
            self.app_instance._show_settings()
        else:
            logger.warning("Could not find show_settings method on app_instance.")

    def _handle_initiate_monitor_swap(self, screen_object):
        """Handle switching from window overlay to monitor overlay for the selected screen.
        
        Args:
            screen_object: The QScreen object representing the monitor to switch to
        """
        try:
            logger.info(f"Initiating monitor swap to screen: {screen_object.name() if screen_object else 'None'}")
            
            if not screen_object:
                logger.error("No screen object provided for monitor swap")
                return
                
            if not self.app_instance:
                logger.error("Cannot switch to monitor: app_instance not available")
                return
                
            # Close the current window overlay
            self.close()
            
            # Request the app to create a new monitor overlay for the selected screen
            self.app_instance.prepare_to_create_monitor_overlay(screen_object)
            
        except Exception as e:
            logger.error(f"Error initiating monitor swap: {e}", exc_info=True)

    def _handle_swap_window(self, new_hwnd):
        logger.info(f"Attempting to swap overlay from HWND {self.hwnd} to {new_hwnd}")
        if new_hwnd == self.hwnd:
            logger.info("Selected window is already the current target. No change.")
            return
        
        # Store current geometry before making any changes
        current_geometry = self.geometry()
        current_screen = QGuiApplication.screenAt(current_geometry.center())
        
        if self.thumbnail and self.hwnd:
            self._cleanup_thumbnail()
        
        self.hwnd = new_hwnd
        self.monitor_index = None
        
        # Update the key passthrough target window before registering the thumbnail
        if hasattr(self, 'key_passthrough') and self.key_passthrough:
            self.key_passthrough.set_target_window(self.hwnd)
            logger.debug(f"Updated key_passthrough target window to HWND {self.hwnd}")
        
        if self.register_thumbnail():
            # Restore the original geometry
            self.setGeometry(current_geometry)
            
            # Ensure the window stays within the current screen bounds
            if current_screen:
                self.ensure_in_monitor_bounds(current_screen)
                
            self.update_thumbnail()
            logger.info(f"Successfully swapped overlay to HWND {new_hwnd}, maintaining position and size")
        else:
            logger.error(f"Failed to register thumbnail for HWND {new_hwnd}. Overlay may not update.")

    def _handle_initiate_monitor_swap(self, screen_object):
        logger.info(f"Initiating swap to monitor: {screen_object.name()}")
        if self.app_instance and hasattr(self.app_instance, 'prepare_to_create_monitor_overlay'):
            self.app_instance.prepare_to_create_monitor_overlay(screen_object)
        else:
            logger.error("app_instance or prepare_to_create_monitor_overlay method not found.")

    def _handle_show_sub_settings(self):
        logger.debug("Showing sub-settings dialog via overlay context menu")
        if self.app_instance and hasattr(self.app_instance, '_show_sub_settings'):
            self.app_instance._show_sub_settings()
        else:
            logger.warning("show_sub_settings method not found on app_instance.")

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
        if self.app_instance and hasattr(self.app_instance, 'cleanup_and_quit'):
            self.app_instance.cleanup_and_quit()
        else:
            logger.warning("cleanup_and_quit method not found on app_instance. Falling back to QApplication.quit()")
            from PySide6.QtWidgets import QApplication
            QApplication.quit()

    def moveEvent(self, event):
        super().moveEvent(event)
        # Update thumbnail position when window is moved
        if hasattr(self, 'thumbnail') and self.thumbnail:
            self.update_thumbnail()
        # Update focus indicator position
        if hasattr(self, '_focus_indicator') and self._focus_indicator.isVisible():
            self._focus_indicator.update_position(self.rect())