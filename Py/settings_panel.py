import sys
import logging
import os
import random
import ctypes
from pathlib import Path
from typing import Dict, Any, Optional
from PySide6.QtCore import QEvent, QPoint, QRect, QTimer, Qt, Signal, QSizeF, QObject, QFile, QTextStream
from PySide6.QtGui import QCursor, QFont, QMouseEvent, QPaintEvent, QPainter, QPen, QPixmap, QScreen, QAction, QColor, QGuiApplication

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QWidget, QFrame, QMessageBox,
                             QScrollArea, QSizePolicy, QDialog, QFileDialog, QListView,
                             QGridLayout, QStyle, QStyleOption)

# Resource paths using QRC
# All resources are accessed via the QRC system (e.g., ":/Resources/filename.png")
# THEMES_DIR and RESOURCES_DIR are kept for compatibility with existing code
THEMES_DIR = ":/themes"
RESOURCES_DIR = ":/Resources"

# Import snap utilities
import snap_utils
from snap_utils import (
    handle_overlay_mouse_press,
    handle_overlay_mouse_move,
    handle_overlay_mouse_release,
    apply_snap,
    get_resize_edge_for_pos,
    debug_monitor_setup,
    get_physical_monitor_info,
    get_physical_monitor_for_screen,
    get_screen_scale_factor
)

logger = logging.getLogger(__name__)

# Windows DPI API constants
MDT_EFFECTIVE_DPI = 0
MDT_ANGULAR_DPI = 1  
MDT_RAW_DPI = 2

class BorderOverlay(QWidget):
    """A transparent overlay that draws a border on top of all other widgets."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # Allow clicks to pass through
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        
    def paintEvent(self, event):
        """Draw a border matching the main window's style."""
        # Only draw if we have a parent that's a SettingsPanel
        parent = self.parent()
        if not hasattr(parent, 'theme'):
            return
            
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Get the border color from the parent's theme
            is_dark = parent.theme.lower() == 'dark'
            border_color = QColor(255, 255, 255) if is_dark else QColor(51, 51, 51)
            
            # Set up the pen for the border (2px width to match main window)
            pen = QPen(border_color, 2)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            
            # Make sure we don't fill the rectangle, only draw the border
            painter.setBrush(Qt.NoBrush)
            
            # Get the window's rectangle, adjusted to account for the border
            rect = self.rect().adjusted(1, 1, -1, -1)
            
            # Draw the border with the same radius as the main window
            radius = 10  # Match the window's border radius
            painter.drawRoundedRect(rect, radius, radius)
            
        except Exception as e:
            logger.error(f"Error in BorderOverlay.paintEvent: {e}")
        finally:
            if painter.isActive():
                painter.end()

class CrosshairPicker(QWidget):
    window_selected = Signal(int)
    monitor_selected = Signal(QScreen)
    
    def __init__(self, mode="window"):
        super().__init__()
        self.mode = mode
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background-color: rgba(0,0,0,0);")
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)
        all_screens_geometry = QRect()
        for screen in QApplication.instance().screens():
            all_screens_geometry = all_screens_geometry.united(screen.geometry())
        self.setGeometry(all_screens_geometry)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.setFocus(Qt.MouseFocusReason)
        QTimer.singleShot(500, self.perform_grab)
    
    def perform_grab(self):
        self.setCursor(Qt.CrossCursor)
        self.grabMouse()
        self.grabKeyboard()
        mouse_grabber_widget = QWidget.mouseGrabber()
        keyboard_grabber_widget = QWidget.keyboardGrabber()
        if mouse_grabber_widget == self and keyboard_grabber_widget == self:
            QApplication.setOverrideCursor(Qt.CrossCursor)
            logger.info("CrosshairPicker: Mouse and Keyboard grabbed successfully.")
        else:
            if QWidget.mouseGrabber() == self:
                self.releaseMouse()
            if QWidget.keyboardGrabber() == self:
                self.releaseKeyboard()
            QApplication.restoreOverrideCursor()
            self.setCursor(Qt.ArrowCursor)
            self.hide()
    
    def hideEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        self.releaseMouse()
        self.releaseKeyboard()
        QApplication.restoreOverrideCursor()
        super().hideEvent(event)
        self.deleteLater()
    
    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        painter = QPainter(self)
        pen_color = QColor(255, 255, 255, 180)
        pen_thickness = 1
        pen = QPen(pen_color, pen_thickness, Qt.SolidLine)
        painter.setPen(pen)
        widget_rect = self.rect()
        current_global_mouse_pos = QCursor.pos()
        local_mouse_pos = self.mapFromGlobal(current_global_mouse_pos)
        painter.drawLine(widget_rect.left(), local_mouse_pos.y(), widget_rect.right(), local_mouse_pos.y())
        painter.drawLine(local_mouse_pos.x(), widget_rect.top(), local_mouse_pos.x(), widget_rect.bottom())
        painter.end()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        self.repaint()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = event.globalPosition().toPoint()
            if self.mode == "window":
                hwnd_initial = ctypes.windll.user32.WindowFromPoint(ctypes.wintypes.POINT(point.x(), point.y()))
                if hwnd_initial:
                    hwnd = hwnd_initial
                    while True:
                        parent = ctypes.windll.user32.GetParent(hwnd)
                        if parent == 0:
                            break
                        hwnd = parent
                    pid = ctypes.wintypes.DWORD()
                    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    if pid.value != os.getpid():
                        self.window_selected.emit(hwnd)
                    else:
                        logger.info(f"Self-picking detected (HWND {hwnd}). Ignoring.")
            else:
                for screen in QApplication.instance().screens():
                    if screen.geometry().contains(point):
                        self.monitor_selected.emit(screen)
                        break
            QTimer.singleShot(0, self.hide)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.hide()

class SettingsPanel(QMainWindow):
    def __init__(self, app_instance=None):
        super().__init__()
        self.setFixedSize(750, 300)
        self.app_instance = app_instance
        self.theme = self.app_instance.current_theme if self.app_instance else "Dark"
        self.picker_active = False
        self.current_selected_qscreen = None
        
        # Dragging state
        self.drag_start_position = None
        self.dragging = False
        self.installEventFilter(self)
        
        # Initialize instance variables first
        self.badge_filenames = [f"Badge{i}.png" for i in range(21)]
        self.current_badge_index = 0
        self.badge_label = None
        self._screen_info = []  # Initialize screen info list
        
        # Create the border overlay
        self.border_overlay = BorderOverlay(self)
        self.border_overlay.hide()  # Hide by default until badge is shown
        
        # Load saved badge index if available
        if self.app_instance and self.app_instance.settings:
            try:
                saved_index = self.app_instance.settings.value("UI/current_badge_index", "0")
                self.current_badge_index = int(saved_index)
                if not 0 <= self.current_badge_index < len(self.badge_filenames):
                    self.current_badge_index = 0
            except ValueError:
                self.current_badge_index = 0
        
        # Setup UI and load data
        self.setup_ui()
        # Load windows and monitors
        self.load_windows()
        self.load_monitors()
        
        # Apply theme after UI is set up
        self.apply_theme(self.theme)
        
        # Ensure the overlay is sized correctly
        self.border_overlay.resize(self.size())
    
    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        # Resize the border overlay to match the window size
        if hasattr(self, 'border_overlay') and self.border_overlay:
            self.border_overlay.resize(self.size())
    
    def showEvent(self, event):
        """Handle window show events to ensure proper badge display."""
        super().showEvent(event)
        # Update the badge display after the window is shown to ensure proper sizing
        if hasattr(self, '_update_badge_display'):
            # Use a single-shot timer to ensure this runs after the window is fully shown
            QTimer.singleShot(50, self._update_badge_display)
    
    def _get_resource_path(self, relative_path):
        """Get the QRC path for a resource file.
        
        Args:
            relative_path (str): The relative path to the resource
            
        Returns:
            str: QRC path (e.g., ":/Resources/filename.png")
        """
        # Convert to forward slashes for consistency
        path = relative_path.replace('\\', '/')
        
        # If it's already a QRC path, return as-is
        if path.startswith(':'):
            return path
            
        # Handle theme files
        if path.lower().endswith('.qss') or 'themes/' in path.lower():
            theme_name = Path(path).stem.lower()
            return f":/themes/{theme_name}.qss"
            
        # Handle resource files
        filename = Path(path).name
        return f":/Resources/{filename}"
        
    def _update_badge_display(self):
        """Update the badge display with the current badge image and theme."""
        if not hasattr(self, 'badge_label') or not self.badge_label:
            return
            
        try:
            # Get the current badge filename
            badge_filename = self.badge_filenames[self.current_badge_index]
            badge_path = self._get_resource_path(badge_filename)
            
            # Load the badge image directly from QRC
            pixmap = QPixmap(badge_path)
            
            if pixmap.isNull():
                logger.warning(f"Failed to load badge from QRC: {badge_path}")
                self.badge_label.setText("Badge N/A")
                # Hide the border overlay if badge is not found
                if hasattr(self, 'border_overlay') and self.border_overlay:
                    self.border_overlay.hide()
                return
            if pixmap.isNull():
                # Hide the border overlay if badge failed to load
                if hasattr(self, 'border_overlay') and self.border_overlay:
                    self.border_overlay.hide()
                logger.error(f"Failed to load badge image: {badge_path}")
                self.badge_label.setText("Badge Error")
                return
                
            # Get screen DPI for proper scaling
            app = QApplication.instance()
            screen = self.screen() or (app.primaryScreen() if app else None)
            dpr = screen.devicePixelRatio() if screen else 1.0
            
            # Calculate target size - make it larger but maintain aspect ratio
            max_size = 240  # Increased from 200 to make it ~20% larger
            
            # Calculate the scaled size that fits within max_size while maintaining aspect ratio
            if pixmap.width() > pixmap.height():
                scaled_pixmap = pixmap.scaledToWidth(
                    int(max_size * dpr),
                    Qt.SmoothTransformation
                )
            else:
                scaled_pixmap = pixmap.scaledToHeight(
                    int(max_size * dpr),
                    Qt.SmoothTransformation
                )
            
            # Set device pixel ratio for high DPI displays
            scaled_pixmap.setDevicePixelRatio(dpr)
            
            # Apply the pixmap to the label
            self.badge_label.setPixmap(scaled_pixmap)
            
            # Show and raise the border overlay
            if hasattr(self, 'border_overlay') and self.border_overlay:
                self.border_overlay.show()
                self.border_overlay.raise_()
            
            # Set size policy to allow growing and shrinking
            self.badge_label.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding
            )
            
            # Calculate logical size in device-independent pixels
            logical_size = scaled_pixmap.size() / dpr
            
            # Set minimum size to the scaled pixmap size
            self.badge_label.setMinimumSize(logical_size)
            self.badge_label.setMaximumSize(logical_size)
            
            # Update the container's size to match the badge
            badge_container = self.badge_label.parent()
            if badge_container:
                badge_container.setMinimumSize(logical_size)
                badge_container.setMaximumSize(logical_size)
                
                # Also update the outer container
                badge_outer_container = badge_container.parent()
                if badge_outer_container:
                    badge_outer_container.setMinimumWidth(int(logical_size.width()))
                    badge_outer_container.setMinimumHeight(int(logical_size.height()))
                    badge_outer_container.setMaximumWidth(int(logical_size.width()))
                    badge_outer_container.setMaximumHeight(int(logical_size.height()))
                
            # Force an update of the layout
            self.updateGeometry()
            QApplication.processEvents()
                
        except Exception as e:
            logger.error(f"Error updating badge display: {e}", exc_info=True)
            self.badge_label.setText("Error")
    
    def _select_random_badge(self):
        if len(self.badge_filenames) <= 1:
            return
        new_index = random.choice([i for i in range(len(self.badge_filenames)) if i != self.current_badge_index])
        self.current_badge_index = new_index
        self._update_badge_display()
        if self.app_instance and self.app_instance.settings:
            self.app_instance.settings.setValue("UI/current_badge_index", str(self.current_badge_index))
    
    def setup_drag(self):
        # Initialize drag state
        self.drag_state = {
            'dragging': False,
            'drag_start_pos': None,
            'window_start_pos': None,
            'edge': None
        }
        
        # Install event filter on the main widget for drag handling
        self.main_widget.installEventFilter(self)
        
        # Set up mouse tracking for the main widget and title bar
        self.main_widget.setMouseTracking(True)
        
        # Connect title bar mouse events
        if hasattr(self, 'title_bar'):
            self.title_bar.mousePressEvent = self.title_mouse_press
            self.title_bar.mouseMoveEvent = self.title_mouse_move
            self.title_bar.mouseReleaseEvent = self.title_mouse_release
            self.title_bar.setMouseTracking(True)

    def _get_physical_monitor_info(self, screen: QScreen) -> Dict[str, Any]:
        """
        Get physical monitor information for a screen using the centralized utility.
        
        Args:
            screen: The Qt screen to get physical info for
            
        Returns:
            Dict containing physical monitor information
        """
        # Get monitor info from the centralized utility
        monitor_info = get_physical_monitor_for_screen(screen)
        
        if not monitor_info:
            # Fallback to Qt information if no physical monitor info is available
            geo = screen.geometry()
            scale_factor = get_screen_scale_factor(screen)
            return {
                'physical_width': int(geo.width() * scale_factor),
                'physical_height': int(geo.height() * scale_factor),
                'position': QPoint(geo.x(), geo.y()),
                'work_area': screen.availableGeometry(),
                'primary': screen == QApplication.primaryScreen(),
                'monitor_rect': geo,
                'dpi': QSizeF(screen.logicalDotsPerInch(), screen.logicalDotsPerInchY()),
                'scale_factor': scale_factor
            }
            
        return monitor_info
    
    def start_window_picker(self):
        if self.picker_active:
            return
        self.picker_active = True
        self.crosshair_picker = CrosshairPicker(mode="window")
        self.crosshair_picker.window_selected.connect(self.on_window_selected)
        self.crosshair_picker.destroyed.connect(self._on_picker_destroyed)
        self.crosshair_picker.show()
    
    def on_window_selected(self, hwnd):
        if hwnd:
            self.selected_hwnd = hwnd
            for i in range(self.window_combo.count()):
                if self.window_combo.itemData(i) == hwnd:
                    self.window_combo.setCurrentIndex(i)
                    break
            else:
                window_title = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetWindowTextW(hwnd, window_title, 256)
                self.window_combo.addItem(window_title.value or "[Untitled Window]", hwnd)
                self.window_combo.setCurrentIndex(self.window_combo.count() - 1)
        self.accept_window()
    
    def on_monitor_selected(self, qscreen_obj):
        """Handle monitor selection from the picker."""
        try:
            if not qscreen_obj or not hasattr(qscreen_obj, 'geometry'):
                logger.error("Invalid screen object received from picker")
                return
                
            logger.info("Monitor selected from picker:")
            geo = qscreen_obj.geometry()
            logger.info(f"  Name: {qscreen_obj.name() if hasattr(qscreen_obj, 'name') else 'N/A'}")
            logger.info(f"  Geometry: {geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
            
            # Find the matching screen in our screens list
            screens = QApplication.instance().screens()
            matching_index = -1
            
            for i, screen in enumerate(screens):
                s_geo = screen.geometry()
                if (s_geo.x() == geo.x() and s_geo.y() == geo.y() and
                    s_geo.width() == geo.width() and s_geo.height() == geo.height()):
                    matching_index = i
                    logger.info(f"  Matched with screen index {i}")
                    break
            
            if matching_index >= 0:
                # Update the current selected screen
                self.current_selected_qscreen = screens[matching_index]
                
                # Find and select the matching item in the combo box
                for i in range(self.monitor_combo.count()):
                    screen = self.monitor_combo.itemData(i, Qt.ItemDataRole.UserRole)
                    if screen and hasattr(screen, 'geometry'):
                        s_geo = screen.geometry()
                        if (s_geo.x() == geo.x() and s_geo.y() == geo.y() and
                            s_geo.width() == geo.width() and s_geo.height() == geo.height()):
                            self.monitor_combo.setCurrentIndex(i)
                            logger.info(f"  Updated combo box selection to index {i}")
                            break
                
                # Accept the selection
                self.accept_monitor()
            else:
                logger.warning("Selected screen not found in current screens list")
                QMessageBox.warning(self, "Error", "Selected screen is no longer available.")
                
        except Exception as e:
            logger.error(f"Error in on_monitor_selected: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to select monitor: {str(e)}")
            self.accept_monitor()

    def start_monitor_picker(self):
        """Start the monitor picker to select a monitor."""
        try:
            if self.picker_active:
                logger.warning("Monitor picker is already active")
                return
                
            logger.info("Starting monitor picker...")
            
            # Log current screens for reference
            screens = QApplication.instance().screens()
            logger.info(f"Current screens available: {len(screens)}")
            for i, screen in enumerate(screens):
                geo = screen.geometry()
                logger.info(f"  Screen {i}: {screen.name() if hasattr(screen, 'name') else 'N/A'} - "
                          f"{geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
            
            self.picker = CrosshairPicker(mode="monitor")
            self.picker.monitor_selected.connect(self.on_monitor_selected)
            self.picker.destroyed.connect(self._on_picker_destroyed)
            
            # Show the picker on the primary screen
            primary_screen = QApplication.primaryScreen()
            if primary_screen:
                geo = primary_screen.geometry()
                logger.info(f"Showing picker on primary screen: {primary_screen.name() if hasattr(primary_screen, 'name') else 'N/A'}")
                self.picker.move(geo.center() - self.picker.rect().center())
            
            self.picker.show()
            self.picker.activateWindow()
            self.picker.raise_()
            self.picker.grabMouse()
            self.picker.grabKeyboard()
            self.picker_active = True
            logger.info("Monitor picker activated")
            
        except Exception as e:
            logger.error(f"Error starting monitor picker: {e}", exc_info=True)
            self.picker_active = False
            if hasattr(self, 'picker') and self.picker:
                try:
                    self.picker.deleteLater()
                except Exception as e:
                    logger.error(f"Error cleaning up picker: {e}")
                    pass
            
            QMessageBox.warning(self, "Error", "Failed to start monitor picker.")

    def _on_picker_destroyed(self):
        self.picker_active = False

    def accept_window(self):
        hwnd = self.window_combo.currentData()
        if hwnd is None:
            QMessageBox.warning(self, "Error", "No window selected.")
            return
        if self.app_instance:
            self.app_instance.prepare_to_create_window_overlay(hwnd)
            self.close()

    
    def accept_monitor(self):
        try:
            # Get the selected index from the combobox
            index = self.monitor_combo.currentIndex()
            if index < 0:
                logger.warning("No monitor selected; falling back to primary screen")
                qscreen_obj = QGuiApplication.primaryScreen()
            else:
                # Get the screen object from the combobox's item data
                qscreen_obj = self.monitor_combo.itemData(index, Qt.ItemDataRole.UserRole)
                
                # Verify we got a valid screen object
                if not isinstance(qscreen_obj, QScreen):
                    logger.warning(f"Invalid screen object at index {index}; falling back to primary screen")
                    qscreen_obj = QGuiApplication.primaryScreen()
            
            if not qscreen_obj:
                logger.error("No screens available")
                QMessageBox.critical(self, "Error", "No monitors available.")
                return
                
            # Get all available screens for verification
            all_screens = QGuiApplication.screens()
            screen_found = False
            
            # Verify the screen object is in the current screen list
            for i, screen in enumerate(all_screens):
                if (screen.name() == qscreen_obj.name() and 
                    screen.geometry() == qscreen_obj.geometry()):
                    qscreen_obj = screen  # Use the current screen object
                    screen_found = True
                    break
            
            if not screen_found:
                logger.warning(f"Selected screen not found in current screen list, using primary screen")
                qscreen_obj = QGuiApplication.primaryScreen()
                
            # Log the selected screen details for debugging
            screen_name = qscreen_obj.name() if hasattr(qscreen_obj, 'name') else 'unnamed'
            geo = qscreen_obj.geometry()
            logger.info(f"Selected screen: {screen_name} - {geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
            
            # Log all available screens for debugging
            logger.info("Available screens at selection time:")
            for i, screen in enumerate(QGuiApplication.screens()):
                g = screen.geometry()
                logger.info(f"  Screen {i}: {screen.name() if hasattr(screen, 'name') else 'N/A'} - "
                          f"{g.width()}x{g.height()} @ ({g.x()},{g.y()})")
            
            if not self.app_instance:
                logger.error("App instance not available")
                QMessageBox.critical(self, "Error", "Application instance not available.")
                return
            
            logger.info(f"Creating monitor overlay for screen: {screen_name}")
            self.app_instance.prepare_to_create_monitor_overlay(qscreen_obj)
            self.current_selected_qscreen = None
            self.close()
            
        except Exception as e:
            logger.error(f"Error in accept_monitor: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to select monitor: {str(e)}")
            self.close()
    
    def quick_start(self):
        active_window = ctypes.windll.user32.GetForegroundWindow()
        if active_window and self.app_instance:
            self.app_instance.prepare_to_create_window_overlay(active_window)
            self.close()
    
    def minimize_to_tray(self):
        if self.app_instance and self.app_instance._tray_icon:
            self.hide()
    
    def show_settings(self):
        if self.app_instance:
            self.app_instance._show_sub_settings()
    
    def show_about_dialog(self):
        if self.app_instance:
            if not hasattr(self, '_about_dialog_instance') or not self._about_dialog_instance.isVisible():
                from about_dialog import AboutDialog
                self._about_dialog_instance = AboutDialog(parent=self, app_instance=self.app_instance)
                # Ensure theme is applied after dialog is shown
                self._about_dialog_instance.show()
                # Force theme application after dialog is shown to ensure styles are applied
                if hasattr(self.app_instance, 'current_theme'):
                    self._about_dialog_instance.apply_theme(self.app_instance.current_theme)
            else:
                self._about_dialog_instance.activateWindow()
                self._about_dialog_instance.raise_()
                # Re-apply theme when bringing to front
                if hasattr(self.app_instance, 'current_theme'):
                    self._about_dialog_instance.apply_theme(self.app_instance.current_theme)
    
    def eventFilter(self, obj, event):
        # Handle badge double-click
        if obj == self.badge_label and event.type() == QEvent.MouseButtonDblClick:
            self._select_random_badge()
            return True
            
        # Handle window dragging for the main widget
        if obj == self or obj == self.centralWidget():
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
                    self.dragging = True
                    return True
                    
            elif event.type() == QEvent.MouseMove and self.dragging:
                if self.drag_start_position is not None:
                    self.move(event.globalPos() - self.drag_start_position)
                    return True
                    
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.dragging = False
                    return True
        
        # Handle other events normally
        return super().eventFilter(obj, event)
    
    def snap_to_edge(self, pos=None):
        """
        Snap the window to screen edges if close enough.
        This is now handled by snap_utils.apply_snap, but kept for compatibility.
        """
        if pos is None:
            pos = self.pos()
        new_pos, _ = apply_snap(pos, self.size(), snap_distance=30)
        if new_pos != pos:
            self.move(new_pos)
    
    def _get_current_screen(self):
        center_point = self.geometry().center()
        app = QApplication.instance()
        return app.screenAt(center_point) or app.primaryScreen()
    
    def load_windows(self):
        if not self.app_instance:
            self.window_combo.clear()
            self.window_combo.addItem("Error: App instance not found", None)
            self.window_combo.setEnabled(False)
            return
        windows_with_icons = self.app_instance.get_menu_ready_windows()
        self.window_combo.clear()
        if not windows_with_icons:
            self.window_combo.addItem("No capturable windows found", None)
            self.window_combo.setEnabled(False)
        else:
            self.window_combo.setEnabled(True)
            for hwnd, title, q_icon in windows_with_icons:
                display_title = f"{title} (0x{hwnd:X})" if title else f"[Untitled Window] (0x{hwnd:X})"
                if q_icon and not q_icon.isNull():
                    self.window_combo.addItem(q_icon, display_title, hwnd)
                else:
                    self.window_combo.addItem(display_title, hwnd)
    
    def get_combo_style(self):
        """Return the stylesheet for QComboBox, loading from combo_styles.qss."""
        try:
            # Try to load from the theme file first
            theme_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'themes')
            style_file = os.path.join(theme_dir, 'combo_styles.qss')
            
            with open(style_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not load combo styles from file, using fallback: {e}")
            return self._get_fallback_combo_style(self.theme.lower() if hasattr(self, 'theme') else 'dark')

    def _get_display_info(self, screen: QScreen, idx: int) -> Dict[str, Any]:
        """Generate display information for the screen using centralized monitor info.
        
        Args:
            screen: QScreen object
            idx: 0-based index of the screen
            
        Returns:
            dict: Contains display_text, tooltip, and screen object
        """
        try:
            # Get physical monitor information
            monitor_info = self._get_physical_monitor_info(screen)
            
            # Get display name, fallback to 'Display N' if not available
            try:
                name = screen.name()
                if not name or name.startswith('\\.\\'):
                    name = f'Display {idx + 1}'
                else:
                    # Clean up the display name
                    name = name.strip()
                    if name.lower().startswith('display'):
                        name = name[0].upper() + name[1:]  # Capitalize first letter
            except Exception:
                name = f'Display {idx + 1}'
            
            # Extract physical and logical dimensions
            physical_width = monitor_info.get('physical_width', 0)
            physical_height = monitor_info.get('physical_height', 0)
            position = monitor_info.get('position', QPoint(0, 0))
            dpi = monitor_info.get('dpi', (96, 96))
            scale_factor = monitor_info.get('scale_factor', 1.0)
            logical_geo = screen.geometry()
            refresh_rate = screen.refreshRate()
            
            # Convert DPI to QSizeF if it's a tuple
            if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
                dpi_size = QSizeF(dpi[0], dpi[1])
            elif hasattr(dpi, 'width') and hasattr(dpi, 'height'):
                dpi_size = QSizeF(dpi.width(), dpi.height())
            else:
                dpi_size = QSizeF(96, 96)
            
            # Log detailed screen information
            screen_name = screen.name() if hasattr(screen, 'name') and callable(screen.name) else f'Display {idx}'
            logger.info(f"=== Processing Monitor {idx} ===")
            logger.info(f"Screen name: {screen_name}")
            logger.info(f"Qt logical geometry: {logical_geo}")
            logger.info(f"Physical resolution: {physical_width}x{physical_height}")
            logger.info(f"Scale factor: {scale_factor:.2f}")
            logger.info(f"DPI: {dpi_size.width():.1f}x{dpi_size.height():.1f}")
            
            # Use physical resolution for display text
            display_text = f"{name} - {physical_width}×{physical_height}"
            
            try:
                manufacturer = screen.manufacturer() if hasattr(screen, 'manufacturer') and callable(screen.manufacturer) else 'N/A'
                model = screen.model() if hasattr(screen, 'model') and callable(screen.model) else 'N/A'
                
                # Log additional screen information
                logger.info(f"Manufacturer: {manufacturer}")
                logger.info(f"Model: {model}")
                logger.info(f"Refresh Rate: {refresh_rate}Hz")
                
                # Ensure we have a unique identifier for each screen
                if hasattr(screen, 'serialNumber') and callable(screen.serialNumber):
                    serial = screen.serialNumber()
                    logger.info(f"Serial Number: {serial}")
                    if not name or name == f'Display {idx + 1}':
                        name = f'Display {idx + 1} ({serial[-4:]})'  # Use last 4 digits of serial if available
                
                tooltip_parts = [
                    f"Display: {name}",
                    f"Physical Resolution: {physical_width} × {physical_height}",
                    f"Qt Logical Resolution: {logical_geo.width()} × {logical_geo.height()}",
                    f"Scale Factor: {scale_factor:.2f}",
                    f"Position: ({position.x()}, {position.y()})",
                    f"DPI: {dpi_size.width():.1f} × {dpi_size.height():.1f}",
                    f"Refresh Rate: {refresh_rate}Hz"
                ]
                
                if manufacturer != 'N/A' or model != 'N/A':
                    tooltip_parts.append(f"{manufacturer} {model}".strip())
                
                tooltip = '\n'.join(tooltip_parts)
                
            except Exception as e:
                logger.warning(f"Could not get detailed display info: {e}")
                tooltip = f"{name}\n{physical_width} × {physical_height} @ {dpi.width():.1f} DPI\n{refresh_rate}Hz"
            
            return {
                'display_text': display_text,
                'tooltip': tooltip,
                'screen': screen,
                'physical_width': physical_width,
                'physical_height': physical_height,
                'scale_factor': scale_factor,
                'dpi': dpi
            }
                
        except Exception as e:
            logger.error(f"Error getting display info: {e}", exc_info=True)
            return {
                'display_text': f"Display {idx + 1} (Error)",
                'tooltip': f"Error getting display information: {e}",
                'screen': screen,
                'physical_width': 0,
                'physical_height': 0,
                'scale_factor': 1.0,
                'dpi': QSizeF(96, 96)
            }
    
    def get_combo_style(self):
        """Return the stylesheet for QComboBox, loading from the current theme's QSS file."""
        try:
            # Get the current theme name
            theme_name = getattr(self, 'theme', 'dark').lower()
            logger.debug(f"Loading combo styles for theme: {theme_name}")
            
            # First try loading from combo_styles.qrc
            try:
                combo_qrc = ":/themes/combo_styles.qss"
                combo_file = QFile(combo_qrc)
                if combo_file.open(QFile.ReadOnly | QFile.Text):
                    stream = QTextStream(combo_file)
                    styles = stream.readAll()
                    combo_file.close()
                    if styles.strip():
                        logger.debug("Successfully loaded combo styles from combo_styles.qss")
                        return styles
                    else:
                        logger.warning("combo_styles.qss is empty")
                else:
                    error = combo_file.errorString()
                    logger.warning(f"Failed to open combo_styles.qrc: {error}")
            except Exception as e:
                logger.warning(f"Error loading combo_styles.qss: {e}", exc_info=True)
            
            # If no styles found in combo_styles.qrc, try loading from the theme file
            try:
                theme_qrc = f":/themes/{theme_name}.qss"
                theme_file = QFile(theme_qrc)
                if theme_file.open(QFile.ReadOnly | QFile.Text):
                    stream = QTextStream(theme_file)
                    content = stream.readAll()
                    theme_file.close()
                    
                    if content:
                        # Look for QComboBox styles in the theme content
                        if 'QComboBox' in content:
                            import re
                            combo_styles = re.search(r'QComboBox\s*\{[^}]*\}', content, re.DOTALL)
                            if combo_styles:
                                logger.debug("Found QComboBox styles in theme file")
                                return combo_styles.group(0)
                            else:
                                logger.debug("No QComboBox styles found in theme file")
                        else:
                            logger.debug("No QComboBox styles found in theme file (no QComboBox selector)")
                    else:
                        logger.warning(f"Theme file {theme_qrc} is empty")
                else:
                    error = theme_file.errorString()
                    logger.warning(f"Failed to open theme file {theme_qrc}: {error}")
            except Exception as e:
                logger.warning(f"Error loading theme file {theme_qrc}: {e}", exc_info=True)
            
            # If we get here, use the fallback styles with the current theme
            logger.warning("No QComboBox styles found in any source, using fallback styles")
            return self._get_fallback_combo_style(theme_name)
            
        except Exception as e:
            logger.error(f"Error loading combo styles: {e}", exc_info=True)
            return self._get_fallback_combo_style('dark')

    def load_monitors(self):
        """Load available monitors with detailed information in tooltips.
        
        Note: Monitor indices are 0-based in Qt, but displayed as 1-based to users.
        """
        if not hasattr(self, 'monitor_combo'):
            logger.warning("monitor_combo not initialized, skipping monitor load")
            return
        
        # Initialize screens list at the start
        screens = QApplication.screens()
        logger.info(f"Loading {len(screens)} available screens...")
            
        try:
            # Store the current selection if any
            current_screen = None
            if self.monitor_combo.currentIndex() >= 0:
                current_screen = self.monitor_combo.currentData(Qt.ItemDataRole.UserRole)
            
            # Disconnect the signal to prevent multiple triggers during update
            try:
                self.monitor_combo.currentIndexChanged.disconnect()
            except (TypeError, RuntimeError) as e:
                # Ignore if not connected or already disconnected
                logger.debug(f"Could not disconnect signal: {e}")
                
            # Clear existing items and screen info
            self.monitor_combo.clear()
            self._screen_info = []
            
            # Add monitors with basic info and detailed tooltips
            for i, screen_obj in enumerate(screens):
                try:
                    if not hasattr(screen_obj, 'geometry'):
                        logger.error(f"Screen {i} has no geometry method!")
                        continue
                        
                    # Get display info for this screen
                    info = self._get_display_info(screen_obj, i)
                    self._screen_info.append(info)
                    
                    # Add item with display text
                    display_text = info.get('display_text', f'Display {i+1}')
                    self.monitor_combo.addItem(display_text)
                    
                    # Store the screen object and its index
                    idx = self.monitor_combo.count() - 1
                    self.monitor_combo.setItemData(idx, info['tooltip'], Qt.ItemDataRole.ToolTipRole)
                    self.monitor_combo.setItemData(idx, screen_obj, Qt.ItemDataRole.UserRole)
                    self.monitor_combo.setItemData(idx, i, Qt.ItemDataRole.UserRole + 1)
                    
                    logger.info(f"  Added monitor {i}: {display_text}")
                    
                except Exception as e:
                    logger.error(f"Error processing screen {i}: {e}", exc_info=True)
            
            # Reconnect the signal after populating
            self.monitor_combo.currentIndexChanged.connect(self._on_monitor_selected)
            
            # Restore selection if possible
            if current_screen:
                for i in range(self.monitor_combo.count()):
                    screen = self.monitor_combo.itemData(i, Qt.ItemDataRole.UserRole)
                    if screen == current_screen:
                        self.monitor_combo.setCurrentIndex(i)
                        break
            else:
                # Select the primary screen by default if available
                primary_screen = QApplication.primaryScreen()
                if primary_screen and primary_screen in screens:
                    try:
                        idx = screens.index(primary_screen)
                        if idx < self.monitor_combo.count():
                            self.monitor_combo.setCurrentIndex(idx)
                            logger.info(f"Set primary screen as default: {primary_screen.name() if hasattr(primary_screen, 'name') else 'N/A'}")
                    except ValueError:
                        pass
            
            # Fallback to first screen if no selection
            if self.monitor_combo.currentIndex() < 0 and self.monitor_combo.count() > 0:
                self.monitor_combo.setCurrentIndex(0)
                logger.info("Falling back to first screen as default")
            
            # Update the current selected screen
            if self.monitor_combo.currentIndex() >= 0:
                self.current_selected_qscreen = self.monitor_combo.currentData(Qt.ItemDataRole.UserRole)
                logger.info(f"Monitor combo box populated with {self.monitor_combo.count()} items, current: {self.monitor_combo.currentText()}")
            else:
                logger.warning("No monitors available in combo box")
            
        except Exception as e:
            logger.error(f"Error loading monitors: {e}", exc_info=True)
            # Fallback to minimal display
            self.monitor_combo.clear()
            for i, screen in enumerate(screens):
                self.monitor_combo.addItem(f"Display {i + 1}", screen)
            if screens:
                self.monitor_combo.setCurrentIndex(0)
                self.current_selected_qscreen = screens[0]
    
    def _on_monitor_selected(self, index):
        """Handle monitor selection change."""
        try:
            if index < 0 or index >= self.monitor_combo.count():
                logger.warning(f"Invalid monitor index selected: {index}")
                return
                
            # Get the screen object from the combo box
            screen = self.monitor_combo.itemData(index, Qt.ItemDataRole.UserRole)
            if not screen or not hasattr(screen, 'geometry'):
                logger.error(f"Invalid screen object at index {index}")
                return
                
            # Get the stored screen index for verification
            screen_idx = self.monitor_combo.itemData(index, Qt.ItemDataRole.UserRole + 1)
            geo = screen.geometry()
            
            # Log detailed information about the selected screen
            logger.info(f"Selected monitor {index} (stored index: {screen_idx}):")
            logger.info(f"  Name: {screen.name() if hasattr(screen, 'name') else 'N/A'}")
            logger.info(f"  Geometry: {geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
            
            # Update the current screen reference
            self.current_selected_qscreen = screen
            
            # Also update the screen info if available
            if 0 <= index < len(self._screen_info):
                logger.info(f"  Display name: {self._screen_info[index].get('display_text', 'N/A')}")
                
            logger.info(f"Current screen reference updated for index {index}")
            
        except Exception as e:
            logger.error(f"Error in _on_monitor_selected: {e}", exc_info=True)
            
    def apply_theme(self, theme_name, from_global=False):
        """Apply the specified theme to the UI.
        
        Args:
            theme_name (str): Name of the theme to apply (e.g., 'dark', 'light')
            from_global (bool): Whether this is being called from a global theme change
        """
        try:
            self.theme = theme_name
            theme_name = theme_name.lower()
            
            # Define QRC paths
            theme_qrc_path = f":/themes/{theme_name}.qss"
            combo_style_qrc_path = ":/themes/combo_styles.qss"
            
            logger.info(f"Attempting to load theme from QRC: {theme_qrc_path}")
            
            # Load and apply the main stylesheet from QRC
            try:
                theme_file = QFile(theme_qrc_path)
                if not theme_file.open(QFile.ReadOnly | QFile.Text):
                    raise IOError(f"Failed to open QRC resource: {theme_qrc_path}")
                
                stream = QTextStream(theme_file)
                stylesheet = stream.readAll()
                theme_file.close()
                
                # Replace any resource paths in the stylesheet
                if stylesheet:
                    stylesheet = stylesheet.replace('url(Resources/', 'url(:/Resources/')
                    stylesheet = stylesheet.replace('url("Resources/', 'url(":/Resources/')
                    
                    self.setStyleSheet(stylesheet)
                    logger.info(f"Successfully loaded theme: {theme_name}")
                else:
                    logger.warning(f"Empty stylesheet loaded from {theme_qrc_path}")
            except Exception as e:
                logger.error(f"Failed to load theme file {theme_qrc_path}: {e}")
                raise
            
            # Load and apply combo box styles from QRC
            combo_styles = ""
            try:
                combo_file = QFile(combo_style_qrc_path)
                if combo_file.open(QFile.ReadOnly | QFile.Text):
                    stream = QTextStream(combo_file)
                    combo_styles = stream.readAll()
                    combo_file.close()
                    
                    # Replace any resource paths in the combo styles
                    if combo_styles:
                        combo_styles = combo_styles.replace('url(Resources/', 'url(:/Resources/')
                    combo_styles = combo_styles.replace('url("Resources/', 'url(":/Resources/')
                    logger.info("Successfully loaded combo box styles from QRC")
                else:
                    raise IOError(f"Failed to open QRC resource: {combo_style_qrc_path}")
            except Exception as e:
                logger.warning(f"Could not load combo box styles from QRC: {e}")
                combo_styles = self._get_fallback_combo_style(theme_name)
            
            # Apply styles to all combo boxes
            for combo_attr in ['window_combo', 'monitor_combo']:
                if hasattr(self, combo_attr):
                    combo = getattr(self, combo_attr)
                    
                    # Ensure we're using QListView for dropdown
                    if not isinstance(combo.view(), QListView):
                        combo.setView(QListView())
                    
                    # Apply the combo box styles
                    combo.setStyleSheet(combo_styles)
                    
                    # Set the property for light/dark theming
                    combo.setProperty('class', 'light' if theme_name == 'light' else '')
                    
                    # Force style update
                    combo.style().unpolish(combo)
                    combo.style().polish(combo)
                    combo.update()
            
            # Update badge display to match theme
            if hasattr(self, 'badge_label'):
                self._update_badge_display()
                
            logger.info(f"Applied {theme_name} theme{' (from global)' if from_global else ''}")
            
        except Exception as e:
            logger.error(f"Error applying theme {theme_name}: {e}", exc_info=True)
            # Fall back to default theme if specified theme fails
            if theme_name.lower() != 'dark':
                logger.info("Falling back to dark theme")
                self.apply_theme('Dark')
            else:
                # If dark theme also fails, apply minimal styles
                self._apply_minimal_styles()
    
    def _apply_minimal_styles(self):
        """Apply minimal fallback styles when theme loading fails."""
        try:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2e2e2e;
                }
                QComboBox {
                    color: white;
                    background-color: #2e2e2e;
                    border: 1px solid #444;
                    padding: 5px;
                    min-height: 24px;
                }
                QComboBox::down-arrow {
                    width: 8px;
                    height: 8px;
                    background-color: white;
                    border-radius: 4px;
                    margin-right: 6px;
                    image: none;
                }
                QComboBox QAbstractItemView {
                    color: white;
                    background-color: #2e2e2e;
                    selection-background-color: #3a3a3a;
                    outline: 1px solid #444;
                }
            """)
            logger.info("Applied minimal fallback styles")
        except Exception as e:
            logger.error(f"Failed to apply fallback styles: {e}", exc_info=True)
    
    def _get_fallback_combo_style(self, theme_name):
        """Generate a fallback style for combo boxes.
        
        Args:
            theme_name (str): Name of the current theme (light/dark)
            
        Returns:
            str: CSS styles for combo boxes
        """
        is_light = theme_name.lower() == 'light'
        bg_color = "#f0f0f0" if is_light else "#2e2e2e"
        text_color = "#000000" if is_light else "#ffffff"
        border_color = "#cccccc" if is_light else "#444444"
        hover_bg = "#e0e0e0" if is_light else "#3a3a3a"
        
        return f"""
            QComboBox {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 5px 10px 5px 5px;
                min-width: 6em;
            }}
            QComboBox:hover {{
                background-color: {hover_bg};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                width: 8px;
                height: 8px;
                background-color: {text_color};
                border-radius: 4px;
                margin-right: 6px;
                image: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {bg_color};
                color: {text_color};
                selection-background-color: {hover_bg};
                outline: 1px solid {border_color};
            }}
        """
            
    def quit_application(self):
        """Handle application quit action."""
        logger.info("Quit application requested from settings panel")
        if hasattr(self, 'app_instance') and self.app_instance:
            # If we have an app instance, use its quit method
            self.app_instance.quit()
        else:
            # Otherwise, just close the window
            self.close()
            
    def title_mouse_press(self, event):
        """Handle mouse press on title bar."""
        if event.button() == Qt.LeftButton:
            self.drag_state = snap_utils.handle_overlay_mouse_press(event, self)
            if not self.drag_state.get('is_resizing', False):
                event.accept()
            else:
                super().mousePressEvent(event)
            
    def title_mouse_move(self, event):
        """Handle mouse move on title bar with snapping."""
        if hasattr(self, 'drag_state'):
            # Use the centralized mouse move handler from snap_utils
            handled = snap_utils.handle_overlay_mouse_move(event, self, self.drag_state)
            if handled:
                event.accept()
                return
        
        # If not handled by the overlay handler, call the parent's implementation
        super().mouseMoveEvent(event)
            
    def title_mouse_release(self, event):
        """Handle mouse release on title bar with final snap."""
        if hasattr(self, 'drag_state') and event.button() == Qt.LeftButton:
            # Use the centralized mouse release handler from snap_utils
            if snap_utils.handle_overlay_mouse_release(event, self, self.drag_state):
                event.accept()
                return
        
        # If not handled by the overlay handler, call the parent's implementation
        super().mouseReleaseEvent(event)
            
    # Removed duplicate eventFilter method - merged into the one above
    
    def setup_ui(self):
        """Set up the main user interface components."""
        # Set window properties
        self.setWindowTitle("SHITTY PICTURE IN PICTURE")
        self.setFixedSize(800, 300)  # Fixed size to match the design
        
        # Enable custom window styling and remove default title bar
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        
        # Create main widget and set it as the central widget
        self.main_widget = QWidget()
        self.main_widget.setObjectName("mainWidget")
        self.setCentralWidget(self.main_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Title bar container - single row with title frame and close button
        self.title_bar = QWidget()
        self.title_bar.setObjectName("titleBar")
        self.title_bar.setFixedHeight(60)  # Set fixed height for the title bar
        
        # Main horizontal layout for the title bar
        title_bar_layout = QHBoxLayout(self.title_bar)
        title_bar_layout.setContentsMargins(30, 10, 30, 10)  # Add padding around the content
        title_bar_layout.setSpacing(0)
        
        # Container for the title frame - spans most of the width
        title_frame = QFrame()
        title_frame.setObjectName("titleFrame")
        title_frame.setFixedHeight(40)
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(10, 0, 10, 0)
        
        title_label = QLabel("SHITTY PICTURE IN PICTURE".replace("PiP", "Pip"))
        title_label.setObjectName("titleLabel")
        title_layout.addWidget(title_label, 0, Qt.AlignCenter)
        
        title_bar_layout.addWidget(title_frame, 1)  # Allow frame to expand
        
        # Add spacer to push the close button to the right
        title_bar_layout.addSpacing(20)  # Space between title and close button
        
        # Close button with adjusted position (2px higher)
        close_button_container = QWidget()
        close_button_container.setObjectName("closeButtonContainer")
        close_button_layout = QVBoxLayout(close_button_container)
        close_button_layout.setContentsMargins(0, 6, 0, 0)  # 6px top padding (2px higher than before)
        close_button_layout.setSpacing(0)
        close_button_layout.addStretch()  # Push button to the bottom
        
        self.close_button = QPushButton()
        self.close_button.setObjectName("closeButton")
        self.close_button.setFixedSize(24, 24)
        self.close_button.clicked.connect(self.quit_application)
        self.close_button.setText("")
        self.close_button.setToolTip("Quit")
        
        close_button_layout.addWidget(self.close_button)
        close_button_layout.addStretch()  # Add space at the bottom
        
        title_bar_layout.addWidget(close_button_container, 0, Qt.AlignTop | Qt.AlignRight)
        
        # Add title bar to main layout
        self.title_bar_container = QWidget()
        self.title_bar_container.setObjectName("titleBarContainer")
        title_container_layout = QVBoxLayout(self.title_bar_container)
        title_container_layout.setContentsMargins(0, 0, 0, 0)
        title_container_layout.setSpacing(0)
        title_container_layout.addWidget(self.title_bar)
        
        # Add title bar container to main layout
        self.main_layout.addWidget(self.title_bar_container)
        
        # Create main content area
        content_widget = QWidget()
        content_widget.setObjectName("contentWidget")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)
        
        # Main content container (left side: controls, right side: badge)
        main_content = QHBoxLayout()
        main_content.setContentsMargins(0, 0, 0, 0)
        main_content.setSpacing(0)
        
        # Left side container for controls
        left_side = QWidget()
        left_side.setObjectName("leftSide")
        left_layout = QVBoxLayout(left_side)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)  # Reduced spacing between rows
        
        # Window selection row with consistent spacing
        window_row = QHBoxLayout()
        window_row.setContentsMargins(0, 0, 0, 0)
        window_row.setSpacing(9)  # Consistent spacing between widgets
        window_row.setAlignment(Qt.AlignLeft)  # Left align all items in the row
        
        self.window_btn = QPushButton("SELECT WINDOW")
        self.window_btn.setObjectName("selectButton")
        self.window_btn.clicked.connect(self.start_window_picker)
        self.window_btn.setFixedSize(108, 32)
        
        # Create window combo with QListView to avoid native styling issues
        self.window_combo = QComboBox()
        self.window_combo.setView(QListView())
        self.window_combo.view().window().setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.window_combo.view().window().setAttribute(Qt.WA_TranslucentBackground)
        self.window_combo.setObjectName("windowCombo")
        self.window_combo.setFixedSize(200, 32)
        
        # Configure view properties
        view = self.window_combo.view()
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Apply styles from theme
        self.window_combo.setStyleSheet(self.get_combo_style())
        
        self.window_start_btn = QPushButton("START")
        self.window_start_btn.setObjectName("startButton")
        self.window_start_btn.clicked.connect(self.accept_window)
        self.window_start_btn.setFixedSize(56, 32)
        
        window_row.addWidget(self.window_btn)
        window_row.addSpacing(4)  # Extra spacing after button
        window_row.addWidget(self.window_combo)
        window_row.addSpacing(4)  # Extra spacing before start button
        window_row.addWidget(self.window_start_btn)
        window_row.addStretch()  # Push everything to the left
        
        left_layout.addLayout(window_row)
        
        # Monitor selection row with consistent spacing
        monitor_row = QHBoxLayout()
        monitor_row.setContentsMargins(0, 0, 0, 0)
        monitor_row.setSpacing(9)  # Consistent spacing between widgets
        monitor_row.setAlignment(Qt.AlignLeft)  # Left align all items in the row
        
        self.monitor_btn = QPushButton("SELECT MONITOR")
        self.monitor_btn.setObjectName("selectButton")
        self.monitor_btn.clicked.connect(self.start_monitor_picker)
        self.monitor_btn.setFixedSize(108, 32)
        
        # Create monitor combo with QListView to avoid native styling issues
        self.monitor_combo = QComboBox()
        self.monitor_combo.setView(QListView())
        self.monitor_combo.view().window().setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.monitor_combo.view().window().setAttribute(Qt.WA_TranslucentBackground)
        self.monitor_combo.setObjectName("monitorCombo")
        self.monitor_combo.setFixedSize(200, 32)
        
        # Configure view properties
        view = self.monitor_combo.view()
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Apply styles from theme
        self.monitor_combo.setStyleSheet(self.get_combo_style())
        # Force update the combo box style
        self.monitor_combo.style().unpolish(self.monitor_combo)
        self.monitor_combo.style().polish(self.monitor_combo)
        
        self.monitor_start_btn = QPushButton("START")
        self.monitor_start_btn.setObjectName("startButton")
        self.monitor_start_btn.clicked.connect(self.accept_monitor)
        self.monitor_start_btn.setFixedSize(56, 32)
        
        monitor_row.addWidget(self.monitor_btn)
        monitor_row.addSpacing(4)  # Extra spacing after button
        monitor_row.addWidget(self.monitor_combo)
        monitor_row.addSpacing(4)  # Extra spacing before start button
        monitor_row.addWidget(self.monitor_start_btn)
        monitor_row.addStretch()  # Push everything to the left
        
        left_layout.addLayout(monitor_row)
        
        # Bottom buttons row
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        
        # Left side buttons (minimize, settings, about)
        left_buttons = QHBoxLayout()
        left_buttons.setSpacing(0)  # Set to 0 and control spacing manually
        
        # Minimize to Tray button - left aligned
        self.minimize_btn = QPushButton("MINIMIZE TO TRAY")
        self.minimize_btn.setObjectName("actionButton")
        self.minimize_btn.clicked.connect(self.minimize_to_tray)
        self.minimize_btn.setFixedHeight(32)
        
        # Settings button - next to minimize
        self.settings_btn = QPushButton("SUBSETTINGS")
        self.settings_btn.setObjectName("actionButton")
        self.settings_btn.clicked.connect(self.show_settings)
        self.settings_btn.setFixedSize(120, 32)  # Increased width to fit 'SUBSETTINGS' text
        
        # Add buttons to left_buttons layout with manual spacing
        left_buttons.addWidget(self.minimize_btn)
        left_buttons.addSpacing(8)  # Reduced from 10px to 8px after minimize button
        left_buttons.addWidget(self.settings_btn)
        left_buttons.addSpacing(3)  # Reduced from 5px to 3px before about button
        
        # Add about button next to settings with adjusted vertical and horizontal position
        self.about_btn = QPushButton("?")
        self.about_btn.setObjectName("aboutButton")
        self.about_btn.clicked.connect(self.show_about_dialog)
        self.about_btn.setFixedSize(32, 32)
        
        # Create a container for the about button to adjust its vertical position
        about_container = QWidget()
        about_layout = QVBoxLayout(about_container)
        about_layout.setContentsMargins(0, 11, 0, 0)  # 11px top margin to lower the button by 3px
        about_layout.addWidget(self.about_btn)
        left_buttons.addWidget(about_container)
        
        # Add stretch to push buttons to the left
        left_buttons.addStretch()
        
        # Add left_buttons to main button_row
        button_row.addLayout(left_buttons)
        
        left_layout.addLayout(button_row)
        left_layout.addStretch()
        
        # Add left side to main content with stretch
        main_content.addWidget(left_side, 1)
        
        # Create a container for the badge that will be placed in the bottom-right
        badge_container = QWidget()
        badge_container.setObjectName("badgeContainer")
        badge_container.setStyleSheet("""
            #badgeContainer {
                background-color: transparent;
                border: none;
                padding: 0;
                margin: 0;
            }
        """)
        
        # Use a grid layout for precise control
        badge_layout = QGridLayout(badge_container)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.setSpacing(0)
        
        # Create and configure the badge label
        self.badge_label = QLabel()
        self.badge_label.setObjectName("badgeLabel")
        self.badge_label.setStyleSheet("""
            #badgeLabel {
                background-color: transparent;
                border: none;
                padding: 0;
                margin: 0;
            }
        """)
        self.badge_label.installEventFilter(self)  # Install event filter for double-click
        
        # Add the badge to the bottom-right of the grid
        badge_layout.addWidget(self.badge_label, 0, 0, Qt.AlignRight | Qt.AlignBottom)
        
        # Create a container for the badge to control its position
        badge_outer_container = QWidget()
        badge_outer_container.setObjectName("badgeOuterContainer")
        badge_outer_layout = QVBoxLayout(badge_outer_container)
        badge_outer_layout.setContentsMargins(0, 0, 0, 0)
        badge_outer_layout.setSpacing(0)
        badge_outer_layout.addStretch()  # Push to bottom
        badge_outer_layout.addWidget(badge_container, 0, Qt.AlignRight | Qt.AlignBottom)
        
        # Add the outer container to the main content
        main_content.addWidget(badge_outer_container, 1, Qt.AlignRight | Qt.AlignBottom)
        
        # Add main content to the content layout
        content_layout.addLayout(main_content)
        
        # Add content widget to main layout
        self.main_layout.addWidget(content_widget, 1)
        
        # The main layout is already set on self.main_widget in the constructor
        
        # Set window properties and setup drag functionality
        self.setup_drag()
        
        # Load initial data
        self.load_windows()
        self.load_monitors()
        
        # Load and display initial badge
        self._update_badge_display()
        
        # Apply theme
        self.apply_theme(self.theme)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SettingsPanel()
    window.show()
    sys.exit(app.exec())
