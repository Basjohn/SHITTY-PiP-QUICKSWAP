import sys
import logging
import os
import random
import ctypes
from pathlib import Path
from typing import Dict, Any, Optional
from PySide6.QtCore import QEvent, QPoint, QRect, QTimer, Qt, Signal, QSize, QSizeF, QFile, QTextStream
from PySide6.QtGui import QCursor, QMouseEvent, QPaintEvent, QPainter, QPen, QPixmap, QScreen, QColor, QGuiApplication
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QWidget, QFrame, QMessageBox,
                             QSizePolicy, QListView,
                             QGridLayout)

# Resource paths using QRC
THEMES_DIR = ":/themes"
RESOURCES_DIR = ":/Resources"

# Import monitor utilities
from monitor_utils import (
    get_physical_monitor_info,
    get_all_monitors
)

# Import snap utilities
from snap_utils import (
    handle_overlay_mouse_press,
    handle_overlay_mouse_move,
    handle_overlay_mouse_release,
    apply_snap
)

logger = logging.getLogger(__name__)

class BorderOverlay(QWidget):
    """A transparent overlay that draws a border on top of all other widgets."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        
    def paintEvent(self, event):
        """Draw a border matching the main window's style."""
        parent = self.parent()
        if not hasattr(parent, 'theme'):
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            is_dark = parent.theme.lower() == 'dark'
            border_color = QColor(255, 255, 255) if is_dark else QColor(51, 51, 51)
            pen = QPen(border_color, 2)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect = self.rect().adjusted(1, 1, -1, -1)
            radius = 10
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
        self.drag_start_position = None
        self.dragging = False
        self.installEventFilter(self)
        self.badge_filenames = [f"Badge{i}.png" for i in range(21)]
        self.current_badge_index = 0
        self.badge_label = None
        self._screen_info = []
        self.border_overlay = BorderOverlay(self)
        self.border_overlay.hide()
        if self.app_instance and self.app_instance.settings:
            try:
                saved_index = self.app_instance.settings.value("UI/current_badge_index", "0")
                self.current_badge_index = int(saved_index)
                if not 0 <= self.current_badge_index < len(self.badge_filenames):
                    self.current_badge_index = 0
            except ValueError:
                self.current_badge_index = 0
        self.setup_ui()
        self.load_windows()
        self.load_monitors()
        self.apply_theme(self.theme)
        self.border_overlay.resize(self.size())
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'border_overlay') and self.border_overlay:
            self.border_overlay.resize(self.size())
    
    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, '_update_badge_display'):
            QTimer.singleShot(50, self._update_badge_display)
    
    def _get_resource_path(self, relative_path):
        path = relative_path.replace('\\', '/')
        if path.startswith(':'):
            return path
        if path.lower().endswith('.qss') or 'themes/' in path.lower():
            theme_name = Path(path).stem.lower()
            return f":/themes/{theme_name}.qss"
        filename = Path(path).name
        return f":/Resources/{filename}"
        
    def _update_badge_display(self):
        if not hasattr(self, 'badge_label') or not self.badge_label:
            return
            
        # Store current badge index in case of failure
        current_badge_index = getattr(self, 'current_badge_index', 0)
        
        try:
            # Get the badge filename and path
            badge_filename = self.badge_filenames[current_badge_index]
            badge_path = self._get_resource_path(badge_filename)
            
            # Load the pixmap with error handling
            pixmap = QPixmap()
            if not pixmap.load(badge_path):
                # Try alternative loading method if direct load fails
                pixmap = QPixmap(badge_path.replace(':/', ''))
                
            if pixmap.isNull():
                logger.warning(f"Failed to load badge from: {badge_path}")
                self.badge_label.setText("Badge N/A")
                if hasattr(self, 'border_overlay') and self.border_overlay:
                    self.border_overlay.hide()
                return
                
            # Get screen information for DPI scaling
            app = QApplication.instance()
            screen = self.screen() or (app.primaryScreen() if app else None)
            dpr = screen.devicePixelRatio() if screen else 1.0
            
            # Calculate scaled size
            max_size = 240
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
                
            # Set up the badge label
            scaled_pixmap.setDevicePixelRatio(dpr)
            self.badge_label.setPixmap(scaled_pixmap)
            
            # Update border overlay if it exists
            if hasattr(self, 'border_overlay') and self.border_overlay:
                self.border_overlay.show()
                self.border_overlay.raise_()
                
            # Set size policies and constraints
            self.badge_label.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Expanding
            )
            
            # Calculate and set logical size for proper DPI scaling
            logical_size = scaled_pixmap.size() / dpr
            self.badge_label.setMinimumSize(logical_size)
            self.badge_label.setMaximumSize(logical_size)
            
            # Update container sizes if they exist
            badge_container = self.badge_label.parent()
            if badge_container:
                badge_container.setMinimumSize(logical_size)
                badge_container.setMaximumSize(logical_size)
                badge_outer_container = badge_container.parent()
                if badge_outer_container:
                    badge_outer_container.setMinimumWidth(int(logical_size.width()))
                    badge_outer_container.setMinimumHeight(int(logical_size.height()))
                    badge_outer_container.setMaximumWidth(int(logical_size.width()))
                    badge_outer_container.setMaximumHeight(int(logical_size.height()))
            
            # Force UI update
            self.badge_label.update()
            self.updateGeometry()
            
            # Process any pending events to ensure UI updates
            QApplication.processEvents()
            
        except Exception as e:
            logger.error(f"Error updating badge display: {e}", exc_info=True)
            if hasattr(self, 'badge_label'):
                self.badge_label.setText("Badge Error")
            # Try to recover by selecting a different badge
            if hasattr(self, 'badge_filenames') and len(self.badge_filenames) > 1:
                self.current_badge_index = (current_badge_index + 1) % len(self.badge_filenames)
                QTimer.singleShot(100, self._update_badge_display)
    
    def _select_random_badge(self):
        if len(self.badge_filenames) <= 1:
            return
        new_index = random.choice([i for i in range(len(self.badge_filenames)) if i != self.current_badge_index])
        self.current_badge_index = new_index
        self._update_badge_display()
        if self.app_instance and self.app_instance.settings:
            self.app_instance.settings.setValue("UI/current_badge_index", str(self.current_badge_index))
    
    def setup_drag(self):
        self.drag_state = {
            'dragging': False,
            'drag_start_pos': None,
            'window_start_pos': None,
            'edge': None
        }
        self.main_widget.installEventFilter(self)
        self.main_widget.setMouseTracking(True)
        if hasattr(self, 'title_bar'):
            self.title_bar.mousePressEvent = self.title_mouse_press
            self.title_bar.mouseMoveEvent = self.title_mouse_move
            self.title_bar.mouseReleaseEvent = self.title_mouse_release
            self.title_bar.setMouseTracking(True)

    def _get_physical_monitor_info(self, screen: QScreen) -> Dict[str, Any]:
        try:
            monitor_info = get_physical_monitor_info(screen)
            if not monitor_info:
                raise ValueError("No monitor info returned from get_physical_monitor_info")
            geo = screen.geometry()
            return {
                'physical_width': monitor_info.get('physical_width', int(geo.width() * monitor_info.get('scale_factor', 1.0))),
                'physical_height': monitor_info.get('physical_height', int(geo.height() * monitor_info.get('scale_factor', 1.0))),
                'position': monitor_info.get('position', QPoint(geo.x(), geo.y())),
                'work_area': monitor_info.get('work_area', screen.availableGeometry()),
                'primary': monitor_info.get('is_primary', screen == QApplication.primaryScreen()),
                'monitor_rect': monitor_info.get('rect', geo),
                'dpi': monitor_info.get('dpi', QSizeF(screen.logicalDotsPerInch(), screen.logicalDotsPerInchY())),
                'scale_factor': monitor_info.get('scale_factor', 1.0),
                'device_name': monitor_info.get('device_name', screen.name() if hasattr(screen, 'name') else 'Unknown')
            }
        except Exception as e:
            logger.warning(f"Error getting physical monitor info: {e}")
            geo = screen.geometry()
            monitor_info = get_physical_monitor_info(screen)
            scale_factor = monitor_info.get('scale_factor', 1.0)
            return {
                'physical_width': int(geo.width() * scale_factor),
                'physical_height': int(geo.height() * scale_factor),
                'position': QPoint(geo.x(), geo.y()),
                'work_area': screen.availableGeometry(),
                'primary': screen == QApplication.primaryScreen(),
                'monitor_rect': geo,
                'dpi': QSizeF(screen.logicalDotsPerInch(), screen.logicalDotsPerInchY()),
                'scale_factor': scale_factor,
                'device_name': screen.name() if hasattr(screen, 'name') else 'Unknown'
            }
    
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
        try:
            if not qscreen_obj or not hasattr(qscreen_obj, 'geometry'):
                logger.error("Invalid screen object received from picker")
                return
            logger.info("Monitor selected from picker:")
            geo = qscreen_obj.geometry()
            logger.info(f"  Name: {qscreen_obj.name() if hasattr(qscreen_obj, 'name') else 'N/A'}")
            logger.info(f"  Geometry: {geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
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
                self.current_selected_qscreen = screens[matching_index]
                for i in range(self.monitor_combo.count()):
                    screen = self.monitor_combo.itemData(i, Qt.ItemDataRole.UserRole)
                    if screen and hasattr(screen, 'geometry'):
                        s_geo = screen.geometry()
                        if (s_geo.x() == geo.x() and s_geo.y() == geo.y() and
                            s_geo.width() == geo.width() and s_geo.height() == geo.height()):
                            self.monitor_combo.setCurrentIndex(i)
                            logger.info(f"  Updated combo box selection to index {i}")
                            break
                self.accept_monitor()
            else:
                logger.warning("Selected screen not found in current screens list")
                QMessageBox.warning(self, "Error", "Selected screen is no longer available.")
        except Exception as e:
            logger.error(f"Error in on_monitor_selected: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to select monitor: {str(e)}")
            self.accept_monitor()

    def start_monitor_picker(self):
        try:
            if self.picker_active:
                logger.warning("Monitor picker is already active")
                return
            logger.info("Starting monitor picker...")
            screens = QApplication.instance().screens()
            logger.info(f"Current screens available: {len(screens)}")
            for i, screen in enumerate(screens):
                geo = screen.geometry()
                logger.info(f"  Screen {i}: {screen.name() if hasattr(screen, 'name') else 'N/A'} - "
                          f"{geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
            self.picker = CrosshairPicker(mode="monitor")
            self.picker.monitor_selected.connect(self.on_monitor_selected)
            self.picker.destroyed.connect(self._on_picker_destroyed)
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
            index = self.monitor_combo.currentIndex()
            if index < 0:
                logger.warning("No monitor selected; falling back to primary screen")
                qscreen_obj = QGuiApplication.primaryScreen()
            else:
                qscreen_obj = self.monitor_combo.itemData(index, Qt.ItemDataRole.UserRole)
                if not isinstance(qscreen_obj, QScreen):
                    logger.warning(f"Invalid screen object at index {index}; falling back to primary screen")
                    qscreen_obj = QGuiApplication.primaryScreen()
            if not qscreen_obj:
                logger.error("No screens available")
                QMessageBox.critical(self, "Error", "No monitors available.")
                return
            all_screens = QGuiApplication.screens()
            screen_found = False
            for i, screen in enumerate(all_screens):
                if (screen.name() == qscreen_obj.name() and 
                    screen.geometry() == qscreen_obj.geometry()):
                    qscreen_obj = screen
                    screen_found = True
                    break
            if not screen_found:
                logger.warning("Selected screen not found in current screen list, using primary screen")
                qscreen_obj = QGuiApplication.primaryScreen()
            screen_name = qscreen_obj.name() if hasattr(qscreen_obj, 'name') else 'unnamed'
            geo = qscreen_obj.geometry()
            logger.info(f"Selected screen: {screen_name} - {geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
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
                self._about_dialog_instance.show()
                if hasattr(self.app_instance, 'current_theme'):
                    self._about_dialog_instance.apply_theme(self.app_instance.current_theme)
            else:
                self._about_dialog_instance.activateWindow()
                self._about_dialog_instance.raise_()
                if hasattr(self.app_instance, 'current_theme'):
                    self._about_dialog_instance.apply_theme(self.app_instance.current_theme)
    
    def keyPressEvent(self, event):
        """Handle key press events for the settings panel."""
        if event.key() == Qt.Key_Escape:
            logger.info("Escape key pressed - closing application")
            if hasattr(self, 'app_instance') and self.app_instance:
                if hasattr(self.app_instance, 'cleanup_and_quit'):
                    self.app_instance.cleanup_and_quit()
                elif hasattr(self.app_instance, 'quit'):
                    self.app_instance.quit()
                else:
                    QApplication.quit()
            else:
                QApplication.quit()
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if obj == self.badge_label and event.type() == QEvent.MouseButtonDblClick:
            self._select_random_badge()
            return True
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
        return super().eventFilter(obj, event)
    
    def snap_to_edge(self, pos=None):
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
        try:
            theme_name = getattr(self, 'theme', 'dark').lower()
            logger.debug(f"Loading combo styles for theme: {theme_name}")
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
            try:
                theme_qrc = f":/themes/{theme_name}.qss"
                theme_file = QFile(theme_qrc)
                if theme_file.open(QFile.ReadOnly | QFile.Text):
                    stream = QTextStream(theme_file)
                    content = stream.readAll()
                    theme_file.close()
                    if content:
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
            logger.warning("No QComboBox styles found in any source, using fallback styles")
            return self._get_fallback_combo_style(theme_name)
        except Exception as e:
            logger.error(f"Error loading combo styles: {e}", exc_info=True)
            return self._get_fallback_combo_style('dark')

    def _get_display_info(self, screen: QScreen, idx: int, monitor_info: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if monitor_info is None:
                monitor_info = self._get_physical_monitor_info(screen)
            display_name = monitor_info.get('device_name', '')
            if not display_name or display_name.startswith('\\\\'):
                display_name = f'Display {idx + 1}'
            width = monitor_info.get('physical_width', 0)
            height = monitor_info.get('physical_height', 0)
            position = monitor_info.get('position', QPoint(0, 0))
            x, y = position.x(), position.y()
            dpi = QSizeF(96, 96)
            if 'dpi' in monitor_info and isinstance(monitor_info['dpi'], (QSizeF, QSize)):
                dpi = monitor_info['dpi']
            elif 'dpi_x' in monitor_info and 'dpi_y' in monitor_info:
                try:
                    dpi = QSizeF(float(monitor_info['dpi_x']), float(monitor_info['dpi_y']))
                except (TypeError, ValueError):
                    pass
            scale_factor = float(monitor_info.get('scale_factor', 1.0)) * 100
            display_text = f"{display_name} ({width}x{height})"
            tooltip_lines = [
                f"Display: {display_name}",
                f"Resolution: {width}x{height}",
                f"Position: ({x}, {y})",
                f"DPI: {dpi.width():.1f}x{dpi.height():.1f}",
                f"Scale: {scale_factor:.0f}%"
            ]
            is_primary = monitor_info.get('primary', False) or monitor_info.get('is_primary', False)
            if is_primary:
                display_text += " (Primary)"
                tooltip_lines.append("Primary Monitor: Yes")
            device_name = monitor_info.get('device_name', '')
            if device_name and device_name != display_name:
                tooltip_lines.append(f"Device: {device_name}")
            return {
                'display_text': display_text,
                'tooltip': '\n'.join(tooltip_lines),
                'screen': screen,
                'physical_width': width,
                'physical_height': height,
                'position': QPoint(x, y),
                'scale_factor': scale_factor / 100.0,
                'dpi': dpi,
                'is_primary': is_primary
            }
        except Exception as e:
            logger.error(f"Error getting display info: {e}", exc_info=True)
            return {
                'display_text': f"Display {idx + 1} (Error)",
                'tooltip': f"Error getting display information: {e}",
                'screen': screen,
                'physical_width': 0,
                'physical_height': 0,
                'position': QPoint(0, 0),
                'scale_factor': 1.0,
                'dpi': QSizeF(96, 96),
                'is_primary': False
            }
    
    def _match_monitor_to_screen(self, monitor, qt_screens):
        """
        Match a monitor dictionary to the best Qt screen based on geometry and properties.
        
        Args:
            monitor: Dictionary containing monitor info (name, width, height, position, is_primary)
            qt_screens: List of QScreen objects to match against
            
        Returns:
            Tuple of (best_matching_screen, match_score) where score is between 0 and 1
        """
        best_match = None
        best_score = 0
        
        monitor_rect = QRect(
            monitor.get('position', QPoint(0, 0)),
            QSize(monitor.get('width', 0), monitor.get('height', 0))
        )
        
        for screen in qt_screens:
            screen_geo = screen.geometry()
            
            # Calculate geometric overlap
            intersection = screen_geo.intersected(monitor_rect)
            if not intersection.isValid():
                continue
                
            # Calculate overlap area as a percentage of both screens
            overlap_area = intersection.width() * intersection.height()
            monitor_area = monitor_rect.width() * monitor_rect.height()
            screen_area = screen_geo.width() * screen_geo.height()
            
            if overlap_area == 0 or monitor_area == 0 or screen_area == 0:
                continue
                
            # Calculate match score (0-1) based on overlap
            overlap_ratio = min(overlap_area / monitor_area, overlap_area / screen_area)
            
            # Bonus for primary screen match
            if monitor.get('is_primary') and screen == QGuiApplication.primaryScreen():
                overlap_ratio = min(1.0, overlap_ratio + 0.2)
                
            if overlap_ratio > best_score:
                best_score = overlap_ratio
                best_match = screen
        
        return best_match, best_score
        
    def load_monitors(self):
        if not hasattr(self, 'monitor_combo'):
            logger.warning("monitor_combo not initialized, skipping monitor load")
            return
        current_screen = None
        if self.monitor_combo.currentIndex() >= 0:
            current_screen = self.monitor_combo.currentData()
        try:
            monitors = get_all_monitors()
            logger.info(f"Loading {len(monitors)} available monitors...")
            self.monitor_combo.clear()
            self._screen_info = []
            qt_screens = QApplication.screens()
            matched_screens = set()
            monitor_screen_map = []
            for monitor in monitors:
                screen, score = self._match_monitor_to_screen(monitor, qt_screens)
                if screen:
                    matched_screens.add(screen)
                    monitor_screen_map.append((monitor, screen, score))
            for screen in qt_screens:
                if screen not in matched_screens:
                    logger.warning(f"No monitor matched for Qt screen: {screen.name()}")
                    monitor = {
                        'name': screen.name(),
                        'width': screen.geometry().width(),
                        'height': screen.geometry().height(),
                        'position': screen.geometry().topLeft(),
                        'is_primary': screen == QApplication.primaryScreen(),
                        'screen_object': screen
                    }
                    monitor_screen_map.append((monitor, screen, 0.5))
            monitor_screen_map.sort(key=lambda x: (x[0]['position'].x(), x[0]['position'].y()))
            for i, (monitor, screen, score) in enumerate(monitor_screen_map):
                try:
                    display_info = self._get_display_info(screen, i, monitor_info=monitor)
                    if not display_info:
                        continue
                    self.monitor_combo.addItem(display_info['display_text'], screen)
                    self.monitor_combo.setItemData(self.monitor_combo.count() - 1, 
                                                 display_info['tooltip'], Qt.ToolTipRole)
                    self._screen_info.append({
                        'screen': screen,
                        'monitor_info': monitor,
                        'is_primary': monitor.get('is_primary', False)
                    })
                    logger.info(f"Added monitor {i}: {display_info['display_text']} (match confidence: {score:.1f})")
                except Exception as e:
                    logger.error(f"Error adding monitor {i} to combo box: {e}", exc_info=True)
            if current_screen:
                for i in range(self.monitor_combo.count()):
                    if self.monitor_combo.itemData(i) == current_screen:
                        self.monitor_combo.setCurrentIndex(i)
                        break
            if self.monitor_combo.count() > 0 and self.monitor_combo.currentIndex() < 0:
                self.monitor_combo.setCurrentIndex(0)
            if self.monitor_combo.count() == 0:
                logger.warning("No monitors available in combo box")
        except Exception as e:
            logger.critical(f"Failed to load monitors: {e}", exc_info=True)
            try:
                self.monitor_combo.clear()
                screens = QApplication.screens()
                for i, screen in enumerate(screens):
                    display_text = f"Display {i+1}"
                    if screen == QApplication.primaryScreen():
                        display_text += " (Primary)"
                    self.monitor_combo.addItem(display_text, screen)
                if screens:
                    self.monitor_combo.setCurrentIndex(0)
            except Exception as fallback_error:
                logger.critical(f"Fallback monitor loading also failed: {fallback_error}")
            if hasattr(self, '_on_monitor_selected'):
                self._on_monitor_selected(self.monitor_combo.currentIndex())
                if current_screen and hasattr(QApplication, 'screens') and current_screen in QApplication.screens():
                    for i, screen_data in enumerate(self._screen_info):
                        if screen_data['screen'] == current_screen:
                            self.monitor_combo.setCurrentIndex(i)
                            logger.info(f"Restored selection to monitor {i + 1}")
                            break
                elif hasattr(QApplication, 'primaryScreen') and self.monitor_combo.count() > 0:
                    primary_screen = QApplication.primaryScreen()
                    if primary_screen:
                        for i, screen_data in enumerate(self._screen_info):
                            if screen_data['screen'] == primary_screen:
                                self.monitor_combo.setCurrentIndex(i)
                                logger.info(f"Set primary screen as default: {primary_screen.name() if hasattr(primary_screen, 'name') else 'N/A'}")
                                break
                    if self.monitor_combo.currentIndex() < 0 and self.monitor_combo.count() > 0:
                        self.monitor_combo.setCurrentIndex(0)
                        logger.info("Set first screen as default (primary not found)")
                if self.monitor_combo.currentIndex() >= 0:
                    self.current_selected_qscreen = self.monitor_combo.currentData(Qt.ItemDataRole.UserRole)
                    logger.info(f"Monitor combo box populated with {self.monitor_combo.count()} items, current: {self.monitor_combo.currentText()}")
                else:
                    logger.warning("No monitors available in combo box")
        except Exception as e:
            logger.error(f"Error loading monitors: {e}", exc_info=True)
            self.monitor_combo.clear()
            qt_screens = QApplication.screens()
            for i, screen in enumerate(qt_screens):
                self.monitor_combo.addItem(f"Display {i + 1}", screen)
            if qt_screens:
                self.monitor_combo.setCurrentIndex(0)
                self.current_selected_qscreen = qt_screens[0]
    
    def _on_monitor_selected(self, index):
        try:
            if index < 0 or index >= self.monitor_combo.count():
                logger.warning(f"Invalid monitor index selected: {index}")
                return
            screen = self.monitor_combo.itemData(index, Qt.ItemDataRole.UserRole)
            if not screen or not hasattr(screen, 'geometry'):
                logger.error(f"Invalid screen object at index {index}")
                return
            screen_idx = self.monitor_combo.itemData(index, Qt.ItemDataRole.UserRole + 1)
            geo = screen.geometry()
            logger.info(f"Selected monitor {index} (stored index: {screen_idx}):")
            logger.info(f"  Name: {screen.name() if hasattr(screen, 'name') else 'N/A'}")
            logger.info(f"  Geometry: {geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
            self.current_selected_qscreen = screen
            if 0 <= index < len(self._screen_info):
                logger.info(f"  Display name: {self._screen_info[index].get('display_text', 'N/A')}")
            logger.info(f"Current screen reference updated for index {index}")
        except Exception as e:
            logger.error(f"Error in _on_monitor_selected: {e}", exc_info=True)
            
    def apply_theme(self, theme_name, from_global=False):
        try:
            theme_name = str(theme_name).strip().lower()
            logger.info(f"Applying theme: {theme_name} (from_global={from_global})")
            
            # Update theme name and store in instance
            self.theme = theme_name
            
            # Load main theme stylesheet
            theme_qrc_path = f":/themes/{theme_name}.qss"
            logger.debug(f"Loading theme from QRC: {theme_qrc_path}")
            
            stylesheet = ""
            try:
                theme_file = QFile(theme_qrc_path)
                if not theme_file.open(QFile.ReadOnly | QFile.Text):
                    raise IOError(f"Failed to open QRC resource: {theme_qrc_path}")
                
                stream = QTextStream(theme_file)
                stylesheet = stream.readAll()
                theme_file.close()
                
                if not stylesheet.strip():
                    logger.warning(f"Empty stylesheet loaded from {theme_qrc_path}")
                    raise ValueError("Empty stylesheet content")
                    
                # Fix resource paths
                stylesheet = stylesheet.replace('url(Resources/', 'url(:/Resources/')
                stylesheet = stylesheet.replace('url("Resources/', 'url(":/Resources/')
                
                # Apply the stylesheet
                self.setStyleSheet(stylesheet)
                logger.info(f"Successfully applied {theme_name} theme styles")
                
            except Exception as e:
                logger.error(f"Failed to load theme file {theme_qrc_path}: {e}")
                if theme_name != 'dark':
                    logger.info("Falling back to dark theme")
                    self.apply_theme('Dark')
                    return
                raise
            
            # Load and apply combo box styles
            combo_style_qrc_path = ":/themes/combo_styles.qss"
            combo_styles = ""
            
            try:
                combo_file = QFile(combo_style_qrc_path)
                if combo_file.open(QFile.ReadOnly | QFile.Text):
                    stream = QTextStream(combo_file)
                    combo_styles = stream.readAll()
                    combo_file.close()
                    
                    if combo_styles.strip():
                        combo_styles = combo_styles.replace('url(Resources/', 'url(:/Resources/')
                        combo_styles = combo_styles.replace('url("Resources/', 'url(":/Resources/')
                        logger.debug("Successfully loaded combo box styles from QRC")
                    else:
                        logger.warning("Empty combo box styles loaded from QRC")
                        combo_styles = self._get_fallback_combo_style(theme_name)
                else:
                    raise IOError(f"Failed to open QRC resource: {combo_style_qrc_path}")
                    
            except Exception as e:
                logger.warning(f"Could not load combo box styles from QRC: {e}")
                combo_styles = self._get_fallback_combo_style(theme_name)
            
            # Apply combo box styles
            for combo_attr in ['window_combo', 'monitor_combo']:
                if hasattr(self, combo_attr):
                    try:
                        combo = getattr(self, combo_attr)
                        if not isinstance(combo.view(), QListView):
                            combo.setView(QListView())
                        
                        # Apply styles and update
                        combo.setStyleSheet(combo_styles)
                        combo.setProperty('class', 'light' if theme_name == 'light' else '')
                        
                        # Force style update
                        combo.style().unpolish(combo)
                        combo.style().polish(combo)
                        combo.update()
                        
                        logger.debug(f"Updated styles for {combo_attr}")
                    except Exception as e:
                        logger.error(f"Error updating {combo_attr}: {e}", exc_info=True)
            
            # Update badge display
            if hasattr(self, 'badge_label') and self.badge_label:
                QTimer.singleShot(100, self._update_badge_display)
                logger.debug("Scheduled badge display update")
            
            # Force UI update
            self.update()
            logger.info(f"Successfully applied {theme_name} theme{' (from global)' if from_global else ''}")
            
        except Exception as e:
            logger.critical(f"Critical error applying theme {theme_name}: {e}", exc_info=True)
            try:
                if theme_name != 'dark':
                    logger.info("Attempting fallback to dark theme")
                    self.apply_theme('Dark')
                else:
                    logger.warning("Already on dark theme, applying minimal styles")
                    self._apply_minimal_styles()
            except Exception:
                logger.critical("Fallback theming failed, UI may be unstyled", exc_info=True)
                self._apply_minimal_styles()
    
    def _apply_minimal_styles(self):
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
        logger.info("Quit application requested from settings panel")
        if hasattr(self, 'app_instance') and self.app_instance:
            self.app_instance.quit()
        else:
            self.close()
            
    def title_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_state = handle_overlay_mouse_press(event, self)
            if not self.drag_state.get('is_resizing', False):
                event.accept()
            else:
                super().mousePressEvent(event)
            
    def title_mouse_move(self, event):
        if hasattr(self, 'drag_state'):
            handled = handle_overlay_mouse_move(event, self, self.drag_state)
            if handled:
                event.accept()
                return
        super().mouseMoveEvent(event)
            
    def title_mouse_release(self, event):
        if hasattr(self, 'drag_state') and event.button() == Qt.LeftButton:
            if handle_overlay_mouse_release(event, self, self.drag_state):
                event.accept()
                return
        super().mouseReleaseEvent(event)
    
    def setup_ui(self):
        self.setWindowTitle("SHITTY PiP QUICKSWAP")
        self.setFixedSize(800, 300)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.main_widget = QWidget()
        self.main_widget.setObjectName("mainWidget")
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.title_bar = QWidget()
        self.title_bar.setObjectName("titleBar")
        self.title_bar.setFixedHeight(60)
        title_bar_layout = QHBoxLayout(self.title_bar)
        title_bar_layout.setContentsMargins(30, 10, 30, 10)  # Completed with symmetric margins
        title_bar_layout.setSpacing(10)
        title_frame = QFrame()
        title_frame.setObjectName("titleFrame")
        title_frame.setFixedHeight(40)
        title_layout = QHBoxLayout(title_frame)
        title_layout.setContentsMargins(10, 0, 10, 0)
        title_label = QLabel("SHITTY PiP QUICKSWAP")
        title_label.setObjectName("titleLabel")
        title_layout.addWidget(title_label, 0, Qt.AlignCenter)
        title_bar_layout.addWidget(title_frame, 1)
        title_bar_layout.addSpacing(20)
        close_button_container = QWidget()
        close_button_container.setObjectName("closeButtonContainer")
        close_button_layout = QVBoxLayout(close_button_container)
        close_button_layout.setContentsMargins(0, 6, 0, 0)
        close_button_layout.setSpacing(0)
        close_button_layout.addStretch()
        self.close_button = QPushButton()
        self.close_button.setObjectName("closeButton")
        self.close_button.setFixedSize(24, 24)
        self.close_button.clicked.connect(self.quit_application)
        self.close_button.setText("")
        self.close_button.setToolTip("Quit")
        close_button_layout.addWidget(self.close_button)
        close_button_layout.addStretch()
        title_bar_layout.addWidget(close_button_container, 0, Qt.AlignTop | Qt.AlignRight)
        self.title_bar_container = QWidget()
        self.title_bar_container.setObjectName("titleBarContainer")
        title_container_layout = QVBoxLayout(self.title_bar_container)
        title_container_layout.setContentsMargins(0, 0, 0, 0)
        title_container_layout.setSpacing(0)
        title_container_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.title_bar_container)
        content_widget = QWidget()
        content_widget.setObjectName("contentWidget")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)
        main_content = QHBoxLayout()
        main_content.setContentsMargins(0, 0, 0, 0)
        main_content.setSpacing(0)
        left_side = QWidget()
        left_side.setObjectName("leftSide")
        left_layout = QVBoxLayout(left_side)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        window_row = QHBoxLayout()
        window_row.setContentsMargins(0, 0, 0, 0)
        window_row.setSpacing(9)
        window_row.setAlignment(Qt.AlignLeft)
        self.window_btn = QPushButton("SELECT WINDOW")
        self.window_btn.setObjectName("selectButton")
        self.window_btn.clicked.connect(self.start_window_picker)
        self.window_btn.setFixedSize(108, 32)
        self.window_combo = QComboBox()
        self.window_combo.setView(QListView())
        self.window_combo.view().window().setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.window_combo.view().window().setAttribute(Qt.WA_TranslucentBackground)
        self.window_combo.setObjectName("windowCombo")
        self.window_combo.setFixedSize(200, 32)
        view = self.window_combo.view()
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.window_combo.setStyleSheet(self.get_combo_style())
        self.window_start_btn = QPushButton("START")
        self.window_start_btn.setObjectName("startButton")
        self.window_start_btn.clicked.connect(self.accept_window)
        self.window_start_btn.setFixedSize(56, 32)
        window_row.addWidget(self.window_btn)
        window_row.addSpacing(4)
        window_row.addWidget(self.window_combo)
        window_row.addSpacing(4)
        window_row.addWidget(self.window_start_btn)
        window_row.addStretch()
        left_layout.addLayout(window_row)
        monitor_row = QHBoxLayout()
        monitor_row.setContentsMargins(0, 0, 0, 0)
        monitor_row.setSpacing(9)
        monitor_row.setAlignment(Qt.AlignLeft)
        self.monitor_btn = QPushButton("SELECT MONITOR")
        self.monitor_btn.setObjectName("selectButton")
        self.monitor_btn.clicked.connect(self.start_monitor_picker)
        self.monitor_btn.setFixedSize(108, 32)
        self.monitor_combo = QComboBox()
        self.monitor_combo.setView(QListView())
        self.monitor_combo.view().window().setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.monitor_combo.view().window().setAttribute(Qt.WA_TranslucentBackground)
        self.monitor_combo.setObjectName("monitorCombo")
        self.monitor_combo.setFixedSize(200, 32)
        view = self.monitor_combo.view()
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.monitor_combo.setStyleSheet(self.get_combo_style())
        self.monitor_start_btn = QPushButton("START")
        self.monitor_start_btn.setObjectName("startButton")
        self.monitor_start_btn.clicked.connect(self.accept_monitor)
        self.monitor_start_btn.setFixedSize(56, 32)
        monitor_row.addWidget(self.monitor_btn)
        monitor_row.addSpacing(4)
        monitor_row.addWidget(self.monitor_combo)
        monitor_row.addSpacing(4)
        monitor_row.addWidget(self.monitor_start_btn)
        monitor_row.addStretch()
        left_layout.addLayout(monitor_row)
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        left_buttons = QHBoxLayout()
        left_buttons.setSpacing(0)
        self.minimize_btn = QPushButton("MINIMIZE TO TRAY")
        self.minimize_btn.setObjectName("actionButton")
        self.minimize_btn.clicked.connect(self.minimize_to_tray)
        self.minimize_btn.setFixedHeight(32)
        self.settings_btn = QPushButton("SUBSETTINGS")
        self.settings_btn.setObjectName("actionButton")
        self.settings_btn.clicked.connect(self.show_settings)
        self.settings_btn.setFixedSize(120, 32)
        left_buttons.addWidget(self.minimize_btn)
        left_buttons.addSpacing(8)
        left_buttons.addWidget(self.settings_btn)
        left_buttons.addSpacing(3)
        self.about_btn = QPushButton("?")
        self.about_btn.setObjectName("aboutButton")
        self.about_btn.clicked.connect(self.show_about_dialog)
        self.about_btn.setFixedSize(32, 32)
        about_container = QWidget()
        about_layout = QVBoxLayout(about_container)
        about_layout.setContentsMargins(0, 11, 0, 0)
        about_layout.addWidget(self.about_btn)
        left_buttons.addWidget(about_container)
        left_buttons.addStretch()
        button_row.addLayout(left_buttons)
        left_layout.addLayout(button_row)
        left_layout.addStretch()
        main_content.addWidget(left_side, 1)
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
        badge_layout = QGridLayout(badge_container)
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.setSpacing(0)
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
        self.badge_label.installEventFilter(self)
        badge_layout.addWidget(self.badge_label, 0, 0, Qt.AlignRight | Qt.AlignBottom)
        badge_outer_container = QWidget()
        badge_outer_container.setObjectName("badgeOuterContainer")
        badge_outer_layout = QVBoxLayout(badge_outer_container)
        badge_outer_layout.setContentsMargins(0, 0, 0, 0)
        badge_outer_layout.setSpacing(0)
        badge_outer_layout.addStretch()
        badge_outer_layout.addWidget(badge_container, 0, Qt.AlignRight | Qt.AlignBottom)
        main_content.addWidget(badge_outer_container, 1, Qt.AlignRight | Qt.AlignBottom)
        content_layout.addLayout(main_content)
        self.main_layout.addWidget(content_widget, 1)
        self.setup_drag()
        self.load_windows()
        self.load_monitors()
        self._update_badge_display()
        self.apply_theme(self.theme)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SettingsPanel()
    window.show()
    sys.exit(app.exec())
