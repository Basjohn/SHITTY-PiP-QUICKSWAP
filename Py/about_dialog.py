"""
AboutDialog module for the Shitty-PiP application.
Displays information about the application and links.
"""

import os
import sys
import logging
from pathlib import Path
from PySide6.QtCore import Qt, QUrl, QMargins, QEvent, QPoint, QFile, QTextStream
from PySide6.QtGui import QFont, QDesktopServices, QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QSpacerItem, QSizePolicy, QApplication, QFrame, QTextEdit
)

# Resource paths using QRC
THEMES_DIR = ":/themes"
RESOURCES_DIR = ":/Resources"

logger = logging.getLogger(__name__)

class AboutDialog(QDialog):
    """Dialog to display application information and links."""

    def __init__(self, parent=None, app_instance=None):
        super().__init__(parent)
        self.app_instance = app_instance


        self.setWindowTitle("About Shitty PiP QuickSwap")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Set initial size - reduced height
        self.resize(350, 220)  # Reduced height by ~30%
        if self.screen():
            screen_geo = self.screen().availableGeometry()
            self.move(screen_geo.center() - self.rect().center()) # Center on current screen
        else: # Fallback if screen() is None
            primary_screen = QGuiApplication.primaryScreen()
            if primary_screen:
                screen_geo = primary_screen.availableGeometry()
                self.move(screen_geo.center() - self.rect().center())


        self.setup_ui()
        self.load_settings_and_apply_theme() # Load theme from app_instance

        # Draggability attributes
        self.drag_position = None
        self.is_dragging = False
        self.snap_threshold = 20
        self.screen_margin = 5

        if hasattr(self, 'title_bar'):
            self.title_bar.installEventFilter(self)
        else:
            logger.warning("AboutDialog: title_bar not found for installing event filter.")


    def setup_ui(self):
        from PySide6.QtWidgets import QFrame
        
        # Root frame for all content
        self.frame = QFrame(self)
        self.frame.setObjectName("aboutFrame")
        
        # Set main layout for the dialog
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Set frame layout
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        
        # Add frame to main layout
        main_layout.addWidget(self.frame)
        
        # Title bar with drag support
        self.title_bar = QWidget(self.frame)
        self.title_bar.setObjectName("titleBar")
        self.title_bar.setFixedHeight(36)
        self.title_bar.setMouseTracking(True)
        
        # Title bar layout
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(12, 4, 4, 4)
        title_layout.setSpacing(8)
        
        # Title label - centered with alignment
        self.title_label = QLabel("ABOUT SHITTY PiP QUICKSWAP")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addStretch()  # Push title to center
        
        
        # Close button - adjusted position
        self.close_btn = QPushButton("×")
        self.close_btn.setObjectName("close_btn")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet("margin-left: 8px; margin-top: -1px;")
        self.close_btn.clicked.connect(self.close)
        
        # Add widgets to title bar
        title_layout.addWidget(self.title_label, 1)  # Stretch
        title_layout.addWidget(self.close_btn, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        # Add title bar to frame
        frame_layout.addWidget(self.title_bar)
        
        # Content area - reduced padding
        content = QWidget()
        content.setObjectName("contentArea")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 12, 16, 12)  # Reduced vertical padding
        content_layout.setSpacing(6)  # Reduced spacing
        
        # Main text
        font_text = QFont("Segoe UI", 10)
        line1 = QLabel("Made for my own shitty productivity, shared freely for yours.")
        line1.setFont(font_text)
        line1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line1.setWordWrap(True)
        content_layout.addWidget(line1)
        
        line2 = QLabel("You can always donate to my dumbass though or buy my shitty literature.")
        line2.setFont(font_text)
        line2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line2.setWordWrap(True)
        content_layout.addWidget(line2)
        
        # Buttons
        links_layout = QHBoxLayout()
        links_layout.setSpacing(10)
        
        self.paypal_btn = QPushButton("PayPal")
        self.paypal_btn.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://www.paypal.com/donate/?business=UBZJY8KHKKLGC&no_recurring=0&item_name=Why+are+you+doing+this?+Are+you+drunk?+&currency_code=USD")))
        
        self.goodreads_btn = QPushButton("Goodreads")
        self.goodreads_btn.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://www.goodreads.com/book/show/25006763-usu")))
        
        self.amazon_btn = QPushButton("Amazon")
        self.amazon_btn.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://www.amazon.com/Usu-Jayde-Ver-Elst-ebook/dp/B00V8A5K7Y")))
        
        links_layout.addStretch()
        links_layout.addWidget(self.paypal_btn)
        links_layout.addWidget(self.goodreads_btn)
        links_layout.addWidget(self.amazon_btn)
        links_layout.addStretch()
        
        content_layout.addLayout(links_layout)
        content_layout.addStretch()
        
        # Attribution
        self.attribution_label = QLabel()
        self.attribution_label.setObjectName("attributionLabel")
        self.attribution_label.setOpenExternalLinks(True)
        self.attribution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.attribution_label)
        
        # Add content to frame (must be done after all content is added)
        frame_layout.addWidget(content)

    def get_button_style(self, theme):
        # Basic button style, can be expanded
        if theme.lower() == "dark":
            return """
                QPushButton { background-color: #444; color: white; border: 1px solid #666; border-radius: 5px; padding: 8px 12px; }
                QPushButton:hover { background-color: #555; border-color: #777; }
                QPushButton:pressed { background-color: #666; }
            """
        else: # Light
            return """
                QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #bbb; border-radius: 5px; padding: 8px 12px; }
                QPushButton:hover { background-color: #d0d0d0; border-color: #aaa; }
                QPushButton:pressed { background-color: #c0c0c0; }
            """

    def get_theme_styles(self, theme):
        """Get the stylesheet for the specified theme."""
        theme = theme.lower()
        
        base_styles = """
            QDialog {
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
            
            #aboutFrame {
                border: 2px solid white;
                border-radius: 10px;
                background: rgba(51, 51, 51, 0.9);
                padding: 0px;
                margin: 0px;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            
            #titleBar {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1.8px solid white;
                border-bottom: 1.8px solid white;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                margin: 0;
                padding: 3px 12px 1px 12px;
            }
            
            #titleLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 15px;
                letter-spacing: 1px;
                padding: 0;
                margin: 0;
                text-transform: uppercase;
            }
            
            #close_btn {
                color: white;
                font-size: 20px;
                width: 28px;
                height: 28px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 0;
                margin: 0;
                padding-bottom: 2px;
            }
            
            #close_btn:hover {
                color: #cccccc;
            }
            
            #contentArea {
                background: rgba(51, 51, 51, 0.9);
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
                padding: 12px;
                margin: 0;
            }
            
            QLabel {
                color: #ffffff;
                font-size: 11px;
            }
            
            #attributionLabel {
                color: #aaaaaa;
                font-size: 10px;
                margin-top: 8px;
            }
            
            #attributionLabel a {
                color: #ffffff;
                text-decoration: none;
            }
            
            #attributionLabel a:hover {
                text-decoration: underline;
            }
            
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #666;
                border-radius: 5px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #555;
                border-color: #777;
            }
            
            QPushButton:pressed {
                background-color: #666;
            }
        """
        
        light_theme_overrides = """
            #aboutFrame {
                background: rgba(248, 248, 248, 0.9);
                border-color: #bbbbbb;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            
            #titleBar {
                background-color: #f0f0f0;
                border-color: #bbbbbb;
                color: #222222;
            }
            
            #titleLabel {
                color: #222222;
            }
            
            #close_btn {
                color: #222222;
            }
            
            #close_btn:hover {
                color: #555555;
            }
            
            #contentArea {
                background: rgba(248, 248, 248, 0.9);
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            
            QLabel {
                color: #222222;
            }
            
            #attributionLabel {
                color: #666666;
            }
            
            #attributionLabel a {
                color: #000000;
            }
            
            QPushButton {
                background-color: #e0e0e0;
                color: #222222;
                border-color: #bbbbbb;
            }
            
            QPushButton:hover {
                background-color: #d0d0d0;
                border-color: #999999;
            }
            
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
        """
        
        if theme == "light":
            return base_styles + "\n" + light_theme_overrides
        return base_styles

    def apply_theme(self, theme, from_global=False):
        """
        Apply the specified theme to the dialog.
        
        Args:
            theme (str): The name of the theme to apply (e.g., 'dark', 'light')
            from_global (bool): Whether the theme change was initiated from the main application
        """
        try:
            theme = theme.lower()
            self.theme = theme
            logger.debug(f"Applying theme: {theme}")
            
            # Get complete stylesheet for the theme
            stylesheet = self.get_theme_styles(theme)
            
            # Apply the complete stylesheet to the dialog
            self.setStyleSheet(stylesheet)
            
            # Update the frame's theme property for style inheritance
            if hasattr(self, 'frame'):
                self.frame.setProperty('theme', theme)
                self.frame.style().unpolish(self.frame)
                self.frame.style().polish(self.frame)
            
            # Force update of all child widgets
            self.update()
            
        except Exception as e:
            logger.error(f"Error applying theme {theme}: {e}", exc_info=True)
            # Fall back to dark theme if there's an error
            if theme != 'dark':
                self.apply_theme('dark')
        
        # Ensure the dialog is properly repainted
        self.repaint()

    def load_settings_and_apply_theme(self):
        """Load settings and apply theme."""
        try:
            if self.app_instance and hasattr(self.app_instance, 'settings'):
                current_theme = self.app_instance.settings.value("UI/theme", "Dark")
            else:
                current_theme = "Dark"
            
            self.apply_theme(current_theme)
        except Exception as e:
            logger.error(f"Error loading settings and applying theme: {e}")

    def _get_resource_path(self, relative_path):
        """Get the full path to a resource file."""
        # First check if the file exists in the current directory
        if os.path.exists(relative_path):
            return relative_path
            
        # Then check in the resources directory
        if relative_path.startswith("Resources/"):
            resource_path = os.path.join(RESOURCES_DIR, relative_path.replace("Resources/", ""))
        else:
            resource_path = os.path.join(RESOURCES_DIR, relative_path)
            
        if os.path.exists(resource_path):
            return resource_path
            
        # Then check in the themes directory
        if relative_path.startswith("themes/"):
            theme_path = os.path.join(THEMES_DIR, relative_path.replace("themes/", ""))
            if os.path.exists(theme_path):
                return theme_path
                
        # If not found, return the original path (will raise an error when used)
        return relative_path

    def _get_current_screen(self):
        """Get the current screen where the dialog is located."""
        try:
            center_point = self.geometry().center()
            current_screen = QGuiApplication.screenAt(center_point)
            if not current_screen:
                current_screen = QGuiApplication.screenAt(self.pos())
            if not current_screen:
                current_screen = QGuiApplication.primaryScreen()
            return current_screen
        except Exception as e:
            logger.error(f"Error getting current screen: {e}")
            return QGuiApplication.primaryScreen()

    def _apply_snap_to(self, pos, size, screen_geo):
        """Apply snap-to behavior for window edges."""
        if abs(pos.x() - screen_geo.left()) < self.snap_threshold:
            pos.setX(screen_geo.left())
        elif abs(pos.x() + size.width() - screen_geo.right()) < self.snap_threshold:
            pos.setX(screen_geo.right() - size.width())
        
        if abs(pos.y() - screen_geo.top()) < self.snap_threshold:
            pos.setY(screen_geo.top())
        elif abs(pos.y() + size.height() - screen_geo.bottom()) < self.snap_threshold:
            pos.setY(screen_geo.bottom() - size.height())
            
        return pos

    def _constrain_to_screen(self, pos, size, screen_geo):
        constrained_x = max(screen_geo.left() + self.screen_margin, min(pos.x(), screen_geo.right() - size.width() - self.screen_margin))
        constrained_y = max(screen_geo.top() + self.screen_margin, min(pos.y(), screen_geo.bottom() - size.height() - self.screen_margin))
        return QPoint(constrained_x, constrained_y)

    def eventFilter(self, obj, event):
        if obj == self.title_bar and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                child_widget = self.title_bar.childAt(event.position().toPoint())
                if child_widget and child_widget != self.title_label and child_widget != self.title_bar:
                    return False 
                self.is_dragging = True
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
        return super().eventFilter(obj, event)

    def mouseMoveEvent(self, event):
        if self.is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            if not self.drag_position:
                self.is_dragging = False 
                return
            new_pos_global = event.globalPosition().toPoint()
            new_pos_local = new_pos_global - self.drag_position
            current_screen = self._get_current_screen()
            if not current_screen:
                self.move(new_pos_local)
                return
            screen_geo = current_screen.availableGeometry()
            dialog_size = self.size()
            snapped_pos = self._apply_snap_to(QPoint(new_pos_local), dialog_size, screen_geo)
            final_pos = self._constrain_to_screen(snapped_pos, dialog_size, screen_geo)
            if self.pos() != final_pos: self.move(final_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_dragging and event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            current_screen = self._get_current_screen()
            if current_screen:
                screen_geo = current_screen.availableGeometry()
                dialog_size = self.size()
                current_pos = self.pos()
                snapped_pos = self._apply_snap_to(QPoint(current_pos), dialog_size, screen_geo)
                final_pos = self._constrain_to_screen(snapped_pos, dialog_size, screen_geo)
                if self.pos() != final_pos: self.move(final_pos)
            self.drag_position = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

if __name__ == '__main__':
    # Example usage:
    import sys
    from PySide6.QtWidgets import QApplication
    
    # Mock app_instance and settings for testing
    class MockAppInstance:
        def __init__(self):
            self.settings = self.MockSettings()

        class MockSettings:
            def value(self, key, default=None, type=None):
                if key == "theme":
                    return "Dark" # or "Light"
                return default
            def setValue(self, key, value): pass
            def sync(self): pass
            def beginGroup(self, group): pass
            def endGroup(self): pass
            def childGroups(self): return []


    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    
    # For testing, create a mock app_instance
    mock_app = MockAppInstance()

    dialog = AboutDialog(app_instance=mock_app)
    dialog.show()
    sys.exit(app.exec())
