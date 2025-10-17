"""
InstanceRunningDialog module
- Displays a styled message when another instance is already running
- Uses centralized QSS (no inline styles)
- Similar styling to AboutDialog for consistency
"""

from core.logging import get_logger
from PySide6.QtCore import Qt, QEvent, QPoint
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QFrame
)

logger = get_logger(__name__)

class InstanceRunningDialog(QDialog):
    """Dialog to display when another instance is already running."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("instanceRunningDialog")
        self.setWindowTitle("SPQ Already Running")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Initial size and center on current screen
        self.resize(400, 200)
        if self.screen():
            screen_geo = self.screen().availableGeometry()
            self.move(screen_geo.center() - self.rect().center())
        else:
            primary_screen = QGuiApplication.primaryScreen()
            if primary_screen:
                screen_geo = primary_screen.availableGeometry()
                self.move(screen_geo.center() - self.rect().center())

        self._setup_ui()
        self._apply_dark_styling()  # Programmatic styling (ThemeManager not init yet)

        # Drag behavior
        self.drag_position = None
        self.is_dragging = False
        self.snap_threshold = 20
        self.screen_margin = 5
        self.title_bar.installEventFilter(self)

    def _setup_ui(self):
        # Root layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Outer border frame (styled by QSS)
        border = QFrame(self)
        border.setObjectName("instanceDialogBorder")
        main_layout.addWidget(border)

        frame_layout = QVBoxLayout(border)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        # Title frame
        self.title_bar = QWidget(border)
        self.title_bar.setObjectName("titleFrame")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(8, 6, 6, 6)
        title_layout.setSpacing(8)

        self.title_label = QLabel("ALREADY RUNNING", self.title_bar)
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.close_btn = QPushButton("X", self.title_bar)
        self.close_btn.setObjectName("closeButton")
        self.close_btn.clicked.connect(self.close)

        title_layout.addStretch()
        title_layout.addWidget(self.title_label, 1)
        title_layout.addWidget(self.close_btn, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        frame_layout.addWidget(self.title_bar)

        # Content frame
        content_frame = QFrame(border)
        content_frame.setObjectName("instanceContentFrame")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(20, 16, 20, 16)
        content_layout.setSpacing(12)

        font_title = QFont("Segoe UI", 11, QFont.Weight.Bold)
        font_text = QFont("Segoe UI", 10)

        # Main message
        main_msg = QLabel("Another instance of SPQ is already running.")
        main_msg.setFont(font_title)
        main_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_msg.setWordWrap(True)
        content_layout.addWidget(main_msg)

        # Instructions
        instructions = QLabel(
            "Only one instance can run at a time to prevent weird crazy shit.\n\n"
            "Close the existing instance from the system tray or right-click menu before starting a new one."
        )
        instructions.setFont(font_text)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setWordWrap(True)
        content_layout.addWidget(instructions)

        content_layout.addStretch()

        # OK button
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.ok_btn = QPushButton("GOD DAMNIT")
        self.ok_btn.setObjectName("instanceOkButton")
        self.ok_btn.setMinimumWidth(100)
        self.ok_btn.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addStretch()

        content_layout.addLayout(button_layout)

        frame_layout.addWidget(content_frame)

    def _apply_dark_styling(self):
        """Apply programmatic dark styling (ThemeManager not initialized yet)."""
        # Dark theme colors matching our QSS theme
        bg_main = "#1e1e1e"
        bg_title = "#252525"
        bg_content = "#2d2d2d"
        fg_text = "#e0e0e0"
        fg_title = "#ffffff"
        accent = "#007acc"
        accent_hover = "#1e88e5"
        border_color = "#3c3c3c"
        
        # Main dialog background
        self.setStyleSheet("""
            QDialog#instanceRunningDialog {
                background-color: transparent;
            }
        """)
        
        # Border frame
        border = self.findChild(QFrame, "instanceDialogBorder")
        if border:
            border.setStyleSheet(f"""
                QFrame#instanceDialogBorder {{
                    background-color: {bg_main};
                    border: 2px solid {border_color};
                    border-radius: 8px;
                }}
            """)
        
        # Title bar
        if self.title_bar:
            self.title_bar.setStyleSheet(f"""
                QWidget#titleFrame {{
                    background-color: {bg_title};
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    border-bottom: 1px solid {border_color};
                }}
                QLabel#titleLabel {{
                    color: {fg_title};
                    font-weight: bold;
                    font-size: 11pt;
                    background: transparent;
                }}
            """)
        
        # Close button
        if self.close_btn:
            self.close_btn.setStyleSheet(f"""
                QPushButton#closeButton {{
                    background-color: transparent;
                    color: {fg_text};
                    border: none;
                    border-radius: 4px;
                    font-size: 12pt;
                    font-weight: bold;
                    padding: 2px 8px;
                    min-width: 24px;
                    min-height: 24px;
                }}
                QPushButton#closeButton:hover {{
                    background-color: #e74c3c;
                    color: white;
                }}
                QPushButton#closeButton:pressed {{
                    background-color: #c0392b;
                }}
            """)
        
        # Content frame
        content_frame = self.findChild(QFrame, "instanceContentFrame")
        if content_frame:
            content_frame.setStyleSheet(f"""
                QFrame#instanceContentFrame {{
                    background-color: {bg_content};
                    border-bottom-left-radius: 6px;
                    border-bottom-right-radius: 6px;
                }}
                QLabel {{
                    color: {fg_text};
                    background: transparent;
                }}
            """)
        
        # OK button
        if self.ok_btn:
            self.ok_btn.setStyleSheet(f"""
                QPushButton#instanceOkButton {{
                    background-color: {accent};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 10pt;
                    font-weight: bold;
                }}
                QPushButton#instanceOkButton:hover {{
                    background-color: {accent_hover};
                }}
                QPushButton#instanceOkButton:pressed {{
                    background-color: #1565c0;
                }}
            """)

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
            if self.pos() != final_pos:
                self.move(final_pos)
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
                if self.pos() != final_pos:
                    self.move(final_pos)
            self.drag_position = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
