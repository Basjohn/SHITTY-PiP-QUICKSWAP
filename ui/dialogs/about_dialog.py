"""
AboutDialog module
- Uses centralized QSS (no inline styles)
- Unique objectNames to avoid confusion with SubSettings
"""

import sys
from core.logging import get_logger
from PySide6.QtCore import Qt, QUrl, QEvent, QPoint
from PySide6.QtGui import QFont, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QFrame
)

# Resource paths using QRC
THEMES_DIR = ":/themes"
RESOURCES_DIR = ":/Resources"

logger = get_logger(__name__)

class AboutDialog(QDialog):
    """Dialog to display application information and links."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("aboutDialog")
        self.setWindowTitle("About Shitty PiP QuickSwap")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Initial size and center on current screen
        self.resize(350, 220)
        if self.screen():
            screen_geo = self.screen().availableGeometry()
            self.move(screen_geo.center() - self.rect().center())
        else:
            primary_screen = QGuiApplication.primaryScreen()
            if primary_screen:
                screen_geo = primary_screen.availableGeometry()
                self.move(screen_geo.center() - self.rect().center())

        self._setup_ui()

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
        border.setObjectName("aboutDialogBorder")
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

        self.title_label = QLabel("ABOUT SHITTY PiP QUICKSWAP", self.title_bar)
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
        content_frame.setObjectName("aboutContentFrame")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(16, 12, 16, 12)
        content_layout.setSpacing(6)

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

        links_layout = QHBoxLayout()
        links_layout.setSpacing(10)

        self.paypal_btn = QPushButton("PayPal")
        self.paypal_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(
            "https://www.paypal.com/donate/?business=UBZJY8KHKKLGC&no_recurring=0&item_name=Why+are+you+doing+this?+Are+you+drunk?+&currency_code=USD"
        )))

        self.goodreads_btn = QPushButton("Goodreads")
        self.goodreads_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(
            "https://www.goodreads.com/book/show/25006763-usu"
        )))

        self.amazon_btn = QPushButton("Amazon")
        self.amazon_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(
            "https://www.amazon.com/Usu-Jayde-Ver-Elst-ebook/dp/B00V8A5K7Y"
        )))

        links_layout.addStretch()
        links_layout.addWidget(self.paypal_btn)
        links_layout.addWidget(self.goodreads_btn)
        links_layout.addWidget(self.amazon_btn)
        links_layout.addStretch()

        content_layout.addLayout(links_layout)
        content_layout.addStretch()

        self.attribution_label = QLabel()
        self.attribution_label.setObjectName("attributionLabel")
        self.attribution_label.setOpenExternalLinks(True)
        self.attribution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.attribution_label)

        frame_layout.addWidget(content_frame)

    # Inline theming removed; styling comes from theme QSS via objectNames


    # Resource helpers removed; resources use centralized managers/QSS

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

if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dialog = AboutDialog()
    dialog.show()
    sys.exit(app.exec())