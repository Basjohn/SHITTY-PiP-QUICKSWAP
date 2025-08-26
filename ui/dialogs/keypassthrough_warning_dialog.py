from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QFrame, QHBoxLayout
from core.logging import get_logger

logger = get_logger(__name__)


class KeyPassthroughWarningDialog(QDialog):
    """One-time warning when enabling Keypassthrough.

    - Frameless, translucent background. No title bar.
    - Styled by QSS using objectNames similar to AboutDialog.
    - Centered text with an "I Understand" button and a secondary "No Way!" button.
    - Size roughly similar to About dialog, but a bit smaller.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("keypassthroughWarningDialog")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Initial size and center on same screen as parent (SubSettings)
        # Use a slightly wider baseline to reduce risk of text wrapping/cutoff
        self.resize(420, 220)
        self._center_on_parent_screen()

        self._setup_ui()

    def showEvent(self, event):
        """Ensure final geometry fits content and stays centered on the parent's screen.

        On some DPIs or font configurations, the automatic size calculation may
        change after the widget is shown. We adjust and recenter here to keep it
        placed correctly relative to the SubSettings dialog's display.
        """
        super().showEvent(event)
        try:
            # Expand to fit content but keep it compact; then recenter.
            self.adjustSize()
        except Exception as e:
            logger.warning(f"KeyPassthroughWarningDialog.adjustSize failed: {e}")
        self._center_on_parent_screen()

    def _center_on_parent_screen(self):
        try:
            screen = None
            if self.parent() is not None:
                # Try to center on the same screen as parent dialog
                screen = self.parent().screen()
            if screen is None:
                # Fallback to screen under parent position
                parent_pos = self.parent().mapToGlobal(self.parent().rect().center()) if self.parent() else self.pos()
                screen = QGuiApplication.screenAt(parent_pos)
            if screen is None:
                screen = QGuiApplication.primaryScreen()
            if screen:
                geo = screen.availableGeometry()
                # Use frameGeometry when available (after show) for more accurate centering
                rect_to_use = self.frameGeometry() if self.isVisible() else self.rect()
                self.move(geo.center() - rect_to_use.center())
        except Exception as e:
            logger.error(f"Failed to center KeyPassthroughWarningDialog: {e}")

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        border = QFrame(self)
        border.setObjectName("aboutDialogBorder")  # Reuse About styling border
        main_layout.addWidget(border)

        content_layout = QVBoxLayout(border)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)

        font_text = QFont("Segoe UI", 10)

        line1 = QLabel("This feature should not be used with online games that feature an anti-cheat or Gacha games.")
        line1.setWordWrap(True)
        line1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line1.setFont(font_text)
        content_layout.addWidget(line1)

        line2 = QLabel("They may see key forwarding incorrectly as a cheating attempt.")
        line2.setWordWrap(True)
        line2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line2.setFont(font_text)
        content_layout.addWidget(line2)

        # Buttons row
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(10)
        buttons_row.setContentsMargins(0, 6, 0, 0)

        reject_btn = QPushButton("No Way!")
        reject_btn.clicked.connect(self.reject)
        buttons_row.addWidget(reject_btn)

        ack_btn = QPushButton("I Understand")
        ack_btn.clicked.connect(self.accept)
        buttons_row.addWidget(ack_btn)

        # Center the buttons row
        row_container = QFrame()
        row_container.setLayout(buttons_row)
        content_layout.addWidget(row_container, alignment=Qt.AlignmentFlag.AlignCenter)

        # Stretch to keep content centered vertically if window grows
        content_layout.addStretch()
