import logging
from debug_utils import debug_enabled
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel, 
                              QLineEdit, QComboBox, QPushButton, QCheckBox, 
                              QKeySequenceEdit, QSlider, QFrame)
from PySide6.QtGui import QGuiApplication, QFont, QKeySequence, QIntValidator, QPalette, QColor, QPainter, QPen, QPainterPath, QBrush
from PySide6.QtCore import Qt, QRect, Signal, QTimer, QMargins, QEvent


class CircleCheckBox(QCheckBox):
    """A custom checkbox with a circular indicator that uses explicit colors."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set minimum size to ensure the indicator is fully visible
        self.setMinimumSize(24, 24)
        # Set up the checkbox to be transparent and prevent system styling
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Set default theme
        self.theme = 'dark'
        # Completely disable the default indicator
        self.setStyleSheet("""
            QCheckBox {
                background: transparent;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 0px;
                height: 0px;
                border: none;
                padding: 0px;
                margin: 0px;
            }
        """)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Determine theme (default to dark if not set)
        theme = getattr(self, 'theme', 'dark').lower() if hasattr(self, 'theme') else 'dark'
        is_light = theme == 'light'

        # Calculate indicator rect (left side of checkbox)
        indicator_size = 18
        indicator_rect = QRect(2, (self.height() - indicator_size) // 2, indicator_size, indicator_size)

        # Draw outer circle (always dark gray for visibility)
        painter.setPen(QPen(QColor(85, 85, 85), 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(indicator_rect)

        # Draw inner circle if checked, color depends on theme
        if self.isChecked():
            inner_size = 10
            inner_rect = QRect(
                indicator_rect.x() + (indicator_rect.width() - inner_size) // 2,
                indicator_rect.y() + (indicator_rect.height() - inner_size) // 2,
                inner_size, inner_size
            )
            painter.save()
            try:
                painter.setPen(Qt.NoPen)
                if is_light:
                    painter.setBrush(QBrush(QColor(0, 0, 0)))  # Black for light theme
                else:
                    painter.setBrush(QBrush(QColor(255, 255, 255)))  # White for dark theme
                painter.drawEllipse(inner_rect)
            finally:
                painter.restore()

        # Draw text, color depends on theme
        if self.text():
            text_rect = QRect(
                indicator_rect.right() + 6,
                0,
                self.width() - indicator_rect.right() - 6,
                self.height()
            )
            painter.save()
            try:
                if is_light:
                    painter.setPen(QColor(0, 0, 0))  # Black for light theme
                else:
                    painter.setPen(QColor(255, 255, 255))  # White for dark theme
                painter.drawText(
                    text_rect,
                    Qt.AlignLeft | Qt.AlignVCenter | Qt.TextShowMnemonic,
                    self.text()
                )
            finally:
                painter.restore()

        # Disabled state (semi-transparent)
        if not self.isEnabled():
            painter.setPen(QPen(QColor(0, 0, 0, 60), 1, Qt.DotLine))
            painter.drawEllipse(indicator_rect)


# Import snap utilities
from snap_utils import apply_snap

logger = logging.getLogger(__name__)

def apply_snap_to(pos, size, screen_geo, threshold):
    return pos

class DoubleClickCheckBox(QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_timer = QTimer()
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(250)
        self._click_timer.timeout.connect(self._on_click_timeout)
        self._clicked = False
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self._clicked:
                self._clicked = True
                self._click_timer.start()
                return
            self._clicked = False
            super().mousePressEvent(event)
    
    def _on_click_timeout(self):
        self._clicked = False

class SubSettingsDialog(QDialog):
    hotkey_settings_changed = Signal(bool, str)
    border_opacity_changed = Signal(int)
    _instance = None
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
            cls._instance.finished.connect(lambda: setattr(cls, '_instance', None))
        return cls._instance

    def __init__(self, parent=None, app_instance=None, modal=True, flags=None):
        # Initialize with flags for transparency and frameless window
        if flags is None:
            flags = Qt.WindowFlags(Qt.FramelessWindowHint | Qt.Tool)
            
        super().__init__(parent, flags)
        
        # Enable transparency at dialog level
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Initialize instance variables before setting up UI
        self.target_hwnd = None
        self.app_instance = app_instance
        self.thumbnail_id = None
        self._border_width = 2
        self._content_margins = QMargins(5, 5, 5, 5)
        self.debug_mode = False
        self.window_sort_combo = None
        self.hotkey_checkbox = None
        self.hotkey_edit = None
        self.fps_entry = None
        self.opacity = 0.8  # 80% opacity
        self.border_opacity = 1.0
        
        # Set window properties for transparency while maintaining interactivity
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        
        # Enable transparency
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowModality(Qt.NonModal)
        
        # Set theme
        self.theme = "dark"
        if self.app_instance and hasattr(self.app_instance, 'current_theme') and self.app_instance.current_theme:
            try:
                theme = str(self.app_instance.current_theme).strip().lower()
                if theme in ["light", "dark"]:
                    self.theme = theme
                    logger.debug(f"Initialized theme from app_instance: {self.theme}")
                else:
                    logger.warning(f"Invalid theme value: {theme}. Using default theme.")
            except (TypeError, AttributeError) as e:
                logger.warning(f"Error processing theme: {e}. Falling back to default theme.")

        # Window setup
        self.setWindowTitle("SUBSETTINGS")
        self.setSizeGripEnabled(False)
        
        # Set initial size and position
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.resize(330, 420)  # Original size
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )
        
        # Initialize drag handling
        self.drag_position = None
        self.is_dragging = False
        self.snap_threshold = 30
        self.screen_margin = 5
        
        # Setup UI components first
        self.setup_ui()
        
        # Connect signals after UI is set up
        self._connect_signals()
        
        # Load settings which might affect theme
        self.load_settings()
        
        # Apply theme immediately after UI is set up
        self._theme_applied = False
        # Force theme application with refresh to ensure all widgets are styled
        self.apply_theme(self.theme, force_refresh=True)
        self._theme_applied = True
        
        if hasattr(self, 'title_bar'):
            self.title_bar.installEventFilter(self)
    
    def setup_ui(self):
        logger.debug("Setting up UI...")
        
        # Main container with transparent background
        main_container = QWidget(self)
        main_container.setObjectName("mainContainer")
        main_container.setAttribute(Qt.WA_TranslucentBackground, True)
        main_container.setAutoFillBackground(False)
        
        # Ensure root dialog is transparent
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        
        # Main layout with no margins
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Main content widget with transparent background
        self.main_widget = QWidget(main_container)
        self.main_widget.setObjectName("mainWidget")
        self.main_widget.setAttribute(Qt.WA_TranslucentBackground, True)
        self.main_widget.setAutoFillBackground(False)
        
        # Content layout with proper margins
        content_layout = QVBoxLayout(self.main_widget)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(9)
        
        # Title bar area
        self.title_bar = QWidget()
        self.title_bar.setObjectName("titleBar")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 12)  # Add bottom margin for separation
        title_layout.setSpacing(0)
        
        self.title_label = QLabel("SUBSETTINGS")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_font = QFont("Segoe UI", 24, QFont.Bold)  # Increased to 24pt for better visibility
        self.title_label.setFont(QFont("Segoe UI", 26, QFont.Bold))  # Increased by 2pt
        # Add some padding to the title for better spacing
        self.title_label.setStyleSheet("padding: 5px 0;")
        
        # Close button - pure X with no button styling and proper hover inversion
        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("close_btn")
        self.close_btn.setFixedSize(36, 36)  # Large size
        self.close_btn.setCursor(Qt.PointingHandCursor)  # Hand cursor on hover
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setFlat(True)  # Make button completely flat
        
        # Add title and close button to title layout
        title_layout.addWidget(self.title_label, 1)
        title_layout.addWidget(self.close_btn, 0)
        
        # Add title bar to content layout
        content_layout.addWidget(self.title_bar)
        
        # Set a fixed large font for the X
        font = QFont("Arial", 20, QFont.Bold)
        self.close_btn.setFont(font)
        
        # Apply enhanced X styling with proper color inversion on hover/click
        if self.theme.lower() == 'light':
            self.close_btn.setStyleSheet("""
            QPushButton#close_btn {
                background-color: transparent;
                border: none;
                color: #000000;
                font-size: 28px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
                text-align: center;
            }
            QPushButton#close_btn:hover {
                color: #ffffff;
                background-color: transparent;
                font-size: 30px;
            }
            QPushButton#close_btn:pressed {
                color: #ffffff;
                background-color: transparent;
                font-size: 28px;
            }
            """)
        else:
            self.close_btn.setStyleSheet("""
            QPushButton#close_btn {
                background-color: transparent;
                border: none;
                color: #ffffff;
                font-size: 28px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
                text-align: center;
            }
            QPushButton#close_btn:hover {
                color: #000000;
                background-color: transparent;
                font-size: 30px;
            }
            QPushButton#close_btn:pressed {
                color: #000000;
                background-color: transparent;
                font-size: 28px;
            }
            """)
        
        # Main widget styling is handled in apply_theme()
        
        main_layout.addWidget(self.main_widget)
        
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(main_container)
        
        # Make the window draggable from the title area
        self.title_label.setMouseTracking(True)
        self.title_bar.setMouseTracking(True)
        
        # Store original methods to restore them later if needed
        self._original_title_mouse_press = self.title_label.mousePressEvent
        self._original_title_mouse_move = self.title_label.mouseMoveEvent
        self._original_title_mouse_release = self.title_label.mouseReleaseEvent
        
        # Setup drag handlers
        self.title_label.mousePressEvent = self.title_mouse_press
        self.title_label.mouseMoveEvent = self.title_mouse_move
        self.title_label.mouseReleaseEvent = self.title_mouse_release
        
        # Make entire title bar draggable
        self.title_bar.mousePressEvent = self.title_mouse_press
        self.title_bar.mouseMoveEvent = self.title_mouse_move
        self.title_bar.mouseReleaseEvent = self.title_mouse_release
        
        overlay_opacity_label = QLabel("Overlay Opacity (%)")
        overlay_opacity_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        overlay_opacity_label.setStyleSheet(f"color: {'#000000' if self.theme.lower() == 'light' else '#ffffff'};")
        content_layout.addWidget(overlay_opacity_label)  # Bold, theme-aware color

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setMinimumHeight(17)
        # Connect to main overlay opacity logic ONLY
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        # Theme-aware slider styling
        slider_theme = self.theme.lower() if hasattr(self, 'theme') else 'dark'
        if slider_theme == 'light':
            slider_style = '''
                QSlider::groove:horizontal {
                    border: 1px solid #bbb;
                    background: #f0f0f0;
                    height: 6px;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #222;
                    border: 1px solid #222;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
            '''
        else:
            slider_style = '''
                QSlider::groove:horizontal {
                    border: 1px solid #444;
                    background: #222;
                    height: 6px;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #eee;
                    border: 1px solid #eee;
                    width: 14px;
                    margin: -4px 0;
                    border-radius: 7px;
                }
            '''
        self.opacity_slider.setStyleSheet(slider_style)
        content_layout.addWidget(self.opacity_slider)

        border_opacity_label = QLabel("Border Opacity (%)")
        border_opacity_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        border_opacity_label.setStyleSheet(f"color: {'#000000' if self.theme.lower() == 'light' else '#ffffff'};")
        content_layout.addWidget(border_opacity_label)  # Bold, theme-aware color
        self.border_opacity_slider = QSlider(Qt.Horizontal)
        self.border_opacity_slider.setRange(0, 100)
        self.border_opacity_slider.setValue(100)
        self.border_opacity_slider.setMinimumHeight(17)
        # Connect to border opacity logic ONLY
        self.border_opacity_slider.valueChanged.connect(self._on_border_opacity_changed)
        self.border_opacity_slider.setStyleSheet(slider_style)
        content_layout.addWidget(self.border_opacity_slider)


        theme_label = QLabel("Theme")
        theme_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        theme_label.setStyleSheet(f"color: {'#000000' if self.theme.lower() == 'light' else '#ffffff'};")
        content_layout.addWidget(theme_label)  # Bold, theme-aware color

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setFont(QFont("Segoe UI", 10))
        # Make ComboBox more visually distinct
        combo_style = self.get_combo_style() + 'QComboBox { border-width: 2px; }'
        self.theme_combo.setStyleSheet(combo_style)
        self.theme_combo.setCurrentText(self.theme)
        self.theme_combo.currentTextChanged.connect(lambda text: self.apply_theme(text, from_global=False))
        content_layout.addWidget(self.theme_combo)

        window_sort_label = QLabel("Window Sort:")
        window_sort_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        window_sort_label.setStyleSheet(f"color: {'#000000' if self.theme.lower() == 'light' else '#ffffff'};")
        content_layout.addWidget(window_sort_label)  # Bold, theme-aware color

        self.window_sort_combo = QComboBox()
        self.window_sort_combo.addItems(["Most Recently Active", "Alphabetical"])
        self.window_sort_combo.setFont(QFont("Segoe UI", 10))
        # Make ComboBox more visually distinct
        combo_style = self.get_combo_style() + 'QComboBox { border-width: 2px; }'
        self.window_sort_combo.setStyleSheet(combo_style)
        self.window_sort_combo.currentTextChanged.connect(self._on_window_sort_changed)
        content_layout.addWidget(self.window_sort_combo)

        # Hotkey section with minimal spacing
        # Create compact Switch Hotkey row with proper alignment
        hotkey_widget = QWidget()
        hotkey_widget.setObjectName("hotkeyWidget")
        hotkey_layout = QHBoxLayout(hotkey_widget)
        hotkey_layout.setContentsMargins(0, 5, 0, 5)  # Reduced vertical margins
        hotkey_layout.setSpacing(4)  # Tight spacing for more compact layout
        
        hotkey_label = QLabel("Switch Hotkey:")
        hotkey_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        hotkey_label.setStyleSheet(f"color: {'#000000' if self.theme.lower() == 'light' else '#ffffff'};")  # Bold, theme-aware color
        
        self.hotkey_edit = QKeySequenceEdit()
        self.hotkey_edit.setMinimumWidth(90)  # Even more reduced width for compactness
        self.hotkey_edit.setMaximumWidth(110)  # Reduced maximum width
        self.hotkey_edit.keySequenceChanged.connect(self._on_hotkey_setting_changed)
        
        self.hotkey_checkbox = CircleCheckBox("Enable")
        self.hotkey_checkbox.setChecked(True)
        self.hotkey_checkbox.stateChanged.connect(self._on_hotkey_setting_changed)
        self.hotkey_checkbox.setFont(QFont("Segoe UI", 10, QFont.Bold))  # Bold
        self.hotkey_checkbox.theme = self.theme.lower() if hasattr(self, 'theme') else 'dark'
        
        # Compact layout with perfect spacing - final adjustment
        hotkey_layout.addWidget(hotkey_label)
        hotkey_layout.addWidget(self.hotkey_edit, 0, Qt.AlignVCenter)
        # Adjust spacing to create perfect alignment
        hotkey_layout.addSpacing(5)  # Small positive spacing to create more room between elements
        hotkey_layout.addWidget(self.hotkey_checkbox, 0, Qt.AlignVCenter)
        
        # Precise margin adjustments to ensure no overlap and perfect alignment
        self.hotkey_checkbox.setStyleSheet("margin-left:5px; padding-left:0px;")
        
        # Reduce minimum width of hotkey edit to create more space
        self.hotkey_edit.setMinimumWidth(75)
        
        content_layout.addWidget(hotkey_widget)

        fps_label = QLabel("Capture FPS (Monitor Overlay)")
        fps_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        fps_label.setStyleSheet(f"color: {'#000000' if self.theme.lower() == 'light' else '#ffffff'};")  # Bold, theme-aware color
        content_layout.addWidget(fps_label)

        self.fps_entry = QLineEdit()
        self.fps_entry.setValidator(QIntValidator(1, 240))
        self.fps_entry.setText("60")
        self.fps_entry.setInputMethodHints(Qt.ImhDigitsOnly)
        self.fps_entry.textChanged.connect(self._on_fps_changed)
        # Theme-aware FPS entry styling
        if self.theme.lower() == 'light':
            self.fps_entry.setStyleSheet("background: #ffffff; color: #000000; border: 2px solid #bbbbbb; border-radius: 4px;")
        else:
            self.fps_entry.setStyleSheet("background: #444444; color: #ffffff; border: 2px solid #888888; border-radius: 4px;")
        content_layout.addWidget(self.fps_entry)

        self.auto_switch_checkbox = CircleCheckBox("Auto Switch")
        self.auto_switch_checkbox.setFont(QFont("Segoe UI", 10, QFont.Bold))  # Bold
        self.auto_switch_checkbox.setStyleSheet("font-weight:bold;")
        self.auto_switch_checkbox.setToolTip("Automatically switch to the most recently used window")
        self.auto_switch_checkbox.stateChanged.connect(self._on_auto_switch_changed)
        self.auto_switch_checkbox.theme = self.theme.lower() if hasattr(self, 'theme') else 'dark'
        content_layout.addWidget(self.auto_switch_checkbox, alignment=Qt.AlignTop)
        
        self.click_through_checkbox = CircleCheckBox("Click-through mode")
        self.click_through_checkbox.setFont(QFont("Segoe UI", 10, QFont.Bold))  # Bold
        self.click_through_checkbox.setStyleSheet("font-weight:bold;")
        self.click_through_checkbox.stateChanged.connect(self._on_click_through_changed)
        self.click_through_checkbox.theme = self.theme.lower() if hasattr(self, 'theme') else 'dark'
        content_layout.addWidget(self.click_through_checkbox, alignment=Qt.AlignTop)
        
        content_layout.addSpacing(8)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #666666;")
        line.setFixedHeight(1)
        content_layout.addWidget(line)
        
        experimental_container = QWidget()
        experimental_container.setObjectName("experimentalContainer")
        experimental_layout = QVBoxLayout(experimental_container)
        experimental_layout.setContentsMargins(0, 4, 0, 8)
        experimental_layout.setSpacing(2)
        
        experimental_label = QLabel("EXPERIMENTAL")
        font = QFont("Segoe UI", 9)
        font.setBold(True)
        font.setUnderline(True)
        experimental_label.setFont(font)
        experimental_label.setStyleSheet("color: #ff6b6b;")
        experimental_label.setAlignment(Qt.AlignCenter)
        
        warning_label = QLabel("DO NOT USE WITH ONLINE GAMING OR GACHA")
        warning_font = QFont("Segoe UI", 8)
        warning_font.setBold(True)
        warning_label.setFont(warning_font)
        warning_label.setStyleSheet("color: #ff6b6b;")
        warning_label.setAlignment(Qt.AlignCenter)
        
        experimental_layout.addWidget(experimental_label)
        experimental_layout.addWidget(warning_label)
        
        content_layout.addWidget(experimental_container)
        
        self.key_passthrough_checkbox = CircleCheckBox("Enable key passthrough (Window Overlay)")
        self.key_passthrough_checkbox.setFont(QFont("Segoe UI", 10))
        self.key_passthrough_checkbox.stateChanged.connect(self._on_key_passthrough_changed)
        self.key_passthrough_checkbox.theme = self.theme.lower() if hasattr(self, 'theme') else 'dark'
        content_layout.addWidget(self.key_passthrough_checkbox, alignment=Qt.AlignTop)
        
        self.aggressive_passthrough_checkbox = CircleCheckBox("Aggressive Key Passthrough")
        self.aggressive_passthrough_checkbox.setFont(QFont("Segoe UI", 10))
        self.aggressive_passthrough_checkbox.stateChanged.connect(self._on_aggressive_passthrough_changed)
        self.aggressive_passthrough_checkbox.theme = self.theme.lower() if hasattr(self, 'theme') else 'dark'
        content_layout.addWidget(self.aggressive_passthrough_checkbox, alignment=Qt.AlignTop)


        content_layout.addStretch()
        
        logger.debug("UI setup complete")

    def paintEvent(self, event):
        # Create a transparent background with rounded corners
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill the entire background with transparent
        painter.fillRect(self.rect(), Qt.transparent)

        # Create a path for the main content area
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(1, 1, -1, -1), 10, 10)

        # Determine theme (default to dark if not set)
        theme = getattr(self, 'theme', 'dark').lower()
        is_light = theme == 'light'

        # Fill with semi-transparent background: white for light, dark for dark
        if is_light:
            bg_color = QColor(255, 255, 255, 200)  # White, ~80% opacity
            border_color = QColor(0, 0, 0, int(255 * self.border_opacity))  # Black border
        else:
            bg_color = QColor(30, 30, 30, 200)  # Dark, ~80% opacity
            border_color = QColor(255, 255, 255, int(255 * self.border_opacity))  # White border

        painter.fillPath(path, bg_color)
        painter.setPen(QPen(border_color, self._border_width))
        painter.drawPath(path)


    def save_settings(self):
        if not hasattr(self, 'app_instance') or not hasattr(self.app_instance, 'settings'):
            return
            
        settings = self.app_instance.settings
        
        if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
            settings.setValue("key_passthrough_enabled", self.key_passthrough_checkbox.isChecked())
        if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
            settings.setValue("aggressive_key_passthrough", self.aggressive_passthrough_checkbox.isChecked())
        if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
            settings.setValue("click_through_enabled", self.click_through_checkbox.isChecked())
        if hasattr(self, 'auto_switch_checkbox') and self.auto_switch_checkbox:
            settings.setValue("auto_switch_enabled", self.auto_switch_checkbox.isChecked())
        
        if hasattr(self, 'opacity_slider') and self.opacity_slider:
            settings.setValue("overlay_opacity", self.opacity_slider.value())
        if hasattr(self, 'border_opacity_slider') and self.border_opacity_slider:
            settings.setValue("border_opacity", self.border_opacity_slider.value())
        
        if hasattr(self, 'window_sort_combo') and self.window_sort_combo:
            settings.setValue("window_sort_order", self.window_sort_combo.currentText())
        
        if hasattr(self, 'fps_entry') and self.fps_entry:
            try:
                fps = int(self.fps_entry.text())
                if 1 <= fps <= 240:
                    settings.setValue("capture_fps", fps)
            except (ValueError, AttributeError):
                pass
        
        if hasattr(self, 'hotkey_checkbox') and hasattr(self, 'hotkey_edit'):
            settings.setValue("hotkey_enabled", self.hotkey_checkbox.isChecked())
            settings.setValue("hotkey_sequence", self.hotkey_edit.keySequence().toString())
        
        if hasattr(self, 'theme_combo') and self.theme_combo:
            settings.setValue("theme", self.theme_combo.currentText().lower())
        
        settings.sync()
        logger.debug("Settings saved successfully")

    def load_settings(self):
        if not hasattr(self, 'app_instance') or not hasattr(self.app_instance, 'settings'):
            logger.error("SubSettingsDialog: app_instance or settings not available.")
            return
            
        settings = self.app_instance.settings
        
        signal_blocks = {}
        if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
            signal_blocks['key_passthrough'] = self.key_passthrough_checkbox.blockSignals(True)
        if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
            signal_blocks['aggressive_passthrough'] = self.aggressive_passthrough_checkbox.blockSignals(True)
        if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
            signal_blocks['click_through'] = self.click_through_checkbox.blockSignals(True)
        if hasattr(self, 'auto_switch_checkbox') and self.auto_switch_checkbox:
            signal_blocks['auto_switch'] = self.auto_switch_checkbox.blockSignals(True)
        
        try:
            if hasattr(self, 'opacity_slider') and self.opacity_slider:
                opacity = settings.value("overlay_opacity", 100, type=int)
                self.opacity_slider.setValue(opacity)
                self.opacity = opacity / 100.0  # Initialize opacity
            if hasattr(self, 'border_opacity_slider') and self.border_opacity_slider:
                border_opacity = settings.value("border_opacity", 100, type=int)
                self.border_opacity_slider.setValue(border_opacity)
                self.border_opacity = border_opacity / 100.0  # Initialize border opacity
            
            if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                key_passthrough = settings.value("key_passthrough_enabled", True, type=bool)
                self.key_passthrough_checkbox.setChecked(key_passthrough)
            if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                aggressive_passthrough = settings.value("aggressive_key_passthrough", False, type=bool)
                self.aggressive_passthrough_checkbox.setChecked(aggressive_passthrough)
            if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
                click_through = settings.value("click_through_enabled", False, type=bool)
                self.click_through_checkbox.setChecked(click_through)
            if hasattr(self, 'auto_switch_checkbox') and self.auto_switch_checkbox:
                auto_switch = settings.value("auto_switch_enabled", False, type=bool)
                self.auto_switch_checkbox.setChecked(auto_switch)
                logger.debug(f"Loaded auto_switch setting: {auto_switch}")
            
            if hasattr(self, 'window_sort_combo') and self.window_sort_combo:
                sort_order = settings.value("window_sort_order", "Most Recently Active", type=str)
                index = self.window_sort_combo.findText(sort_order)
                if index >= 0:
                    self.window_sort_combo.setCurrentIndex(index)
                    
            if hasattr(self, 'fps_entry') and self.fps_entry:
                fps = settings.value("capture_fps", 60, type=int)
                self.fps_entry.setText(str(fps))
                
            if hasattr(self, 'hotkey_checkbox') and hasattr(self, 'hotkey_edit'):
                hotkey_enabled = settings.value("hotkey_enabled", True, type=bool)
                hotkey_sequence = settings.value("hotkey_sequence", "Ctrl+Alt+Space", type=str)
                self.hotkey_checkbox.setChecked(hotkey_enabled)
                self.hotkey_edit.setKeySequence(QKeySequence(hotkey_sequence))
            
            if hasattr(self, 'theme_combo') and self.theme_combo:
                theme = settings.value("theme", "Dark", type=str).capitalize()
                index = self.theme_combo.findText(theme, Qt.MatchFixedString)
                if index >= 0:
                    self.theme_combo.setCurrentIndex(index)
                    
            if hasattr(self.app_instance, 'key_passthrough_setting_changed'):
                self.app_instance.key_passthrough_setting_changed.emit(key_passthrough, aggressive_passthrough)
                
            logger.debug(f"Loaded settings - key_passthrough: {key_passthrough}, aggressive: {aggressive_passthrough}")
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}", exc_info=True)
        finally:
            if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                self.key_passthrough_checkbox.blockSignals(signal_blocks.get('key_passthrough', False))
            if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                self.aggressive_passthrough_checkbox.blockSignals(signal_blocks.get('aggressive_passthrough', False))
            if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
                self.click_through_checkbox.blockSignals(signal_blocks.get('click_through', False))
            if hasattr(self, 'auto_switch_checkbox') and self.auto_switch_checkbox:
                self.auto_switch_checkbox.blockSignals(signal_blocks.get('auto_switch', False))

    def _connect_signals(self):
        if hasattr(self, 'theme_combo'):
            self.theme_combo.currentTextChanged.connect(
                lambda text: self.apply_theme(text, from_global=False))
        
        if hasattr(self, 'key_passthrough_checkbox'):
            self.key_passthrough_checkbox.stateChanged.connect(self._on_key_passthrough_changed)
        if hasattr(self, 'aggressive_passthrough_checkbox'):
            self.aggressive_passthrough_checkbox.stateChanged.connect(self._on_aggressive_passthrough_changed)
        if hasattr(self, 'click_through_checkbox'):
            self.click_through_checkbox.stateChanged.connect(self._on_click_through_changed)
        if hasattr(self, 'auto_switch_checkbox'):
            self.auto_switch_checkbox.stateChanged.connect(self._on_auto_switch_changed)
            
        if hasattr(self, 'opacity_slider'):
            self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        if hasattr(self, 'border_opacity_slider'):
            self.border_opacity_slider.valueChanged.connect(self._on_border_opacity_changed)
        if hasattr(self, 'window_sort_combo'):
            self.window_sort_combo.currentTextChanged.connect(self._on_window_sort_changed)
        if hasattr(self, 'fps_entry') and hasattr(self.fps_entry, 'textChanged'):
            self.fps_entry.textChanged.connect(self._on_fps_changed)
        if hasattr(self, 'hotkey_checkbox') and hasattr(self, 'hotkey_edit'):
            self.hotkey_checkbox.stateChanged.connect(self._on_hotkey_setting_changed)
            self.hotkey_edit.keySequenceChanged.connect(self._on_hotkey_setting_changed)
            
        self.finished.connect(self.save_settings)

    def title_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def title_mouse_move(self, event):
        if event.buttons() & Qt.LeftButton and self.drag_position is not None:
            new_pos = event.globalPosition().toPoint() - self.drag_position
            new_pos, _ = apply_snap(new_pos, self.size(), self.snap_threshold)
            self.move(new_pos)
            event.accept()
    
    def title_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            current_pos = self.pos()
            new_pos, _ = apply_snap(current_pos, self.size(), self.snap_threshold)
            if new_pos != current_pos:
                self.move(new_pos)
            self.drag_position = None
            event.accept()
    
    def apply_theme(self, theme, from_global=False, force_refresh=False):
        """Apply theme to the dialog and all child widgets.
        
        Args:
            theme (str): Theme name ('light' or 'dark', case-insensitive)
            from_global (bool): If True, this is called from a global theme change
            force_refresh (bool): If True, forces theme reapplication even if unchanged
        """
        try:
            logger.debug(f"Applying theme: {theme}, from_global: {from_global}, force: {force_refresh}")
            
            # Normalize and validate theme
            theme = str(theme).strip().lower() if theme else "dark"
            if theme not in ["light", "dark"]:
                logger.warning(f"Invalid theme '{theme}'. Defaulting to 'dark'.")
                theme = "dark"
                
            # Check if theme needs to be updated
            current_theme = getattr(self, 'theme', '').lower()
            theme_changed = current_theme != theme
            
            if not theme_changed and not force_refresh:
                logger.debug(f"Theme unchanged ({theme}), skipping reapplication")
                return
                
            logger.debug(f"Theme changing from '{current_theme}' to '{theme}'")
            self.theme = theme
            
            # Set color scheme based on theme
            if theme == "light":
                colors = {
                    'bg': "#f0f0f0", 'fg': "#000000",
                    'title_bg': "#f0f0f0", 'title_fg': "#000000",
                    'input_bg': "#ffffff", 'input_fg': "#000000",
                    'line_edit_bg': "#ffffff", 'line_edit_fg': "#000000", 'line_edit_border': "#999999",
                    'combo_bg': "#ffffff", 'combo_fg': "#000000", 'combo_border': "#999999",
                    'combo_dropdown_bg': "#ffffff",
                    'slider_groove': "#cccccc", 'slider_handle': "#505050",
                    'slider_handle_border': "#333333", 'slider_handle_hover': "#707070",
                    'button_bg': "#e1e1e1", 'button_text': "#000000", 'button_border': "#999999",
                    'button_hover': "#f0f0f0", 'button_pressed': "#d0d0d0",
                    'close_btn_hover': "#ff6b6b", 'close_btn_pressed': "#ff3b3b"
                }
            else:  # dark theme
                colors = {
                    'bg': "#2a2a2a", 'fg': "#ffffff",
                    'title_bg': "#2a2a2a", 'title_fg': "#ffffff",
                    'input_bg': "#404040", 'input_fg': "#ffffff",
                    'line_edit_bg': "#404040", 'line_edit_fg': "#ffffff", 'line_edit_border': "#666666",
                    'combo_bg': "#404040", 'combo_fg': "#ffffff", 'combo_border': "#666666",
                    'combo_dropdown_bg': "#404040",
                    'slider_groove': "#555555", 'slider_handle': "#a0a0a0",
                    'slider_handle_border': "#cccccc", 'slider_handle_hover': "#c0c0c0",
                    'button_bg': "#404040", 'button_text': "#ffffff", 'button_border': "#666666",
                    'button_hover': "#4a4a4a", 'button_pressed': "#363636",
                    'close_btn_hover': "#ff6b6b", 'close_btn_pressed': "#ff3b3b"
                }

            # Get checkbox style
            checkbox_style = self.get_checkbox_style(theme)

            # Build the style sheet with transparency support
            bg_color = 'rgba(240, 240, 240, 0.7)' if theme == 'light' else 'rgba(42, 42, 42, 0.7)'
            style_sheet = f"""
                /* Base dialog and root container styling - fully transparent */
                QDialog {{
                    background: transparent;
                }}
                
                /* Main container - completely transparent */
                QWidget#mainContainer {{
                    background: transparent;
                }}
                
                /* Main widget - single semi-transparent background layer */
                QWidget#mainWidget {{
                    background: {bg_color};
                    border-radius: 10px;
                    border: 1px solid {colors['button_border']};
                    color: {colors['fg']};
                    font-family: 'Segoe UI';
                    font-size: 11px;
                }}
                
                /* Ensure child widgets of mainWidget are truly transparent */
                QWidget#mainWidget > QWidget {{
                    background: transparent;
                }}
                
                /* Base widget styling for all other widgets */
                QWidget {{
                    color: {colors['fg']};
                    font-family: 'Segoe UI';
                    font-size: 11px;
                }}
                
                /* Checkbox styling with proper visual feedback */
                QCheckBox {{
                    color: {colors['fg']};
                    background: transparent;
                    spacing: 8px;
                    padding: 2px 0;
                }}
                
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    border: 1px solid {colors['button_border']};
                    border-radius: 8px;
                    background: {colors['input_bg']};
                }}
                
                QCheckBox::indicator:checked {{
                    background: {colors['button_bg']};
                    border: 1px solid {colors['button_border']};
                    image: url('');  /* Clear any default image */
                }}
                
                QCheckBox::indicator:checked:hover {{
                    background: {colors['button_hover']};
                }}
                
                QCheckBox::indicator:unchecked:hover {{
                    background: {colors['input_bg']};
                    border: 1px solid {colors['button_hover']};
                }}
                
                QCheckBox::indicator:disabled {{
                    border: 1px solid {colors['button_border']};
                    background: {'#e0e0e0' if theme == 'light' else '#3a3a3a'};
                }}
                
                QCheckBox::indicator:disabled:checked {{
                    background: {'#c0c0c0' if theme == 'light' else '#505050'};
                }}
                
                /* Sliders */
                QSlider::groove:horizontal {{
                    height: 4px;
                    background: {colors['slider_groove']};
                    border-radius: 2px;
                }}
                
                QSlider::handle:horizontal {{
                    background: {colors['slider_handle']};
                    border: 1px solid {colors['slider_handle_border']};
                    width: 12px;
                    margin: -4px 0;
                    border-radius: 6px;
                }}
                
                QSlider::handle:horizontal:hover {{
                    background: {colors['slider_handle_hover']};
                }}
                
                /* Title bar */
                QLabel#titleLabel {{
                    color: {colors['title_fg']};
                    background: transparent;
                    font-weight: bold;
                    font-size: 24px;
                    padding: 5px 0;
                }}
                
                /* Buttons */
                QPushButton {{
                    background-color: {colors['button_bg']};
                    color: {colors['button_text']};
                    border: 1px solid {colors['button_border']};
                    padding: 4px 8px;
                    border-radius: 3px;
                }}
                
                QPushButton:hover {{
                    background-color: {colors['button_hover']};
                }}
                
                QPushButton:pressed {{
                    background-color: {colors['button_pressed']};
                }}
                
                /* Line edits */
                QLineEdit, QTextEdit, QPlainTextEdit {{
                    background-color: {colors['line_edit_bg']};
                    color: {colors['line_edit_fg']};
                    border: 1px solid {colors['line_edit_border']};
                    padding: 3px;
                    border-radius: 3px;
                }}
                
                /* Combo boxes */
                QComboBox {{
                    background-color: {colors['combo_bg']};
                    color: {colors['combo_fg']};
                    border: 1px solid {colors['combo_border']};
                    padding: 2px 6px 2px 3px;
                    border-radius: 3px;
                }}
                
                QComboBox::drop-down {{
                    border: 0px;
                    background: transparent;
                }}
                
                QComboBox::down-arrow {{
                    image: url(none);
                    width: 0px;
                    height: 0px;
                }}
                
                /* Close button */
                QPushButton#closeButton {{
                    background-color: transparent;
                    color: {colors['fg']};
                    border: none;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 0 8px;
                }}
                
                QPushButton#closeButton:hover {{
                    color: {colors['close_btn_hover']};
                    background: transparent;
                }}
                
                QPushButton#closeButton:pressed {{
                    color: {colors['close_btn_pressed']};
                }}
                """

            # Apply the style sheet to the dialog
            self.setStyleSheet(style_sheet + "\nQDialog { background: transparent; }")
            
            # Update the palette
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor(colors['bg']))
            palette.setColor(QPalette.WindowText, QColor(colors['fg']))
            # Use fully transparent window background
            transparent_bg = QColor(0, 0, 0, 0)
            palette.setColor(QPalette.Window, transparent_bg)
            
            # Set other colors normally
            palette.setColor(QPalette.WindowText, QColor(colors['fg']))
            palette.setColor(QPalette.Base, QColor(colors['input_bg']))
            palette.setColor(QPalette.AlternateBase, QColor(colors['bg']))
            palette.setColor(QPalette.ToolTipBase, QColor(colors['fg']))
            palette.setColor(QPalette.ToolTipText, QColor(colors['bg']))
            palette.setColor(QPalette.Text, QColor(colors['fg']))
            palette.setColor(QPalette.Button, QColor(colors['button_bg']))
            palette.setColor(QPalette.ButtonText, QColor(colors['button_text']))
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, Qt.white)
            
            # Disabled state colors
            palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(colors['fg']).darker(150))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(colors['fg']).darker(150))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(colors['button_text']).darker(150))
            
            # Tooltip colors
            palette.setColor(QPalette.ToolTipBase, QColor(colors['bg']).lighter(150))
            palette.setColor(QPalette.ToolTipText, QColor(colors['fg']).darker(150))
            
            # Set the palette
            self.setPalette(palette)
            
            # Apply custom styles that should not be overridden by the theme
            if hasattr(self, '_apply_custom_styles'):
                self._apply_custom_styles()
                
            # Update the theme combo box to reflect current theme
            if hasattr(self, 'theme_combo') and self.theme_combo:
                theme_text = theme.capitalize()
                index = self.theme_combo.findText(theme_text, Qt.MatchFixedString)
                if index >= 0 and self.theme_combo.currentIndex() != index:
                    self.theme_combo.setCurrentIndex(index)
            
            # Force style update on all child widgets
            if hasattr(self, 'update_styles'):
                self.update_styles()
            
            # Notify parent application of theme change if needed
            if not from_global and hasattr(self, 'app_instance') and self.app_instance:
                try:
                    # Try apply_theme_globally first (main PiPApplication method)
                    if hasattr(self.app_instance, 'apply_theme_globally') and callable(self.app_instance.apply_theme_globally):
                        logger.debug(f"Notifying app instance of theme change to {theme} via apply_theme_globally")
                        self.app_instance.apply_theme_globally(theme, from_global=False)
                    # Fall back to apply_theme for backward compatibility
                    elif hasattr(self.app_instance, 'apply_theme') and callable(self.app_instance.apply_theme):
                        logger.debug(f"Notifying app instance of theme change to {theme} via apply_theme")
                        self.app_instance.apply_theme(theme)
                    else:
                        logger.debug("Neither app_instance.apply_theme_globally nor app_instance.apply_theme available, skipping notification")
                except Exception as e:
                    logger.error(f"Error notifying app instance of theme change: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error applying theme: {e}", exc_info=True)
            # Apply a basic fallback theme on error
            try:
                fallback_style = """
                    QDialog {
                        background: transparent;
                    }
                    QWidget#mainContainer {
                        background: transparent;
                    }
                    QWidget#mainWidget {
                        background: rgba(42, 42, 42, 0.7);
                        border-radius: 10px;
                        border: 1px solid #666666;
                        color: #ffffff;
                        font-family: 'Segoe UI';
                        font-size: 11px;
                    }
                    QLabel { 
                        background: transparent; 
                        color: #ffffff; 
                    }
                    QPushButton {
                        background-color: #404040;
                        color: #ffffff;
                        border: 1px solid #666666;
                        padding: 4px 12px;
                        border-radius: 4px;
                    }
                    QPushButton:hover { background-color: #4a4a4a; }
                    QPushButton:pressed { background-color: #363636; }
                    QLineEdit, QComboBox, QSlider::handle {
                        background-color: #404040;
                        color: #ffffff;
                        border: 1px solid #666666;
                        padding: 4px 8px;
                        border-radius: 4px;
                    }
                """
                self.setStyleSheet(fallback_style)
            except Exception as fallback_error:
                logger.error(f"Failed to apply fallback theme: {fallback_error}")
        finally:
            # Ensure theme application is marked as complete
            self._theme_applied = True
            logger.debug("Theme application completed")
            return
            
        # Apply the style sheet to the main widget
        if hasattr(self, 'main_widget'):
            self.main_widget.setStyleSheet(style_sheet)
        
        # Set up the palette with all necessary colors
        palette = self.palette()
        
        # Base colors
        palette.setColor(QPalette.Window, QColor(bg_color))
        palette.setColor(QPalette.WindowText, QColor(fg_color))
        palette.setColor(QPalette.Base, QColor(input_bg))
        palette.setColor(QPalette.AlternateBase, QColor(button_hover))
        
        # Text colors
        palette.setColor(QPalette.Text, QColor(fg_color))
        palette.setColor(QPalette.ButtonText, QColor(button_text))
        palette.setColor(QPalette.BrightText, QColor("#ffffff"))
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(button_bg))
        
        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor("#0078d7"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        
        # Disabled state colors
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(fg_color).darker(150))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(fg_color).darker(150))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(button_text).darker(150))
        
        # Tooltip colors
        palette.setColor(QPalette.ToolTipBase, QColor(bg_color).lighter(150))
        palette.setColor(QPalette.ToolTipText, QColor(fg_color).darker(150))
        
        # Apply the palette to the dialog and all its children
        self.setPalette(palette)
        
        # Force update of all child widgets
        QApplication.processEvents()
        
        # Apply theme-specific palette settings
        if self.theme == "light":
            palette.setColor(QPalette.AlternateBase, QColor("#f0f0f0"))
            palette.setColor(QPalette.Text, QColor("#000000"))
            palette.setColor(QPalette.Button, QColor("#e1e1e1"))
            palette.setColor(QPalette.ButtonText, QColor("#000000"))
        else:  # dark
            palette.setColor(QPalette.Window, QColor("#2a2a2a"))
            palette.setColor(QPalette.WindowText, QColor("#ffffff"))
            palette.setColor(QPalette.Base, QColor("#404040"))
            palette.setColor(QPalette.AlternateBase, QColor("#2a2a2a"))
            palette.setColor(QPalette.Text, QColor("#ffffff"))
            palette.setColor(QPalette.Button, QColor("#404040"))
            palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
        
        # Apply the palette to the dialog and all its children
        self.setPalette(palette)
        
        # Force style refresh
        self.style().unpolish(self)
        self.style().polish(self)
        
        # Update all CircleCheckBox instances with the new theme
        checkbox_style = self.get_checkbox_style(theme)
        for widget in self.findChildren(CircleCheckBox):
            widget.theme = theme.lower()
            widget.setStyleSheet(checkbox_style)
            # Force immediate repaint of each checkbox
            widget.update()
            
        # Update theme-aware label colors
        text_color = '#000000' if theme.lower() == 'light' else '#ffffff'
        for widget in self.findChildren(QLabel):
            current_style = widget.styleSheet()
            if 'color:' in current_style and ('#000000' in current_style or '#ffffff' in current_style):
                widget.setStyleSheet(f"color: {text_color};")
        
        # Update widgets
        self.update()
        QApplication.processEvents()
        
        # Notify parent if this was a global theme change
        if not from_global and hasattr(self, 'app_instance') and self.app_instance is not None:
            if hasattr(self.app_instance, 'apply_theme') and callable(self.app_instance.apply_theme):
                logger.debug("Notifying app instance of theme change")
                try:
                    self.app_instance.apply_theme(self.theme)
                except Exception as e:
                    logger.error(f"Error notifying app instance of theme change: {e}")
            else:
                logger.debug("app_instance does not have apply_theme method")
        
        # Mark theme as applied
        self._theme_applied = True
        logger.debug("Theme application complete")
        
        # ... (rest of the code remains the same)
        # Force a complete repaint
        self.repaint()
        QApplication.processEvents()
    
    def get_checkbox_style(self, theme):
        if theme.lower() == "light":
            return """
                QCheckBox { color: #000000; spacing: 4px; font-size: 11px; padding: 4px 0; }
                QCheckBox::indicator { width: 16px; height: 16px; border: 2px solid #000000; border-radius: 9px; background: #ffffff; }
                QCheckBox::indicator:checked { background: #000000 !important; border: 2px solid #000000 !important; }
                QCheckBox::indicator:unchecked:hover { border-color: #333333; }
                QCheckBox:checked { font-weight: bold; }
            """
        else:
            return """
                QCheckBox { color: #ffffff; spacing: 4px; font-size: 11px; padding: 4px 0; }
                QCheckBox::indicator { width: 16px; height: 16px; border: 2px solid #888888; border-radius: 9px; background: #333333; }
                QCheckBox::indicator:checked { border: 2px solid #ffffff; background: transparent; }
                QCheckBox::indicator:unchecked:hover { border-color: #aaaaaa; }
                QCheckBox:checked { font-weight: bold; }
            """

    def _on_key_passthrough_changed(self, state):
        if not self.app_instance or not hasattr(self.app_instance, 'settings'):
            return
            
        key_passthrough_blocked = False
        aggressive_blocked = False
        
        try:
            key_passthrough_cb = getattr(self, 'key_passthrough_checkbox', None)
            aggressive_cb = getattr(self, 'aggressive_passthrough_checkbox', None)
            
            if key_passthrough_cb:
                key_passthrough_cb.blockSignals(True)
                key_passthrough_blocked = True
            if aggressive_cb:
                aggressive_cb.blockSignals(True)
                aggressive_blocked = True
            
            enabled = state == Qt.Checked
            aggressive = False
            
            settings = self.app_instance.settings
            settings.setValue("key_passthrough_enabled", enabled)
            settings.setValue("aggressive_key_passthrough", aggressive)
            settings.sync()
            
            if key_passthrough_cb:
                key_passthrough_cb.setChecked(enabled)
            if aggressive_cb:
                aggressive_cb.setChecked(aggressive)
            
            logger.debug(f"Key passthrough: enabled={enabled}, aggressive={aggressive}")
            
            if hasattr(self.app_instance, 'key_passthrough_setting_changed'):
                self.app_instance.key_passthrough_setting_changed.emit(enabled, aggressive)
                
        except Exception as e:
            logger.error(f"Error updating key passthrough setting: {e}", exc_info=True)
        finally:
            if key_passthrough_blocked and hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                self.key_passthrough_checkbox.blockSignals(False)
            if aggressive_blocked and hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                self.aggressive_passthrough_checkbox.blockSignals(False)
            if hasattr(self, 'save_settings'):
                self.save_settings()
    
    def _on_aggressive_passthrough_changed(self, state):
        if not self.app_instance or not hasattr(self.app_instance, 'settings'):
            return
            
        key_passthrough_blocked = False
        aggressive_blocked = False
        
        try:
            key_passthrough_cb = getattr(self, 'key_passthrough_checkbox', None)
            aggressive_cb = getattr(self, 'aggressive_passthrough_checkbox', None)
            
            if key_passthrough_cb:
                key_passthrough_cb.blockSignals(True)
                key_passthrough_blocked = True
            if aggressive_cb:
                aggressive_cb.blockSignals(True)
                aggressive_blocked = True
            
            aggressive = state == Qt.Checked
            enabled = aggressive
            
            settings = self.app_instance.settings
            settings.setValue("key_passthrough_enabled", enabled)
            settings.setValue("aggressive_key_passthrough", aggressive)
            settings.sync()
            
            if key_passthrough_cb:
                key_passthrough_cb.setChecked(enabled)
            if aggressive_cb:
                aggressive_cb.setChecked(aggressive)
            
            logger.debug(f"Aggressive key passthrough: enabled={enabled}, aggressive={aggressive}")
            
            if hasattr(self.app_instance, 'key_passthrough_setting_changed'):
                self.app_instance.key_passthrough_setting_changed.emit(enabled, aggressive)
                
        except Exception as e:
            logger.error(f"Error updating aggressive key passthrough setting: {e}", exc_info=True)
        finally:
            if key_passthrough_blocked and hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                self.key_passthrough_checkbox.blockSignals(False)
            if aggressive_blocked and hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                self.aggressive_passthrough_checkbox.blockSignals(False)
            if hasattr(self, 'save_settings'):
                self.save_settings()

    def _on_click_through_changed(self, state):
        if hasattr(self, 'app_instance') and self.app_instance:
            self.app_instance.set_click_through_mode(state == Qt.Checked)
        self.save_settings()
        
    def _on_auto_switch_changed(self, state):
        if hasattr(self, 'auto_switch_checkbox') and self.auto_switch_checkbox:
            is_checked = self.auto_switch_checkbox.isChecked()
            logger.debug(f"Auto-switch changed to: {is_checked}")
            if hasattr(self, 'app_instance') and self.app_instance:
                if hasattr(self.app_instance, 'set_auto_switch'):
                    self.app_instance.set_auto_switch(is_checked)
        self.save_settings()
            
    def _on_fps_changed(self, text):
        if text and text.isdigit():
            try:
                fps = int(text)
                if 1 <= fps <= 240:
                    if self.app_instance:
                        if hasattr(self.app_instance, 'set_capture_fps'):
                            self.app_instance.set_capture_fps(fps)
                        if hasattr(self.app_instance, 'settings'):
                            self.app_instance.settings.setValue("capture_fps", fps)
                            logger.debug(f"Capture FPS changed to: {fps}")
                else:
                    logger.warning(f"FPS value {fps} is outside the allowed range (1-240)")
            except (ValueError, TypeError):
                logger.debug(f"Invalid FPS value: {text}")

    def _on_opacity_changed(self, value):
        """Handle main overlay opacity slider change (affects all overlays, not dialog visuals).
        
        Args:
            value (int): Opacity value from slider (0-100)
            
        Note:
            Converts 0-100 slider value to 0.0-1.0 for internal use.
            The UI shows percentages (0-100%) but internally uses 0.0-1.0.
        """
        if not self.app_instance:
            return
            
        try:
            # Convert from 0-100 slider value to 0.0-1.0 for internal use
            opacity = float(value) / 100.0
            
            # Set opacity on all overlays
            if hasattr(self.app_instance, 'set_all_overlays_opacity'):
                if debug_enabled():
                    logger.debug(f"[UI] Setting all overlays opacity to {opacity:.3f} (from {value}%)")
                self.app_instance.set_all_overlays_opacity(opacity)
            
            # Save the raw 0-100 value for UI consistency
            if hasattr(self.app_instance, 'settings'):
                self.app_instance.settings.setValue("overlay_opacity", value)
                if debug_enabled():
                    logger.debug(f"[UI] Saved overlay opacity setting: {value}%")
                    
        except (ValueError, TypeError) as e:
            logger.error(f"Error setting overlay opacity: {e}")
            if debug_enabled():
                logger.debug(f"[UI] Failed to set opacity with value: {value} (type: {type(value)})")

    def _on_border_opacity_changed(self, value):
        """Handle border opacity slider change (affects overlay border opacity, not dialog visuals).
        
        Args:
            value (int): Border opacity value from slider (0-100)
            
        Note:
            Converts 0-100 slider value to 0.0-1.0 for internal use.
            The UI shows percentages (0-100%) but internally uses 0.0-1.0.
        """
        if not self.app_instance:
            return
            
        try:
            # Convert from 0-100 slider value to 0.0-1.0 for internal use
            opacity = float(value) / 100.0
            
            # Set border opacity on all overlays
            if hasattr(self.app_instance, 'set_overlay_border_opacity'):
                if debug_enabled():
                    logger.debug(f"[UI] Setting border opacity to {opacity:.3f} (from {value}%)")
                self.app_instance.set_overlay_border_opacity(opacity)
            
            # Save the raw 0-100 value for UI consistency
            if hasattr(self.app_instance, 'settings'):
                self.app_instance.settings.setValue("border_opacity", value)
                if debug_enabled():
                    logger.debug(f"[UI] Saved border opacity setting: {value}%")
            
            # Emit the signal for any other components that need to listen
            self.border_opacity_changed.emit(opacity)
                    
        except (ValueError, TypeError) as e:
            logger.error(f"Error setting border opacity: {e}")
            if debug_enabled():
                logger.debug(f"[UI] Failed to set border opacity with value: {value} (type: {type(value)})")

    def _on_hotkey_setting_changed(self):
        if not self.app_instance or not hasattr(self.app_instance, 'settings'):
            return
        # Get current hotkey state from widgets
        enabled = False
        sequence = ""
        try:
            if hasattr(self, 'hotkey_checkbox') and self.hotkey_checkbox:
                enabled = self.hotkey_checkbox.isChecked()
            if hasattr(self, 'hotkey_edit') and self.hotkey_edit:
                sequence = self.hotkey_edit.keySequence().toString()
        except Exception as e:
            logger.error(f"Error reading hotkey widgets: {e}")
        # Call main app to update hotkey binding immediately
        try:
            if hasattr(self.app_instance, 'update_switch_hotkey'):
                ok = self.app_instance.update_switch_hotkey(enabled, sequence)
                if not ok:
                    logger.error("Failed to update switch hotkey dynamically from subsettings dialog")
        except Exception as e:
            logger.error(f"Exception updating switch hotkey: {e}")
    
    def get_button_style(self):
        if self.theme.lower() == "light":
            return (
                'QPushButton { background-color: #e6e6e6; color: #000000; border: 2px solid #000000; border-radius: 5px; padding: 6px 12px; font-family: "Segoe UI"; font-size: 12px; font-weight: 500; min-height: 30px; min-width: 80px; } '
                'QPushButton:hover { background-color: #f0f0f0; border-color: #333333; } '
                'QPushButton:pressed { background-color: #d9d9d9; border-color: #000000; } '
                'QPushButton:disabled { background-color: #aaaaaa; border-color: #999999; color: #666666; }'
            )
        else:
            return (
                'QPushButton { background-color: #444444; color: #ffffff; border: 2px solid #dddddd; border-radius: 5px; padding: 6px 12px; font-family: "Segoe UI"; font-size: 12px; font-weight: 600; min-height: 30px; min-width: 80px; } '
                'QPushButton:hover { background-color: #555555; border-color: #ffffff; } '
                'QPushButton:pressed { background-color: #333333; border-color: #ffffff; } '
                'QPushButton:disabled { background-color: #555555; border-color: #666666; color: #999999; }'
            )

    def get_combo_style(self):
        if self.theme.lower() == "light":
            # 20% darker than #f5f5f5 is roughly #cccccc; thick black border, solid popup color
            return (
                'QComboBox { background-color: #cccccc; color: #000000; border: 3px solid #000000; border-radius: 4px; padding: 5px 8px; font-family: "Segoe UI"; font-size: 12px; min-height: 24px; } '
                'QComboBox:hover { border-color: #222222; } '
                'QComboBox::drop-down { width: 20px; border: none; background: #cccccc; } '
                'QComboBox::down-arrow { image: none; width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #000000; } '
                'QComboBox::drop-down:hover { background-color: #b3b3b3; } '
                'QComboBox QAbstractItemView { background-color: #cccccc; color: #000000; border: none; selection-background-color: #0078d7; selection-color: white; outline: none; } QComboBox QAbstractItemView::item { border: none; }'
            )
        else:
            # 30% darker than #404040 is roughly #232323; white border, solid popup color
            return (
                'QComboBox { background-color: #232323; color: #ffffff; border: 2px solid #ffffff; border-radius: 4px; padding: 5px 8px; font-family: "Segoe UI"; font-size: 12px; min-height: 24px; } '
                'QComboBox:hover { border-color: #bbbbbb; } '
                'QComboBox::drop-down { width: 20px; border: none; background: #232323; } '
                'QComboBox::down-arrow { image: none; width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #ffffff; } '
                'QComboBox::drop-down:hover { background-color: #333333; } '
                'QComboBox QAbstractItemView { background-color: #232323; color: #ffffff; border: none; selection-background-color: #0078d7; selection-color: white; outline: none; } QComboBox QAbstractItemView::item { border: none; }'
            )

    def _on_window_sort_changed(self, sort_order):
        if self.app_instance:
            if hasattr(self.app_instance, 'update_window_sort_order'):
                self.app_instance.update_window_sort_order(sort_order)
            if hasattr(self.app_instance, 'settings'):
                self.app_instance.settings.setValue("window_sort_order", sort_order)
        logger.debug(f"Window sort order changed to: {sort_order}")

    def eventFilter(self, obj, event):
        if obj == self.title_bar and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                child_widget = self.title_bar.childAt(event.position().toPoint())
                if child_widget and child_widget != self.title_label and child_widget != self.title_bar:
                    return False
                self.is_dragging = True
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
        return super().eventFilter(obj, event)

    def mouseMoveEvent(self, event):
        if self.is_dragging and event.buttons() == Qt.LeftButton:
            if not self.drag_position:
                self.is_dragging = False
                return
            new_pos_global = event.globalPosition().toPoint()
            new_pos_local = new_pos_global - self.drag_position
            self.move(new_pos_local)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_dragging and event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.drag_position = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.is_dragging = False
        
    def showEvent(self, event):
        """Handle the show event to ensure theme is applied when dialog is shown."""
        super().showEvent(event)
        
        # Ensure theme is applied when dialog is shown
        if not hasattr(self, '_theme_applied') or not self._theme_applied:
            logger.debug("Applying theme in showEvent")
            self.apply_theme(self.theme, force_refresh=True)
            self._theme_applied = True
        
        # Ensure window is properly activated and raised
        self.activateWindow()
        self.raise_()
        
        # Apply theme again on show to ensure proper styling
        logger.debug("Show event - refreshing theme")
        self.apply_theme(self.theme, force_refresh=True)
        
        def update_styles(widget):
            """Recursively update styles for all child widgets."""
            if not widget:
                return
                
            try:
                widget_class = widget.metaObject().className()
                
                # Update widget styles
                if hasattr(widget, 'style'):
                    try:
                        widget.style().unpolish(widget)
                        widget.style().polish(widget)
                    except Exception as e:
                        logger.debug(f"Error updating style for {widget_class}: {str(e)}")
                
                # Update widget if it has the method and it's not a QListView
                if hasattr(widget, 'update') and callable(getattr(widget, 'update')):
                    try:
                        if widget_class != 'QListView':
                            widget.update()
                        else:
                            # For QListView, call update with a full repaint
                            widget.update(widget.rect())
                    except Exception as e:
                        logger.debug(f"Error updating {widget_class}: {str(e)}")
                
                # Process direct children
                for child in widget.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
                    if child is not None and child != widget:  # Prevent infinite recursion
                        update_styles(child)
                        
            except Exception as e:
                logger.error(f"Error in update_styles for {widget}: {e}")
        
        # Apply style updates to all widgets
        update_styles(self)
        
        # Force a complete style refresh of the dialog itself
        try:
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()
        except Exception as e:
            logger.error(f"Error refreshing dialog styles: {e}")
        
        # Force an immediate repaint
        self.repaint()
        
        # Process any pending events to ensure UI updates
        QApplication.processEvents()
        
        logger.debug("Theme refresh complete")
        
        # Apply our custom styles after a short delay to ensure they're not overridden
        QTimer.singleShot(100, self._apply_custom_styles)
        self.update()
        self.raise_()
        self.activateWindow()
        
    def _apply_custom_styles(self):
        """Apply consistent theming and transparency to all widgets."""
        if not hasattr(self, 'main_widget') or self.main_widget is None:
            return
        
        try:
            # Set transparency attributes for main window and main widget
            for widget in [self, self.main_widget]:
                if widget:
                    widget.setAttribute(Qt.WA_TranslucentBackground, True)
                    widget.setAutoFillBackground(False)
            
            # Get current theme colors
            theme = getattr(self, 'theme', 'dark').lower()
            is_dark = theme == 'dark'
            
            # Define colors based on theme
            colors = {
                'text': '#ffffff' if is_dark else '#000000',
                'base': '#2a2a2a' if is_dark else '#f0f0f0',
                'highlight': '#60a8ff' if is_dark else '#0078d7',
                'button_bg': '#404040' if is_dark else '#e1e1e1',
                'button_hover': '#4a4a4a' if is_dark else '#f0f0f0',
                'button_pressed': '#363636' if is_dark else '#d0d0d0',
                'border': '#666666' if is_dark else '#999999',
                'disabled': '#666666',
                'disabled_bg': '#333333' if is_dark else '#e0e0e0'
            }
            
            # Base style for all widgets
            base_style = f"""
                /* Scrollbars */
                QScrollBar:vertical {{
                    background: transparent;
                    width: 10px;
                    margin: 0px;
                }}
                
                QScrollBar::handle:vertical {{
                    background: {colors['border']};
                    min-height: 20px;
                    border-radius: 5px;
                }}
                
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                    height: 0px;
                }}
                
                /* Tooltips */
                QToolTip {{
                    color: {colors['text']};
                    background-color: {colors['base']};
                    border: 1px solid {colors['border']};
                    padding: 2px;
                }}
            """
            
            # Apply base style to dialog
            self.setStyleSheet(base_style)
            
            # Apply specific styles to child widgets
            for widget in self.findChildren(QWidget):
                try:
                    if widget != self and hasattr(widget, 'setStyleSheet'):
                        widget.setStyleSheet('')
                    
                    # Set transparency attributes
                    widget.setAttribute(Qt.WA_TranslucentBackground, True)
                    widget.setAutoFillBackground(False)
                    
                    # Update widget to apply styles
                    widget.update()
                    
                except Exception as e:
                    logger.debug(f"Error styling widget {widget}: {e}")
            
            # --- MANUALLY REAPPLY COMBOBOX STYLES TO OVERRIDE RESETS ---
            combo_style = self.get_combo_style()
            if hasattr(self, 'theme_combo') and self.theme_combo:
                self.theme_combo.setStyleSheet(combo_style)
            if hasattr(self, 'window_sort_combo') and self.window_sort_combo:
                self.window_sort_combo.setStyleSheet(combo_style)
            # ----------------------------------------------------------

            # --- MANUALLY STYLE FPS ENTRY ---
            if hasattr(self, 'fps_entry') and self.fps_entry:
                if self.theme.lower() == 'light':
                    self.fps_entry.setStyleSheet("background: #ffffff; color: #000000; border: 2px solid #bbbbbb; border-radius: 4px;")
                else:
                    self.fps_entry.setStyleSheet("background: #444444; color: #ffffff; border: 2px solid #888888; border-radius: 4px;")
            # --------------------------------

            # --- BOLD PASSTHROUGH LABELS ---
            for cb in [getattr(self, 'key_passthrough_checkbox', None), getattr(self, 'aggressive_passthrough_checkbox', None)]:
                if cb and hasattr(cb, 'setFont'):
                    font = cb.font()
                    font.setBold(True)
                    cb.setFont(font)
            # --------------------------------

            # --- SLIDER STYLING ---
            slider_style = ""
            if self.theme.lower() == 'light':
                slider_style = (
                    "QSlider::groove:horizontal { border: none; height: 3px; background: #000000; margin: 0px; border-radius: 2px; } "
                    "QSlider::handle:horizontal { background: #ffffff; border: 1px solid #000000; width: 15.3px; height: 15.3px; margin: -6.65px 0; border-radius: 7.65px; } "
                    "QSlider::sub-page:horizontal { background: #000000; } "
                    "QSlider::add-page:horizontal { background: #000000; } "
                )
            else:
                slider_style = (
                    "QSlider::groove:horizontal { border: none; height: 3px; background: #ffffff; margin: 0px; border-radius: 2px; } "
                    "QSlider::handle:horizontal { background: #232323; border: 1px solid #ffffff; width: 15.3px; height: 15.3px; margin: -6.65px 0; border-radius: 7.65px; } "
                    "QSlider::sub-page:horizontal { background: #ffffff; } "
                    "QSlider::add-page:horizontal { background: #ffffff; } "
                )
            # Find all sliders and apply style
            for slider in self.findChildren(QSlider):
                slider.setStyleSheet(slider_style)
                
            # Explicitly update CircleCheckBox theme and force repaint
            for checkbox in self.findChildren(CircleCheckBox):
                checkbox.theme = self.theme
                checkbox.update()  # Force immediate repaint
            # --------------------------------

            # --- STYLE HOTKEY ENTRY LIKE FPS ENTRY ---
            if hasattr(self, 'hotkey_edit') and self.hotkey_edit:
                if self.theme.lower() == 'light':
                    self.hotkey_edit.setStyleSheet("background: #ffffff; color: #000000; border: 2px solid #bbbbbb; border-radius: 4px;")
                else:
                    self.hotkey_edit.setStyleSheet("background: #444444; color: #ffffff; border: 2px solid #888888; border-radius: 4px;")
            # --------------------------------

            # --- ENSURE ALL TEXT IN DARK THEME IS WHITE ---
            if self.theme.lower() == 'dark':
                for label in self.findChildren(QLabel):
                    label.setStyleSheet("color: #ffffff;")
                for entry in self.findChildren(QLineEdit):
                    entry.setStyleSheet(entry.styleSheet() + " color: #ffffff;")
            # --------------------------------

            # Ensure main widget is properly set up
            self.main_widget.raise_()
            self.main_widget.activateWindow()
            self.main_widget.setFocus()
            
            # Force update to apply all styles
            self.update()
            
            logger.debug("Custom styles applied successfully")
                
        except Exception as e:
            logger.error(f"Error applying custom styles: {e}", exc_info=True)

    def event(self, event):
        if event.type() == QEvent.HoverMove:
            widget = self.childAt(event.position().toPoint()) if hasattr(event, 'position') else None
            if not isinstance(widget, QLineEdit):
                self.setCursor(Qt.ArrowCursor)
        elif event.type() == QEvent.WindowActivate:
            self.raise_()
            self.activateWindow()
        return super().event(event)