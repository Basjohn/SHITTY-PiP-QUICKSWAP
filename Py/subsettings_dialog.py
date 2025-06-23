import sys
import logging
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QWidget, QHBoxLayout, QLabel, 
                              QLineEdit, QComboBox, QPushButton, QCheckBox, 
                              QKeySequenceEdit, QSlider, QFrame, QSpacerItem, QSizePolicy,
                              QStyle, QStyleOptionButton)
from PySide6.QtGui import QGuiApplication, QFont, QKeySequence, QMouseEvent, QIntValidator, QPalette, QColor, QPainter
from PySide6.QtCore import Qt, QMargins, QPoint, QEvent, Signal, QSize, QSettings, QTimer

# Import snap utilities
from snap_utils import apply_snap


logger = logging.getLogger(__name__)

def apply_snap_to(pos, size, screen_geo, threshold):
    # Placeholder for snap logic
    return pos

class DoubleClickCheckBox(QCheckBox):
    """A checkbox that requires a double-click to toggle its state."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_timer = QTimer()
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(250)  # Double-click interval in ms
        self._click_timer.timeout.connect(self._on_click_timeout)
        self._clicked = False
    
    def mousePressEvent(self, event):
        """Handle mouse press events to implement double-click requirement."""
        if event.button() == Qt.LeftButton:
            if not self._clicked:
                self._clicked = True
                self._click_timer.start()
                return  # Don't process the first click
            
            # This is the second click - process as normal
            self._clicked = False
            super().mousePressEvent(event)
    
    def _on_click_timeout(self):
        """Handle click timeout - reset the click state."""
        self._clicked = False


class SubSettingsDialog(QDialog):
    """A customizable settings dialog with theming support."""
    hotkey_settings_changed = Signal(bool, str)
    _instance = None
    
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """Get or create the single instance of the dialog."""
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
            # Clean up reference when dialog is closed
            cls._instance.finished.connect(lambda: setattr(cls, '_instance', None))
        return cls._instance  # enabled, combo

    def __init__(self, target_hwnd=None, parent=None, app_instance=None):
        """Initialize the settings dialog.
        
        Args:
            target_hwnd: The target window handle (optional)
            parent: The parent widget (optional)
            app_instance: Reference to the main application instance (optional)
        """
        super().__init__(parent)
        self.target_hwnd = target_hwnd
        self.app_instance = app_instance
        self.thumbnail_id = None
        self._border_width = 2
        self._content_margins = QMargins(5, 5, 5, 5)
        self.debug_mode = False
        self.window_sort_combo = None
        self.hotkey_checkbox = None
        self.hotkey_edit = None
        self.fps_entry = None
        
        # Initialize theme
        self.theme = "Dark"
        if self.app_instance and hasattr(self.app_instance, 'current_theme'):
            self.theme = self.app_instance.current_theme
            # Ensure theme is properly capitalized (Dark/Light) for the combobox
            self.theme = self.theme[0].upper() + self.theme[1:].lower() if self.theme else "Dark"

        self.setWindowTitle("SUBSETTINGS")
        # Set window attributes for a frameless window with rounded corners
        self.setWindowFlags(Qt.FramelessWindowHint | 
                          Qt.WindowStaysOnTopHint | 
                          Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # Enable window drop shadow with transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, False)
        
        # Set window background to transparent
        self.setStyleSheet("""
            QDialog {
                background: transparent;
            }
        """)
        
        # Disable window resizing (we're using fixed size)
        self.setSizeGripEnabled(False)
        
        # Main dialog styling with white border and rounded corners
        self.setStyleSheet("""
            QDialog {
                background: transparent;
                padding: 0px;
                margin: 0px;
            }
            
            #mainContainer {
                border: 2px solid white;
                border-radius: 10px;
                background: rgba(51, 51, 51, 0.7);
                padding: 0px;
                margin: 0px;
                background-color: rgba(42, 42, 42, 0.9);  /* Add transparency to main container */
            }
            
            #titleBar {
                background-color: #2a2a2a;  /* Solid color, no transparency */
                color: #ffffff;
                border: 1.8px solid white;
                border-bottom: 1.8px solid white;
                border-top-left-radius: 7.2px;
                border-top-right-radius: 7.2px;
                font-size: 15px;
                font-weight: bold;
                margin: 0;
                padding: 3.4px 10.8px 5.6px 10.8px;
            }
            
            [theme="light"] #titleBar {
                border-color: #000000;
            }
            
            #titleLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 17px;  /* Increased by 2pt total */
                letter-spacing: 1px;
                padding: 0;
                margin: 0;
                text-transform: uppercase;
            }
            
            #mainWidget {
                background: rgba(51, 51, 51, 0.7);  /* 70% opacity */
                border-radius: 0 0 7.2px 7.2px;
                padding: 10.8px;
                margin: 0;
            }
            
            #mainWidget > * {
                background-color: transparent;
            }
            
            /* Opacity slider with more padding */
            QSlider::groove:horizontal {
                border: 0.9px solid #666666;  /* 10% smaller border */
                height: 5.4px;  /* 10% smaller height */
                background: #555555;
                margin: 7.2px 0;  /* 10% smaller margin */
                border-radius: 2.7px;  /* 10% smaller radius */
            }
            
            QSlider::handle:horizontal {
                background: #a0a0a0;
                border: 0.9px solid #cccccc;  /* 10% smaller border */
                width: 14.4px;  /* 10% smaller width */
                height: 14.4px;  /* 10% smaller height */
                margin: -5.4px 0;  /* 10% smaller margin */
                border-radius: 7.2px;  /* 10% smaller radius */
            }
            
            /* Hotkey section with more vertical spacing */
            #hotkeyWidget {
                padding: 12px 0;
            }
            
            /* Checkbox styling */
            QCheckBox {
                spacing: 2.4px;
                color: #ffffff;
                font-size: 11px;
                margin: 1.8px 0;
            }
            
            QCheckBox:checked {
                font-weight: 800;  /* Extra bold */
                color: #ffffff;
            }
            
            QCheckBox::indicator {
                width: 14.4px;  /* 10% smaller */
                height: 14.4px;  /* 10% smaller */
                border: 1.8px solid #888888;  /* 10% smaller border */
                border-radius: 8.1px;  /* 10% smaller radius */
                background: #333333;
                padding: 1.8px;  /* 10% smaller padding */
            }
            
            QCheckBox::indicator:checked {
                border: 2px solid #ffffff;  /* Thinner border when checked */
                border-radius: 9px;  /* Increased to match unchecked state */
                padding: 3px;  /* Increased padding to make the border appear thinner */
                background: transparent;
            }
            
            QCheckBox::indicator:unchecked:hover {
                border-color: #aaaaaa;
            }
            
            [theme="light"] QCheckBox {
                color: #000000;
                padding: 2.4px 0;
                margin: 1.8px 0;
            }
            
            [theme="light"] QCheckBox:checked {
                font-weight: 800;  /* Extra bold */
                color: #000000;
            }
            
            [theme="light"] QCheckBox::indicator {
                background: #ffffff;  /* White background when OFF */
                border: 2px solid #000000;  /* Black border */
                border-radius: 9px;
                width: 16px;
                height: 16px;
            }
            
            [theme="light"] QCheckBox::indicator:checked {
                background: #000000 !important;  /* Force black fill when ON */
                border: 2px solid #000000 !important;
                border-radius: 9px;
                width: 16px;
                height: 16px;
            }
            
            [theme="light"] QCheckBox::indicator:unchecked:hover {
                border-color: #000000;
            }
            
            /* Close button styling */
            #close_btn {
                background: transparent;
                border: none;
                color: #ffffff;
                font-size: 23px;  /* 1pt larger */
                width: 30px;
                height: 30px;
                min-width: 30px;
                max-width: 30px;
                border-radius: 0;
                padding: 0;
                margin: -3px 1px 3px 0;  /* 1px higher, 1px more right */
                font-weight: normal;
            }
            
            #close_btn:hover {
                color: #000000;
                background: transparent;
            }
        """)
        
        # Set fixed size (slightly larger than original to accommodate new font sizes)
        self.setFixedSize(320, 520)  # Slightly reduced from 338x547
        
        # Center on screen
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )

        self.drag_position = None
        self.is_dragging = False
        self.snap_threshold = 30
        self.screen_margin = 5
        
        # Set up UI and load settings before connecting signals
        self.setup_ui()
        self.load_settings()
        self._connect_signals()  # Connect signals after loading settings
        self.apply_theme(self.theme)  # Initialize theme
        
        if hasattr(self, 'title_bar'):
            self.title_bar.installEventFilter(self)
    
    def setup_ui(self):
        """Set up the user interface components."""
        logger.debug("Setting up UI...")
        
        # Main container with border and rounded corners
        main_container = QWidget()
        main_container.setObjectName("mainContainer")
        
        # Main layout for the container
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create title bar
        self.title_bar = QWidget()
        self.title_bar.setObjectName("titleBar")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(12, 8, 12, 8)
        title_layout.setSpacing(0)
        
        # Title label
        self.title_label = QLabel("SUBSETTINGS")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)  # Center the title text
        
        # Close button container
        close_btn_container = QWidget()
        close_btn_container.setObjectName("closeBtnContainer")
        close_btn_layout = QHBoxLayout(close_btn_container)
        close_btn_layout.setContentsMargins(0, 0, 0, 0)
        close_btn_layout.setSpacing(0)
        
        # Close button with larger size and adjusted position
        self.close_btn = QPushButton("×")
        self.close_btn.setObjectName("close_btn")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setCursor(Qt.ArrowCursor)
        self.close_btn.clicked.connect(self.close)
        
        # Apply minimal styling
        self.close_btn.setStyleSheet("""
            QPushButton {
                font-size: 22px;
                font-weight: normal;
                color: white;
                background: transparent;
                border: none;
                padding: 0;
                margin: -1px 1px 1px 0;  /* 1px higher, 1px more right */
            }
            QPushButton:hover {
                color: black;
            }
        """)
        
        # Add close button to its container
        close_btn_layout.addWidget(self.close_btn)
        
        # Add widgets to title layout with proper spacing
        title_layout.addStretch(1)
        title_layout.addWidget(self.title_label, 2, Qt.AlignCenter)  # Centered title with more stretch
        title_layout.addStretch(1)
        title_layout.addWidget(close_btn_container, 0, Qt.AlignRight | Qt.AlignTop)
        
        # Add title bar to main container
        main_layout.addWidget(self.title_bar)
        
        # Create content widget for the settings
        self.main_widget = QWidget()
        self.main_widget.setObjectName("mainWidget")
        content_layout = QVBoxLayout(self.main_widget)
        content_layout.setContentsMargins(16, 16, 16, 16)  # Reduced margins for better space usage
        content_layout.setSpacing(4)  # Reduced spacing between all widgets to 4px
        
        # Add content widget to main container
        main_layout.addWidget(self.main_widget)
        
        # Set main container as the central widget
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(main_container)
        
        # Enable dragging on the title bar
        self.title_bar.setMouseTracking(True)
        self.title_bar.mousePressEvent = self.title_mouse_press
        self.title_bar.mouseMoveEvent = self.title_mouse_move
        self.title_bar.mouseReleaseEvent = self.title_mouse_release
        
        opacity_label = QLabel("Opacity (%)")
        opacity_label.setFont(QFont("Segoe UI", 10))
        content_layout.addWidget(opacity_label)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setMinimumHeight(24)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        content_layout.addWidget(self.opacity_slider)

        theme_label = QLabel("Theme")
        theme_label.setFont(QFont("Segoe UI", 10))
        content_layout.addWidget(theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setFont(QFont("Segoe UI", 10))
        self.theme_combo.setStyleSheet(self.get_combo_style())
        # Set current theme before connecting the signal
        self.theme_combo.setCurrentText(self.theme)
        self.theme_combo.currentTextChanged.connect(lambda text: self.apply_theme(text, from_global=False))
        content_layout.addWidget(self.theme_combo)

        window_sort_label = QLabel("Window Sort:")
        window_sort_label.setFont(QFont("Segoe UI", 10))
        content_layout.addWidget(window_sort_label)

        self.window_sort_combo = QComboBox()
        self.window_sort_combo.addItems(["Most Recently Active", "Alphabetical"])
        self.window_sort_combo.setFont(QFont("Segoe UI", 10))
        self.window_sort_combo.setStyleSheet(self.get_combo_style())
        self.window_sort_combo.currentTextChanged.connect(self._on_window_sort_changed)
        content_layout.addWidget(self.window_sort_combo)

        # Hotkey section with improved layout and spacing
        hotkey_widget = QWidget()
        hotkey_widget.setObjectName("hotkeyWidget")  # Add ID for styling
        hotkey_layout = QHBoxLayout(hotkey_widget)
        hotkey_layout.setContentsMargins(0, 8, 0, 8)  # Add vertical padding
        hotkey_layout.setSpacing(12)
        
        # Label
        hotkey_label = QLabel("Switch Hotkey:")
        hotkey_label.setFont(QFont("Segoe UI", 10))
        
        # Input field with fixed width
        self.hotkey_edit = QKeySequenceEdit()
        self.hotkey_edit.setFixedWidth(120)
        self.hotkey_edit.keySequenceChanged.connect(self._on_hotkey_setting_changed)
        
        # Checkbox with right alignment
        self.hotkey_checkbox = QCheckBox("Enabled")
        self.hotkey_checkbox.setChecked(True)
        self.hotkey_checkbox.stateChanged.connect(self._on_hotkey_setting_changed)
        
        # Add widgets with proper spacing
        hotkey_layout.addWidget(hotkey_label, 0)
        hotkey_layout.addWidget(self.hotkey_edit, 1)  # Takes remaining space
        
        # Add a spacer before the checkbox
        spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        hotkey_layout.addItem(spacer)
        
        hotkey_layout.addWidget(self.hotkey_checkbox, 0, Qt.AlignRight)
        
        content_layout.addWidget(hotkey_widget)

        fps_label = QLabel("Capture FPS (Monitor Overlay)")
        fps_label.setFont(QFont("Segoe UI", 10))
        content_layout.addWidget(fps_label)

        self.fps_entry = QLineEdit()
        self.fps_entry.setValidator(QIntValidator(1, 240))
        self.fps_entry.setText("60")
        self.fps_entry.setInputMethodHints(Qt.ImhDigitsOnly)
        self.fps_entry.textChanged.connect(self._on_fps_changed)
        content_layout.addWidget(self.fps_entry)

        # Add checkboxes with minimal spacing
        self.click_through_checkbox = QCheckBox("Click-through mode")
        self.click_through_checkbox.setFont(QFont("Segoe UI", 10))
        self.click_through_checkbox.stateChanged.connect(self._on_click_through_changed)
        content_layout.addWidget(self.click_through_checkbox, alignment=Qt.AlignTop)
        
        # Add EXPERIMENTAL section
        content_layout.addSpacing(8)
        
        # Add a horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #666666;")
        line.setFixedHeight(1)
        content_layout.addWidget(line)
        
        # Container for experimental section
        experimental_container = QWidget()
        experimental_layout = QVBoxLayout(experimental_container)
        experimental_layout.setContentsMargins(0, 4, 0, 8)
        experimental_layout.setSpacing(2)
        
        # EXPERIMENTAL label with underline
        experimental_label = QLabel("EXPERIMENTAL")
        font = QFont("Segoe UI", 9)
        font.setBold(True)
        font.setUnderline(True)
        experimental_label.setFont(font)
        experimental_label.setStyleSheet("color: #ff6b6b;")
        experimental_label.setAlignment(Qt.AlignCenter)
        
        # Warning message
        warning_label = QLabel("DO NOT USE WITH ONLINE GAMING OR GACHA")
        warning_font = QFont("Segoe UI", 8)
        warning_font.setBold(True)
        warning_label.setFont(warning_font)
        warning_label.setStyleSheet("color: #ff6b6b;")
        warning_label.setAlignment(Qt.AlignCenter)
        
        # Add labels to container
        experimental_layout.addWidget(experimental_label)
        experimental_layout.addWidget(warning_label)
        
        # Add container to main layout
        content_layout.addWidget(experimental_container)
        
        # Add the experimental checkboxes with double-click requirement
        self.key_passthrough_checkbox = DoubleClickCheckBox("Enable key passthrough (Window Overlay)")
        self.key_passthrough_checkbox.setFont(QFont("Segoe UI", 10))
        self.key_passthrough_checkbox.stateChanged.connect(self._on_key_passthrough_changed)
        content_layout.addWidget(self.key_passthrough_checkbox, alignment=Qt.AlignTop)
        
        self.aggressive_passthrough_checkbox = DoubleClickCheckBox("Aggressive Key Passthrough")
        self.aggressive_passthrough_checkbox.setFont(QFont("Segoe UI", 10))
        self.aggressive_passthrough_checkbox.stateChanged.connect(self._on_aggressive_passthrough_changed)
        content_layout.addWidget(self.aggressive_passthrough_checkbox)

        content_layout.addStretch()
        
        # Add main widget to main layout
        main_layout.addWidget(self.main_widget)
        main_layout.setStretch(1, 1)
        
        self._connect_signals()
        
        logger.debug("UI setup complete")

    def save_settings(self):
        """Save all settings to the application settings."""
        if not hasattr(self, 'app_instance') or not hasattr(self.app_instance, 'settings'):
            return
            
        settings = self.app_instance.settings
        
        # Save checkbox states
        if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
            settings.setValue("key_passthrough_enabled", 
                           self.key_passthrough_checkbox.isChecked())
                           
        if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
            settings.setValue("aggressive_key_passthrough", 
                           self.aggressive_passthrough_checkbox.isChecked())
                           
        if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
            settings.setValue("click_through_enabled", 
                           self.click_through_checkbox.isChecked())
        
        # Save other settings
        if hasattr(self, 'opacity_slider') and self.opacity_slider:
            settings.setValue("overlay_opacity", self.opacity_slider.value())
            
        if hasattr(self, 'window_sort_combo') and self.window_sort_combo:
            settings.setValue("window_sort_order", self.window_sort_combo.currentText())
            
        if hasattr(self, 'fps_entry') and self.fps_entry:
            try:
                fps = int(self.fps_entry.text())
                settings.setValue("capture_fps", fps)
            except (ValueError, AttributeError):
                pass
                
        # Save hotkey settings
        if hasattr(self, 'hotkey_checkbox') and hasattr(self, 'hotkey_edit'):
            settings.setValue("hotkey_enabled", self.hotkey_checkbox.isChecked())
            settings.setValue("hotkey_sequence", self.hotkey_edit.keySequence().toString())
        
        # Save theme
        if hasattr(self, 'theme_combo') and self.theme_combo:
            settings.setValue("theme", self.theme_combo.currentText().lower())
        
        # Ensure settings are written to disk
        settings.sync()
        logger.debug("Settings saved successfully")

    def load_settings(self):
        """Load settings from the application settings."""
        if not hasattr(self, 'app_instance') or not hasattr(self.app_instance, 'settings'):
            return
            
        settings = self.app_instance.settings
        
        # Block signals while loading to prevent unnecessary updates
        signal_blocks = {}
        
        # Store signal block states
        if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
            signal_blocks['key_passthrough'] = self.key_passthrough_checkbox.blockSignals(True)
            
        if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
            signal_blocks['aggressive_passthrough'] = self.aggressive_passthrough_checkbox.blockSignals(True)
            
        if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
            signal_blocks['click_through'] = self.click_through_checkbox.blockSignals(True)
        
        try:
            # Load checkbox states with explicit False defaults
            if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                if settings.contains("key_passthrough_enabled"):
                    enabled = settings.value("key_passthrough_enabled", "false").lower() == "true"
                else:
                    enabled = False
                    settings.setValue("key_passthrough_enabled", False)
                self.key_passthrough_checkbox.setChecked(enabled)
                
            if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                if settings.contains("aggressive_key_passthrough"):
                    aggressive = settings.value("aggressive_key_passthrough", "false").lower() == "true"
                else:
                    aggressive = False
                    settings.setValue("aggressive_key_passthrough", False)
                self.aggressive_passthrough_checkbox.setChecked(aggressive)
                
            if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
                if settings.contains("click_through_enabled"):
                    click_through = settings.value("click_through_enabled", "false").lower() == "true"
                else:
                    click_through = False
                    settings.setValue("click_through_enabled", False)
                self.click_through_checkbox.setChecked(click_through)
            
            # Load other settings
            if hasattr(self, 'opacity_slider') and self.opacity_slider:
                opacity = int(settings.value("overlay_opacity", 100))
                self.opacity_slider.setValue(opacity)
                
            if hasattr(self, 'window_sort_combo') and self.window_sort_combo:
                sort_order = settings.value("window_sort_order", "Most Recently Active")
                index = self.window_sort_combo.findText(sort_order)
                if index >= 0:
                    self.window_sort_combo.setCurrentIndex(index)
                    
            if hasattr(self, 'fps_entry') and self.fps_entry:
                fps = settings.value("capture_fps", "30")
                self.fps_entry.setText(str(fps))
                
            # Load hotkey settings
            if hasattr(self, 'hotkey_checkbox') and hasattr(self, 'hotkey_edit'):
                hotkey_enabled = settings.value("hotkey_enabled", "true").lower() == "true"
                hotkey_sequence = settings.value("hotkey_sequence", "Ctrl+Alt+Space")
                self.hotkey_checkbox.setChecked(hotkey_enabled)
                self.hotkey_edit.setKeySequence(QKeySequence(hotkey_sequence))
            
            # Load theme
            if hasattr(self, 'theme_combo') and self.theme_combo:
                theme = settings.value("theme", "dark").capitalize()
                index = self.theme_combo.findText(theme, Qt.MatchFixedString)
                if index >= 0:
                    self.theme_combo.setCurrentIndex(index)
                    
            logger.debug("Settings loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading settings: {e}", exc_info=True)
            
        finally:
            # Restore signal block states
            if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                self.key_passthrough_checkbox.blockSignals(
                    signal_blocks.get('key_passthrough', False))
                    
            if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                self.aggressive_passthrough_checkbox.blockSignals(
                    signal_blocks.get('aggressive_passthrough', False))
                    
            if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
                self.click_through_checkbox.blockSignals(
                    signal_blocks.get('click_through', False))

    def _connect_signals(self):
        """Connect all signal handlers after initial setup."""
        # Connect theme combo box
        if hasattr(self, 'theme_combo'):
            self.theme_combo.currentTextChanged.connect(
                lambda text: self.apply_theme(text, from_global=False))
        
        # Connect passthrough checkboxes
        if hasattr(self, 'key_passthrough_checkbox'):
            self.key_passthrough_checkbox.stateChanged.connect(self._on_key_passthrough_changed)
        if hasattr(self, 'aggressive_passthrough_checkbox'):
            self.aggressive_passthrough_checkbox.stateChanged.connect(self._on_aggressive_passthrough_changed)
        if hasattr(self, 'click_through_checkbox'):
            self.click_through_checkbox.stateChanged.connect(self._on_click_through_changed)
            
        # Connect other UI elements
        if hasattr(self, 'opacity_slider'):
            self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        if hasattr(self, 'window_sort_combo'):
            self.window_sort_combo.currentTextChanged.connect(self._on_window_sort_changed)
        if hasattr(self, 'fps_entry') and hasattr(self.fps_entry, 'textChanged'):
            self.fps_entry.textChanged.connect(self._on_fps_changed)
        if hasattr(self, 'hotkey_checkbox') and hasattr(self, 'hotkey_edit'):
            self.hotkey_checkbox.stateChanged.connect(self._on_hotkey_setting_changed)
            self.hotkey_edit.keySequenceChanged.connect(self._on_hotkey_setting_changed)
            
        # Connect save settings on close
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
    
    def apply_theme(self, theme, from_global=False):
        """
        Apply the specified theme to the dialog.
        
        Args:
            theme (str): Name of the theme to apply ('dark' or 'light')
            from_global (bool): Whether this is being called from a global theme change
        """
        
        if hasattr(self, '_applying_theme') and self._applying_theme:
            return

        self._applying_theme = True
        try:
            logger.debug(f"Applying theme: {theme}, from_global: {from_global}")
            self.theme = theme.lower() if theme else "dark"
            
            # Set theme colors
            if self.theme == "light":
                bg_color = "#f0f0f0"
                fg_color = "#000000"
                border_color = "#cccccc"
                title_bg = "#e0e0e0"
                title_fg = "#000000"
                input_bg = "#ffffff"
                line_edit_bg = "#ffffff"
                line_edit_fg = "#000000"
                line_edit_border = "#999999"
                combo_bg = "#ffffff"
                combo_fg = "#000000"
                combo_border = "#999999"
                combo_dropdown_bg = "#ffffff"
                slider_groove = "#cccccc"
                slider_handle = "#505050"
                slider_handle_border = "#333333"
                slider_handle_hover = "#707070"
                button_bg = "#e1e1e1"
                button_text = "#000000"
                button_border = "#999999"
                button_hover = "#f0f0f0"
                button_pressed = "#d0d0d0"
                close_btn_hover = "#ff6b6b"
                close_btn_pressed = "#ff3b3b"
            else:  # Dark theme
                bg_color = "#333333"
                fg_color = "#ffffff"
                border_color = "#666666"
                title_bg = "#2a2a2a"
                title_fg = "#ffffff"
                input_bg = "#404040"
                line_edit_bg = "#404040"
                line_edit_fg = "#ffffff"
                line_edit_border = "#666666"
                combo_bg = "#404040"
                combo_fg = "#ffffff"
                combo_border = "#666666"
                combo_dropdown_bg = "#404040"
                slider_groove = "#555555"
                slider_handle = "#a0a0a0"
                slider_handle_border = "#cccccc"
                slider_handle_hover = "#c0c0c0"
                button_bg = "#404040"
                button_text = "#ffffff"
                button_border = "#666666"
                button_hover = "#4a4a4a"
                button_pressed = "#363636"
                close_btn_hover = "#ff6b6b"
                close_btn_pressed = "#ff3b3b"

            # Get checkbox style based on theme
            checkbox_style = self.get_checkbox_style(theme)

            # Build the stylesheet
            style_sheet = f"""
                #mainContainer {{
                    border: 2px solid {border_color};
                    border-radius: 8px;
                    background-color: {bg_color};
                }}
                #titleBar {{
                    background-color: {title_bg};
                    color: {title_fg};
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    padding: 6px 12px 4px 12px;
                    border: none;
                }}
                #mainWidget {{
                    background-color: {bg_color};
                    border-bottom-left-radius: 6px;
                    border-bottom-right-radius: 6px;
                    padding: 12px;
                }}
                QLabel {{
                    color: {fg_color};
                    background-color: transparent;
                }}
                {checkbox_style}
                #close_btn {{
                    background-color: transparent;
                    border: none;
                    color: {title_fg};
                    font-size: 18px;
                    font-weight: normal;
                    width: 24px;
                    height: 24px;
                    min-width: 24px;
                    max-width: 24px;
                    border-radius: 0;
                    padding: 0;
                    margin: 0;
                }}
                #close_btn:hover {{
                    color: {close_btn_hover};
                    font-weight: bold;
                }}
                QLineEdit, QKeySequenceEdit, QLineEdit#fps_entry {{
                    background-color: {line_edit_bg};
                    color: {line_edit_fg};
                    border: 1px solid {line_edit_border};
                    border-radius: 4px;
                    padding: 4px 8px;
                    min-height: 24px;
                    selection-background-color: {fg_color};
                    selection-color: {bg_color};
                }}
                
                QLineEdit:focus, QKeySequenceEdit:focus, QLineEdit#fps_entry:focus {{
                    border: 1px solid {fg_color};
                }}
                QComboBox {{
                    background-color: {combo_bg};
                    color: {combo_fg};
                    border: 1px solid {combo_border};
                    border-radius: 4px;
                    padding: 4px 8px;
                }}
                QComboBox::drop-down {{
                    border: none;
                }}
                QSlider::groove:horizontal {{
                    border: 1px solid {slider_groove};
                    height: 4px;
                    background: {slider_groove};
                    margin: 0px;
                    border-radius: 2px;
                }}
                QSlider::handle:horizontal {{
                    background: {slider_handle};
                    border: 1px solid {slider_handle_border};
                    width: 12px;
                    height: 12px;
                    margin: -4px 0;
                    border-radius: 6px;
                }}
                QSlider::handle:horizontal:hover {{
                    background: {slider_handle_hover};
                }}
                QSlider::sub-page:horizontal {{
                    background: {slider_handle};
                    border: 1px solid {slider_groove};
                    height: 4px;
                    border-radius: 2px;
                }}
                QSlider::add-page:horizontal {{
                    background: {slider_groove};
                    border: 1px solid {slider_groove};
                    height: 4px;
                    border-radius: 2px;
                }}
                QSlider {{
                    min-height: 24px;
                }}
                QPushButton {{
                    background-color: {button_bg};
                    color: {button_text};
                    border: 1px solid {button_border};
                    border-radius: 4px;
                    padding: 4px 12px;
                    min-width: 80px;
                }}
                QPushButton:hover {{
                    background-color: {button_hover};
                }}
                QPushButton:pressed {{
                    background-color: {button_pressed};
                }}
            """
            
            # Apply the stylesheet to the main widget
            self.main_widget.setStyleSheet(style_sheet)
            
            # Update the palette for proper theming
            palette = self.palette()
            palette.setColor(QPalette.Window, QColor(bg_color))
            palette.setColor(QPalette.WindowText, QColor(fg_color))
            palette.setColor(QPalette.Base, QColor(input_bg))
            palette.setColor(QPalette.AlternateBase, QColor(button_hover))
            palette.setColor(QPalette.Text, QColor(fg_color))
            palette.setColor(QPalette.Button, QColor(button_bg))
            palette.setColor(QPalette.ButtonText, QColor(button_text))
            palette.setColor(QPalette.BrightText, QColor("#ffffff"))
            palette.setColor(QPalette.Highlight, QColor("#0078d7"))
            palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
            self.setPalette(palette)
            
            # Update the close button text
            if hasattr(self, 'close_btn'):
                self.close_btn.setText("×")
            
            # Update theme in app instance if needed
            if self.app_instance and not from_global:
                self.app_instance.current_theme = theme
                if hasattr(self.app_instance, 'settings'):
                    self.app_instance.settings.setValue("theme", theme)
                    self.app_instance.settings.sync()
                if hasattr(self.app_instance, 'apply_theme_globally'):
                    self.app_instance.apply_theme_globally(theme)
            
            # Update the theme combo box if this is not from a global theme change
            if not from_global and hasattr(self, 'theme_combo'):
                self.theme_combo.blockSignals(True)
                self.theme_combo.setCurrentText(theme)
                self.theme_combo.blockSignals(False)
                
        except Exception as e:
            logger.error(f"Error applying theme: {e}", exc_info=True)
        finally:
            self._applying_theme = False
    
    def get_checkbox_style(self, theme):
        """Get checkbox styling based on theme."""
        if theme.lower() == "light":
            return """
                QCheckBox {
                    color: #000000;
                    spacing: 4px;
                    font-size: 11px;
                    padding: 4px 0;
                }
                
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 2px solid #000000;
                    border-radius: 9px;
                    background: #ffffff;
                }
                
                QCheckBox::indicator:checked {
                    background: #000000 !important;
                    border: 2px solid #000000 !important;
                }
                
                QCheckBox::indicator:unchecked:hover {
                    border-color: #333333;
                }
                
                QCheckBox:checked {
                    font-weight: bold;
                }
            """
        else:  # Dark theme
            return """
                QCheckBox {
                    color: #ffffff;
                    spacing: 4px;
                    font-size: 11px;
                    padding: 4px 0;
                }
                
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 2px solid #888888;
                    border-radius: 9px;
                    background: #333333;
                }
                
                QCheckBox::indicator:checked {
                    border: 2px solid #ffffff;
                    background: transparent;
                }
                
                QCheckBox::indicator:unchecked:hover {
                    border-color: #aaaaaa;
                }
                
                QCheckBox:checked {
                    font-weight: bold;
                }
            """

    def _on_key_passthrough_changed(self, state):
        """Handle key passthrough checkbox state change."""
        if not self.app_instance or not hasattr(self.app_instance, 'settings'):
            return
            
        # Block signals to prevent feedback loop
        key_passthrough_blocked = False
        aggressive_blocked = False
        
        try:
            # Get the checkboxes
            key_passthrough_cb = getattr(self, 'key_passthrough_checkbox', None)
            aggressive_cb = getattr(self, 'aggressive_passthrough_checkbox', None)
            
            # Block signals if checkboxes exist
            if key_passthrough_cb:
                key_passthrough_cb.blockSignals(True)
                key_passthrough_blocked = True
            if aggressive_cb:
                aggressive_cb.blockSignals(True)
                aggressive_blocked = True
            
            # Get the new state
            enabled = state == Qt.Checked
            aggressive = False  # When enabling regular passthrough, disable aggressive mode
            
            # Save the states
            settings = self.app_instance.settings
            settings.setValue("key_passthrough_enabled", enabled)
            settings.setValue("aggressive_key_passthrough", aggressive)
            settings.sync()
            
            # Update the checkboxes
            if key_passthrough_cb:
                key_passthrough_cb.setChecked(enabled)
            if aggressive_cb:
                aggressive_cb.setChecked(aggressive)
            
            logger.debug(f"Key passthrough: enabled={enabled}, aggressive={aggressive}")
            
            # Emit signal to update window overlays
            if hasattr(self.app_instance, 'key_passthrough_setting_changed'):
                self.app_instance.key_passthrough_setting_changed.emit(enabled, aggressive)
                
        except Exception as e:
            logger.error(f"Error updating key passthrough setting: {e}", exc_info=True)
        finally:
            # Re-enable signals
            if key_passthrough_blocked and hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                self.key_passthrough_checkbox.blockSignals(False)
            if aggressive_blocked and hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                self.aggressive_passthrough_checkbox.blockSignals(False)
                
            # Ensure settings are saved
            if hasattr(self, 'save_settings'):
                self.save_settings()
    
    def _on_aggressive_passthrough_changed(self, state):
        """Handle aggressive key passthrough checkbox state change."""
        if not self.app_instance or not hasattr(self.app_instance, 'settings'):
            return
            
        # Block signals to prevent feedback loop
        key_passthrough_blocked = False
        aggressive_blocked = False
        
        try:
            # Get the checkboxes
            key_passthrough_cb = getattr(self, 'key_passthrough_checkbox', None)
            aggressive_cb = getattr(self, 'aggressive_passthrough_checkbox', None)
            
            # Block signals if checkboxes exist
            if key_passthrough_cb:
                key_passthrough_cb.blockSignals(True)
                key_passthrough_blocked = True
            if aggressive_cb:
                aggressive_cb.blockSignals(True)
                aggressive_blocked = True
            
            # Get the new state
            aggressive = state == Qt.Checked
            enabled = aggressive  # When enabling aggressive mode, also enable regular passthrough
            
            # Save the states
            settings = self.app_instance.settings
            settings.setValue("key_passthrough_enabled", enabled)
            settings.setValue("aggressive_key_passthrough", aggressive)
            settings.sync()
            
            # Update the checkboxes
            if key_passthrough_cb:
                key_passthrough_cb.setChecked(enabled)
            if aggressive_cb:
                aggressive_cb.setChecked(aggressive)
            
            logger.debug(f"Aggressive key passthrough: enabled={enabled}, aggressive={aggressive}")
            
            # Emit signal to update window overlays
            if hasattr(self.app_instance, 'key_passthrough_setting_changed'):
                self.app_instance.key_passthrough_setting_changed.emit(enabled, aggressive)
                
        except Exception as e:
            logger.error(f"Error updating aggressive key passthrough setting: {e}", exc_info=True)
        finally:
            # Re-enable signals
            if key_passthrough_blocked and hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                self.key_passthrough_checkbox.blockSignals(False)
            if aggressive_blocked and hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                self.aggressive_passthrough_checkbox.blockSignals(False)
                
            # Ensure settings are saved
            if hasattr(self, 'save_settings'):
                self.save_settings()

    def _on_click_through_changed(self, state):
        """Handle click-through checkbox state change."""
        if not self.app_instance or not hasattr(self.app_instance, 'toggle_click_through_mode'):
            return
            
        try:
            # Block signals to prevent feedback loop
            if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
                self.click_through_checkbox.blockSignals(True)
            
            # Toggle the click-through mode
            self.app_instance.toggle_click_through_mode()
            
            # Get the new state to update the checkbox
            new_state = self.app_instance.settings.value("click_through_enabled", False, type=bool)
            
            # Update the checkbox state if needed
            if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
                self.click_through_checkbox.setChecked(new_state)
            
            logger.debug(f"Click-through mode {'enabled' if new_state else 'disabled'})")
            
        except Exception as e:
            logger.error(f"Error in _on_click_through_changed: {e}", exc_info=True)
        finally:
            # Always unblock signals
            if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
                self.click_through_checkbox.blockSignals(False)
            
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
                    # Show a warning but don't prevent typing
                    logger.warning(f"FPS value {fps} is outside the allowed range (1-240)")
            except (ValueError, TypeError) as e:
                logger.debug(f"Invalid FPS value: {text}")

    def load_settings(self):
        if not self.app_instance or not hasattr(self.app_instance, 'settings'):
            logger.error("SubSettingsDialog: app_instance or settings not available.")
            return

        app_settings = self.app_instance.settings
        opacity = app_settings.value("opacity", 95, type=int)
        theme = app_settings.value("theme", "Dark", type=str)
        sort_order = app_settings.value("windowSortOrder", "Most Recently Active", type=str)
        hotkey_enabled = app_settings.value("SwitchHotkeyEnabled", True, type=bool)
        hotkey_combo = app_settings.value("SwitchHotkeySequence", "Ctrl+Shift+Q", type=str)
        fps = app_settings.value("capture_fps", 60, type=int)
        click_through = app_settings.value("click_through_enabled", False, type=bool)
        key_passthrough = app_settings.value("key_passthrough_enabled", True, type=bool)
        aggressive_passthrough = app_settings.value("aggressive_key_passthrough", False, type=bool)

        # Block signals while loading to prevent unnecessary updates
        if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
            self.key_passthrough_checkbox.blockSignals(True)
        if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
            self.aggressive_passthrough_checkbox.blockSignals(True)
        if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
            self.click_through_checkbox.blockSignals(True)

        try:
            self.opacity_slider.setValue(opacity)
            self.theme_combo.setCurrentText(theme)
            if self.window_sort_combo:
                self.window_sort_combo.setCurrentText(sort_order)
            if self.hotkey_checkbox:
                self.hotkey_checkbox.setChecked(hotkey_enabled)
            if self.hotkey_edit:
                seq = QKeySequence(hotkey_combo.replace("ctrl", "Ctrl").replace("alt", "Alt").replace("shift", "Shift").replace("win", "Meta"))
                self.hotkey_edit.setKeySequence(seq)
            if self.fps_entry:
                self.fps_entry.setText(str(fps))
            if self.click_through_checkbox:
                self.click_through_checkbox.setChecked(click_through)
            if self.key_passthrough_checkbox:
                self.key_passthrough_checkbox.setChecked(key_passthrough)
            if self.aggressive_passthrough_checkbox:
                self.aggressive_passthrough_checkbox.setChecked(aggressive_passthrough)
                
            logger.debug(f"Loaded settings - key_passthrough: {key_passthrough}, aggressive: {aggressive_passthrough}")
            
            # Emit signal to update key passthrough state
            if hasattr(self.app_instance, 'key_passthrough_setting_changed'):
                self.app_instance.key_passthrough_setting_changed.emit(key_passthrough, aggressive_passthrough)
                
        except Exception as e:
            logger.error(f"Error loading settings: {e}", exc_info=True)
        finally:
            # Re-enable signals
            if hasattr(self, 'key_passthrough_checkbox') and self.key_passthrough_checkbox:
                self.key_passthrough_checkbox.blockSignals(False)
            if hasattr(self, 'aggressive_passthrough_checkbox') and self.aggressive_passthrough_checkbox:
                self.aggressive_passthrough_checkbox.blockSignals(False)
            if hasattr(self, 'click_through_checkbox') and self.click_through_checkbox:
                self.click_through_checkbox.blockSignals(False)

    def _format_hotkey_for_keyboard(self, seq_str):
        return seq_str.replace("+", "+").replace("Ctrl", "ctrl").replace("Alt", "alt").replace("Shift", "shift").replace("Meta", "win").lower()

    def save_settings(self):
        if not self.app_instance or not hasattr(self.app_instance, 'settings'):
            logger.error("SubSettingsDialog: app_instance or settings not available for saving.")
            return
            
        app_settings = self.app_instance.settings
        
        # Save hotkey settings
        hotkey_enabled = self.hotkey_checkbox.isChecked()
        if hasattr(self, 'hotkey_edit') and self.hotkey_edit:
            raw_seq = self.hotkey_edit.keySequence().toString(QKeySequence.NativeText)
            hotkey_combo = self._format_hotkey_for_keyboard(raw_seq)
        else:
            hotkey_combo = self._format_hotkey_for_keyboard("Ctrl+Shift+Q")
            
        # Save key passthrough settings
        key_passthrough = False
        aggressive_passthrough = False
        if hasattr(self, 'key_passthrough_checkbox'):
            key_passthrough = self.key_passthrough_checkbox.isChecked()
        if hasattr(self, 'aggressive_passthrough_checkbox'):
            aggressive_passthrough = self.aggressive_passthrough_checkbox.isChecked()
        
        # Save all settings
        app_settings.setValue("SwitchHotkeyEnabled", hotkey_enabled)
        app_settings.setValue("SwitchHotkeySequence", hotkey_combo)
        app_settings.setValue("key_passthrough_enabled", key_passthrough)
        app_settings.setValue("aggressive_key_passthrough", aggressive_passthrough)
        app_settings.sync()
        
        logger.info(f"Settings saved: SwitchHotkeyEnabled={hotkey_enabled}, "
                   f"SwitchHotkeySequence='{hotkey_combo}', "
                   f"key_passthrough_enabled={key_passthrough}, "
                   f"aggressive_key_passthrough={aggressive_passthrough}")
                   
        # Emit signals
        self.hotkey_settings_changed.emit(hotkey_enabled, hotkey_combo)
        if hasattr(self.app_instance, 'key_passthrough_setting_changed'):
            self.app_instance.key_passthrough_setting_changed.emit(key_passthrough, aggressive_passthrough)

    def _on_hotkey_setting_changed(self, *args):
        self.save_settings()

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
            return (
                'QComboBox { background-color: #ffffff; color: #000000; border: 1px solid #cccccc; border-radius: 4px; padding: 5px 8px; font-family: "Segoe UI"; font-size: 12px; min-height: 24px; } '
                'QComboBox:hover { border-color: #999999; } '
                'QComboBox::drop-down { width: 20px; border: none; background: #ffffff; } '
                'QComboBox::down-arrow { image: none; width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #000000; } '
                'QComboBox::drop-down:hover { background-color: #f0f0f0; } '
                'QComboBox QAbstractItemView { background-color: #ffffff; color: #000000; border: 1px solid #cccccc; selection-background-color: #0078d7; selection-color: white; }'
            )
        else:
            return (
                'QComboBox { background-color: #404040; color: #ffffff; border: 1px solid #666666; border-radius: 4px; padding: 5px 8px; font-family: "Segoe UI"; font-size: 12px; min-height: 24px; } '
                'QComboBox:hover { border-color: #999999; } '
                'QComboBox::drop-down { width: 20px; border: none; background: #404040; } '
                'QComboBox::down-arrow { image: none; width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 6px solid #ffffff; } '
                'QComboBox::drop-down:hover { background-color: #4a4a4a; } '
                'QComboBox QAbstractItemView { background-color: #404040; color: #ffffff; border: 1px solid #666666; selection-background-color: #0078d7; selection-color: white; }'
            )

    def _on_opacity_changed(self, value):
        if self.app_instance:
            opacity = value / 100.0
            if hasattr(self.app_instance, 'set_all_overlays_opacity'):
                self.app_instance.set_all_overlays_opacity(opacity)
            if hasattr(self.app_instance, 'settings'):
                self.app_instance.settings.setValue("opacity", value)
                logger.debug(f"Opacity changed to: {value}%")

    def _on_window_sort_changed(self, sort_order):
        if self.app_instance:
            if hasattr(self.app_instance, 'update_window_sort_order'):
                self.app_instance.update_window_sort_order(sort_order)
            if hasattr(self.app_instance, 'settings'):
                self.app_instance.settings.setValue("windowSortOrder", sort_order)
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

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, False)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self.apply_theme(self.theme))
        self.update()
        self.raise_()
        self.activateWindow()

    def event(self, event):
        if event.type() == QEvent.HoverMove:
            widget = self.childAt(event.position().toPoint()) if hasattr(event, 'position') else None
            if not isinstance(widget, QLineEdit):
                self.setCursor(Qt.ArrowCursor)
        elif event.type() == QEvent.WindowActivate:
            self.raise_()
            self.activateWindow()
        return super().event(event)