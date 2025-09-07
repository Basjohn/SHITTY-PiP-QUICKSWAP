from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFrame,
    QLabel, QComboBox, QStackedWidget, QSizePolicy, QHBoxLayout,
    QPushButton, QButtonGroup, QSpacerItem, QSlider, QSystemTrayIcon,
    QLineEdit, QCheckBox, QRadioButton, QTextEdit, QPlainTextEdit, QAbstractSpinBox,
    QGraphicsDropShadowEffect
)

from PySide6.QtCore import Qt, QPoint, QEvent, QRect, QSize, QDir
from PySide6.QtGui import QIcon, QPixmap, QColor, QPainter, QPen
import win32gui

from core.window.enumerator import WindowEnumerator
from core.logging import get_logger
from core.threading import ThreadManager
from utils.theme.theme_manager import get_theme_manager
from utils.window.behavior import WindowBehaviorManager
from core.settings.settings_manager import SettingsManager
from core.opacity.manager import get_opacity_manager
from core.ui.tray import SystemTrayManager
from utils.monitor_utils import get_all_monitors
from core.graphics import get_overlay_manager
from core.graphics.backends import BackendType

# Constants
MIN_WINDOW_WIDTH = 700      # pixels (was 675)
MIN_WINDOW_HEIGHT = 540     # pixels (was 500)
DRAGGABLE_MARGIN = 1       # pixels from top for draggable region

# Badge behavior constants (logical pixels / ratios)
BADGE_SCALE_FACTOR = 0.9            # 20% smaller than baseline 400–500 logical px
BADGE_LEFT_SHIFT_FACTOR = 0.55      # shift badge left by 55% of its width (relative)
BADGE_UNDERLAP_FACTOR = 0.03        # slightly deeper underlap (3% of height, min 3px)

# Set up logging
logger = get_logger(__name__)


class BorderPaintOverlay(QWidget):
    """Top-layer sibling overlay that reliably paints the window border above children."""
    def __init__(self, parent=None, radius: int = 8, width: int = 2, color: QColor = QColor(255, 255, 255, 255)):
        super().__init__(parent)
        self._radius = radius
        self._width = width
        self._color = color
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_color(self, color: QColor):
        """Update border color and repaint."""
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(self._color)
        pen.setWidth(self._width)
        painter.setPen(pen)
        # Inset by half pen width to keep stroke fully visible
        inset = max(1, int(self._width / 2))
        rect = self.rect().adjusted(inset, inset, -inset, -inset)
        painter.drawRoundedRect(rect, self._radius, self._radius)


class MainDialog(QMainWindow):
    """Main settings panel window with theme support.
    
    QMainWindow-based dialog with centralized window behavior management
    for dragging, resizing, and snapping.
    """
    
    def __init__(self, app_instance=None):
        """Initialize the settings panel with a basic frame and theme support.
        
        Args:
            app_instance: Reference to the main application instance (optional)
        """
        # Store app instance reference
        self.app_instance = app_instance
        
        # Set up logging FIRST
        global logger
        logger = get_logger(__name__)
        logger.info("Initializing MainDialog...")
        
        # Initialize managers
        self.theme_manager = get_theme_manager()
        self.settings_manager = SettingsManager()
        self.window_enumerator = WindowEnumerator()
        self.opacity_manager = get_opacity_manager()
        self.overlay_manager = get_overlay_manager()
        # React to theme/style changes to refresh visuals that depend on QSS timing
        try:
            if hasattr(self.theme_manager, 'theme_changed'):
                self.theme_manager.theme_changed.connect(self._on_theme_changed)
            if hasattr(self.theme_manager, 'style_changed'):
                self.theme_manager.style_changed.connect(self._on_style_changed)
        except Exception as e:
            logger.error(f"Failed to connect theme signals: {e}", exc_info=True)
        
        # Initialize the base class
        super().__init__(None)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        # Set window flags - frameless with custom title bar
        self.setWindowFlags(
            Qt.Window |
            Qt.FramelessWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        # Set minimum window size
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        
        # Window behavior is initialized centrally in _setup_window_behavior()

        # Build UI
        self._build_ui()
        
        # Apply theme - use dark.qss as default
        self.apply_theme("dark")
        
        # Center window
        self._center_window()
        
        # Populate comboboxes
        self._populate_window_combobox()
        self._populate_monitor_combobox()
        
        # Set initial combobox visibility
        self._update_combobox_visibility(self.window_mode_button.isChecked())
        
        # Restore window geometry
        self._restore_window_geometry()
        
        # Set up system tray
        self._setup_system_tray()
        
        # Set up badge system
        self._setup_badge_system()
        
        # Initialize window behavior manager
        self._setup_window_behavior()
        
        # Connect signals
        self._connect_signals()
        
        # Enable mouse tracking for all widgets
        self._enable_mouse_tracking()

    def keyPressEvent(self, event):
        """Quit the application when ESC is pressed."""
        try:
            if event and event.key() == Qt.Key_Escape:
                event.accept()
                # Use the same cleanup path as tray quit / window close
                try:
                    self._on_tray_quit()
                except Exception:
                    pass
                return
        except Exception:
            pass
        super().keyPressEvent(event)
        
        logger.info("MainDialog initialization complete")

    def _setup_border_overlay(self):
        """Create paint-based top-layer border overlay as a sibling of main_frame."""
        try:
            # Match radius/width to theme (8px radius, 2px stroke)
            self.border_overlay = BorderPaintOverlay(self, radius=8, width=2)
            # Geometry set in show/resize events
            self.border_overlay.hide()
            # Apply current theme border color
            self._apply_border_color()
        except Exception as e:
            logger.error(f"Error setting up border overlay: {e}", exc_info=True)
    
    def _build_ui(self):
        """Build the main user interface."""
        # Create main frame
        self.main_frame = QFrame()
        self.main_frame.setObjectName("main_frame")
        self.main_frame.setMouseTracking(True)
        
        # Set main frame as central widget
        self.setCentralWidget(self.main_frame)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.main_frame.setLayout(main_layout)
        
        # Create content widget
        self.content_widget = QWidget()
        self.content_widget.setObjectName("contentWidget")
        
        # Content layout
        self.content_layout = QVBoxLayout()
        # Remove top margin so top controls sit flush under the title bar
        self.content_layout.setContentsMargins(10, 0, 10, 10)
        self.content_layout.setSpacing(0)
        self.content_widget.setLayout(self.content_layout)
        
        # Create title bar
        self._create_title_bar(main_layout)
        
        # Add content widget to main layout
        main_layout.addWidget(self.content_widget, 1)
        
        # Create mode buttons
        self._create_mode_buttons()
        
        # Create combobox container
        self._create_combobox_container()
        
        # Set up opacity control
        self._setup_opacity_control()
        
        self.apply_theme(self.theme_manager._current_theme)

        # Create a non-interactive border overlay so frame borders render above children (e.g., badge)
        self._setup_border_overlay()

        # Center window
        self._center_window()
        
        # Populate comboboxes
        self._populate_window_combobox()
        self._populate_monitor_combobox()
        
        # Set initial combobox visibility
        self._update_combobox_visibility(self.window_mode_button.isChecked())
        
        # Restore window geometry
        self._restore_window_geometry()
        
        logger.info("MainDialog initialization complete")
    
    def _build_ui_alt(self):
        """Build the main user interface."""
        # Create main frame
        self.main_frame = QFrame()
        self.main_frame.setObjectName("main_frame")
        self.main_frame.setMouseTracking(True)
        
        # Set main frame as central widget
        self.setCentralWidget(self.main_frame)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.main_frame.setLayout(main_layout)
        
        # Create content widget
        self.content_widget = QWidget()
        self.content_widget.setObjectName("contentWidget")
        
        # Content layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(0)
        self.content_widget.setLayout(self.content_layout)
        
        # Create mode buttons
        self._create_mode_buttons()
        
        # Create combobox container
        self._create_combobox_container()
        
        # Set up opacity control
        self._setup_opacity_control()
        
        # Setup badge system
        self._setup_badge_system()
        
        # Setup system tray
        self._setup_system_tray()
        
        # Connect signals
        self._connect_signals()
        
        # Enable mouse tracking for all widgets
        self._enable_mouse_tracking()
    
    def _create_title_bar(self, main_layout):
        """Create the title bar."""
        self.title_bar = QFrame()
        self.title_bar.setObjectName("titleBar")
        # Rely on QSS for height; do not force fixed height here
        self.title_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(10, 0, 5, 0)
        # Slightly increase spacing to avoid tight packing that can visually clip adjacent widgets
        title_layout.setSpacing(8)
        
        # Title icon (DPI-aware for crispness). QSS controls the visual size; do not fix sizes here.
        self.title_icon = QLabel()
        self.title_icon.setObjectName("titleIcon")
        self.title_icon.setScaledContents(True)  # allow pixmap to conform to QSS-defined 45px box
        # Use embedded Qt resource icon
        icon_path = ":/icons/ShittyPIP.ico"
        try:
            screen = (self.windowHandle().screen() if self.windowHandle() else QApplication.primaryScreen())
            dpr = float(getattr(screen, 'devicePixelRatio', lambda: 1.0)()) if screen else 1.0
            if hasattr(screen, 'devicePixelRatio') and not callable(screen.devicePixelRatio):
                dpr = float(screen.devicePixelRatio)
        except Exception:
            dpr = 1.0
        # Request a high-resolution pixmap and let QLabel scale it per QSS box
        base_px = 64
        icon_pm = QIcon(icon_path).pixmap(int(base_px * dpr), int(base_px * dpr))
        try:
            icon_pm.setDevicePixelRatio(dpr)
        except Exception:
            pass
        self.title_icon.setPixmap(icon_pm)
        self.title_icon.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        
        # Title label
        self.title = QLabel("SHITTY PiP QUICKSWAP")
        self.title.setObjectName("titleLabel")
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        
        # Spacer
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        # Close button
        self.close_button = QPushButton("X")
        self.close_button.setObjectName("closeButton")
        # Match AboutDialog: no fixed size or custom cursor; styling via QSS
        self.close_button.clicked.connect(self.close)
        
        # Add widgets to title bar
        title_layout.addWidget(self.title_icon)
        title_layout.addSpacing(5)
        title_layout.addWidget(self.title, 1, Qt.AlignCenter)
        title_layout.addItem(spacer)
        title_layout.addWidget(self.close_button)
        
        # Set window properties
        self.setWindowTitle("SPQ")
        self.setProperty("_q_customTitleBarTitle", "SHITTY PiP QUICKSWAP")
        
        try:
            icon = QIcon(icon_path)
            if not icon.isNull():
                self.setWindowIcon(icon)
                logger.info(f"Window icon set from Qt resource: {icon_path}")
            else:
                logger.error(f"Qt resource window icon is null: {icon_path}")
        except Exception as e:
            logger.error(f"Failed to set window icon from Qt resource: {e}", exc_info=True)
        
        # Add title bar to main layout
        main_layout.addWidget(self.title_bar)
        
    
    
    def _create_mode_buttons(self):
        """Create mode buttons for window/monitor selection."""
        # Create mode buttons
        self.window_mode_button = QPushButton("WINDOW MODE")
        self.window_mode_button.setObjectName("QSmolselect")
        self.window_mode_button.setCursor(Qt.PointingHandCursor)
        self.window_mode_button.setCheckable(True)
        
        self.monitor_mode_button = QPushButton("MONITOR MODE")
        self.monitor_mode_button.setObjectName("QSmolselect")
        self.monitor_mode_button.setCursor(Qt.PointingHandCursor)
        self.monitor_mode_button.setCheckable(True)
        
        # Load saved mode button state
        saved_mode = self.settings_manager.get("ui.mode_button_state", "window")
        if saved_mode == "window":
            self.window_mode_button.setChecked(True)
            self.monitor_mode_button.setChecked(False)
        else:
            self.window_mode_button.setChecked(False)
            self.monitor_mode_button.setChecked(True)
        
        # Create mode buttons layout
        mode_buttons_layout = QHBoxLayout()
        mode_buttons_layout.setContentsMargins(0, 0, 0, 0)
        mode_buttons_layout.setSpacing(10)
        mode_buttons_layout.addWidget(self.window_mode_button)
        mode_buttons_layout.addWidget(self.monitor_mode_button)
        
        mode_buttons_widget = QWidget()
        mode_buttons_widget.setLayout(mode_buttons_layout)
        
        # Create button group for exclusive toggling
        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.setExclusive(True)
        self.mode_button_group.addButton(self.window_mode_button, 0)
        self.mode_button_group.addButton(self.monitor_mode_button, 1)
        
        # Create left layout
        left_layout = QVBoxLayout()
        # Top-left pinning: no extra top margin here; overall top margin is handled by content_layout
        left_layout.setContentsMargins(10, 0, 0, 0)
        left_layout.setSpacing(5)  # Increase spacing for better visual separation
        left_layout.addWidget(mode_buttons_widget, alignment=Qt.AlignLeft | Qt.AlignTop)
        
        # Create right button layout
        right_button_layout = QVBoxLayout()
        # No extra per-group top padding; use main_top_layout's top margin for both sides
        right_button_layout.setContentsMargins(0, 0, 0, 0)
        right_button_layout.setSpacing(5)
        
        self.minimize_tray_button = QPushButton("MINIMIZE TO TRAY")
        self.minimize_tray_button.setObjectName("QBasicBitchButton")
        self.minimize_tray_button.setCursor(Qt.PointingHandCursor)
        self.minimize_tray_button.setFixedWidth(140)
        right_button_layout.addWidget(self.minimize_tray_button, alignment=Qt.AlignRight)
        
        self.subsettings_button = QPushButton("SUBSETTINGS")
        self.subsettings_button.setObjectName("QBasicBitchButton")
        self.subsettings_button.setCursor(Qt.PointingHandCursor)
        self.subsettings_button.setFixedWidth(140)
        right_button_layout.addWidget(self.subsettings_button, alignment=Qt.AlignRight)
        
        self.about_button = QPushButton("ABOUT")
        self.about_button.setObjectName("QBasicBitchButton")
        self.about_button.setCursor(Qt.PointingHandCursor)
        self.about_button.setFixedWidth(90)
        right_button_layout.addWidget(self.about_button, alignment=Qt.AlignRight)
        
        right_buttons_widget = QWidget()
        right_buttons_widget.setLayout(right_button_layout)
        
        # Create main top layout
        main_top_layout = QHBoxLayout()
        # Normal horizontal padding with uniform top padding for both groups
        main_top_layout.setContentsMargins(10, 10, 10, 0)
        main_top_layout.setSpacing(10)
        
        # Create container widgets for left and right sides to maintain grouping
        left_container = QWidget()
        left_container.setLayout(left_layout)
        left_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        
        # Set size policies to keep elements grouped
        right_buttons_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Add widgets to main top layout with appropriate stretch factors
        # Left group pinned top-left
        main_top_layout.addWidget(left_container, 0, Qt.AlignTop | Qt.AlignLeft)
        main_top_layout.addStretch(1)  # Add stretch in the middle to push elements to sides
        main_top_layout.addWidget(right_buttons_widget, 0, Qt.AlignTop | Qt.AlignRight)  # No stretch to maintain compact size
        
        # Insert this as the first item in the content layout
        self.content_layout.insertLayout(0, main_top_layout)
        
        # Store reference to left layout for later use
        self.left_layout = left_layout
    
    def _create_combobox_container(self):
        """Create the combobox container for window/monitor selection."""
        # Create a container for the comboboxes
        self.combobox_container = QWidget()
        self.combobox_layout = QVBoxLayout(self.combobox_container)
        self.combobox_layout.setContentsMargins(0, 0, 0, 0)
        self.combobox_layout.setSpacing(0)
        
        # Create a stacked widget to hold both combobox rows
        self.combobox_stack = QStackedWidget()
        
        # Function to create a combobox row
        def create_combobox_row(placeholder):
            # Create container widget
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 2, 0, 2)
            layout.setSpacing(8)
            
            # Create combobox
            combobox = QComboBox()
            combobox.setObjectName("windowComboBox")
            combobox.addItem(placeholder)
            combobox.setItemData(0, "", Qt.UserRole - 1)
            combobox.setMinimumWidth(220)
            
            # Create arrow button
            arrow_button = QPushButton(">>")
            arrow_button.setObjectName("QComboArrow")
            arrow_button.setCursor(Qt.PointingHandCursor)
            
            # Add widgets to layout
            layout.addWidget(combobox)
            layout.addWidget(arrow_button)
            
            # Let the layout handle the sizing
            container.adjustSize()

            return container, combobox, arrow_button
        
        # Create window row
        self.window_row, self.window_combobox, self.window_arrow_button = create_combobox_row("Select a window...")
        
        # Create monitor row
        self.monitor_row, self.monitor_combobox, self.monitor_arrow_button = create_combobox_row("Select a monitor...")
        
        # Let rows size naturally under layout/QSS
        
        # Add both rows to the stacked widget
        self.combobox_stack.addWidget(self.window_row)
        self.combobox_stack.addWidget(self.monitor_row)
        
        # Add the stacked widget to the main layout
        self.combobox_layout.addWidget(self.combobox_stack, alignment=Qt.AlignLeft)
        
        # Add combobox container to the left layout (keep left-aligned)
        self.left_layout.addWidget(self.combobox_container, alignment=Qt.AlignLeft)
    
    def _setup_opacity_control(self):
        """Set up the opacity control UI elements."""
        try:
            # Create container frame for the opacity control
            self.opacity_frame = QFrame()
            self.opacity_frame.setObjectName("opacityControlFrame")
            opacity_layout = QVBoxLayout(self.opacity_frame)
            opacity_layout.setContentsMargins(0, 2, 0, 0)
            opacity_layout.setSpacing(2)
            
            # Create container for the opacity bar
            self.opacity_container = QFrame()
            self.opacity_container.setObjectName("opacityControlContainer")
            self.opacity_container.setFixedHeight(14)
            self.opacity_container.setMouseTracking(True)
            
            # Set up the layout
            container_layout = QHBoxLayout(self.opacity_container)
            container_layout.setContentsMargins(3, 2, 3, 2)
            container_layout.setSpacing(0)
            
            # Create the fill and empty bars
            self.opacity_fill = QFrame()
            self.opacity_fill.setObjectName("opacityFill")
            self.opacity_fill.setFixedHeight(10)
            
            self.opacity_empty = QFrame()
            self.opacity_empty.setObjectName("opacityEmpty")
            self.opacity_empty.setFixedHeight(10)
            self.opacity_empty.setStyleSheet("background: transparent; border: none; margin: 0; padding: 0;")
            
            container_layout.addWidget(self.opacity_fill)
            container_layout.addWidget(self.opacity_empty)
            
            # Create the label
            self.opacity_label = QLabel("OPACITY")
            self.opacity_label.setObjectName("opacityLabel")
            self.opacity_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            
            # Add to layout
            opacity_layout.addWidget(self.opacity_container)
            opacity_layout.addWidget(self.opacity_label)
            
            # Add to left layout directly below combobox
            self.left_layout.addSpacing(5)
            self.left_layout.addWidget(self.opacity_frame, alignment=Qt.AlignLeft | Qt.AlignTop)
            
            # Load saved opacity
            saved_opacity = self.settings_manager.get("appearance.opacity", 100)
            self.current_opacity = saved_opacity
            
            # Connect to opacity manager signal
            from core.opacity.manager import get_opacity_manager
            opacity_manager = get_opacity_manager()
            opacity_manager.opacityChanged.connect(self.set_opacity)
            
            # Connect mouse events for dragging
            self.opacity_container.mousePressEvent = self._on_opacity_bar_clicked
            self.opacity_container.mouseMoveEvent = self._on_opacity_bar_dragged
            
        except Exception as e:
            logger.error(f"Error in _setup_opacity_control: {e}", exc_info=True)
            raise
    
    def _setup_window_behavior(self):
        """Initialize the window behavior manager for dragging, resizing, and snapping."""
        self.window_behavior = WindowBehaviorManager(
            widget=self,
            min_width=MIN_WINDOW_WIDTH,
            min_height=MIN_WINDOW_HEIGHT
        )
    
    def _setup_badge_system(self):
        """Set up the badge system by initializing the badge display."""
        self._initialize_badge()
        
    def _initialize_badge(self):
        """Initialize the badge system with a persistent badge image."""
        try:
            # Create badge label
            self.badge_label = QLabel(self.main_frame)
            self.badge_label.setObjectName("badgeLabel")
            self.badge_label.setCursor(Qt.ArrowCursor)
            self.badge_label.setMouseTracking(True)
            self.badge_label.mouseDoubleClickEvent = self._on_badge_double_clicked
            # Ensure badge receives mouse events (needed for double-click)
            self.badge_label.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            # Python custom styling (permitted exception): drop shadow for visibility
            shadow = QGraphicsDropShadowEffect(self.badge_label)
            shadow.setBlurRadius(16)
            shadow.setOffset(0, 2)
            shadow.setColor(QColor(0, 0, 0, 180))
            self.badge_label.setGraphicsEffect(shadow)
            
            # Load badge from settings or default to Badge19.png
            saved_badge = self.settings_manager.get("ui.badge_file", None)
            if not saved_badge:
                saved_badge = "Badge19.png"
            self._select_badge(saved_badge)
            self._position_badge()
            self.badge_label.show()
            
        except Exception as e:
            logger.error(f"Error initializing badge: {e}", exc_info=True)
            raise RuntimeError("Failed to initialize badge system.") from e
            
    def _select_badge(self, badge_file=None):
        """Select and load a badge image by filename from Qt resources; if None, pick a random Badge*.png.
        Saves the selected filename (not path) to settings under 'ui.badge_file'.
        """
        try:
            import random
            # List embedded badge files from Qt resource system
            qdir = QDir(":/badges")
            entries = qdir.entryList(["Badge*.png"], QDir.Files | QDir.Readable, QDir.Name)
            badge_files = list(entries)
            logger.debug(f"Qt resource badges found: {badge_files}")
            if not badge_files:
                raise RuntimeError("No badge files found in Qt resources under :/badges")
            
            if badge_file is None:
                # Choose a random badge, preferably different from current
                current = getattr(self, 'current_badge_name', None)
                choices = [f for f in badge_files if f != current] or badge_files
                selected_badge = random.choice(choices)
            else:
                if badge_file not in badge_files:
                    raise RuntimeError(f"Badge file '{badge_file}' not found in Qt resources.")
                selected_badge = badge_file
            
            badge_path = f":/badges/{selected_badge}"
            pm = QPixmap(badge_path)
            if pm.isNull():
                raise RuntimeError(f"Failed to load badge '{badge_path}'.")
            
            # Preserve original pixmap and defer scaling to dynamic updater
            self._badge_pm_orig = pm
            self._update_badge_visuals()
            
            # Track and persist selection
            self.current_badge_name = selected_badge
            self.current_badge_path = badge_path
            self.settings_manager.set("ui.badge_file", selected_badge)
            logger.debug(f"Badge set to {selected_badge}")
            
        except Exception as e:
            logger.error(f"Error loading badge: {e}", exc_info=True)
            raise
    
    def _update_badge_visuals(self):
        """Scale badge between 400–500px width based on window size and position bottom-right."""
        try:
            if not hasattr(self, 'badge_label') or not hasattr(self, '_badge_pm_orig'):
                return
            pm = self._badge_pm_orig
            if pm is None or pm.isNull():
                return
            parent = self.main_frame if hasattr(self, 'main_frame') else self
            # Compute target width in [400, 500] (logical px) based on parent width
            base = MIN_WINDOW_WIDTH
            delta = max(0, parent.width() - base)
            incr = min(100, int(delta // 4))  # +1px per 4px beyond base, up to +100
            target_w_logical = max(400, min(500, 400 + incr))
            # Apply requested 20% reduction in logical size
            target_w_logical = max(1, int(target_w_logical * BADGE_SCALE_FACTOR))

            # DPI-aware scaling: scale in device pixels, then mark DPR for crisp rendering
            try:
                screen = (self.windowHandle().screen() if self.windowHandle() else QApplication.primaryScreen())
                dpr = float(getattr(screen, 'devicePixelRatio', lambda: 1.0)()) if screen else 1.0
                if hasattr(screen, 'devicePixelRatio') and not callable(screen.devicePixelRatio):
                    # PySide6 may expose devicePixelRatio as property on QScreen
                    dpr = float(screen.devicePixelRatio)
            except Exception:
                dpr = 1.0

            target_w_device = int(target_w_logical * dpr)
            scaled = pm.scaled(target_w_device, target_w_device, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            try:
                scaled.setDevicePixelRatio(dpr)
            except Exception:
                pass

            # Apply pixmap and logical size to label
            logical_w = max(1, int(scaled.width() / dpr))
            logical_h = max(1, int(scaled.height() / dpr))
            self.badge_label.setPixmap(scaled)
            self.badge_label.setFixedSize(logical_w, logical_h)
            self._position_badge()
        except Exception as e:
            logger.error(f"Error updating badge visuals: {e}", exc_info=True)
            
    def _position_badge(self):
        """Position the badge in the bottom right corner of the window."""
        try:
            if hasattr(self, 'badge_label') and self.badge_label and self.badge_label.pixmap():
                # Position relative to main_frame (badge's parent), flush/slight underlap
                parent = self.main_frame
                badge_w = self.badge_label.width()
                badge_h = self.badge_label.height()
                # Use positive overlap to intentionally underlap beyond parent's right/bottom edges
                overlap = max(2, int(badge_h * BADGE_UNDERLAP_FACTOR))
                left_shift = int(badge_w * BADGE_LEFT_SHIFT_FACTOR)
                x = parent.width() - badge_w - left_shift + overlap
                y = parent.height() - badge_h + overlap
                self.badge_label.move(x, y)
                self.badge_label.raise_()
                logger.debug(f"Positioned badge at ({x}, {y})")
                
        except Exception as e:
            logger.error(f"Error positioning badge: {e}", exc_info=True)
            
    def _on_badge_double_clicked(self, event):
        """Handle double-click on the badge to change to a different image."""
        try:
            logger.debug("Badge double-clicked, selecting new badge")
            self._select_badge()  # pick random
            self._update_badge_visuals()
            event.accept()
        except Exception as e:
            logger.error(f"Error handling badge double-click: {e}", exc_info=True)
            raise RuntimeError("Failed to handle badge double-click.") from e

    def _showEvent_badge_refresh(self, event):
        """Private helper retained to avoid duplicate defs; not used."""
        super().showEvent(event)
        try:
            if hasattr(self, 'badge_label'):
                self.badge_label.show()
                self._update_badge_visuals()
                self.badge_label.raise_()
        except Exception as e:
            logger.error(f"Error in _showEvent_badge_refresh: {e}", exc_info=True)

    def _setup_system_tray(self):
        """Set up the system tray manager."""
        try:
            self.tray_manager = SystemTrayManager(self)
            self.tray_manager.show_main_window_requested.connect(self.show)
            self.tray_manager.show_settings_requested.connect(self._on_tray_show_settings)
            # Route Quit to a full shutdown sequence
            self.tray_manager.quit_requested.connect(self._on_tray_quit)
            # Connect tray toggles (removed click-through)
            if hasattr(self.tray_manager, 'toggle_overlay_lock_requested'):
                self.tray_manager.toggle_overlay_lock_requested.connect(self._on_tray_toggle_overlay_lock)
            if hasattr(self.tray_manager, 'toggle_auto_switch_requested'):
                self.tray_manager.toggle_auto_switch_requested.connect(self._on_tray_toggle_auto_switch)

            # Ensure tray icon is cleaned up on any app quit path
            try:
                from PySide6.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    app.aboutToQuit.connect(self.tray_manager.cleanup)
            except Exception:
                pass

            # Sync initial tray labels/states with current settings (removed click-through)
            try:
                locked = bool(self.overlay_manager.is_overlay_locked())
                self.tray_manager.set_overlay_lock_state(locked)
            except Exception:
                pass
            try:
                auto_switch = bool(self.settings_manager.get("features.autoswitch_enabled", False))
                self.tray_manager.set_auto_switch_state(auto_switch)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error setting up system tray: {e}", exc_info=True)
            raise RuntimeError("Failed to set up system tray.") from e

    def _on_tray_quit(self) -> None:
        """Perform a clean application shutdown from the tray menu."""
        try:
            # Best-effort core shutdown first (unregister hotkeys, stop controllers, pools)
            core = getattr(self, 'app_instance', None)
            if core and hasattr(core, 'shutdown'):
                try:
                    core.shutdown()
                except Exception:
                    pass

            # Remove tray icon/menu promptly
            try:
                if hasattr(self, 'tray_manager') and self.tray_manager:
                    self.tray_manager.cleanup()
            except Exception:
                pass

            # Ask Qt to quit
            from PySide6.QtCore import QCoreApplication
            QCoreApplication.quit()
        except Exception as e:
            logger.error(f"Failed to quit from tray: {e}")

    # Removed click-through tray toggle handler

    def _on_tray_show_settings(self) -> None:
        """Handle tray request to show settings dialog."""
        try:
            self._show_subsettings_dialog()
        except Exception as e:
            logger.error(f"Failed to show settings from tray: {e}")

    def _on_tray_toggle_overlay_lock(self) -> None:
        """Handle tray request to toggle overlay lock."""
        try:
            locked = bool(self.overlay_manager.is_overlay_locked())
            new_locked = not locked
            self.overlay_manager.set_overlay_lock(new_locked)
            # Reflect new state in tray label
            try:
                self.tray_manager.set_overlay_lock_state(new_locked)
            except Exception:
                pass
            logger.info(f"Tray toggled overlay lock -> {new_locked}")
        except Exception as e:
            logger.error(f"Failed to toggle overlay lock from tray: {e}")

    def _on_tray_toggle_auto_switch(self) -> None:
        """Handle tray request to toggle auto-switch setting."""
        try:
            current = bool(self.settings_manager.get("features.autoswitch_enabled", False))
            new_state = not current
            self.settings_manager.set("features.autoswitch_enabled", new_state)
            
            # Apply to overlay manager
            try:
                from core.graphics import get_overlay_manager
                overlay_manager = get_overlay_manager()
                if overlay_manager:
                    overlay_manager.set_auto_switch_enabled(new_state)
            except Exception as e:
                logger.error(f"Failed to apply auto-switch setting to overlay manager: {e}")
            
            # Keep tray checkbox state in sync
            try:
                self.tray_manager.set_auto_switch_state(new_state)
            except Exception:
                pass
            logger.info(f"Tray toggled auto-switch -> {new_state}")
        except Exception as e:
            logger.error(f"Failed to toggle auto-switch from tray: {e}")
            
    def _setup_window_behavior(self):
        """Initialize the window behavior manager for dragging, resizing, and snapping.
        
        This is the centralized initialization point for all window behavior.
        """
        try:
            # Initialize the window behavior manager with this window as the target widget
            self.window_behavior = WindowBehaviorManager(
                widget=self,
                min_width=MIN_WINDOW_WIDTH,
                min_height=MIN_WINDOW_HEIGHT
            )
            logger.info("Window behavior manager initialized")
        except Exception as e:
            logger.error(f"Error setting up window behavior: {e}", exc_info=True)
            raise RuntimeError("Failed to set up window behavior manager.") from e
    
    def _connect_signals(self):
        """Connect all signals and slots."""
        # Connect mode button signals
        self.window_mode_button.toggled.connect(self._on_mode_button_toggled)
        self.monitor_mode_button.toggled.connect(self._on_mode_button_toggled)
        
        # Connect combobox signals
        self.window_combobox.currentIndexChanged.connect(self._on_window_selected)
        self.monitor_combobox.currentIndexChanged.connect(self._on_monitor_selected)
        # idToggled emits (id: int, checked: bool); ensure correct mapping to window/monitor rows
        self.mode_button_group.idToggled.connect(self._on_mode_id_toggled)
        
        # Connect close button
        self.close_button.clicked.connect(self.close)
        
        # Connect right button actions
        self.minimize_tray_button.clicked.connect(self._minimize_to_tray)
        self.subsettings_button.clicked.connect(self._open_subsettings_dialog)
        self.about_button.clicked.connect(self._toggle_about_dialog)
        
        # Connect launch buttons (double-arrow) to explicit overlay creation
        try:
            if hasattr(self, 'window_arrow_button') and self.window_arrow_button:
                self.window_arrow_button.clicked.connect(self._on_launch_overlay_clicked)
            if hasattr(self, 'monitor_arrow_button') and self.monitor_arrow_button:
                self.monitor_arrow_button.clicked.connect(self._on_launch_overlay_clicked)
        except Exception as e:
            logger.error(f"Failed to connect launch buttons: {e}", exc_info=True)
    
    def _enable_mouse_tracking(self):
        """Enable mouse tracking and install an event filter to capture child mouse events for dragging."""
        def enable_and_filter(widget):
            widget.setMouseTracking(True)
            # Only install filter on children; let self use its normal handlers
            for child in widget.findChildren(QWidget):
                child.setMouseTracking(True)
                child.installEventFilter(self)
        enable_and_filter(self)

    def eventFilter(self, obj, event):
        """Forward mouse events from children to WindowBehaviorManager with coordinates mapped to self."""
        if obj is self or not hasattr(self, 'window_behavior'):
            return False
        et = event.type()
        if et in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            # Only handle events originating from this window's hierarchy
            if isinstance(obj, QWidget) and self.isAncestorOf(obj):
                # Map to this window's coordinate system
                try:
                    gp = event.globalPosition().toPoint() if hasattr(event, 'globalPosition') else event.globalPos()
                    local = self.mapFromGlobal(gp)

                    # Lightweight proxy carrying the API expected by WindowBehaviorManager
                    class _Proxy:
                        def __init__(self, btn, btns, lp, gp):
                            self._btn = btn
                            self._btns = btns
                            self._lp = lp
                            self._gp = gp
                        def button(self):
                            return self._btn
                        def buttons(self):
                            return self._btns
                        def position(self):
                            class _P:
                                def __init__(self, p):
                                    self._p = p
                                def toPoint(self):
                                    return self._p
                            return _P(self._lp)
                        def globalPosition(self):
                            class _P:
                                def __init__(self, p):
                                    self._p = p
                                def toPoint(self):
                                    return self._p
                            return _P(self._gp)
                        def pos(self):
                            return self._lp
                        def globalPos(self):
                            return self._gp

                    proxy = _Proxy(event.button(), event.buttons(), local, gp)

                    state = self.window_behavior.state
                    if et == QEvent.MouseButtonPress:
                        # Only initiate drag if region is draggable
                        if self.is_draggable_region(local) and proxy.button() == Qt.LeftButton:
                            self.window_behavior.handle_mouse_press(proxy, self.is_draggable_region)
                            return True
                    elif et == QEvent.MouseMove:
                        if state.is_dragging or state.is_resizing:
                            self.window_behavior.handle_mouse_move(proxy)
                            return True
                    elif et == QEvent.MouseButtonRelease:
                        if state.is_dragging or state.is_resizing:
                            self.window_behavior.handle_mouse_release(proxy)
                            return True
                    elif et == QEvent.MouseButtonDblClick:
                        if self.is_draggable_region(local):
                            self.showNormal() if self.isMaximized() else self.showMaximized()
                            return True
                except Exception:
                    pass
        return False
    
    def _center_window(self):
        """Center the window on the primary screen."""
        screen = QApplication.primaryScreen().availableGeometry()
        frame_geometry = self.frameGeometry()
        frame_geometry.moveCenter(screen.center())
        self.move(frame_geometry.topLeft())
    
    # --- Event Handlers ---
    
    def mousePressEvent(self, event):
        """Handle mouse press events for dragging and resizing."""
        if hasattr(self, 'window_behavior'):
            self.window_behavior.handle_mouse_press(event, self.is_draggable_region)
        else:
            super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click: toggle maximize on any draggable blank space or title bar."""
        try:
            if hasattr(self, 'window_behavior') and self.is_draggable_region(event.pos()):
                self.showNormal() if self.isMaximized() else self.showMaximized()
            else:
                super().mouseDoubleClickEvent(event)
        except Exception:
            super().mouseDoubleClickEvent(event)
    
    def mouseMoveEvent(self, event):
        """Update cursor on hover to show resize handles."""
        if hasattr(self, 'window_behavior'):
            # Get the current drag state
            drag_state = self.window_behavior.state
            
            # Only change cursor if we're not dragging or resizing
            if not drag_state.is_dragging and not drag_state.is_resizing:
                self.window_behavior.handle_mouse_move(event)
            else:
                # If we're dragging or resizing, let the behavior manager handle it
                self.window_behavior.handle_mouse_move(event)
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events for drag and resize operations."""
        if hasattr(self, 'window_behavior'):
            self.window_behavior.handle_mouse_release(event)
        else:
            super().mouseReleaseEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave events to reset cursor."""
        if hasattr(self, 'window_behavior'):
            self.window_behavior.handle_leave()
        super().leaveEvent(event)
    
    def is_draggable_region(self, pos: QPoint) -> bool:
        """Check if the given position is in a draggable region.
        
        The title bar and any blank space in the window should be draggable, except for controls.
        """
        # Allow dragging from the entire title bar height, except on window control buttons
        title_h = self.title_bar.height() if hasattr(self, 'title_bar') and self.title_bar else 55
        if pos.y() <= title_h:
            w = self.childAt(pos)
            if w and getattr(w, 'objectName', lambda: '')() in ["minimizeButton", "maximizeButton", "closeButton"]:
                return False
            return True

        # Outside the title bar region, do NOT allow dragging
        # This prevents accidental drags from content areas and focuses UX on the title bar
        return False

        # Helper: walk up parents to detect interactive controls or right-side buttons
        interactive_types = (QComboBox, QPushButton, QSlider, QLineEdit, QCheckBox, QRadioButton, QTextEdit, QPlainTextEdit, QAbstractSpinBox)
        cur = w
        while cur is not None and cur is not self:
            # Allow dragging from the badge as requested
            if hasattr(self, 'badge_label') and cur is self.badge_label:
                return True
            if isinstance(cur, interactive_types):
                return False
            # Avoid dragging from the right-side buttons container
            if hasattr(self, 'right_buttons_widget') and cur is self.right_buttons_widget:
                return False
            cur = cur.parentWidget()

        # For container widgets within the content area, allow dragging
        if self.content_widget.isAncestorOf(w):
            return True

        return False
    
    def showEvent(self, event):
        """Handle window show event to restore geometry and refresh badge visuals."""
        try:
            super().showEvent(event)
            self._restore_window_geometry()
            
            if hasattr(self, 'badge_label'):
                self.badge_label.show()
                # Ensure correct size and flush positioning on show
                self._update_badge_visuals()
                self.badge_label.raise_()
            # Position overlay over main_frame in window coords and raise it
            if hasattr(self, 'border_overlay') and self.border_overlay:
                from PySide6.QtCore import QPoint, QRect
                top_left = self.main_frame.mapTo(self, QPoint(0, 0))
                self.border_overlay.setGeometry(QRect(top_left, self.main_frame.size()))
                self.border_overlay.show()
                self.border_overlay.raise_()
                
            if hasattr(self, 'current_opacity'):
                self.set_opacity(self.current_opacity)
                
        except Exception as e:
            logger.error(f"Error restoring window geometry: {e}", exc_info=True)
            raise RuntimeError("Failed to restore window geometry.") from e
    
    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        if hasattr(self, "opacity_container"):
            self.set_opacity(getattr(self, 'current_opacity', 100))
        # Update badge visuals on resize (scaling + underlap positioning)
        try:
            if hasattr(self, 'badge_label'):
                self._update_badge_visuals()
            if hasattr(self, 'border_overlay') and self.border_overlay:
                from PySide6.QtCore import QPoint, QRect
                top_left = self.main_frame.mapTo(self, QPoint(0, 0))
                self.border_overlay.setGeometry(QRect(top_left, self.main_frame.size()))
                self.border_overlay.raise_()
        except Exception as e:
            logger.error(f"Error in resizeEvent updating badge: {e}", exc_info=True)
    
    def closeEvent(self, event):
        """Handle window close event to save geometry."""
        try:
            # Save window geometry
            geometry = {
                "x": self.x(),
                "y": self.y(),
                "width": self.width(),
                "height": self.height(),
                "maximized": self.isMaximized()
            }
            self.settings_manager.set("ui.main_window_geometry", geometry)
            
            # Hide badge if it exists
            if hasattr(self, 'badge_label'):
                self.badge_label.hide()
                
            logger.debug(f"Saved main window geometry: {geometry}")
            # Route to centralized shutdown so all resources are cleaned, then quit
            try:
                self._on_tray_quit()
            except Exception:
                pass
            event.accept()
            
        except Exception as e:
            logger.error(f"Error saving window geometry: {e}", exc_info=True)
            event.ignore()
            raise RuntimeError("Failed to save window geometry.") from e
    
    # --- UI Update Methods ---
    
    def _populate_window_combobox(self):
        """Populate the window combobox with available windows."""
        try:
            current_data = self.window_combobox.currentData()
            self.window_combobox.clear()
            self.window_combobox.addItem("Select a window...")
            self.window_combobox.setItemData(0, "", Qt.UserRole - 1)
            
            windows = self.window_enumerator.refresh_window_list(force=True)
            
            for hwnd, title in windows:
                icon = self.window_enumerator.icon_manager.get_window_icon(hwnd)
                if not icon or icon.isNull():
                    icon = WindowEnumerator._blank_icon
                self.window_combobox.addItem(icon, title, hwnd)
            
            if current_data:
                index = self.window_combobox.findData(current_data)
                if index > 0:
                    self.window_combobox.setCurrentIndex(index)
                    
        except Exception as e:
            logger.error(f"Error populating window combobox: {e}", exc_info=True)
            self.window_combobox.addItem("Error loading windows")
            self.window_combobox.setEnabled(False)
    
    def _on_window_selected(self, index: int):
        """Handle window selection from combobox."""
        if index <= 0:
            return
            
        hwnd = self.window_combobox.currentData()
        if not hwnd:
            return
            
        try:
            # Build minimal window info for logging/preview only; do NOT create overlay here
            window_info = self._build_window_info(hwnd)
            logger.info(f"Selected window: {window_info.get('title', 'Unknown')} (HWND: {hwnd})")
            self.selected_window = window_info
            self._update_window_preview()
            
        except Exception as e:
            logger.error(f"Error selecting window {hwnd}: {e}", exc_info=True)
    
    def _populate_monitor_combobox(self):
        """Populate the monitor combobox with available monitors."""
        try:
            current_data = self.monitor_combobox.currentData()
            self.monitor_combobox.clear()
            self.monitor_combobox.addItem("Select a monitor...")
            self.monitor_combobox.setItemData(0, "", Qt.UserRole - 1)
            
            monitors = get_all_monitors()
            
            if not monitors:
                self.monitor_combobox.addItem("No monitors found")
                return
                
            for i, monitor in enumerate(monitors):
                is_primary = monitor.get('is_primary', False)
                monitor_name = f"Monitor {i + 1}: {monitor['rect'].width()}x{monitor['rect'].height()}"
                if is_primary:
                    monitor_name += " (Primary)"
                self.monitor_combobox.addItem(monitor_name)
                # Store the full monitor info dict for selection handling
                self.monitor_combobox.setItemData(i + 1, monitor, Qt.UserRole)
                
            if current_data:
                for i in range(1, self.monitor_combobox.count()):
                    if self.monitor_combobox.itemData(i) == current_data:
                        self.monitor_combobox.setCurrentIndex(i)
                        break
                        
        except Exception as e:
            logger.error(f"Error populating monitor combobox: {e}", exc_info=True)
            self.monitor_combobox.addItem("Error loading monitors")
            self.monitor_combobox.setEnabled(False)
            raise RuntimeError("Failed to populate monitor combobox.") from e
    
    def _on_monitor_selected(self, index: int):
        """Handle monitor selection from combobox."""
        if index <= 0:
            return
            
        monitor = self.monitor_combobox.currentData()
        if not monitor:
            return
            
        logger.info(f"Selected monitor: {monitor.get('name', 'Unknown')}")
        # Store selection only; overlay is launched explicitly via the double-arrow
        self.selected_monitor = monitor

    def _on_launch_overlay_clicked(self):
        """Launch the overlay based on current mode and selection via the double-arrow button."""
        try:
            if hasattr(self, 'mode_button_group') and self.mode_button_group.checkedId() == 0:
                if not getattr(self, 'selected_window', None):
                    raise RuntimeError("No window selected for overlay launch")
                self._create_overlay_for_window(self.selected_window)
            else:
                if not getattr(self, 'selected_monitor', None):
                    raise RuntimeError("No monitor selected for overlay launch")
                self._create_overlay_for_monitor(self.selected_monitor)
        except Exception as e:
            logger.error(f"Overlay launch failed: {e}", exc_info=True)
            raise
    
    def _update_combobox_visibility(self, window_mode: bool):
        """Update combobox visibility based on selected mode."""
        self.combobox_stack.setCurrentIndex(0 if window_mode else 1)
        
        if window_mode:
            self._populate_window_combobox()
        else:
            self._populate_monitor_combobox()

    def _on_mode_id_toggled(self, id: int, checked: bool):
        """Handle exclusive mode toggle by id to update combobox visibility and contents.
        id == 0 -> window mode, id == 1 -> monitor mode. Only react when checked is True.
        """
        try:
            if not checked:
                return
            window_mode = (id == 0)
            self._update_combobox_visibility(window_mode)
            # Do not auto-create overlays on mode toggle; creation is explicit via the launch button
        except Exception as e:
            logger.error(f"Error handling mode toggle (id={id}, checked={checked}): {e}", exc_info=True)
    
    def _on_mode_button_toggled(self, checked):
        """Save the mode button state when toggled."""
        if checked:
            if self.window_mode_button.isChecked():
                self.settings_manager.set("ui.mode_button_state", "window")
            else:
                self.settings_manager.set("ui.mode_button_state", "monitor")
    
    # --- Opacity Control Methods ---
    
    def set_opacity(self, percent: int):
        """Set the opacity fill bar to the given percentage (0-100)."""
        percent = max(0, min(100, percent))
        self.current_opacity = percent
        
        if not hasattr(self, 'opacity_container') or self.opacity_container.width() <= 0:
            ThreadManager.single_shot(10, self.set_opacity, percent)
            return
            
        container_layout = self.opacity_container.layout()
        if not container_layout:
            return
        
        fill_stretch = percent
        empty_stretch = 100 - percent
        container_layout.setStretch(0, fill_stretch)
        container_layout.setStretch(1, empty_stretch)
        
        self.opacity_container.updateGeometry()
        self.opacity_fill.updateGeometry()
        self.opacity_empty.updateGeometry()
        
        self.settings_manager.set("appearance.opacity", percent)

        # Route opacity to active overlay via OverlayManager (0.0–1.0)
        try:
            self.overlay_manager.set_opacity(percent / 100.0)
        except Exception as e:
            logger.error(f"Failed to update overlay opacity: {e}", exc_info=True)

    # --- Overlay Creation Helpers ---
    def _ensure_min_overlay_size(self, rect: QRect) -> QRect:
        """Ensure overlay rect meets minimum size of 640x360 logical pixels."""
        try:
            min_size = QSize(640, 360)
            if rect.size().width() < min_size.width() or rect.size().height() < min_size.height():
                new_size = rect.size().expandedTo(min_size)
                rect = QRect(rect.topLeft(), new_size)
            return rect
        except Exception as e:
            logger.error(f"Failed to enforce minimum overlay size: {e}", exc_info=True)
            raise

    def _compute_min_size_for_display(self) -> QSize:
        """Compute minimum overlay size honoring current display aspect ratio.
        Not smaller than 427x240, scaled to preserve AR of the active screen.
        """
        screen = self.screen()
        if not screen:
            return QSize(427, 240)
        geo = screen.geometry()
        w, h = geo.width(), geo.height()
        if h <= 0:
            return QSize(427, 240)
        ar = float(w) / float(h)
        base_w, base_h = 427, 240
        # Two candidates preserving AR, at least base size
        cand1 = QSize(max(base_w, int(round(base_h * ar))), base_h)
        cand2 = QSize(base_w, max(base_h, int(round(base_w / ar))))
        area1 = cand1.width() * cand1.height()
        area2 = cand2.width() * cand2.height()
        return cand1 if area1 <= area2 else cand2

    def _default_overlay_rect_top_left(self) -> QRect:
        """Default top-left rect on current display using computed min size."""
        screen = self.screen()
        if not screen:
            return QRect(0, 0, 427, 240)
        geo = screen.geometry()
        size = self._compute_min_size_for_display()
        return QRect(geo.left(), geo.top(), size.width(), size.height())

    def _get_window_rect(self, hwnd: int) -> QRect:
        """Get the QRect of a Win32 window by handle."""
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            return QRect(left, top, right - left, bottom - top)
        except Exception as e:
            logger.error(f"Failed to get window rect for hwnd={hwnd}: {e}", exc_info=True)
            raise RuntimeError("Failed to get window rect") from e

    def _get_window_title(self, hwnd: int) -> str:
        """Get the title of a Win32 window by handle."""
        try:
            return win32gui.GetWindowText(hwnd) or "Window Overlay"
        except Exception as e:
            logger.error(f"Failed to get window title for hwnd={hwnd}: {e}", exc_info=True)
            return "Window Overlay"

    def _build_window_info(self, hwnd: int) -> dict:
        """Build a minimal window info dict with 'rect' and 'title'."""
        rect = self._get_window_rect(hwnd)
        rect = self._ensure_min_overlay_size(rect)
        title = self._get_window_title(hwnd)
        return {"hwnd": hwnd, "rect": rect, "title": title}

    def _create_overlay_for_window(self, window_info) -> None:
        """Create a window overlay using the DWM backend for the given window_info."""
        # Validate window_info and geometry
        if not window_info:
            raise RuntimeError("No window_info provided for overlay creation")
        # Title for display/logging
        if isinstance(window_info, dict):
            title = window_info.get('title', 'Window Overlay')
        else:
            title = getattr(window_info, 'title', 'Window Overlay')
        # Ignore target window rect for initial placement per UX: open at top-left minimum size
        rect = self._default_overlay_rect_top_left()
        opacity = (getattr(self, 'current_opacity', 100) or 100) / 100.0
        # Extract hwnd from window_info
        if isinstance(window_info, dict):
            hwnd = int(window_info.get('hwnd', 0))
        else:
            hwnd = int(getattr(window_info, 'hwnd', 0) or 0)
        created_id = self.overlay_manager.create_window_overlay(
            rect,
            title=title,
            opacity=opacity,
            backend=BackendType.DWM,
            hwnd=hwnd,
        )
        if not created_id:
            raise RuntimeError("Failed to create window overlay")
        logger.info(f"Created window overlay: {created_id} at {rect}")
        
        # Minimize main dialog to tray after successful overlay creation
        self._minimize_to_tray()

    def _create_overlay_for_monitor(self, monitor: dict) -> None:
        """Create a monitor overlay using the Monitor backend for the given monitor dict."""
        if not monitor or 'rect' not in monitor:
            raise RuntimeError("Monitor info with 'rect' is required for overlay creation")
        # Place at top-left of the launching display, minimum size honoring AR
        rect = self._default_overlay_rect_top_left()
        opacity = (getattr(self, 'current_opacity', 100) or 100) / 100.0
        created_id = self.overlay_manager.create_monitor_overlay(rect, title="Monitor Overlay", opacity=opacity, backend=BackendType.MONITOR, monitor_target=monitor)
        if not created_id:
            raise RuntimeError("Failed to create monitor overlay")
        logger.info(f"Created monitor overlay: {created_id} at {rect}")
        
        # Minimize main dialog to tray after successful overlay creation
        self._minimize_to_tray()
    
    def _update_opacity_from_click(self, pos):
        """Update opacity based on click/drag position."""
        container_width = self.opacity_container.width()
        if container_width <= 0:
            return
            
        percent = int((pos.x() / container_width) * 100)
        percent = max(0, min(100, percent))
        self.set_opacity(percent)
        
        logger.debug(f"Opacity set to {percent}%")
    
    def _on_opacity_bar_clicked(self, event):
        """Handle mouse click on the opacity bar."""
        self._update_opacity_from_click(event.pos())
    
    def _on_opacity_bar_dragged(self, event):
        """Handle mouse drag on the opacity bar."""
        if event.buttons() & Qt.LeftButton:
            self._on_opacity_bar_clicked(event)
    
    def increase_opacity(self, amount: int = 5):
        """Increase the opacity by the specified amount."""
        from core.opacity.manager import get_opacity_manager
        opacity_manager = get_opacity_manager()
        opacity_manager.increase_opacity(amount)
        self.set_opacity(opacity_manager.get_opacity())
    
    def decrease_opacity(self, amount: int = 1):
        """Decrease the opacity by the specified amount."""
        from core.opacity.manager import get_opacity_manager
        opacity_manager = get_opacity_manager()
        opacity_manager.decrease_opacity(amount)
        self.set_opacity(opacity_manager.get_opacity())
    
    # --- Badge System Methods ---
    
    def _select_badge_legacy(self, badge_file=None):
        """Legacy badge selection (unused)."""
        try:
            import random
            import os
            resources_dir = "resources"
            badge_files = [f for f in os.listdir(resources_dir) if f.startswith("Badge") and f.endswith(".png")]
            if not badge_files:
                raise RuntimeError("No badge files found in resources directory.")
            if badge_file and badge_file in badge_files:
                selected_badge = badge_file
            else:
                selected_badge = random.choice(badge_files)
            badge_path = os.path.join(resources_dir, selected_badge)
            pixmap = QPixmap(badge_path)
            if pixmap.isNull():
                raise RuntimeError(f"Failed to load badge {badge_path}.")
            # Legacy path expected _scale_badge_pixmap; now no-op
            self.current_badge_path = badge_path
            self.settings_manager.set("ui.badge_file", selected_badge)
        except Exception as e:
            logger.error(f"Error in _select_badge_legacy: {e}", exc_info=True)
    
    def _scale_badge_pixmap(self, pixmap: QPixmap):
        """Scale the badge pixmap based on window state."""
        try:
            base_size = 400
            max_size = 500
            target_size = max_size if self.isMaximized() else base_size
            
            scaled_pixmap = pixmap.scaled(
                target_size, 
                target_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.badge_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Error scaling badge pixmap: {e}", exc_info=True)
            raise RuntimeError("Failed to scale badge pixmap.") from e
    
    def _position_badge_legacy(self):
        """Legacy badge positioning (kept for reference; unused)."""
        try:
            if not hasattr(self, 'badge_label') or not self.badge_label.pixmap():
                return
            content_geo = self.content_widget.geometry() if hasattr(self, 'content_widget') else self.rect()
            badge_size = self.badge_label.pixmap().size()
            x_pos = content_geo.width() - badge_size.width() + 2
            y_pos = content_geo.height() - badge_size.height() + 2
            self.badge_label.move(x_pos, y_pos)
        except Exception as e:
            logger.error(f"Error in _position_badge_legacy: {e}", exc_info=True)
    
    def _on_badge_double_clicked_legacy(self, event):
        """Legacy double-click handler (unused)."""
        try:
            logger.debug("Badge double-clicked, selecting new badge (legacy)")
            self._select_badge()
            self._position_badge_legacy()
        except Exception as e:
            logger.error(f"Error in legacy badge double-click: {e}", exc_info=True)
    
    # --- Window Management Methods ---
    
    def _restore_window_geometry(self):
        """Restore window geometry from settings."""
        try:
            geometry = self.settings_manager.get("ui.main_window_geometry", None)
            if geometry:
                self.setGeometry(
                    geometry.get("x", self.x()),
                    geometry.get("y", self.y()),
                    geometry.get("width", self.width()),
                    geometry.get("height", self.height())
                )
                
                if geometry.get("maximized", False):
                    self.showMaximized()
                    
                logger.debug(f"Restored main window geometry: {geometry}")
                
        except Exception as e:
            logger.error(f"Error restoring window geometry: {e}", exc_info=True)
            raise RuntimeError("Failed to restore window geometry.") from e
    
    def _minimize_to_tray(self):
        """Minimize the window to the system tray."""
        try:
            self.hide()
            self.tray_manager.show_message(
                "SPQ Running",
                "Application is running in the system tray.",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
            
        except Exception as e:
            logger.error(f"Error minimizing to tray: {e}", exc_info=True)
            raise RuntimeError("Failed to minimize to tray.") from e
    
    def _toggle_about_dialog(self):
        """Toggle the about dialog."""
        try:
            if hasattr(self, '_about_dialog') and self._about_dialog:
                if self._about_dialog.isVisible():
                    self._about_dialog.hide()
                else:
                    self._about_dialog.show()
                    self._about_dialog.raise_()
                    self._about_dialog.activateWindow()
            else:
                self._about_dialog = self._create_about_dialog()
                self._about_dialog.show()
                self._about_dialog.raise_()
                self._about_dialog.activateWindow()
        except Exception as e:
            logger.error(f"Error toggling about dialog: {e}", exc_info=True)
            raise RuntimeError("Failed to toggle about dialog.") from e

    def _create_about_dialog(self):
        """Create the centralized AboutDialog instance."""
        try:
            from ui.dialogs.about_dialog import AboutDialog
            return AboutDialog(self)
        except Exception as e:
            logger.error(f"Error creating AboutDialog: {e}", exc_info=True)
            raise RuntimeError("Failed to create AboutDialog.") from e
    
    def _open_subsettings_dialog(self):
        """Open the subsettings dialog."""
        try:
            from ui.dialogs.subsettings_dialog import SubSettingsDialog
            
            # Reset cursor before opening dialog to prevent stuck resize cursor
            if hasattr(self, 'window_behavior'):
                self.window_behavior.handle_leave()
            
            if not hasattr(self, '_subsettings_dialog') or not self._subsettings_dialog:
                self._subsettings_dialog = SubSettingsDialog(self)
                
            self._subsettings_dialog.show()
            self._subsettings_dialog.raise_()
            self._subsettings_dialog.activateWindow()
            
        except Exception as e:
            logger.error(f"Error opening subsettings dialog: {e}", exc_info=True)
            raise RuntimeError("Failed to open subsettings dialog.") from e
    
    def _update_window_preview(self):
        """Update the window preview (placeholder for future implementation)."""
        pass
    
    def apply_theme(self, theme_name: str = None):
        """Apply the specified theme to the dialog.
        
        Args:
            theme_name: Name of the theme to apply. If None, uses the current theme.
        """
        try:
            if theme_name is None and hasattr(self, 'theme_manager'):
                # Use current theme if none specified
                theme_name = self.theme_manager.get_current_theme()
            if hasattr(self, 'theme_manager'):
                # Apply theme using theme manager's built-in method
                self.theme_manager.apply_theme(theme_name)
                logger.debug(f"Applied theme: {theme_name}")
                # Ensure visuals remain correct after stylesheet changes
                try:
                    if hasattr(self, 'badge_label') and self.badge_label:
                        self._update_badge_visuals()
                        self.badge_label.raise_()
                        self.badge_label.show()
                        logger.debug("Refreshed badge visuals after theme apply")
                    # Update border overlay color after stylesheet reapply
                    self._apply_border_color()
                except Exception as e:
                    logger.error(f"Post-theme visual refresh failed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error applying theme: {e}", exc_info=True)
            raise RuntimeError("Failed to apply theme.") from e

    def _on_theme_changed(self, theme_name: str) -> None:
        """Handle theme change to refresh visuals after stylesheet updates."""
        try:
            if hasattr(self, 'badge_label') and self.badge_label:
                self._update_badge_visuals()
                self._position_badge()
                self.badge_label.raise_()
                self.badge_label.show()
                logger.debug(f"Badge refreshed on theme_changed: {theme_name}")
            # Ensure border overlay matches theme on theme change
            self._apply_border_color()
        except Exception as e:
            logger.error(f"Error refreshing visuals on theme_changed: {e}", exc_info=True)

    def _on_style_changed(self) -> None:
        """Handle style reapplication events to keep visuals correct."""
        try:
            if hasattr(self, 'badge_label') and self.badge_label:
                self._update_badge_visuals()
                self._position_badge()
                self.badge_label.raise_()
                self.badge_label.show()
                logger.debug("Badge refreshed on style_changed")
            # Ensure border overlay matches theme on style changes too
            self._apply_border_color()
        except Exception as e:
            logger.error(f"Error refreshing visuals on style_changed: {e}", exc_info=True)

    def _apply_border_color(self):
        """Apply themed border color to the paint overlay (10% lighter in light theme)."""
        try:
            if not hasattr(self, 'border_overlay') or not self.border_overlay:
                return
            # Pull base border token from theme
            theme_name = self.theme_manager.get_current_theme() if hasattr(self, 'theme_manager') else 'dark'
            if theme_name.lower() == 'dark':
                # Dark theme requires a white border for MainDialog
                col = QColor(255, 255, 255, 255)
            else:
                # Light theme: start from theme token and lighten by 10%
                base_hex = self.theme_manager.get_token('border', theme_name)
                col = QColor(base_hex)
                if theme_name.lower() == 'light':
                    col = col.lighter(110)
                col.setAlpha(255)
            self.border_overlay.set_color(col)
        except Exception as e:
            logger.error(f"Failed to apply border color: {e}", exc_info=True)
