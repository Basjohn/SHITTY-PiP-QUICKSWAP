"""
Consolidated Theme Manager

A unified solution for managing application theming, styling, and assets.
Replaces and combines functionality from style_manager.py and theme_manager.py.
"""
import json
import os
import weakref
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, TypeVar, Union

from PySide6.QtCore import QObject, Signal, Qt, QSize, QFile
from core.threading import ThreadManager
from PySide6.QtGui import QColor, QFont, QIcon, QPixmap
from core.settings import settings_manager
from core.logging import get_logger
from utils.resource_manager import get_resource_manager, ResourceType
from utils.paths import get_data_dir

# Project root (no sys.path mutation; assume proper package setup)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = get_logger(__name__)
try:
    # Ensure Qt resources (e.g., :/themes/*.qss) are registered even during tests
    # When running the app, main.py imports this; tests may not.
    import ui.resources_rc  # type: ignore  # noqa: F401
    logger.debug("Registered Qt resources via ui.resources_rc")
except Exception as _e:
    # Non-fatal: ThemeManager will fall back to filesystem if resource import is unavailable
    logger.debug(f"ui.resources_rc import failed; QSS will try filesystem fallback. err={_e}")

# Type aliases
ColorLike = Union[QColor, str, tuple[int, int, int], tuple[int, int, int, int]]
IconLike = Union[QIcon, str, QPixmap]

# Type variable for theme enums
T = TypeVar('T', bound=Enum)

class ThemeVariant(Enum):
    """Supported theme variants."""
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"

class ThemeColorRole(Enum):
    """Standard color roles for theming."""
    # Base colors
    BASE = "base"
    TEXT = "text"
    BORDER = "border"
    HIGHLIGHT = "highlight"
    HIGHLIGHT_TEXT = "highlight_text"
    DISABLED = "disabled"
    DISABLED_TEXT = "disabled_text"
    
    # Button colors
    BUTTON = "button"
    BUTTON_HOVER = "button_hover"
    BUTTON_PRESSED = "button_pressed"
    BUTTON_TEXT = "button_text"
    BUTTON_BORDER = "button_border"
    
    # Input colors
    INPUT_BG = "input_bg"
    INPUT_TEXT = "input_text"
    
    # Additional UI elements
    TOOLTIP_BG = "tooltip_bg"
    TOOLTIP_TEXT = "tooltip_text"
    LINK = "link"
    LINK_VISITED = "link_visited"

@dataclass
class ThemeColors:
    """Container for theme color definitions."""
    colors: Dict[str, str] = field(default_factory=dict)
    
    def get(self, role: Union[ThemeColorRole, str], default: str = None) -> Optional[str]:
        """Get a color by role.

        Strict behavior: do NOT silently return a color when the role is missing.
        Return None unless an explicit default is provided. This allows callers
        like ThemeManager.get_token to fail fast per no-fallback policy.
        """
        role_name = role.value if isinstance(role, ThemeColorRole) else role
        if default is not None:
            return self.colors.get(role_name, default)
        return self.colors.get(role_name)
    
    def set(self, role: Union[ThemeColorRole, str], value: str) -> None:
        """Set a color for a role."""
        role_name = role.value if isinstance(role, ThemeColorRole) else role
        self.colors[role_name] = value

class ThemeManager(QObject):
    """
    Centralized theme management for the application.
    
    Handles:
    - Loading and applying themes
    - Managing theme assets (icons, images, fonts)
    - Styling Qt widgets
    - Theme switching and customization
    - High-DPI support
    - Resource cleanup
    """
    # Signals
    theme_changed = Signal(str)  # Emits the name of the new theme
    style_changed = Signal()     # Emitted when styles are reapplied
    
    # Default theme definitions
    DEFAULT_THEMES = {
        ThemeVariant.DARK.value: {
            'name': 'dark',
            'base': '#1e1e1e',
            'text': '#ffffff',
            'button': '#444444',
            'button_hover': '#555555',
            'button_pressed': '#333333',
            'button_text': '#ffffff',
            'button_border': '#dddddd',
            'input_bg': '#2d2d2d',
            'input_text': '#ffffff',
            'border': '#666666',
            'overlay.border.stroke': '#404040',
            'overlay.border.thickness.base': '2.0',
            'overlay.border.accent': '#808080',  # Dark grey accent on white border
            'overlay.border.rounded.enabled': 'false',
            'highlight': '#0078d7',
            'highlight_text': '#ffffff',
            'disabled': '#555555',
            'disabled_text': '#999999',
            'tooltip_bg': '#2d2d2d',
            'tooltip_text': '#ffffff',
            'link': '#4a9cff',
            'link_visited': '#b07bff',
        },
        ThemeVariant.LIGHT.value: {
            'name': 'light',
            'base': '#f5f5f5',
            'text': '#000000',
            'button': '#e6e6e6',
            'button_hover': '#f0f0f0',
            'button_pressed': '#d9d9d9',
            'button_text': '#000000',
            'button_border': '#000000',
            'input_bg': '#ffffff',
            'input_text': '#000000',
            # Use a very dark grey for borders to match QSS light theme intent
            'border': '#141414',
            'overlay.border.stroke': '#000000',
            'overlay.border.thickness.base': '2.5',
            'overlay.border.accent': '#cccccc',  # Light grey accent on black border
            'overlay.border.rounded.enabled': 'false',
            'highlight': '#0066cc',
            'highlight_text': '#ffffff',
            'disabled': '#e0e0e0',
            'disabled_text': '#999999',
            'tooltip_bg': '#ffffff',
            'tooltip_text': '#000000',
            'link': '#0066cc',
            'link_visited': '#7b42ff',
        }
    }
    
    # Singleton instance
    _instance: Optional['ThemeManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Implement singleton pattern - lock-free via UI thread confinement."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    @classmethod
    def instance(cls, app_instance=None) -> 'ThemeManager':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            if app_instance:
                cls._instance._app_ref = weakref.ref(app_instance)
        return cls._instance
    
    def __init__(self, app=None):
        """Initialize the theme manager - idempotent.
        
        Args:
            app: Reference to the main application instance (weak reference will be stored)
        """
        if self._initialized:
            return
            
        super().__init__()
        self._app_ref = weakref.ref(app) if app else None
        self._settings_manager = None
        self._current_theme = ThemeVariant.DARK.value
        self._theme_colors = {}
        self._styles = {}
        self._assets = {}
        self._initialized = True
        # Use ThreadManager instead of direct QTimer for consistent timing management
        self._pending_theme = None
        self._applying_theme = False
        
        # Initialize caches
        self._icon_cache = {}
        self._pixmap_cache = {}
        self._font_cache = {}
        
        # Initialize centralized resource manager singleton
        self.resource_manager = get_resource_manager()
        self._resource_id = None
        
        # Set up paths (use centralized portable/runtime-aware paths)
        data_dir = get_data_dir()
        self._themes_dir = data_dir / 'themes'
        self._resources_dir = data_dir / 'resources'
        
        # Ensure directories exist
        self._themes_dir.mkdir(exist_ok=True, parents=True)
        (self._resources_dir / 'icons').mkdir(exist_ok=True, parents=True)
        (self._resources_dir / 'images').mkdir(exist_ok=True, parents=True)
        
        # Initialize default themes
        self._init_default_themes()
        
        # Load any custom themes
        self._load_custom_themes()
        
        # Connect to settings changes
        if app:
            ThreadManager.single_shot(0, self._initialize_settings_connection)

        # Register ThemeManager with ResourceManager for deterministic cleanup
        try:
            if self.resource_manager:
                self._resource_id = self.resource_manager.register(
                    self,
                    ResourceType.CUSTOM,
                    "ThemeManager",
                    cleanup_handler=lambda obj: obj.shutdown(),
                    tags={"theme_manager"}
                )
        except Exception as e:
            logger.debug(f"ThemeManager ResourceManager.register failed: {e}")
    
    @property
    def app(self):
        """Get the application instance (or None if it no longer exists)."""
        # Try weak reference first
        if self._app_ref:
            app = self._app_ref()
            if app is not None:
                return app
        
        # Fallback to QApplication.instance() if weak reference is invalid
        from PySide6.QtWidgets import QApplication
        return QApplication.instance()
    
    def _init_default_themes(self) -> None:
        """Initialize the default themes if they don't exist."""
        # Initialize default themes from the DEFAULT_THEMES class variable
        for theme_name, theme_data in self.DEFAULT_THEMES.items():
            if theme_name not in self._theme_colors:
                self._theme_colors[theme_name] = ThemeColors(theme_data)
        
    def _load_custom_themes(self) -> None:
        """Load any custom themes from the filesystem.
        
        Looks for theme files in the 'themes' directory at the project root.
        Theme files should be JSON files with a .json extension.
        """
        themes_dir = self._themes_dir
        if not themes_dir.exists():
            logger.warning(f"Themes directory not found at {themes_dir}")
            return
            
        for theme_file in themes_dir.glob('*.json'):
            try:
                with open(theme_file, 'r', encoding='utf-8') as f:
                    theme_data = json.load(f)
                
                if not isinstance(theme_data, dict):
                    logger.warning(f"Invalid theme file format in {theme_file.name}")
                    continue
                    
                theme_name = theme_data.get('name')
                if not theme_name:
                    logger.warning(f"Theme name not found in {theme_file.name}")
                    continue
                    
                # Add or update the theme, then validate required tokens to fail fast
                self._theme_colors[theme_name] = ThemeColors(theme_data)
                try:
                    self._validate_theme(theme_name)
                except Exception as ve:
                    # Remove invalid theme entry and fail fast per no-fallback policy
                    self._theme_colors.pop(theme_name, None)
                    error_msg = (
                        f"Theme '{theme_name}' from {theme_file.name} is invalid: {ve}. "
                        f"Required tokens missing. Fix the theme file and restart."
                    )
                    logger.error(error_msg)
                    raise
                logger.info(f"Loaded theme: {theme_name} from {theme_file.name}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing theme file {theme_file.name}: {e}")
            except Exception as e:
                logger.error(f"Error loading theme from {theme_file.name}: {e}")
        
    def _initialize_settings_connection(self):
        """Initialize connection to settings manager after app is fully initialized."""
        # SettingsManager is a singleton, so we can create a new instance
        # and it will return the existing one if it already exists
        self._settings_manager = settings_manager.SettingsManager()
        self._settings_manager.setting_changed.connect(self._on_setting_changed)
        
        # Apply initial theme from settings
        theme = self._settings_manager.get('theme', None)
        if theme is not None and theme != self._current_theme:
            self.apply_theme(theme)
    
    def apply_theme(self, theme: Union[str, ThemeVariant], from_settings: bool = False) -> None:
        """Apply a theme to the application.
        
        Args:
            theme: Theme name or ThemeVariant enum
            from_settings: If True, this was triggered by a settings change
            
        Raises:
            ValueError: If the theme is invalid or cannot be applied
        """
        theme_name = theme.value if isinstance(theme, ThemeVariant) else str(theme).lower()
        
        # If this is coming from settings and matches current theme, do nothing
        if from_settings and theme_name == self._current_theme:
            return
        
        # If we're in the middle of a settings change, queue this request
        if not from_settings and self._settings_manager and self._settings_manager.get('theme') != theme_name:
            self._pending_theme = theme_name
            # Use ThreadManager instead of direct QTimer for consistent timing management
            ThreadManager.single_shot(100, self._apply_pending_theme_change)
            return
        
        # Clear any pending theme changes
        self._pending_theme = None

        # Re-entrancy guard
        if self._applying_theme:
            logger.debug("apply_theme ignored due to ongoing application")
            return
        
        # Validate theme exists and has all required colors
        if theme_name not in self._theme_colors:
            error_msg = f"Theme '{theme_name}' not found"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Validate the theme has all required color definitions
        self._validate_theme(theme_name)
        
        old_theme = self._current_theme
        self._current_theme = theme_name
        
        # Apply the theme
        try:
            self._applying_theme = True
            # Clear caches on UI thread prior to applying new theme
            self._clear_caches_async()
            # Update the application palette first
            self._update_application_palette()
            
            # Apply styles
            self._apply_styles()
            
            # Update settings if this wasn't triggered by a settings change
            if not from_settings and self._settings_manager:
                self._settings_manager.set('theme', theme_name, save_immediately=True)
            
            # Emit signal
            self.theme_changed.emit(theme_name)
            logger.info(f"Applied theme: {theme_name} (was: {old_theme})")
            
        except Exception as e:
            error_msg = f"Failed to apply theme '{theme_name}': {e}"
            logger.error(error_msg)
            # Clear the current theme to prevent partial application
            self._current_theme = ""
            # Re-raise the exception to fail fast
            raise RuntimeError(error_msg) from e
        finally:
            self._applying_theme = False

    def get_current_theme(self) -> str:
        """Return the current theme name (e.g., 'dark' or 'light')."""
        return self._current_theme

    def get_token(self, name: Union[str, ThemeColorRole], theme_name: Optional[str] = None) -> str:
        """Get a color token by name from the specified or current theme.
        
        Args:
            name: Token name or ThemeColorRole, e.g. 'border', 'base', 'text'.
            theme_name: Optional theme name; if None, uses the current theme.
        
        Returns:
            str: Hex color string for the token.
        
        Raises:
            ValueError: If the theme or token is not found.
        """
        colors = self.get_theme_colors(theme_name)
        value = colors.get(name)
        if value is None:
            raise ValueError(f"Token '{name}' not found in theme '{theme_name or self._current_theme}'")
        return value
    
    def _apply_pending_theme_change(self):
        """Apply a pending theme change that was queued."""
        if self._pending_theme is not None:
            self.apply_theme(self._pending_theme)
    
    def _on_setting_changed(self, setting_name: str, value):
        """Handle changes to settings that affect theming."""
        if setting_name == 'theme':
            self.apply_theme(value, from_settings=True)
    
    def get_theme_colors(self, theme: str = None) -> ThemeColors:
        """Get color definitions for a theme.
        
        Args:
            theme: Theme name. If None, uses current theme.
            
        Returns:
            ThemeColors: The theme's color definitions
            
        Raises:
            ValueError: If the theme is not found or invalid
        """
        theme_name = theme or self._current_theme
        if not theme_name:
            raise ValueError("No theme specified and no current theme set")
            
        if theme_name not in self._theme_colors:
            raise ValueError(f"Theme '{theme_name}' not found")
            
        # Return the existing ThemeColors container as-is (no double wrapping)
        return self._theme_colors[theme_name]
    
    def get_icon(self, icon_name: str, size: int = 32) -> QIcon:
        """Get an icon from the current theme.
        
        Args:
            icon_name: Name of the icon (without extension)
            size: Desired icon size in pixels
            
        Returns:
            QIcon: The requested icon, or a fallback if not found
        """
        cache_key = (icon_name, size)
        if cache_key in self._icon_cache:
            return self._icon_cache[cache_key]
            
        # Try to load the icon from the current theme
        icon_path = self._get_icon_path(icon_name)
        if icon_path and os.path.exists(icon_path):
            icon = QIcon(icon_path)
            self._icon_cache[cache_key] = icon
            return icon
        # Fallback: return an empty icon and warn
        logger.warning(f"Icon not found: {icon_name}")
        return QIcon()
    
    def get_pixmap(self, image_name: str, size: Optional[QSize | int] = None) -> QPixmap:
        """Get a pixmap from the current theme.
        
        Args:
            image_name: Name of the image (without extension)
            size: Desired size in pixels or QSize. If None, original size is used.
        
        Returns:
            QPixmap: The requested pixmap, possibly scaled. Empty if not found.
        """
        # Normalize size to QSize
        qsize: Optional[QSize]
        if size is None:
            qsize = None
        elif isinstance(size, QSize):
            qsize = size
        else:
            qsize = QSize(int(size), int(size))
        
        cache_key = (image_name, qsize.width() if qsize else 0, qsize.height() if qsize else 0)
        if cache_key in self._pixmap_cache:
            return self._pixmap_cache[cache_key]
        
        image_path = self._get_image_path(image_name)
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if qsize and not pixmap.isNull():
                pixmap = pixmap.scaled(qsize, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self._pixmap_cache[cache_key] = pixmap
            return pixmap
        
        # Fallback to blank pixmap
        logger.warning(f"Image not found: {image_name}")
        return QPixmap(qsize) if qsize else QPixmap()
    
    def _get_image_path(self, image_name: str) -> Optional[str]:
        """Get the filesystem path for an image."""
        for ext in ['.png', '.jpg', '.jpeg', '.svg', '.bmp', '.gif']:
            image_path = self._resources_dir / 'images' / f"{image_name}{ext}"
            if image_path.exists():
                return str(image_path)
        return None
    
    def get_font(self, font_name: str = "default") -> QFont:
        """Get a font from the current theme.
        
        Args:
            font_name: Name of the font (e.g., 'default', 'heading')
            
        Returns:
            QFont: The requested font, or the application default if not found
        """
        if font_name in self._font_cache:
            return self._font_cache[font_name]
            
        # Fall back to default font
        font = QFont("Segoe UI" if self._current_theme == ThemeVariant.DARK.value else "Segoe UI", 10)
        self._font_cache[font_name] = font
        return font
    
    def get_theme_stylesheet(self, theme_name: str = None) -> str:
        """Get a stylesheet string for the specified theme.
        
        Args:
            theme_name: Name of the theme. If None, uses current theme.
            
        Returns:
            str: Stylesheet string for the theme
        """
        if theme_name is None:
            theme_name = self._current_theme
            
        # Get theme colors
        colors = self.get_theme_colors(theme_name)
        
        # Convert QColor objects to hex strings if needed
        def color_to_hex(color):
            if hasattr(color, 'name'):
                return color.name()
            return str(color)
        
        # Generate stylesheet based on theme colors
        return f"""
            QWidget {{
                background-color: {color_to_hex(colors.get('base', '#1e1e1e'))};
                color: {color_to_hex(colors.get('text', '#ffffff'))};
                border: 1px solid {color_to_hex(colors.get('border', '#404040'))};
            }}
            QPushButton {{
                background-color: {color_to_hex(colors.get('button', '#444444'))};
                color: {color_to_hex(colors.get('button_text', '#ffffff'))};
                border: 1px solid {color_to_hex(colors.get('button_border', '#dddddd'))};
                padding: 5px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {color_to_hex(colors.get('button_hover', '#555555'))};
            }}
            QPushButton:pressed {{
                background-color: {color_to_hex(colors.get('button_pressed', '#333333'))};
            }}
            QComboBox {{
                background-color: {color_to_hex(colors.get('input_bg', '#2d2d2d'))};
                color: {color_to_hex(colors.get('input_text', '#ffffff'))};
                border: 1px solid {color_to_hex(colors.get('border', '#404040'))};
                border-radius: 4px;
                padding: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {color_to_hex(colors.get('input_bg', '#2d2d2d'))};
                color: {color_to_hex(colors.get('input_text', '#ffffff'))};
                selection-background-color: {color_to_hex(colors.get('highlight', '#0078d7'))};
                selection-color: {color_to_hex(colors.get('highlight_text', '#ffffff'))};
            }}
        """
    
    def _get_icon_path(self, icon_name: str) -> Optional[str]:
        """Get the filesystem path for an icon.
        
        Args:
            icon_name: Name of the icon (without extension)
            
        Returns:
            Optional[str]: Path to the icon file, or None if not found
        """
        # Try different extensions in order of preference
        for ext in ['.svg', '.png', '.ico']:
            icon_path = self._resources_dir / 'icons' / f"{icon_name}{ext}"
            if icon_path.exists():
                return str(icon_path)
        return None
    
    def _get_font_path(self, font_file: str) -> Optional[str]:
        """Get the filesystem path for a font file.
        
        Args:
            font_file: Name of the font file (with extension)
            
        Returns:
            Optional[str]: Path to the font file, or None if not found
        """
        # Check in font directories
        font_dirs = [
            Path(":/fonts"),  # Resource path
            self._resources_dir / "fonts",  # Filesystem (portable-aware)
        ]
        
        for font_dir in font_dirs:
            font_path = font_dir / font_file
            if font_path.exists():
                return str(font_path)
        return None
        
    def _validate_theme(self, theme_name: str) -> None:
        """Validate that a theme has all required color definitions.
        
        Args:
            theme_name: Name of the theme to validate
            
        Raises:
            ValueError: If the theme is missing required color definitions
        """
        if theme_name not in self._theme_colors:
            raise ValueError(f"Theme '{theme_name}' not found")
            
        required_colors = [
            'base', 'text', 'border', 'overlay.border.stroke', 'highlight', 'highlight_text',
            'disabled', 'disabled_text', 'button', 'button_hover',
            'button_pressed', 'button_text', 'button_border',
            'input_bg', 'input_text', 'tooltip_bg', 'tooltip_text',
            'link', 'link_visited'
        ]
        
        missing = [color for color in required_colors 
                  if not self._theme_colors[theme_name].get(color)]
        
        if missing:
            raise ValueError(
                f"Theme '{theme_name}' is missing required color definitions: {', '.join(missing)}"
            )
        
    def _update_application_palette(self):
        """Apply the current theme's colors to the application palette.
        
        Intentionally left empty because we use QSS for styling. Kept for
        backward compatibility with older callers.
        """
        
    def apply_theme_to_widget(self, widget):
        """Apply the current theme to a specific widget.
        
        Args:
            widget: The widget to apply theme to
            
        This ensures consistent styling for dynamically created widgets.
        """
        if not widget:
            return
            
        try:
            # Apply current theme's stylesheet
            theme_name = self._current_theme
            
            # If it's a QMenu, apply specialized menu styling
            if hasattr(widget, 'objectName') and widget.objectName() == 'overlayContextMenu':
                # Use the context menu's own theming method if available
                if hasattr(widget.parent(), 'apply_theme'):
                    widget.parent().apply_theme(theme_name)
                    return
                    
            # For regular widgets, just apply base styling
            stylesheet = self.get_theme_stylesheet(theme_name)
            widget.setStyleSheet(stylesheet)
            
            # If widget has any children, propagate theme
            if hasattr(widget, 'findChildren'):
                for child in widget.findChildren(QObject):
                    # Only apply to widgets that can have stylesheets
                    if hasattr(child, 'setStyleSheet'):
                        child.setStyleSheet(stylesheet)
                        
            logger.debug(f"Applied theme to widget {widget.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Failed to apply theme to widget: {e}")
    
    def _apply_styles(self) -> None:
        """Apply the current theme's styles to the application.
        
        This loads the appropriate QSS file based on the current theme and applies it.
        Any styles defined in QSS will take precedence over Python-applied styles.
        """
        logger.debug(f"Applying styles for theme: {self._current_theme}")
        
        if not self.app:
            logger.error("Cannot apply styles: No application instance available")
            return
            
        # Load the appropriate QSS file based on the current theme
        # Prefer Qt resource path first (embedded): :/themes/<name>.qss
        style_sheet = None
        res_qss_path = f":/themes/{self._current_theme}.qss"
        try:
            f = QFile(res_qss_path)
            if f.exists() and f.open(QFile.ReadOnly | QFile.Text):
                data = f.readAll()
                try:
                    style_sheet = bytes(data).decode('utf-8')
                except Exception:
                    style_sheet = str(bytes(data), 'utf-8', errors='ignore')
                f.close()
                logger.debug(f"Loaded theme stylesheet from Qt resource: {res_qss_path} ({len(style_sheet)} chars)")
        except Exception as e:
            logger.debug(f"Failed to read QSS from Qt resource {res_qss_path}: {e}")
        
        # Fallback to filesystem portable path under data/themes
        theme_file = self._themes_dir / f"{self._current_theme}.qss"
        logger.debug(f"Looking for theme file at: {theme_file}")
        
        if style_sheet is None:
            if not theme_file.exists():
                error_msg = f"Theme file not found: {theme_file}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            try:
                with open(theme_file, 'r', encoding='utf-8') as f2:
                    style_sheet = f2.read()
                if not style_sheet.strip():
                    error_msg = f"Theme file is empty: {theme_file}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                logger.debug(f"Successfully read {len(style_sheet)} characters from theme file")
            except Exception as e:
                error_msg = f"Failed to load theme file {theme_file}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
        
        try:
            # Apply the style to the application (propagates to all widgets)
            self.app.setStyleSheet(style_sheet)
            logger.debug("Successfully applied stylesheet to application")
            
            # Emit style changed signal
            self.style_changed.emit()
            logger.debug("Emitted style_changed signal")
            
        except Exception as e:
            error_msg = f"Failed to apply theme stylesheet: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def _clear_caches_async(self) -> None:
        """Clear icon/pixmap/font caches on the UI thread safely."""
        def _do_clear():
            try:
                icon_n = len(self._icon_cache)
                pix_n = len(self._pixmap_cache)
                font_n = len(self._font_cache)
                self._icon_cache.clear()
                self._pixmap_cache.clear()
                self._font_cache.clear()
                logger.debug(f"ThemeManager caches cleared (icons={icon_n}, pixmaps={pix_n}, fonts={font_n})")
            except Exception as e:
                logger.debug(f"ThemeManager cache clear failed: {e}")
        try:
            ThreadManager.run_on_ui_thread(_do_clear)
        except Exception:
            # Best-effort fallback
            _do_clear()

    def shutdown(self) -> None:
        """Deterministic cleanup: disconnect signals and clear caches."""
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        # Disconnect settings signal
        try:
            if getattr(self, "_settings_manager", None) is not None:
                try:
                    self._settings_manager.setting_changed.disconnect(self._on_setting_changed)
                except Exception:
                    pass
        except Exception:
            logger.debug("ThemeManager settings disconnect failed", exc_info=True)
        # Clear caches on UI thread
        self._clear_caches_async()
        # Do not unregister self here to avoid recursion when called from ResourceManager cleanup
        logger.info("ThemeManager shutdown complete")
    
    def _get_application_style(self) -> str:
        """Get the complete application stylesheet for the current theme.
        
        Note: This method is kept for backward compatibility but should not be used
        directly. Styles should be defined in the theme's QSS file.
        """
        # Get the current theme's colors
        colors = self.get_theme_colors()
        
        # Create a dictionary of style variables
        style_vars = {
            'base': colors.get('base', '#1e1e1e'),
            'text': colors.get('text', '#ffffff'),
            'border': colors.get('border', '#666666'),
            'highlight': colors.get('highlight', '#0078d7'),
            'highlight_text': colors.get('highlight_text', '#ffffff'),
            'disabled': colors.get('disabled', '#555555'),
            'disabled_text': colors.get('disabled_text', '#999999'),
            'button': colors.get('button', '#444444'),
            'button_hover': colors.get('button_hover', '#555555'),
            'button_pressed': colors.get('button_pressed', '#333333'),
            'button_text': colors.get('button_text', '#ffffff'),
            'button_border': colors.get('button_border', '#dddddd'),
            'input_bg': colors.get('input_bg', '#2d2d2d'),
            'input_text': colors.get('input_text', '#ffffff'),
            'tooltip_bg': colors.get('tooltip_bg', '#2d2d2d'),
            'tooltip_text': colors.get('tooltip_text', '#ffffff'),
            'link': colors.get('link', '#4a9cff'),
            'link_visited': colors.get('link_visited', '#b07bff'),
            'panel_bg': 'rgba(26, 26, 26, 0.8)',  # Semi-transparent background for panel
            'title_bg': '#2d2d2d',                # Slightly lighter for title bar
            'title_text': '#ffffff',               # White text for title
            'section_bg': '#262626',               # Slightly lighter than panel for sections
            'section_border': '#3d3d3d'            # Border color for sections
        }
        
        # Format the stylesheet with all required variables using string formatting
        return """
            /* Base application styles */
            QWidget {{
                font-family: "Segoe UI";
                font-size: 12px;
                color: {text};
                background: transparent;
            }}
            
            /* Main Dialog */
            MainDialog {{
                background-color: {panel_bg};
                border: 1px solid {section_border};
                border-radius: 8px;
            }}
            
            /* Title Bar */
            #titleBar {{
                background-color: {title_bg};
                padding: 8px 12px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                border-bottom: 1px solid {section_border};
            }}
            
            #titleLabel {{
                color: {title_text};
                font-size: 14px;
                font-weight: bold;
                padding: 0;
                margin: 0;
            }}
            
            /* Close Button */
            #closeButton {{
                background: transparent;
                border: none;
                color: {text};
                font-size: 16px;
                font-weight: bold;
                padding: 0 8px;
                margin: 0;
                min-width: 20px;
                min-height: 20px;
                border-radius: 3px;
            }}
            
            #closeButton:hover {{
                background: #ff4444;
                color: white;
            }}
            
            /* Badge Section */
            #badgeSection {{
                background-color: {section_bg};
                border: 1px solid {section_border};
                border-radius: 4px;
                padding: 12px;
                margin: 8px;
            }}
            
            #badgeLabel {{
                font-weight: bold;
                margin-bottom: 8px;
            }}
            
            /* Buttons */
            {button_style}
            
            /* Combo boxes */
            {combo_style}
            
            /* Line edits */
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, 
            QDateEdit, QDateTimeEdit, QTimeEdit {{
                background-color: {input_bg};
                color: {input_text};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
                min-width: 200px;
            }}
            
            /* Sections */
            .section {{
                background-color: {section_bg};
                border: 1px solid {section_border};
                border-radius: 4px;
                padding: 12px;
                margin: 8px;
            }}
            
            .section-title {{
                font-weight: bold;
                margin-bottom: 8px;
                color: {highlight};
            }}
            
            /* Form Layouts */
            .form-row {{
                margin-bottom: 8px;
            }}
            
            .form-label {{
                min-width: 120px;
            }}
            
            /* Checkboxes and radio buttons */
            QCheckBox::indicator, QRadioButton::indicator {{
                width: 16px;
                height: 16px;
            }}
            
            QCheckBox::indicator:unchecked {{
                border: 1px solid {border};
                background: {input_bg};
            }}
            
            QCheckBox::indicator:checked {{
                border: 1px solid {highlight};
                background: {highlight};
            }}
            QRadioButton::indicator:unchecked {{
                border: 1px solid {border};
                background: {input_bg};
                border-radius: 8px;
            }}
            QRadioButton::indicator:checked {{
                border: 1px solid {highlight};
                background: {highlight};
                border-radius: 8px;
            }}
            
            /* Group boxes */
            QGroupBox {{
                border: 1px solid {border};
                border-radius: 4px;
                margin-top: 1em;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 3px;
            }}
            
            /* Tab widgets */
            QTabWidget::pane {{
                border: 1px solid {border};
                border-top: none;
                border-radius: 0 0 4px 4px;
                padding: 5px;
                background: {base};
            }}
            QTabBar::tab {{
                background: {button};
                color: {button_text};
                border: 1px solid {border};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px 10px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected, QTabBar::tab:hover {{
                background: {button_hover};
            }}
            QTabBar::tab:selected {{
                border-bottom: 1px solid {base};
                margin-bottom: -1px;
            }}
            
            /* Tool tips */
            QToolTip {{
                background-color: {tooltip_bg};
                color: {tooltip_text};
                border: 1px solid {border};
                padding: 4px;
            }}
            
            /* Links */
            QLabel[href] {{
                color: {link};
                text-decoration: underline;
            }}
            QLabel[href]:hover {{
                color: {highlight};
            }}
            
            /* Disabled elements */
            QWidget:disabled {{
                color: {disabled_text};
                background-color: {disabled};
            }}
        """.format(**style_vars)
    
    def _update_application_palette(self) -> None:
        """Update the application palette based on the current theme."""
        if not self.app:
            return
            
        from PySide6.QtGui import QPalette, QColor
        from PySide6.QtCore import Qt
            
        palette = self.app.palette()
        colors = self.get_theme_colors()
        
        # Set base colors with fallback values
        palette.setColor(QPalette.Window, QColor(colors.get('base', '#1e1e1e')))
        palette.setColor(QPalette.WindowText, QColor(colors.get('text', '#ffffff')))
        palette.setColor(QPalette.Base, QColor(colors.get('input_bg', '#2d2d2d')))
        palette.setColor(QPalette.AlternateBase, QColor(colors.get('base', '#1e1e1e')))
        palette.setColor(QPalette.ToolTipBase, QColor(colors.get('tooltip_bg', '#2d2d2d')))
        palette.setColor(QPalette.ToolTipText, QColor(colors.get('tooltip_text', '#ffffff')))
        palette.setColor(QPalette.Text, QColor(colors.get('input_text', '#ffffff')))
        palette.setColor(QPalette.Button, QColor(colors.get('button', '#444444')))
        palette.setColor(QPalette.ButtonText, QColor(colors.get('button_text', '#ffffff')))
        palette.setColor(QPalette.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.Link, QColor(colors.get('link', '#4a9cff')))
        palette.setColor(QPalette.Highlight, QColor(colors.get('highlight', '#0078d7')))
        palette.setColor(QPalette.HighlightedText, QColor(colors.get('highlight_text', '#ffffff')))
        
        # Set disabled colors with fallback values
        disabled_text = colors.get('disabled_text', '#999999')
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, 
                        QColor(disabled_text))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, 
                        QColor(disabled_text))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, 
                        QColor(disabled_text))
        # Set the application palette
        self.app.setPalette(palette)
        
        # Force style refresh
        self.app.setStyle(self.app.style())
    
    def get_combo_style(self, theme_name: str = None) -> str:
        """Get the combo box style for the current theme.
        
        Args:
            theme_name: Optional theme name. If not provided, uses current theme.
            
        Returns:
            str: CSS style string for combo boxes
        """
        colors = self.get_theme_colors(theme_name or self._current_theme)
        
        return f"""
            QComboBox {{
                background-color: {colors.get('input_bg')};
                color: {colors.get('input_text')};
                border: 1px solid {colors.get('border')};
                border-radius: 4px;
                padding: 5px 8px;
                min-width: 6em;
                min-height: 24px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                width: 8px;
                height: 8px;
                background-color: {colors.get('text')};
                border-radius: 4px;
                margin-right: 6px;
                image: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors.get('input_bg')};
                color: {colors.get('input_text')};
                selection-background-color: {colors.get('highlight')};
                selection-color: {colors.get('highlight_text')};
                outline: 1px solid {colors.get('border')};
            }}
            QComboBox:hover {{
                background-color: {colors.get('button_hover')};
            }}
            QComboBox:disabled {{
                background-color: {colors.get('disabled')};
                color: {colors.get('disabled_text')};
            }}
        """

# Global instance
theme_manager = ThemeManager()

def get_theme_manager(app_instance=None) -> ThemeManager:
    """Get the theme manager instance, creating it if necessary."""
    if ThemeManager._instance is None:
        ThemeManager._instance = ThemeManager(app_instance)
    return ThemeManager._instance