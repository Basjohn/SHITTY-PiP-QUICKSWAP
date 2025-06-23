#!/usr/bin/env python3
"""
Common constants used across multiple files in Shitty PiP
"""
from PySide6.QtGui import QColor


# --- Global Default Constants ---
DEFAULT_POSITION_PRESET = "Top Left"
DEFAULT_WINDOW_OVERLAY_WIDTH = 480
DEFAULT_WINDOW_OVERLAY_HEIGHT = 320
DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR = 0.30  # 30% of screen width
DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR = 0.30 # 30% of screen height
# --- End Global Default Constants ---

# --- Theme Constants ---
class ThemeColors:
    """Centralized theme color definitions for consistent theming across the application."""
    
    # Dark Theme
    DARK = {
        'name': 'dark',
        'primary': QColor(0, 120, 215),  # Blue accent
        'background': QColor(30, 30, 30),
        'foreground': QColor(240, 240, 240),
        'border': QColor(40, 40, 40),
        'fill': QColor(20, 20, 20),
        'highlight': QColor(0, 102, 204),
        'button': {
            'background': QColor(68, 68, 68),
            'text': QColor(255, 255, 255),
            'border': QColor(119, 119, 119),
            'hover': QColor(85, 85, 85),
            'pressed': QColor(51, 51, 51)
        },
        'menu': {
            'background': QColor(30, 30, 30),
            'text': QColor(240, 240, 240),
            'border': QColor(40, 40, 40),
            'highlight': QColor(0, 120, 215)
        }
    }
    
    # Light Theme
    LIGHT = {
        'name': 'light',
        'primary': QColor(0, 102, 204),  # Slightly darker blue for better contrast
        'background': QColor(240, 240, 240),
        'foreground': QColor(30, 30, 30),
        'border': QColor(180, 180, 180),
        'fill': QColor(200, 200, 200),
        'highlight': QColor(0, 122, 204),
        'button': {
            'background': QColor(224, 224, 224),  # ~30% lighter than dark theme
            'text': QColor(0, 0, 0),  # Black text
            'border': QColor(0, 0, 0),  # Black border
            'hover': QColor(240, 240, 240),
            'pressed': QColor(208, 208, 208)
        },
        'menu': {
            'background': QColor(240, 240, 240),
            'text': QColor(30, 30, 30),
            'border': QColor(180, 180, 180),
            'highlight': QColor(0, 102, 204)
        }
    }
    
    @classmethod
    def get_theme_colors(cls, theme_name):
        """Get the color dictionary for the specified theme."""
        theme_name = theme_name.lower() if theme_name else 'dark'
        return cls.DARK if theme_name == 'dark' else cls.LIGHT
    
    @classmethod
    def get_theme_stylesheet(cls, theme_name):
        """Get a stylesheet string for the specified theme."""
        # Prevent Nuitka from removing unused parameter

        colors = cls.get_theme_colors(theme_name)
        menu = colors['menu']
        
        # Base styles for menus
        base_styles = f"""
            QMenu {{
                background-color: rgb({menu['background'].red()}, {menu['background'].green()}, {menu['background'].blue()});
                color: rgb({menu['text'].red()}, {menu['text'].green()}, {menu['text'].blue()});
                border: 1px solid rgb({menu['border'].red()}, {menu['border'].green()}, {menu['border'].blue()});
                padding: 5px;
            }}
            QMenu::item:selected {{
                background-color: rgb({menu['highlight'].red()}, {menu['highlight'].green()}, {menu['highlight'].blue()});
            }}
            QMenu::item:disabled {{
                color: gray;
            }}
            QMenu::separator {{
                height: 1px;
                background: rgb({menu['border'].red()}, {menu['border'].green()}, {menu['border'].blue()});
                margin: 4px 8px;
            }}
        """
        
        # Try to load the full theme stylesheet from resources
        try:
            from PySide6.QtCore import QFile, QTextStream, QIODevice
            
            # Map theme name to resource path
            theme_map = {
                'dark': ':/themes/dark.qss',
                'light': ':/themes/light.qss'
            }
            
            theme_path = theme_map.get(theme_name.lower())
            if theme_path:
                file = QFile(theme_path)
                if file.open(QIODevice.ReadOnly | QIODevice.Text):
                    stream = QTextStream(file)
                    theme_styles = stream.readAll()
                    file.close()
                    return base_styles + '\n' + theme_styles
                else:
                    print(f"Warning: Could not open theme file: {theme_path}")
        except Exception as e:
            print(f"Warning: Could not load theme from resources: {e}")
            import traceback
            traceback.print_exc()
        
        # Fall back to basic styles if theme loading fails
        return base_styles
# --- End Theme Constants ---