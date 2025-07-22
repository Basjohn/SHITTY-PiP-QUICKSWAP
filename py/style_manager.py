import logging
from typing import Optional, Dict
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QObject, Signal, QFile, QIODevice

logger = logging.getLogger(__name__)

class StyleManager(QObject):
    """
    Manages application theming and styles.
    """
    theme_changed = Signal(str)  # Signal emitted when theme changes
    
    _instance = None
    
    # Theme color definitions
    THEMES = {
        'dark': {
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
            'highlight': '#0078d7',
            'highlight_text': '#ffffff',
            'disabled': '#555555',
            'disabled_text': '#999999'
        },
        'light': {
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
            'border': '#cccccc',
            'highlight': '#0066cc',
            'highlight_text': '#ffffff',
            'disabled': '#e0e0e0',
            'disabled_text': '#999999'
        }
    }
    
    def __init__(self, app_instance=None):
        super().__init__()
        self.app_instance = app_instance
        self.current_theme = 'dark'
        self.theme_styles = {}
        
    @classmethod
    def instance(cls, app_instance=None):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(app_instance)
        return cls._instance
    
    def initialize(self):
        """Initialize the style manager and load default theme."""
        self.load_theme_resources()
        
        # Load saved theme if available
        if self.app_instance and hasattr(self.app_instance, 'settings'):
            saved_theme = self.app_instance.settings.value('theme', 'dark')
            self.apply_theme(saved_theme)
    
    def load_theme_resources(self):
        """Load theme resources from QRC files."""
        # Main theme styles
        for theme in ['dark', 'light']:
            style_file = QFile(f":/themes/{theme}.qss")
            if style_file.open(QIODevice.ReadOnly | QIODevice.Text):
                self.theme_styles[theme] = bytes(style_file.readAll()).decode('utf-8')
                style_file.close()
            else:
                logger.warning(f"Failed to load {theme} theme from resources")
                self.theme_styles[theme] = self._get_fallback_style(theme)
        
        # Subsettings specific styles
        self.subsettings_styles = {}
        for theme in ['dark', 'light']:
            style_file = QFile(f":/themes/subsettings{theme}.qss")
            if style_file.open(QIODevice.ReadOnly | QIODevice.Text):
                self.subsettings_styles[theme] = bytes(style_file.readAll()).decode('utf-8')
                style_file.close()
    
    def _get_fallback_style(self, theme: str) -> str:
        """Generate a fallback style if theme files can't be loaded."""
        colors = self.THEMES.get(theme, self.THEMES['dark'])
        return f"""
            QWidget {{
                background-color: {colors['base']};
                color: {colors['text']};
                font-family: "Segoe UI";
            }}
            QPushButton {{
                background-color: {colors['button']};
                color: {colors['button_text']};
                border: 2px solid {colors['button_border']};
                border-radius: 5px;
                padding: 6px 12px;
                min-height: 30px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {colors['button_hover']};
            }}
            QPushButton:pressed {{
                background-color: {colors['button_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {colors['disabled']};
                color: {colors['disabled_text']};
            }}
            QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
                background-color: {colors['input_bg']};
                color: {colors['input_text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 5px 8px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {colors['text']};
            }}
        """
    
    def get_theme_colors(self, theme: str = None) -> Dict[str, str]:
        """Get color definitions for a theme."""
        if theme is None:
            theme = self.current_theme
        return self.THEMES.get(theme, self.THEMES['dark'])
    
    def apply_theme(self, theme: str, widget: Optional[QWidget] = None, is_subsettings: bool = False):
        """
        Apply a theme to the application or a specific widget.
        
        Args:
            theme: Theme name ('light' or 'dark')
            widget: Optional widget to apply theme to. If None, applies to QApplication.
            is_subsettings: Whether to apply subsettings-specific styles.
        """
        theme = theme.lower()
        if theme not in ['light', 'dark']:
            theme = 'dark'
            logger.warning(f"Invalid theme '{theme}'. Using 'dark'.")
        
        self.current_theme = theme
        
        # Get the appropriate style sheet
        if is_subsettings:
            style = self.subsettings_styles.get(theme, self.theme_styles.get(theme, ''))
            logger.debug("SUBSETTING THEME'D! Applying subsettings theme: %s", theme)
        else:
            style = self.theme_styles.get(theme, '')
            logger.debug("Applying main theme: %s to %s", theme, 
                        widget.__class__.__name__ if widget else 'QApplication')
        
        # Apply the style
        target = widget if widget is not None else QApplication.instance()
        if not target:
            logger.warning("No target widget or QApplication instance found for theme application")
            return style
            
        try:
            target.setStyleSheet(style)
            
            # Force style refresh - only call update() on widgets, not on QApplication
            if isinstance(target, QWidget):
                target.style().unpolish(target)
                target.style().polish(target)
                target.update()
                logger.debug("Successfully applied theme to widget: %s", target.objectName() or target.__class__.__name__)
        except Exception as e:
            logger.error("Error applying theme to %s: %s", 
                        target.__class__.__name__, str(e), exc_info=True)
        
        # If this is the main application, emit theme changed signal
        if widget is None:
            logger.debug("Emitting theme_changed signal for theme: %s", theme)
            self.theme_changed.emit(theme)
            
            # Save theme preference if we have an app instance with settings
            if self.app_instance and hasattr(self.app_instance, 'settings'):
                self.app_instance.settings.setValue('theme', theme)
                self.app_instance.settings.sync()
                logger.debug("Saved theme preference: %s", theme)
        
        return style
    
    def get_button_style(self) -> str:
        """Get button style for the current theme."""
        colors = self.get_theme_colors()
        return f"""
            QPushButton {{
                background-color: {colors['button']};
                color: {colors['button_text']};
                border: 2px solid {colors['button_border']};
                border-radius: 5px;
                padding: 6px 12px;
                font-family: "Segoe UI";
                font-size: 12px;
                font-weight: {'600' if self.current_theme == 'dark' else '500'};
                min-height: 30px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {colors['button_hover']};
                {'border-color: #ffffff;' if self.current_theme == 'dark' else 'border-color: #333333;'}
            }}
            QPushButton:pressed {{
                background-color: {colors['button_pressed']};
            }}
            QPushButton:disabled {{
                background-color: {colors['disabled']};
                color: {colors['disabled_text']};
                border-color: #666666;
            }}
        """
    
    def get_combo_style(self) -> str:
        """Get combo box style for the current theme."""
        colors = self.get_theme_colors()
        return f"""
            QComboBox {{
                background-color: {colors['input_bg']};
                color: {colors['input_text']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 5px 8px;
                font-family: "Segoe UI";
                font-size: 12px;
                min-height: 24px;
            }}
            QComboBox:hover {{
                border-color: #999999;
            }}
            QComboBox::drop-down {{
                width: 20px;
                border: none;
                background: {colors['input_bg']};
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {colors['input_text']};
            }}
            QComboBox::drop-down:hover {{
                background-color: {colors['button_hover']};
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['input_bg']};
                color: {colors['input_text']};
                border: 1px solid {colors['border']};
                selection-background-color: {colors['highlight']};
                selection-color: {colors['highlight_text']};
            }}
        """
    
    def get_checkbox_style(self) -> str:
        """Get checkbox style for the current theme."""
        colors = self.get_theme_colors()
        if self.current_theme == 'light':
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
        else:
            return f"""
                QCheckBox {{
                    color: {colors['text']};
                    spacing: 4px;
                    font-size: 11px;
                    padding: 4px 0;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                    border: 2px solid #888888;
                    border-radius: 9px;
                    background: #333333;
                }}
                QCheckBox::indicator:checked {{
                    border: 2px solid #ffffff;
                    background: transparent;
                }}
                QCheckBox::indicator:unchecked:hover {{
                    border-color: #aaaaaa;
                }}
                QCheckBox:checked {{
                    font-weight: bold;
                }}
            """

# Global instance
style_manager = StyleManager()
