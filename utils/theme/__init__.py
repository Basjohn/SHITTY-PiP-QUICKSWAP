"""
Theme Management Package

Provides centralized theming, styling, and asset management for the application.
This package consolidates the functionality previously in style_manager.py and theme_manager.py.
"""

from .theme_manager import ThemeManager, theme_manager, get_theme_manager

# For backward compatibility
__all__ = ['ThemeManager', 'theme_manager', 'get_theme_manager']