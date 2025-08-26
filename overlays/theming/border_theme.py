from __future__ import annotations

from PySide6.QtGui import QColor
from core.logging import get_logger


class BorderTheme:
    """Centralized theme management for borders with strict token enforcement."""
    
    def __init__(self, theme_manager):
        self._theme_manager = theme_manager
        self._logger = get_logger("BorderTheme")
        
    def get_border_color(self) -> QColor:
        """Get current theme border color with strict enforcement."""
        try:
            # Check if we're in dark theme - always use white in dark theme per user requirement
            current_theme = self.get_theme_name().lower()
            if 'dark' in current_theme:
                return QColor(255, 255, 255)  # Always white in dark theme
            
            # Light theme - use theme token
            color_str = self._theme_manager.get_token('overlay.border.stroke')
            return QColor(color_str)
        except Exception as e:
            # Fail-fast: no fallback colors per user policy
            self._logger.critical(f"REQUIRED theme token 'overlay.border.stroke' missing: {e}")
            raise ValueError(f"Required theme token 'overlay.border.stroke' missing: {e}")
            
    def get_border_thickness_base(self) -> float:
        """Get base border thickness from theme."""
        try:
            thickness_str = self._theme_manager.get_token('overlay.border.thickness.base')
            return float(thickness_str)
        except Exception:
            # Light theme default: 2.0px, Dark theme: 1.5px
            # Try to infer from current theme name
            try:
                current_theme = getattr(self._theme_manager, '_current_theme', 'light')
                if 'dark' in current_theme.lower():
                    return 1.5  # Thinner for dark theme
                else:
                    return 2.0  # Standard for light theme
            except Exception:
                return 2.0  # Safe fallback
            
    def get_accent_color(self) -> QColor:
        """Get inner accent color for depth effect."""
        try:
            color_str = self._theme_manager.get_token('overlay.border.accent')
            return QColor(color_str)
        except Exception:
            # Fallback to theme-appropriate accent color
            current_theme = self.get_theme_name().lower()
            if 'dark' in current_theme:
                return QColor(100, 150, 255, 128)  # Subtle blue accent for dark theme
            else:
                return QColor(0, 0, 0, 64)  # Subtle dark accent for light theme
            
    def is_rounded_enabled_by_theme(self) -> bool:
        """Check if rounded borders are enabled in theme (theme-level override)."""
        try:
            enabled_str = self._theme_manager.get_token('overlay.border.rounded.enabled')
            return enabled_str.lower() == 'true'
        except Exception:
            return False  # Default to sharp corners
            
    def get_theme_name(self) -> str:
        """Get current theme name for debugging."""
        try:
            return getattr(self._theme_manager, '_current_theme', 'unknown')
        except Exception:
            return 'unknown'
            
    def get_accent_thickness(self) -> float:
        """Get inner accent thickness."""
        try:
            return float(self._theme_manager.get_token('overlay.border.accent.thickness'))
        except Exception:
            return 1.0  # Default thickness
            
    def get_accent_inset(self) -> float:
        """Get inner accent inset distance."""
        try:
            return float(self._theme_manager.get_token('overlay.border.accent.inset'))
        except Exception:
            return 3.0  # Default inset
    
    def validate_required_tokens(self) -> bool:
        """Validate that all required theme tokens are present."""
        required_tokens = ['overlay.border.stroke']
        
        for token in required_tokens:
            try:
                self._theme_manager.get_token(token)
            except Exception as e:
                self._logger.error(f"Missing required theme token '{token}': {e}")
                return False
                
        return True
