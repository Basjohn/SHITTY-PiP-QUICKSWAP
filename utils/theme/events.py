"""
Theme System Events

Provides centralized event handling for theme-related events and errors.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional, List

from PySide6.QtCore import QObject, Signal


class ThemeErrorType(Enum):
    """Types of theme errors that can occur."""
    MISSING_TOKEN = auto()
    INVALID_TOKEN_VALUE = auto()
    THEME_NOT_FOUND = auto()
    THEME_LOAD_FAILED = auto()
    VALIDATION_FAILED = auto()
    QSS_ERROR = auto()


@dataclass
class ThemeError:
    """Represents a theme-related error."""
    error_type: ThemeErrorType
    message: str
    theme_name: str
    component: Optional[str] = None
    token: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ThemeEvents(QObject):
    """
    Central event bus for theme-related events.
    
    This class provides signals for theme-related events:
    - theme_changed: Emitted when the theme changes
    - theme_error: Emitted when a theme error occurs
    - theme_validated: Emitted when a theme is validated
    - theme_component_registered: Emitted when a component registers with the theme system
    """
    # Signals
    theme_changed = Signal(str)  # theme_name
    theme_error = Signal(ThemeError)  # error
    theme_validated = Signal(str, bool)  # theme_name, is_valid
    theme_component_registered = Signal(str, List[str])  # component_name, required_tokens
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def instance(cls) -> 'ThemeEvents':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the event bus."""
        super().__init__()
        
    def emit_theme_error(self, 
                       error_type: ThemeErrorType,
                       message: str,
                       theme_name: str,
                       component: Optional[str] = None,
                       token: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit a theme error event.
        
        Args:
            error_type: Type of error
            message: Error message
            theme_name: Name of the theme where the error occurred
            component: Optional component name
            token: Optional token name
            details: Optional additional details
        """
        error = ThemeError(
            error_type=error_type,
            message=message,
            theme_name=theme_name,
            component=component,
            token=token,
            details=details
        )
        self.theme_error.emit(error)


# Global instance
theme_events = ThemeEvents.instance()
