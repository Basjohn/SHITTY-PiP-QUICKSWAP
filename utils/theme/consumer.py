"""
Theme Consumer Interface

Defines the ThemeConsumer interface for components that consume theme data.
Allows for standardized theme application and automatic updates.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from core.logging import get_logger

logger = get_logger(__name__)


class ThemeConsumer(ABC):
    """Interface for components that consume theme data."""
    
    @abstractmethod
    def apply_theme(self, theme_name: str) -> None:
        """
        Apply the given theme to this component.
        
        Args:
            theme_name: Name of the theme to apply
        """
        pass
    
    @abstractmethod
    def get_required_tokens(self) -> List[str]:
        """
        Get the list of theme tokens required by this component.
        
        Returns:
            List of token names required by this component
        """
        pass
    
    def on_theme_changed(self, theme_name: str) -> None:
        """
        Called when the theme changes.
        
        Args:
            theme_name: Name of the new theme
        """
        try:
            self.apply_theme(theme_name)
        except Exception as e:
            logger.error(f"Failed to apply theme {theme_name}: {e}")
            
    def validate_theme(self, theme_name: str, theme_data: Dict[str, Any]) -> bool:
        """
        Validate the theme data against the required tokens.
        
        Args:
            theme_name: Name of the theme
            theme_data: Theme data dictionary
            
        Returns:
            True if the theme is valid, False otherwise
        """
        missing = []
        for token in self.get_required_tokens():
            if token not in theme_data:
                missing.append(token)
                
        if missing:
            logger.error(f"Theme {theme_name} missing required tokens: {', '.join(missing)}")
            return False
            
        return True
