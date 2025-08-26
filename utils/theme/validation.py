"""
Theme Token Validation Utilities

Provides centralized validation for theme tokens across the application.
Ensures all required theme tokens are present in theme files.
"""
from typing import Dict, List, Set, Optional
from .theme_manager import ThemeManager
from core.logging import get_logger

logger = get_logger(__name__)

# Registry of required theme tokens by component
_REQUIRED_TOKENS: Dict[str, Set[str]] = {
    # Core theme requirements
    "core": {
        "base",
        "text",
        "border",
        "highlight",
        "highlight_text",
    },
    
    # Border overlay requirements
    "border_overlay": {
        "overlay.border.stroke",
        "overlay.border.thickness.base",
        "overlay.border.accent",
        "overlay.border.rounded.enabled",
    }
}


def register_required_tokens(component: str, tokens: List[str]) -> None:
    """
    Register required tokens for a component.
    
    Args:
        component: Component name
        tokens: List of required tokens
    """
    global _REQUIRED_TOKENS
    
    if component not in _REQUIRED_TOKENS:
        _REQUIRED_TOKENS[component] = set()
        
    for token in tokens:
        _REQUIRED_TOKENS[component].add(token)
    
    logger.debug(f"Registered {len(tokens)} required tokens for {component}")
    

def get_required_tokens(component: Optional[str] = None) -> Set[str]:
    """
    Get required tokens for a component or all components.
    
    Args:
        component: Optional component name. If None, returns all required tokens.
        
    Returns:
        Set of required token names
    """
    if component is not None:
        return _REQUIRED_TOKENS.get(component, set())
        
    # Combine all tokens
    all_tokens = set()
    for tokens in _REQUIRED_TOKENS.values():
        all_tokens.update(tokens)
    return all_tokens


def validate_theme(theme_name: str, component: Optional[str] = None) -> bool:
    """
    Validate that a theme contains all required tokens.
    
    Args:
        theme_name: Name of the theme to validate
        component: Optional component name to validate tokens for
        
    Returns:
        True if the theme is valid, False otherwise
        
    Raises:
        ValueError: If theme manager is not initialized
    """
    theme_manager = ThemeManager.instance()
    
    required_tokens = get_required_tokens(component)
    missing_tokens = []
    
    for token in required_tokens:
        try:
            theme_manager.get_token(token, theme_name)
        except ValueError:
            missing_tokens.append(token)
    
    if missing_tokens:
        logger.error(f"Theme '{theme_name}' missing required tokens: {', '.join(missing_tokens)}")
        return False
        
    return True


def validate_all_themes(component: Optional[str] = None) -> Dict[str, bool]:
    """
    Validate all themes against required tokens.
    
    Args:
        component: Optional component name to validate tokens for
        
    Returns:
        Dictionary of theme names to validation results
    """
    theme_manager = ThemeManager.instance()
    
    # Get all themes from theme manager
    themes = getattr(theme_manager, "_themes", {}).keys()
    
    results = {}
    for theme in themes:
        results[theme] = validate_theme(theme, component)
    
    return results
