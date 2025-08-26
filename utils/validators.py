"""
Input validation utilities for the application.

This module provides centralized validation functions for various input types
used throughout the application.
"""

from typing import Union

# Debugging and logging utilities
from core.logging import get_logger

# Get logger instance
logger = get_logger(__name__)

# Constants for validation
VALID_THEMES = {'dark', 'light', 'system'}
VALID_WINDOW_SORT_ORDERS = {
    'alphabetical', 'reverse_alphabetical',
    'most_recent', 'least_recent',
    'largest_first', 'smallest_first'
}


def is_valid_theme(theme_name: str) -> bool:
    """
    Check if a theme name is valid.
    
    Args:
        theme_name: Name of the theme to validate
        
    Returns:
        bool: True if the theme is valid, False otherwise
    """
    return theme_name.lower() in VALID_THEMES


def is_valid_window_sort_order(sort_order: str) -> bool:
    """
    Check if a window sort order is valid.
    
    Args:
        sort_order: Sort order string to validate
        
    Returns:
        bool: True if the sort order is valid, False otherwise
    """
    return sort_order.lower() in VALID_WINDOW_SORT_ORDERS


def is_valid_opacity(opacity: Union[int, float]) -> bool:
    """
    Check if an opacity value is valid (0-100).
    
    Args:
        opacity: Opacity value to validate
        
    Returns:
        bool: True if the opacity is valid, False otherwise
    """
    try:
        opacity_float = float(opacity)
        return 0 <= opacity_float <= 100
    except (TypeError, ValueError):
        return False


def is_valid_fps(fps: Union[int, float, str]) -> bool:
    """
    Check if an FPS value is valid (1-144).
    
    Args:
        fps: FPS value to validate
        
    Returns:
        bool: True if the FPS is valid, False otherwise
    """
    try:
        fps_float = float(fps)
        return 1 <= fps_float <= 144
    except (TypeError, ValueError):
        return False


def is_valid_hotkey(hotkey: str) -> bool:
    """
    Check if a hotkey string is valid.
    
    Args:
        hotkey: Hotkey string to validate (e.g., 'Ctrl+Alt+P')
        
    Returns:
        bool: True if the hotkey is valid, False otherwise
    """
    if not hotkey or not isinstance(hotkey, str):
        return False
    
    # Split by + and check each part
    parts = [p.strip().lower() for p in hotkey.split('+') if p.strip()]
    if not parts:
        return False
    
    # Check modifiers and key
    modifiers = {'ctrl', 'alt', 'shift', 'win'}
    key = parts[-1]
    mods = parts[:-1]
    
    # Check if all modifiers are valid
    if not all(m in modifiers for m in mods):
        return False
    
    # Check if key is valid (simple check for now)
    return len(key) == 1 or key in {
        'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
        'esc', 'tab', 'capslock', 'space', 'enter', 'backspace', 'delete',
        'home', 'end', 'pageup', 'pagedown', 'insert', 'printscreen', 'scrolllock',
        'pause', 'break', 'numlock', 'add', 'subtract', 'multiply', 'divide',
        'decimal', 'left', 'right', 'up', 'down'
    }


def sanitize_theme_name(theme_name: str) -> str:
    """
    Sanitize and normalize a theme name.
    
    Args:
        theme_name: Theme name to sanitize
        
    Returns:
        str: Sanitized theme name, or 'dark' if invalid
    """
    if not theme_name or not isinstance(theme_name, str):
        return 'dark'
    
    theme_name = theme_name.lower().strip()
    return theme_name if theme_name in VALID_THEMES else 'dark'


def sanitize_window_sort_order(sort_order: str) -> str:
    """
    Sanitize and normalize a window sort order.
    
    Args:
        sort_order: Sort order to sanitize
        
    Returns:
        str: Sanitized sort order, or 'alphabetical' if invalid
    """
    if not sort_order or not isinstance(sort_order, str):
        return 'alphabetical'
    
    sort_order = sort_order.lower().strip()
    return sort_order if sort_order in VALID_WINDOW_SORT_ORDERS else 'alphabetical'