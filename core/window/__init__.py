"""
Window management and utilities.

This package provides core window management functionality including window
enumeration, validation, icon handling, and window management.
"""

from .enumerator import WindowEnumerator
from .validation import (
    is_window_visible,
    get_window_rect,
    get_window_style,
    get_window_title,
    get_window_class,
    get_window_process_info,
    is_system_window,
    is_media_player,
    is_valid_window,
    # Constants
    SYSTEM_WINDOW_TITLES,
    SYSTEM_WINDOW_CLASSES,
    SYSTEM_PROCESSES,
    VIDEO_EXTENSIONS,
    MEDIA_PLAYER_PROCESSES,
    MEDIA_PLAYER_CLASSES
)
from .types import WindowType, WindowState, WindowInfo
from .adapter import create_window_manager, WindowManagerAdapter
from .manager_impl import WindowManagerImpl

__all__ = [
    # Window management
    'create_window_manager',
    'WindowManagerAdapter',
    'WindowManagerImpl',
    'WindowType',
    'WindowState',
    'WindowInfo',
    
    # Window enumeration and validation
    'WindowEnumerator',
    'is_window_visible',
    'get_window_rect',
    'get_window_style',
    'get_window_title',
    'get_window_class',
    'get_window_process_info',
    'is_system_window',
    'is_media_player',
    'is_valid_window',
    
    # Constants
    'SYSTEM_WINDOW_TITLES',
    'SYSTEM_WINDOW_CLASSES',
    'SYSTEM_PROCESSES',
    'VIDEO_EXTENSIONS',
    'MEDIA_PLAYER_PROCESSES',
    'MEDIA_PLAYER_CLASSES'
]
