"""
Centralized window menu population utilities for SPQ application.

This module provides functions to populate window switch menus with proper filtering,
icon handling, and caching to ensure consistent behavior across the application.
"""

import time
from typing import List, Tuple, Optional, Callable
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import QMenu

# Centralized logging and debugging
from debug_utils import get_logger, log_perf, debug_enabled, DebugTimer

# Initialize logger
logger = get_logger(__name__)

# Cache for window lists with timestamp and data
_window_list_cache = {
    'timestamp': 0,
    'data': None,
    'cached_icons': {}
}

# Cache duration in seconds (5 minutes)
CACHE_DURATION = 300

@log_perf(level=logging.DEBUG, threshold_ms=10.0)
def get_cached_window_list(app_instance) -> Optional[List[Tuple[int, str, Optional[QIcon]]]]:
    """
    Get the cached window list or refresh it if needed.
    
    Args:
        app_instance: The main application instance with window enumeration capabilities
        
    Returns:
        List of tuples (hwnd, title, icon) or None if no windows found
    """
    try:
        current_time = time.time()
        
        # Return cached data if it's still fresh
        if (_window_list_cache['data'] is not None and 
            current_time - _window_list_cache['timestamp'] < CACHE_DURATION):
            logger.debug("Using cached window list")
            return _window_list_cache['data']
        
        # Otherwise refresh the cache
        return refresh_window_list_cache(app_instance)
    except Exception:
        logger.exception("Error getting cached window list")
        return None

@log_perf(level=logging.DEBUG, threshold_ms=100.0)  # Higher threshold due to potential system calls
def refresh_window_list_cache(app_instance) -> Optional[List[Tuple[int, str, Optional[QIcon]]]]:
    """
    Refresh the window list cache with current window data.
    
    Args:
        app_instance: The main application instance
        
    Returns:
        List of tuples (hwnd, title, icon) or None if no windows found
    """
    try:
        with DebugTimer("Getting window list from system"):
            windows = app_instance.get_windows()  # This should return List[Tuple[int, str]]
            
        if not windows:
            logger.warning("No windows found in refresh_window_list_cache")
            return None
            
        # Convert to list of tuples with (hwnd, title, icon)
        window_list = []
        icon_errors = 0
        
        for hwnd, title in windows:
            # Skip invalid windows
            if not hwnd or not title or title.strip() == '':
                continue
                
            # Get window icon (this might be slow, consider caching)
            icon = None
            if hasattr(app_instance, 'get_window_icon'):
                try:
                    with DebugTimer(f"Getting icon for window {hwnd}"):
                        icon = app_instance.get_window_icon(hwnd)
                except Exception as e:
                    icon_errors += 1
                    if debug_enabled():
                        logger.debug(f"Error getting icon for window {hwnd}: {e}")
            
            window_list.append((hwnd, title, icon))
        
        # Update cache
        _window_list_cache['data'] = window_list
        _window_list_cache['timestamp'] = time.time()
        
        logger.info(f"Refreshed window list cache with {len(window_list)} windows")
        if icon_errors > 0:
            logger.debug(f"Encountered {icon_errors} errors while fetching window icons")
            
        return window_list
        
    except Exception:
        logger.exception("Error refreshing window list cache")
        return None

@log_perf(level=logging.DEBUG, threshold_ms=50.0)
def populate_window_switch_menu(menu: QMenu, app_instance, on_window_selected: Callable[[int], None]):
    """
    Populate a menu with window switch options.
    
    Args:
        menu: The QMenu to populate
        app_instance: The main application instance
        on_window_selected: Callback function that takes an HWND parameter
    """
    try:
        with DebugTimer("Clearing and populating window switch menu"):
            menu.clear()
            
            # Get window list (will use cache if recent)
            window_list = get_cached_window_list(app_instance)
            if not window_list:
                no_windows = menu.addAction("No windows found")
                no_windows.setEnabled(False)
                logger.info("No windows available to populate menu")
                return
            
            logger.debug(f"Populating menu with {len(window_list)} windows")
            
            # Add window items to menu
            for hwnd, title, icon in window_list:
                try:
                    action = QAction(menu)
                    action.setText(title)
                    if icon:
                        action.setIcon(icon)
                    
                    # Connect action to the callback with the HWND
                    action.triggered.connect(lambda checked, h=hwnd: on_window_selected(h))
                    menu.addAction(action)
                except Exception as e:
                    logger.error(f"Error adding window {hwnd} to menu: {e}")
                    continue
                    
    except Exception:
        logger.exception("Error populating window switch menu")
        # Add error item to menu
        menu.clear()
        error_action = menu.addAction("Error loading windows")
        error_action.setEnabled(False)
        return

def force_refresh_window_list(app_instance):
    """
    Force a refresh of the window list cache.
    
    Args:
        app_instance: The main application instance
    """
    global _window_list_cache
    logger.info("Forcing refresh of window list cache")
    _window_list_cache = {
        'timestamp': 0,  # Force refresh
        'data': None,
        'cached_icons': {}
    }
