"""
Window management utilities for the application.

This package contains modules for managing windows, including window positioning,
snapping, monitor detection, DWM thumbnails, overlay management, state persistence,
and integration with the application core.

IMPORTANT: All window behavior functionality is now centralized in behavior.py.
Use WindowBehaviorManager and related classes/functions from there.
"""

# Centralized window behavior
from .behavior import (
    WindowBehaviorManager,
    WindowManagementCore,
    WindowState,
    WindowStateManager,
    apply_snap,
    get_resize_edge_for_pos,
    get_cursor_for_edge
)

# Overlay and thumbnail management
from core.graphics.overlay_manager import OverlayManager
from core.graphics.overlay import OverlayConfig
from .thumbnail_manager import ThumbnailManager

__all__ = [
    # Centralized window behavior
    'WindowBehaviorManager',
    'WindowManagementCore',
    'WindowState',
    'WindowStateManager',
    'apply_snap',
    'get_resize_edge_for_pos',
    'get_cursor_for_edge',
    
    # Overlay and thumbnail management
    'OverlayManager',
    'OverlayConfig',
    'ThumbnailManager'
]

from .monitors import (  # noqa: F401
    MonitorInfo, get_physical_monitor_info, get_all_monitor_rects,
    get_virtual_screen_rect, find_monitor_for_window, get_screen_scale_factor,
    ensure_within_available_desktop, invalidate_cache, get_monitor_at_position,
    get_primary_monitor, calculate_window_center, is_window_on_monitor,
    get_best_monitor_for_window, get_available_geometry_for_monitor,
    center_window_on_monitor, get_monitor_count, is_multi_monitor_setup,
    get_snap_zones_for_monitor, CACHE_TTL_SECONDS, DEFAULT_DPI
)

# Add monitor utilities to __all__
__all__.extend([
    'MonitorInfo', 'get_physical_monitor_info', 'get_all_monitor_rects',
    'get_virtual_screen_rect', 'find_monitor_for_window', 'get_screen_scale_factor',
    'ensure_within_available_desktop', 'invalidate_cache', 'get_monitor_at_position',
    'get_primary_monitor', 'calculate_window_center', 'is_window_on_monitor',
    'get_best_monitor_for_window', 'get_available_geometry_for_monitor',
    'center_window_on_monitor', 'get_monitor_count', 'is_multi_monitor_setup',
    'get_snap_zones_for_monitor', 'CACHE_TTL_SECONDS', 'DEFAULT_DPI'
])

# Version information
__version__ = "1.0.0"
