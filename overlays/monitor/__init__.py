"""
Monitor overlay UI components.

This module provides UI components for monitor capture overlays with
consistent styling and behavior matching the integrated DWM overlays.
"""

from .monitor_overlay import MonitorOverlay
from .capture_display_widget import CaptureDisplayWidget

__all__ = [
    'MonitorOverlay',
    'CaptureDisplayWidget'
]
