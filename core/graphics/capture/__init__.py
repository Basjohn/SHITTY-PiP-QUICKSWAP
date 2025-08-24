"""
Monitor capture system for screen recording and display.

This package provides screen capture managers and a QWidget-based renderer
for displaying captured frames without OpenGL.
"""

from .monitor_capture_manager import MonitorCaptureManager, get_monitor_capture_manager, CaptureFrame
from .d3d11_monitor_renderer import D3D11MonitorRenderer
from .dwm_capture_manager import DwmCaptureManager, get_dwm_capture_manager, DwmContentRect

__all__ = [
    'MonitorCaptureManager',
    'get_monitor_capture_manager', 
    'D3D11MonitorRenderer',
    'CaptureFrame',
    'DwmCaptureManager',
    'get_dwm_capture_manager',
    'DwmContentRect'
]
