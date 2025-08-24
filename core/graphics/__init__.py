"""
Core overlay management system.

This package provides a unified interface for managing overlay windows with
support for multiple rendering backends (DWM, OpenGL, etc.).
"""
from typing import Optional

from PySide6.QtCore import QRect

from .backends import BackendType
from .overlay_manager import OverlayManager

# Public API
__all__ = [
    'OverlayManager',
    'BackendType',
    'create_overlay',
    'update_overlay',
    'destroy_overlay',
    'get_overlay_manager'
]

# Singleton instance
_overlay_manager: Optional[OverlayManager] = None

def get_overlay_manager() -> OverlayManager:
    """Get the global overlay manager instance."""
    global _overlay_manager
    if _overlay_manager is None:
        _overlay_manager = OverlayManager()
    return _overlay_manager

def create_overlay(hwnd: int, rect: QRect, 
                 backend: BackendType = BackendType.AUTO) -> bool:
    """
    Create a new overlay window.
    
    Args:
        hwnd: Window handle to attach overlay to
        rect: Initial position and size
        backend: Rendering backend to use
        
    Returns:
        bool: True if overlay was created successfully
    """
    return get_overlay_manager().create_overlay(hwnd, rect, backend)

def update_overlay(hwnd: int, rect: QRect) -> bool:
    """
    Update an existing overlay.
    
    Args:
        hwnd: Window handle of the overlay
        rect: New position and size
        
    Returns:
        bool: True if overlay was updated successfully
    """
    return get_overlay_manager().update_overlay(hwnd, rect)

def destroy_overlay(hwnd: int) -> bool:
    """
    Destroy an overlay window.
    
    Args:
        hwnd: Window handle of the overlay to destroy
        
    Returns:
        bool: True if overlay was destroyed successfully
    """
    return get_overlay_manager().destroy_overlay(hwnd)
