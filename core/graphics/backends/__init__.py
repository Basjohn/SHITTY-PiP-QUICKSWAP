"""
Overlay rendering backends.

This module provides implementations of different rendering backends for overlays.
"""
from enum import Enum, auto
from typing import Type, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...base import OverlayBase

class BackendType(Enum):
    """Enumeration of available overlay backends."""
    AUTO = auto()
    DWM = auto()
    OPENGL = auto()
    SOFTWARE = auto()
    MONITOR = auto()

# Backend registry
_backend_registry: Dict[BackendType, Type['OverlayBase']] = {}

def register_backend(backend_type: BackendType, backend_class: Type['OverlayBase']) -> None:
    """Register a backend implementation.
    
    Args:
        backend_type: Type of the backend
        backend_class: Class implementing the backend
    """
    _backend_registry[backend_type] = backend_class

def get_backend(backend_type: BackendType) -> Optional[Type['OverlayBase']]:
    """Get a backend implementation by type.
    
    Args:
        backend_type: Type of the backend to get
        
    Returns:
        The backend class, or None if not found
    """
    return _backend_registry.get(backend_type)

__all__ = ['BackendType', 'register_backend', 'get_backend']
