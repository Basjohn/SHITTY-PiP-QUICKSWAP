"""
Docking System - Multi-overlay window management for SPQ Docker.

This module provides a complete docking system that manages multiple overlays
in a coordinated layout, allowing users to view and interact with multiple
windows simultaneously in a docked configuration.

Key Components:
- DockingOverlayManager: Orchestrates the 3-overlay system with single-source positioning
- DockingOverlay: Individual overlay wrapper with synchronization
"""

from .manager import DockingOverlayManager, get_docking_manager
from .overlay import DockingOverlay

__all__ = [
    'DockingOverlayManager',
    'DockingOverlay',
    'get_docking_manager'
]
