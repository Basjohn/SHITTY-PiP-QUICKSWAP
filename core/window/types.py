"""
Window types and data structures.

This module defines the data structures and types used by the window management system.
"""
from __future__ import annotations

import time
from enum import Enum, auto
from typing import Any, Dict, Optional

from PySide6.QtCore import QPoint, QSize
from PySide6.QtWidgets import QWidget

class WindowType(Enum):
    """Types of windows that can be created."""
    MAIN = auto()
    DIALOG = auto()
    TOOL = auto()
    OVERLAY = auto()
    CUSTOM = auto()

class WindowState:
    """Represents the state of a window."""
    
    def __init__(self, 
                 window_id: str,
                 window_type: WindowType = WindowType.CUSTOM,
                 title: str = "",
                 size: QSize = None,
                 position: QPoint = None,
                 is_visible: bool = True,
                 is_minimized: bool = False,
                 is_maximized: bool = False,
                 is_fullscreen: bool = False,
                 parent_id: str = None):
        """Initialize window state."""
        self.window_id = window_id
        self.window_type = window_type
        self.title = title
        self.size = size or QSize(800, 600)
        self.position = position or QPoint(100, 100)
        self.is_visible = is_visible
        self.is_minimized = is_minimized
        self.is_maximized = is_maximized
        self.is_fullscreen = is_fullscreen
        self.parent_id = parent_id
        self.properties: Dict[str, Any] = {}  # noqa: F821
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert window state to a dictionary."""
        return {
            'window_id': self.window_id,
            'window_type': self.window_type.name,
            'title': self.title,
            'size': (self.size.width(), self.size.height()),
            'position': (self.position.x(), self.position.y()),
            'is_visible': self.is_visible,
            'is_minimized': self.is_minimized,
            'is_maximized': self.is_maximized,
            'is_fullscreen': self.is_fullscreen,
            'parent_id': self.parent_id,
            'properties': self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WindowState':
        """Create a WindowState from a dictionary."""
        state = cls(
            window_id=data['window_id'],
            window_type=WindowType[data['window_type']],
            title=data['title'],
            size=QSize(*data['size']) if 'size' in data else None,
            position=QPoint(*data['position']) if 'position' in data else None,
            is_visible=data.get('is_visible', True),
            is_minimized=data.get('is_minimized', False),
            is_maximized=data.get('is_maximized', False),
            is_fullscreen=data.get('is_fullscreen', False),
            parent_id=data.get('parent_id')
        )
        state.properties = data.get('properties', {})
        return state

class WindowInfo:
    """Information about a managed window."""
    
    def __init__(
        self,
        window_id: str,
        window: QWidget,
        window_type: WindowType = WindowType.CUSTOM,
        parent: Optional[QWidget] = None
    ) -> None:
        """Initialize window information."""
        self.window_id = window_id
        self.window = window
        self.window_type = window_type
        self.parent = parent
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.state = WindowState(
            window_id=window_id,
            window_type=window_type,
            title=window.windowTitle(),
            size=window.size(),
            position=window.pos(),
            is_visible=window.isVisible(),
            is_minimized=window.isMinimized(),
            is_maximized=window.isMaximized(),
            is_fullscreen=window.isFullScreen(),
            parent_id=parent.objectName() if parent else None
        )
    
    def update_state(self) -> None:
        """Update the window state from the actual window."""
        if not self.window:
            return
            
        self.state.title = self.window.windowTitle()
        self.state.size = self.window.size()
        self.state.position = self.window.pos()
        self.state.is_visible = self.window.isVisible()
        self.state.is_minimized = self.window.isMinimized()
        self.state.is_maximized = self.window.isMaximized()
        self.state.is_fullscreen = self.window.isFullScreen()
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert window info to a dictionary."""
        return {
            'window_id': self.window_id,
            'window_type': self.window_type.name,
            'parent_id': self.parent.objectName() if self.parent else None,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'state': self.state.to_dict()
        }
