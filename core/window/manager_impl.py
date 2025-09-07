"""
Window manager implementation.

This module contains the concrete implementation of the window manager
that handles window creation, management, and layout.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar, Union
import uuid

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from core.logging import get_logger
from core.window.types import WindowInfo, WindowType
from core.threading import ThreadManager

T = TypeVar('T')  # noqa: WPS111

class WindowManagerImpl(QObject):
    """
    Implementation of the window manager.
    
    This class handles the actual window management functionality.
    """
    
    # Signals
    window_created = Signal(str)  # window_id
    window_closed = Signal(str)   # window_id
    window_shown = Signal(str)    # window_id
    window_hidden = Signal(str)   # window_id
    window_focus_changed = Signal(str, bool)  # window_id, has_focus
    
    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Initialize the window manager.
        
        Args:
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._logger = get_logger(__name__)
        self._windows: Dict[str, WindowInfo] = {}
        # Lock-free: All mutations dispatched to UI thread via ThreadManager
        self._initialized = False
        
        # Initialize after the event loop starts
        ThreadManager.single_shot(0, self._initialize)
    
    def _initialize(self) -> None:
        """Initialize the window manager after the event loop starts."""
        # Lock-free: UI thread only access
        if self._initialized:
            return
            
        self._logger.info("Initializing window manager...")
        
        # Install event filter to track window events
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)
        
        self._initialized = True
        self._logger.info("Window manager initialized")
    
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Handle application events."""
        event_type = event.type()
        
        # Handle window activation changes
        if event_type == QEvent.WindowActivate:
            window = obj.window()
            if window:
                self._on_window_activated(window)
        
        # Let the event continue to be processed
        return super().eventFilter(obj, event)
    
    def _on_window_activated(self, window: QWidget) -> None:
        """Handle window activation."""
        window_id = getattr(window, 'window_id', None)
        if window_id and window_id in self._windows:
            self.window_focus_changed.emit(window_id, True)
    
    def create_window(  # noqa: WPS210
        self,
        window_type: Union[WindowType, str],
        **kwargs: Any
    ) -> str:
        """
        Create a new window.
        
        Args:
            window_type: Type of window to create
            **kwargs: Additional window parameters
                - window_id: Optional window ID (auto-generated if not provided)
                - title: Window title
                - size: Window size as QSize or (width, height) tuple
                - position: Window position as QPoint or (x, y) tuple
                - parent: Parent widget
                - flags: Window flags
                - visible: Whether to show the window (default: True)
        
        Returns:
            str: The window ID of the created window
        """
        if isinstance(window_type, str):
            try:
                window_type = WindowType[window_type.upper()]
            except KeyError:
                self._logger.error("Unknown window type: %s", window_type)
                raise ValueError(f"Unknown window type: {window_type!r}")
        
        # Generate window ID if not provided
        window_id = kwargs.get('window_id', f"window_{uuid.uuid4().hex[:8]}")
        
        # Lock-free: UI thread only access
        if window_id in self._windows:
            self._logger.warning(f"Window with ID {window_id} already exists")
            return ""
            
            # Create the window
            parent = kwargs.get('parent')
            window = QMainWindow(parent) if window_type == WindowType.MAIN else QWidget(parent)
            window.setObjectName(window_id)
            
            # Set window properties
            if 'title' in kwargs:
                window.setWindowTitle(kwargs['title'])
            
            if 'size' in kwargs:
                size = kwargs['size']
                if isinstance(size, (tuple, list)) and len(size) == 2:
                    window.resize(QSize(*size))
                elif isinstance(size, QSize):
                    window.resize(size)
            
            if 'position' in kwargs:
                pos = kwargs['position']
                if isinstance(pos, (tuple, list)) and len(pos) == 2:
                    window.move(QPoint(*pos))
                elif isinstance(pos, QPoint):
                    window.move(pos)
            
            if 'flags' in kwargs:
                window.setWindowFlags(kwargs['flags'])
            
            # Store window info
            window_info = WindowInfo(window_id, window, window_type, parent)
            self._windows[window_id] = window_info
            
            # Connect signals
            window.destroyed.connect(lambda: self._on_window_destroyed(window_id))
            
            # Show window if requested
            if kwargs.get('visible', True):
                window.show()
            
            self.window_created.emit(window_id)
            return window_id
    
    def _on_window_destroyed(self, window_id: str) -> None:
        """Handle window destruction."""
        # Lock-free: UI thread only access
        if window_id in self._windows:
            del self._windows[window_id]
            self.window_closed.emit(window_id)
    
    def close_window(self, window_id: str) -> bool:
        """Close the specified window.
        
        Args:
            window_id: ID of the window to close
            
        Returns:
            bool: True if window was found and closed, False otherwise
        """
        # Lock-free: UI thread only access
        if window_id not in self._windows:
            self._logger.warning("Window %s not found", window_id)
            return False
            
            window_info = self._windows[window_id]
            if window_info.window:
                window_info.window.close()
                return True
            return False
    
    def get_window(self, window_id: str) -> Optional[QWidget]:
        """Get a window by its ID.
        
        Args:
            window_id: ID of the window to retrieve
            
        Returns:
            Optional[QWidget]: The window widget if found, None otherwise
        """
        # Lock-free: UI thread only access
        if window_id in self._windows:
            return self._windows[window_id].window
        return None
    
    def get_window_info(self, window_id: str) -> Optional[WindowInfo]:
        """Get window information by ID."""
        # Lock-free: UI thread only access
        return self._windows.get(window_id)
    
    def show_window(self, window_id: str) -> bool:
        """Show the specified window.
        
        Args:
            window_id: ID of the window to show
            
        Returns:
            bool: True if window was shown, False otherwise
        """
        # Lock-free: UI thread only access
        if window_id not in self._windows:
            return False
            
            window_info = self._windows[window_id]
            if window_info.window:
                window = window_info.window
                window.show()
                window.raise_()
                window.activateWindow()
                window_info.update_state()
                self.window_shown.emit(window_id)
                return True
            return False
    
    def hide_window(self, window_id: str) -> bool:
        """Hide the specified window."""
        # Lock-free: UI thread only access
        if window_id not in self._windows:
            return False
            
            window_info = self._windows[window_id]
            if window_info.window:
                window_info.window.hide()
                window_info.update_state()
                self.window_hidden.emit(window_id)
                return True
            return False
    
    def set_focus(self, window_id: str) -> bool:
        """Set focus to the specified window."""
        return self.show_window(window_id)
    
    def minimize_window(self, window_id: str) -> bool:
        """Minimize the specified window."""
        # Lock-free: UI thread only access
        if window_id not in self._windows:
            return False
            
            window_info = self._windows[window_id]
            if window_info.window:
                window_info.window.showMinimized()
                window_info.update_state()
                return True
            return False
    
    def maximize_window(self, window_id: str) -> bool:
        """Maximize the specified window."""
        # Lock-free: UI thread only access
        if window_id not in self._windows:
            return False
            
            window_info = self._windows[window_id]
            if window_info.window:
                window_info.window.showMaximized()
                window_info.update_state()
                return True
            return False
    
    def restore_window(self, window_id: str) -> bool:
        """Restore the specified window to normal state."""
        # Lock-free: UI thread only access
        if window_id not in self._windows:
            return False
            
            window_info = self._windows[window_id]
            if window_info.window:
                window_info.window.showNormal()
                window_info.update_state()
                return True
            return False
    
    def set_fullscreen(self, window_id: str, fullscreen: bool = True) -> bool:
        """Set fullscreen state for the specified window."""
        # Lock-free: UI thread only access
        if window_id not in self._windows:
            return False
            
            window_info = self._windows[window_id]
            if window_info.window:
                if fullscreen:
                    window_info.window.showFullScreen()
                else:
                    window_info.window.showNormal()
                window_info.update_state()
                return True
            return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Lock-free: UI thread only access
        # Close all windows
        for window_id in list(self._windows.keys()):
            self.close_window(window_id)
            
            # Remove event filter
            app = QApplication.instance()
            if app:
                app.removeEventFilter(self)
