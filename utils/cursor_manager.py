"""
Cursor Management Module

Provides centralized cursor management to prevent conflicts between different
systems that need to control cursor appearance.
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

# Local imports - Core
from core.logging import get_logger


class CursorPriority(Enum):
    """Priority levels for cursor management requests."""
    CONTEXT_MENU = 100      # Highest priority - context menus
    WINDOW_BEHAVIOR = 50    # Medium priority - drag/resize cursors
    DEFAULT = 1             # Lowest priority - default cursors


@dataclass
class CursorRequest:
    """Represents a cursor management request."""
    requester: str
    widget: QWidget
    cursor: Qt.CursorShape
    priority: CursorPriority
    reason: str = ""


class CursorManager:
    """Coordinates cursor management between different systems to prevent conflicts."""
    
    _instance: Optional['CursorManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Implement singleton pattern - lock-free via UI thread confinement."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the cursor manager - idempotent."""
        if self._initialized:
            return
            
        self._logger = get_logger(__name__)
        self._current_cursor: Optional[CursorRequest] = None
        self._cursor_stack: list[CursorRequest] = []
        # Anti-spam signature for repeated identical set requests
        self._last_set_signature: Optional[tuple] = None
        
        # Register with ResourceManager for deterministic cleanup
        try:
            from utils.resource_manager import get_resource_manager, ResourceType
            self._resource_manager = get_resource_manager()
            self._resource_id = self._resource_manager.register(
                self,
                ResourceType.CUSTOM,
                "CursorManager singleton",
                cleanup_handler=lambda obj: obj._cleanup()
            )
            self._logger.debug("Registered CursorManager with ResourceManager")
        except Exception as e:
            self._logger.warning(f"Failed to register with ResourceManager: {e}")
            self._resource_manager = None
            self._resource_id = None
        
        self._initialized = True
    
    def set_cursor(
        self, 
        requester: str, 
        widget: QWidget, 
        cursor: Qt.CursorShape,
        priority: CursorPriority,
        reason: str = ""
    ) -> bool:
        """Set cursor with specified priority.
        
        Args:
            requester: Identifier of the requesting system
            widget: Widget that needs cursor change
            cursor: Cursor shape to set
            priority: Priority level for this cursor request
            reason: Optional reason for the cursor change
            
        Returns:
            bool: True if cursor was set, False if denied
        """
        # UI thread operation - no lock needed
        request = CursorRequest(requester, widget, cursor, priority, reason)
        
        # Short-circuit: if the exact same cursor request is already active, do nothing
        if (self._current_cursor is not None and
            self._current_cursor.requester == requester and
            self._current_cursor.widget is widget and
            self._current_cursor.cursor == cursor and
            self._current_cursor.priority == priority):
            # Avoid logging spam for identical no-op
            return True

        # Check if we can grant this request
        if self._current_cursor is None or priority.value >= self._current_cursor.priority.value:
            # Store previous cursor if we're overriding
            if self._current_cursor is not None:
                self._cursor_stack.append(self._current_cursor)
            
            self._current_cursor = request
            widget.setCursor(cursor)
            
            # Reduce spam: only log when signature changes
            sig = (requester, id(widget), cursor, priority)
            if self._last_set_signature != sig:
                self._logger.debug(f"Cursor set by {requester} on {widget} to {cursor.name} (priority {priority.name})")
                self._last_set_signature = sig
            return True
        else:
            # Reduce deny spam: only log when requester/priority changes
            sig = ("DENY", requester, priority, self._current_cursor.requester, self._current_cursor.priority)
            if self._last_set_signature != sig:
                self._logger.debug(f"Cursor request denied for {requester} (priority {priority.name} < current {self._current_cursor.priority.name})")
                self._last_set_signature = sig
            return False
    
    def unset_cursor(self, requester: str, widget: QWidget) -> bool:
        """Unset cursor for the specified requester and widget.
        
        Args:
            requester: Identifier of the system releasing cursor
            widget: Widget to restore cursor for
            
        Returns:
            bool: True if cursor was unset, False if not held by requester
        """
        # UI thread operation - no lock needed
        if self._current_cursor is None:
            # Reduce cursor override logging spam
            return True
        
        if self._current_cursor.requester != requester:
            self._logger.debug(f"Cannot unset cursor: held by {self._current_cursor.requester}, not {requester}")
            return False
        
        # Restore previous cursor from stack or default
        if self._cursor_stack:
            previous = self._cursor_stack.pop()
            self._current_cursor = previous
            widget.setCursor(previous.cursor)
            self._logger.debug(f"Cursor restored to {previous.requester} ({previous.cursor.name})")
        else:
            self._current_cursor = None
            widget.unsetCursor()
            # Reduce spam: do not log default restoration repeatedly
            if self._last_set_signature != ("DEFAULT",):
                self._logger.debug("Cursor restored to default")
                self._last_set_signature = ("DEFAULT",)
        
        return True
    
    def is_cursor_managed_by(self, requester: str, widget: QWidget) -> bool:
        """Check if cursor is currently managed by the specified requester for widget.
        
        Args:
            requester: Identifier to check
            widget: Widget to check
            
        Returns:
            bool: True if cursor is managed by requester for widget, False otherwise
        """
        # UI thread operation - no lock needed
        return (self._current_cursor is not None and 
               self._current_cursor.requester == requester and
               self._current_cursor.widget == widget)
    
    def get_current_cursor_manager(self) -> Optional[str]:
        """Get the identifier of the current cursor manager.
        
        Returns:
            Optional[str]: Current cursor manager identifier or None if no override
        """
        # UI thread operation - no lock needed
        return self._current_cursor.requester if self._current_cursor else None
    
    def force_unset_all(self) -> None:
        """Force unset all cursor overrides (emergency cleanup)."""
        # UI thread operation - no lock needed
        if self._current_cursor:
            self._logger.warning(f"Force unsetting cursor from {self._current_cursor.requester}")
            try:
                self._current_cursor.widget.unsetCursor()
            except Exception as e:
                self._logger.error(f"Failed to unset cursor: {e}")
        
        self._current_cursor = None
        self._cursor_stack.clear()
        self._logger.info("All cursor overrides cleared")
    
    def _cleanup(self):
        """Cleanup handler for ResourceManager."""
        try:
            # Clear all cursor requests
            self._current_cursor = None
            self._cursor_stack.clear()
            self._logger.debug("CursorManager cleanup completed")
        except Exception as e:
            self._logger.error(f"Error during CursorManager cleanup: {e}")
    
    def shutdown(self):
        """Explicit shutdown method."""
        if hasattr(self, '_resource_id') and self._resource_id and hasattr(self, '_resource_manager') and self._resource_manager:
            try:
                self._resource_manager.unregister(self._resource_id)
                self._resource_id = None
            except Exception as e:
                self._logger.warning(f"Failed to unregister from ResourceManager: {e}")
        self._cleanup()
    
    def _apply_cursor(self, request: CursorRequest) -> bool:
        """Apply cursor for the specified request.
        
        Args:
            request: Cursor request to apply
            
        Returns:
            bool: True if cursor was successfully applied
        """
        try:
            request.widget.setCursor(request.cursor)
            self._current_cursor = request
            self._logger.debug(
                f"Cursor set: {request.requester} "
                f"(priority {request.priority.value}) - {request.reason}"
            )
            return True
        except Exception as e:
            self._logger.error(f"Failed to set cursor for {request.requester}: {e}")
            return False


# Global cursor manager instance
_cursor_manager: Optional[CursorManager] = None


def get_cursor_manager() -> CursorManager:
    """Get the global cursor manager instance."""
    global _cursor_manager
    if _cursor_manager is None:
        _cursor_manager = CursorManager()
    return _cursor_manager


def set_managed_cursor(
    requester: str, 
    widget: QWidget, 
    cursor: Qt.CursorShape,
    priority: CursorPriority,
    reason: str = ""
) -> bool:
    """Convenience function to set managed cursor.
    
    Args:
        requester: Identifier of the requesting system
        widget: Widget that needs cursor change
        cursor: Cursor shape to set
        priority: Priority level for this cursor request
        reason: Optional reason for the cursor change
        
    Returns:
        bool: True if cursor was set, False if denied
    """
    manager = get_cursor_manager()
    return manager.set_cursor(requester, widget, cursor, priority, reason)


def unset_managed_cursor(requester: str, widget: QWidget) -> bool:
    """Convenience function to unset managed cursor.
    
    Args:
        requester: Identifier of the system releasing cursor
        widget: Widget to restore cursor for
        
    Returns:
        bool: True if cursor was unset, False if not held by requester
    """
    manager = get_cursor_manager()
    return manager.unset_cursor(requester, widget)


def is_cursor_managed_by(requester: str, widget: QWidget) -> bool:
    """Convenience function to check if cursor is managed by requester.
    
    Args:
        requester: Identifier of the system to check
        widget: Widget to check cursor for
        
    Returns:
        bool: True if cursor is managed by the specified requester
    """
    manager = get_cursor_manager()
    return manager.is_cursor_managed_by(requester, widget)
