"""
Cursor Management Module

Provides centralized cursor management to prevent conflicts between different
systems that need to control cursor appearance.
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum
from threading import RLock

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt

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
    
    def __init__(self):
        """Initialize the cursor manager."""
        self._lock = RLock()
        self._current_cursor: Optional[CursorRequest] = None
        self._cursor_stack: list[CursorRequest] = []
        self._logger = get_logger(__name__)
    
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
        with self._lock:
            request = CursorRequest(requester, widget, cursor, priority, reason)
            
            # Check if we can grant this request
            if self._current_cursor is None:
                # No current cursor override, grant immediately
                return self._apply_cursor(request)
            
            # Check if this is the same widget and requester
            if (self._current_cursor.widget == widget and 
                self._current_cursor.requester == requester):
                # Same requester and widget, update cursor
                return self._apply_cursor(request)
            
            # Check priority
            if priority.value > self._current_cursor.priority.value:
                # Higher priority, override current cursor
                # Reduce cursor override logging spam
                if not hasattr(self, '_last_cursor_log') or self._last_cursor_log != requester:
                    self._logger.debug(
                        f"Cursor override: {self._current_cursor.requester} "
                        f"(priority {self._current_cursor.priority.value}) -> "
                        f"{requester} (priority {priority.value})"
                    )
                    self._last_cursor_log = requester
                
                # Store current cursor in stack for restoration
                self._cursor_stack.append(self._current_cursor)
                
                # Apply new cursor
                return self._apply_cursor(request)
            
            elif priority.value == self._current_cursor.priority.value:
                # Same priority, allow if same widget
                if self._current_cursor.widget == widget:
                    return self._apply_cursor(request)
                else:
                    self._logger.debug(
                        f"Cursor denied: {requester} (same priority, different widget)"
                    )
                    return False
            
            else:
                # Lower priority, deny request
                self._logger.debug(
                    f"Cursor denied: {requester} (priority {priority.value} < current {self._current_cursor.priority.value})"
                )
                return False
    
    def unset_cursor(self, requester: str, widget: QWidget) -> bool:
        """Unset cursor for the specified requester and widget.
        
        Args:
            requester: Identifier of the system releasing cursor
            widget: Widget to restore cursor for
            
        Returns:
            bool: True if cursor was unset, False if not held by requester
        """
        with self._lock:
            if self._current_cursor is None:
                # Reduce cursor override logging spam
                if not hasattr(self, '_last_cursor_log') or self._last_cursor_log != requester:
                    self._logger.debug(f"Cursor unset ignored: no current cursor override (from {requester})")
                    self._last_cursor_log = requester
                return False
            
            if (self._current_cursor.requester != requester or 
                self._current_cursor.widget != widget):
                self._logger.debug(
                    f"Cursor unset denied: {requester} does not hold cursor for this widget"
                )
                return False
            
            # Restore default cursor
            try:
                widget.unsetCursor()
                self._logger.debug(f"Cursor unset: {requester}")
            except Exception as e:
                self._logger.error(f"Failed to unset cursor for {requester}: {e}")
            
            self._current_cursor = None
            
            # Restore previous cursor if any
            if self._cursor_stack:
                previous_request = self._cursor_stack.pop()
                self._logger.debug(f"Restoring previous cursor: {previous_request.requester}")
                return self._apply_cursor(previous_request)
            
            return True
    
    def is_cursor_managed_by(self, requester: str, widget: QWidget) -> bool:
        """Check if cursor is currently managed by the specified requester for widget.
        
        Args:
            requester: Identifier to check
            widget: Widget to check
            
        Returns:
            bool: True if cursor is managed by requester for widget, False otherwise
        """
        with self._lock:
            return (self._current_cursor is not None and 
                   self._current_cursor.requester == requester and
                   self._current_cursor.widget == widget)
    
    def get_current_cursor_manager(self) -> Optional[str]:
        """Get the identifier of the current cursor manager.
        
        Returns:
            Optional[str]: Current cursor manager identifier or None if no override
        """
        with self._lock:
            return self._current_cursor.requester if self._current_cursor else None
    
    def force_unset_all(self) -> None:
        """Force unset all cursor overrides (emergency cleanup)."""
        with self._lock:
            if self._current_cursor:
                self._logger.warning(f"Force unsetting cursor from {self._current_cursor.requester}")
                try:
                    self._current_cursor.widget.unsetCursor()
                except Exception as e:
                    self._logger.error(f"Failed to force unset cursor: {e}")
            
            self._current_cursor = None
            self._cursor_stack.clear()
            self._logger.debug("All cursor overrides force unset")
    
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
        requester: Identifier to check
        widget: Widget to check
        
    Returns:
        bool: True if cursor is managed by requester for widget, False otherwise
    """
    manager = get_cursor_manager()
    return manager.is_cursor_managed_by(requester, widget)
