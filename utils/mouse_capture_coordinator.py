"""
Mouse Capture Coordination Module

Provides centralized coordination of mouse capture between different systems
to prevent conflicts between window behavior and context menu systems.
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum
from threading import RLock

from PySide6.QtWidgets import QWidget

from core.logging import get_logger


class MouseCapturePriority(Enum):
    """Priority levels for mouse capture requests."""
    CONTEXT_MENU = 100      # Highest priority - context menus
    WINDOW_BEHAVIOR = 50    # Medium priority - drag/resize
    DEFAULT = 1             # Lowest priority - default handling


@dataclass
class CaptureRequest:
    """Represents a mouse capture request."""
    requester: str
    widget: QWidget
    priority: MouseCapturePriority
    reason: str = ""


class MouseCaptureCoordinator:
    """Coordinates mouse capture between different systems to prevent conflicts."""
    
    def __init__(self):
        """Initialize the mouse capture coordinator."""
        self._lock = RLock()
        self._current_capture: Optional[CaptureRequest] = None
        self._capture_stack: list[CaptureRequest] = []
        self._logger = get_logger(__name__)
    
    def request_capture(
        self, 
        requester: str, 
        widget: QWidget, 
        priority: MouseCapturePriority,
        reason: str = ""
    ) -> bool:
        """Request mouse capture with specified priority.
        
        Args:
            requester: Identifier of the requesting system
            widget: Widget that needs mouse capture
            priority: Priority level for this capture request
            reason: Optional reason for the capture request
            
        Returns:
            bool: True if capture was granted, False if denied
        """
        with self._lock:
            request = CaptureRequest(requester, widget, priority, reason)
            
            # Check if we can grant this request
            if self._current_capture is None:
                # No current capture, grant immediately
                return self._grant_capture(request)
            
            # Check priority
            if priority.value > self._current_capture.priority.value:
                # Higher priority, override current capture
                self._logger.debug(
                    f"Mouse capture override: {self._current_capture.requester} "
                    f"(priority {self._current_capture.priority.value}) -> "
                    f"{requester} (priority {priority.value})"
                )
                
                # Store current capture in stack for restoration
                self._capture_stack.append(self._current_capture)
                
                # Release current capture
                self._release_current_capture()
                
                # Grant new capture
                return self._grant_capture(request)
            
            elif priority.value == self._current_capture.priority.value:
                # Same priority, deny request but log
                self._logger.debug(
                    f"Mouse capture denied: {requester} (same priority as current {self._current_capture.requester})"
                )
                return False
            
            else:
                # Lower priority, deny request
                self._logger.debug(
                    f"Mouse capture denied: {requester} (priority {priority.value} < current {self._current_capture.priority.value})"
                )
                return False
    
    def release_capture(self, requester: str) -> bool:
        """Release mouse capture for the specified requester.
        
        Args:
            requester: Identifier of the system releasing capture
            
        Returns:
            bool: True if capture was released, False if not held by requester
        """
        with self._lock:
            if self._current_capture is None:
                self._logger.debug(f"Mouse capture release ignored: no current capture (from {requester})")
                return False
            
            if self._current_capture.requester != requester:
                self._logger.warning(
                    f"Mouse capture release denied: {requester} does not hold capture "
                    f"(held by {self._current_capture.requester})"
                )
                return False
            
            # Release current capture
            self._release_current_capture()
            
            # Restore previous capture if any
            if self._capture_stack:
                previous_request = self._capture_stack.pop()
                self._logger.debug(f"Restoring previous mouse capture: {previous_request.requester}")
                return self._grant_capture(previous_request)
            
            return True
    
    def is_captured_by(self, requester: str) -> bool:
        """Check if mouse is currently captured by the specified requester.
        
        Args:
            requester: Identifier to check
            
        Returns:
            bool: True if captured by requester, False otherwise
        """
        with self._lock:
            return (self._current_capture is not None and 
                   self._current_capture.requester == requester)
    
    def get_current_capturer(self) -> Optional[str]:
        """Get the identifier of the current mouse capturer.
        
        Returns:
            Optional[str]: Current capturer identifier or None if no capture
        """
        with self._lock:
            return self._current_capture.requester if self._current_capture else None
    
    def force_release_all(self) -> None:
        """Force release all mouse captures (emergency cleanup)."""
        with self._lock:
            if self._current_capture:
                self._logger.warning(f"Force releasing mouse capture from {self._current_capture.requester}")
                self._release_current_capture()
            
            self._capture_stack.clear()
            self._logger.debug("All mouse captures force released")
    
    def _grant_capture(self, request: CaptureRequest) -> bool:
        """Grant mouse capture to the specified request.
        
        Args:
            request: Capture request to grant
            
        Returns:
            bool: True if capture was successfully granted
        """
        try:
            request.widget.grabMouse()
            self._current_capture = request
            self._logger.debug(
                f"Mouse capture granted: {request.requester} "
                f"(priority {request.priority.value}) - {request.reason}"
            )
            return True
        except Exception as e:
            self._logger.error(f"Failed to grant mouse capture to {request.requester}: {e}")
            return False
    
    def _release_current_capture(self) -> None:
        """Release the current mouse capture."""
        if self._current_capture:
            try:
                self._current_capture.widget.releaseMouse()
                self._logger.debug(f"Mouse capture released: {self._current_capture.requester}")
            except Exception as e:
                self._logger.error(f"Failed to release mouse capture from {self._current_capture.requester}: {e}")
            finally:
                self._current_capture = None


# Global coordinator instance
_capture_coordinator: Optional[MouseCaptureCoordinator] = None


def get_mouse_capture_coordinator() -> MouseCaptureCoordinator:
    """Get the global mouse capture coordinator instance."""
    global _capture_coordinator
    if _capture_coordinator is None:
        _capture_coordinator = MouseCaptureCoordinator()
    return _capture_coordinator


def request_mouse_capture(
    requester: str, 
    widget: QWidget, 
    priority: MouseCapturePriority,
    reason: str = ""
) -> bool:
    """Convenience function to request mouse capture.
    
    Args:
        requester: Identifier of the requesting system
        widget: Widget that needs mouse capture
        priority: Priority level for this capture request
        reason: Optional reason for the capture request
        
    Returns:
        bool: True if capture was granted, False if denied
    """
    coordinator = get_mouse_capture_coordinator()
    return coordinator.request_capture(requester, widget, priority, reason)


def release_mouse_capture(requester: str) -> bool:
    """Convenience function to release mouse capture.
    
    Args:
        requester: Identifier of the system releasing capture
        
    Returns:
        bool: True if capture was released, False if not held by requester
    """
    coordinator = get_mouse_capture_coordinator()
    return coordinator.release_capture(requester)


def is_mouse_captured_by(requester: str) -> bool:
    """Convenience function to check if mouse is captured by requester.
    
    Args:
        requester: Identifier to check
        
    Returns:
        bool: True if captured by requester, False otherwise
    """
    coordinator = get_mouse_capture_coordinator()
    return coordinator.is_captured_by(requester)
