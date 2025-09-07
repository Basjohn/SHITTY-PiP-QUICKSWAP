"""
Window manager adapter.

This module provides an adapter that implements the IWindowManager interface
and delegates to WindowManagerImpl, resolving the metaclass conflict between
QObject and Protocol.
"""
from __future__ import annotations

import weakref
from typing import Any, Callable, Optional, TypeVar

from PySide6.QtCore import QObject, Signal

from core.threading import ThreadManager

from core.interfaces import IWindowManager
from core.logging import get_logger
from core.window.manager_impl import WindowManagerImpl

T = TypeVar('T')  # noqa: WPS111

class WindowManagerAdapter(QObject):
    """
    Adapter that implements IWindowManager protocol and delegates to WindowManagerImpl.
    
    This class uses composition to implement the IWindowManager protocol
    while inheriting from QObject, avoiding metaclass conflicts.
    """
    
    # Signals
    window_created = Signal(str)  # window_id
    window_closed = Signal(str)   # window_id
    window_shown = Signal(str)    # window_id
    window_hidden = Signal(str)   # window_id
    window_focus_changed = Signal(str, bool)  # window_id, has_focus
    
    def __init__(self, parent: Optional[QObject] = None) -> None:
        """Initialize the window manager adapter.
        
        Args:
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._logger = get_logger(__name__)
        self._impl: Optional[WindowManagerImpl] = None
        self._initialized = False
        self._pending_operations: list[Callable[[], None]] = []
        
        # Use weakref to avoid reference cycles
        self._weak_self = weakref.proxy(self)
        
        # Initialize the implementation in the main thread
        ThreadManager.single_shot(0, self._initialize_impl)
    
    def _initialize_impl(self) -> None:
        """Initialize the implementation in the main thread."""
        try:
            self._impl = WindowManagerImpl(self)
            self._initialized = True
            
            # Connect signals
            self._impl.window_created.connect(self.window_created)
            self._impl.window_closed.connect(self.window_closed)
            self._impl.window_shown.connect(self.window_shown)
            self._impl.window_hidden.connect(self.window_hidden)
            self._impl.window_focus_changed.connect(self.window_focus_changed)
            
            # Process any pending operations
            for op in self._pending_operations:
                op()
            self._pending_operations.clear()
            
            self._logger.debug("Window manager adapter initialized")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize window manager: {e}", exc_info=True)
            raise
    
    def _run_when_initialized(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Run the function when the implementation is initialized.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
            
        Raises:
            RuntimeError: If the window manager implementation is not available
            Exception: Any exception raised by the function
        """
        if self._initialized and self._impl is not None:
            return func(self._impl, *args, **kwargs)
        
        # Lock-free: Queue operation and use ThreadManager callback instead of threading.Event
        result: Optional[T] = None
        exception: Optional[Exception] = None
        operation_complete = False
        
        def wrapper() -> None:
            nonlocal result, exception, operation_complete
            try:
                if self._impl is not None:
                    result = func(self._impl, *args, **kwargs)
                else:
                    exception = RuntimeError("Window manager implementation not available")
            except Exception as e:
                exception = e
            finally:
                operation_complete = True
        
        self._pending_operations.append(wrapper)
        
        # Lock-free: Busy wait with ThreadManager yielding instead of blocking Event.wait()
        from core.threading import ThreadManager
        tm = ThreadManager()
        while not operation_complete:
            tm.yield_ui_thread()
        
        if exception:
            raise exception
        return result
    
    # IWindowManager implementation
    
    def create_window(self, window_type: str, **kwargs) -> str:
        """Create a new window."""
        return self._run_when_initialized(
            lambda impl, t, **kw: impl.create_window(t, **kw),
            window_type, **kwargs
        )
    
    def close_window(self, window_id: str) -> bool:
        """Close the specified window."""
        return self._run_when_initialized(
            lambda impl, wid: impl.close_window(wid),
            window_id
        )
    
    def get_window(self, window_id: str) -> Optional[Any]:
        """Get a window by its ID."""
        return self._run_when_initialized(
            lambda impl, wid: impl.get_window(wid),
            window_id
        )
    
    def show_window(self, window_id: str) -> bool:
        """Show the specified window."""
        return self._run_when_initialized(
            lambda impl, wid: impl.show_window(wid),
            window_id
        )
    
    def hide_window(self, window_id: str) -> bool:
        """Hide the specified window."""
        return self._run_when_initialized(
            lambda impl, wid: impl.hide_window(wid),
            window_id
        )
    
    def set_focus(self, window_id: str) -> bool:
        """Set focus to the specified window."""
        return self._run_when_initialized(
            lambda impl, wid: impl.set_focus(wid),
            window_id
        )
    
    def minimize_window(self, window_id: str) -> bool:
        """Minimize the specified window."""
        return self._run_when_initialized(
            lambda impl, wid: impl.minimize_window(wid),
            window_id
        )
    
    def maximize_window(self, window_id: str) -> bool:
        """Maximize the specified window."""
        return self._run_when_initialized(
            lambda impl, wid: impl.maximize_window(wid),
            window_id
        )
    
    def restore_window(self, window_id: str) -> bool:
        """Restore the specified window to normal state."""
        return self._run_when_initialized(
            lambda impl, wid: impl.restore_window(wid),
            window_id
        )
    
    def set_fullscreen(self, window_id: str, fullscreen: bool = True) -> bool:
        """Set fullscreen state for the specified window."""
        return self._run_when_initialized(
            lambda impl, wid, fs: impl.set_fullscreen(wid, fs),
            window_id, fullscreen
        )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._impl is not None:
            self._impl.cleanup()
        
        # Clear pending operations
        self._pending_operations.clear()

    def shutdown(self) -> None:
        """Shutdown the window manager adapter safely.
        
        - Invokes cleanup on the underlying implementation if present
        - Resets initialization state and implementation reference
        - Clears any pending operations
        - Never raises; logs errors internally
        """
        try:
            if self._impl is not None:
                try:
                    # Prefer an explicit shutdown on impl if it exists, else cleanup
                    if hasattr(self._impl, "shutdown"):
                        self._impl.shutdown()  # type: ignore[attr-defined]
                    else:
                        self._impl.cleanup()
                except Exception as e:
                    self._logger.error(f"WindowManagerAdapter.shutdown: impl cleanup failed: {e}", exc_info=True)
        finally:
            # Reset internal state regardless of impl outcome
            self._impl = None
            self._initialized = False
            try:
                self._pending_operations.clear()
            except Exception:
                pass
            try:
                self._logger.debug("WindowManagerAdapter shutdown complete")
            except Exception:
                pass


def create_window_manager(parent: Optional[QObject] = None) -> IWindowManager:
    """Create and return a properly typed window manager.
    
    Args:
        parent: Optional parent QObject
        
    Returns:
        IWindowManager: A window manager instance that delegates to a WindowManagerImpl
    """
    return WindowManagerAdapter(parent)
