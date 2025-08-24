"""
Core interfaces for the application's central services.

This module defines the abstract base classes (interfaces) for all core services
in the application. These interfaces define the contracts that concrete
implementations must follow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, runtime_checkable, TYPE_CHECKING

from PySide6.QtCore import QObject

# Type checking
if TYPE_CHECKING:
    pass  # Any type-checking only imports can go here

# Type variable for generic window types
T = TypeVar('T', bound='QObject')


class ResourceType(Enum):
    """Types of resources that can be managed by the resource manager."""
    WINDOW = "window"
    WINDOW_MANAGER = "window_manager"
    THREAD = "thread"
    FILE = "file"
    NETWORK = "network"
    MEMORY = "memory"
    GRAPHICS = "graphics"
    AUDIO = "audio"
    OTHER = "other"


@dataclass
class ResourceInfo:
    """Metadata about a registered resource."""
    resource_id: str
    resource_type: ResourceType
    description: str
    created_at: float
    metadata: Dict[str, Any] = None


@runtime_checkable
class IResourceManager(Protocol):
    """Interface for resource management."""
    
    def register(self, resource: Any, resource_type: ResourceType, 
               description: str = "", **metadata) -> str:
        """Register a resource for management.
        
        Args:
            resource: The resource to register
            resource_type: Type of the resource
            description: Human-readable description
            **metadata: Additional metadata
            
        Returns:
            str: Unique resource ID
        """
        ...
    
    def unregister(self, resource_id: str) -> bool:
        """Unregister and clean up a resource.
        
        Args:
            resource_id: ID of the resource to unregister
            
        Returns:
            bool: True if resource was found and unregistered, False otherwise
        """
        ...
    
    def get_resource(self, resource_id: str) -> Any:
        """Get a registered resource by ID.
        
        Args:
            resource_id: ID of the resource to retrieve
            
        Returns:
            The registered resource or None if not found
        """
        ...
    
    def list_resources(self, resource_type: Optional[ResourceType] = None) -> List[ResourceInfo]:
        """List all registered resources, optionally filtered by type.
        
        Args:
            resource_type: Optional resource type filter
            
        Returns:
            List of resource information objects
        """
        ...


@runtime_checkable
class IThreadManager(Protocol):
    """Interface for thread and task management."""
    
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for execution.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            str: Task ID that can be used to track or cancel the task
        """
        ...
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get the result of a submitted task.
        
        Args:
            task_id: ID of the task to get results for
            timeout: Maximum time to wait for the result (None = wait forever)
            
        Returns:
            The result of the task execution
            
        Raises:
            TimeoutError: If the timeout is reached before the task completes
            Exception: If the task raised an exception
        """
        ...
    
    def cancel(self, task_id: str) -> bool:
        """Attempt to cancel a running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            bool: True if the task was successfully cancelled, False otherwise
        """
        ...
    
    def create_ui_coalescer(self, name: str, capacity: int = 64, mode: str = "latest", window_ms: int = 7) -> Any:
        """Create a UI coalescer for batching UI tasks.
        
        Args:
            name: Name for the coalescer
            capacity: Queue capacity
            mode: Coalescing mode ('latest' or 'merge')
            window_ms: Batching window in milliseconds
            
        Returns:
            UICoalescer instance
        """
        ...
    
    def create_triple_buffer(self) -> Any:
        """Create a TripleBuffer for latest-value exchange.
        
        Returns:
            TripleBuffer instance
        """
        ...
    
    def create_spsc_queue(self, capacity: int) -> Any:
        """Create a bounded SPSC ring buffer.
        
        Args:
            capacity: Fixed capacity (>1)
            
        Returns:
            SPSCQueue instance
        """
        ...


@runtime_checkable
class IEventSystem(Protocol):
    """Interface for the event system."""
    
    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> str:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when the event is emitted
            
        Returns:
            str: Subscription ID that can be used to unsubscribe
        """
        ...
    
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events.
        
        Args:
            subscription_id: ID of the subscription to remove
        """
        ...
    
    def publish(self, event_type: str, data: Any = None) -> None:
        """Publish an event.
        
        Args:
            event_type: Type of event being published
            data: Optional data to pass to subscribers
        """
        ...


@runtime_checkable
class IWindowManager(Protocol[T]):
    """Interface for window management.
    
    This is a Protocol that defines the interface for window management.
    Implementations should either inherit from this Protocol or implement
    all its methods to be considered compatible.
    """
    
    # Signals (type hints only, actual signals should be defined in implementation)
    window_created: Any  # pyqtSignal(str)
    window_closed: Any   # pyqtSignal(str)
    window_shown: Any    # pyqtSignal(str)
    window_hidden: Any   # pyqtSignal(str)
    window_focus_changed: Any  # pyqtSignal(str, bool)
    
    def create_window(self, window_type: str, **kwargs) -> Optional[T]:
        """Create a new window.
        
        Args:
            window_type: Type of window to create
            **kwargs: Additional window-specific parameters
            
        Returns:
            The created window object or None if creation failed
        """
        ...
    
    def close_window(self, window_id: str) -> bool:
        """Close a window.
        
        Args:
            window_id: ID of the window to close
            
        Returns:
            bool: True if the window was found and closed, False otherwise
        """
        ...
    
    def get_window(self, window_id: str) -> Optional[T]:
        """Get a window by ID.
        
        Args:
            window_id: ID of the window to retrieve
            
        Returns:
            The window object or None if not found
        """
        ...
        
    def connect(self, signal: str, slot: Callable) -> None:
        """Connect a signal to a slot.
        
        Args:
            signal: Name of the signal to connect
            slot: Callable to connect to the signal
        """
        ...
        
    def disconnect(self, signal: str = None, slot: Callable = None) -> None:
        """Disconnect a signal from a slot or all slots.
        
        Args:
            signal: Name of the signal to disconnect (or None for all signals)
            slot: Callable to disconnect (or None for all slots)
        """
        ...


class ISettingsManager(ABC):
    """Interface for settings management.
    
    This is an abstract base class that defines the interface for settings management.
    Concrete implementations should inherit from this class and implement all abstract methods.
    """
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value.
        
        Args:
            key: Setting key in dot notation (e.g., 'app.theme.color')
            default: Default value to return if key is not found
            
        Returns:
            The setting value or default if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set a setting value.
        
        Args:
            key: Setting key in dot notation
            value: Value to set
        """
        pass
    
    @abstractmethod
    def save(self) -> None:
        """Save all settings to persistent storage."""
        pass
    
    @abstractmethod
    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        pass


@runtime_checkable
class IHotkeyManager(Protocol):
    """Interface for hotkey management."""
    
    def register_hotkey(self, key_combination: str, callback: Callable[[], None]) -> str:
        """Register a global hotkey.
        
        Args:
            key_combination: String representing the key combination (e.g., 'Ctrl+Alt+P')
            callback: Function to call when the hotkey is pressed
            
        Returns:
            str: Hotkey ID that can be used to unregister the hotkey
        """
        ...
        
    def unregister_hotkey(self, hotkey_id: str) -> bool:
        """Unregister a previously registered hotkey.
        
        Args:
            hotkey_id: ID of the hotkey to unregister
            
        Returns:
            bool: True if the hotkey was unregistered, False otherwise
        """
        ...
        
    def unregister_all_hotkeys(self) -> None:
        """Unregister all hotkeys."""
        ...


@runtime_checkable
class ILogger(Protocol):
    """Interface for logging."""
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        ...
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        ...
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        ...
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        ...
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        ...
    
    def exception(self, message: str, **kwargs) -> None:
        """Log an exception with stack trace."""
        ...
