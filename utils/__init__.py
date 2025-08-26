"""
Utils Package

This package contains utility modules for the application, including thread management,
resource management, and other helper functions.
"""

__all__ = ['thread_manager', 'resource_manager', 'cache_cleaner']

# Use lazy imports to avoid circular dependencies
import importlib
from typing import Any, Dict, TypeVar

# Type variable for generic type hints
T = TypeVar('T')

# Lazy loading cache
_import_cache: Dict[str, Any] = {}

def _lazy_import(name: str) -> Any:
    """Lazily import a module or attribute."""
    # Do not attempt to lazy import dunder attributes (e.g., __wrapped__ probed by inspect)
    if name.startswith('__') and name.endswith('__'):
        raise AttributeError(name)
    if name not in _import_cache:
        if name == 'ThreadManager':
            from .thread_manager import ThreadManager
            _import_cache[name] = ThreadManager
        elif name == 'ThreadPoolType':
            from .thread_manager import ThreadPoolType
            _import_cache[name] = ThreadPoolType
        elif name == 'TaskPriority':
            from .thread_manager import TaskPriority
            _import_cache[name] = TaskPriority
        elif name == 'TaskResult':
            from .thread_manager import TaskResult
            _import_cache[name] = TaskResult
        elif name == 'Task':
            from .thread_manager import Task
            _import_cache[name] = Task
        elif name == 'create_capture_manager':
            from .thread_manager import create_capture_manager
            _import_cache[name] = create_capture_manager
        elif name == 'ResourceManager':
            from .resource_manager import ResourceManager
            _import_cache[name] = ResourceManager
        elif name == 'ResourceType':
            from .resource_manager import ResourceType
            _import_cache[name] = ResourceType
        elif name == 'BufferPool':
            from .resource_manager import BufferPool
            _import_cache[name] = BufferPool
        elif name == 'OverlayState':
            from .resource_manager import OverlayState
            _import_cache[name] = OverlayState
        elif name == 'create_buffer_pool':
            from .resource_manager import create_buffer_pool
            _import_cache[name] = create_buffer_pool
        elif name == 'manage_overlay_state':
            from .resource_manager import manage_overlay_state
            _import_cache[name] = manage_overlay_state
        elif name == 'cleanup_pycache':
            from .cache_cleaner import cleanup_pycache
            _import_cache[name] = cleanup_pycache
        elif name == 'register_cleanup_on_exit':
            from .cache_cleaner import register_cleanup_on_exit
            _import_cache[name] = register_cleanup_on_exit
        else:
            # For any other attribute, try to import it directly
            parts = name.split('.')
            module = importlib.import_module('.' + parts[0], __name__)
            _import_cache[name] = getattr(module, parts[1]) if len(parts) > 1 else module
    
    return _import_cache[name]

# Define __getattr__ to enable lazy loading
# This allows us to use dot notation for imports while still lazy loading
def __getattr__(name: str) -> Any:
    """Lazily import and return the requested attribute."""
    # Avoid intercepting Python's internal dunder lookups
    if name.startswith('__') and name.endswith('__'):
        raise AttributeError(name)
    return _lazy_import(name)

# Define __dir__ to help with autocompletion
def __dir__():
    """List all available attributes, including lazily loaded ones."""
    return [
        'ThreadManager', 'ThreadPoolType', 'TaskPriority', 'TaskResult', 'Task',
        'create_capture_manager', 'ResourceManager', 'ResourceType', 'BufferPool',
        'OverlayState', 'create_buffer_pool', 'manage_overlay_state',
        'cleanup_pycache', 'register_cleanup_on_exit'
    ]