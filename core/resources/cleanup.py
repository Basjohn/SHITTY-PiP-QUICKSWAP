"""
Resource cleanup utilities.

This module provides cleanup handlers for various resource types,
including file system resources like __pycache__ directories.
"""
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from .manager import ResourceManager
from .types import ResourceType


def register_pycache_cleaner(
    base_dir: str,
    resource_manager: Optional[ResourceManager] = None,
) -> str:
    """Register a cleanup handler for __pycache__ directories.
    
    Args:
        base_dir: Base directory to search for __pycache__ directories
        resource_manager: Optional resource manager instance. If not provided,
                        the default resource manager will be used.
                        
    Returns:
        str: Resource ID for the registered cleanup handler
    """
    if resource_manager is None:
        from ..resources import get_resource_manager
        resource_manager = get_resource_manager()
    
    def cleanup_pycache() -> None:
        """Clean up all __pycache__ directories under the base directory."""
        base_path = Path(base_dir).resolve()
        if not base_path.exists() or not base_path.is_dir():
            return
            
        for root, dirs, _ in os.walk(base_path, topdown=False):
            if "__pycache__" in root.split(os.sep):
                continue  # Skip nested __pycache__ directories
                
            pycache = Path(root) / "__pycache__"
            if pycache.exists() and pycache.is_dir():
                try:
                    shutil.rmtree(pycache, ignore_errors=False)
                except Exception as e:
                    # Log the error but don't fail the cleanup
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to remove {pycache}: {e}")
    
    # Register the cleanup function with the resource manager
    return resource_manager.register(
        resource=cleanup_pycache,
        resource_type=ResourceType.CLEANUP_HANDLER,
        description=f"__pycache__ cleaner for {base_dir}",
        cleanup_handler=lambda func: func(),  # Execute the registered cleanup callable
        base_dir=str(base_dir),
    )


def register_cleanup_handler(
    cleanup_func: Callable[[], None],
    description: str = "",
    resource_manager: Optional[ResourceManager] = None,
    **metadata,
) -> str:
    """Register a generic cleanup handler.
    
    Args:
        cleanup_func: Function to call during cleanup
        description: Description of the cleanup handler
        resource_manager: Optional resource manager instance
        **metadata: Additional metadata for the resource
        
    Returns:
        str: Resource ID for the registered cleanup handler
    """
    if resource_manager is None:
        from ..resources import get_resource_manager
        resource_manager = get_resource_manager()
    
    return resource_manager.register(
        resource=cleanup_func,
        resource_type=ResourceType.CLEANUP_HANDLER,
        description=description or "Custom cleanup handler",
        cleanup_handler=lambda func: func(),  # Execute the registered cleanup callable
        **metadata,
    )
