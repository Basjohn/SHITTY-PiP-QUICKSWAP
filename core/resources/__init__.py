"""
Resource management.

This package provides resource tracking and cleanup functionality.
"""
import atexit
from typing import Any, Dict, List, Optional, Union

from .manager import ResourceManager
from .types import CleanupProtocol, ResourceInfo, ResourceType
from . import cleanup

# Create a singleton instance
_resource_manager: Optional[ResourceManager] = None

def get_resource_manager() -> ResourceManager:
    """Get the singleton instance of the resource manager.
    
    Returns:
        ResourceManager: The singleton instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager

def register_resource(
    resource: Any,
    resource_type: Union[ResourceType, str] = ResourceType.UNKNOWN,
    description: str = "",
    **metadata
) -> str:
    """Register a resource for management.
    
    Args:
        resource: The resource to register
        resource_type: Type of the resource (enum or string)
        description: Human-readable description
        **metadata: Additional metadata about the resource
        
    Returns:
        str: Unique resource ID
    """
    if isinstance(resource_type, str):
        try:
            resource_type = ResourceType[resource_type.upper()]
        except KeyError:
            resource_type = ResourceType.UNKNOWN
    
    return get_resource_manager().register(
        resource=resource,
        resource_type=resource_type,
        description=description,
        **metadata
    )

def unregister_resource(resource_id: str) -> bool:
    """Unregister and clean up a resource.
    
    Args:
        resource_id: ID of the resource to unregister
        
    Returns:
        bool: True if resource was found and unregistered, False otherwise
    """
    return get_resource_manager().unregister(resource_id)

def get_resource(resource_id: str) -> Any:
    """Get a registered resource by ID.
    
    Args:
        resource_id: ID of the resource to retrieve
        
    Returns:
        The registered resource or None if not found or already garbage collected
    """
    return get_resource_manager().get(resource_id)

def list_resources(
    resource_type: Optional[Union[ResourceType, str]] = None,
    include_metadata: bool = False
) -> List[Union[ResourceInfo, Dict[str, Any]]]:
    """List all registered resources, optionally filtered by type.
    
    Args:
        resource_type: Optional resource type filter (can be enum or string)
        include_metadata: If True, returns full resource info objects
                        If False, returns just the resource IDs
                        
    Returns:
        List of resource information objects or resource IDs
    """
    resources = get_resource_manager().list_resources(resource_type)
    if include_metadata:
        return resources
    # Return just IDs when metadata is not requested
    return [ri.resource_id for ri in resources]

def cleanup_all() -> None:
    """Clean up all registered resources."""
    get_resource_manager().cleanup_all()

def shutdown() -> None:
    """Shut down the resource manager and clean up all resources."""
    global _resource_manager
    if _resource_manager is not None:
        _resource_manager.shutdown()
        _resource_manager = None

# --- OpenGL convenience registration helpers -------------------------------
def register_gl_handle(handle: int, *, description: str = "", delete_fn=None,
                       resource_type: ResourceType = ResourceType.CUSTOM,
                       make_current=None, done_current=None, **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_handle(
        handle,
        description=description,
        delete_fn=delete_fn,
        resource_type=resource_type,
        make_current=make_current,
        done_current=done_current,
        **metadata,
    )

def register_gl_texture(texture_id: int, *, delete_textures, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_texture(texture_id, delete_textures=delete_textures, description=description, **metadata)

def register_gl_buffer(buffer_id: int, *, delete_buffer, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_buffer(buffer_id, delete_buffer=delete_buffer, description=description, **metadata)

def register_gl_vertex_array(vao_id: int, *, delete_vertex_array, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_vertex_array(vao_id, delete_vertex_array=delete_vertex_array, description=description, **metadata)

def register_gl_framebuffer(fbo_id: int, *, delete_framebuffer, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_framebuffer(fbo_id, delete_framebuffer=delete_framebuffer, description=description, **metadata)

def register_gl_renderbuffer(rbo_id: int, *, delete_renderbuffer, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_renderbuffer(rbo_id, delete_renderbuffer=delete_renderbuffer, description=description, **metadata)

def register_gl_query(query_id: int, *, delete_query, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_query(query_id, delete_query=delete_query, description=description, **metadata)

def register_gl_shader(shader_id: int, *, delete_shader, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_shader(shader_id, delete_shader=delete_shader, description=description, **metadata)

def register_gl_program(program_id: int, *, delete_program, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_program(program_id, delete_program=delete_program, description=description, **metadata)

def register_gl_qt_program(program_obj: object, *, description: str = "", **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_qt_program(program_obj, description=description, **metadata)

def register_gl_sync(sync_obj: object, *, delete_sync, description: str = "", make_current=None, done_current=None, **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_sync(sync_obj, delete_sync=delete_sync, description=description,
                               make_current=make_current, done_current=done_current, **metadata)

def register_gl_context(ctx_obj: object, *, destroy_context, description: str = "", make_current=None, done_current=None, **metadata) -> str:
    rm = get_resource_manager()
    return rm.register_gl_context(ctx_obj, destroy_context=destroy_context, description=description,
                                  make_current=make_current, done_current=done_current, **metadata)

# Clean up on module unload
atexit.register(shutdown)

__all__ = [
    'ResourceManager',
    'ResourceType',
    'ResourceInfo',
    'CleanupProtocol',
    'get_resource_manager',
    'register_resource',
    'unregister_resource',
    'get_resource',
    'list_resources',
    'cleanup_all',
    'shutdown',
    'cleanup',  # Export the cleanup module
    # GL helpers
    'register_gl_handle',
    'register_gl_texture',
    'register_gl_buffer',
    'register_gl_vertex_array',
    'register_gl_framebuffer',
    'register_gl_renderbuffer',
    'register_gl_query',
    'register_gl_shader',
    'register_gl_program',
    'register_gl_qt_program',
    'register_gl_sync',
    'register_gl_context',
]
