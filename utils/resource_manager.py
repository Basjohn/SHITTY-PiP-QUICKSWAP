"""
Centralized Resource Manager access point.

This module re-exports the canonical ResourceManager implementation from
`core.resources` to provide a single import location for resource management
throughout the app. This avoids duplication while enabling future evolution
of the utils-layer without changing call sites.
"""
from __future__ import annotations

# Re-export core implementation as the single source of truth
from core.resources import (
    ResourceManager as ResourceManager,
    ResourceType as ResourceType,
    ResourceInfo as ResourceInfo,
    CleanupProtocol as CleanupProtocol,
    get_resource_manager as get_resource_manager,
    register_resource as register_resource,
    unregister_resource as unregister_resource,
    get_resource as get_resource,
    list_resources as list_resources,
    cleanup_all as cleanup_all,
    register_gl_handle as register_gl_handle,
    register_gl_texture as register_gl_texture,
    register_gl_buffer as register_gl_buffer,
    register_gl_vertex_array as register_gl_vertex_array,
    register_gl_framebuffer as register_gl_framebuffer,
    register_gl_renderbuffer as register_gl_renderbuffer,
    register_gl_query as register_gl_query,
    register_gl_shader as register_gl_shader,
    register_gl_program as register_gl_program,
    register_gl_qt_program as register_gl_qt_program,
    register_gl_sync as register_gl_sync,
    register_gl_context as register_gl_context,
)

__all__ = [
    "ResourceManager",
    "ResourceType",
    "ResourceInfo",
    "CleanupProtocol",
    "get_resource_manager",
    "register_resource",
    "unregister_resource",
    "get_resource",
    "list_resources",
    "cleanup_all",
    # GL helpers
    "register_gl_handle",
    "register_gl_texture",
    "register_gl_buffer",
    "register_gl_vertex_array",
    "register_gl_framebuffer",
    "register_gl_renderbuffer",
    "register_gl_query",
    "register_gl_shader",
    "register_gl_program",
    "register_gl_qt_program",
    "register_gl_sync",
    "register_gl_context",
    # Utilities
    "cleanup_via_temp_registration",
    # Finders
    "find_resource_by_description",
    "find_resources_by_type",
]

# --- Utilities ---------------------------------------------------------------
def cleanup_via_temp_registration(
    resource: object,
    *,
    cleanup_handler,
    resource_type: ResourceType = ResourceType.CUSTOM,
    description: str = "",
    **metadata,
) -> bool:
    """Perform cleanup through ResourceManager using a temp registration.

    Registers the provided resource with the given cleanup_handler and then
    immediately unregisters it so the cleanup runs on the correct thread
    with proper ordering and context activation.

    Args:
        resource: The object/handle box to register temporarily.
        cleanup_handler: Callable performing actual cleanup.
        resource_type: ResourceType bucket for ordering.
        description: Optional description for logging/traceability.
        **metadata: Additional resource metadata.

    Returns:
        True if unregister dispatch was initiated.
    """
    rm = get_resource_manager()
    rid = rm.register(
        resource,
        resource_type=resource_type,
        description=description,
        cleanup_handler=cleanup_handler,
        **metadata,
    )
    return rm.unregister(rid)

# --- Finder helpers -----------------------------------------------------------
def find_resource_by_description(description: str):
    """Return the first live resource whose description matches exactly.

    This is useful for retrieving singletons (e.g., "DockingOverlayManager")
    without knowing their internal resource_id. Returns None if not found.
    """
    try:
        rm = get_resource_manager()
        infos = rm.list_resources()
        for ri in infos:
            try:
                if getattr(ri, "description", "") == description:
                    return rm.get(ri.resource_id)
            except Exception:
                continue
        return None
    except Exception:
        return None

def find_resources_by_type(resource_type):
    """Return a list of live resources matching the given ResourceType or name.

    resource_type can be a ResourceType enum or its string name.
    """
    try:
        rm = get_resource_manager()
        infos = rm.list_resources(resource_type)
        results = []
        for ri in infos:
            try:
                obj = rm.get(ri.resource_id)
                if obj is not None:
                    results.append(obj)
            except Exception:
                continue
        return results
    except Exception:
        return []
