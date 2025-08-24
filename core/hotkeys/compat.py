"""
Hotkey Manager Compatibility Layer

This module provides backward compatibility for code that hasn't been updated
to use the new HotkeyManager API directly.
"""

raise ImportError(
    "core.hotkeys.compat is removed. Use core.hotkeys.manager.HotkeyManager directly "
    "(e.g., from core.hotkeys.manager import HotkeyManager). No compatibility wrappers remain."
)
import warnings
from typing import Callable, Dict

from .manager import HotkeyManager

# Global instance for backward compatibility
_hotkey_manager_instance = None

def get_hotkey_manager() -> HotkeyManager:
    """
    Get the global HotkeyManager instance (singleton).
    
    Returns:
        HotkeyManager: The global HotkeyManager instance
    """
    global _hotkey_manager_instance
    if _hotkey_manager_instance is None:
        _hotkey_manager_instance = HotkeyManager()
    return _hotkey_manager_instance

def register_hotkey(hotkey_id: str, sequence: str, callback: Callable, 
                  suppress: bool = True) -> bool:
    """
    Register a global hotkey (compatibility function).
    
    Args:
        hotkey_id: Unique identifier for the hotkey
        sequence: The hotkey sequence (e.g., "ctrl+alt+s")
        callback: Function to call when the hotkey is pressed
        suppress: Whether to suppress the hotkey from normal processing
        
    Returns:
        bool: True if registration was successful, False otherwise
    """
    warnings.warn(
        "register_hotkey() is deprecated. Use HotkeyManager.register_hotkey() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_hotkey_manager().register_hotkey(hotkey_id, sequence, callback, suppress)

def unregister_hotkey(hotkey_id: str) -> bool:
    """
    Unregister a global hotkey (compatibility function).
    
    Args:
        hotkey_id: The ID of the hotkey to unregister
        
    Returns:
        bool: True if unregistration was successful, False otherwise
    """
    warnings.warn(
        "unregister_hotkey() is deprecated. Use HotkeyManager.unregister_hotkey() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_hotkey_manager().unregister_hotkey(hotkey_id)

def unregister_all_hotkeys() -> None:
    """Unregister all registered hotkeys (compatibility function)."""
    warnings.warn(
        "unregister_all_hotkeys() is deprecated. Use HotkeyManager.unregister_all_hotkeys() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    get_hotkey_manager().unregister_all_hotkeys()

def is_hotkey_registered(hotkey_id: str) -> bool:
    """
    Check if a hotkey is currently registered (compatibility function).
    
    Args:
        hotkey_id: The ID of the hotkey to check
        
    Returns:
        bool: True if the hotkey is registered, False otherwise
    """
    return get_hotkey_manager().is_hotkey_registered(hotkey_id)

def get_registered_hotkeys() -> Dict[str, str]:
    """
    Get all registered hotkeys (compatibility function).
    
    Returns:
        Dict[str, str]: Mapping of hotkey IDs to their sequences
    """
    return get_hotkey_manager().get_registered_hotkeys()

def cleanup() -> None:
    """Clean up resources and unregister all hotkeys (compatibility function)."""
    warnings.warn(
        "cleanup() is deprecated. Use HotkeyManager.cleanup() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    get_hotkey_manager().cleanup()

# For backward compatibility
HotkeyManager = HotkeyManager
