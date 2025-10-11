"""
Hide/Show All Overlays Controller

Provides Ctrl+Shift+H hotkey to toggle visibility of all overlays.
Works with both single overlay and docking modes.
"""
from typing import Optional

from core.logging import get_logger
from core.hotkeys.manager import HotkeyManager
from core.overlay_state_manager import get_overlay_state_manager
from core.settings import get_settings_manager
from utils.resource_manager import get_resource_manager, ResourceType


class HideShowController:
    """Controller for hide/show all overlays hotkey."""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._hotkey_manager = HotkeyManager()  # Singleton via __new__
        self._state_manager = get_overlay_state_manager()
        self._settings_manager = get_settings_manager()
        self._hotkey_id = "hide_show_overlays"
        
        # Register with ResourceManager (correct signature and cleanup handler)
        rm = get_resource_manager()
        try:
            rm.register(
                self,
                ResourceType.CUSTOM,
                "HideShowController",
                cleanup_handler=lambda obj: obj.cleanup(),
                tags={"hotkeys", "controller"},
                cleanup_priority=9,  # Stop before HotkeyManager to avoid races
            )
        except Exception:
            # Best-effort registration; controller will still function
            pass
        
        # Register hotkey from settings (only if enabled)
        self._register_hotkey()
        
        self._logger.info(f"HideShowController initialized with {self._get_current_sequence()} hotkey")
    
    def _get_current_sequence(self) -> str:
        """Get current hotkey sequence from settings."""
        return self._settings_manager.get("hotkeys.hide_show_overlays", "ctrl+shift+h")
    
    def _register_hotkey(self) -> None:
        """Register hide/show hotkey from settings (only if enabled)."""
        try:
            # Check if hotkey is enabled
            enabled = self._settings_manager.get("hotkeys.hide_show_enabled", False)
            if not enabled:
                self._logger.debug("Hide/Show hotkey disabled in settings, not registering")
                return
            
            sequence = self._get_current_sequence()
            # No suppression (like other multipress hotkeys)
            success = self._hotkey_manager.register_hotkey(
                hotkey_id=self._hotkey_id,
                callback=self._on_hotkey_pressed,
                sequence=sequence,
                suppress=False,  # No suppression like other multipress hotkeys
                global_hotkey=True
            )
            
            if success:
                self._logger.info(f"Hide/Show hotkey registered: {sequence}")
            else:
                self._logger.warning(f"Failed to register hide/show hotkey: {sequence}")
                
        except Exception as e:
            self._logger.error(f"Failed to register hide/show hotkey: {e}", exc_info=True)
    
    def update_hotkeys(self) -> None:
        """Update hotkey registration based on current settings (enable/disable + sequence)."""
        try:
            # Unregister existing hotkey first
            self._hotkey_manager.unregister_hotkey(self._hotkey_id)
            self._logger.debug("Unregistered old hide/show hotkey")
            
            # Re-register based on enabled setting
            self._register_hotkey()
                
        except Exception as e:
            self._logger.error(f"Failed to update hide/show hotkey: {e}", exc_info=True)
    
    def update_hotkey(self, new_sequence: str) -> None:
        """Update hotkey sequence when settings change.
        
        Args:
            new_sequence: New hotkey sequence string
        """
        try:
            # Unregister old hotkey
            self._hotkey_manager.unregister_hotkey(self._hotkey_id)
            self._logger.debug("Unregistered old hide/show hotkey")
            
            # Check if enabled before re-registering
            enabled = self._settings_manager.get("hotkeys.hide_show_enabled", False)
            if not enabled:
                self._logger.debug("Hide/Show hotkey disabled, not re-registering")
                return
            
            # Register new hotkey
            success = self._hotkey_manager.register_hotkey(
                hotkey_id=self._hotkey_id,
                callback=self._on_hotkey_pressed,
                sequence=new_sequence,
                suppress=False,
                global_hotkey=True
            )
            
            if success:
                self._logger.info(f"Updated hide/show hotkey to: {new_sequence}")
            else:
                self._logger.warning(f"Failed to register new hide/show hotkey: {new_sequence}")
                
        except Exception as e:
            self._logger.error(f"Failed to update hide/show hotkey: {e}", exc_info=True)
    
    def _on_hotkey_pressed(self) -> None:
        """Handle Ctrl+Shift+H press - toggle overlay visibility."""
        try:
            if self._state_manager.are_overlays_hidden():
                # Show overlays
                success = self._state_manager.show_all_overlays()
                if success:
                    self._logger.info("Shown all overlays via Ctrl+Shift+H")
                else:
                    self._logger.warning("Failed to show overlays")
            else:
                # Hide overlays
                success = self._state_manager.hide_all_overlays()
                if success:
                    self._logger.info("Hidden all overlays via Ctrl+Shift+H")
                else:
                    self._logger.warning("Failed to hide overlays")
                    
        except Exception as e:
            self._logger.error(f"Error toggling overlay visibility: {e}", exc_info=True)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self._hotkey_manager.unregister_hotkey(self._hotkey_id)
            self._logger.info("HideShowController cleanup complete")
        except Exception as e:
            self._logger.debug(f"Cleanup error: {e}")


# Singleton instance
_hide_show_controller: Optional[HideShowController] = None


def get_hide_show_controller() -> HideShowController:
    """Get singleton hide/show controller.
    
    Returns:
        HideShowController instance
    """
    global _hide_show_controller
    if _hide_show_controller is None:
        _hide_show_controller = HideShowController()
    return _hide_show_controller
