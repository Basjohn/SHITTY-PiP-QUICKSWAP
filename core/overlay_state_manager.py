"""
Overlay State Manager - Manages hide/show state for all overlay types.

Handles state capture, persistence, and restoration for both single overlay
and docking modes. Provides a unified interface for the Hide/Show All Overlays feature.
"""
from __future__ import annotations
from typing import Optional

from core.logging import get_logger
from core.threading import get_thread_manager
from utils.resource_manager import get_resource_manager, ResourceType


class OverlayStateManager:
    """Manages hide/show state for all overlay types."""
    
    def __init__(self):
        self._hidden = False
        self._saved_state: Optional[dict] = None
        self._logger = get_logger(__name__)
        self._thread_manager = get_thread_manager()
        
        # Register with ResourceManager for cleanup
        rm = get_resource_manager()
        rm.register(self, "OverlayStateManager", ResourceType.CUSTOM)
    
    def are_overlays_hidden(self) -> bool:
        """Check if overlays are currently hidden.
        
        Returns:
            True if overlays are hidden, False otherwise
        """
        return self._hidden
    
    def hide_all_overlays(self) -> bool:
        """Hide all overlays and save their state.
        
        Returns:
            True if overlays were successfully hidden, False otherwise
        """
        # Must run on main thread for Qt operations
        return self._thread_manager.run_in_main_thread(self._hide_all_overlays_impl)
    
    def _hide_all_overlays_impl(self) -> bool:
        """Implementation of hide_all_overlays - runs on main thread."""
        try:
            # Detect current mode
            mode = self._detect_current_mode()
            
            if not mode:
                self._logger.warning("No active overlays to hide")
                return False
            
            if mode == "single":
                success = self._hide_single_overlay()
            elif mode == "docking":
                success = self._hide_docking_system()
            else:
                self._logger.error(f"Unknown mode: {mode}")
                return False
            
            if success:
                self._saved_state = {"mode": mode}  # Just track mode, not full state
                self._hidden = True
                self._logger.info(f"[HIDE_SHOW] Hid all overlays (mode={mode})")
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to hide overlays: {e}", exc_info=True)
            return False
    
    def show_all_overlays(self) -> bool:
        """Show all overlays from saved state.
        
        Returns:
            True if overlays were successfully shown, False otherwise
        """
        # Must run on main thread for Qt operations
        return self._thread_manager.run_in_main_thread(self._show_all_overlays_impl)
    
    def _show_all_overlays_impl(self) -> bool:
        """Implementation of show_all_overlays - runs on main thread."""
        try:
            if not self._hidden or not self._saved_state:
                self._logger.warning("No saved state to restore")
                return False
            
            mode = self._saved_state.get("mode")
            
            if mode == "single":
                success = self._show_single_overlay()
            elif mode == "docking":
                success = self._show_docking_system()
            else:
                self._logger.error(f"Unknown saved mode: {mode}")
                return False
            
            if success:
                self._hidden = False
                self._saved_state = None
                self._logger.info(f"[HIDE_SHOW] Showed all overlays (mode={mode})")
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Failed to show overlays: {e}", exc_info=True)
            return False
    
    # --- Private Methods: Detection ---
    
    def _detect_current_mode(self) -> Optional[str]:
        """Detect if single or docking mode is active.
        
        Returns:
            "single", "docking", or None
        """
        try:
            from utils.resource_manager import find_resource_by_description
            
            # Check for docking manager first
            # Check if overlays exist (visible or hidden), not just if system is active
            try:
                docking = find_resource_by_description("DockingOverlayManager")
                if docking:
                    # Check if overlays exist by checking _main_overlay attribute
                    if hasattr(docking, '_main_overlay') and docking._main_overlay is not None:
                        self._logger.debug("Detected docking mode (overlays exist)")
                        return "docking"
            except Exception as e:
                self._logger.debug(f"Docking detection failed: {e}")
            
            # Check for overlay manager
            try:
                # Correct description per overlay manager registration
                overlay_mgr = find_resource_by_description("OverlayManager singleton")
                if overlay_mgr and hasattr(overlay_mgr, 'get_all_overlays'):
                    overlays = overlay_mgr.get_all_overlays()
                    if overlays:
                        self._logger.debug("Detected single overlay mode")
                        return "single"
                # Fallback: instantiate OverlayManager singleton and query overlays directly
                try:
                    from core.graphics.overlay_manager import OverlayManager as _OM
                    om = _OM()
                    if hasattr(om, 'get_all_overlays'):
                        overlays = om.get_all_overlays()
                        if overlays:
                            self._logger.debug("Detected single overlay mode via fallback")
                            return "single"
                except Exception as _e:
                    self._logger.debug(f"Single overlay detection fallback failed: {_e}")
            except Exception as e:
                self._logger.debug(f"Single overlay detection failed: {e}")
            
        except Exception as e:
            self._logger.debug(f"Mode detection failed: {e}")
        
        return None
    
    # --- Private Methods: Hide/Show (Visibility Toggle) ---
    
    def _hide_single_overlay(self) -> bool:
        """Hide single overlay without destroying it.
        
        Overlay remains alive but invisible. All state preserved.
        
        Returns:
            True if successful
        """
        try:
            from utils.resource_manager import find_resource_by_description
            
            overlay_mgr = find_resource_by_description("OverlayManager")
            
            if not overlay_mgr:
                return False
            
            # Hide all overlays
            if hasattr(overlay_mgr, 'get_all_overlays'):
                overlays = overlay_mgr.get_all_overlays()
                for overlay in overlays:
                    try:
                        if hasattr(overlay, 'hide'):
                            overlay.hide()
                        self._logger.debug("[HIDE_SHOW] Hid single overlay")
                    except Exception as e:
                        self._logger.warning(f"Failed to hide overlay: {e}")
            
            self._logger.info("[HIDE_SHOW] Hid single overlay (kept alive)")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to hide single overlay: {e}")
            return False
    
    def _hide_docking_system(self) -> bool:
        """Hide docking system without destroying it.
        
        Overlays remain alive but invisible. All state preserved.
        
        Returns:
            True if successful
        """
        try:
            from utils.resource_manager import find_resource_by_description
            
            docking = find_resource_by_description("DockingOverlayManager")
            if not docking:
                return False
            
            # Hide main overlay
            if hasattr(docking, '_main_overlay') and docking._main_overlay:
                try:
                    if hasattr(docking._main_overlay, '_dwm_overlay') and docking._main_overlay._dwm_overlay:
                        if hasattr(docking._main_overlay._dwm_overlay, '_host'):
                            docking._main_overlay._dwm_overlay._host.hide()
                    self._logger.debug("[HIDE_SHOW] Hid main overlay")
                except Exception as e:
                    self._logger.warning(f"Failed to hide main overlay: {e}")
            
            # Hide secondary overlays
            if hasattr(docking, '_secondary_overlays'):
                for i, overlay in enumerate(docking._secondary_overlays):
                    try:
                        if overlay and hasattr(overlay, '_dwm_overlay') and overlay._dwm_overlay:
                            if hasattr(overlay._dwm_overlay, '_host'):
                                overlay._dwm_overlay._host.hide()
                        self._logger.debug(f"[HIDE_SHOW] Hid secondary_{i}")
                    except Exception as e:
                        self._logger.warning(f"Failed to hide secondary_{i}: {e}")
            
            self._logger.info("[HIDE_SHOW] Hid docking overlays (kept alive)")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to hide docking system: {e}")
            return False
    
    # --- Private Methods: Restoration ---
    
    def _show_single_overlay(self) -> bool:
        """Show previously hidden single overlay.
        
        Overlay was never destroyed, so all state is preserved.
        
        Returns:
            True if successful
        """
        try:
            from utils.resource_manager import find_resource_by_description
            
            overlay_mgr = find_resource_by_description("OverlayManager")
            if not overlay_mgr:
                self._logger.error("OverlayManager not found")
                return False
            
            # Show all overlays
            if hasattr(overlay_mgr, 'get_all_overlays'):
                overlays = overlay_mgr.get_all_overlays()
                for overlay in overlays:
                    try:
                        if hasattr(overlay, 'show'):
                            overlay.show()
                        self._logger.debug("[HIDE_SHOW] Showed single overlay")
                    except Exception as e:
                        self._logger.warning(f"Failed to show overlay: {e}")
            
            self._logger.info("[HIDE_SHOW] Showed single overlay (from hidden state)")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to show single overlay: {e}", exc_info=True)
            return False
    
    def _show_docking_system(self) -> bool:
        """Show previously hidden docking overlays.
        
        Overlays were never destroyed, so all state is preserved.
        
        Returns:
            True if successful
        """
        try:
            from utils.resource_manager import find_resource_by_description
            
            docking = find_resource_by_description("DockingOverlayManager")
            if not docking:
                self._logger.error("DockingOverlayManager not found")
                return False
            
            # Show main overlay
            if hasattr(docking, '_main_overlay') and docking._main_overlay:
                try:
                    if hasattr(docking._main_overlay, '_dwm_overlay') and docking._main_overlay._dwm_overlay:
                        if hasattr(docking._main_overlay._dwm_overlay, '_host'):
                            docking._main_overlay._dwm_overlay._host.show()
                    self._logger.debug("[HIDE_SHOW] Showed main overlay")
                except Exception as e:
                    self._logger.warning(f"Failed to show main overlay: {e}")
            
            # Show secondary overlays
            if hasattr(docking, '_secondary_overlays'):
                for i, overlay in enumerate(docking._secondary_overlays):
                    try:
                        if overlay and hasattr(overlay, '_dwm_overlay') and overlay._dwm_overlay:
                            if hasattr(overlay._dwm_overlay, '_host'):
                                overlay._dwm_overlay._host.show()
                        self._logger.debug(f"[HIDE_SHOW] Showed secondary_{i}")
                    except Exception as e:
                        self._logger.warning(f"Failed to show secondary_{i}: {e}")
            
            self._logger.info("[HIDE_SHOW] Showed docking overlays (from hidden state)")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to show docking system: {e}", exc_info=True)
            return False
    
    def cleanup(self, *args) -> None:
        """Cleanup resources before shutdown.
        
        Args:
            *args: Ignored (ResourceManager may pass resource object)
        """
        try:
            # Clear saved state
            self._saved_state = None
            self._hidden = False
            self._logger.info("OverlayStateManager cleanup complete")
        except Exception as e:
            self._logger.debug(f"Cleanup error: {e}")


# Singleton instance
_overlay_state_manager: Optional[OverlayStateManager] = None


def get_overlay_state_manager() -> OverlayStateManager:
    """Get singleton overlay state manager.
    
    Returns:
        OverlayStateManager instance
    """
    global _overlay_state_manager
    if _overlay_state_manager is None:
        _overlay_state_manager = OverlayStateManager()
    return _overlay_state_manager
