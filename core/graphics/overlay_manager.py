"""
Centralized Overlay Manager with MRU and Z-Order Management.

This module provides an enhanced version of the OverlayManager class with support for:
- Most Recently Used (MRU) tracking for quick switching
- Overlay locking to prevent unwanted overlay changes
- Integrated z-order management via ZOrderManager
- Centralized lifecycle management for all overlay types
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Any, Callable

from PySide6.QtCore import QRect

from core.logging import get_logger
from .overlay import Overlay as OverlayBase
from .types import OverlayConfig, OverlayType
from .backend_manager import BackendManager, BackendType

logger = get_logger(__name__)

class OverlayManager:
    """Manages the lifecycle of overlay instances with MRU and locking support."""
    
    # Maximum number of items to keep in MRU list
    MAX_MRU_ITEMS = 10
    
    _instance: Optional['OverlayManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the overlay manager with MRU and locking support."""
        if self._initialized:
            return
            
        self._overlays: Dict[str, OverlayBase] = {}
        self._backend_manager = BackendManager()
        self._logger = get_logger(__name__)
        self._initialized = True
        
        # App instance provider for DWM overlays
        self._app_instance_provider: Optional[Callable[[], Any]] = None

        # Track the single active overlay id (enforce one-at-a-time semantics)
        self._active_overlay_id: Optional[str] = None
        
        # MRU tracking
        self._mru_overlays: List[str] = []
        self._overlays_locked: bool = False  # Initialize to False - overlays start unlocked
        self._lock = threading.RLock()
        
        # Debug overlay lock state
        self._logger.debug(f"OverlayManager initialized with overlays_locked={self._overlays_locked}")
        
        # Track focus timestamps for MRU
        self._focus_timestamps: Dict[str, float] = {}

        # Set via set_app_instance_provider() by the composition root (ApplicationCore)
        self._app_instance_provider: Optional[Callable[[], object]] = None
        
        
        # Z-order management handled by centralized ZOrderManager
        
        # Log available backends
        available = self._backend_manager.get_available_backends()
        if available:
            self._logger.info("Available overlay backends: %s", 
                            ", ".join(b.name for b in available))
        else:
            self._logger.warning("No overlay backends available")

    def set_app_instance_provider(self, provider: Callable[[], object]) -> None:
        """Set the provider used to inject the application instance into overlays.
        
        Args:
            provider: A zero-arg callable returning the application instance.
        """
        with self._lock:
            self._app_instance_provider = provider
            self._logger.debug("OverlayManager app_instance provider set")
    
    # MRU and Locking Methods
    def update_mru(self, overlay_id: str) -> None:
        """Update the MRU list with the given overlay ID.
        
        Args:
            overlay_id: The ID of the overlay to update in MRU
        """
        with self._lock:
            if self._overlays_locked or overlay_id not in self._overlays:
                return
                
            # Update timestamp
            self._focus_timestamps[overlay_id] = time.time()
            
            # Update MRU list
            if overlay_id in self._mru_overlays:
                self._mru_overlays.remove(overlay_id)
            self._mru_overlays.insert(0, overlay_id)
            
            # Trim MRU list if needed
            if len(self._mru_overlays) > self.MAX_MRU_ITEMS:
                removed = self._mru_overlays.pop()
                self._focus_timestamps.pop(removed, None)
    
    def is_overlay_locked(self) -> bool:
        """Check if overlays are currently locked.
        
        Returns:
            bool: True if overlays are locked, False otherwise
        """
        return self._overlays_locked
    
    def set_overlay_lock(self, locked: bool = True) -> None:
        """Set the overlay lock state.
        
        Args:
            locked: Whether to lock the overlays
        """
        with self._lock:
            self._overlays_locked = locked
            self._logger.info("Overlay lock %s", "enabled" if locked else "disabled")
    
    def get_mru_overlays(self, limit: int = 0) -> List[OverlayBase]:
        """Get overlays sorted by most recently used.
        
        Args:
            limit: Maximum number of overlays to return (0 for all)
            
        Returns:
            List of overlays sorted by most recently used
        """
        with self._lock:
            overlays = [self._overlays[oid] for oid in self._mru_overlays 
                      if oid in self._overlays]
            return overlays[:limit] if limit > 0 else overlays
    
    # Core Overlay Management Methods
    def create_overlay(self, 
                     rect: Optional[QRect] = None,
                     overlay_type: Optional[OverlayType] = OverlayType.WINDOW,
                     title: str = "Overlay",
                     opacity: float = 1.0,
                     backend: BackendType = BackendType.AUTO,
                     properties: Optional[Dict[str, Any]] = None,
                     config: Optional[Any] = None) -> Optional[str]:
        """Create a new overlay with the given parameters.
        
        Backward compatibility:
        - Accepts optional 'config' kwarg (ignored) to support older tests calling
          create_overlay(overlay_type=..., config=...).
        - If rect is None, uses a small default geometry as a placeholder.
        - If overlay_type is None, defaults to OverlayType.WINDOW.
        
        Args:
            rect: Position and size of the overlay. If None, a default QRect is used.
            overlay_type: Type of overlay to create. If None, defaults to WINDOW.
            title: Window title for the overlay
            opacity: Initial opacity (0.0 to 1.0)
            backend: Preferred backend to use
            properties: Backend-specific properties
            config: Ignored; accepted for backward compatibility only
            
        Returns:
            Overlay ID if successful, None otherwise
        """
        with self._lock:
            if self._overlays_locked:
                self._logger.warning("Cannot create overlay: overlays are locked")
                return None

            # Check if we can reuse the existing overlay when swapping windows
            reuse_existing = False
            existing_overlay = None
            
            if self._active_overlay_id is not None:
                existing_overlay = self._overlays.get(self._active_overlay_id)
                
                # If current overlay is flagged for recreation, tear it down instead of reusing
                if existing_overlay is not None and getattr(existing_overlay, "_needs_recreate", False):
                    self._logger.warning(
                        "Active overlay %s marked _needs_recreate; tearing down for fresh creation",
                        self._active_overlay_id,
                    )
                    self._remove_overlay_internal(self._active_overlay_id)
                    self._active_overlay_id = None
                    existing_overlay = None
                
                # Only reuse for window overlays with the same backend type
                if (existing_overlay and 
                    overlay_type == OverlayType.WINDOW and 
                    existing_overlay.get_config().overlay_type == OverlayType.WINDOW and
                    backend == existing_overlay.get_backend_type()):
                    
                    # Keep the existing overlay and just update its properties
                    self._logger.info("Reusing existing overlay %s for window swap", self._active_overlay_id)
                    reuse_existing = True
                else:
                    # Different overlay type or backend, destroy the existing one
                    if self._active_overlay_id is not None and existing_overlay is not None:
                        self._logger.info("Replacing active overlay %s with a new one", self._active_overlay_id)
                        self._remove_overlay_internal(self._active_overlay_id)
                        self._active_overlay_id = None

            # Normalize defaults for optional args
            if rect is None:
                rect = QRect(0, 0, 1, 1)
            if overlay_type is None:
                overlay_type = OverlayType.WINDOW

            # Create overlay configuration
            config_obj = OverlayConfig(
                overlay_type=overlay_type,
                position=rect.topLeft(),
                size=rect.size(),
                opacity=opacity,
                title=title,
                properties=properties or {}
            )
            
            if reuse_existing and existing_overlay:
                # Reuse the existing overlay - just update its properties
                overlay_id = self._active_overlay_id
                overlay = existing_overlay
                
                # Update the overlay with new properties
                try:
                    # Update window/source properties without recreating
                    if 'hwnd' in config_obj.properties and hasattr(overlay, 'update_source'):
                        self._logger.info("Updating source for overlay %s", overlay_id)
                        overlay.update_source(config_obj.properties['hwnd'])
                    
                    # Update position and size if changed
                    current_config = overlay.get_config()
                    if (current_config.position != config_obj.position or 
                        current_config.size != config_obj.size):
                        self._logger.info("Updating geometry for overlay %s", overlay_id)
                        overlay.set_geometry(rect)
                    
                    # Update opacity if changed
                    if current_config.opacity != config_obj.opacity:
                        self._logger.info("Updating opacity for overlay %s", overlay_id)
                        overlay.set_opacity(config_obj.opacity)
                        
                    # Update title if changed
                    if current_config.title != config_obj.title:
                        self._logger.info("Updating title for overlay %s", overlay_id)
                        overlay.set_title(config_obj.title)
                        
                    # Update MRU
                    self.update_mru(overlay_id)
                    
                    self._logger.info("Reused overlay %s with updated properties", overlay_id)
                    return overlay_id
                except Exception as e:
                    # If updating fails, fall back to creating a new overlay
                    self._logger.error("Failed to update overlay %s: %s", overlay_id, str(e))
                    self._remove_overlay_internal(overlay_id)
                    self._active_overlay_id = None
                    reuse_existing = False
            
            # Create a new overlay if not reusing
            overlay = self._backend_manager.create_overlay(config_obj, backend)
            if not overlay:
                self._logger.error("Failed to create overlay")
                return None
            
            # Inject the application instance into DWM overlays before initialization.
            # Enables centralized OverlayContextMenu population and strict wiring.
            try:
                # Local import to avoid circular dependency at module import time
                from .backends.dwm.integrated_dwm_backend import IntegratedDWMOverlay as _DWMOverlay

                if isinstance(overlay, _DWMOverlay):
                    # Require overlay attributes for safe switching
                    if not hasattr(overlay, "app_instance"):
                        self._logger.error("DWMOverlay lacks 'app_instance' attribute; aborting overlay creation")
                        return None
                    if not hasattr(overlay, "_handle_swap_window"):
                        self._logger.error("DWMOverlay missing '_handle_swap_window'; aborting overlay creation")
                        return None

                    # Obtain app instance from injected provider (strict, no fallback)
                    if self._app_instance_provider is None:
                        self._logger.error("No app_instance provider set; cannot create DWM overlay")
                        return None
                    try:
                        app_instance = self._app_instance_provider()
                    except Exception as prov_e:
                        self._logger.error("app_instance provider failed: %s", prov_e)
                        return None

                    overlay.app_instance = app_instance
                    self._logger.debug("Injected app_instance into DWMOverlay for context menu window switching")
            except Exception as e:
                # Enforce strict no-fallback policy
                self._logger.error("Failed to prepare DWMOverlay injection: %s", str(e))
                return None

            # Ensure window-mode features are lazily initialized when creating window overlays
            try:
                if config_obj.overlay_type == OverlayType.WINDOW and self._app_instance_provider is not None:
                    app_instance = self._app_instance_provider()
                    if hasattr(app_instance, "ensure_window_mode_features"):
                        app_instance.ensure_window_mode_features()
                        self._logger.debug("Ensured window-mode features are initialized for window overlay")
            except Exception as e:
                # Do not abort overlay creation for lazy-init errors; log and continue
                self._logger.debug(f"ensure_window_mode_features failed: {e}")

            # Generate a unique ID for the overlay
            overlay_id = overlay.id
            self._overlays[overlay_id] = overlay
            
            # Initialize the overlay
            if not overlay.initialize():
                self._logger.error("Failed to initialize overlay %s", overlay_id)
                self._overlays.pop(overlay_id, None)
                return None
            
            # Add to MRU
            self.update_mru(overlay_id)
            
            # Show the overlay first to ensure window handle is stable
            overlay.show()
            
            # Register overlay with ZOrderManager for centralized z-order enforcement
            try:
                main_widget = getattr(overlay, "_host", None)
                if main_widget is None:
                    self._logger.error(f"Cannot register overlay {overlay_id}: missing host widget")
                    try:
                        overlay.close()
                    except Exception:
                        pass
                    self._overlays.pop(overlay_id, None)
                    return None
                
                # Defer registration to ensure window handle is stable
                from core.threading import ThreadManager
                
                def _register_after_stable():
                    if main_widget.isVisible() and main_widget.winId():
                        # Register with unified z-order manager
                        from utils.resource_manager import get_resource_manager
                        rm = get_resource_manager()
                        rm.register_overlay(overlay_id, main_widget)
                
                # Register after a short delay to ensure window handle stability
                ThreadManager.single_shot(50, _register_after_stable)
                
            except Exception as e:
                self._logger.error(f"Failed to setup z-order registration for {overlay_id}: {e}")
            
            # Mark as active
            self._active_overlay_id = overlay_id
            
            self._logger.info("Created overlay %s with backend %s", overlay_id, overlay.__class__.__name__)
            return overlay_id
    
    def get_overlay(self, overlay_id: str) -> Optional[OverlayBase]:
        """Get an overlay by its ID and update MRU.
        
        Args:
            overlay_id: The ID of the overlay to retrieve
            
        Returns:
            The overlay, or None if not found
        """
        with self._lock:
            overlay = self._overlays.get(overlay_id)
            if overlay:
                self.update_mru(overlay_id)
            return overlay

    def get_active_overlay(self) -> Optional[OverlayBase]:
        """Return the currently active overlay instance, if any.

        Thread-safe accessor used by modules needing to interact with the
        active overlay (e.g., UI feedback on input decisions).
        """
        with self._lock:
            if self._active_overlay_id is None:
                return None
            return self._overlays.get(self._active_overlay_id)
    
    def remove_overlay(self, overlay_id: str) -> bool:
        """Remove and clean up an overlay.
        
        Args:
            overlay_id: The ID of the overlay to remove
            
        Returns:
            True if the overlay was removed, False otherwise
        """
        with self._lock:
            if self._overlays_locked:
                self._logger.warning("Cannot remove overlay: overlays are locked")
                return False
                
            return self._remove_overlay_internal(overlay_id)

    def _remove_overlay_internal(self, overlay_id: str) -> bool:
        """Internal helper to remove and clean up an overlay without acquiring the lock twice."""
        overlay = self._overlays.pop(overlay_id, None)
        if overlay is None:
            return False

        # Remove from MRU and timestamps
        if overlay_id in self._mru_overlays:
            self._mru_overlays.remove(overlay_id)
        self._focus_timestamps.pop(overlay_id, None)

        try:

            # Ensure it is hidden and closed/cleaned up
            try:
                overlay.hide()
            except Exception:
                pass
            try:
                overlay.close()
            except Exception:
                pass

            # Unregister from unified z-order management
            try:
                from utils.resource_manager import get_resource_manager
                get_resource_manager().unregister_overlay(overlay_id)
            except Exception:
                pass
            
            # Clear active id if this was active
            if self._active_overlay_id == overlay_id:
                self._active_overlay_id = None
                
            return True
        except Exception as e:
            self._logger.exception("Error removing overlay %s: %s", overlay_id, str(e))
            return False

    def close_active(self) -> None:
        """Close and remove the currently active overlay, if any."""
        with self._lock:
            if self._active_overlay_id is not None:
                self._remove_overlay_internal(self._active_overlay_id)
                self._active_overlay_id = None

    def set_opacity(self, value: float) -> bool:
        """Route opacity updates to the active overlay.

        Returns True if an active overlay exists and was updated.
        """
        with self._lock:
            if self._active_overlay_id is None:
                return False
            overlay = self._overlays.get(self._active_overlay_id)
            if overlay is None:
                return False
            try:
                overlay.update_config(opacity=value)
                return True
            except Exception as e:
                self._logger.exception("Failed to set opacity on active overlay: %s", str(e))
                return False

    # Convenience creation helpers (window/monitor paths)
    def create_window_overlay(self, rect: QRect, title: str = "Overlay", opacity: float = 1.0,
                               backend: BackendType = BackendType.AUTO,
                               hwnd: Optional[int] = None) -> Optional[str]:
        # DWM window overlays require a valid source hwnd; enforce no-fallback policy
        props: Dict[str, Any] = {}
        if backend == BackendType.DWM:
            if not hwnd or int(hwnd) == 0:
                self._logger.error("DWM window overlay creation requires a valid hwnd; none provided")
                return None
            props["hwnd"] = int(hwnd)
        return self.create_overlay(rect, overlay_type=OverlayType.WINDOW, title=title, opacity=opacity, backend=backend, properties=props)

    def create_monitor_overlay(self, rect: QRect, title: str = "Overlay", opacity: float = 1.0,
                                backend: BackendType = BackendType.AUTO, monitor_target: Optional[Dict] = None) -> Optional[str]:
        props = {}
        if monitor_target:
            props["monitor_target"] = monitor_target
        return self.create_overlay(rect, overlay_type=OverlayType.MONITOR, title=title, opacity=opacity, backend=backend, properties=props)
    
    def update_overlay(self, overlay_id: str, **kwargs) -> bool:
        """Update an overlay's configuration.
        
        Args:
            overlay_id: The ID of the overlay to update
            **kwargs: Configuration values to update
            
        Returns:
            True if the update was successful, False otherwise
        """
        with self._lock:
            if self._overlays_locked:
                self._logger.warning("Cannot update overlay: overlays are locked")
                return False
                
            overlay = self._overlays.get(overlay_id)
            if overlay is None:
                return False
                
            try:
                # Update position if provided
                if 'position' in kwargs:
                    overlay.set_position(kwargs['position'])
                    
                # Update size if provided
                if 'size' in kwargs:
                    overlay.set_size(kwargs['size'])
                    
                # Update opacity if provided
                if 'opacity' in kwargs:
                    overlay.set_opacity(kwargs['opacity'])
                    
                # Update visibility if provided
                if 'visible' in kwargs:
                    if kwargs['visible']:
                        overlay.show()
                    else:
                        overlay.hide()
                
                # Update MRU if any changes were made
                if kwargs:
                    self.update_mru(overlay_id)
                
                self._logger.debug("Updated overlay %s: %s", overlay_id, 
                                 ", ".join(f"{k}={v}" for k, v in kwargs.items()))
                return True
                
            except Exception as e:
                self._logger.exception("Failed to update overlay %s: %s", 
                                     overlay_id, str(e))
                return False
    
    def get_all_overlays(self) -> List[OverlayBase]:
        """Get all managed overlays.
        
        Returns:
            A list of all managed overlays
        """
        with self._lock:
            return list(self._overlays.values())
    
    def enforce_z_order(self, overlay_id: str) -> bool:
        """Enforce z-order for the specified overlay using the centralized ZOrderManager.
        
        Args:
            overlay_id: ID of the overlay to enforce z-order for
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            from utils.resource_manager import get_resource_manager
            rm = get_resource_manager()
            return rm.enforce_z_order(overlay_id)
    
    def schedule_z_order_enforcement(self, overlay_id: str, delay_ms: int = 0) -> None:
        """Schedule z-order enforcement with optional delay using ThreadManager.
        
        Args:
            overlay_id: ID of the overlay to enforce z-order for
            delay_ms: Delay in milliseconds before enforcement
        """
        try:
            from core.threading import ThreadManager
            from utils.resource_manager import get_resource_manager
            rm = get_resource_manager()
            # Use ThreadManager for scheduling (centralized timer policy)
            ThreadManager.single_shot(delay_ms, lambda: rm.enforce_z_order(overlay_id))
        except Exception as e:
            self._logger.debug(f"Z-order scheduling for {overlay_id}: {e}")
    
    def clear_all(self) -> None:
        """Remove and clean up all overlays if not locked."""
        with self._lock:
            if self._overlays_locked:
                self._logger.warning("Cannot clear overlays: overlays are locked")
                return
                
            for overlay_id in list(self._overlays.keys()):
                self.remove_overlay(overlay_id)
            
            # Clear MRU and timestamps
            self._mru_overlays.clear()
            self._focus_timestamps.clear()
            
            self._logger.info("Cleared all overlays")
