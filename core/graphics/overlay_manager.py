"""
Centralized Overlay Manager with MRU and Z-Order Management.

This module provides an enhanced version of the OverlayManager class with support for:
- Most Recently Used (MRU) tracking for quick switching
- Overlay locking to prevent unwanted overlay changes
- Integrated z-order management via ZOrderManager
"""
from __future__ import annotations

# Standard library imports
import time
from typing import Any, Callable, Dict, List, Optional

# Third-party imports
from PySide6.QtCore import QRect

# Local imports - Core
from core.graphics.backend_manager import BackendManager
from core.graphics.backends import BackendType
from core.graphics.overlay import Overlay as OverlayBase
from core.logging import get_logger
from core.threading import ThreadManager

# Local imports - Relative
from .types import OverlayConfig, OverlayType

logger = get_logger(__name__)

class OverlayManager:
    """Manages the lifecycle of overlay instances with MRU and locking support."""
    
    # Maximum number of items to keep in MRU list
    MAX_MRU_ITEMS = 12  # Expanded for docking mode support
    
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
        # Lock-free implementation - all mutations happen on UI thread via ThreadManager
        
        # Debug overlay lock state
        self._logger.debug(f"OverlayManager initialized with overlays_locked={self._overlays_locked}")
        
        # Track focus timestamps for MRU
        self._focus_timestamps: Dict[str, float] = {}

        # Set via set_app_instance_provider() by the composition root (ApplicationCore)
        self._app_instance_provider: Optional[Callable[[], object]] = None
        
        # Closed-window switch manager for handling window closures
        self._closed_window_switch_manager = None
        
        # Register with ResourceManager for deterministic cleanup
        try:
            from utils.resource_manager import get_resource_manager, ResourceType
            self._resource_manager = get_resource_manager()
            self._resource_id = self._resource_manager.register(
                self,
                ResourceType.CUSTOM,
                "OverlayManager singleton",
                cleanup_handler=lambda obj: obj._cleanup()
            )
            self._logger.debug("Registered OverlayManager with ResourceManager")
        except Exception as e:
            self._logger.warning(f"Failed to register with ResourceManager: {e}")
            self._resource_manager = None
            self._resource_id = None
        
        # Z-order management handled by centralized ZOrderManager
        
        # Log available backends
        available = self._backend_manager.get_available_backends()
        if available:
            self._logger.info("Available overlay backends: %s", 
                            ", ".join(b.name for b in available))
        else:
            self._logger.warning("No overlay backends available")
            
        # Initialize closed-window switch manager
        self._initialize_closed_window_switch()
        
        # Apply initial auto-switch setting from configuration
        self._apply_initial_auto_switch_setting()

    def set_app_instance_provider(self, provider: Callable[[], object]) -> None:
        """Set the provider used to inject the application instance into overlays.
        
        Args:
            provider: A zero-arg callable returning the application instance.
        """
        # UI thread operation - no lock needed
        self._app_instance_provider = provider
        self._logger.debug("OverlayManager app_instance provider set")
    
    # MRU and Locking Methods
    def update_mru(self, overlay_id: str) -> None:
        """Update the MRU list with the given overlay ID.
        
        Args:
            overlay_id: The ID of the overlay to update in MRU
        """
        # UI thread operation - no lock needed
        if self._overlays_locked or overlay_id not in self._overlays:
            return
            
        # Update timestamp
        self._focus_timestamps[overlay_id] = time.time()
        
        # Remove from current position if exists
        if overlay_id in self._mru_overlays:
            self._mru_overlays.remove(overlay_id)
        
        # Add to front
        self._mru_overlays.insert(0, overlay_id)
        
        # Trim to max size
        while len(self._mru_overlays) > self.MAX_MRU_ITEMS:
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
        # UI thread operation - no lock needed
        self._overlays_locked = locked
        self._logger.info("Overlay lock %s", "enabled" if locked else "disabled")
        
        # Update focus indicator state for all overlays immediately
        self._update_focus_indicator_lock_state(locked)
    
    def get_mru_overlays(self, limit: int = 0) -> List[OverlayBase]:
        """Get overlays sorted by most recently used.
        
        Args:
            limit: Maximum number of overlays to return (0 for all)
            
        Returns:
            List of overlays sorted by most recently used
        """
        # UI thread operation - no lock needed
        overlays = [self._overlays[oid] for oid in self._mru_overlays 
                  if oid in self._overlays]
        return overlays[:limit] if limit > 0 else overlays

    def get_mru_window_list(self, limit: int = 0) -> List[int]:
        """Get MRU window HWNDs from centralized MRUManager (SINGLE SOURCE OF TRUTH).
        
        Args:
            limit: Maximum number of HWNDs to return (0 = all)
            
        Returns:
            List of window HWNDs sorted by most recently used
            
        Note: This now reads directly from MRUManager instead of overlay tracking.
        MRUManager is updated by FocusTracker and is the authoritative source.
        """
        try:
            from core.switching.mru_manager import get_mru_manager
            mru_mgr = get_mru_manager()
            # Get from centralized MRU (FocusTracker keeps this updated)
            return mru_mgr.get_recent(limit=limit if limit > 0 else None)
        except Exception as e:
            self._logger.warning(f"Failed to get MRU from centralized manager: {e}")
            return []

    def update_mru_from_hwnd_list(self, hwnd_list: List[int]) -> None:
        """Update MRU tracking from a list of HWNDs.
        
        Updates BOTH:
        1. Centralized MRUManager (window HWND MRU)
        2. Local overlay ID MRU (UI interaction tracking)
        
        Args:
            hwnd_list: List of HWNDs in desired MRU order
        """
        # Update centralized MRUManager (SINGLE SOURCE OF TRUTH for window HWNDs)
        try:
            from core.switching.mru_manager import get_mru_manager
            mru_mgr = get_mru_manager()
            for hwnd in hwnd_list:
                if hwnd:
                    mru_mgr.record(hwnd)
        except Exception as e:
            self._logger.warning(f"Failed to update centralized MRU: {e}")
        
        # Update local overlay ID MRU (for UI interaction tracking)
        new_mru_order = []
        for hwnd in hwnd_list:
            for overlay_id, overlay in self._overlays.items():
                if hasattr(overlay, 'get_source_hwnd') and overlay.get_source_hwnd() == hwnd:
                    if overlay_id not in new_mru_order:
                        new_mru_order.append(overlay_id)
                    break
        
        # Add any existing overlays not in the HWND list
        for overlay_id in self._mru_overlays:
            if overlay_id not in new_mru_order:
                new_mru_order.append(overlay_id)
        
        self._mru_overlays = new_mru_order[:self.MAX_MRU_ITEMS]
        self._logger.debug(f"Updated both centralized MRU and overlay MRU: {len(self._mru_overlays)} overlay IDs")
    
    # Core Overlay Management Methods
    def create_overlay(self, 
                     rect: Optional[QRect] = None,
                     overlay_type: Optional[OverlayType] = None,
                     title: str = "",
                     opacity: float = 1.0,
                     backend: Optional[BackendType] = None,
                     properties: Optional[Dict[str, Any]] = None,
                     config: Optional[Any] = None,
                     bypass_lock: bool = False) -> Optional[str]:
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
            bypass_lock: If True, bypass overlay lock for system operations
            
        Returns:
            Overlay ID if successful, None otherwise
        """
        # UI thread operation - no lock needed
        if self._overlays_locked and not bypass_lock:
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
        
        # Inject the application instance into DWM and Monitor overlays before initialization.
        # Enables centralized OverlayContextMenu population and strict wiring.
        try:
            # Local import to avoid circular dependency at module import time
            from .backends.dwm.integrated_dwm_backend import IntegratedDWMOverlay as _DWMOverlay
            from .backends.monitor.monitor_backend import MonitorBackend as _MonitorBackend

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
            
            elif isinstance(overlay, _MonitorBackend):
                # MonitorBackend needs app_instance for context menu monitor switching
                if self._app_instance_provider is not None:
                    try:
                        app_instance = self._app_instance_provider()
                        # Store on backend - will be propagated to host widget after initialization
                        overlay._app_instance_for_host = app_instance
                        self._logger.debug("Stored app_instance for MonitorBackend host widget")
                    except Exception as prov_e:
                        self._logger.warning("app_instance provider failed for MonitorBackend: %s", prov_e)
        except Exception as e:
            # Enforce strict no-fallback policy for DWM, log warning for Monitor
            self._logger.error("Failed to prepare overlay injection: %s", str(e))
            # Check if overlay is DWM type (strict failure) - use duck typing to avoid undefined variable
            if backend == BackendType.DWM:
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
        
        # Register for auto-switch monitoring if DWM overlay with hwnd
        if backend == BackendType.DWM and properties and properties.get("hwnd"):
            self._register_overlay_window(overlay_id, properties["hwnd"])
        
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
                    try:
                        from utils.resource_manager import get_resource_manager
                        rm = get_resource_manager()
                        ok = rm.register_overlay(overlay_id, main_widget)
                        if not ok:
                            logger.debug(f"Z-order registration returned False for {overlay_id}")
                    except Exception as zex:
                        logger.error(f"Z-order registration failed for {overlay_id}: {zex}")
            
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
        # UI thread operation - no lock needed
        overlay = self._overlays.get(overlay_id)
        if overlay:
            self.update_mru(overlay_id)
        return overlay

    def get_active_overlay(self) -> Optional[OverlayBase]:
        """Return the currently active overlay instance, if any.

        Thread-safe accessor used by modules needing to interact with the
        active overlay (e.g., UI feedback on input decisions).
        """
        # UI thread operation - no lock needed
        if self._active_overlay_id is None:
            return None
        return self._overlays.get(self._active_overlay_id)
    
    def remove_overlay(self, overlay_id: str, bypass_lock: bool = False) -> bool:
        """Remove an overlay by ID.
        
        Args:
            overlay_id: The ID of the overlay to remove
            bypass_lock: If True, bypass overlay lock for system operations
            
        Returns:
            True if the overlay was removed, False otherwise
        """
        # UI thread operation - no lock needed
        if self._overlays_locked and not bypass_lock:
            self._logger.warning("Cannot remove overlay: overlays are locked")
            return False
            
        return self._remove_overlay_internal(overlay_id)
    
    def destroy_overlay(self, overlay_id: str) -> bool:
        """Destroy and clean up an overlay (alias for remove_overlay).
        
        Args:
            overlay_id: The ID of the overlay to destroy
            
        Returns:
            True if the overlay was destroyed, False otherwise
        """
        return self.remove_overlay(overlay_id)

    def _remove_overlay_internal(self, overlay_id: str) -> bool:
        """Internal helper to remove and clean up an overlay without acquiring the lock twice."""
        overlay = self._overlays.pop(overlay_id, None)
        if overlay is None:
            return False

        # Remove from MRU and timestamps
        if overlay_id in self._mru_overlays:
            self._mru_overlays.remove(overlay_id)
        self._focus_timestamps.pop(overlay_id, None)

        # Unregister from auto-switch monitoring
        self._unregister_overlay_window(overlay_id)

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
        # UI thread operation - no lock needed
        if self._active_overlay_id is not None:
            self._remove_overlay_internal(self._active_overlay_id)
            self._active_overlay_id = None

    def set_opacity(self, value: float) -> bool:
        """Route opacity updates to the active overlay.

        Returns True if an active overlay exists and was updated.
        """
        # UI thread operation - no lock needed
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
        
        This method must be called from the UI thread to maintain lock-free operation.
        Cross-thread calls will be automatically dispatched to the UI thread.
        
        Args:
            overlay_id: The ID of the overlay to update
            **kwargs: Configuration values to update
            
        Returns:
            True if the update was successful, False otherwise
        """
        # Ensure UI thread operation for lock-free design
        from PySide6.QtCore import QThread, QCoreApplication
        app = QCoreApplication.instance()
        if app and QThread.currentThread() != app.thread():
            # Dispatch to UI thread and return immediately
            ThreadManager.run_on_ui_thread(self.update_overlay, overlay_id, **kwargs)
            return True
        
        # UI thread operation - no lock needed
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
        # UI thread operation - no lock needed
        return list(self._overlays.values())
    
    def enforce_z_order(self, overlay_id: str) -> bool:
        """Enforce z-order for the specified overlay using the centralized ZOrderManager.
        
        Args:
            overlay_id: ID of the overlay to enforce z-order for
            
        Returns:
            True if successful, False otherwise
        """
        # UI thread operation - no lock needed
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
        # UI thread operation - no lock needed
        if self._overlays_locked:
            self._logger.warning("Cannot clear overlays: overlays are locked")
            return
            
        for overlay_id in list(self._overlays.keys()):
            self.remove_overlay(overlay_id)
        
        # Clear MRU and timestamps
        self._mru_overlays.clear()
        self._focus_timestamps.clear()
        
        self._logger.info("Cleared all overlays")
    
    def _cleanup(self):
        """Cleanup handler for ResourceManager."""
        try:
            # Clear all overlays
            self.clear_all()
            # Reset state
            self._active_overlay_id = None
            self._overlays_locked = False
            self._mru_overlays.clear()
            self._focus_timestamps.clear()
            self._logger.debug("OverlayManager cleanup completed")
        except Exception as e:
            self._logger.error(f"Error during OverlayManager cleanup: {e}")
    
    def _initialize_closed_window_switch(self) -> None:
        """Initialize the closed-window switch manager."""
        try:
            from .window_monitor import initialize_closed_window_switch_manager
            self._closed_window_switch_manager = initialize_closed_window_switch_manager(self)
            self._logger.debug("Closed-window switch manager initialized")
        except Exception as e:
            self._logger.warning(f"Failed to initialize closed-window switch manager: {e}")
            self._closed_window_switch_manager = None
    
    def set_auto_switch_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-switching when windows close.
        
        Args:
            enabled: Whether to enable auto-switching
        """
        if self._closed_window_switch_manager:
            self._closed_window_switch_manager.set_auto_switch_enabled(enabled)
            self._logger.info(f"Closed-window switching {'enabled' if enabled else 'disabled'}")
        else:
            self._logger.warning("Closed-window switch manager not available")
    
    def _register_overlay_window(self, overlay_id: str, hwnd: int) -> None:
        """Register an overlay's source window for auto-switch monitoring.
        
        Args:
            overlay_id: ID of the overlay
            hwnd: Source window handle
        """
        if self._closed_window_switch_manager and hwnd:
            self._closed_window_switch_manager.register_overlay_window(overlay_id, hwnd)
    
    def _unregister_overlay_window(self, overlay_id: str) -> None:
        """Unregister an overlay from auto-switch monitoring.
        
        Args:
            overlay_id: ID of the overlay
        """
        if self._closed_window_switch_manager:
            self._closed_window_switch_manager.unregister_overlay(overlay_id)
    
    def _apply_initial_auto_switch_setting(self) -> None:
        """Apply the initial auto-switch setting from configuration."""
        try:
            from core.settings import get_settings_manager
            settings_manager = get_settings_manager()
            if settings_manager:
                # Prefer canonical feature flag; fall back to legacy key for backward compatibility
                auto_switch_enabled = bool(
                    settings_manager.get(
                        "features.autoswitch_enabled",
                        settings_manager.get("behavior.auto_switch", True),
                    )
                )
                self.set_auto_switch_enabled(auto_switch_enabled)
                self._logger.debug(f"Applied initial auto-switch setting: {auto_switch_enabled}")
        except Exception as e:
            self._logger.warning(f"Failed to apply initial auto-switch setting: {e}")
            # Default to enabled if settings unavailable
            self.set_auto_switch_enabled(True)

    def shutdown(self):
        """Explicit shutdown method."""
        # Shutdown closed-window switch manager
        if self._closed_window_switch_manager:
            try:
                from .window_monitor import shutdown_closed_window_switch_manager
                shutdown_closed_window_switch_manager()
                self._closed_window_switch_manager = None
            except Exception as e:
                self._logger.warning(f"Failed to shutdown closed-window switch manager: {e}")
        
        if hasattr(self, '_resource_id') and self._resource_id and hasattr(self, '_resource_manager') and self._resource_manager:
            try:
                self._resource_manager.unregister(self._resource_id)
                self._resource_id = None
            except Exception as e:
                self._logger.warning(f"Failed to unregister from ResourceManager: {e}")
        self._cleanup()
    
    def _update_focus_indicator_lock_state(self, locked: bool) -> None:
        """Update focus indicator lock state for all overlays.
        
        Args:
            locked: Whether overlays are locked
        """
        try:
            # Update all overlays, not just active one
            for overlay_id, overlay in self._overlays.items():
                if hasattr(overlay, '_overlay_host'):
                    host = getattr(overlay, '_overlay_host')
                    if host and hasattr(host, '_focus_indicator'):
                        indicator = getattr(host, '_focus_indicator')
                        if indicator and hasattr(indicator, 'set_locked'):
                            indicator.set_locked(locked)
                            self._logger.debug(f"Updated focus indicator lock state for overlay {overlay_id}: {locked}")
        except Exception as e:
            self._logger.debug(f"Failed to update focus indicator lock state: {e}")

    # Removed click-through functionality - was incompatible with overlay architecture
