"""
Docking Overlay Manager - Orchestrates the 3-overlay docking system.

This manager handles the lifecycle of three synchronized DWM overlays:
- Main overlay (100% size) - displays current MRU[0]
- Secondary overlay 1 (60% size) - displays MRU[1] 
- Secondary overlay 2 (40% size) - displays MRU[2]
"""
from __future__ import annotations
from typing import Optional, List
from PySide6.QtCore import QObject, QRect, QCoreApplication
from PySide6.QtGui import QGuiApplication
from core.logging import get_logger
from core.threading import get_thread_manager, ThreadManager
from core.settings import get_settings_manager
from utils.debug.log_suppressor import suppress_debug_log
from core.switching.mru_manager import get_mru_manager
from .overlay import DockingOverlay
from .config import DockingConfig
from utils.window.overlay_constants import OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT
from utils.window_validation import is_valid_window as _is_valid_window
from utils.window.monitors import (
    get_available_geometry_for_monitor,
)
from utils.window.overlay_persistence import (
    nearest_corner_state_from_rect,
    geometry_from_state,
)

logger = get_logger(__name__)

# Global singleton instance
_docking_manager_instance: Optional['DockingOverlayManager'] = None


def get_docking_manager() -> Optional['DockingOverlayManager']:
    """Get the global docking manager instance if it exists."""
    return _docking_manager_instance


class DockingOverlayManager(QObject):
    """Manages the lifecycle of the 3-overlay docking system."""
    
    # ANSI color code for bright pink/magenta for HWND highlighting in MRU logs
    _PINK = "\033[95m"
    _RESET = "\033[0m"
    
    @staticmethod
    def _pink_hwnd(hwnd: int) -> str:
        """Format HWND in bright pink for MRU/focus logs."""
        return f"{DockingOverlayManager._PINK}{hwnd}{DockingOverlayManager._RESET}"
    
    def __init__(self):
        """Initialize the docking overlay manager."""
        super().__init__()
        global _docking_manager_instance
        _docking_manager_instance = self
        self._main_overlay: Optional[DockingOverlay] = None
        self._secondary_overlays: List[DockingOverlay] = []
        self._positioner = None
        self._config: Optional[DockingConfig] = None
        self._thread_manager = get_thread_manager()
        self._logger = get_logger(__name__)
        self._geom_logger = get_logger("DOCK_GEOM")  # Purple-colored geometry diagnostics
        self._mru_logger = get_logger("MRU_FOCUS")  # Purple-colored MRU/focus diagnostics
        self._resource_manager = None
        self._is_active = False
        self._force_log_next_sync = False
        # Use centralized MRU manager instead of local list
        self._mru_manager = None
        # Suppress push-based MRU updates during user-initiated focus changes
        self._mru_push_suppressed: bool = False

        # State tracking
        self._is_active = False
        # Persist logging de-noise: only log when geometry changes
        self._last_persisted_rect = None
        self._is_initializing = False
        self._bound_as_unit = False
        self._initialization_complete = False
        self._overlay_locks: dict[str, bool] = {}
        self._app_instance_provider = None
        self._closed_window_switch_manager = None
        # Initialize overlay lock state (prevents AttributeError in interaction handlers)
        self._overlays_locked: bool = False
        # When true, we are restoring from OverlayStateManager and must not mutate MRU
        # or override geometry with persisted state during creation.
        self._is_restoring: bool = False
        # When True, do not persist geometry on next destroy (used by hide/show toggle)
        self._suppress_persist_on_destroy: bool = False
        # MRU state for docking mode
        try:
            settings = get_settings_manager()
            self._mru_capacity = int(settings.get("docking.mru_capacity", 12))
        except Exception:
            self._mru_capacity = 12
        # NOTE: No local MRU list - MRUManager is the SINGLE SOURCE OF TRUTH

        # Docking mode (normal | cycle) and overlay count (2-5)
        try:
            settings = get_settings_manager()
            self._docking_mode = str(settings.get("docking.mode", "normal") or "normal").lower()
        except Exception:
            self._docking_mode = "normal"
        try:
            settings = get_settings_manager()
            self._overlay_count = max(2, min(5, int(settings.get("docking.overlay_count", 3) or 3)))
        except Exception:
            self._overlay_count = 3

        # Cycle-mode state
        self._cycle_active = False
        self._cycle_interval_ms = 1200

        # Initialize centralized MRU manager used by autoswitch/cycling logic
        try:
            self._mru_manager = get_mru_manager()
            self._mru_manager.set_capacity(self._mru_capacity)
            suppress_debug_log(self._logger, f"Initialized MRUManager (cap={self._mru_capacity})", "DockingManager")
        except Exception as e:
            self._logger.warning(f"Failed to initialize MRUManager: {e}")
            self._mru_manager = None
        
        # CRITICAL: Initialize FocusTracker to track user window focus changes
        # This keeps MRU updated so Normal mode swaps use the correct window
        try:
            from core.switching.focus_tracker import get_focus_tracker
            self._focus_tracker = get_focus_tracker()
            suppress_debug_log(self._logger, "Initialized FocusTracker for docking mode", "DockingManager")
        except Exception as e:
            self._logger.warning(f"Failed to initialize FocusTracker: {e}")
            self._focus_tracker = None
        
        # Initialize WindowMonitor for closed window detection
        self._window_monitor = None
        try:
            from core.graphics.window_monitor import WindowMonitor
            self._window_monitor = WindowMonitor()
            self._window_monitor.window_closed.connect(self._on_window_closed)
            suppress_debug_log(self._logger, "Initialized WindowMonitor for docking mode", "DockingManager")
        except Exception as e:
            self._logger.warning(f"Failed to initialize WindowMonitor: {e}")
            self._window_monitor = None
        
        # NOTE: No MRU listener needed - DockingManager reads directly from MRUManager
        # This is the SINGLE SOURCE OF TRUTH architecture (no synchronization, no stale data)
        
        # DWM-style swap queuing system for main overlay
        self._main_swap_in_flight = False
        self._pending_main_swap_hwnd: Optional[int] = None
        self._pending_main_swap_record_mru = False
        
        # Normal mode: Track explicit overlay assignments (which window each overlay shows)
        # In Normal mode, overlays maintain sticky assignments unless explicitly changed
        
        # AR validation control: validate on creation/restoration/swaps, skip during user resizes
        self._last_validated_main_hwnd = None  # Track which HWND was last validated
        self._user_resize_in_progress = False  # True during wheel/handle resizes
        # Format: {"main": hwnd, "secondary_0": hwnd, "secondary_1": hwnd, ...}
        self._normal_mode_assignments: dict[str, Optional[int]] = {}
        for i in range(5):  # Support up to 5 secondary overlays
            self._normal_mode_assignments[f"secondary_{i}"] = None
        self._normal_mode_assignments["main"] = None
        
        # React to docking.mode and docking.overlay_count setting changes to start/stop cycle and rebuild
        try:
            self._settings_ref = get_settings_manager()
            self._settings_ref.register_change_handler("docking.mode", self._on_docking_mode_changed)
            self._settings_ref.register_change_handler("docking.overlay_count", self._on_overlay_count_changed)
        except Exception as e:
            self._logger.debug(f"Failed to register docking.mode change handler: {e}")
        # Debug verbosity flag for docking logs (SIZING/FIT/POSITION)
        try:
            self._docking_verbose = bool(get_settings_manager().get("debug.docking_verbose", False))
            if getattr(self, "_settings_ref", None):
                self._settings_ref.register_change_handler("debug.docking_verbose", self._on_docking_verbose_changed)
        except Exception as e:
            self._docking_verbose = False
            try:
                self._logger.debug(f"Failed to initialize docking verbose flag: {e}")
            except Exception:
                pass

        # Register with ResourceManager for deterministic cleanup
        try:
            from utils.resource_manager import get_resource_manager, ResourceType
            self._resource_manager = get_resource_manager()
            self._resource_id = None
            self._resource_id = self._resource_manager.register(
                self,
                ResourceType.CUSTOM,
                "DockingOverlayManager",
                cleanup_handler=lambda obj: obj._cleanup()
            )
            pass  # ResourceManager registration successful (debug suppressed)
        except Exception as e:
            self._logger.warning(f"Failed to register with ResourceManager: {e}")
            self._resource_manager = None
            self._resource_id = None
        
        # CRITICAL FIX: Integrate with hotkey system for opacity and other controls
        self._setup_hotkey_integration()
        
        # Initialize modules that DWM overlay also initializes
        self._initialize_missing_modules()

        # Ensure app_instance provider is available for context menu enumeration
        # This enables centralized window_menu_utils to function in docking mode
        try:
            self._initialize_app_instance_provider()
        except Exception as e:
            self._logger.warning(f"Failed to initialize app_instance provider: {e}")


    def destroy_docking_system(self, persist: bool = True) -> None:
        """Clean shutdown of all docking overlays.

        Args:
            persist: If True, saves main geometry before teardown. If False, skips persistence
                     (used for transient hide/show cycles).
        """
        self._is_active = False
        
        # Stop window monitoring
        try:
            if self._window_monitor:
                self._window_monitor.clear_all()
                self._window_monitor.window_closed.disconnect(self._on_window_closed)
        except Exception as e:
            self._logger.debug(f"Failed to cleanup window monitor: {e}")
        
        # Persist current main geometry before tearing down
        try:
            if persist and not getattr(self, "_suppress_persist_on_destroy", False):
                # Cancel any pending debounced save timer
                if hasattr(self, '_persist_timer'):
                    try:
                        self._persist_timer.cancel()
                    except Exception:
                        pass
                # Immediately flush current geometry (bypasses debounce)
                self._persist_main_geometry()
            else:
                try:
                    # Reset the flag after honoring it once
                    self._suppress_persist_on_destroy = False
                except Exception:
                    pass
        except Exception:
            pass
        
        # Unregister from ResourceManager first
        if self._resource_manager and self._resource_id:
            try:
                self._resource_manager.unregister(self._resource_id)
                self._resource_id = None
                pass  # ResourceManager unregistration successful (debug suppressed)
            except Exception as e:
                self._logger.warning(f"Failed to unregister from ResourceManager: {e}")
        
        # Cleanup secondary overlays with resource cleanup
        for i, overlay in enumerate(self._secondary_overlays):
            if overlay:
                try:
                    # Unregister from auto-switch manager
                    overlay_id = f"secondary_{i}"
                    self.unregister_overlay_from_auto_switch(overlay_id)
                    
                    # Unregister from ResourceManager if registered
                    if self._resource_manager and hasattr(overlay, '_resource_id') and overlay._resource_id:
                        try:
                            self._resource_manager.unregister(overlay._resource_id)
                        except Exception as e:
                            self._logger.warning(f"Failed to unregister secondary overlay {i} from ResourceManager: {e}")
                    
                    overlay.cleanup()
                except Exception as e:
                    self._logger.warning(f"Error cleaning up secondary overlay {i}: {e}")
        self._secondary_overlays.clear()
        
        # Cleanup main overlay with resource cleanup
        if self._main_overlay:
            try:
                # Unregister from auto-switch manager
                self.unregister_overlay_from_auto_switch('main')
                
                # Unregister from ResourceManager if registered
                if self._resource_manager and hasattr(self._main_overlay, '_resource_id') and self._main_overlay._resource_id:
                    try:
                        self._resource_manager.unregister(self._main_overlay._resource_id)
                    except Exception as e:
                        self._logger.warning(f"Failed to unregister main overlay from ResourceManager: {e}")
                
                self._main_overlay.cleanup()
            except Exception as e:
                self._logger.warning(f"Error cleaning up main overlay: {e}")
            self._main_overlay = None
        
        # Leave manager inactive; no re-initialization here
        self._config = DockingConfig()
        self._positioner = None
        self._bound_as_unit = False
        
        # Clear global singleton instance
        global _docking_manager_instance
        if _docking_manager_instance is self:
            _docking_manager_instance = None
        
        # Clear overlay pool to prevent stale overlays with deleted Qt widgets
        try:
            from .overlay_pool import get_docking_overlay_pool
            pool = get_docking_overlay_pool()
            if pool:
                pool.clear_pool()
                self._logger.debug("Cleared docking overlay pool")
        except Exception as e:
            self._logger.debug(f"Failed to clear overlay pool: {e}")
        
        self._logger.info("Docking system destroyed")

    def set_restoring(self, restoring: bool) -> None:
        """Enable/disable restore mode to avoid MRU and geometry churn during creation."""
        self._is_restoring = bool(restoring)

    def create_docking_system(self, target_hwnds: List[int]) -> bool:
        """Create the complete docking system with main and secondary overlays."""
        try:
            if self._is_active:
                self._logger.warning("Docking system already active")
                return True
            
            if not target_hwnds:
                self._logger.error("No target windows provided for docking system")
                return False
            
            # Set initializing flag BEFORE overlay creation to prevent context menu creation
            self._is_initializing = True
            
            # Create overlay config for main overlay with default size/position
            from core.graphics.types import OverlayConfig, OverlayType
            from PySide6.QtCore import QSize, QPoint
            default_size = QSize(400, 300)  # Default main overlay size
            default_position = QPoint(100, 100)  # Default position
            
            main_config = OverlayConfig(
                overlay_type=OverlayType.DOCKING,
                title="Main Docking Overlay",
                size=default_size,
                position=default_position,
                parent_hwnd=target_hwnds[0]
            )
        
            self._main_overlay = DockingOverlay("main", main_config, is_main=True, manager=self)
            if not self._main_overlay.initialize(target_hwnd=target_hwnds[0]):
                self._logger.error("Failed to initialize main docking overlay")
                return False
        
            # Inject app instance for context menu functionality
            self._inject_app_instance_into_overlay(self._main_overlay)
            # Register main overlay for closed-window autoswitch monitoring
            try:
                if target_hwnds[0]:
                    self.register_overlay_for_auto_switch("main", int(target_hwnds[0]))
            except Exception:
                pass
            
            # Populate MRU with additional windows if needed
            from core.window.enumerator import WindowEnumerator
            needed = max(1, int(self._overlay_count))
            if len(target_hwnds) < needed:
                try:
                    windows = WindowEnumerator.enum_windows()
                    additional_hwnds = []
                    for hwnd, title in windows:
                        if hwnd not in target_hwnds and hwnd != 0:
                            additional_hwnds.append(hwnd)
                            if len(target_hwnds) + len(additional_hwnds) >= needed:
                                break
                    target_hwnds.extend(additional_hwnds)
                except Exception as e:
                    self._logger.debug(f"Failed to populate additional target windows: {e}")

            # Create secondary overlays with unique target windows
            self._secondary_overlays = []
            secondary_overlay_ids = [f"secondary_{i}" for i in range(max(0, self._overlay_count - 1))]
            
            for i, overlay_id in enumerate(secondary_overlay_ids):
                # Get unique target window from expanded target list
                mru_index = i + 1
                target_hwnd = target_hwnds[mru_index] if mru_index < len(target_hwnds) else target_hwnds[0]
                
                # Calculate proper size for secondary overlay with descending hierarchy
                main_size = main_config.size
                secondary_ratio = 0.7 if i == 0 else 0.5  # For D and E, keep 50%
                secondary_width = int(main_size.width() * secondary_ratio)
                secondary_height = int(main_size.height() * secondary_ratio)
                
                # Position directly adjacent with NO gaps - bound together
                main_pos = main_config.position
                if i == 0:
                    # First secondary: directly adjacent to main overlay (no gap)
                    secondary_x = main_pos.x() + main_size.width()
                else:
                    # Subsequent secondaries: place after previous secondary (no gap)
                    prev_total_w = int(main_size.width() * 0.7)
                    if i > 1:
                        # Add widths of previous secondaries (C, D...) at 50%
                        prev_total_w += int(main_size.width() * 0.5 * (i - 1))
                    secondary_x = main_pos.x() + main_size.width() + prev_total_w
                
                # Align Y position exactly with main overlay for perfect horizontal alignment
                secondary_y = main_pos.y()
                
                secondary_config = OverlayConfig(
                    overlay_type=OverlayType.DOCKING,
                    title=f"Secondary Docking Overlay {i}",
                    size=QSize(secondary_width, secondary_height),
                    position=QPoint(secondary_x, secondary_y),
                    parent_hwnd=target_hwnd
                )
                
                overlay = DockingOverlay(overlay_id, secondary_config, is_main=False, manager=self)
                if overlay.initialize(target_hwnd=target_hwnd):
                    # Inject app instance for context menu functionality
                    self._inject_app_instance_into_overlay(overlay)
                    self._secondary_overlays.append(overlay)
                    # Register secondary overlay for closed-window autoswitch monitoring
                    try:
                        if target_hwnd:
                            self.register_overlay_for_auto_switch(overlay_id, int(target_hwnd))
                    except Exception:
                        pass
                    pass  # Secondary overlay initialized (debug suppressed)
                else:
                    self._logger.warning(f"Failed to initialize secondary overlay: {overlay_id}")
            
            self._logger.info(f"Created docking system with {len(self._secondary_overlays)} secondary overlays")
            
            # Try to restore persisted main overlay geometry before positioning
            # Skip this during state restoration; OverlayStateManager will apply saved geometry.
            if not self._is_restoring:
                try:
                    self._apply_persisted_main_geometry()
                except Exception:
                    pass

            # Mark active BEFORE any positioning so initial syncs aren't aborted
            self._is_active = True
            
            # Set up positioning and synchronization
            self._setup_positioning()
            self._setup_overlay_synchronization()

            # Apply initial opacity from OpacityManager before showing overlays
            try:
                self._set_initial_opacity()
            except Exception:
                pass

            # Show all overlays unless we're in a state restoration flow.
            # During restoration, OverlayStateManager will apply the saved geometry
            # and then show overlays to avoid a flash at default size/position.
            if not getattr(self, "_is_restoring", False):
                self._show_all_overlays()
            
            # Initialize Normal mode assignments with current overlay windows
            # This captures the initial state for sticky assignment behavior
            try:
                if self._main_overlay:
                    main_hwnd = self._main_overlay.get_target_hwnd()
                    self._normal_mode_assignments["main"] = main_hwnd
                    
                for i, overlay in enumerate(self._secondary_overlays):
                    overlay_id = f"secondary_{i}"
                    overlay_hwnd = overlay.get_target_hwnd()
                    self._normal_mode_assignments[overlay_id] = overlay_hwnd
                    
                self._logger.debug(f"Initialized Normal mode assignments: {self._normal_mode_assignments}")
            except Exception as e:
                self._logger.warning(f"Failed to initialize Normal mode assignments: {e}")
            
            # Prevent MRU updates from clearing secondary overlays during initialization
            self._initialization_complete = True
            
            # Populate MRU with all target windows for quickswitch and swap operations
            # Skip this during state restoration to preserve the saved MRU ordering.
            if not self._is_restoring:
                try:
                    # Record all assigned windows in centralized MRUManager (main first, then secondaries)
                    if self._main_overlay:
                        main_hwnd = self._main_overlay.get_target_hwnd()
                        if main_hwnd:
                            self._mru_manager.record(main_hwnd)
                    
                    for overlay in self._secondary_overlays:
                        overlay_hwnd = overlay.get_target_hwnd()
                        if overlay_hwnd:
                            self._mru_manager.record(overlay_hwnd)
                    
                    # Populate with additional windows from system to have a full MRU
                    current_mru_count = len(self._get_current_mru())
                    min_mru_count = max(6, current_mru_count + 3)  # Want at least 6 windows for good quickswitch
                    if current_mru_count < min_mru_count:
                        self._populate_mru_with_visible_windows(min_mru_count)
                    
                    self._logger.debug(f"Initialized MRU with {len(self._get_current_mru())} windows")
                except Exception as e:
                    self._logger.warning(f"Failed to populate initial MRU: {e}")

            # Bind overlays for cohesive movement
            self._bind_overlays_for_movement()
            
            # Clear initializing flag BEFORE sync to allow it to run
            self._is_initializing = False
            
            # Delayed initial sync after overlays are stable (100ms)
            # This prevents cascade during restoration
            def _delayed_initial_sync():
                try:
                    self.sync_overlay_properties()
                except Exception as e:
                    self._logger.warning(f"Delayed initial sync failed: {e}")
            
            if QCoreApplication.instance() is not None:
                ThreadManager.single_shot(100, _delayed_initial_sync)
            else:
                self._thread_manager.run_on_ui_thread(_delayed_initial_sync)
            
            # Note: Do not manipulate opacity at creation time in docking mode.
            # IntegratedDWMOverlay manages initial opacity consistently; forcing
            # 0% then raising causes visible flicker and repeated clamp logs.
            
            self._logger.info(f"Docking system created successfully with {1 + len(self._secondary_overlays)} bound overlays (mode={self._docking_mode})")

            # Start/stop cycle loop based on mode
            try:
                if self.is_cycle_mode():
                    self._start_cycle_mode()
                else:
                    self._stop_cycle_mode()
            except Exception:
                pass
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to create docking system: {e}")
            # Clear initializing flag on failure
            self._is_initializing = False
            self._cleanup_partial_creation()
            return False

    def _apply_persisted_main_geometry(self) -> bool:
        """Restore last saved nearest-corner geometry for the main overlay, if available.
        Returns True if applied, else False.
        """
        try:
            # Do not apply persisted geometry during Hide/Show restoration
            if getattr(self, "_is_restoring", False):
                try:
                    suppress_debug_log(self._logger, "[DOCK_PERSIST] Skipped during restoration", "DockingManager")
                except Exception:
                    pass
                return False
            if not self._main_overlay:
                return False
            
            # Prevent duplicate application - mark that we've applied once
            if getattr(self, "_persist_applied_once", False):
                suppress_debug_log(self._logger, "[DOCK_PERSIST] Already applied, skipping duplicate", "DockingManager")
                return False
            self._persist_applied_once = True
            
            settings = get_settings_manager()
            state = settings.get("docking.last_state", None)
            if not isinstance(state, dict) or not state:
                return False
            # Log the loaded state for diagnostics
            try:
                corner = state.get("corner")
                mon_idx_dbg = state.get("monitor_index")
                self._logger.info(f"[DOCK_PERSIST] Loaded state: corner={corner}, monitor_index={mon_idx_dbg}, size={state.get('width')}x{state.get('height')}")
            except Exception:
                pass
            mon_idx = int(state.get("monitor_index", 0))
            # Use full monitor geometry for docking (physical pixels), not work area or logical pixels
            try:
                screens = QGuiApplication.screens()
                if 0 <= mon_idx < len(screens):
                    screen = screens[mon_idx]
                    # Get logical geometry and convert to physical pixels for DPI-scaled displays
                    logical_geom = screen.geometry()
                    dpr = screen.devicePixelRatio()
                    # Convert to physical pixels
                    phys_x = int(logical_geom.x() * dpr)
                    phys_y = int(logical_geom.y() * dpr)
                    phys_w = int(logical_geom.width() * dpr)
                    phys_h = int(logical_geom.height() * dpr)
                    avail = QRect(phys_x, phys_y, phys_w, phys_h)
                    self._logger.debug(f"[DOCK_PERSIST] Screen {mon_idx}: logical={logical_geom.width()}x{logical_geom.height()}, DPR={dpr:.2f}, physical={phys_w}x{phys_h}")
                else:
                    screen = QGuiApplication.primaryScreen()
                    logical_geom = screen.geometry()
                    dpr = screen.devicePixelRatio()
                    phys_x = int(logical_geom.x() * dpr)
                    phys_y = int(logical_geom.y() * dpr)
                    phys_w = int(logical_geom.width() * dpr)
                    phys_h = int(logical_geom.height() * dpr)
                    avail = QRect(phys_x, phys_y, phys_w, phys_h)
                    self._logger.debug(f"[DOCK_PERSIST] Primary screen: logical={logical_geom.width()}x{logical_geom.height()}, DPR={dpr:.2f}, physical={phys_w}x{phys_h}")
            except Exception as e:
                self._logger.warning(f"[DOCK_PERSIST] Failed to get physical geometry: {e}, using fallback")
                avail = get_available_geometry_for_monitor(mon_idx)
                self._logger.debug(f"[DOCK_PERSIST] Using fallback geometry: ({avail.x()},{avail.y()}) {avail.width()}x{avail.height()}")
            # Optional insets from underlying canvas; aspect left to system logic for docking
            from PySide6.QtCore import QSize as _QSize
            min_size = _QSize(OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT)
            insets = (0, 0)
            try:
                if hasattr(self._main_overlay, '_dwm_overlay') and self._main_overlay._dwm_overlay:
                    d = self._main_overlay._dwm_overlay
                    c = getattr(d, '_canvas', None)
                    if c and hasattr(c, 'get_content_insets'):
                        cx, cy = c.get_content_insets()
                        insets = (max(0, int(cx)), max(0, int(cy)))
            except Exception:
                pass
            rect_out = geometry_from_state(state, avail, min_size, aspect=None, insets=insets, constrain="physical")
            if rect_out is None:
                self._logger.warning("[DOCK_PERSIST] geometry_from_state returned None")
                return False
            self._logger.debug(f"[DOCK_PERSIST] Calculated physical geometry: ({rect_out.x()},{rect_out.y()}) {rect_out.width()}x{rect_out.height()}")
            
            # Convert from physical back to logical pixels for Qt widget
            screens = QGuiApplication.screens()
            if 0 <= mon_idx < len(screens):
                dpr = screens[mon_idx].devicePixelRatio()
            else:
                dpr = QGuiApplication.primaryScreen().devicePixelRatio()
            
            logical_x = int(rect_out.x() / dpr)
            logical_y = int(rect_out.y() / dpr)
            logical_w = int(rect_out.width() / dpr)
            logical_h = int(rect_out.height() / dpr)
            self._logger.debug(f"[DOCK_PERSIST] Converted to logical: ({logical_x},{logical_y}) {logical_w}x{logical_h} (DPR={dpr:.2f})")
            
            # Apply logical coordinates for Qt widget
            self._main_overlay.set_geometry(logical_x, logical_y, logical_w, logical_h)
            
            # Suppress spurious persistence saves for 2 seconds after restoration
            import time
            self._persist_suppression_until = time.time() + 2.0
            
            try:
                # Fetch final geometry after scheduling set (best-effort)
                final_rect = self._main_overlay.get_geometry()
                self._logger.info(
                    f"[DOCK_PERSIST] Applied persisted docking main geometry -> ({final_rect.x()},{final_rect.y()}) {final_rect.width()}x{final_rect.height()}"
                )
            except Exception:
                self._logger.info("[DOCK_PERSIST] Applied persisted docking main geometry")
            return True
        except Exception as e:
            suppress_debug_log(self._logger, f"Apply persisted docking geometry failed: {e}", "DockingManager")
            return False

    def hide_all_overlays(self) -> None:
        """Hide all overlays using robust OverlayStateManager.
        
        This properly captures state for restoration via hotkey or tray menu.
        Replaces simple .hide() calls with proper state management.
        """
        try:
            from core.overlay_state_manager import get_overlay_state_manager
            state_mgr = get_overlay_state_manager()
            
            if state_mgr.hide_all_overlays():
                self._logger.info("Docking: all overlays hidden with state capture")
            else:
                # Fallback to simple hide if state manager fails
                self._logger.warning("State manager failed, using simple hide fallback")
                if self._main_overlay:
                    self._main_overlay.hide()
                for overlay in self._secondary_overlays:
                    try:
                        overlay.hide()
                    except Exception:
                        pass
        except Exception as e:
            self._logger.error(f"Failed to hide all overlays: {e}")

    def _on_main_geometry_changed(self) -> None:
        """Handle main overlay geometry changes (move/resize) to persist state.
        
        Uses debouncing to avoid excessive saves during drag/resize operations.
        """
        try:
            # Cancel any pending persistence timer
            if hasattr(self, '_persist_timer'):
                try:
                    self._persist_timer.cancel()
                except Exception:
                    pass
            
            # Schedule persistence after 500ms of inactivity (debounce)
            def _delayed_persist():
                try:
                    self._persist_main_geometry()
                except Exception as e:
                    self._logger.debug(f"Delayed persist failed: {e}")
            
            self._persist_timer = self._thread_manager.single_shot(500, _delayed_persist)
        except Exception as e:
            suppress_debug_log(self._logger, f"Failed to schedule geometry persistence: {e}", "DockingManager")

    def _persist_main_geometry(self) -> None:
        """Persist current main overlay geometry as nearest-corner state."""
        try:
            if not self._main_overlay:
                return
            
            # Suppress spurious saves for 2 seconds after restoration
            if hasattr(self, '_persist_suppression_until'):
                import time
                if time.time() < self._persist_suppression_until:
                    suppress_debug_log(self._logger, "[DOCK_PERSIST] Suppressed spurious save during restoration cooldown", "DockingManager")
                    return
            
            rect = self._main_overlay.get_geometry()
            if not isinstance(rect, QRect) or rect.isEmpty():
                return
            state = nearest_corner_state_from_rect(rect, constrain="physical")
            if not state:
                return
            get_settings_manager().set("docking.last_state", state)
            # INFO-level diagnostic: only emit when geometry changes
            try:
                rect_tuple = (int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))
                if getattr(self, "_last_persisted_rect", None) != rect_tuple:
                    self._logger.info(
                        f"[DOCK_PERSIST] Saved main geometry {rect_tuple[2]}x{rect_tuple[3]} at ({rect_tuple[0]},{rect_tuple[1]})"
                    )
                    self._last_persisted_rect = rect_tuple
            except Exception:
                pass
        except Exception as e:
            suppress_debug_log(self._logger, f"Persist docking geometry failed: {e}", "DockingManager")

    def _bind_overlays_for_movement(self) -> None:
        """Bind overlays for cohesive movement using event filter on overlay hosts."""
        try:
            if not self._main_overlay:
                self._logger.warning("Cannot bind overlays: main overlay not ready")
                return
            
            # Get overlay host widgets for event filtering
            main_host = None
            if hasattr(self._main_overlay, '_dwm_overlay') and self._main_overlay._dwm_overlay:
                main_host = getattr(self._main_overlay._dwm_overlay, '_host', None)
            
            if not main_host:
                # Schedule retry if host not ready
                self._thread_manager.single_shot(100, self._bind_overlays_for_movement)
                return
            
            # Install event filter on main overlay host
            main_host.installEventFilter(self)
            
            # Connect main overlay geometryChanged to persistence (enables size persistence on resize)
            try:
                if hasattr(main_host, 'geometryChanged'):
                    main_host.geometryChanged.connect(self._on_main_geometry_changed)
                    suppress_debug_log(self._logger, "Connected main overlay geometryChanged to persistence", "DockingManager")
            except Exception as e:
                self._logger.warning(f"Failed to connect geometryChanged signal: {e}")
            
            # Install event filter on secondary overlay hosts
            for i, overlay in enumerate(self._secondary_overlays):
                if overlay and hasattr(overlay, '_dwm_overlay') and overlay._dwm_overlay:
                    sec_host = getattr(overlay._dwm_overlay, '_host', None)
                    if sec_host:
                        sec_host.installEventFilter(self)
            
            self._bound_as_unit = True
            suppress_debug_log(self._logger, "Docking overlays bound with event filters", "DockingManager")
        except Exception as e:
            self._logger.error(f"Failed to bind overlays for movement: {e}")

    def eventFilter(self, obj, event) -> bool:
        """Filter events from overlay hosts to handle drag and sync."""
        from PySide6.QtCore import QEvent
        from PySide6.QtCore import Qt

        # Check if event comes from any overlay host
        try:
            is_main_host = False
            is_any_overlay_host = False
            is_secondary_host = False
            secondary_overlay_ref = None
            
            # Check if event comes from main overlay
            if self._main_overlay and hasattr(self._main_overlay, '_dwm_overlay'):
                main_dwm = self._main_overlay._dwm_overlay
                main_host = getattr(main_dwm, '_host', None)
                if main_host is not None and obj is main_host:
                    is_main_host = True
                    is_any_overlay_host = True
            
            # Check if event comes from any secondary overlay
            if not is_any_overlay_host:
                for overlay in self._secondary_overlays:
                    if overlay and hasattr(overlay, '_dwm_overlay'):
                        sec_dwm = overlay._dwm_overlay
                        sec_host = getattr(sec_dwm, '_host', None)
                        if sec_host is not None and obj is sec_host:
                            is_any_overlay_host = True
                            is_secondary_host = True
                            secondary_overlay_ref = overlay
                            break
            
            if not is_any_overlay_host:
                return False
        except Exception:
            pass

        # If we are applying a batch geometry update, ignore host events to avoid feedback loops
        try:
            if getattr(self, '_batch_applying', False):
                return False
        except Exception:
            pass

        et = event.type()
        if et == QEvent.Type.Move:
            # Hide secondaries during drag for performance (only from main host)
            if is_main_host and not hasattr(self, '_drag_in_progress'):
                self._drag_in_progress = True
                if getattr(self, '_hide_secondaries_during_drag', False):
                    self._hide_secondary_overlays_for_drag()
            # If a secondary overlay host is moved, translate main by the same delta to keep the dock cohesive
            if is_secondary_host and secondary_overlay_ref is not None and self._main_overlay and getattr(self, '_secondary_drag_active', False):
                try:
                    sec_host = getattr(getattr(secondary_overlay_ref, '_dwm_overlay', None), '_host', None)
                except Exception:
                    sec_host = None
                if sec_host is not None:
                    try:
                        curr_pos = sec_host.pos()
                    except Exception:
                        curr_pos = None
                    last_key = f"_last_pos_{id(sec_host)}"
                    last_pos = getattr(self, last_key, None)
                    if curr_pos is not None and last_pos is not None:
                        try:
                            dx = int(curr_pos.x()) - int(last_pos.x())
                            dy = int(curr_pos.y()) - int(last_pos.y())
                            if dx or dy:
                                # Skip translating main when batch/apply is in progress to avoid teleport
                                if not getattr(self, '_batch_applying', False):
                                    main_rect = self._main_overlay.get_geometry()
                                    new_x = int(main_rect.x()) + dx
                                    new_y = int(main_rect.y()) + dy
                                    self._main_overlay.set_geometry(new_x, new_y, int(main_rect.width()), int(main_rect.height()))
                        except Exception:
                            pass
                    # Update last position for next delta
                    if curr_pos is not None:
                        try:
                            setattr(self, last_key, curr_pos)
                        except Exception:
                            pass
            self._coalesced_sync(5)
        elif et == QEvent.Type.Resize:
            # Sync on resize from any overlay
            # ARCHITECTURAL RULE: Main → Secondaries ONLY (never reverse)
            # Secondaries should NEVER resize main; that creates feedback loops
            
            # Skip all resize-triggered syncs during:
            # 1. Batch apply mode (sync in progress)
            # 2. Initialization (system not stable)
            if getattr(self, '_batch_applying', False):
                return False
            if getattr(self, '_is_initializing', False):
                return False
            
            # Only trigger sync if main was resized (not secondaries)
            if is_main_host:
                # Mark as user resize if not during batch operations
                # This prevents AR validation from running during manual resize
                if not getattr(self, '_batch_applying', False):
                    try:
                        self._user_resize_in_progress = True
                        # Clear flag after 500ms of no resize events
                        if hasattr(self, '_clear_resize_timer'):
                            try:
                                self._clear_resize_timer.cancel()
                            except Exception:
                                pass
                        # Schedule clearing the flag
                        def clear_resize_flag():
                            try:
                                self._user_resize_in_progress = False
                            except Exception:
                                pass
                        self._clear_resize_timer = self._thread_manager.single_shot(500, clear_resize_flag)
                    except Exception:
                        pass
                self._coalesced_sync(5)
        elif et == QEvent.Type.MouseButtonPress:
            try:
                if hasattr(event, 'button') and event.button() == Qt.MouseButton.LeftButton:
                    # Docking: rely on WindowBehaviorManager for drag/resize; no manual drag state here
                    if is_main_host and getattr(self, '_hide_secondaries_during_drag', False):
                        self._hide_secondary_overlays_for_drag()
                    # Track active drag from secondary to allow group translation only while held
                    if is_secondary_host:
                        try:
                            setattr(self, '_secondary_drag_active', True)
                        except Exception:
                            pass
            except Exception:
                pass
        elif et == QEvent.Type.MouseMove:
            # If dragging from a secondary, translate main by mouse delta to avoid host jitter
            try:
                if is_secondary_host and getattr(self, '_secondary_drag_active', False) and self._main_overlay:
                    # Use global mouse position delta for smooth group move
                    curr_gp = None
                    try:
                        if hasattr(event, 'globalPosition'):
                            curr_gp = event.globalPosition().toPoint()
                        elif hasattr(event, 'globalPos'):
                            curr_gp = event.globalPos()
                    except Exception:
                        curr_gp = None
                    last_gp = getattr(self, '_secondary_drag_global_last', None)
                    if curr_gp is not None and last_gp is not None:
                        dx = int(curr_gp.x()) - int(last_gp.x())
                        dy = int(curr_gp.y()) - int(last_gp.y())
                        if dx or dy:
                            try:
                                main_rect = self._main_overlay.get_geometry()
                                new_x = int(main_rect.x()) + dx
                                new_y = int(main_rect.y()) + dy
                                self._main_overlay.set_geometry(new_x, new_y, int(main_rect.width()), int(main_rect.height()))
                            except Exception:
                                pass
                        try:
                            setattr(self, '_secondary_drag_global_last', curr_gp)
                        except Exception:
                            pass
                    # Schedule a sync to reposition secondaries after main moves
                    self._coalesced_sync(5)
            except Exception:
                pass
        elif et == QEvent.Type.MouseButtonRelease:
            try:
                needs_restore = False
                if hasattr(self, '_drag_in_progress'):
                    needs_restore = True
                if needs_restore:
                    if getattr(self, '_hide_secondaries_during_drag', False):
                        self._show_secondary_overlays_after_drag()
                    else:
                        if hasattr(self, '_drag_in_progress'):
                            try:
                                delattr(self, '_drag_in_progress')
                            except Exception:
                                pass
                # Clear secondary drag flag
                try:
                    if hasattr(self, '_secondary_drag_active'):
                        setattr(self, '_secondary_drag_active', False)
                except Exception:
                    pass
                if hasattr(self, '_last_sync_state'):
                    try:
                        delattr(self, '_last_sync_state')
                    except Exception:
                        pass
                self._coalesced_sync(1)
            except Exception:
                pass
        elif et == QEvent.Type.Wheel:
            # Mark user resize in progress during wheel events
            # This prevents AR validation from interfering with manual resizing
            try:
                self._user_resize_in_progress = True
                # Clear flag after 500ms of no wheel events
                if hasattr(self, '_clear_resize_timer'):
                    try:
                        self._clear_resize_timer.cancel()
                    except Exception:
                        pass
                # Schedule clearing the flag
                def clear_resize_flag():
                    try:
                        self._user_resize_in_progress = False
                    except Exception:
                        pass
                self._clear_resize_timer = self._thread_manager.single_shot(500, clear_resize_flag)
            except Exception:
                pass
        return False

    def _show_all_overlays(self) -> None:
        """Show all overlays in the docking system."""
        # Apply persisted geometry on restore/show to keep corner+size memory
        try:
            # Do not apply persisted geometry during Hide/Show restoration; the
            # OverlayStateManager has already applied the exact saved rect.
            if not getattr(self, "_is_restoring", False):
                self._apply_persisted_main_geometry()
        except Exception:
            pass

        if self._main_overlay:
            self._main_overlay.show()
            # Enforce z-order for main overlay to ensure it's above taskbar
            try:
                from utils.z_order_manager import get_z_order_manager
                zm = get_z_order_manager()
                if hasattr(self._main_overlay, 'id') and self._main_overlay.id:
                    # CRITICAL priority enforces immediately (no debounce)
                    zm.enforce_z_order_critical(self._main_overlay.id)
            except Exception as e:
                self._logger.debug(f"Failed to enforce main overlay z-order on show: {e}")
                
            for overlay in self._secondary_overlays:
                overlay.show()
                # CRITICAL: Enforce z-order immediately after show to prevent taskbar coverage
                # Secondary overlays frequently get covered by taskbar without explicit enforcement
                # Use CRITICAL priority for immediate HWND_TOPMOST without debouncing
                try:
                    from utils.z_order_manager import get_z_order_manager
                    zm = get_z_order_manager()
                    if hasattr(overlay, 'id') and overlay.id:
                        # CRITICAL priority enforces immediately (no debounce)
                        zm.enforce_z_order_critical(overlay.id)
                except Exception as e:
                    self._logger.debug(f"Failed to enforce secondary overlay z-order on show: {e}")

    def _get_default_overlay_size(self):
        """Default size used by tests; returns QSize with a sensible default."""
        try:
            from PySide6.QtCore import QSize
            return QSize(400, 300)
        except Exception:
            return None

    def _get_default_overlay_position(self):
        """Default position used by tests; returns QPoint with a sensible default."""
        try:
            from PySide6.QtCore import QPoint
            return QPoint(100, 100)
        except Exception:
            return None

    def _coalesced_sync(self, delay_ms: int) -> None:
        """Coalesce repeated sync requests using ThreadManager without Qt timers."""
        try:
            # Suppress syncs during batch mode or initialization
            if getattr(self, '_batch_applying', False):
                return
            if getattr(self, '_is_initializing', False):
                return
            
            if getattr(self, '_sync_coalesce_pending', False):
                return
            self._sync_coalesce_pending = True
            def _do_sync():
                try:
                    self.sync_overlay_properties()
                finally:
                    try:
                        self._sync_coalesce_pending = False
                    except Exception:
                        pass
            if QCoreApplication.instance() is not None:
                ThreadManager.single_shot(max(1, int(delay_ms)), _do_sync)
            else:
                self._thread_manager.run_on_ui_thread(_do_sync)
        except Exception:
            # Keep this debug minimal to avoid spam
            try:
                self._thread_manager.run_on_ui_thread(self.sync_overlay_properties)
            except Exception:
                pass

    # --- Cycle Mode -------------------------------------------------------
    def is_cycle_mode(self) -> bool:
        try:
            return str(get_settings_manager().get("docking.mode", getattr(self, "_docking_mode", "normal"))).lower() == "cycle"
        except Exception:
            return str(getattr(self, "_docking_mode", "normal")).lower() == "cycle"

    def _start_cycle_mode(self) -> None:
        try:
            if getattr(self, "_cycle_active", False):
                return
            self._cycle_active = True
            
            # Start polling for MRU changes
            self._schedule_cycle_tick()
                
            self._logger.info("Docking cycle mode started")
        except Exception as e:
            self._logger.warning(f"Failed to start cycle mode: {e}")

    def _stop_cycle_mode(self) -> None:
        try:
            if not getattr(self, "_cycle_active", False):
                return
            self._cycle_active = False
            
            # No listener management needed - both modes read directly from MRUManager
            # Just clear cycle-specific flags
            try:
                self._cycle_push_enabled = False
            except Exception:
                pass
                
            self._logger.info("Docking cycle mode stopped")
        except Exception:
            pass

    def _schedule_cycle_tick(self) -> None:
        if not getattr(self, "_cycle_active", False):
            return
        try:
            ThreadManager.single_shot(int(getattr(self, "_cycle_interval_ms", 1200)), self._cycle_tick)
        except Exception:
            pass

    def _cycle_tick(self) -> None:
        """Polling mechanism for cycle mode - checks MRU and updates overlays."""
        if not getattr(self, "_cycle_active", False) or not getattr(self, "_is_active", False):
            return
        try:
            # Use the centralized update mechanism which handles all the logic
            self._update_overlay_displays()
            
            # Trigger sync after cycle updates to validate AR for new windows
            if getattr(self, '_initialization_complete', False):
                self._coalesced_sync(10)  # Small delay to let swap complete
            
            # Continue polling
            self._schedule_cycle_tick()
        except Exception as e:
            self._logger.debug(f"Cycle tick error: {e}")
            # Continue polling even on error
            try:
                self._schedule_cycle_tick()
            except Exception:
                pass
    
    def _cycle_tick_old_implementation(self) -> None:
        """OLD: Manual assignment approach - replaced by _update_overlay_displays()."""
        if not getattr(self, "_cycle_active", False) or not getattr(self, "_is_active", False):
            return
        try:
            mru = self._mru_manager or get_mru_manager()
            count = 1 + len(self._secondary_overlays)
            recent = mru.get_recent(limit=count + 1) if mru else []
            # Exclude the current foreground window in cycle mode to avoid showing it in overlays
            try:
                import win32gui  # type: ignore
                fg = int(win32gui.GetForegroundWindow() or 0)
                if fg:
                    recent = [h for h in recent if h != fg]
            except Exception:
                pass
            # Assign MRU windows to overlays in order while respecting locks
            if self._main_overlay and recent:
                try:
                    if not self._is_overlay_locked(self._main_overlay):
                        cur = getattr(self._main_overlay, '_target_hwnd', None)
                        if recent[0] and recent[0] != cur and _is_valid_window(recent[0]):
                            self._main_overlay.set_target_window(int(recent[0]))
                except Exception:
                    pass
            for idx, overlay in enumerate(self._secondary_overlays, start=1):
                try:
                    if idx < len(recent) and overlay and not self._is_overlay_locked(overlay):
                        target = recent[idx]
                        cur = getattr(overlay, '_target_hwnd', None)
                        if target and target != cur and _is_valid_window(target):
                            overlay.set_target_window(int(target))
                except Exception:
                    pass
            # Re-sync layout
            self._coalesced_sync(10)
        finally:
            self._schedule_cycle_tick()

    def _on_docking_mode_changed(self, key: str, value) -> None:
        """Live react to docking.mode changes to start/stop cycle loop."""
        try:
            prev_mode = self._docking_mode
            mode = str(value or "normal").lower()
            self._docking_mode = mode
            
            if mode == "cycle":
                if self._is_active:
                    self._start_cycle_mode()
            else:
                # Switching to Normal mode
                self._stop_cycle_mode()
                
                # Reinitialize Normal mode assignments from current overlay states
                # This captures what's currently displayed as the sticky assignments
                if self._is_active and prev_mode == "cycle":
                    try:
                        if self._main_overlay:
                            main_hwnd = self._main_overlay.get_target_hwnd()
                            self._normal_mode_assignments["main"] = main_hwnd
                        
                        for i, overlay in enumerate(self._secondary_overlays):
                            overlay_id = f"secondary_{i}"
                            overlay_hwnd = overlay.get_target_hwnd()
                            self._normal_mode_assignments[overlay_id] = overlay_hwnd
                        
                        self._logger.info("Switched to Normal mode - initialized sticky assignments from current state")
                    except Exception as e:
                        self._logger.warning(f"Failed to initialize Normal mode assignments on mode switch: {e}")
        except Exception as e:
            self._logger.debug(f"Failed to handle docking.mode change: {e}")

    def _on_overlay_count_changed(self, key: str, value) -> None:
        """Live react to overlay count changes by rebuilding secondaries when active."""
        try:
            try:
                new_count = max(2, min(5, int(value)))
            except Exception:
                new_count = 3
            if not self._is_active:
                self._overlay_count = new_count
            if not self._is_active or new_count < 2 or new_count > 5:
                return
            # Schedule rebuild on UI thread shortly to avoid reentrancy
            from core.threading import ThreadManager
            from PySide6.QtCore import QCoreApplication
            if QCoreApplication.instance() is not None:
                ThreadManager.single_shot(10, lambda: self._recreate_for_overlay_count_change(new_count))
            else:
                self._logger.debug("Skipping overlay_count change (app not initialized)")
        except Exception as e:
            self._logger.debug(f"Failed to handle overlay_count change: {e}")

    def _recreate_for_overlay_count_change(self, new_count: int) -> None:
        """Recreate docking system to apply new overlay count, preserving assignments when possible."""
        try:
            if not self._is_active or new_count < 2 or new_count > 5:
                return
            # Collect current assignments in order: main, secondaries
            targets: list[int] = []
            def _append(hwnd: int):
                try:
                    if hwnd and hwnd not in targets and _is_valid_window(hwnd):
                        targets.append(int(hwnd))
                except Exception:
                    pass

            # Main
            try:
                _append(getattr(self._main_overlay, '_target_hwnd', None))
            except Exception:
                pass
            # Secondaries
            for ov in self._secondary_overlays:
                try:
                    _append(getattr(ov, '_target_hwnd', None))
                except Exception:
                    pass

            # Pad from MRU manager
            try:
                mru = self._mru_manager or get_mru_manager()
                for h in (mru.get_recent(limit=10) if mru else []):
                    if len(targets) >= new_count:
                        break
                    _append(h)
            except Exception:
                pass

            # If still short, attempt to enumerate windows
            if len(targets) < new_count:
                try:
                    from core.window.enumerator import WindowEnumerator
                    windows = WindowEnumerator.enum_windows()
                    for hwnd, title in windows:
                        if len(targets) >= new_count:
                            break
                        _append(hwnd)
                except Exception:
                    pass

            # Clamp to needed
            targets = targets[:new_count]

            # Persist geometry before teardown will already be handled in destroy_docking_system
            self.destroy_docking_system()
            # Ensure updated count is stored
            try:
                get_settings_manager().set("docking.overlay_count", new_count)
            except Exception:
                pass
            self.create_docking_system(targets if targets else [])
        except Exception as e:
            self._logger.error(f"Failed to recreate docking system for overlay_count change: {e}")

    def _fade_in_overlays_on_creation(self) -> None:
        """DWM-style fade-in animation for all overlays on creation to prevent flicker."""
        try:
            # Get target opacity from opacity manager or use default
            target_opacity = 1.0
            try:
                from core.opacity.manager import get_opacity_manager
                opacity_mgr = get_opacity_manager()
                if opacity_mgr:
                    # Get opacity as percentage and convert to float
                    opacity_percent = opacity_mgr.get_opacity()
                    target_opacity = opacity_percent / 100.0
            except Exception as e:
                suppress_debug_log(self._logger, f"Failed to get opacity from manager: {e}", "DockingManager")
            
            # Start all overlays at 0 opacity
            overlays_to_fade = []
            if self._main_overlay:
                self._main_overlay.set_opacity(0.0)
                overlays_to_fade.append(self._main_overlay)
            
            for overlay in self._secondary_overlays:
                if overlay and overlay.is_initialized():
                    overlay.set_opacity(0.0)
                    overlays_to_fade.append(overlay)
            
            # Fade in all overlays (1000ms like DWM overlay creation)
            self._fade_in_overlays(overlays_to_fade, target_opacity, duration_ms=1000)
            
        except Exception as e:
            self._logger.error(f"Failed to fade in overlays on creation: {e}")
    
    def _fade_in_overlays(self, overlays: list, target_opacity: float, duration_ms: int = 1000) -> None:
        """Fade in multiple overlays simultaneously."""
        try:
            if not overlays:
                return
                
            # Calculate steps for smooth animation (60 FPS)
            frame_time = 16  # ~60 FPS
            max_steps = max(1, duration_ms // frame_time)
            
            def fade_step(step: int):
                if step >= max_steps:
                    # Ensure final opacity is set
                    for overlay in overlays:
                        if overlay:
                            overlay.set_opacity(target_opacity)
                    return
                    
                # Smooth easing curve (ease-out)
                progress = step / max_steps
                eased_progress = 1 - (1 - progress) ** 2
                current_opacity = target_opacity * eased_progress
                
                for overlay in overlays:
                    if overlay:
                        overlay.set_opacity(current_opacity)
                    
                # Schedule next step
                ThreadManager.single_shot(frame_time, lambda: fade_step(step + 1))
                
            # Start fade animation
            fade_step(0)
            
        except Exception as e:
            self._logger.error(f"Fade-in animation failed: {e}")

    def _set_initial_opacity(self) -> None:
        """Set proper initial opacity for all overlays (legacy method)."""
        try:
            # Get target opacity from opacity manager or use default
            target_opacity = 1.0
            try:
                from core.opacity.manager import get_opacity_manager
                opacity_mgr = get_opacity_manager()
                if opacity_mgr:
                    # Get opacity as percentage and convert to float
                    opacity_percent = opacity_mgr.get_opacity()
                    target_opacity = opacity_percent / 100.0
                    pass  # Using opacity manager value (debug suppressed)
            except Exception as e:
                suppress_debug_log(self._logger, f"Failed to get opacity from manager: {e}", "DockingManager")
            
            # Set opacity for all overlays
            if self._main_overlay:
                self._main_overlay.set_opacity(target_opacity)
                pass  # Main overlay opacity set (debug suppressed)
            
            for i, overlay in enumerate(self._secondary_overlays):
                if overlay:
                    overlay.set_opacity(target_opacity)
                    pass  # Secondary overlay opacity set (debug suppressed)
                    
        except Exception as e:
            self._logger.error(f"Failed to set initial opacity: {e}")
            # Fallback to 100% opacity
            try:
                if self._main_overlay:
                    self._main_overlay.set_opacity(1.0)
                for overlay in self._secondary_overlays:
                    if overlay:
                        overlay.set_opacity(1.0)
            except Exception:
                pass

    def _is_overlay_locked(self, overlay) -> bool:
        """Check if a given docking overlay is locked via its backend (focus indicator)."""
        try:
            if overlay is None:
                return False
            if hasattr(overlay, '_is_window_locked'):
                return bool(getattr(overlay, '_is_window_locked', False))
            # Docking mode uses its own overlay system, no DWM overlay access needed
        except Exception:
            pass
        return False

    def _initialize_mru_from_overlay_manager(self) -> None:
        """Initialize or refresh MRU list from the main OverlayManager, with fallback population.

        Ensures at least 3 items by enumerating visible windows when necessary.
        """
        # Seed centralized MRUManager with windows from overlay manager
        try:
            from core.graphics import get_overlay_manager
            overlay_manager = get_overlay_manager()
            hwnd_list = overlay_manager.get_mru_window_list(self._mru_capacity)
            # Record each window in centralized MRUManager
            for hwnd in hwnd_list:
                if hwnd:
                    self._mru_manager.record(hwnd)
            suppress_debug_log(self._logger, f"Seeded MRU with {len(self._get_current_mru())} items from overlay manager", "DockingManager")
        except Exception as e:
            self._logger.warning(f"Failed to initialize MRU from overlay manager: {e}")
        
        # Populate with visible windows if needed
        current_count = len(self._get_current_mru())
        if current_count < 3:
            suppress_debug_log(self._logger, f"MRU insufficient ({current_count}), populating with visible windows", "DockingManager")
            self._populate_mru_with_visible_windows()

    # ========================================================================
    # MRU SINGLE SOURCE OF TRUTH - Helper Methods
    # ========================================================================
    # All MRU data lives in MRUManager. DockingManager reads directly.
    # No local cache, no synchronization, no stale data.
    
    def _get_current_mru(self, limit: Optional[int] = None) -> List[int]:
        """Get current MRU list from centralized MRUManager (SINGLE SOURCE OF TRUTH).
        
        This is the ONLY way to read MRU data in DockingManager.
        Never cache MRU data locally - always read fresh from MRUManager.
        
        Args:
            limit: Maximum number of entries to return (default: self._mru_capacity)
            
        Returns:
            List of window handles in MRU order (most recent first)
        """
        if limit is None:
            limit = self._mru_capacity
        return self._mru_manager.get_recent(limit=limit) if self._mru_manager else []
    
    def _reorder_mru(self, new_order: List[int]) -> None:
        """Update MRU order in centralized MRUManager.
        
        This is used for MRU rotation (quickswitch) where we need to
        modify the MRU order directly.
        
        Args:
            new_order: New MRU order (most recent first)
        """
        if not self._mru_manager or not new_order:
            return
        
        try:
            # Clear and rebuild MRU with new order
            self._mru_manager.clear()
            for hwnd in new_order:
                # record() will dedupe and validate automatically
                self._mru_manager.record(hwnd)
        except Exception as e:
            self._logger.error(f"Failed to reorder MRU: {e}", exc_info=True)
    
    # ========================================================================
    
    def _dedupe_preserve_order(self, seq: list[int]) -> list[int]:
        """Return a new list with duplicates removed, preserving the first occurrence order."""
        seen = set()
        out = []
        try:
            for x in seq or []:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
        except Exception:
            pass
        return out

    def _populate_mru_with_visible_windows(self, min_unique: int = 3) -> None:
        """Enumerate visible top-level windows and populate MRU until we have at least min_unique unique items.

        Test-friendly: uses win32gui.EnumWindows(cb, container) with a simple Python callback and a
        container list, and ctypes.windll.user32.IsWindowVisible for visibility checks. Relies on
        utils.window_validation.is_valid_window to validate candidates. Silently skips errors.
        """
        try:
            import win32gui  # type: ignore
            import ctypes  # type: ignore
            from utils.window_validation import is_valid_window  # type: ignore
        except Exception:
            return
        try:
            windows: list[int] = []
            def enum_windows_callback(hwnd, container):
                try:
                    if ctypes.windll.user32.IsWindowVisible(hwnd):
                        if is_valid_window(int(hwnd)):
                            container.append(int(hwnd))
                except Exception:
                    pass
                return True
            try:
                win32gui.EnumWindows(enum_windows_callback, windows)
            except Exception:
                windows = []
            if not windows:
                return
            # Record unique unseen windows in MRUManager until we have at least min_unique
            current_mru = self._get_current_mru()
            seen = set(current_mru)
            for h in windows:
                # Check current count after each addition
                if len(self._get_current_mru()) >= int(min_unique):
                    break
                if h not in seen:
                    self._mru_manager.record(h)
                    seen.add(h)
        except Exception:
            pass

    def _get_screen_geometry_for_main(self, main_rect) -> 'QRect':
        """Pick the full screen geometry containing the main overlay (ignores taskbar).

        We intentionally use full screen geometry here so secondary overlays align perfectly
        with the main overlay even when the taskbar is auto-hidden or not present. Clamping
        is done against the physical screen bounds to avoid false offsets.
        """
        try:
            from PySide6.QtCore import QPoint
            # Use the center of the main rect to select screen
            center = QPoint(int(main_rect.x() + main_rect.width() / 2), int(main_rect.y() + main_rect.height() / 2))
            screen = QGuiApplication.screenAt(center)
            if screen:
                return screen.geometry()
        except Exception as e:
            suppress_debug_log(self._logger, f"screenAt failed, falling back to primary: {e}", "DockingManager")
        return self._get_screen_geometry()

    def _calculate_horizontal_left_position(self, main_rect, width: int, height: int, index: int):
        """Calculate horizontal placement to the LEFT of main overlay, bottom-align aware."""
        from PySide6.QtCore import QPoint
        screen_geometry = self._get_screen_geometry_for_main(main_rect)
        near_bottom = (main_rect.y() + main_rect.height()) >= (screen_geometry.y() + screen_geometry.height() - 100)
        if index == 0:
            x = main_rect.x() - width
            y = (main_rect.y() + main_rect.height() - height) if near_bottom else main_rect.y()
        else:
            # Second secondary sits to the left of the first secondary
            first_width = int(main_rect.width() * 0.7)
            x = main_rect.x() - first_width - width
            y = (main_rect.y() + main_rect.height() - height) if near_bottom else main_rect.y()
        return QPoint(x, y)

    def _on_main_overlay_moved(self) -> None:
        """Handle main overlay movement - immediately sync all secondary overlays."""
        if self._is_active and self._bound_as_unit:
            self.sync_overlay_properties()

    def _hide_secondary_overlays_for_drag(self) -> None:
        """No-op: secondary overlays must remain visible; opacity is controlled by the Opacity Manager only."""
        suppress_debug_log(self._logger, "Drag-hide suppressed: secondary overlays remain visible by policy", "DockingManager")

    def _show_secondary_overlays_after_drag(self) -> None:
        """No-op: since we do not hide during drag, only ensure a geometry re-sync occurs."""
        try:
            if hasattr(self, '_drag_in_progress'):
                delattr(self, '_drag_in_progress')
            if hasattr(self, '_last_sync_state'):
                try:
                    delattr(self, '_last_sync_state')
                except Exception:
                    pass
            self.sync_overlay_properties()
            suppress_debug_log(self._logger, "Post-drag: ensured geometry re-sync without opacity changes", "DockingManager")
        except Exception as e:
            self._logger.warning(f"Post-drag sync failed: {e}")

    def sync_overlay_properties(self) -> None:
        """Synchronize size and position of secondary overlays relative to main overlay.

        - Does not modify opacity (opacity is handled by hotkey/opacity manager).
        - Uses snapshot-based positioning for horizontal flush alignment.
        - Applies geometry directly to overlays with UI thread execution.
        """
        # Debug spam suppression for sync calls
        if not self._is_active or not self._main_overlay:
            self._logger.error(f"Sync aborted: active={self._is_active}, main_overlay={self._main_overlay is not None}")
            return
        try:
            # Enter batch apply mode to prevent recursive Move/Resize reactions
            setattr(self, '_batch_applying', True)
            # Get main overlay geometry as reference point
            main_rect = None
            try:
                if self._main_overlay and self._main_overlay.is_initialized():
                    main_rect = self._main_overlay.get_geometry()
                else:
                    main_rect = None
            except Exception as e:
                suppress_debug_log(self._logger, f"Failed to get main overlay geometry: {e}", "DockingManager")
                main_rect = None
                
            if main_rect is None or main_rect.width() <= 0 or main_rect.height() <= 0:
                suppress_debug_log(self._logger, "Main overlay has no valid geometry; skipping sync", "DockingManager")
                return

            # STEP 0: Validate main overlay AR when needed (creation/restoration/swaps)
            # Skip during user-initiated resizes to avoid interfering with wheel/handle resizing
            try:
                # Get current main overlay source HWND
                current_main_hwnd = None
                try:
                    if hasattr(self._main_overlay, 'get_source_hwnd'):
                        current_main_hwnd = self._main_overlay.get_source_hwnd()
                except Exception:
                    pass
                
                # Only validate AR if:
                # 1. Not currently being resized by user, AND
                # 2. HWND changed (new window swapped in) OR never validated before
                should_validate = (
                    not getattr(self, '_user_resize_in_progress', False) and
                    (current_main_hwnd != self._last_validated_main_hwnd or self._last_validated_main_hwnd is None)
                )
                
                if should_validate:
                    cached_main_ar = self._main_overlay.get_cached_source_aspect()
                    if cached_main_ar and cached_main_ar > 0:
                        # Get canvas insets for main overlay
                        main_ix = main_iy = 0
                        try:
                            if hasattr(self._main_overlay, '_dwm_overlay') and self._main_overlay._dwm_overlay:
                                canvas = getattr(self._main_overlay._dwm_overlay, '_canvas', None)
                                if canvas and hasattr(canvas, 'get_content_insets'):
                                    cx, cy = canvas.get_content_insets()
                                    main_ix = max(0, int(cx))
                                    main_iy = max(0, int(cy))
                        except Exception:
                            main_ix = main_iy = 0
                        
                        # Check if current geometry matches AR
                        current_inner_w = max(1, main_rect.width() - 2 * main_ix)
                        current_inner_h = max(1, main_rect.height() - 2 * main_iy)
                        current_ar = current_inner_w / current_inner_h
                        
                        # If AR mismatch > 5%, resize main to match cached AR
                        if abs(current_ar - cached_main_ar) / cached_main_ar > 0.05:
                            # Keep width, adjust height to match AR
                            target_inner_h = max(1, int(current_inner_w / cached_main_ar))
                            target_outer_h = target_inner_h + 2 * main_iy
                            
                            self._logger.info(f"SIZING: main AR mismatch detected: current={current_ar:.3f}, cached={cached_main_ar:.3f}, resizing {main_rect.width()}x{main_rect.height()} -> {main_rect.width()}x{target_outer_h}")
                            
                            # Resize main overlay (preserves position)
                            self._main_overlay.set_geometry(main_rect.x(), main_rect.y(), main_rect.width(), target_outer_h)
                            
                            # Refresh main_rect for secondary calculations
                            main_rect = self._main_overlay.get_geometry()
                        
                        # Mark this HWND as validated
                        self._last_validated_main_hwnd = current_main_hwnd
            except Exception as e:
                self._logger.debug(f"Main AR validation failed: {e}")

            # DOCKING SINGLE-PASS SNAPSHOT SIZING/POSITIONING
            # Strictly decreasing ratios to enforce hierarchy: A > B > C > D > E
            # B starts at ~70% of A; each subsequent secondary decays by ~15%
            sec_count = len(self._secondary_overlays)
            if sec_count <= 0:
                size_ratios = [1.0]
            else:
                decay = 0.85
                size_ratios = [1.0] + [max(0.1, 0.7 * (decay ** i)) for i in range(sec_count)]
            spacing = 2  # Increased spacing for better visual separation
            
            # Determine alignment based on nearest corner of the MAIN overlay (adaptive top/bottom)
            screen_geometry = self._get_screen_geometry_for_main(main_rect)
            try:
                state = nearest_corner_state_from_rect(main_rect)
                corner = state.get("corner") if isinstance(state, dict) else None
                align_bottom = bool(corner) and str(corner).startswith("bottom")
            except Exception:
                # Fallback: compare centers if state resolution fails
                try:
                    align_bottom = (main_rect.center().y() >= (screen_geometry.y() + screen_geometry.height() // 2))
                except Exception:
                    align_bottom = True

            # Allow dragging anywhere within the screen bounds (do NOT clamp to work-area/taskbar)
            def aligned_y(h: int, overlay_index: int = -1) -> int:
                """Calculate Y position for secondary overlay aligned to main.
                
                Secondaries rely on Z-order enforcement to stay visible above taskbar,
                just like the main overlay. No artificial work-area clamping.
                """
                main_top = main_rect.y()
                main_bottom = main_rect.y() + main_rect.height()
                proposed_y = (main_bottom - h) if align_bottom else main_top
                
                # Simple screen bounds check (not taskbar/work area)
                top_edge = screen_geometry.y()
                bottom_edge = screen_geometry.y() + screen_geometry.height()
                
                if proposed_y < top_edge:
                    proposed_y = top_edge
                if proposed_y + h > bottom_edge:
                    proposed_y = bottom_edge - h
                
                return proposed_y
            
            # STEP 1: Compute sizes snapshot using cached backend AR
            sec_sizes: list[tuple[int, int]] = []  # (w,h)
            sec_meta: list[dict] = []  # per-index meta: {'ix':int,'iy':int,'ar':Optional[float]}
            for i, overlay in enumerate(self._secondary_overlays):
                if not overlay or not overlay.is_initialized():
                    sec_sizes.append((0, 0))
                    sec_meta.append({'ix': 0, 'iy': 0, 'ar': None})
                    continue
                    
                ratio = size_ratios[i + 1] if (i + 1) < len(size_ratios) else 0.5
                sec_h = max(1, int(main_rect.height() * float(ratio)))

                # Best-effort: read canvas insets so OUTER geometry hugs content tightly
                ix = iy = 0
                try:
                    if hasattr(overlay, '_dwm_overlay') and overlay._dwm_overlay:
                        canvas = getattr(overlay._dwm_overlay, '_canvas', None)
                        if canvas and hasattr(canvas, 'get_content_insets'):
                            cx, cy = canvas.get_content_insets()
                            ix = max(0, int(cx))
                            iy = max(0, int(cy))
                except Exception:
                    ix = iy = 0
                
                # Use cached AR from backend (client-area based; avoids timing issues)
                cached_ar = overlay.get_cached_source_aspect()
                # Clamp to sane bounds just like the DWM backend does, to prevent
                # ultra-wide/tall sources (e.g., title-bar rects) from exploding geometry
                if cached_ar is not None:
                    try:
                        # Validate AR is a finite number
                        if not isinstance(cached_ar, (int, float)) or not (0 < float(cached_ar) < float('inf')):
                            self._logger.warning(f"SIZING: sec={i} invalid cached_ar={cached_ar}, ignoring")
                            cached_ar = None
                        elif not (0.2 <= float(cached_ar) <= 5.0):
                            original_ar = cached_ar
                            cached_ar = max(0.2, min(5.0, float(cached_ar)))
                            self._logger.debug(f"SIZING: sec={i} AR clamped from {original_ar:.3f} to {cached_ar:.3f}")
                    except Exception as e:
                        self._logger.debug(f"SIZING: sec={i} AR validation failed: {e}")
                        cached_ar = None
                if cached_ar and cached_ar > 0:
                    # Compute OUTER width AND height so INNER (after insets) exactly matches AR
                    # inner_h = (sec_h - 2*iy), inner_w = inner_h * AR
                    # outer_w = inner_w + 2*ix, outer_h = sec_h + 2*iy (symmetric inset handling)
                    inner_h = max(1, int(sec_h - 2 * iy))
                    inner_w = max(1, int(round(inner_h * float(cached_ar))))
                    
                    # DEFENSIVE: Clamp computed width to prevent excessive values from malformed ARs
                    # Maximum reasonable width is 3x the main overlay width (even for ultra-wide sources)
                    max_reasonable_w = main_rect.width() * 3
                    if inner_w > max_reasonable_w:
                        self._logger.warning(f"SIZING: sec={i} inner_w={inner_w} exceeds max={max_reasonable_w}, clamping (AR={cached_ar:.3f})")
                        inner_w = int(max_reasonable_w)
                    
                    sec_w = max(1, int(inner_w + 2 * ix))
                    sec_h = max(1, int(sec_h + 2 * iy))  # Add canvas insets to height (was missing!)
                    ar_source = f"cached={cached_ar:.3f} (insets {ix},{iy})"
                else:
                    # Three-tier AR fallback:
                    # 1. Valid cached AR (already tried above)
                    # 2. Display AR where window is located (try here)
                    # 3. DEFAULT_ASPECT 16:9 (final fallback)
                    
                    from utils.window.overlay_constants import DEFAULT_ASPECT
                    fallback_ar = None
                    ar_source = None
                    
                    # Try to get monitor AR for the window this overlay is showing
                    try:
                        overlay_hwnd = overlay.get_target_hwnd()
                        if overlay_hwnd:
                            import ctypes
                            
                            # Get monitor handle for this window
                            user32 = ctypes.WinDLL('user32')
                            MONITOR_DEFAULTTONEAREST = 0x00000002
                            monitor_handle = user32.MonitorFromWindow(int(overlay_hwnd), MONITOR_DEFAULTTONEAREST)
                            
                            if monitor_handle:
                                # Get the QScreen for this monitor to use logical dimensions (DPI-independent)
                                from PySide6.QtGui import QGuiApplication
                                screens = QGuiApplication.screens()
                                
                                # Find matching screen by checking if window center is on it
                                try:
                                    import win32gui
                                    wrect = win32gui.GetWindowRect(int(overlay_hwnd))
                                    center_x = (wrect[0] + wrect[2]) // 2
                                    center_y = (wrect[1] + wrect[3]) // 2
                                    
                                    for screen in screens:
                                        screen_geo = screen.geometry()
                                        if screen_geo.contains(center_x, center_y):
                                            # Use LOGICAL dimensions (DPI-independent)
                                            logical_w = screen_geo.width()
                                            logical_h = screen_geo.height()
                                            
                                            if logical_w > 0 and logical_h > 0:
                                                display_ar = logical_w / logical_h
                                                
                                                # Validate display AR is within real-world display bounds
                                                # Wider than window AR bounds (0.4-3.0) because displays are manufactured products
                                                # Min 0.3: Extreme portrait (1080x3840, theoretical but possible)
                                                # Max 4.0: Super-ultrawide 32:9 displays like Samsung Odyssey G9 (AR ~3.556)
                                                if 0.3 <= display_ar <= 4.0:
                                                    fallback_ar = display_ar
                                                    ar_source = f"display={display_ar:.3f} (monitor logical)"
                                                    self._logger.debug(f"SIZING: sec={i} using display AR {display_ar:.3f} from monitor {screen.name()}")
                                                    break
                                                else:
                                                    self._logger.debug(f"SIZING: sec={i} rejecting display AR {display_ar:.3f} (out of bounds 0.3-4.0)")
                                            break
                                except Exception as e:
                                    self._logger.debug(f"SIZING: sec={i} failed to get window center: {e}")
                    except Exception as e:
                        self._logger.debug(f"SIZING: sec={i} failed to get monitor AR: {e}")
                    
                    # Final fallback to 16:9 if display AR wasn't usable
                    if fallback_ar is None:
                        fallback_ar = DEFAULT_ASPECT[0] / DEFAULT_ASPECT[1]  # 16:9 = 1.777...
                        ar_source = f"default={fallback_ar:.3f} (16:9 fallback)"
                    
                    # Compute inner dimensions using fallback AR, then add insets
                    inner_h = max(1, int(sec_h - 2 * iy))
                    inner_w = max(1, int(round(inner_h * fallback_ar)))
                    
                    # Add canvas insets to get outer dimensions
                    sec_w = max(1, int(inner_w + 2 * ix))
                    sec_h = max(1, int(sec_h + 2 * iy))
                    # ar_source already set above
                
                sec_sizes.append((sec_w, sec_h))
                sec_meta.append({'ix': ix, 'iy': iy, 'ar': (float(cached_ar) if cached_ar and cached_ar > 0 else None)})
                # Verbose sizing logs behind settings flag or forced once
                if getattr(self, '_docking_verbose', False) or getattr(self, '_force_log_next_sync', False):
                    self._logger.info(f"SIZING: sec={i} ar={ar_source} size={sec_w}x{sec_h}")
            
            # STEP 2: Compute positions using screen-aware inward/outward positioning with fit-scaling (min-size aware)
            positions = []

            # Filter valid secondary sizes and count; preserve indices for per-role floors
            valid_items = [(i, (w, h)) for i, (w, h) in enumerate(sec_sizes) if w > 0 and h > 0]
            valid_sec = [wh for _i, wh in valid_items]
            n_valid = len(valid_sec)
            spacing_total = max(0, (n_valid - 1) * spacing)
            nominal_total_width = sum(w for w, _ in valid_sec) + spacing_total

            # Pre-compute per-secondary width floors proportional to role ratio
            floors_w: list[int] = []
            for i, (w, h) in valid_items:
                role_ratio = size_ratios[i + 1] if (i + 1) < len(size_ratios) else 0.4
                floors_w.append(max(1, int(round(OVERLAY_MIN_WIDTH * float(role_ratio)))))

            # Screen edges and available horizontal space on each side
            screen_left_edge = screen_geometry.x()
            screen_right_edge = screen_geometry.x() + screen_geometry.width()
            available_right = max(0, screen_right_edge - (main_rect.x() + main_rect.width()) - spacing)
            available_left = max(0, (main_rect.x() - screen_left_edge) - spacing)

            # Decide side and compute scale if needed (accounting for min width floors)
            def total_width_for_scale(scale: float) -> int:
                tw = 0
                for idx, (w, _h) in enumerate(valid_sec):
                    floor_w = floors_w[idx] if idx < len(floors_w) else OVERLAY_MIN_WIDTH
                    tw += max(floor_w, int(round(w * scale)))
                return tw + spacing_total

            # Side selection preference: pick the side with more available space if neither fits at scale=1
            if nominal_total_width <= available_right:
                position_inward = False
                fit_scale = 1.0
            elif nominal_total_width <= available_left:
                position_inward = True
                fit_scale = 1.0
            else:
                # Choose max available side and binary-search scale including min width floors
                position_inward = available_left > available_right
                available = max(available_left, available_right)
                lo, hi = 0.1, 1.0
                fit_scale = lo
                for _ in range(14):
                    mid = (lo + hi) / 2.0
                    if total_width_for_scale(mid) <= available:
                        fit_scale = mid
                        lo = mid
                    else:
                        hi = mid
                # Clamp to a reasonable floor
                fit_scale = max(0.1, min(1.0, fit_scale))
            if getattr(self, '_docking_verbose', False) or getattr(self, '_force_log_next_sync', False):
                self._logger.info(f"FIT: inward={position_inward} availL={available_left} availR={available_right} nominal={nominal_total_width} scale={fit_scale:.3f}")

            # Apply proportional scaling to maintain AR and fit within available space
            sec_sizes_use: list[tuple[int, int]] = []
            for i, (w, h) in enumerate(sec_sizes):
                if w > 0 and h > 0:
                    # Enforce per-role minimum floors to allow secondaries to shrink more than main
                    role_ratio = size_ratios[i + 1] if (i + 1) < len(size_ratios) else 0.4
                    floor_w = max(1, int(round(OVERLAY_MIN_WIDTH * float(role_ratio))))
                    floor_h = max(1, int(round(OVERLAY_MIN_HEIGHT * float(role_ratio))))
                    sw = max(floor_w, int(round(w * fit_scale)))
                    sh = max(floor_h, int(round(h * fit_scale)))
                    sec_sizes_use.append((sw, sh))
                else:
                    sec_sizes_use.append((0, 0))

            # Enforce strict A>B>C>D>E hierarchy on heights post-scale; recalc widths to preserve AR
            try:
                prev_h = None
                for idx, (w, h) in enumerate(sec_sizes_use):
                    if w <= 0 or h <= 0:
                        prev_h = h
                        continue
                    # Compute per-role minimum height floor
                    role_ratio = size_ratios[idx + 1] if (idx + 1) < len(size_ratios) else 0.4
                    floor_h = max(1, int(round(OVERLAY_MIN_HEIGHT * float(role_ratio))))
                    if prev_h is not None and h >= prev_h:
                        new_h = max(floor_h, prev_h - 1)
                        if new_h != h:
                            # Recompute width based on cached AR and insets if available
                            meta = sec_meta[idx] if idx < len(sec_meta) else {'ix': 0, 'iy': 0, 'ar': None}
                            ix = int(meta.get('ix', 0) or 0)
                            iy = int(meta.get('iy', 0) or 0)
                            arv = meta.get('ar', None)
                            if arv and arv > 0:
                                inner_h2 = max(1, int(new_h - 2 * iy))
                                inner_w2 = max(1, int(round(inner_h2 * float(arv))))
                                new_w = max(1, int(inner_w2 + 2 * ix))
                            else:
                                # Maintain proportional width scaling
                                scale_h = float(new_h) / float(h)
                                new_w = max(1, int(round(w * scale_h)))
                            sec_sizes_use[idx] = (new_w, new_h)
                            if getattr(self, '_docking_verbose', False) or getattr(self, '_force_log_next_sync', False):
                                self._logger.info(f"HIERARCHY_ENFORCE: sec={idx} adjusted to {new_w}x{new_h}")
                        h = new_h
                    prev_h = h
            except Exception as e:
                suppress_debug_log(self._logger, f"Hierarchy enforcement failed: {e}", "DockingManager")

            # Position overlays using selected side and scaled sizes with sequence-wide offset
            total_scaled_width = sum(w for w, _ in sec_sizes_use if w > 0) + spacing_total
            if position_inward:
                # LEFT of main: compute leftmost nominal X and shift right if needed
                x_cursor_nominal = main_rect.x() - spacing
                x_leftmost_nominal = x_cursor_nominal - total_scaled_width
                offset = 0
                if x_leftmost_nominal < screen_left_edge:
                    offset = screen_left_edge - x_leftmost_nominal
                for i, (sec_w, sec_h) in enumerate(sec_sizes_use):
                    if sec_w > 0 and sec_h > 0:
                        x_nominal = x_cursor_nominal - sec_w
                        x = x_nominal + offset
                        y = aligned_y(sec_h, overlay_index=i)
                        positions.append((x, y, sec_w, sec_h))
                        # Advance nominal cursor without offset to preserve spacing
                        x_cursor_nominal = x_nominal - spacing
                        if getattr(self, '_docking_verbose', False) or getattr(self, '_force_log_next_sync', False):
                            self._logger.info(f"POSITION: sec={i} x={x} y={y} w={sec_w} h={sec_h} (inward)")
                    else:
                        positions.append((0, 0, 0, 0))
            else:
                # RIGHT of main: compute rightmost nominal X and shift left if needed
                x_start_nominal = main_rect.x() + main_rect.width() + spacing
                x_rightmost_nominal = x_start_nominal + total_scaled_width
                overflow = x_rightmost_nominal - screen_right_edge
                offset = -overflow if overflow > 0 else 0
                # Track nominal cursor separately - offset shifts entire group uniformly
                x_cursor_nominal = x_start_nominal
                for i, (sec_w, sec_h) in enumerate(sec_sizes_use):
                    if sec_w > 0 and sec_h > 0:
                        x = x_cursor_nominal + offset  # Apply uniform offset to each position
                        y = aligned_y(sec_h, overlay_index=i)
                        positions.append((x, y, sec_w, sec_h))
                        x_cursor_nominal += sec_w + spacing  # Advance nominal cursor without offset
                        if getattr(self, '_docking_verbose', False) or getattr(self, '_force_log_next_sync', False):
                            self._logger.info(f"POSITION: sec={i} x={x} y={y} w={sec_w} h={sec_h} (outward)")
                    else:
                        positions.append((0, 0, 0, 0))
            
            # STEP 3: Apply geometry using the computed positions with explicit UI thread execution
            for i, overlay in enumerate(self._secondary_overlays):
                if not overlay or i >= len(positions):
                    continue
                x, y, w, h = positions[i]
                def apply_geometry(overlay=overlay, x=x, y=y, w=w, h=h, i=i):
                    try:
                        overlay.set_geometry(x, y, w, h)
                        suppress_debug_log(self._logger, f"Applied geometry to secondary {i}: x={x}, y={y}, w={w}, h={h}", "DockingManager")
                    except Exception as e:
                        self._logger.error(f"Failed to apply geometry to secondary {i}: {e}")
                # Apply immediately on UI thread
                if QCoreApplication.instance() is not None:
                    ThreadManager.single_shot(0, apply_geometry)
                else:
                    self._thread_manager.run_on_ui_thread(apply_geometry)

            # Optionally log layout direction once per reset/creation
            try:
                if getattr(self, '_force_log_next_sync', False):
                    self._logger.info(f"Sync complete: inward={position_inward}, main=({main_rect.x()},{main_rect.y()},{main_rect.width()}x{main_rect.height()})")
                    self._force_log_next_sync = False
            except Exception:
                pass
        except Exception as e:
            self._logger.error(f"sync_overlay_properties failed: {e}")
        finally:
            # Leave batch apply mode after UI tasks have a chance to run
            def _clear_batch():
                try:
                    self._batch_applying = False
                except Exception:
                    pass
            try:
                if QCoreApplication.instance() is not None:
                    ThreadManager.single_shot(12, _clear_batch)
                else:
                    self._thread_manager.run_on_ui_thread(_clear_batch)
            except Exception:
                # Best effort fallback
                _clear_batch()
            def _mark_for_next_log_on_events():
                """Mark manager to log on next creation/reset/snap events."""
                self._force_log_next_sync = True
                
            # Expose method for reset/creation events to trigger logging
            if not hasattr(self, '_mark_for_next_log'):
                self._mark_for_next_log = _mark_for_next_log_on_events

            # Suppressed verbose sync completion log to reduce spam


    def _setup_overlay_synchronization(self) -> None:
        """Initialize synchronization behavior flags (no Qt timers)."""
        # Enable drag-hide behavior for smoother dragging
        self._hide_secondaries_during_drag = True

    def _on_docking_verbose_changed(self, key: str, value) -> None:
        """Live toggle for verbose docking logs via settings."""
        try:
            self._docking_verbose = bool(value)
        except Exception:
            self._docking_verbose = False

    def _setup_hotkey_integration(self) -> None:
        """Integrate docking manager with global hotkey system."""
        try:
            # Connect to opacity manager signals for hotkey events
            from core.opacity.manager import OpacityManager
            opacity_manager = OpacityManager()
            
            # Connect opacity change signals to affect all docked overlays
            opacity_manager.opacityChanged.connect(self._on_global_opacity_change)
            
            suppress_debug_log(self._logger, "Integrated docking manager with hotkey system", "DockingManager")
        except Exception as e:
            self._logger.warning(f"Failed to integrate with hotkey system: {e}")

    def handle_overlay_interaction(self, overlay_id: str, interaction_type: str) -> None:
        """Handle overlay interactions like quickswitch in docking mode."""
        try:
            if not self._is_active:
                suppress_debug_log(self._logger, f"Docking mode not active, ignoring {interaction_type} on {overlay_id}", "DockingManager")
                return
            
            if interaction_type == "quickswitch":
                # Handle locally to avoid recursion with QuickSwitchController's docking path
                # Map to the same behavior as a main overlay double-click
                self._handle_double_click("main")
            elif interaction_type == "double_click":
                self._handle_double_click(overlay_id)
            else:
                self._logger.debug(f"Unknown interaction type: {interaction_type}")
                
        except Exception as e:
            self._logger.error(f"Error handling overlay interaction {interaction_type}: {e}", exc_info=True)

    def _on_global_opacity_change(self, opacity_percent: int) -> None:
        """Handle global opacity changes from hotkeys."""
        try:
            if not self._is_active:
                return
                
            opacity_value = opacity_percent / 100.0
            
            # Apply opacity to all docked overlays
            if self._main_overlay:
                self._main_overlay.set_opacity(opacity_value)
                
            for overlay in self._secondary_overlays:
                if overlay and overlay.is_initialized():
                    # Skip if overlay is hidden during drag
                    if not getattr(overlay, '_drag_hidden', False):
                        overlay.set_opacity(opacity_value)
                        
            self._logger.debug(f"Applied global opacity {opacity_percent}% to all docked overlays")
        except Exception as e:
            self._logger.warning(f"Failed to apply global opacity to docked overlays: {e}")

    def _setup_periodic_sync(self) -> None:
        """Set up periodic synchronization as fallback for overlay binding."""
        # Disable periodic sync to prevent constant flickering and debug spam
        # Use signal-based synchronization only
        self._logger.debug("Periodic sync disabled - using signal-based synchronization only")

    def _handle_double_click(self, overlay_id: str) -> None:
        """Handle double-click on specific overlay.
        
        Normal mode: Swap overlay content with foreground (foreground → overlay, overlay → foreground)
        Cycle mode: Focus overlay's window, all overlays update based on MRU
        """
        # Respect global lock and per-overlay individual lock
        try:
            overlay = self._main_overlay if overlay_id == "main" else self._get_overlay_by_id(overlay_id)
        except Exception:
            overlay = None
        if self._overlays_locked or self._is_overlay_locked(overlay):
            # When locked, just bring window to focus without MRU changes
            self._bring_overlay_window_to_focus(overlay_id)
            return
            
        # Handle based on mode
        if self.is_cycle_mode():
            # Cycle mode: Focus the window currently displayed in the overlay
            # This will make it foreground and exclude it from overlay display
            # All overlays will update to show next N items from MRU
            try:
                if overlay_id == "main" and self._main_overlay:
                    target = getattr(self._main_overlay, 'get_target_hwnd', lambda: None)()
                    if target:
                        self._focus_and_promote_to_mru(int(target))
                elif overlay_id.startswith("secondary_"):
                    sec = self._get_overlay_by_id(overlay_id)
                    if sec is not None:
                        target = getattr(sec, 'get_target_hwnd', lambda: None)()
                        if target:
                            self._focus_and_promote_to_mru(int(target))
            except Exception as e:
                self._logger.debug(f"Cycle mode double-click failed: {e}")
        else:
            # Normal mode: Swap clicked overlay with foreground
            # This is a targeted operation - only affects the clicked overlay
            if overlay_id == "main":
                # Main overlay: Use standard quickswitch rotation
                # This updates main overlay to show next MRU item
                self._rotate_mru_forward()
            else:
                # Secondary overlay: Swap with foreground
                # Clicked overlay gets foreground, its old content goes to foreground
                self._normal_mode_swap_with_foreground(overlay_id)

    def _rotate_mru_forward(self) -> None:
        """Rotate MRU list forward and update main overlay (Normal) or all overlays (Cycle).
        
        Normal mode: Rotates MRU and updates ONLY main overlay assignment
        Cycle mode: Rotates MRU and updates all overlays via _focus_and_promote_to_mru
        
        This implements the classic Alt+Tab behavior:
        1. Find the current foreground window
        2. If it's MRU[0], switch to MRU[1] and move MRU[0] to position 1
        3. Focus the new MRU[0]
        4. Update overlay(s) based on mode
        """
        # Get current MRU from centralized manager (SINGLE SOURCE OF TRUTH)
        current_mru = self._get_current_mru()
        
        if len(current_mru) < 2:
            self._mru_logger.debug("QUICKSWITCH: Not enough windows in MRU for rotation")
            return
            
        current_focus = self._get_current_focus_hwnd()
        
        # Get all windows currently displayed in overlays
        displayed_windows = set()
        if self._main_overlay:
            main_hwnd = self._main_overlay.get_target_hwnd()
            if main_hwnd:
                displayed_windows.add(int(main_hwnd))
        for sec_overlay in self._secondary_overlays:
            sec_hwnd = sec_overlay.get_target_hwnd()
            if sec_hwnd:
                displayed_windows.add(int(sec_hwnd))
        
        # Standard MRU rotation logic with displayed window filtering
        if current_focus and len(current_mru) >= 2 and current_focus == current_mru[0]:
            # Rotate: MRU[1] becomes new MRU[0], old MRU[0] moves to position 1
            new_order = [current_mru[1], current_mru[0]] + current_mru[2:]
            self._reorder_mru(new_order)
            
            # Get fresh MRU after reorder
            current_mru = self._get_current_mru()
            
            # Find first non-displayed window in MRU
            target_hwnd = None
            for mru_hwnd in current_mru:
                if int(mru_hwnd) not in displayed_windows:
                    target_hwnd = int(mru_hwnd)
                    break
            
            if not target_hwnd:
                # All MRU windows are displayed, just use MRU[0]
                target_hwnd = current_mru[0]
                self._mru_logger.debug(f"QUICKSWITCH: All MRU windows displayed, using MRU[0]: {self._pink_hwnd(target_hwnd)}")
            else:
                self._mru_logger.debug(f"QUICKSWITCH: Rotating from {self._pink_hwnd(current_focus)} to first non-displayed {self._pink_hwnd(target_hwnd)}")
        else:
            # Current focus is not MRU[0], find first non-displayed MRU window
            target_hwnd = None
            for mru_hwnd in current_mru:
                if int(mru_hwnd) not in displayed_windows:
                    target_hwnd = int(mru_hwnd)
                    break
            
            if not target_hwnd:
                # All MRU windows are displayed, just use MRU[0]
                target_hwnd = current_mru[0]
                self._mru_logger.debug(f"QUICKSWITCH: All MRU windows displayed, using MRU[0]: {self._pink_hwnd(target_hwnd)}")
            else:
                self._mru_logger.debug(f"QUICKSWITCH: Bringing first non-displayed MRU to focus: {self._pink_hwnd(target_hwnd)}")
        
        # Mode-aware overlay updates
        if self.is_cycle_mode():
            # Cycle mode: Focus and update all overlays
            self._focus_and_promote_to_mru(target_hwnd)
        else:
            # Normal mode: Simple swap logic
            # Always focus the window being swapped OUT and swap in MRU[0]
            
            # Get the window currently in main overlay (the one being swapped OUT)
            current_main_hwnd = None
            if self._main_overlay:
                current_main_hwnd = self._main_overlay.get_target_hwnd()
            
            # In Normal mode, use MRU[0] as the new window (after rotation)
            # Don't search for "first non-displayed" - that's wrong
            new_main_hwnd = current_mru[0] if current_mru else None
            # Capture if this new window is currently displayed in a secondary overlay
            dup_overlay_before = self._find_overlay_showing_window(int(new_main_hwnd)) if new_main_hwnd else None
            
            if not new_main_hwnd:
                self._mru_logger.warning("QUICKSWITCH (Normal): No valid MRU[0] after rotation")
                return
            
            # Focus the window being swapped OUT (current main overlay content)
            if current_main_hwnd:
                self._bring_hwnd_to_focus(int(current_main_hwnd))
                self._mru_logger.debug(f"QUICKSWITCH (Normal): Focusing outgoing window {self._pink_hwnd(current_main_hwnd)}")
            
            # Update MRU order to promote the new window
            self._update_mru_order_only(int(new_main_hwnd))
            
            # CRITICAL FIX: Always focus the window being swapped OUT and swap in MRU[0]
            # The old logic had separate cases for foreground location, but we simplified it:
            # - Always focus the current main overlay window (the one being swapped out)
            # - Always swap in the new MRU[0] (after rotation)
            # This is the correct behavior for Normal mode quickswitch
            
            # Swap main overlay to show new MRU[0]
            self._assign_overlay("main", int(new_main_hwnd), update_tracking=True)
            self._mru_logger.debug(f"QUICKSWITCH (Normal): Main overlay swapped from {self._pink_hwnd(current_main_hwnd) if current_main_hwnd else 'None'} to {self._pink_hwnd(new_main_hwnd)}")

            # Duplication-aware swap: if the new main was previously shown in a secondary overlay,
            # and that secondary is unlocked, move that secondary to show the old main content.
            if (
                dup_overlay_before
                and dup_overlay_before != "main"
                and current_main_hwnd
                and self._is_window_valid(int(current_main_hwnd))
                and not self._is_overlay_locked_by_key(dup_overlay_before)
            ):
                self._assign_overlay(dup_overlay_before, int(current_main_hwnd), update_tracking=True)
                self._mru_logger.debug(
                    f"QUICKSWITCH (Normal): Swapped contents between main and {dup_overlay_before} to avoid duplication"
                )

    def _bring_secondary_to_focus(self, secondary_index: int) -> None:
        """Bring secondary overlay's window to focus.
        
        In Normal mode: Focus the window that should be displayed in that secondary overlay.
        In Cycle mode: Focus the actual window currently displayed in that overlay.
        """
        # In Cycle mode we must focus the actual displayed secondary window,
        # not a naive MRU index mapping (which excludes foreground and shifts indices).
        if self.is_cycle_mode():
            try:
                sec_id = f"secondary_{secondary_index}"
                sec = self._get_overlay_by_id(sec_id)
                if sec is not None:
                    target_hwnd = getattr(sec, 'get_target_hwnd', lambda: None)()
                    if target_hwnd:
                        self._mru_logger.debug(f"SECONDARY_FOCUS: Cycle mode - focusing secondary_{secondary_index} target {self._pink_hwnd(target_hwnd)}")
                        self._focus_and_promote_to_mru(int(target_hwnd))
                        return
            except Exception as e:
                self._logger.debug(f"Cycle mode secondary focus failed: {e}")
                # Fall back to MRU-based mapping below
                pass
        
        # Normal mode: Use MRU index mapping
        # secondary_0 displays MRU[1], secondary_1 displays MRU[2], etc.
        current_mru = self._get_current_mru()
        mru_index = secondary_index + 1
        if mru_index < len(current_mru):
            target_hwnd = current_mru[mru_index]
            self._mru_logger.debug(f"SECONDARY_FOCUS: Normal mode - focusing secondary_{secondary_index} (MRU[{mru_index}]) = {self._pink_hwnd(target_hwnd)}")
            self._focus_and_promote_to_mru(int(target_hwnd))
        else:
            self._mru_logger.warning(f"SECONDARY_FOCUS: MRU index {mru_index} out of bounds for MRU list length {len(current_mru)}")

    def _update_overlay_displays(self) -> None:
        """Update which windows are displayed in each overlay based on mode.
        
        Normal mode: Preserve sticky assignments, only update invalid/closed windows
        Cycle mode: Dynamic assignment based on MRU order (excluding foreground)
        """
        if not self._is_active or not self._main_overlay:
            return
        
        # Don't update displays during initialization to preserve assigned windows
        if not getattr(self, '_initialization_complete', False):
            self._logger.debug("Skipping overlay display update during initialization")
            return
        
        # Mode-aware dispatch
        if self.is_cycle_mode():
            self._update_cycle_mode_displays()
        else:
            self._update_normal_mode_displays()
    
    def _update_cycle_mode_displays(self) -> None:
        """Cycle mode: Dynamic assignment based on MRU order (excluding foreground)."""
        
        try:
            # Build a unique list of candidates; ensure enough unique items
            base_needed = 1 + len(self._secondary_overlays)
            # Exclude current foreground window only in Cycle mode
            fg_hwnd = None
            if self.is_cycle_mode():
                try:
                    import win32gui  # type: ignore
                    fg_hwnd = int(win32gui.GetForegroundWindow() or 0)
                except Exception:
                    fg_hwnd = None
            desired_count = max(3, base_needed + (1 if (self.is_cycle_mode() and fg_hwnd) else 0))
            unique_mru = self._get_current_mru()
            if len(unique_mru) < desired_count:
                try:
                    self._populate_mru_with_visible_windows(desired_count)
                    unique_mru = self._get_current_mru()
                except Exception:
                    pass
            # Filter out the current foreground window if present (Cycle mode only)
            if self.is_cycle_mode() and fg_hwnd:
                unique_mru = [h for h in unique_mru if h != fg_hwnd]

            # Validate MRU candidates to ensure only real, valid top-level windows are assigned
            try:
                validated_mru = []
                for h in unique_mru:
                    try:
                        ih = int(h)
                    except Exception:
                        continue
                    try:
                        if _is_valid_window(ih):
                            validated_mru.append(ih)
                    except Exception:
                        # Best-effort: skip invalid entries
                        pass
            except Exception:
                validated_mru = [int(h) for h in unique_mru if isinstance(h, int)]
            assigned = set()
            if self.is_cycle_mode() and fg_hwnd:
                assigned.add(int(fg_hwnd))  # prevent assignment to the foreground window
            changed_overlays = []

            # PRE-POPULATE assigned with ALL locked overlay windows to prevent duplicates
            # This must happen BEFORE any assignment logic to ensure locked windows are reserved
            main_locked = self._is_overlay_locked(self._main_overlay)
            if main_locked:
                current_main = getattr(self._main_overlay, 'get_target_hwnd', lambda: None)()
                if current_main:
                    assigned.add(int(current_main))
                    self._logger.info(f"[CYCLE_LOCK] Main overlay locked; preserving HWND {current_main}")
            
            # Also reserve all locked secondary overlay windows BEFORE assignment
            locked_count = 0
            for i, overlay in enumerate(self._secondary_overlays):
                if self._is_overlay_locked(overlay):
                    current_target = overlay.get_target_hwnd()
                    if current_target:
                        assigned.add(int(current_target))
                        locked_count += 1
                        self._logger.info(f"[CYCLE_LOCK] Secondary overlay {i} locked; preserving HWND {current_target}")
            
            # Log lock state summary for visibility
            total_locked = (1 if main_locked else 0) + locked_count
            if total_locked > 0:
                self._logger.debug(f"[CYCLE_LOCK] {total_locked} overlay(s) locked, {len(assigned)} window(s) reserved")
            
            # Now proceed with assignment logic (locked windows already in 'assigned' set)
            if main_locked:
                # Locked overlay: skip assignment, it keeps its current target
                current_main = getattr(self._main_overlay, 'get_target_hwnd', lambda: None)()
                if current_main:
                    self._logger.debug(f"[CYCLE_LOCK] Skipping main overlay assignment (locked to {current_main})")
            else:
                # Main overlay uses the first valid unique item NOT already assigned (or clears if none)
                main_hwnd = None
                for hwnd in validated_mru:
                    if hwnd not in assigned:
                        main_hwnd = hwnd
                        break
                
                if main_hwnd is not None:
                    prev_hwnd = getattr(self._main_overlay, 'get_target_hwnd', lambda: None)()
                    if prev_hwnd != main_hwnd:
                        self._main_overlay.set_target_window(main_hwnd)
                        changed_overlays.append(self._main_overlay)
                    assigned.add(main_hwnd)
                    self._logger.debug(f"Set main overlay target to HWND {main_hwnd}")
                else:
                    self._main_overlay.set_target_window(None)
                    self._logger.debug("Cleared main overlay target (no available MRU not already assigned)")

            # Assign each secondary the next available unique HWND, respecting locks
            for i, overlay in enumerate(self._secondary_overlays):
                if self._is_overlay_locked(overlay):
                    # Already reserved in pre-population phase above; skip assignment
                    continue

                # Find next valid unique not yet assigned
                chosen = None
                for hwnd in validated_mru:
                    if hwnd not in assigned:
                        chosen = hwnd
                        break
                if chosen is not None:
                    prev = overlay.get_target_hwnd()
                    if prev != chosen:
                        overlay.set_target_window(chosen)
                        changed_overlays.append(overlay)
                    assigned.add(chosen)
                    self._logger.debug(f"Set secondary overlay {i} target to HWND {chosen}")
                else:
                    # Preserve existing if any, otherwise clear
                    current_target = overlay.get_target_hwnd()
                    if current_target and current_target not in assigned:
                        assigned.add(current_target)
                        self._logger.debug(f"Preserving secondary overlay {i} target HWND {current_target} (no unique MRU left)")
                    else:
                        overlay.set_target_window(None)
                        self._logger.debug(f"Cleared secondary overlay {i} target (no unique MRU left)")

            # Apply a short fade only in cycle mode and only for overlays whose window changed
            try:
                if changed_overlays and self.is_cycle_mode():
                    target_opacity = 1.0
                    try:
                        from core.opacity.manager import get_opacity_manager
                        om = get_opacity_manager()
                        if om:
                            target_opacity = max(0.0, min(1.0, float(om.get_opacity() or 100) / 100.0))
                    except Exception:
                        target_opacity = 1.0
                    # Short fade ~200ms
                    self._fade_in_overlays(changed_overlays, target_opacity=target_opacity, duration_ms=200)
            except Exception:
                pass
                    
        except Exception as e:
            self._logger.error(f"Error updating cycle mode displays: {e}")
            # Continue operation - don't fail completely on display update errors
    
    def _update_normal_mode_displays(self) -> None:
        """Normal mode: Preserve sticky assignments, only update invalid/closed windows.
        
        Overlays maintain their assigned windows until:
        1. The window becomes invalid/closed
        2. The overlay has no assignment yet (initialization)
        3. Explicit user action (quickswitch, swap, etc.)
        """
        try:
            # Get validated MRU (no foreground exclusion in Normal mode)
            validated_mru = self._get_validated_mru()
            assigned = set()  # Track which windows are assigned to prevent duplicates
            
            # Update main overlay if needed
            main_locked = self._is_overlay_locked(self._main_overlay)
            if main_locked:
                # Preserve locked overlay assignment
                current_main = self._main_overlay.get_target_hwnd()
                if current_main:
                    assigned.add(int(current_main))
                    self._logger.debug(f"Main overlay locked; preserving HWND {current_main}")
            else:
                # Check current assignment
                current_assignment = self._normal_mode_assignments.get("main")
                current_actual = self._main_overlay.get_target_hwnd()
                
                # Need new assignment if:
                # 1. No assignment tracked
                # 2. Current assignment is invalid
                # 3. Overlay shows different window than tracked (desynced)
                if (current_assignment is None or 
                    not self._is_window_valid(current_assignment) or
                    current_actual != current_assignment):
                    
                    # Find next available window
                    new_hwnd = self._get_next_available_mru(validated_mru, assigned)
                    if new_hwnd:
                        self._assign_overlay("main", new_hwnd, update_tracking=True)
                        assigned.add(new_hwnd)
                        self._logger.debug(f"NORMAL: Main overlay updated to {new_hwnd} (prev invalid/missing)")
                    else:
                        self._logger.debug("NORMAL: No valid MRU for main overlay")
                else:
                    # Keep current assignment
                    assigned.add(current_assignment)
                    # Ensure overlay actually shows what we think it should
                    if current_actual != current_assignment:
                        self._assign_overlay("main", current_assignment, update_tracking=False)
            
            # Update secondary overlays if needed
            for i, overlay in enumerate(self._secondary_overlays):
                overlay_id = f"secondary_{i}"
                
                sec_locked = self._is_overlay_locked(overlay)
                if sec_locked:
                    current_target = overlay.get_target_hwnd()
                    if current_target:
                        assigned.add(int(current_target))
                        self._logger.debug(f"Secondary {i} locked; preserving HWND {current_target}")
                    continue
                
                # Check current assignment
                current_assignment = self._normal_mode_assignments.get(overlay_id)
                current_actual = overlay.get_target_hwnd()
                
                # Need new assignment if:
                # 1. No assignment tracked
                # 2. Current assignment is invalid
                # 3. Assignment already used by another overlay
                # 4. Overlay shows different window than tracked (desynced)
                if (current_assignment is None or
                    not self._is_window_valid(current_assignment) or
                    current_assignment in assigned or
                    current_actual != current_assignment):
                    
                    # Find next available window
                    new_hwnd = self._get_next_available_mru(validated_mru, assigned)
                    if new_hwnd:
                        self._assign_overlay(overlay_id, new_hwnd, update_tracking=True)
                        assigned.add(new_hwnd)
                        self._logger.debug(f"NORMAL: Secondary {i} updated to {new_hwnd} (prev invalid/missing)")
                    else:
                        # No more windows available
                        if current_assignment and current_assignment not in assigned:
                            # Keep current if still valid and not duplicate
                            assigned.add(current_assignment)
                        else:
                            self._logger.debug(f"NORMAL: No valid MRU for secondary {i}")
                else:
                    # Keep current assignment
                    assigned.add(current_assignment)
                    # Ensure overlay actually shows what we think it should
                    if current_actual != current_assignment:
                        self._assign_overlay(overlay_id, current_assignment, update_tracking=False)
                        
        except Exception as e:
            self._logger.error(f"Error updating normal mode displays: {e}", exc_info=True)
            # Continue operation - don't fail completely on display update errors

    def _populate_mru_with_visible_windows(self, min_unique: int = 3) -> None:
        """Enumerate visible top-level windows and populate MRU until we have at least min_unique unique items.

        Best-effort: uses win32gui and utils.window_validation.is_valid_window; silently skips errors.
        Prioritizes windows on the same monitor as the main overlay (soft priority).
        """
        try:
            import win32gui
            import ctypes
            from utils.window_validation import is_valid_window
        except Exception:
            return
        try:
            # Get main overlay's monitor for prioritization
            main_monitor = None
            try:
                if self._main_overlay and hasattr(self._main_overlay, 'get_host'):
                    host = self._main_overlay.get_host()
                    if host:
                        from utils.monitor_utils import get_monitor_from_point
                        screen_geom = host.geometry()
                        center_x = screen_geom.x() + screen_geom.width() // 2
                        center_y = screen_geom.y() + screen_geom.height() // 2
                        main_monitor = get_monitor_from_point(center_x, center_y)
            except Exception:
                pass
            
            windows: list[int] = []
            def enum_windows_callback(hwnd, container):
                try:
                    if ctypes.windll.user32.IsWindowVisible(hwnd):
                        if is_valid_window(int(hwnd)):
                            container.append(int(hwnd))
                except Exception:
                    pass
                return True
            try:
                win32gui.EnumWindows(enum_windows_callback, windows)
            except Exception:
                windows = []
            if not windows:
                return
            
            # Prioritize windows on same monitor (soft priority)
            same_monitor = []
            other_monitor = []
            
            if main_monitor:
                try:
                    for hwnd in windows:
                        try:
                            rect = win32gui.GetWindowRect(hwnd)
                            win_center_x = (rect[0] + rect[2]) // 2
                            win_center_y = (rect[1] + rect[3]) // 2
                            from utils.monitor_utils import get_monitor_from_point
                            win_monitor = get_monitor_from_point(win_center_x, win_center_y)
                            if win_monitor and win_monitor.get('name') == main_monitor.get('name'):
                                same_monitor.append(hwnd)
                            else:
                                other_monitor.append(hwnd)
                        except Exception:
                            other_monitor.append(hwnd)
                except Exception:
                    # Fallback: no prioritization
                    same_monitor = windows
                    other_monitor = []
            else:
                # No main monitor detected, no prioritization
                same_monitor = windows
                other_monitor = []
            
            # Record unique unseen windows in MRU until we have at least min_unique
            # Prioritize same-monitor windows first
            current_mru = self._get_current_mru()
            seen = set(current_mru)
            for h in same_monitor + other_monitor:
                if len(self._get_current_mru()) >= int(min_unique):
                    break
                if h not in seen:
                    self._mru_manager.record(h)
                    seen.add(h)
        except Exception:
            pass

    def _setup_positioning(self) -> None:
        """Set up initial positioning for all overlays."""
        if not self._main_overlay:
            return
            
        # Position persistence is handled automatically by _apply_persisted_main_geometry()
        # which is called during overlay creation (line 376)
        
        # Position main overlay (this should be handled by existing overlay system)
        # Secondary overlays will be positioned relative to main overlay
        self.sync_overlay_properties()

    # Position persistence is handled by:
    # - _persist_main_geometry() - Saves geometry on destroy/drag  
    # - _apply_persisted_main_geometry() - Restores geometry on create
    # This is the ONLY positioning system per SST policy.
    # See audits/todo_fixme_evaluation_2025_10_06.md for historical context.

    def _apply_saved_position(self, position_data: dict = None) -> None:
        """Apply saved position - redirects to active persistence system.
        
        Note: This method exists for backward compatibility but delegates to
        the active _apply_persisted_main_geometry() method which is the SST.
        """
        try:
            # Use the active persistence system
            self._apply_persisted_main_geometry()
        except Exception as e:
            self._logger.warning(f"Failed to apply saved position: {e}")

    def _get_screen_geometry(self) -> QRect:
        """Get the geometry of the screen containing the main overlay."""
        # This is a simplified implementation - should use proper screen detection
        screen = QGuiApplication.primaryScreen()
        return screen.geometry() if screen else QRect(0, 0, 1920, 1080)

    def _get_current_focus_hwnd(self) -> Optional[int]:
        """Get the HWND of the currently focused window."""
        try:
            import win32gui  # type: ignore
            hwnd = int(win32gui.GetForegroundWindow() or 0)
            return hwnd if hwnd != 0 else None
        except Exception:
            return None

    def _focus_and_promote_to_mru(self, hwnd: int) -> None:
        """Bring hwnd to foreground and promote it to front of MRU, demoting previous focus.

        This keeps MRU consistent across docking and the global OverlayManager, and
        then refreshes overlay displays (which, in Cycle mode, will exclude the new foreground).
        """
        if not self._is_active or not hwnd:
            return
        try:
            from utils.window_validation import is_valid_window
            if not is_valid_window(int(hwnd)):
                return
        except Exception:
            pass
        try:
            # Bring window to focus
            self._bring_hwnd_to_focus(int(hwnd))
            
            # Record in centralized MRUManager (this will become MRU[0])
            # FocusTracker will also record this when it detects the focus change
            self._mru_manager.record(int(hwnd))
            
            # Refresh overlays (will exclude foreground in Cycle mode)
            self._update_overlay_displays()
            
            # Trigger sync after autoswitch to validate AR for new windows
            if getattr(self, '_initialization_complete', False):
                self._coalesced_sync(10)  # Small delay to let swap complete
        except Exception as e:
            self._logger.debug(f"focus_and_promote_to_mru failed for {hwnd}: {e}")

    def _bring_hwnd_to_focus(self, hwnd: int, attempt: int = 0) -> None:
        """Bring window to focus using proven approach that preserves window state.
        
        Based on quickswitch_controller._focus_window which has never failed.
        This approach:
            - Restores minimized windows with SW_RESTORE
            - NEVER uses SW_SHOWNORMAL (which destroys fullscreen/maximized state)
            - Uses SetWindowPos with SWP_SHOWWINDOW to ensure visibility
            - Falls back through multiple focus strategies
        """
        try:
            if not hwnd:
                return
            
            self._mru_logger.debug(f"FOCUS: Attempting to focus hwnd {self._pink_hwnd(hwnd)} (attempt {attempt})")
            
            # Suppress OverlayHost deactivation clears during handoff
            try:
                from core.graphics.overlay_host import OverlayHost
                OverlayHost.suppress_deactivation_clears(400)
            except Exception:
                pass
            import win32gui
            import win32con
            import win32api
            import win32process
            
            # Step 1: Restore if minimized (preserves fullscreen/maximized state)
            try:
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    self._mru_logger.debug(f"FOCUS: Restored minimized window {self._pink_hwnd(hwnd)}")
            except Exception:
                pass
            
            # Step 2: Try simple SetForegroundWindow first
            try:
                win32gui.SetForegroundWindow(hwnd)
                self._mru_logger.debug(f"FOCUS: SetForegroundWindow succeeded for {self._pink_hwnd(hwnd)}")
                return
            except Exception as e:
                self._mru_logger.debug(f"FOCUS: SetForegroundWindow failed: {e}")
            
            # Step 3: Try SetWindowPos sequence (preserves window state)
            try:
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
                win32gui.SetForegroundWindow(hwnd)
                self._mru_logger.debug(f"FOCUS: SetWindowPos sequence succeeded for {self._pink_hwnd(hwnd)}")
                return
            except Exception as e:
                self._mru_logger.debug(f"FOCUS: SetWindowPos sequence failed: {e}")
            
            # Step 4: Thread input attachment trick
            try:
                fg = win32gui.GetForegroundWindow()
                if fg:
                    fg_tid, _ = win32process.GetWindowThreadProcessId(fg)
                    tgt_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
                    cur_tid = win32api.GetCurrentThreadId()
                    # Attach our thread to both to allow SetForegroundWindow
                    win32api.AttachThreadInput(cur_tid, fg_tid, True)
                    win32api.AttachThreadInput(cur_tid, tgt_tid, True)
                    try:
                        win32gui.BringWindowToTop(hwnd)
                        win32gui.SetForegroundWindow(hwnd)
                        self._mru_logger.debug(f"FOCUS: Thread attachment succeeded for {self._pink_hwnd(hwnd)}")
                    finally:
                        win32api.AttachThreadInput(cur_tid, fg_tid, False)
                        win32api.AttachThreadInput(cur_tid, tgt_tid, False)
                    return
            except Exception as e:
                self._mru_logger.debug(f"FOCUS: Thread attachment failed: {e}")
            
            # Step 5: Alt keystroke trick to satisfy foreground lock
            try:
                VK_MENU = 0x12
                win32api.keybd_event(VK_MENU, 0, 0, 0)
                win32api.keybd_event(VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32gui.SetForegroundWindow(hwnd)
                self._mru_logger.debug(f"FOCUS: Alt keystroke trick succeeded for {self._pink_hwnd(hwnd)}")
                return
            except Exception as e:
                self._mru_logger.debug(f"FOCUS: Alt keystroke trick failed: {e}")

            # Verify foreground and schedule one short retry if it didn't stick
            try:
                current = int(win32gui.GetForegroundWindow() or 0)
                if current != int(hwnd):
                    self._mru_logger.warning(f"FOCUS: All strategies failed - foreground is {self._pink_hwnd(current)}, wanted {self._pink_hwnd(hwnd)}")
                    if attempt < 1:
                        self._mru_logger.debug(f"FOCUS: Scheduling retry for {self._pink_hwnd(hwnd)}")
                        ThreadManager.single_shot(80, lambda: self._bring_hwnd_to_focus(int(hwnd), attempt + 1))
                else:
                    self._mru_logger.debug(f"FOCUS: Verification passed - {self._pink_hwnd(hwnd)} is now foreground")
            except Exception as e:
                self._mru_logger.debug(f"FOCUS: Verification check failed: {e}")
        except Exception as e:
            self._mru_logger.debug(f"_bring_hwnd_to_focus failed for {self._pink_hwnd(hwnd)}: {e}")

    def _bring_overlay_window_to_focus(self, overlay_id: str) -> None:
        """Bring the window displayed in the specified overlay to focus.
        
        Used when overlays are locked - just brings the window to focus without MRU changes.
        """
        try:
            overlay = None
            if overlay_id == "main" and self._main_overlay:
                overlay = self._main_overlay
            else:
                for sec_overlay in self._secondary_overlays:
                    if sec_overlay.overlay_id == overlay_id:
                        overlay = sec_overlay
                        break
        
            if not overlay:
                self._logger.warning(f"Overlay {overlay_id} not found for focus operation")
                return
        
            target_hwnd = getattr(overlay, '_target_hwnd', None)
            if not target_hwnd:
                self._logger.warning(f"No target window for overlay {overlay_id}")
                return
        
            self._mru_logger.debug(f"LOCKED_FOCUS: Bringing overlay {overlay_id} target {self._pink_hwnd(target_hwnd)} to focus (no MRU change)")
            self._bring_hwnd_to_focus(target_hwnd)
            
        except Exception as e:
            self._logger.error(f"Error in _bring_overlay_window_to_focus: {e}")


    def update_mru_list(self, hwnd_list: List[int]) -> None:
        """Update the MRU with new window order.
        
        This method is kept for backwards compatibility with tests.
        It now seeds the centralized MRUManager instead of a local list.
        """
        # Only update MRU if we have meaningful data and initialization is complete
        if hwnd_list and len(hwnd_list) > 0 and getattr(self, '_initialization_complete', False):
            # Seed centralized MRUManager with provided windows
            for hwnd in hwnd_list:
                if hwnd:
                    self._mru_manager.record(hwnd)
            self._logger.debug(f"Updated MRU with {len(self._get_current_mru())} items")
            # Always update overlay displays - mode-specific logic handles the difference:
            # Cycle mode: _update_cycle_mode_displays() reassigns all overlays
            # Normal mode: _update_normal_mode_displays() only replaces invalid/closed windows
            self._update_overlay_displays()
            
            # Trigger sync after autoswitch/cycle swaps to validate AR for new windows
            # This ensures AR validation runs immediately after source window changes
            if getattr(self, '_initialization_complete', False):
                self._coalesced_sync(10)  # Small delay to let swap complete
        else:
            self._logger.debug(f"Skipping MRU update: hwnd_list={len(hwnd_list) if hwnd_list else 0} items, init_complete={getattr(self, '_initialization_complete', False)}")
        
        # Sync with main overlay manager only after initialization
        if getattr(self, '_initialization_complete', False):
            self._sync_with_overlay_manager()

    def _sync_with_overlay_manager(self) -> None:
        """Synchronize MRU state with the main OverlayManager.
        
        This method is kept for backwards compatibility.
        It now reads from centralized MRUManager (single source of truth).
        """
        try:
            from core.graphics import get_overlay_manager
            overlay_manager = get_overlay_manager()
            current_mru = self._get_current_mru()
            overlay_manager.update_mru_from_hwnd_list(current_mru)
        except Exception as e:
            self._logger.warning(f"Failed to sync with OverlayManager: {e}")

    # --- Public API used by docking overlays and context menu ---------------
    def swap_main_overlay_source(self, hwnd: Optional[int], record_mru: bool = True) -> None:
        """Swap the main overlay's source using DWM overlay pattern.
        
        Follows DWM overlay _handle_swap_window pattern:
        1. Queue system for swap requests
        2. Window validation
        3. Fade effect reset
        4. Source swap with thumbnail re-registration
        5. MRU recording and sync
        6. Auto-switch tracking notification
        """
        try:
            if not self._is_active or not self._main_overlay:
                self._logger.debug("swap_main_overlay_source: docking not active or main overlay missing")
                return

            if not hwnd or int(hwnd) == 0:
                self._logger.debug("swap_main_overlay_source: ignored null hwnd")
                return

            target_hwnd = int(hwnd)
            
            # Check if swap is in flight for main overlay
            if getattr(self, '_main_swap_in_flight', False):
                # Queue the request for later
                self._pending_main_swap_hwnd = target_hwnd
                self._pending_main_swap_record_mru = record_mru
                self._logger.debug(f"Main swap in flight, queuing hwnd {target_hwnd}")
                return
            
            self._main_swap_in_flight = True
            self._pending_main_swap_hwnd = None
            self._pending_main_swap_record_mru = False
            
            # Perform the swap
            success = self._swap_main_source_hwnd(target_hwnd, record_mru=record_mru)
            if not success:
                self._logger.error(f"Failed to swap main overlay to hwnd {target_hwnd}")
            else:
                # Notify overlay manager of window change for auto-switch tracking
                self._notify_main_window_change(target_hwnd)
                
        except Exception as e:
            self._logger.error(f"swap_main_overlay_source failed: {e}")
        finally:
            # Process any pending swap
            try:
                pending = getattr(self, '_pending_main_swap_hwnd', None)
                pending_record_mru = getattr(self, '_pending_main_swap_record_mru', False)
                if pending:
                    # Clear pending state
                    self._pending_main_swap_hwnd = None
                    self._pending_main_swap_record_mru = False
                    # Allow next swap to start
                    self._main_swap_in_flight = False
                    self._logger.debug(f"Dispatching pending main swap to hwnd {pending}")
                    from core.threading import ThreadManager
                    ThreadManager.single_shot(10, lambda: self.swap_main_overlay_source(pending, record_mru=pending_record_mru))
                else:
                    self._main_swap_in_flight = False
            except Exception:
                self._main_swap_in_flight = False

    def set_overlays_locked(self, locked: bool) -> None:
        """Set the overlay lock state."""
        self._overlays_locked = locked
        self._logger.debug(f"Docking overlays lock state: {locked}")

    def is_active(self) -> bool:
        """Check if the docking system is active."""
        return self._is_active

    # --- Public API used by DockingContextMenu and overlays ---
    def is_overlay_locked(self, overlay_id: str) -> bool:
        """Return lock state for a specific overlay id ("main" or "secondary_X")."""
        try:
            return bool(self._overlay_locks.get(overlay_id, False))
        except Exception:
            return False

    def set_overlay_locked(self, overlay_id: str, locked: bool) -> None:
        """Set lock state and propagate to backend if possible."""
        try:
            self._overlay_locks[overlay_id] = bool(locked)
            ov = self._get_overlay_by_id(overlay_id)
            if ov is not None:
                try:
                    # Docking mode uses its own overlay system
                    setattr(ov, '_is_window_locked', bool(locked))
                except Exception:
                    pass
        except Exception as e:
            self._logger.debug(f"set_overlay_locked failed for {overlay_id}: {e}")

    def swap_overlay_source(self, overlay_id: str, hwnd: int, record_mru: bool = True) -> None:
        """Swap the given overlay's source using DWM overlay pattern.
        
        Handles both main and secondary overlays with proper queuing system.
        """
        try:
            if not self._is_active or not hwnd or int(hwnd) == 0:
                return
                
            if overlay_id == 'main':
                self.swap_main_overlay_source(hwnd, record_mru=record_mru)
                return
                
            # Handle secondary overlay swap
            target_hwnd = int(hwnd)
            
            # Check if swap is in flight for this secondary overlay
            swap_key = f'_secondary_swap_in_flight_{overlay_id}'
            pending_key = f'_pending_secondary_swap_{overlay_id}'
            pending_mru_key = f'_pending_secondary_mru_{overlay_id}'
            
            if getattr(self, swap_key, False):
                # Queue the request for later
                setattr(self, pending_key, target_hwnd)
                setattr(self, pending_mru_key, record_mru)
                self._logger.debug(f"Secondary {overlay_id} swap in flight, queuing hwnd {target_hwnd}")
                return
            
            setattr(self, swap_key, True)
            setattr(self, pending_key, None)
            setattr(self, pending_mru_key, False)
            
            # Perform the swap
            success = self._swap_secondary_source_hwnd(overlay_id, target_hwnd, record_mru=record_mru)
            if not success:
                self._logger.error(f"Failed to swap secondary {overlay_id} to hwnd {target_hwnd}")
            else:
                # Notify overlay manager of window change for auto-switch tracking
                self._notify_secondary_window_change(overlay_id, target_hwnd)
                
        except Exception as e:
            self._logger.error(f"swap_overlay_source failed for {overlay_id}: {e}")
        finally:
            # Process any pending swap
            try:
                swap_key = f'_secondary_swap_in_flight_{overlay_id}'
                pending_key = f'_pending_secondary_swap_{overlay_id}'
                pending_mru_key = f'_pending_secondary_mru_{overlay_id}'
                
                pending = getattr(self, pending_key, None)
                pending_record_mru = getattr(self, pending_mru_key, False)
                
                if pending:
                    # Clear pending state
                    setattr(self, pending_key, None)
                    setattr(self, pending_mru_key, False)
                    # Allow next swap to start
                    setattr(self, swap_key, False)
                    self._logger.debug(f"Dispatching pending secondary {overlay_id} swap to hwnd {pending}")
                    from core.threading import ThreadManager
                    ThreadManager.single_shot(10, lambda: self.swap_overlay_source(overlay_id, pending, record_mru=pending_record_mru))
                else:
                    setattr(self, swap_key, False)
            except Exception:
                setattr(self, swap_key, False)

    def swap_primary_with_secondary(self, overlay_id: str) -> None:
        """Swap the main overlay's window with the specified secondary overlay.
        Promotes the secondary's source to main, and moves the old main to that secondary.
        """
        try:
            if not self._is_active or not self._main_overlay:
                return
            sec = self._get_overlay_by_id(overlay_id)
            if sec is None or sec is self._main_overlay:
                return
            main_hwnd = getattr(self._main_overlay, 'get_target_hwnd', lambda: None)()
            sec_hwnd = getattr(sec, 'get_target_hwnd', lambda: None)()
            if not sec_hwnd or int(sec_hwnd) == 0:
                return

            # First promote secondary's source to primary via unified path (handles AR + MRU)
            self.swap_main_overlay_source(int(sec_hwnd), record_mru=True)

            # Then move old main into this secondary overlay
            if main_hwnd and int(main_hwnd) != 0:
                sec.set_target_window(int(main_hwnd))
            else:
                sec.set_target_window(None)

            # Re-sync layout after swap
            self.sync_overlay_properties()
        except Exception as e:
            self._logger.error(f"swap_primary_with_secondary failed for {overlay_id}: {e}")

    def reset_overlay(self, overlay_id: str) -> None:
        """Reset overlay position and geometry exactly like DWM overlay reset.
        
        Docking mode reset: resets position/geometry without changing source window assignments.
        """
        try:
            if not self._is_active:
                return
                
            overlay = self._get_overlay_by_id(overlay_id)
            if not overlay:
                self._logger.warning(f"Overlay {overlay_id} not found for reset")
                return
            
            # Docking mode reset: simple position reset without DWM-specific logic
            try:
                # Reset overlay to default position/size while preserving source window
                if hasattr(overlay, 'reset_position'):
                    overlay.reset_position()
                    self._logger.info(f"Reset overlay {overlay_id} position")
                
            except Exception as e:
                self._logger.error(f"Overlay reset failed for {overlay_id}: {e}")
            
            # Force docking layout re-sync to position overlays correctly
            self._force_log_next_sync = True
            self.sync_overlay_properties()
            
            self._logger.info(f"Successfully reset overlay {overlay_id} using DWM pattern")
            
        except Exception as e:
            self._logger.error(f"reset_overlay failed for {overlay_id}: {e}")

    # --- Helpers ---
    def _get_overlay_by_id(self, overlay_id: str):
        if overlay_id == 'main':
            return self._main_overlay
        for ov in self._secondary_overlays:
            try:
                if getattr(ov, 'overlay_id', None) == overlay_id:
                    return ov
            except Exception:
                continue
        return None

    def get_overlay_count(self) -> int:
        """Get the total number of overlays in the docking system."""
        return 1 + len(self._secondary_overlays) if self._is_active else 0

    def _cleanup(self) -> None:
        """Cleanup method called by ResourceManager."""
        self.destroy_docking_system()
        
        if self._resource_manager and self._resource_id:
            try:
                self._resource_manager.unregister(self._resource_id)
            except Exception as e:
                self._logger.warning(f"Error unregistering from ResourceManager: {e}")

    def _cleanup_partial_creation(self) -> None:
        """Cleanup any overlays created so far when create_docking_system fails mid-way."""
        try:
            # Clean up secondary overlays first
            for i, overlay in enumerate(getattr(self, '_secondary_overlays', []) or []):
                try:
                    overlay.cleanup()
                except Exception as e:
                    self._logger.debug(f"Partial cleanup: failed to cleanup secondary {i}: {e}")
            # Clear list
            try:
                self._secondary_overlays.clear()
            except Exception:
                self._secondary_overlays = []

            # Clean up main overlay if present
            if getattr(self, '_main_overlay', None):
                try:
                    self._main_overlay.cleanup()
                except Exception as e:
                    self._logger.debug(f"Partial cleanup: failed to cleanup main overlay: {e}")
                self._main_overlay = None

            # Reset basic state
            self._is_active = False
            self._bound_as_unit = False
        except Exception as e:
            self._logger.debug(f"Partial creation cleanup encountered an error: {e}")

    # --- DWM-style swap implementation methods ---
    
    def _swap_main_source_hwnd(self, new_hwnd: int, record_mru: bool = True) -> bool:
        """Swap the main overlay's source window handle using DWM pattern."""
        try:
            # Validate window
            from utils.window_validation import is_valid_window
            if not is_valid_window(new_hwnd):
                self._logger.error(f"Invalid main swap target: {new_hwnd}")
                return False

            # Note: Docking mode uses its own crossfade animation instead of DWM fade logic

            # Perform the source swap
            self._main_overlay.set_target_window(new_hwnd)
            
            # Register with auto-switch manager
            self.register_overlay_for_auto_switch('main', new_hwnd)
            
            # Update MRU if requested
            if record_mru:
                self._mru_manager.record(new_hwnd)

            # Crossfade the main overlay around the source switch
            self._quick_crossfade_main(duration_ms=120)

            # Re-sync layout/opacity coupling
            self.sync_overlay_properties()
            
            self._logger.debug(f"Main overlay swapped to hwnd {new_hwnd}")
            return True
            
        except Exception as e:
            self._logger.error(f"Main source swap failed: {e}")
            return False
    
    def _quick_crossfade_main(self, duration_ms: int = 120) -> None:
        """Quick crossfade animation for main overlay during source switch."""
        try:
            if not self._main_overlay:
                return
                
            # Calculate steps for smooth animation (60 FPS)
            frame_time = 16  # ~60 FPS
            max_steps = max(1, duration_ms // frame_time)
            original_opacity = self._main_overlay.get_opacity()
            
            def fade_step(step: int):
                if step >= max_steps:
                    # Ensure final opacity is restored
                    self._main_overlay.set_opacity(original_opacity)
                    return
                    
                # Crossfade curve: fade out then in (creates brief flash effect)
                progress = step / max_steps
                if progress < 0.5:
                    # Fade out phase
                    fade_progress = 1.0 - (progress * 2)
                else:
                    # Fade in phase
                    fade_progress = (progress - 0.5) * 2
                
                current_opacity = original_opacity * fade_progress
                self._main_overlay.set_opacity(current_opacity)
                    
                # Schedule next step
                from core.threading import ThreadManager
                ThreadManager.single_shot(frame_time, lambda: fade_step(step + 1))
                
            # Start crossfade animation
            fade_step(0)
            
        except Exception as e:
            self._logger.error(f"Quick crossfade failed: {e}")
    
    def _swap_secondary_source_hwnd(self, overlay_id: str, new_hwnd: int, record_mru: bool = True) -> bool:
        """Swap a secondary overlay's source window handle using DWM pattern."""
        try:
            # Validate window
            from utils.window_validation import is_valid_window
            if not is_valid_window(new_hwnd):
                self._logger.error(f"Invalid secondary swap target: {new_hwnd}")
                return False

            overlay = self._get_overlay_by_id(overlay_id)
            if not overlay:
                self._logger.error(f"Secondary overlay {overlay_id} not found")
                return False

            # Note: Docking mode uses its own overlay system, no DWM fade reset needed

            # Perform the source swap
            overlay.set_target_window(new_hwnd)
            
            # Register with auto-switch manager
            self.register_overlay_for_auto_switch(overlay_id, new_hwnd)
            
            # Update MRU if requested
            if record_mru:
                self._mru_manager.record(new_hwnd)

            # Re-sync layout to update secondary positioning
            self.sync_overlay_properties()
            
            self._logger.debug(f"Secondary overlay {overlay_id} swapped to hwnd {new_hwnd}")
            return True
            
        except Exception as e:
            self._logger.error(f"Secondary {overlay_id} source swap failed: {e}")
            return False
    
    def _notify_main_window_change(self, new_hwnd: int) -> None:
        """Notify overlay manager of main window change for auto-switch tracking."""
        try:
            from core.graphics import get_overlay_manager
            overlay_manager = get_overlay_manager()
            if overlay_manager:
                # Register main overlay window change
                overlay_manager._register_overlay_window('main_docking', new_hwnd)
                self._logger.debug(f"Updated auto-switch tracking: main docking -> window {new_hwnd}")
        except Exception as e:
            self._logger.debug(f"Failed to notify main window change: {e}")
    
    def _notify_secondary_window_change(self, overlay_id: str, new_hwnd: int) -> None:
        """Notify overlay manager of secondary window change for auto-switch tracking."""
        try:
            from core.graphics import get_overlay_manager
            overlay_manager = get_overlay_manager()
            if overlay_manager:
                # Register secondary overlay window change
                overlay_manager._register_overlay_window(f'{overlay_id}_docking', new_hwnd)
                self._logger.debug(f"Updated auto-switch tracking: {overlay_id} docking -> window {new_hwnd}")
        except Exception as e:
            self._logger.debug(f"Failed to notify secondary window change: {e}")
    
    def update_source(self, overlay_id: str, hwnd: int) -> bool:
        """Update overlay source window handle (OverlayManager compatibility).
        
        This is a thin wrapper that validates the argument and delegates to
        the existing swap handler, which queues the work appropriately.
        
        Returns True if the request was accepted for processing.
        """
        try:
            new_hwnd = int(hwnd) if hwnd is not None else 0
        except Exception:
            new_hwnd = 0

        if not new_hwnd:
            self._logger.error(f"update_source rejected invalid hwnd: {hwnd}")
            return False

        # Delegate to unified swap path
        self.swap_overlay_source(overlay_id, new_hwnd)
        return True

    def _initialize_missing_modules(self) -> None:
        """Initialize docking dependencies (composition + autoswitch controller)."""
        # Initialize DWM composition manager (singleton)
        try:
            self._initialize_dwm_composition_manager()
        except Exception as e:
            self._logger.warning(f"Failed to initialize DWM composition manager: {e}")
        # Ensure centralized autoswitch controller is running (foreground-based)
        try:
            from core.switching.autoswitch_controller import get_foreground_autoswitch_controller
            self._foreground_autoswitch_controller = get_foreground_autoswitch_controller()
            suppress_debug_log(self._logger, "ForegroundAutoswitchController initialized for docking mode", "DockingManager")
        except Exception as e:
            self._logger.warning(f"Failed to initialize ForegroundAutoswitchController: {e}")
        # Initialize closed-window switch manager for closed-window handling
        try:
            self._initialize_closed_window_switch_manager()
        except Exception as e:
            self._logger.warning(f"Failed to initialize ClosedWindowSwitchManager: {e}")

    def handle_autoswitch_event(self, new_hwnd: int) -> None:
        """Handle autoswitch events with MRU-aware logic for overlays A/B/C.
        
        Implements duplicate prevention, priority resolution, and cycling logic
        as specified in Spec.md MRU-Aware Autoswitch Logic.
        """
        if not self._is_active or not self._main_overlay:
            return
            
        try:
            # Get current overlay assignments
            current_assignments = self._get_current_overlay_assignments()
            
            # Check if this is a cycling scenario (A returns to same window)
            if current_assignments.get('main') == new_hwnd:
                self._handle_cycling_logic(new_hwnd, current_assignments)
                return
            
            # Standard autoswitch: assign new_hwnd to overlay A
            self._assign_window_with_duplicate_prevention('main', new_hwnd, current_assignments)
            
        except Exception as e:
            self._logger.error(f"Autoswitch handling failed: {e}")

    def _get_current_overlay_assignments(self) -> dict[str, Optional[int]]:
        """Get current window assignments for all overlays."""
        assignments = {}
        
        if self._main_overlay:
            assignments['main'] = getattr(self._main_overlay, '_target_hwnd', None)
        
        for i, overlay in enumerate(self._secondary_overlays):
            overlay_key = f'secondary_{i}'
            if overlay:
                assignments[overlay_key] = getattr(overlay, '_target_hwnd', None)
            else:
                assignments[overlay_key] = None
                
        return assignments

    def _handle_cycling_logic(self, returning_hwnd: int, current_assignments: dict[str, Optional[int]]) -> None:
        """Handle cycling when overlay A returns to the same window.
        
        Push A→B, B→C, and assign most recent valid MRU to A.
        """
        try:
            # Check if main overlay is locked - if so, skip cycling entirely
            if self._is_overlay_locked(self._main_overlay):
                self._logger.info("CYCLE: Main overlay locked - skipping cycle logic")
                return
            
            # Get fresh MRU list
            mru_list = self._mru_manager.get_recent(limit=5)  # Get more for cycling
            
            # Find most recent MRU that's not currently assigned
            new_main_hwnd = None
            for hwnd in mru_list:
                if hwnd not in current_assignments.values():
                    new_main_hwnd = hwnd
                    break
            
            if not new_main_hwnd:
                suppress_debug_log(self._logger, "No available MRU for cycling", "DockingManager")
                return
            
            # Perform the cycling: A→B, B→C, new→A
            old_main = current_assignments.get('main')
            old_secondary_0 = current_assignments.get('secondary_0')
            
            # Main is not locked (checked above), perform cycle
            self._assign_window_to_overlay('main', new_main_hwnd)
            
            # Only cycle to secondary_0 if it's not locked AND we have a valid window to assign
            # Also check it's not creating a duplicate with the new main
            if len(self._secondary_overlays) > 0 and not self._is_overlay_locked(self._secondary_overlays[0]):
                if old_main and old_main != new_main_hwnd:
                    self._assign_window_to_overlay('secondary_0', old_main)
                else:
                    self._logger.debug(f"CYCLE: Skipping secondary_0 assignment (old_main={old_main}, would duplicate)")
            
            # Only cycle to secondary_1 if it's not locked AND we have a valid window to assign
            # Check it's not creating duplicates with main or secondary_0
            if len(self._secondary_overlays) > 1 and not self._is_overlay_locked(self._secondary_overlays[1]):
                # Get fresh assignment to check what secondary_0 actually has now (after cycle above)
                current_sec0_hwnd = getattr(self._secondary_overlays[0], '_target_hwnd', None) if self._secondary_overlays else None
                if old_secondary_0 and old_secondary_0 != new_main_hwnd and old_secondary_0 != current_sec0_hwnd:
                    self._assign_window_to_overlay('secondary_1', old_secondary_0)
                else:
                    self._logger.debug(f"CYCLE: Skipping secondary_1 assignment (old_sec0={old_secondary_0}, would duplicate main={new_main_hwnd} or sec0={current_sec0_hwnd})")
            
            suppress_debug_log(self._logger, f"Cycling: {new_main_hwnd}→A, {old_main}→B, {old_secondary_0}→C", "DockingManager")
            
        except Exception as e:
            self._logger.error(f"Cycling logic failed: {e}")

    def _assign_window_with_duplicate_prevention(self, target_overlay: str, new_hwnd: int, current_assignments: dict[str, Optional[int]]) -> None:
        """Assign window to target overlay with duplicate prevention and priority resolution."""
        try:
            # Check if new_hwnd is already assigned to another overlay
            duplicate_overlay = None
            for overlay_key, assigned_hwnd in current_assignments.items():
                if assigned_hwnd == new_hwnd and overlay_key != target_overlay:
                    duplicate_overlay = overlay_key
                    break
            
            if duplicate_overlay:
                # Check if duplicate overlay is locked
                if self._is_overlay_locked_by_key(duplicate_overlay):
                    # Locked overlay has this window - can't swap it out
                    # Instead, find next available MRU for target overlay (skip the duplicate)
                    self._logger.info(f"AUTOSWITCH: Window {new_hwnd} in locked {duplicate_overlay}, skipping and finding next MRU for {target_overlay}")
                    
                    # Get next available window that's not already assigned
                    mru_list = self._mru_manager.get_recent(limit=10)
                    assigned_hwnds = set(current_assignments.values())
                    
                    alternative_hwnd = None
                    for hwnd in mru_list:
                        if hwnd and hwnd not in assigned_hwnds and self._is_window_valid(hwnd):
                            alternative_hwnd = hwnd
                            break
                    
                    if alternative_hwnd and not self._is_overlay_locked_by_key(target_overlay):
                        self._assign_window_to_overlay(target_overlay, alternative_hwnd)
                        self._logger.info(f"AUTOSWITCH: Assigned alternative window {alternative_hwnd} to {target_overlay}")
                    else:
                        self._logger.debug(f"AUTOSWITCH: No alternative window available for {target_overlay}")
                else:
                    # Duplicate overlay is unlocked - can swap
                    old_target_hwnd = current_assignments.get(target_overlay)
                    
                    # Assign new window to target overlay
                    if not self._is_overlay_locked_by_key(target_overlay):
                        self._assign_window_to_overlay(target_overlay, new_hwnd)
                    
                    # Assign old target window to duplicate overlay
                    if old_target_hwnd:
                        self._assign_window_to_overlay(duplicate_overlay, old_target_hwnd)
                    else:
                        # Find next available MRU for the duplicate overlay
                        self._assign_next_available_mru(duplicate_overlay, current_assignments)
                    
                    suppress_debug_log(self._logger, f"Swapped: {target_overlay}←→{duplicate_overlay} to prevent duplicate", "DockingManager")
            else:
                # No duplicate, direct assignment
                if not self._is_overlay_locked_by_key(target_overlay):
                    self._assign_window_to_overlay(target_overlay, new_hwnd)
                
        except Exception as e:
            self._logger.error(f"Duplicate prevention failed: {e}")

    def _assign_next_available_mru(self, overlay_key: str, current_assignments: dict[str, Optional[int]]) -> None:
        """Assign next available MRU window to the specified overlay."""
        try:
            mru_list = self._mru_manager.get_recent(limit=10)
            assigned_hwnds = set(current_assignments.values())
            
            for hwnd in mru_list:
                if hwnd and hwnd not in assigned_hwnds:
                    self._assign_window_to_overlay(overlay_key, hwnd)
                    return
            
            # Fallback: enumerate visible windows
            try:
                from core.window.enumerator import WindowEnumerator
                windows = WindowEnumerator.enum_windows()
                for hwnd, _title in windows:
                    if hwnd not in assigned_hwnds:
                        self._assign_window_to_overlay(overlay_key, hwnd)
                        return
            except Exception as e:
                suppress_debug_log(self._logger, f"Fallback enumeration failed: {e}", "DockingManager")
                
        except Exception as e:
            self._logger.error(f"Next available MRU assignment failed: {e}")
    def _assign_window_to_overlay(self, overlay_key: str, hwnd: int) -> None:
        """Assign a window to a specific overlay by key."""
        try:
            if overlay_key == 'main':
                # Use swap path for consistency and MRU sync
                self.swap_main_overlay_source(int(hwnd), record_mru=True)
                return
            if overlay_key.startswith('secondary_'):
                # Route via unified secondary swap
                self.swap_overlay_source(overlay_key, int(hwnd), record_mru=True)
                return
        
        except Exception as e:
            self._logger.error(f"Window assignment to {overlay_key} failed: {e}")

    def _is_overlay_locked_by_key(self, overlay_key: str) -> bool:
        """Check if overlay is locked by its key."""
        try:
            if overlay_key == 'main':
                return self._is_overlay_locked(self._main_overlay)
            elif overlay_key.startswith('secondary_'):
                index = int(overlay_key.split('_')[1])
                if 0 <= index < len(self._secondary_overlays):
                    return self._is_overlay_locked(self._secondary_overlays[index])
        except Exception:
            pass
        return False
    
    def _initialize_closed_window_switch_manager(self) -> None:
        """Initialize closed-window switch manager for docking mode overlays."""
        try:
            from core.graphics.window_monitor import get_closed_window_switch_manager, initialize_closed_window_switch_manager
            # Get or create closed-window switch manager
            closed_window_switch_manager = get_closed_window_switch_manager()
            if not closed_window_switch_manager:
                # Initialize with main overlay manager
                from core.graphics import get_overlay_manager
                overlay_manager = get_overlay_manager()
                closed_window_switch_manager = initialize_closed_window_switch_manager(overlay_manager)
            
            self._closed_window_switch_manager = closed_window_switch_manager
            self._logger.debug("Closed-window switch manager initialized for docking mode")
            
            # Apply initial dead_switch setting for closed-window auto-replacement
            try:
                from core.settings import get_settings_manager
                settings = get_settings_manager()
                auto_switch_enabled = bool(settings.get("behavior.dead_switch", True))
                self.set_auto_switch_enabled(auto_switch_enabled)
                self._logger.debug(f"Applied initial dead_switch setting: {auto_switch_enabled}")
            except Exception as e:
                self._logger.warning(f"Failed to apply initial auto-switch setting: {e}")
            
        except Exception as e:
            self._logger.warning(f"Failed to initialize closed-window switch manager: {e}")
            self._closed_window_switch_manager = None
    
    def _initialize_dwm_composition_manager(self) -> None:
        """Initialize DWM composition manager for proper overlay attributes."""
        try:
            from core.graphics.dwm_composition_manager import get_dwm_composition_manager
            
            # Get DWM composition manager (singleton)
            self._dwm_composition_manager = get_dwm_composition_manager()
            self._logger.debug("DWM composition manager initialized for docking mode")
            
        except Exception as e:
            self._logger.warning(f"Failed to initialize DWM composition manager: {e}")
            self._dwm_composition_manager = None
    
    def _initialize_app_instance_provider(self) -> None:
        """Initialize app instance provider for window enumeration."""
        try:
            from core.graphics import get_overlay_manager
            
            # Get main overlay manager and its app instance provider
            overlay_manager = get_overlay_manager()
            if hasattr(overlay_manager, '_app_instance_provider') and overlay_manager._app_instance_provider:
                self._app_instance_provider = overlay_manager._app_instance_provider
                suppress_debug_log(self._logger, "App instance provider initialized for docking mode", "DockingManager")
            else:
                suppress_debug_log(self._logger, "No app instance provider available from overlay manager", "DockingManager")
                self._app_instance_provider = None
                
        except Exception as e:
            self._logger.warning(f"Failed to initialize app instance provider: {e}")
            self._app_instance_provider = None
    
    def _inject_app_instance_into_overlay(self, overlay) -> None:
        """Inject app instance into overlay for context menu functionality."""
        try:
            if self._app_instance_provider:
                app_instance = self._app_instance_provider()
                if app_instance:
                    overlay.app_instance = app_instance
                    suppress_debug_log(self._logger, f"Injected app_instance into overlay {overlay.overlay_id}", "DockingManager")
                else:
                    self._logger.warning(f"App instance provider returned None for overlay {overlay.overlay_id}")
            else:
                self._logger.warning(f"No app instance provider available for overlay {overlay.overlay_id}")
        except Exception as e:
            self._logger.error(f"Failed to inject app instance into overlay {overlay.overlay_id}: {e}")
    
    def register_overlay_for_auto_switch(self, overlay_id: str, hwnd: int) -> None:
        """Register an overlay with the auto-switch manager.
        
        Args:
            overlay_id: ID of the overlay
            hwnd: Source window handle
        """
        try:
            if self._closed_window_switch_manager:
                # Use docking-specific overlay ID format
                docking_overlay_id = f"{overlay_id}_docking"
                self._closed_window_switch_manager.register_overlay_window(docking_overlay_id, hwnd)
                self._logger.debug(f"Registered {overlay_id} for auto-switch monitoring")
        except Exception as e:
            self._logger.debug(f"Failed to register {overlay_id} for auto-switch: {e}")
    
    def unregister_overlay_from_auto_switch(self, overlay_id: str) -> None:
        """Unregister an overlay from the auto-switch manager.
        
        Args:
            overlay_id: ID of the overlay
        """
        try:
            if self._closed_window_switch_manager:
                # Use docking-specific overlay ID format
                docking_overlay_id = f"{overlay_id}_docking"
                self._closed_window_switch_manager.unregister_overlay(docking_overlay_id)
                self._logger.debug(f"Unregistered {overlay_id} from auto-switch monitoring")
        except Exception as e:
            self._logger.debug(f"Failed to unregister {overlay_id} from auto-switch: {e}")
    
    def set_auto_switch_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-switching for docking overlays.
        
        Args:
            enabled: Whether to enable auto-switching
        """
        try:
            if self._closed_window_switch_manager:
                self._closed_window_switch_manager.set_auto_switch_enabled(enabled)
                self._logger.info(f"Docking closed-window switching {'enabled' if enabled else 'disabled'}")
            else:
                self._logger.debug("Closed-window switch manager not available for docking mode")
        except Exception as e:
            self._logger.error(f"Failed to set auto-switch enabled state: {e}")
    
    # ========================================================================
    # Mode-Aware Helper Methods
    # ========================================================================
    
    def _update_mru_order_only(self, hwnd: int) -> None:
        """Update MRU order without triggering overlay reassignment.
        
        Used in Normal mode to track MRU for later use without
        changing current overlay assignments.
        
        Args:
            hwnd: Window handle to promote in MRU
        """
        try:
            if not hwnd:
                return
            
            # Record in centralized MRU (this promotes to MRU[0])
            self._mru_manager.record(int(hwnd))
            self._logger.debug(f"Updated MRU order (hwnd {hwnd} promoted to MRU[0])")
            
        except Exception as e:
            self._logger.debug(f"MRU order update failed: {e}")
    
    def _assign_overlay(self, overlay_id: str, hwnd: Optional[int], update_tracking: bool = True) -> bool:
        """Assign a window to a specific overlay.
        
        Args:
            overlay_id: "main", "secondary_0", "secondary_1", etc.
            hwnd: Window handle to assign (None to clear)
            update_tracking: Update Normal mode assignment tracking
            
        Returns:
            True if assignment succeeded, False otherwise
        """
        try:
            overlay = self._get_overlay_by_id(overlay_id)
            if overlay is None:
                self._logger.warning(f"Cannot assign - overlay {overlay_id} not found")
                return False
            
            prev_hwnd = overlay.get_target_hwnd()
            if prev_hwnd != hwnd:
                overlay.set_target_window(hwnd)
                
                # Update Normal mode tracking if requested
                if update_tracking and not self.is_cycle_mode():
                    self._normal_mode_assignments[overlay_id] = hwnd
                
                # Update window monitoring
                if self._window_monitor:
                    try:
                        # Stop monitoring previous window if it's not shown in other overlays
                        if prev_hwnd and prev_hwnd != hwnd:
                            other_overlays_showing = False
                            for other_id in ["main"] + [f"secondary_{i}" for i in range(len(self._secondary_overlays))]:
                                if other_id != overlay_id:
                                    other_overlay = self._get_overlay_by_id(other_id)
                                    if other_overlay and other_overlay.get_target_hwnd() == prev_hwnd:
                                        other_overlays_showing = True
                                        break
                            if not other_overlays_showing:
                                self._window_monitor.remove_window(prev_hwnd)
                        
                        # Start monitoring new window
                        if hwnd:
                            self._window_monitor.add_window(hwnd)
                    except Exception as e:
                        self._logger.debug(f"Window monitor update failed: {e}")
                
                self._logger.debug(f"Assigned {overlay_id} → HWND {hwnd} (prev: {prev_hwnd})")
                return True
            return False
                
        except Exception as e:
            self._logger.error(f"Failed to assign overlay {overlay_id} to hwnd {hwnd}: {e}")
            return False
    
    def _get_next_available_mru(self, validated_mru: list[int], assigned: set[int]) -> Optional[int]:
        """Get next available MRU window not already assigned.
        
        Args:
            validated_mru: List of validated window handles in MRU order
            assigned: Set of already-assigned window handles
            
        Returns:
            Next available hwnd or None
        """
        for hwnd in validated_mru:
            if hwnd not in assigned:
                return hwnd
        return None
    
    def _is_window_valid(self, hwnd: Optional[int]) -> bool:
        """Check if window is still valid and open.
        
        Args:
            hwnd: Window handle to check
            
        Returns:
            True if window is valid, False otherwise
        """
        if not hwnd:
            return False
        try:
            return _is_valid_window(int(hwnd))
        except Exception:
            return False
    
    def _get_validated_mru(self) -> list[int]:
        """Get validated MRU list with mode-specific filtering.
        
        Returns:
            List of validated window handles
        """
        validated = []
        
        # Exclude foreground in Cycle mode only
        fg_hwnd = None
        if self.is_cycle_mode():
            try:
                import win32gui
                fg_hwnd = int(win32gui.GetForegroundWindow() or 0)
            except Exception:
                pass
        
        current_mru = self._get_current_mru()
        for h in current_mru:
            try:
                ih = int(h)
                # Skip foreground in Cycle mode
                if self.is_cycle_mode() and fg_hwnd and ih == fg_hwnd:
                    continue
                # Validate window
                if self._is_window_valid(ih):
                    validated.append(ih)
            except Exception:
                continue
                
        return validated
    
    def _find_overlay_showing_window(self, hwnd: Optional[int]) -> Optional[str]:
        """Find which overlay (if any) is showing the given window.
        
        Args:
            hwnd: Window handle to search for
            
        Returns:
            overlay_id ("main", "secondary_0", etc.) or None if not found
        """
        if not hwnd:
            return None
        
        try:
            # Check main overlay
            if self._main_overlay:
                main_hwnd = self._main_overlay.get_target_hwnd()
                if main_hwnd and int(main_hwnd) == int(hwnd):
                    return "main"
            
            # Check secondary overlays
            for i, overlay in enumerate(self._secondary_overlays):
                overlay_hwnd = overlay.get_target_hwnd()
                if overlay_hwnd and int(overlay_hwnd) == int(hwnd):
                    return f"secondary_{i}"
        
        except Exception as e:
            self._logger.debug(f"Error finding overlay for hwnd {hwnd}: {e}")
        
        return None
    
    def _normal_mode_swap_with_foreground(self, overlay_id: str) -> None:
        """Normal mode: Swap overlay content with current foreground window.
        
        This is used for double-click in Normal mode - the clicked overlay
        gets the foreground window from MRU (not GetForegroundWindow which returns
        the overlay itself after click), and the previous overlay content is focused.
        
        Args:
            overlay_id: Overlay to swap ("main", "secondary_0", etc.)
        """
        try:
            # Get current content of clicked overlay
            overlay = self._get_overlay_by_id(overlay_id)
            if overlay is None:
                return
                
            current_overlay_hwnd = overlay.get_target_hwnd()
            
            # Get all windows currently displayed in overlays
            displayed_windows = set()
            if self._main_overlay:
                main_hwnd = self._main_overlay.get_target_hwnd()
                if main_hwnd:
                    displayed_windows.add(int(main_hwnd))
            for sec_overlay in self._secondary_overlays:
                sec_hwnd = sec_overlay.get_target_hwnd()
                if sec_hwnd:
                    displayed_windows.add(int(sec_hwnd))
            
            # MRU should have user's last focused window at [0] (maintained by FocusTracker)
            # Find first MRU window that's NOT currently displayed in any overlay
            current_mru = self._get_current_mru()
            self._logger.debug(f"NORMAL_SWAP: MRU has {len(current_mru)} windows, {len(displayed_windows)} currently displayed")
            fg_hwnd = None
            for mru_hwnd in current_mru:
                if int(mru_hwnd) not in displayed_windows and self._is_window_valid(int(mru_hwnd)):
                    fg_hwnd = int(mru_hwnd)
                    self._logger.debug(f"NORMAL_SWAP: Selected MRU window {fg_hwnd} for swap (not in overlays)")
                    break
            
            if not fg_hwnd:
                self._logger.debug(f"NORMAL_SWAP: No valid MRU window for swap (MRU: {current_mru[:5]}, displayed: {displayed_windows})")
                return
            
            # Swap: clicked overlay gets foreground from MRU
            self._assign_overlay(overlay_id, fg_hwnd, update_tracking=True)
            
            # Focus the window that was in the overlay (swap to foreground)
            # DO NOT update MRU order here - swaps should be silent and not affect MRU
            # Only explicit user focus changes should update MRU order
            focus_target = None
            if current_overlay_hwnd and self._is_window_valid(current_overlay_hwnd):
                focus_target = int(current_overlay_hwnd)
                self._logger.debug(f"NORMAL_SWAP: Attempting to focus swapped-out window {focus_target}")
                try:
                    self._bring_hwnd_to_focus(focus_target)
                except Exception as focus_err:
                    self._logger.warning(f"NORMAL_SWAP: Failed to focus {focus_target}: {focus_err}")
            else:
                # No valid previous content, just focus what's now in overlay
                focus_target = fg_hwnd
                self._logger.debug(f"NORMAL_SWAP: No valid prev window (prev={current_overlay_hwnd}), focusing new content {focus_target}")
                try:
                    self._bring_hwnd_to_focus(focus_target)
                except Exception as focus_err:
                    self._logger.warning(f"NORMAL_SWAP: Failed to focus {focus_target}: {focus_err}")
                
            self._logger.debug(f"NORMAL_SWAP: {overlay_id} swapped with MRU window (fg={fg_hwnd}, prev={current_overlay_hwnd}, focus_target={focus_target})")
            
            # After swap, resolve any duplicates across unlocked overlays
            self._resolve_duplicates_across_overlays()
            
        except Exception as e:
            self._logger.error(f"Normal mode swap failed for {overlay_id}: {e}", exc_info=True)
            self._logger.debug(f"Failed to unregister {overlay_id} from auto-switch: {e}")
    
    def _on_window_closed(self, hwnd: int) -> None:
        """Handle window close event from WindowMonitor.
        
        When a window closes, find which overlay(s) show it and replace with next valid MRU.
        
        Args:
            hwnd: Handle of closed window
        """
        try:
            self._logger.info(f"Window closed: {hwnd} - refreshing affected overlays")
            
            # In Normal mode, only update affected overlays
            # In Cycle mode, update all overlays (they'll re-populate from MRU)
            if self.is_cycle_mode():
                # Cycle mode: all overlays update from MRU automatically
                self._update_cycle_mode_displays()
            else:
                # Normal mode: find and update only affected overlays
                affected_overlays = []
                
                # Check main overlay
                if self._main_overlay and self._main_overlay.get_target_hwnd() == hwnd:
                    affected_overlays.append("main")
                
                # Check secondary overlays
                for i, overlay in enumerate(self._secondary_overlays):
                    if overlay and overlay.get_target_hwnd() == hwnd:
                        affected_overlays.append(f"secondary_{i}")
                
                if affected_overlays:
                    self._logger.debug(f"Affected overlays: {affected_overlays}")
                    # Trigger Normal mode display update which will replace invalid windows
                    self._update_normal_mode_displays()
                else:
                    self._logger.debug(f"Window {hwnd} not displayed in any overlay")
                    
        except Exception as e:
            self._logger.error(f"Error handling window close: {e}", exc_info=True)
    
    def _resolve_duplicates_across_overlays(self) -> None:
        """Resolve duplicate window assignments across all unlocked overlays.
        
        When one overlay is locked and others are updated via autoswitch, duplicates
        can occur. This method finds and resolves them by reassigning unlocked overlays
        to next available MRU windows.
        
        Only runs in Normal mode (Cycle mode handles this automatically).
        """
        if self.is_cycle_mode():
            return
        
        try:
            # Build map of which windows are shown where
            window_to_overlays: dict[int, list[str]] = {}
            
            # Check main overlay
            if self._main_overlay:
                main_hwnd = self._main_overlay.get_target_hwnd()
                if main_hwnd:
                    window_to_overlays.setdefault(int(main_hwnd), []).append("main")
            
            # Check secondary overlays
            for i, overlay in enumerate(self._secondary_overlays):
                if overlay:
                    sec_hwnd = overlay.get_target_hwnd()
                    if sec_hwnd:
                        overlay_id = f"secondary_{i}"
                        window_to_overlays.setdefault(int(sec_hwnd), []).append(overlay_id)
            
            # Find duplicates
            duplicates = {hwnd: overlay_ids for hwnd, overlay_ids in window_to_overlays.items() if len(overlay_ids) > 1}
            
            if not duplicates:
                return  # No duplicates found
            
            self._logger.warning(f"DUPLICATE_RESOLVE: Found {len(duplicates)} duplicate window(s): {duplicates}")
            
            # For each duplicate, keep it in the first unlocked overlay and reassign others
            validated_mru = self._get_validated_mru()
            assigned = set(window_to_overlays.keys())  # Start with all currently assigned windows
            
            for hwnd, overlay_ids in duplicates.items():
                # Find which overlays showing this duplicate are locked vs unlocked
                locked_showing_this = []
                unlocked_showing_this = []
                for oid in overlay_ids:
                    if self._is_overlay_locked_by_id(oid):
                        locked_showing_this.append(oid)
                    else:
                        unlocked_showing_this.append(oid)
                
                # If ANY locked overlay has this window, ALL unlocked overlays must be reassigned
                if locked_showing_this:
                    if unlocked_showing_this:
                        self._logger.warning(f"DUPLICATE_RESOLVE: Window {hwnd} in locked overlay(s) {locked_showing_this} AND unlocked {unlocked_showing_this}")
                        reassign_overlays = unlocked_showing_this  # Reassign ALL unlocked showing this window
                    else:
                        # All overlays showing this are locked, no action possible
                        continue
                else:
                    # No locked overlays have this window
                    if len(unlocked_showing_this) <= 1:
                        # Only one unlocked overlay, no duplicate issue
                        continue
                    
                    # Multiple unlocked overlays have same window - keep first, reassign others
                    reassign_overlays = unlocked_showing_this[1:]
                
                # Log resolution strategy
                if locked_showing_this:
                    self._logger.debug(f"DUPLICATE_RESOLVE: Window {hwnd} locked in {locked_showing_this}, reassigning {reassign_overlays}")
                else:
                    keep_in = unlocked_showing_this[0] if unlocked_showing_this else "none"
                    self._logger.debug(f"DUPLICATE_RESOLVE: Keeping {hwnd} in {keep_in}, reassigning {reassign_overlays}")
                
                # Reassign duplicate overlays to next available MRU windows
                for oid in reassign_overlays:
                    new_hwnd = self._get_next_available_mru(validated_mru, assigned)
                    if new_hwnd:
                        self._assign_overlay(oid, new_hwnd, update_tracking=True)
                        assigned.add(new_hwnd)
                        self._logger.info(f"DUPLICATE_RESOLVE: Reassigned {oid} from duplicate {hwnd} to {new_hwnd}")
                    else:
                        self._logger.warning(f"DUPLICATE_RESOLVE: No available MRU for {oid}, leaving duplicate")
                        
        except Exception as e:
            self._logger.error(f"Error resolving duplicates: {e}", exc_info=True)
    
    def _is_overlay_locked_by_id(self, overlay_id: str) -> bool:
        """Check if an overlay is locked by its ID.
        
        Args:
            overlay_id: Overlay identifier ("main", "secondary_0", etc.)
            
        Returns:
            True if overlay is locked
        """
        try:
            overlay = self._get_overlay_by_id(overlay_id)
            if overlay:
                return self._is_overlay_locked(overlay)
        except Exception:
            pass
        return False
