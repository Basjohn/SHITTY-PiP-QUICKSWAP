"""
Docking Overlay - Individual overlay wrapper with synchronization capabilities.

This class wraps a DWM overlay and provides docking-specific functionality
including size synchronization, interaction handling, and MRU integration.
"""
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import QObject, QRect, QSize, QEvent, Qt
from core.logging import get_logger
from core.threading import get_thread_manager
from utils.debug.log_suppressor import suppress_debug_log
from .overlay_pool import get_docking_overlay_pool
from utils.resource_manager import get_resource_manager
from utils.window.overlay_constants import OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT

logger = get_logger(__name__)


class DockingOverlay(QObject):
    """Individual overlay wrapper for the docking system."""
    
    def __init__(self, overlay_id: str, config, is_main: bool = False, manager=None):
        """Initialize a docking overlay.
        
        Args:
            overlay_id: Unique identifier for this overlay
            config: Overlay configuration
            is_main: Whether this is the main overlay (True) or secondary (False)
            manager: Optional docking overlay manager reference
        """
        super().__init__()
        self.overlay_id = overlay_id
        self._config = config
        self._is_main = is_main
        self._manager = manager
        self._dwm_overlay = None
        self._logger = logger
        self._thread_manager = get_thread_manager()
        self._dwm_overlay = None
        self._initialized = False
        self._last_geometry = None
        self._last_opacity = None
        self._fade_animation = None
        self._target_hwnd: Optional[int] = None
        # Resource manager registrations
        self._resource_id: Optional[str] = None
        self._host_resource_id: Optional[str] = None
        
        # Synchronization state
        self._base_size: Optional[QSize] = None
        self._size_ratio: float = 1.0 if is_main else (0.7 if "0" in overlay_id else 0.5)
        
        # Initialization state
        self._is_initialized: bool = False
        # Z-order registration guard
        self._zorder_registered: bool = False

    def _get_valid_target_window(self) -> Optional[int]:
        """Get a valid target window handle for the overlay.
        
        Returns:
            Optional[int]: Valid window handle or None if no valid window found
        """
        try:
            # Import here to avoid circular imports
            import win32gui
            import ctypes
            from utils.window_validation import is_valid_window
            
            # Try to find a suitable visible window first (not desktop)
            try:
                def enum_windows_callback(hwnd, windows):
                    try:
                        if ctypes.windll.user32.IsWindowVisible(hwnd):
                            title = win32gui.GetWindowText(hwnd)
                            if title and len(title.strip()) > 0:  # Has a title
                                if is_valid_window(hwnd):
                                    windows.append((hwnd, title))
                    except Exception:
                        pass
                    return True
                
                windows = []
                win32gui.EnumWindows(enum_windows_callback, windows)
                
                # Use the first valid window we find
                if windows:
                    hwnd, title = windows[0]
                    suppress_debug_log(self._logger, f"Using visible window as target: {hwnd} ({title})", "DockingOverlay")
                    return hwnd
                    
            except Exception as e:
                suppress_debug_log(self._logger, f"Failed to enumerate windows: {e}", "DockingOverlay")
            
            # Fallback to shell window (better than desktop for DWM)
            try:
                shell_hwnd = win32gui.GetShellWindow()
                if shell_hwnd and shell_hwnd != 0:
                    if ctypes.windll.user32.IsWindow(shell_hwnd) and is_valid_window(shell_hwnd):
                        suppress_debug_log(self._logger, f"Using shell window as target: {shell_hwnd}", "DockingOverlay")
                        return shell_hwnd
                        
            except Exception as e:
                suppress_debug_log(self._logger, f"Failed to get shell window: {e}", "DockingOverlay")
            
            self._logger.warning("No valid target window found")
            return None
            
        except Exception as e:
            self._logger.error(f"Error in _get_valid_target_window: {e}")
            return None

    def initialize(self, target_hwnd: Optional[int] = None) -> bool:
        """Initialize the docking overlay with streamlined backend integration."""
        try:
            self._logger.debug(f"Starting initialization for docking overlay {self.overlay_id} (target_hwnd={target_hwnd})")
            if self._is_initialized:
                self._logger.debug(f"Docking overlay {self.overlay_id} already initialized")
                return True
            
            # Set target window configuration
            if target_hwnd and target_hwnd != 0:
                self._target_hwnd = target_hwnd
                self._logger.debug(f"Using provided target_hwnd {target_hwnd} for {self.overlay_id}")
            elif self._is_main:
                # For main overlay, get fallback target window
                self._target_hwnd = self._get_valid_target_window()
                self._logger.debug(f"Main overlay {self.overlay_id} using fallback target_hwnd {self._target_hwnd}")
            
            # Configure overlay properties
            if self._target_hwnd:
                if not hasattr(self._config, 'properties') or self._config.properties is None:
                    self._config.properties = {}
                self._config.properties['hwnd'] = self._target_hwnd
            
            # Create DWM overlay - try pool first, then direct creation
            self._logger.debug(f"Creating DWM overlay for {self.overlay_id}")
            overlay_pool = get_docking_overlay_pool()
            self._dwm_overlay = overlay_pool.acquire_overlay(self.overlay_id, self._config)
            self._logger.debug(f"Pool acquired overlay for {self.overlay_id}: {self._dwm_overlay is not None}")
            
            if not self._dwm_overlay:
                self._logger.debug(f"Pool failed, creating directly via BackendManager for {self.overlay_id}")
                from core.graphics.backend_manager import BackendManager, BackendType
                backend_manager = BackendManager()
                self._dwm_overlay = backend_manager.create_overlay(self._config, BackendType.DWM)
                self._logger.debug(f"BackendManager created overlay for {self.overlay_id}: {self._dwm_overlay is not None}")
            
            if not self._dwm_overlay:
                self._logger.error(f"Failed to create DWM overlay for {self.overlay_id}")
                return False
            
            # Initialize DWM overlay (allow secondary overlays to initialize without valid source)
            try:
                initialization_success = self._dwm_overlay.initialize()
                if not initialization_success and not self._is_main:
                    initialization_success = True  # Secondary overlays can initialize without valid source
            except Exception as e:
                if "Invalid source window" in str(e) and not self._is_main:
                    initialization_success = True  # Expected for secondary overlays
                else:
                    raise
            
            if not initialization_success:
                self._logger.error(f"Failed to initialize DWM overlay for {self.overlay_id}")
                return False
            
            # Register with ResourceManager (backend overlay and its Qt host)
            try:
                from utils.resource_manager import get_resource_manager, ResourceType
                rm = get_resource_manager()
                # Backend overlay (non-Qt object): use WINDOW bucket for cleanup ordering
                try:
                    self._resource_id = rm.register(
                        self._dwm_overlay,
                        resource_type=ResourceType.WINDOW,
                        description=f"DockingOverlay_{self.overlay_id}_backend",
                        cleanup_handler=(lambda obj: obj.cleanup() if hasattr(obj, 'cleanup') else None),
                        role=("main" if self._is_main else "secondary")
                    )
                except Exception as e:
                    self._logger.warning(f"Failed to register backend for {self.overlay_id} with ResourceManager: {e}")
                    self._resource_id = None

                # Qt host (if present): ensure UI-thread-safe cleanup via register_qt
                try:
                    host = getattr(self._dwm_overlay, '_host', None)
                    if host is not None:
                        self._host_resource_id = rm.register_qt(
                            host,
                            resource_type=ResourceType.WINDOW,
                            description=f"DockingOverlayHost_{self.overlay_id}",
                            role=("main" if self._is_main else "secondary")
                        )
                except Exception as e:
                    self._logger.debug(f"Host registration skipped/failed for {self.overlay_id}: {e}")
            except Exception as e:
                self._logger.warning(f"ResourceManager registration failed for {self.overlay_id}: {e}")
                self._resource_id = None
                self._host_resource_id = None
            
            # Set up interaction handling
            self._logger.debug(f"Setting up interaction handling for {self.overlay_id}")
            self._setup_interaction_handling()
            self._logger.debug(f"Interaction handling setup complete for {self.overlay_id}")
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            self._logger.error(f"Critical error initializing docking overlay {self.overlay_id}: {e}")
            self._cleanup_partial_initialization()
            return False
    
    def _cleanup_partial_initialization(self) -> None:
        """Clean up any partially initialized resources."""
        try:
            if hasattr(self, '_dwm_overlay') and self._dwm_overlay:
                try:
                    if hasattr(self._dwm_overlay, 'cleanup'):
                        self._dwm_overlay.cleanup()
                    elif hasattr(self._dwm_overlay, 'close'):
                        self._dwm_overlay.close()
                except Exception as e:
                    suppress_debug_log(self._logger, f"Error during partial cleanup: {e}", "DockingOverlay")
                finally:
                    self._dwm_overlay = None
            
            self._is_initialized = False
            self._target_hwnd = None
        except Exception as e:
            suppress_debug_log(self._logger, f"Error during partial initialization cleanup: {e}", "DockingOverlay")

    def cleanup(self) -> None:
        """Clean up resources and unregister from ResourceManager with enhanced error recovery."""
        cleanup_errors = []
        
        try:
            # Save position before cleanup if we have a manager
            if self._manager and hasattr(self._manager, 'save_overlay_position'):
                try:
                    self._manager.save_overlay_position(self.overlay_id, self.x(), self.y())
                except Exception as e:
                    cleanup_errors.append(f"Failed to save position: {e}")
                    suppress_debug_log(self._logger, f"Failed to save position during cleanup: {e}", "DockingOverlay")
            
            # Clean up DWM overlay - try to return to pool first for performance
            if self._dwm_overlay:
                cleanup_success = False
                
                # Try to return to pool first
                try:
                    overlay_pool = get_docking_overlay_pool()
                    overlay_pool.release_overlay(self.overlay_id, return_to_pool=True)
                    cleanup_success = True
                    suppress_debug_log(self._logger, f"Returned overlay {self.overlay_id} to pool", "DockingOverlay")
                except Exception as e:
                    cleanup_errors.append(f"Pool return failed: {e}")
                    suppress_debug_log(self._logger, f"Failed to return overlay to pool: {e}", "DockingOverlay")
                
                # Fallback to direct cleanup if pool return failed
                if not cleanup_success:
                    # Try close() method first
                    if hasattr(self._dwm_overlay, 'close'):
                        try:
                            self._dwm_overlay.close()
                            cleanup_success = True
                        except Exception as e:
                            cleanup_errors.append(f"DWM overlay close() failed: {e}")
                            suppress_debug_log(self._logger, f"Error calling close() on DWM overlay: {e}", "DockingOverlay")
                    
                    # Try cleanup() method if close() failed
                    if not cleanup_success and hasattr(self._dwm_overlay, 'cleanup'):
                        try:
                            self._dwm_overlay.cleanup()
                            cleanup_success = True
                        except Exception as e:
                            cleanup_errors.append(f"DWM overlay cleanup() failed: {e}")
                            suppress_debug_log(self._logger, f"Error calling cleanup() on DWM overlay: {e}", "DockingOverlay")
                
                # Always clear the reference
                self._dwm_overlay = None
                
                if cleanup_success:
                    pass  # DWM overlay cleanup successful (debug suppressed)
                else:
                    self._logger.warning(f"DWM overlay cleanup had issues for {self.overlay_id}")
            
            # Unregister from ResourceManager
            try:
                from utils.resource_manager import get_resource_manager
                resource_manager = get_resource_manager()
                # Unregister Qt host first (UI object), then backend
                try:
                    if getattr(self, '_host_resource_id', None):
                        resource_manager.unregister(self._host_resource_id)
                        self._host_resource_id = None
                except Exception as e:
                    cleanup_errors.append(f"RM host unregister failed: {e}")
                try:
                    if getattr(self, '_resource_id', None):
                        resource_manager.unregister(self._resource_id)
                        self._resource_id = None
                except Exception as e:
                    cleanup_errors.append(f"RM backend unregister failed: {e}")
                # Also unregister z-order overlay registration if applied
                try:
                    if self._zorder_registered and getattr(self, 'id', None):
                        resource_manager.unregister_overlay(self.id)
                        self._zorder_registered = False
                except Exception:
                    pass
            except Exception as e:
                cleanup_errors.append(f"ResourceManager unregister failed: {e}")
                suppress_debug_log(self._logger, f"Error unregistering from ResourceManager: {e}", "DockingOverlay")
            
            # Reset state
            self._is_initialized = False
            self._target_hwnd = None
            
            # Log cleanup summary
            if cleanup_errors:
                self._logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors for {self.overlay_id}: {'; '.join(cleanup_errors)}")
            else:
                pass  # Docking overlay cleanup successful (debug suppressed)
            
        except Exception as e:
            self._logger.error(f"Critical error during docking overlay cleanup for {self.overlay_id}: {e}")
            # Ensure state is reset even if cleanup fails
            self._is_initialized = False
            self._target_hwnd = None
            self._dwm_overlay = None

    
    def x(self) -> int:
        """Get overlay X position."""
        if self._dwm_overlay and hasattr(self._dwm_overlay, 'x'):
            return self._dwm_overlay.x()
        return 0
    
    def y(self) -> int:
        """Get overlay Y position."""
        if self._dwm_overlay and hasattr(self._dwm_overlay, 'y'):
            return self._dwm_overlay.y()
        return 0
    
    def width(self) -> int:
        """Get overlay width."""
        if self._dwm_overlay and hasattr(self._dwm_overlay, 'width'):
            return self._dwm_overlay.width()
        return 0
    
    def height(self) -> int:
        """Get overlay height."""
        if self._dwm_overlay and hasattr(self._dwm_overlay, 'height'):
            return self._dwm_overlay.height()
        return 0






    def show(self) -> None:
        """Show the overlay."""
        def _show():
            if self._dwm_overlay:
                try:
                    self._dwm_overlay.show()
                    # Overlay shown successfully (debug suppressed)
                except Exception as e:
                    self._logger.warning(f"Error showing overlay {self.overlay_id}: {e}")
        
        self._thread_manager.run_on_ui_thread(_show)

    def hide(self) -> None:
        """Hide the overlay."""
        def _hide():
            if self._dwm_overlay:
                try:
                    self._dwm_overlay.hide()
                    # Overlay hidden successfully (debug suppressed)
                except Exception as e:
                    self._logger.warning(f"Error hiding overlay {self.overlay_id}: {e}")
        
        self._thread_manager.run_on_ui_thread(_hide)

    def set_size_ratio(self, ratio: float) -> None:
        """Set the size ratio relative to the main overlay."""
        self._size_ratio = ratio
        if self._base_size:
            self._apply_size_ratio()

    def sync_with_main(self, main_geometry: QRect, main_opacity: float) -> None:
        """Synchronize properties with the main overlay."""
        if self._is_main:
            return
            
        # Calculate scaled size based on ratio
        scaled_size = QSize(
            int(main_geometry.width() * self._size_ratio),
            int(main_geometry.height() * self._size_ratio)
        )
        
        # Position will be calculated by DockingPositioner
        # For now, just update size and opacity
        scaled_rect = QRect(main_geometry.topLeft(), scaled_size)
        self.set_geometry(scaled_rect)
        self.set_opacity(main_opacity)

    def set_geometry_with_letterbox(self, x: int, y: int, width: int, height: int, letterbox_type: str) -> None:
        """Set geometry with letterboxing/pillarboxing for aspect ratio compliance."""
        def _set_letterbox_geometry():
            if self._dwm_overlay:
                try:
                    # Set flag to allow resize from manager sync
                    self._manager_sync_in_progress = True
                    
                    # Apply letterboxing through DWM overlay
                    self._dwm_overlay.set_geometry(x, y, width, height)
                    # Note: Actual letterboxing rendering would be handled by the DWM overlay's rendering system
                    # Letterbox geometry applied successfully (debug suppressed)
                except Exception as e:
                    self._logger.warning(f"Error setting letterbox geometry for {self.overlay_id}: {e}")
                finally:
                    # Clear flag after geometry change
                    self._manager_sync_in_progress = False
        
        self._thread_manager.run_on_ui_thread(_set_letterbox_geometry)

    def get_target_hwnd(self) -> Optional[int]:
        """Get the target window HWND for this overlay."""
        return self._target_hwnd

    def get_cached_source_aspect(self) -> Optional[float]:
        """Get the cached source aspect ratio from the DWM backend.
        
        Returns:
            Optional[float]: Cached aspect ratio from IntegratedDWMOverlay._source_aspect, or None
        """
        if self._dwm_overlay and hasattr(self._dwm_overlay, '_source_aspect'):
            return getattr(self._dwm_overlay, '_source_aspect', None)
        return None

    

    @property
    def _is_window_locked(self) -> bool:
        """Get the window lock state from the underlying DWM overlay.
        
        Returns:
            bool: True if the window is locked, False otherwise
        """
        if self._dwm_overlay and hasattr(self._dwm_overlay, '_is_window_locked'):
            return bool(getattr(self._dwm_overlay, '_is_window_locked', False))
        return False

    def set_target_window(self, hwnd: Optional[int]) -> None:
        """Set the target window for this overlay."""
        def _set_target():
            if self._dwm_overlay and hwnd:
                try:
                    # Update the source window using DWM overlay's update_source method
                    self._dwm_overlay.update_source(hwnd)
                    self._target_hwnd = hwnd
                    # Keep config properties synchronized for consumers relying on config fallback
                    try:
                        if not hasattr(self._config, 'properties') or self._config.properties is None:
                            self._config.properties = {}
                        self._config.properties['hwnd'] = int(hwnd)
                    except Exception:
                        pass
                    suppress_debug_log(self._logger, f"Updated {self.overlay_id} target window to HWND {hwnd}", "DockingOverlay")
                except Exception as e:
                    self._logger.warning(f"Error updating target window for {self.overlay_id}: {e}")
            elif hwnd is None:
                self._target_hwnd = None
                suppress_debug_log(self._logger, f"Cleared target window for {self.overlay_id}", "DockingOverlay")
        
        self._thread_manager.run_on_ui_thread(_set_target)

    def get_geometry(self) -> QRect:
        """Get the current geometry of the overlay."""
        if self._dwm_overlay:
            try:
                return self._dwm_overlay.get_geometry()
            except Exception as e:
                suppress_debug_log(self._logger, f"Geometry retrieval failed for {self.overlay_id}: {e}", "DockingOverlay")
        
        # Fallback to default geometry
        return QRect(100, 100, 400, 300)

    def set_geometry(self, *args) -> None:
        """Set the geometry of the overlay."""
        def _set_geometry():
            if self._dwm_overlay:
                try:
                    # Set flag to allow resize from manager sync
                    self._manager_sync_in_progress = True
                    
                    if len(args) == 1 and isinstance(args[0], QRect):
                        # QRect argument
                        rect = args[0]
                        self._dwm_overlay.set_geometry(rect.x(), rect.y(), rect.width(), rect.height())
                    elif len(args) == 4:
                        # x, y, width, height arguments
                        self._dwm_overlay.set_geometry(args[0], args[1], args[2], args[3])
                    else:
                        self._logger.warning(f"Invalid geometry arguments for {self.overlay_id}: {args}")
                        return
                        
                    # Geometry set successfully (debug suppressed)
                except Exception as e:
                    self._logger.warning(f"Error setting geometry for {self.overlay_id}: {e}")
                finally:
                    # Clear flag after geometry change
                    self._manager_sync_in_progress = False
        
        self._thread_manager.run_on_ui_thread(_set_geometry)

    def get_opacity(self) -> float:
        """Get the current opacity of the overlay."""
        if self._dwm_overlay:
            try:
                return self._dwm_overlay.get_opacity()
            except Exception as e:
                suppress_debug_log(self._logger, f"Opacity retrieval failed for {self.overlay_id}: {e}", "DockingOverlay")
        return 1.0

    def set_opacity(self, opacity: float) -> None:
        """Set the opacity of the overlay."""
        def _set_opacity():
            if self._dwm_overlay:
                try:
                    self._dwm_overlay.set_opacity(opacity)
                    # Opacity set successfully (debug suppressed)
                except Exception as e:
                    self._logger.warning(f"Error setting opacity for {self.overlay_id}: {e}")
        
        self._thread_manager.run_on_ui_thread(_set_opacity)

    def is_initialized(self) -> bool:
        """Check if the overlay is initialized."""
        return self._is_initialized
        
    def _show_context_menu(self, global_pos=None) -> None:
        """Show the unified context menu for this overlay at the given global position."""
        try:
            if hasattr(self, '_context_menu_handler') and self._context_menu_handler:
                self._context_menu_handler.show_menu(global_pos)
            else:
                self._logger.warning(f"No context menu handler available for {self.overlay_id}")
        except Exception as e:
            self._logger.error(f"Context menu display failed for {self.overlay_id}: {e}")

    def _apply_size_ratio(self) -> None:
        """Apply the current size ratio to the overlay."""
        if not self._base_size or not self._dwm_overlay:
            return
            
        scaled_size = QSize(
            int(self._base_size.width() * self._size_ratio),
            int(self._base_size.height() * self._size_ratio)
        )
        
        current_rect = self.get_geometry()
        new_rect = QRect(current_rect.topLeft(), scaled_size)
        self.set_geometry(new_rect)


    def is_main_overlay(self) -> bool:
        """Check if this is the main overlay."""
        return self._is_main

    def get_size_ratio(self) -> float:
        """Get the current size ratio."""
        return self._size_ratio
    
    def _setup_interaction_handling(self) -> None:
        """Set up interaction handling and host flags for docking overlays."""
        if not self._dwm_overlay:
            return
        
        # Connect double-click for interaction routing
        try:
            if hasattr(self._dwm_overlay, 'doubleClicked'):
                self._dwm_overlay.doubleClicked.connect(self._handle_double_click)
        except Exception as e:
            self._logger.warning(f"Failed to connect double-click signal: {e}")
        
        # Context menu integration for both main and secondary overlays
        self._setup_context_menu_integration()
        
        # Configure the underlying host window depending on role
        if hasattr(self._dwm_overlay, '_host'):
            try:
                host = self._dwm_overlay._host
                if host:
                    # Ensure host can reach the real DWM overlay for lock toggle and key routing
                    try:
                        # Expose backend overlay explicitly without overwriting DockingOverlay parent reference
                        # Docking context menu integration sets host._parent_overlay = self (DockingOverlay)
                        setattr(host, '_backend_overlay', self._dwm_overlay)
                        # Sync focus indicator lock state if indicator exists
                        if hasattr(host, '_focus_indicator') and host._focus_indicator is not None:
                            try:
                                locked = bool(getattr(self._dwm_overlay, '_is_window_locked', False))
                                host._focus_indicator.set_locked(locked)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Docking mode: increase snap strength slightly on the host
                    try:
                        if hasattr(host, '_behavior') and hasattr(host._behavior, 'set_snap_distance'):
                            host._behavior.set_snap_distance(50)
                    except Exception:
                        pass
                    # Sanitize pooled host: remove old DockingOverlay filters and reset transparency
                    self._sanitize_host_flags_and_filters(host)
                if host and not self._is_main:
                    # Secondary overlay: allow free movement and resize interaction
                    flags = host.windowFlags() | Qt.WindowType.WindowStaysOnTopHint
                    # Remove blocking flags to allow movement and resize (keep Qt.Tool to avoid taskbar)
                    flags &= ~Qt.WindowType.WindowDoesNotAcceptFocus
                    host.setWindowFlags(flags)
                    # Allow all mouse events for free interaction
                    host.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
                    host.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # Allow focus on click for interaction
                    host.installEventFilter(self)
                    # Also intercept events from the integrated canvas to prevent wheel self-resize distortion
                    try:
                        if hasattr(host, 'canvas') and host.canvas is not None:
                            host.canvas.installEventFilter(self)
                    except Exception:
                        pass
                    self._logger.debug(f"Configured secondary overlay {self.overlay_id} for free movement and resize")
                    # Ensure focus indicator is visible on non-focusable secondary overlays
                    try:
                        if hasattr(host, "_focus_indicator") and host._focus_indicator is not None:
                            # Show without altering global FocusState
                            host._focus_indicator.set_visible_no_focus(True)
                    except Exception as _fi_e:
                        self._logger.debug(f"Secondary focus indicator enable failed: {_fi_e}")
                elif host and self._is_main:
                    # Main overlay: ensure interactive and on top; remove any secondary-only flags
                    try:
                        flags = host.windowFlags() | Qt.WindowType.WindowStaysOnTopHint
                        flags &= ~Qt.WindowType.WindowDoesNotAcceptFocus
                        # Keep Qt.Tool flag to prevent taskbar presence
                        host.setWindowFlags(flags)
                        host.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
                        host.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                        # Do not install event filter for main host; manager handles main host events
                        self._logger.debug(f"Configured main overlay {self.overlay_id} to stay interactive and on top")
                    except Exception as e:
                        self._logger.warning(f"Failed to configure main overlay {self.overlay_id}: {e}")
            except Exception as e:
                self._logger.warning(f"Failed to set up interaction handling for {self.overlay_id}: {e}")

    def eventFilter(self, obj, event):
        """Filter events to handle context menus and trigger sync on any interaction.

        No blocking - all overlays can interact freely, but sync to keep them bound.
        """
        if hasattr(self, '_dwm_overlay') and hasattr(self._dwm_overlay, '_host'):
            host = self._dwm_overlay._host
            if obj == host or obj == getattr(host, 'canvas', None):
                # Context menu on right-click for all overlays
                if event.type() == QEvent.Type.MouseButtonPress:
                    try:
                        if hasattr(event, 'button') and event.button() == Qt.MouseButton.RightButton:
                            # Use global position if present
                            pos = None
                            try:
                                if hasattr(event, 'globalPosition'):
                                    gp = event.globalPosition()
                                    pos = gp.toPoint() if hasattr(gp, 'toPoint') else gp
                                elif hasattr(event, 'globalPos'):
                                    pos = event.globalPos()
                            except Exception:
                                pos = None
                            self._show_context_menu(pos)
                            return True
                    except Exception:
                        return True

                # Trigger sync on interaction events to keep overlays bound
                # All events pass through without blocking to prevent invisibility
                sync_events = [
                    QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Wheel,
                    QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, 
                    QEvent.Type.MouseMove, QEvent.Type.WindowStateChange
                ]
                if event.type() in sync_events:
                    # During manager batch-apply, ignore sync triggers to avoid feedback loops
                    try:
                        if self._manager and getattr(self._manager, '_batch_applying', False):
                            return False
                    except Exception:
                        pass
                    # Intercept wheel-resize on secondary overlays to scale the entire dock via main.
                    # This prevents the host from briefly resizing itself (visual distortion) before sync applies.
                    if (event.type() == QEvent.Type.Wheel) and (not self._is_main) and self._manager and getattr(self._manager, '_main_overlay', None):
                        try:
                            # Determine wheel direction
                            delta = 0
                            try:
                                if hasattr(event, 'angleDelta') and event.angleDelta().y() != 0:
                                    delta = int(event.angleDelta().y())
                                elif hasattr(event, 'pixelDelta') and event.pixelDelta().y() != 0:
                                    delta = int(event.pixelDelta().y())
                            except Exception:
                                delta = 0
                            if delta != 0:
                                main = self._manager._main_overlay
                                main_rect = main.get_geometry()
                                # Scale factor per wheel notch (~120). Use small step to avoid jumpiness.
                                step = 1.0 + (0.06 if delta > 0 else -0.06)
                                # Preserve AR using cached source aspect if available
                                aspect = None
                                try:
                                    if hasattr(main, 'get_cached_source_aspect'):
                                        aspect = main.get_cached_source_aspect()
                                except Exception:
                                    aspect = None
                                new_w = max(OVERLAY_MIN_WIDTH, int(round(main_rect.width() * step)))
                                if aspect and aspect > 0:
                                    new_h = max(OVERLAY_MIN_HEIGHT, int(round(new_w / aspect)))
                                else:
                                    # Fallback: scale height similarly
                                    new_h = max(OVERLAY_MIN_HEIGHT, int(round(main_rect.height() * step)))
                                # Apply to main only; manager sync will resize and reposition secondaries
                                main.set_geometry(int(main_rect.x()), int(main_rect.y()), new_w, new_h)
                                # Quick sync
                                try:
                                    self._manager._coalesced_sync(1)
                                except Exception:
                                    pass
                                return True  # consume wheel to prevent host self-resize
                        except Exception:
                            # If anything fails, fall through to normal handling
                            pass
                    try:
                        if self._manager and hasattr(self._manager, '_coalesced_sync'):
                            delay = 10 if event.type() == QEvent.Type.Wheel else 5
                            self._manager._coalesced_sync(delay)
                    except Exception:
                        pass
                    # Never block events - always return False to prevent invisibility
                    if event.type() in [QEvent.Type.Wheel, QEvent.Type.Resize]:
                        return False
                        
                # Handle double-click for secondary overlays
                if not self._is_main and event.type() == QEvent.Type.MouseButtonDblClick:
                    try:
                        self._handle_double_click()
                    except Exception:
                        pass
                    return True
                        
        return super().eventFilter(obj, event)

    def _handle_secondary_right_click(self) -> None:
        """Handle right-click on secondary overlay to trigger quickswitch-like behavior."""
        try:
            if self._manager:
                # Use double_click route to engage the same swap/bring-to-front logic
                self._manager.handle_overlay_interaction(self.overlay_id, "double_click")
        except Exception as e:
            self._logger.debug(f"Right-click handler failed for {self.overlay_id}: {e}")

    def _sanitize_host_flags_and_filters(self, host) -> None:
        """Remove stale DockingOverlay event filters and reset critical attributes on a pooled host."""
        try:
            # Remove any DockingOverlay-installed filters from previous uses
            removed = 0
            try:
                filters = list(host.eventFilters()) if hasattr(host, 'eventFilters') else []
            except Exception:
                filters = []
            for f in filters:
                try:
                    if isinstance(f, DockingOverlay) or f.__class__.__name__ == 'DockingOverlay':
                        host.removeEventFilter(f)
                        removed += 1
                except Exception:
                    pass
            if removed:
                self._logger.debug(f"Sanitized host for {self.overlay_id}: removed {removed} DockingOverlay event filters")
            # Reset mouse transparency; role setup will set desired value
            host.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        except Exception as e:
            self._logger.debug(f"Host sanitization skipped due to error: {e}")

    def _handle_double_click(self, event=None) -> None:
        """Handle double-click events on the overlay.
        Accepts an optional event parameter for compatibility with QWidget handlers.
        """
        try:
            if self._manager:
                self._manager.handle_overlay_interaction(self.overlay_id, "double_click")
        except Exception as e:
            self._logger.error(f"Error handling double-click on {self.overlay_id}: {e}", exc_info=True)

    def _setup_context_menu_integration(self) -> None:
        """Set up context menu integration for docking overlays."""
        try:
            # Resolve host from backend overlay
            host = getattr(self._dwm_overlay, '_host', None)
            if not host:
                self._logger.warning(f"No host available for context menu setup on {self.overlay_id}")
                return

            # Ensure menu resolves to this DockingOverlay for enumeration and manager actions
            try:
                setattr(host, '_parent_overlay', self)
            except Exception:
                pass

            # Apply per-role minimum sizes so secondaries can shrink more than main
            try:
                ratio = 1.0 if self._is_main else float(self._size_ratio or 1.0)
                min_w = max(1, int(round(OVERLAY_MIN_WIDTH * ratio)))
                min_h = max(1, int(round(OVERLAY_MIN_HEIGHT * ratio)))
                host.setMinimumSize(QSize(min_w, min_h))
                # Also set dynamic attributes used by WindowBehaviorManager during manual resize
                try:
                    setattr(host, 'min_width', min_w)
                    setattr(host, 'min_height', min_h)
                except Exception:
                    pass
                # Ensure the integrated canvas minimum is relaxed to match per-role floors
                try:
                    canvas = getattr(host, 'canvas', None)
                    if canvas is not None and hasattr(canvas, 'setMinimumSize'):
                        canvas.setMinimumSize(QSize(min_w, min_h))
                except Exception:
                    pass
                self._logger.debug(
                    f"Applied per-role minimum size for {self.overlay_id}: {min_w}x{min_h}"
                )
            except Exception as e:
                self._logger.debug(f"Failed to apply per-role minimum size: {e}")

            # Register with unified z-order so menu lifecycle can prioritize this overlay
            try:
                rm = get_resource_manager()
                rm.register_overlay(self.id, host)
                self._zorder_registered = True
            except Exception as e:
                self._logger.debug(f"Z-order registration failed/skipped: {e}")

            # Build and attach centralized context menu
            try:
                from utils.overlay_context_menu import OverlayContextMenu
                docking_config = {
                    'docking_mode': True,
                    'is_main_overlay': self._is_main,
                    'overlay_id': self.overlay_id,
                    'manager': self._manager,
                    'actions': {
                        # Enable Quit Application item for docking overlays
                        'quit': self._handle_quit_application,
                        # Enable Switch to Single Overlay mode
                        'switch_to_single_overlay': self._handle_switch_to_single_overlay,
                    },
                }
                self._context_menu_handler = OverlayContextMenu(host, overlay_type='dwm', config=docking_config)
                self._context_menu_handler.attach_to_overlay(host)
                self._logger.debug(f"Integrated docking overlay {self.overlay_id} with centralized context menu system")
            except Exception as e:
                self._logger.error(f"Failed to build/attach context menu: {e}")
        except Exception as e:
            self._logger.error(f"Error setting up context menu integration: {e}")

    def _handle_quit_application(self) -> None:
        """Quit the application from the context menu (docking-safe)."""
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                self._logger.info("Quitting application from docking overlay context menu")
                app.quit()
            else:
                self._logger.warning("No QApplication instance found")
        except Exception as e:
            self._logger.error(f"Failed to quit application: {e}")

    def _handle_switch_to_single_overlay(self) -> None:
        """Switch from docking mode to single overlay mode."""
        try:
            if not self._manager:
                self._logger.warning("No manager available for mode switch")
                return
            
            # Get current main overlay's source window to preserve
            target_hwnd = None
            try:
                if hasattr(self._dwm_overlay, 'get_source_hwnd'):
                    target_hwnd = self._dwm_overlay.get_source_hwnd()
            except Exception as e:
                self._logger.debug(f"Could not get source hwnd: {e}")
            
            self._logger.info(f"Switching to single overlay mode (preserving hwnd={target_hwnd})")
            
            # Destroy docking system
            if hasattr(self._manager, 'destroy_docking_system'):
                self._manager.destroy_docking_system()
            
            # Create single overlay with preserved window
            from core.graphics import get_overlay_manager
            from core.graphics.types import OverlayType
            from core.graphics.backends import BackendType
            from PySide6.QtCore import QRect
            
            overlay_mgr = get_overlay_manager()
            
            self._logger.debug(f"Switch state: overlay_mgr={overlay_mgr}, target_hwnd={target_hwnd}")
            
            if overlay_mgr and target_hwnd:
                # Validate the window is still valid
                from utils.window_validation import is_valid_window
                import os
                is_valid = is_valid_window(target_hwnd, os.getpid())
                self._logger.debug(f"Window validation: hwnd={target_hwnd}, valid={is_valid}")
                
                if is_valid:
                    has_create = hasattr(overlay_mgr, 'create_overlay')
                    self._logger.debug(f"OverlayManager has create_overlay: {has_create}")
                    
                    if has_create:
                        # Create DWM overlay with proper parameters
                        from utils.window_validation import get_window_title
                        window_title = get_window_title(target_hwnd) or "Unknown"
                        
                        self._logger.info(f"Creating single overlay: hwnd={target_hwnd}, title='{window_title}'")
                        
                        # Use default positioning (will be adjusted by persistence)
                        overlay_mgr.create_overlay(
                            rect=QRect(100, 100, 600, 400),  # Default size, persistence will override
                            overlay_type=OverlayType.WINDOW,
                            title=window_title,
                            backend=BackendType.DWM,
                            properties={'hwnd': target_hwnd}
                        )
                        self._logger.info(f"Created single DWM overlay for hwnd={target_hwnd}")
                    else:
                        self._logger.error("OverlayManager missing create_overlay method!")
                else:
                    self._logger.warning(f"Target hwnd {target_hwnd} no longer valid")
            else:
                if not overlay_mgr:
                    self._logger.error("Could not get OverlayManager from ResourceManager")
                if not target_hwnd:
                    self._logger.error("No target hwnd to preserve")
            
        except Exception as e:
            self._logger.error(f"Failed to switch to single overlay: {e}", exc_info=True)

    def _setup_docking_context_menu(self) -> None:
        """Deprecated helper; context menu is now set up in _setup_context_menu_integration()."""
        return

    @property
    def id(self) -> str:
        """Stable overlay ID used by context menu/z-order.

        Prefer the backend DWM overlay id; fall back to a synthetic docking id.
        """
        try:
            if self._dwm_overlay and hasattr(self._dwm_overlay, 'id'):
                return str(getattr(self._dwm_overlay, 'id'))
        except Exception:
            pass
        return f"docking_{self.overlay_id}"

    def get_menu_ready_windows(self) -> list:
        """Get list of windows ready for context menu display using DWM overlay method."""
        try:
            # CRITICAL PATH: Prefer centralized WindowEnumerator with icons
            # This avoids app_instance dependency and provides richer menu entries
            try:
                from core.window.enumerator import WindowEnumerator
                enumerator = WindowEnumerator()
                win_with_icons = enumerator.get_capturable_windows_with_icons()
                if win_with_icons:
                    # Already in (hwnd, title, icon) form
                    self._logger.debug(f"ENUMERATION: Retrieved {len(win_with_icons)} windows via WindowEnumerator.get_capturable_windows_with_icons()")
                    return [(hwnd, title, icon) for hwnd, title, icon in win_with_icons if hwnd and title]
            except Exception as e:
                self._logger.debug(f"Icon enumeration failed: {e}")
            
            # Secondary: simple direct enumeration (no icons)
            try:
                from core.window.enumerator import WindowEnumerator
                windows_data = WindowEnumerator.enum_windows()
                if windows_data:
                    windows = [(hwnd, title) for hwnd, title in windows_data if hwnd and title]
                    self._logger.debug(f"ENUMERATION: Retrieved {len(windows)} windows via WindowEnumerator.enum_windows()")
                    return windows
            except Exception as e:
                self._logger.debug(f"Direct enumeration failed: {e}")
            
            # Method 2: Fallback to app_instance if available
            app_instance = getattr(self, 'app_instance', None)
            if not app_instance and self._manager and hasattr(self._manager, '_app_instance'):
                app_instance = self._manager._app_instance
        
            if app_instance:
                try:
                    from utils.window_menu_utils import get_cached_window_list
                    window_list = get_cached_window_list(app_instance)
                    if window_list:
                        windows = [(hwnd, title, icon) for hwnd, title, icon in window_list]
                        self._logger.debug(f"ENUMERATION: Retrieved {len(windows)} windows using cached method")
                        return windows
                except Exception as e:
                    self._logger.debug(f"Cached enumeration failed: {e}")
            
            # Method 3: Last resort - empty list with warning
            self._logger.warning("ENUMERATION: All window enumeration methods failed, returning empty list")
            return []
                
        except Exception as e:
            self._logger.error(f"ENUMERATION: Critical error getting menu ready windows: {e}")
            return []

    def _handle_swap_window(self, target_hwnd: int) -> None:
        """Handle window swap from context menu."""
        try:
            if self._manager:
                # For main overlay, swap main; for secondary, swap that specific secondary
                if getattr(self, '_is_main', False):
                    self._manager.swap_main_overlay_source(int(target_hwnd), record_mru=True)
                else:
                    self._manager.swap_overlay_source(self.overlay_id, int(target_hwnd), record_mru=True)
        except Exception as e:
            self._logger.error(f"Error swapping window: {e}")

    def reset_position(self) -> None:
        """Reset overlay position/size using backend's reset handler.
        Ensures aspect ratio and masking are reapplied by the DWM backend.
        """
        try:
            if self._dwm_overlay and hasattr(self._dwm_overlay, '_handle_reset_position'):
                self._dwm_overlay._handle_reset_position()
            else:
                # Fallback: reapply last known geometry if available
                rect = self.get_geometry()
                if rect:
                    self.set_geometry(rect)
        except Exception as e:
            self._logger.warning(f"reset_position failed for {self.overlay_id}: {e}")
