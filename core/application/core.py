"""
Application Core Module
======================

This module provides the main application class that coordinates all core services.
It uses explicit dependency injection and follows the composition root pattern.

Key Features:
- Centralized service initialization
- Explicit dependencies
- Clean separation of concerns
- Type-safe service access
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Type, TypeVar, cast, List, Tuple, Optional
from PySide6.QtCore import QCoreApplication

from .. import events as core_events
from .. import logging as core_logging
from .. import resources
from .. import settings
from .. import threading
from .. import window as windows
from .window_enumerator import WindowEnumerator
from utils.window_filter import WindowFilter, set_verbose_exclusion_logs
import win32gui
from ..interfaces import (
    IEventSystem,
    ILogger,
    IResourceManager,
    ISettingsManager,
    IThreadManager,
    IWindowManager,
)

# Type variable for generic type hints
T = TypeVar('T')

class ComponentType(Enum):
    """Core service components managed by the application."""
    THREADS = "threads"
    RESOURCES = "resources"
    SETTINGS = "settings"
    DEBUG = "debug"
    OVERLAYS = "overlays"
    STYLES = "styles"
    EVENTS = "events"
    OPACITY = "opacity"

class ApplicationCore:
    """
    Central application controller with explicit dependencies.
    
    This class serves as the composition root for the application,
    initializing and providing access to all core services.
    """
    
    def __init__(self):
        """Initialize core services in the correct order."""
        # Initialize logger first
        self._logger: ILogger = core_logging.AppLogger()
        
        try:
            # Initialize other core services
            self._logger.info("Initializing resource manager...")
            self._resources: IResourceManager = resources.ResourceManager()
            
            self._logger.info("Initializing thread manager...")
            self._threads: IThreadManager = threading.get_thread_manager()
            
            self._logger.info("Initializing event system...")
            self._events: IEventSystem = core_events.EventSystem()
            
            self._logger.info("Initializing settings manager...")
            # Use local settings file instead of user appdata
            settings_file = Path(__file__).parent.parent.parent / 'settings' / 'settings.json'
            self._settings: ISettingsManager = settings.SettingsManager(settings_file=settings_file)
            # Apply initial WindowFilter verbose setting and register change handler
            try:
                initial_verbose = bool(self._settings.get('debug.window_filter_verbose', False))
                set_verbose_exclusion_logs(initial_verbose)

                def _on_window_filter_verbose_changed(key: str, value: object) -> None:
                    try:
                        set_verbose_exclusion_logs(bool(value))
                        self._logger.info(f"[WINFILTER] Setting changed: {key} -> {value}")
                    except Exception as e:
                        self._logger.error(f"Failed to apply {key} change: {e}")

                self._settings.register_change_handler('debug.window_filter_verbose', _on_window_filter_verbose_changed)
            except Exception as e:
                self._logger.error(f"Failed to wire debug.window_filter_verbose: {e}")
            
            self._logger.info("Initializing window manager...")
            self._windows: IWindowManager = windows.create_window_manager()

            # Initialize opacity manager to register hotkeys
            self._logger.info("Initializing opacity manager...")
            from ..opacity.manager import get_opacity_manager
            self._opacity_manager = get_opacity_manager()

            # Defer window-mode features until a window overlay is used
            self._quickswitch_controller = None  # type: ignore[assignment]
            self._autoswitch_controller = None   # type: ignore[assignment]
            self._focus_tracker = None           # type: ignore[assignment]
            self._media_keepalive = None         # type: ignore[assignment]
            self._window_mode_features_ready: bool = False

            # Initialize window enumerator for system windows/icons (centralized)
            self._logger.info("Initializing window enumerator...")
            self._window_enumerator: WindowEnumerator = WindowEnumerator()

            # Provide app_instance to overlays (e.g., DWMOverlay) via OverlayManager DI
            # This avoids any top-level or local imports of app_core in overlay_manager.
            try:
                from ..graphics.overlay_manager import OverlayManager as _OverlayManager
                _OverlayManager().set_app_instance_provider(lambda: self)
                self._logger.debug("OverlayManager app_instance provider wired to ApplicationCore")
            except Exception as e:
                self._logger.error(f"Failed to set app_instance provider on OverlayManager: {e}")

            self._logger.info("Application core initialization complete")
            
        except Exception as e:
            self._logger.critical(f"Failed to initialize application core: {e}", exc_info=True)
            raise
    
    @property
    def logger(self) -> ILogger:
        """Get the logging service."""
        return self._logger
    
    @property
    def resources(self) -> IResourceManager:
        """Get the resource manager."""
        return self._resources
    
    @property
    def threads(self) -> IThreadManager:
        """Get the thread manager."""
        return self._threads
    
    @property
    def events(self) -> IEventSystem:
        """Get the event system."""
        return self._events
    
    @property
    def settings(self) -> ISettingsManager:
        """Get the settings manager."""
        return self._settings
    
    @property
    def windows(self) -> IWindowManager:
        """Get the window manager."""
        return self._windows
    
    @property
    def opacity_manager(self):
        """Get the opacity manager."""
        return self._opacity_manager

    @property
    def quickswitch_controller(self):
        """Get the QuickSwitch controller singleton."""
        # Ensure lazy initialization for window-mode features
        self.ensure_window_mode_features()
        return self._quickswitch_controller

    @property
    def autoswitch_controller(self):
        """Get the AutoSwitch controller singleton."""
        # Ensure lazy initialization for window-mode features
        self.ensure_window_mode_features()
        return self._autoswitch_controller

    def ensure_window_mode_features(self) -> None:
        """Lazily initialize controllers used only in window mode.

        Initializes QuickSwitch, FocusTracker, AutoSwitch, and Media keepalive
        exactly once, and only when needed by window overlays.
        """
        if getattr(self, "_window_mode_features_ready", False):
            return
        try:
            self._logger.info("Lazy-initializing window-mode features...")

            # Switching controllers
            from ..switching.quickswitch_controller import get_quickswitch_controller
            from ..switching.focus_tracker import get_focus_tracker
            from ..switching.autoswitch_controller import get_autoswitch_controller

            self._quickswitch_controller = get_quickswitch_controller()
            self._focus_tracker = get_focus_tracker()
            self._autoswitch_controller = get_autoswitch_controller()

            # Media system (keepalive) remains gated by settings
            self._initialize_media_system()

            self._window_mode_features_ready = True
            self._logger.info("Window-mode features initialized")
        except Exception as e:
            self._logger.error(f"Failed to initialize window-mode features: {e}", exc_info=True)

    def _initialize_media_system(self) -> None:
        """Initialize media control system with efficient keepalive monitoring."""
        try:
            # Only initialize if media control is enabled
            if not self._settings.get("features.media_control_enabled", False):
                self._logger.debug("Media control disabled, skipping keepalive initialization")
                self._media_keepalive = None
                return
            
            # Initialize MediaPlayerKeepAlive with CPU-efficient settings
            from ..media.keepalive import get_media_keepalive
            self._media_keepalive = get_media_keepalive()
            
            # Start the keepalive service
            self._media_keepalive.start()
            self._logger.info("MediaPlayerKeepAlive started with efficient polling")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize media system: {e}")
            self._media_keepalive = None

    # --- Centralized window enumeration/icon APIs ---
    def get_windows(self) -> List[Tuple[int, str]]:
        """Return filtered top-level windows as (hwnd, title) tuples.

        Uses WindowFilter to enumerate and filter capturable windows. A filtered
        out window is not an error. On non-critical errors, logs and returns an
        empty list (graceful failure). No silent paths.
        """
        try:
            hwnds = WindowFilter.get_filtered_windows()
            results: List[Tuple[int, str]] = []
            for hwnd in hwnds:
                try:
                    title = win32gui.GetWindowText(hwnd)
                    if title and title.strip():
                        results.append((hwnd, title))
                except Exception as e:
                    self._logger.debug(f"get_windows: skipping hwnd {hwnd} due to title error: {e}")
                    continue
            return results
        except Exception as e:
            self._logger.error(f"get_windows failed: {e}")
            return []

    def get_window_icon(self, hwnd: int):
        """Return a QIcon for the given hwnd, or None if not available.

        Delegates to the centralized WindowEnumerator. Missing icons are not
        treated as failures; this returns None and logs at debug level.
        """
        try:
            if not hasattr(self, "_window_enumerator") or self._window_enumerator is None:
                self._logger.error("get_window_icon called before window enumerator initialized")
                return None
            return self._window_enumerator.get_window_icon(hwnd)
        except Exception as e:
            self._logger.debug(f"get_window_icon error for hwnd {hwnd}: {e}")
            return None
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Get a service by type.
        
        Args:
            service_type: The type of service to retrieve (interface or implementation)
            
        Returns:
            The requested service instance
            
        Raises:
            ValueError: If the service type is not recognized
        """
        if service_type in (ILogger, core_logging.AppLogger):
            return cast(T, self._logger)
        if service_type in (IThreadManager, threading.ThreadManager):
            return cast(T, self._threads)
        if service_type in (IResourceManager, resources.ResourceManager):
            return cast(T, self._resources)
        if service_type in (ISettingsManager, settings.SettingsManager):
            return cast(T, self._settings)
        if service_type in (IEventSystem, core_events.EventSystem):
            return cast(T, self._events)
        if service_type in (IWindowManager, windows.WindowManager):
            return cast(T, self._windows)
        from ..opacity.manager import OpacityManager
        if service_type in (OpacityManager,):
            return cast(T, self._opacity_manager)
        else:
            raise ValueError(f"Unknown service type: {service_type.__name__}")
    
    def shutdown(self) -> None:
        """Shut down all services in the correct order."""
        self._logger.info("Shutting down application core...")
        
        try:
            # Shutdown in reverse order of initialization
            # Shutdown media system first
            if hasattr(self, '_media_keepalive') and self._media_keepalive:
                self._logger.info("Shutting down media keepalive...")
                self._media_keepalive.stop()
            
            if hasattr(self, '_windows') and self._windows:
                self._logger.info("Shutting down window manager...")
                self._windows.shutdown()
            
            if hasattr(self, '_settings') and self._settings:
                self._logger.info("Saving settings...")
                self._settings.save()
            
            if hasattr(self, '_resources') and self._resources:
                self._logger.info("Cleaning up resources...")
                # ResourceManager exposes shutdown()/cleanup_all(), not cleanup()
                try:
                    self._resources.shutdown()
                except AttributeError:
                    # Fallback for older manager API
                    self._resources.cleanup_all()
            
            if hasattr(self, '_threads') and self._threads:
                self._logger.info("Shutting down thread pool...")
                self._threads.shutdown()
            
            # Opacity manager doesn't need explicit shutdown as it's a singleton
            # and its timers will be stopped when the application exits
            
            self._logger.info("Application core shutdown complete")
            
        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}", exc_info=True)
            raise
        finally:
            # Ensure logger is flushed
            if hasattr(self, '_logger') and self._logger:
                self._logger.shutdown()

# Lazy accessor for a single ApplicationCore instance
_app_core: Optional[ApplicationCore] = None

def get_app_core() -> ApplicationCore:
    """Return the singleton ApplicationCore, creating it after Qt is ready.

    Ensures a Q(Core)Application exists to avoid early ThreadManager.single_shot
    usage inside components (e.g., window adapter, opacity manager).
    """
    global _app_core
    if _app_core is None:
        if QCoreApplication.instance() is None:
            raise RuntimeError("get_app_core called before QApplication is created")
        _app_core = ApplicationCore()
    return _app_core
