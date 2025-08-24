"""
Application lifecycle management.

This module handles the startup and shutdown sequences for the application,
ensuring proper resource management and cleanup.

Access pattern (dependency injection; no top-level app_core import in this module):
    from core.application.core import app_core
    from core.application.lifecycle import ApplicationLifecycle

    lifecycle = ApplicationLifecycle(qapp, app_core.resources, app_core.settings)
"""
from __future__ import annotations

import atexit
import os
import sys
from contextlib import contextmanager
from typing import Callable, List

from PySide6.QtCore import QObject, Signal, QCoreApplication
from core.threading.manager import ThreadManager
from core.logging import get_logger

logger = get_logger(__name__)

class ApplicationLifecycle(QObject):
    """Manages application lifecycle and resource cleanup."""
    
    # Signals
    about_to_quit = Signal()
    initialization_complete = Signal()
    shutdown_initiated = Signal()
    
    def __init__(self, app: QCoreApplication, resources_manager, settings_manager):
        """Initialize the lifecycle manager.
        
        Args:
            app: The QApplication instance
            resources_manager: Injected ResourceManager instance
            settings_manager: Injected SettingsManager instance
        """
        super().__init__()
        self.app = app
        self._resources = resources_manager
        self._settings = settings_manager
        self._cleanup_handlers: List[Callable[[], None]] = []
        self._initialized = False
        self._shutting_down = False
        
        # Connect signals
        self.app.aboutToQuit.connect(self._on_about_to_quit)
        atexit.register(self._on_exit)
    
    def initialize(self) -> bool:
        """Initialize the application.
        
        Returns:
            bool: True if initialization was successful
        """
        if self._initialized:
            return True
            
        try:
            # Initialize core managers
            self._init_resource_manager()
            self._init_thread_manager()
            
            # Register cleanup handlers
            self._register_cleanup_handlers()
            
            self._initialized = True
            self.initialization_complete.emit()
            logger.info("Application initialized successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Failed to initialize application: {e}", exc_info=True)
            self._emergency_shutdown()
            return False
    
    def register_cleanup_handler(self, handler: Callable[[], None]) -> None:
        """Register a cleanup handler to be called during shutdown.
        
        Args:
            handler: A callable that takes no arguments and returns None
        """
        if callable(handler):
            self._cleanup_handlers.append(handler)
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        if self._shutting_down:
            return
            
        self._shutting_down = True
        self.shutdown_initiated.emit()
        
        logger.info("Starting application cleanup...")
        
        # Call cleanup handlers in reverse order of registration
        for handler in reversed(self._cleanup_handlers):
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in cleanup handler: {e}", exc_info=True)
        
        # Clean up core managers
        try:
            ThreadManager.shutdown()
            if hasattr(self._resources, "cleanup_all"):
                resource_manager = self._resources
                resource_manager.cleanup_all()
            elif hasattr(self._resources, "cleanup"):
                # Maintain strict no-fallback: prefer cleanup_all if available; otherwise cleanup
                self._resources.cleanup()
            else:
                raise AttributeError("Resource manager lacks cleanup or cleanup_all")
        except Exception as e:
            logger.error(f"Error during core cleanup: {e}", exc_info=True)
        
        logger.info("Cleanup completed")
    
    def _on_about_to_quit(self) -> None:
        """Handle application about to quit event."""
        self.about_to_quit.emit()
        self.cleanup()
    
    def _on_exit(self) -> None:
        """Handle Python interpreter exit."""
        if not self._shutting_down:
            self.cleanup()
    
    def _init_resource_manager(self) -> None:
        """Initialize the resource manager."""
        try:
            # Resource manager is injected; ensure it is present
            if self._resources is None:
                raise RuntimeError("Resource manager not provided to ApplicationLifecycle")
            logger.debug("Resource manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize resource manager: {e}")
            raise
    
    def _init_thread_manager(self) -> None:
        """Initialize the thread manager."""
        try:
            ThreadManager.initialize()
            logger.debug("Thread manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize thread manager: {e}")
            raise
    
    def _register_cleanup_handlers(self) -> None:
        """Register default cleanup handlers."""
        # Core managers cleanup
        self.register_cleanup_handler(ThreadManager.shutdown)
        if hasattr(self._resources, "cleanup_all"):
            self.register_cleanup_handler(self._resources.cleanup_all)
        elif hasattr(self._resources, "cleanup"):
            self.register_cleanup_handler(self._resources.cleanup)
        else:
            raise AttributeError("Resource manager lacks cleanup or cleanup_all")
        
        # Save settings on exit
        if hasattr(self._settings, "save"):
            self.register_cleanup_handler(self._settings.save)
        else:
            raise AttributeError("Settings manager lacks save method")
    
    def _emergency_shutdown(self) -> None:
        """Perform emergency shutdown of the application."""
        try:
            logger.critical("Performing emergency shutdown")
            
            # Log the error instead of showing a message box to avoid UI dependencies
            logger.critical("A critical error occurred. The application will now exit.")
            
            # Force exit
            sys.exit(1)
            
        except Exception as e:
            # Last resort - just exit
            print(f"FATAL: {e}", file=sys.stderr)
            os._exit(1)

@contextmanager
def application_lifecycle(app: QCoreApplication, resources_manager, settings_manager) -> ApplicationLifecycle:
    """Context manager for application lifecycle.
    
    Example:
        with application_lifecycle(app, app_core.resources, app_core.settings) as lifecycle:
            # Application code here
            pass
    """
    lifecycle = ApplicationLifecycle(app, resources_manager, settings_manager)
    try:
        if not lifecycle.initialize():
            raise RuntimeError("Failed to initialize application")
        yield lifecycle
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        lifecycle._emergency_shutdown()
    finally:
        if not lifecycle._shutting_down:
            lifecycle.cleanup()
