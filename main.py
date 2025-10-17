#!/usr/bin/env python3
"""
Main entry point for the SPQ application.

This module initializes and runs the application core components.
"""
import sys
import signal
import traceback
import os
import logging
import ctypes
from utils.paths import get_runtime_root

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

# Resources are loaded directly from the resources/ directory

from core.application import get_app_core
from core.logging import configure_logging, get_logger


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Configure logging early (safe before Qt)
    # Support a --debug flag to enable DEBUG on console (file logging remains DEBUG by default)
    debug_flag = False
    try:
        if "--debug" in sys.argv:
            debug_flag = True
            # Remove to avoid confusing Qt arg parsing
            sys.argv = [arg for arg in sys.argv if arg != "--debug"]
    except Exception:
        debug_flag = False
    # Propagate debug to environment for modules that check SPQ_DEBUG
    if debug_flag:
        try:
            os.environ["SPQ_DEBUG"] = "1"
        except Exception:
            pass

    # On Windows, allocate a console window only when --debug is used.
    # The executable is built windowed (no console) by default.
    try:
        if debug_flag and os.name == 'nt':
            ctypes.windll.kernel32.AllocConsole()
            # Rebind stdout/stderr to the new console for print/log handlers
            sys.stdout = open('CONOUT$', 'w', encoding='utf-8', buffering=1)
            sys.stderr = open('CONOUT$', 'w', encoding='utf-8', buffering=1)
    except Exception:
        # Non-fatal; continue without a console if allocation fails
        pass

    # Logs: relative ./logs directory, per-session timestamped file.
    # Without --debug: only ERROR/CRITICAL to console and file.
    # With --debug: DEBUG to console and file.
    # Logs to portable runtime root
    logs_dir = get_runtime_root() / 'logs'
    _ = configure_logging(
        name="spq",
        log_dir=logs_dir,
        console_level=(logging.DEBUG if debug_flag else logging.ERROR),
        file_level=(logging.DEBUG if debug_flag else logging.ERROR),
        # Approximate ~2000 lines cap by bytes (assumes ~130 bytes/line)
        max_bytes=262144,
        # Replace own content on rollover (no backups)
        backup_count=0,
    )
    logger = get_logger(__name__)
    if debug_flag:
        try:
            logger.info("Debug mode enabled: console logging set to DEBUG")
        except Exception:
            pass
    
    # Startup hygiene: unset deprecated QT_OPENGL (Qt6 ignores ANGLE env)
    try:
        if 'QT_OPENGL' in os.environ:
            removed = os.environ.pop('QT_OPENGL', None)
            logger.info(f"Unset deprecated QT_OPENGL={removed}; Qt6 uses native OpenGL by default.")
    except Exception as e:
        logger.debug(f"Failed to unset QT_OPENGL: {e}")
    
    # Register cleanup handlers via centralized utilities
    try:
        from utils.cache_cleaner import register_cleanup_on_exit
        
        # Register cleanup for the project directory and all subdirectories
        project_root = os.path.dirname(os.path.abspath(__file__))
        register_cleanup_on_exit(project_root)
        logger.info(f"Registered __pycache__ cleanup on exit for: {project_root}")
        
    except Exception as e:
        logger.error(f"Failed to register cleanup handlers: {e}", exc_info=True)
        raise
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum: int, _) -> None:
        """Handle shutdown signals gracefully."""
        signame = signal.Signals(signum).name
        logger.info(f"Received signal {signame}, shutting down...")
        try:
            get_app_core().shutdown()
        except Exception:
            pass
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the application
        logger.info("Starting application...")
        
        # Ensure we have a QApplication instance (needed for dialogs)
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        
        # Single instance enforcement to prevent keyboard hook conflicts
        # Must be after QApplication creation to show dialog
        try:
            from core.application.instance import ApplicationInstanceManager
            
            instance_mgr = ApplicationInstanceManager(app_name="SPQDocker")
            
            if instance_mgr.is_another_instance_running():
                logger.error("Another instance of ShittyPiPQuickSwap is already running.")
                logger.error("Exiting to prevent fuck ups.")
                
                # Show styled dialog to inform user
                try:
                    from ui.dialogs.instance_running_dialog import InstanceRunningDialog
                    dialog = InstanceRunningDialog()
                    dialog.exec()
                except Exception as e:
                    logger.error(f"Failed to show instance dialog: {e}")
                
                return 1
            
            logger.info("Single instance check passed - this is the primary instance")
            
            # Register cleanup to release mutex on exit
            import atexit
            atexit.register(instance_mgr.cleanup)
            
        except Exception as e:
            # Non-fatal: if instance check fails, continue anyway (fail-open)
            logger.warning(f"Could not perform instance check: {e}")
            instance_mgr = None
        
        # Register compiled Qt resources (no-op if module absent)
        try:
            import ui.resources_rc  # noqa: F401
            logger.debug("Qt resources registered from ui.resources_rc")
        except Exception as e:
            logger.debug(f"ui.resources_rc import failed or not present yet: {e}")
        
        # Ensure any quit path triggers a clean shutdown of the core
        try:
            def _on_about_to_quit():
                logger.info("Qt aboutToQuit received; shutting down core...")
                try:
                    get_app_core().shutdown()
                except Exception:
                    # Do not block app quit on shutdown errors
                    pass
            app.aboutToQuit.connect(_on_about_to_quit)
        except Exception:
            # Non-fatal: signal connection may fail in some environments
            pass
            
        # Set application icon from embedded Qt resources
        try:
            app_icon = QIcon(":/icons/ShittyPIP.ico")
            if not app_icon.isNull():
                app.setWindowIcon(app_icon)
                logger.info("Application icon set from Qt resource :/icons/ShittyPIP.ico")
            else:
                logger.error("Qt resource app icon is null: :/icons/ShittyPIP.ico")
        except Exception as e:
            logger.error(f"Failed to set application icon from Qt resources: {e}", exc_info=True)
        
        # Import and launch main dialog
        try:
            from ui.main_dialog import MainDialog
            from utils.theme.theme_manager import get_theme_manager
            from core.opacity.manager import get_opacity_manager
            
            # First ensure core systems are initialized (after QApplication exists)
            logger.info("Ensuring core systems are initialized...")
            core = get_app_core()
            
            # Initialize theme manager after core systems
            theme_manager = get_theme_manager(app)
            logger.info(f"Theme manager initialized with theme: {theme_manager._current_theme}")
            
            # Initialize opacity manager (singleton initialization for hotkeys)
            _ = get_opacity_manager()
            logger.info("Opacity manager initialized")
            
            # Create main dialog
            logger.info("Creating main dialog...")
            main_dialog = MainDialog(app_instance=core)
            
            # Apply theme before showing the window
            logger.info("Applying theme to main dialog...")
            main_dialog.apply_theme(theme_manager._current_theme)
            
            # Make sure window manager is initialized
            if hasattr(core, 'window_manager'):
                logger.info("Ensuring window manager is ready...")
                core.window_manager.ensure_initialized()
            
            # Show the window
            logger.info("Showing main dialog...")
            main_dialog.show()
            main_dialog.raise_()
            main_dialog.activateWindow()
            logger.info("Main dialog shown")
            
            # Debug: Print all top-level widgets
            logger.info(f"Top level widgets: {app.topLevelWidgets()}")
            
            # Run the application event loop
            sys.exit(app.exec())
                
        except ImportError as e:
            logger.error(f"Failed to import MainDialog: {e}")
            logger.error(traceback.format_exc())
            return 1
        except Exception as e:
            logger.error(f"Failed to initialize main dialog: {e}", exc_info=True)
            return 1
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
