"""
System Tray Manager for the application.

This module provides a system tray icon with a context menu for the application.
It handles all tray-related functionality including showing/hiding windows,
managing application settings, and providing quick access to common actions.
"""

from typing import Dict, Optional, TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMenu, QStyle, QSystemTrayIcon

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


class SystemTrayManager(QObject):
    """Manages the system tray icon and its menu for the application.
    
    This class handles the creation and management of the system tray icon,
    including its context menu and all associated actions. It provides a clean
    interface for the main application to interact with the system tray.
    """
    
    # Signal emitted when the user requests to show the main window
    show_main_window_requested = Signal()
    
    # Signal emitted when the user requests to show the settings window
    show_settings_requested = Signal()
    
    # Removed click-through signal
    
    # Signal emitted when the user requests to toggle overlay lock
    toggle_overlay_lock_requested = Signal()
    
    # Signal emitted when the user requests to toggle auto-switch
    toggle_auto_switch_requested = Signal()
    
    # Signal emitted when the user requests to quit the application
    quit_requested = Signal()
    
    def __init__(self, parent: Optional['QWidget'] = None) -> None:
        """Initialize the SystemTrayManager.
        
        Args:
            parent: The parent widget for the system tray icon.
        """
        super().__init__(parent)
        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._tray_menu: Optional[QMenu] = None
        self._actions: Dict[str, QAction] = {}
        
        self._create_actions()
        self._setup_tray()
    
    def _create_actions(self) -> None:
        """Create all actions for the system tray menu."""
        self._actions["main_window"] = QAction("Main Window", self)
        self._actions["main_window"].triggered.connect(self.show_main_window_requested.emit)
        
        self._actions["settings"] = QAction("Settings", self)
        self._actions["settings"].triggered.connect(self.show_settings_requested.emit)
        
        # Removed click-through action
        
        self._actions["toggle_overlay_lock"] = QAction("Lock Overlays", self)
        self._actions["toggle_overlay_lock"].triggered.connect(
            self.toggle_overlay_lock_requested.emit
        )
        
        self._actions["toggle_auto_switch"] = QAction("Auto-switch", self, checkable=True)
        self._actions["toggle_auto_switch"].triggered.connect(
            self.toggle_auto_switch_requested.emit
        )
        
        self._actions["quit"] = QAction("Quit", self)
        self._actions["quit"].triggered.connect(self.quit_requested.emit)
    
    def _setup_tray(self) -> bool:
        """Set up the system tray icon and menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return False
        
        self._tray_icon = QSystemTrayIcon(self.parent())
        self._tray_menu = QMenu()
        
        # Set up the tray icon from embedded Qt resources
        try:
            tray_icon = QIcon(":/icons/ShittyPIP.ico")
        except Exception:
            tray_icon = QIcon()
        if not tray_icon.isNull():
            self._tray_icon.setIcon(tray_icon)
        else:
            # Fallback to system icon if custom icon fails to load
            self._tray_icon.setIcon(
                self.parent().style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
            )
        
        # Add actions to the menu
        self._tray_menu.addAction(self._actions["main_window"])
        self._tray_menu.addAction(self._actions["settings"])
        self._tray_menu.addSeparator()
        # Removed click-through menu item
        self._tray_menu.addAction(self._actions["toggle_overlay_lock"])
        self._tray_menu.addAction(self._actions["toggle_auto_switch"])
        self._tray_menu.addSeparator()
        
        # Add hidden overlay restoration
        restore_action = QAction("Restore Hidden Overlays", self._tray_menu)
        restore_action.triggered.connect(self._restore_hidden_overlays)
        self._tray_menu.addAction(restore_action)
        self._actions["restore_overlays"] = restore_action
        
        self._tray_menu.addSeparator()
        self._tray_menu.addAction(self._actions["quit"])
        
        # Connect the tray icon
        self._tray_icon.setContextMenu(self._tray_menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        
        # Show the tray icon
        self._tray_icon.show()
    
    @Slot(QSystemTrayIcon.ActivationReason)
    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle activation of the system tray icon.
        
        Args:
            reason: The reason for the activation.
        """
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_main_window_requested.emit()
    
    # Removed click-through state setter
    
    def set_overlay_lock_state(self, locked: bool) -> None:
        """Update the overlay lock action text based on current state.
        
        Args:
            locked: Whether overlays are currently locked.
        """
        if "toggle_overlay_lock" in self._actions:
            text = "Unlock Overlays" if locked else "Lock Overlays"
            self._actions["toggle_overlay_lock"].setText(text)
    
    def set_auto_switch_state(self, enabled: bool) -> None:
        """Update the auto-switch toggle action state.
        
        Args:
            enabled: Whether auto-switch is currently enabled.
        """
        if "toggle_auto_switch" in self._actions:
            self._actions["toggle_auto_switch"].setChecked(enabled)
    
    def show_message(self, title: str, message: str, icon: QSystemTrayIcon.MessageIcon = None, 
                    timeout: int = 10000) -> None:
        """Show a message in the system tray.
        
        Args:
            title: The title of the message.
            message: The message text to display.
            icon: The icon to show with the message. If None, no icon is shown.
            timeout: How long to show the message in milliseconds.
        """
        if self._tray_icon:
            if icon is None:
                icon = QSystemTrayIcon.MessageIcon.Information
            self._tray_icon.showMessage(title, message, icon, timeout)
    
    def set_visible(self, visible: bool) -> None:
        """Set the visibility of the system tray icon.
        
        Args:
            visible: Whether the tray icon should be visible.
        """
        if self._tray_icon:
            self._tray_icon.setVisible(visible)
    
    def cleanup(self) -> None:
        """Clean up resources used by the system tray."""
        if self._tray_icon:
            self._tray_icon.hide()
            self._tray_icon.deleteLater()
            self._tray_icon = None
        
        if self._tray_menu:
            self._tray_menu.clear()
            self._tray_menu.deleteLater()
            self._tray_menu = None
        
        for action in self._actions.values():
            if action:
                action.deleteLater()
        
        self._actions.clear()

    def _restore_hidden_overlays(self) -> None:
        """Restore all hidden overlays by making them visible."""
        try:
            from core.graphics.overlay_manager import OverlayManager
            from core.logging import get_logger
            logger = get_logger("SystemTray")
            
            om = OverlayManager()
            if not om:
                logger.error("OverlayManager not available for overlay restoration")
                return
            
            # Get all overlays and show hidden ones
            overlays = om.get_all_overlays()
            restored_count = 0
            
            for overlay in overlays:
                try:
                    # Check if overlay has a host widget that can be shown
                    if hasattr(overlay, '_host') and overlay._host:
                        host = overlay._host
                        if hasattr(host, 'isVisible') and not host.isVisible():
                            host.show()
                            restored_count += 1
                            logger.debug(f"Restored hidden overlay: {getattr(overlay, 'id', 'unknown')}")
                    elif hasattr(overlay, 'show') and hasattr(overlay, 'isVisible'):
                        if not overlay.isVisible():
                            overlay.show()
                            restored_count += 1
                            logger.debug(f"Restored hidden overlay: {getattr(overlay, 'id', 'unknown')}")
                except Exception as e:
                    logger.debug(f"Failed to restore overlay {getattr(overlay, 'id', 'unknown')}: {e}")
            
            if restored_count > 0:
                logger.info(f"Restored {restored_count} hidden overlays")
            else:
                logger.debug("No hidden overlays found to restore")
                # If no hidden overlays, recreate the most recent overlay from MRU
                self._recreate_most_recent_overlay(logger)
                
        except Exception as e:
            from core.logging import get_logger
            logger = get_logger("SystemTray")
            logger.error(f"Failed to restore hidden overlays: {e}", exc_info=True)
    
    def _recreate_most_recent_overlay(self, logger) -> None:
        """Recreate the most recent overlay from MRU when no hidden overlays exist."""
        try:
            from core.switching.mru_manager import get_mru_manager
            from PySide6.QtCore import QRect
            
            mru = get_mru_manager()
            if not mru:
                logger.debug("MRU manager not available")
                return
            
            # Get most recent window
            recent_hwnd = mru.get_most_recent()
            if not recent_hwnd:
                logger.debug("No recent windows in MRU")
                return
            
            # Get overlay manager
            from core.graphics.overlay_manager import OverlayManager
            om = OverlayManager()
            if not om:
                logger.debug("OverlayManager not available")
                return
            
            # Create overlay for most recent window with default size/position
            rect = QRect(100, 100, 640, 360)  # Default position and size
            
            new_overlay = om.create_overlay(
                rect=rect,
                opacity=1.0,
                title="Restored Overlay",
                properties={'hwnd': recent_hwnd},
                bypass_lock=True
            )
            
            if new_overlay:
                logger.info(f"Recreated overlay for most recent window: {recent_hwnd}")
            else:
                logger.debug(f"Failed to recreate overlay for window: {recent_hwnd}")
                
        except Exception as e:
            logger.error(f"Failed to recreate most recent overlay: {e}", exc_info=True)
