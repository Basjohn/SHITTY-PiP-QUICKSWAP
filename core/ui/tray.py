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
from utils.paths import get_data_dir

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
    
    # Signal emitted when the user requests to toggle click-through mode
    toggle_click_through_requested = Signal()
    
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
        
        self._actions["toggle_click_through"] = QAction("Toggle Click-through", self)
        self._actions["toggle_click_through"].triggered.connect(
            self.toggle_click_through_requested.emit
        )
        
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
    
    def _setup_tray(self) -> None:
        """Set up the system tray icon and menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        
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
        self._tray_menu.addAction(self._actions["toggle_click_through"])
        self._tray_menu.addAction(self._actions["toggle_overlay_lock"])
        self._tray_menu.addAction(self._actions["toggle_auto_switch"])
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
    
    def set_click_through_state(self, enabled: bool) -> None:
        """Update the click-through toggle action text based on current state.
        
        Args:
            enabled: Whether click-through is currently enabled.
        """
        if "toggle_click_through" in self._actions:
            text = "Disable Click-through" if enabled else "Enable Click-through"
            self._actions["toggle_click_through"].setText(text)
    
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
