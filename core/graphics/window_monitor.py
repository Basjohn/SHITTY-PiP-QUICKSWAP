"""
Window Monitor for Auto-Switch Functionality.

This module provides window monitoring capabilities to detect when DWM capture
windows close and automatically switch overlays to the next valid window.
"""
from __future__ import annotations

from typing import Optional, Set
from PySide6.QtCore import QTimer, QObject, Signal

from core.logging import get_logger
from utils import window_validation as winval
from core.application.window_enumerator import WindowEnumerator


class WindowMonitor(QObject):
    """Monitors window validity and triggers auto-switch when windows close."""
    
    # Signal emitted when a monitored window becomes invalid
    window_closed = Signal(int)  # hwnd
    
    def __init__(self):
        super().__init__()
        self._logger = get_logger("WindowMonitor")
        
        # Set of window handles being monitored
        self._monitored_windows: Set[int] = set()
        
        # Timer for periodic validation checks
        self._validation_timer = QTimer()
        self._validation_timer.timeout.connect(self._validate_windows)
        self._validation_timer.setSingleShot(False)
        
        # Window enumerator for getting valid windows
        self._window_enumerator = WindowEnumerator()
        
        # Check interval in milliseconds (2 seconds)
        self._check_interval_ms = 2000
        
        self._logger.debug("WindowMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start the window monitoring timer."""
        if not self._validation_timer.isActive():
            self._validation_timer.start(self._check_interval_ms)
            self._logger.debug("Window monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the window monitoring timer."""
        if self._validation_timer.isActive():
            self._validation_timer.stop()
            self._logger.debug("Window monitoring stopped")
    
    def add_window(self, hwnd: int) -> None:
        """Add a window to the monitoring list.
        
        Args:
            hwnd: Window handle to monitor
        """
        if hwnd and hwnd not in self._monitored_windows:
            self._monitored_windows.add(hwnd)
            self._logger.debug(f"Added window {hwnd} to monitoring")
            
            # Start monitoring if this is the first window
            if len(self._monitored_windows) == 1:
                self.start_monitoring()
    
    def remove_window(self, hwnd: int) -> None:
        """Remove a window from the monitoring list.
        
        Args:
            hwnd: Window handle to stop monitoring
        """
        if hwnd in self._monitored_windows:
            self._monitored_windows.remove(hwnd)
            self._logger.debug(f"Removed window {hwnd} from monitoring")
            
            # Stop monitoring if no windows left
            if not self._monitored_windows:
                self.stop_monitoring()
    
    def clear_all(self) -> None:
        """Clear all monitored windows and stop monitoring."""
        self._monitored_windows.clear()
        self.stop_monitoring()
        self._logger.debug("Cleared all monitored windows")
    
    def _validate_windows(self) -> None:
        """Validate all monitored windows and emit signals for closed ones."""
        try:
            closed_windows = []
            
            for hwnd in list(self._monitored_windows):
                if not winval.is_valid_window(hwnd):
                    closed_windows.append(hwnd)
                    self._monitored_windows.remove(hwnd)
                    self._logger.info(f"Detected closed window: {hwnd}")
            
            # Emit signals for closed windows
            for hwnd in closed_windows:
                self.window_closed.emit(hwnd)
            
            # Stop monitoring if no windows left
            if not self._monitored_windows:
                self.stop_monitoring()
                
        except Exception as e:
            self._logger.error(f"Window validation failed: {e}")
    
    def get_next_valid_window(self, exclude_hwnds: Optional[Set[int]] = None) -> Optional[int]:
        """Get the next valid window for auto-switching.
        
        Args:
            exclude_hwnds: Set of window handles to exclude from selection
            
        Returns:
            Next valid window handle or None if no valid windows found
        """
        try:
            exclude_set = exclude_hwnds or set()
            
            # Get all capturable windows
            windows = self._window_enumerator.refresh_window_list(force=True)
            
            # Find first valid window not in exclude set
            for hwnd, title, icon in windows:
                if hwnd not in exclude_set and winval.is_valid_window(hwnd):
                    self._logger.debug(f"Found next valid window: {hwnd} ({title})")
                    return hwnd
            
            self._logger.warning("No valid windows found for auto-switch")
            return None
            
        except Exception as e:
            self._logger.error(f"Failed to get next valid window: {e}")
            return None


class AutoSwitchManager(QObject):
    """Manages auto-switching of overlay sources when windows close."""
    
    def __init__(self, overlay_manager):
        super().__init__()
        self._logger = get_logger("AutoSwitchManager")
        self._overlay_manager = overlay_manager
        
        # Window monitor for detecting closed windows
        self._window_monitor = WindowMonitor()
        self._window_monitor.window_closed.connect(self._handle_window_closed)
        
        # Track overlay to window mappings
        self._overlay_windows: dict[str, int] = {}
        
        # Auto-switch enabled flag
        self._auto_switch_enabled = True
        
        self._logger.debug("AutoSwitchManager initialized")
    
    def set_auto_switch_enabled(self, enabled: bool) -> None:
        """Enable or disable auto-switching.
        
        Args:
            enabled: Whether to enable auto-switching
        """
        self._auto_switch_enabled = enabled
        self._logger.info(f"Auto-switch {'enabled' if enabled else 'disabled'}")
        
        if not enabled:
            # Clear monitoring when disabled
            self._window_monitor.clear_all()
            self._overlay_windows.clear()
    
    def register_overlay_window(self, overlay_id: str, hwnd: int) -> None:
        """Register an overlay's source window for monitoring.
        
        Args:
            overlay_id: ID of the overlay
            hwnd: Source window handle
        """
        if not self._auto_switch_enabled:
            return
            
        # Remove previous window if exists
        if overlay_id in self._overlay_windows:
            old_hwnd = self._overlay_windows[overlay_id]
            self._window_monitor.remove_window(old_hwnd)
        
        # Register new window
        self._overlay_windows[overlay_id] = hwnd
        self._window_monitor.add_window(hwnd)
        
        self._logger.debug(f"Registered overlay {overlay_id} with window {hwnd}")
    
    def unregister_overlay(self, overlay_id: str) -> None:
        """Unregister an overlay from monitoring.
        
        Args:
            overlay_id: ID of the overlay to unregister
        """
        if overlay_id in self._overlay_windows:
            hwnd = self._overlay_windows.pop(overlay_id)
            self._window_monitor.remove_window(hwnd)
            self._logger.debug(f"Unregistered overlay {overlay_id}")
    
    def _handle_window_closed(self, closed_hwnd: int) -> None:
        """Handle a window being closed by auto-switching affected overlays.
        
        Args:
            closed_hwnd: Handle of the closed window
        """
        try:
            if not self._auto_switch_enabled:
                return
            
            # Find overlays using this window
            affected_overlays = [
                overlay_id for overlay_id, hwnd in self._overlay_windows.items()
                if hwnd == closed_hwnd
            ]
            
            if not affected_overlays:
                return
            
            self._logger.info(f"Window {closed_hwnd} closed, auto-switching {len(affected_overlays)} overlays")
            
            # Get MRU overlays to prioritize recent windows
            mru_overlays = self._overlay_manager.get_mru_overlays()
            mru_hwnds = set()
            
            # Collect HWNDs from MRU overlays for prioritization
            for overlay in mru_overlays:
                if hasattr(overlay, 'get_source_hwnd'):
                    source_hwnd = overlay.get_source_hwnd()
                    if source_hwnd and winval.is_valid_window(source_hwnd):
                        mru_hwnds.add(source_hwnd)
            
            # Get next valid window, excluding closed one and current overlay windows
            exclude_hwnds = {closed_hwnd}
            exclude_hwnds.update(self._overlay_windows.values())
            
            next_hwnd = self._window_monitor.get_next_valid_window(exclude_hwnds)
            
            if not next_hwnd:
                self._logger.warning("No valid window found for auto-switch")
                # Remove affected overlays from tracking
                for overlay_id in affected_overlays:
                    self._overlay_windows.pop(overlay_id, None)
                return
            
            # Switch affected overlays to the next valid window
            for overlay_id in affected_overlays:
                self._switch_overlay_source(overlay_id, next_hwnd)
                
        except Exception as e:
            self._logger.error(f"Auto-switch failed: {e}")
    
    def _switch_overlay_source(self, overlay_id: str, new_hwnd: int) -> None:
        """Switch an overlay's source to a new window.
        
        Args:
            overlay_id: ID of the overlay to switch
            new_hwnd: New source window handle
        """
        try:
            overlay = self._overlay_manager.get_overlay(overlay_id)
            if not overlay:
                self._logger.warning(f"Overlay {overlay_id} not found for auto-switch")
                return
            
            # Update overlay source if it supports it
            if hasattr(overlay, 'update_source'):
                success = overlay.update_source(new_hwnd)
                if success:
                    # Update our tracking
                    self._overlay_windows[overlay_id] = new_hwnd
                    self._window_monitor.add_window(new_hwnd)
                    
                    self._logger.info(f"Auto-switched overlay {overlay_id} to window {new_hwnd}")
                else:
                    self._logger.error(f"Failed to switch overlay {overlay_id} to window {new_hwnd}")
            else:
                self._logger.warning(f"Overlay {overlay_id} does not support source switching")
                
        except Exception as e:
            self._logger.error(f"Failed to switch overlay {overlay_id}: {e}")


# Global instance
_auto_switch_manager: Optional[AutoSwitchManager] = None


def get_auto_switch_manager() -> Optional[AutoSwitchManager]:
    """Get the global auto-switch manager instance."""
    return _auto_switch_manager


def initialize_auto_switch_manager(overlay_manager) -> AutoSwitchManager:
    """Initialize the global auto-switch manager.
    
    Args:
        overlay_manager: The overlay manager instance
        
    Returns:
        The initialized auto-switch manager
    """
    global _auto_switch_manager
    if _auto_switch_manager is None:
        _auto_switch_manager = AutoSwitchManager(overlay_manager)
    return _auto_switch_manager


def shutdown_auto_switch_manager() -> None:
    """Shutdown the global auto-switch manager."""
    global _auto_switch_manager
    if _auto_switch_manager:
        _auto_switch_manager._window_monitor.clear_all()
        _auto_switch_manager = None
