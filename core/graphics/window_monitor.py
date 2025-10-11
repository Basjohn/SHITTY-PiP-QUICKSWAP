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
from utils.resource_manager import get_resource_manager


class WindowMonitor(QObject):
    """Monitors window validity and triggers auto-switch when windows close."""
    
    # Signal emitted when a monitored window becomes invalid
    window_closed = Signal(int)  # hwnd
    
    def __init__(self):
        super().__init__()
        self._logger = get_logger("WindowMonitor")
        self._resource_manager = get_resource_manager()
        
        # Set of window handles being monitored
        self._monitored_windows: Set[int] = set()
        
        # Timer for periodic validation checks - register with ResourceManager
        self._validation_timer = QTimer()
        self._validation_timer.timeout.connect(self._validate_windows)
        self._validation_timer.setSingleShot(False)
        self._timer_resource_id = self._resource_manager.register_qt_timer(
            self._validation_timer,
            description="WindowMonitor validation timer"
        )
        
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
    
    def cleanup(self) -> None:
        """Cleanup resources before shutdown."""
        try:
            self.stop_monitoring()
            self._monitored_windows.clear()
            if hasattr(self, '_timer_resource_id') and self._timer_resource_id:
                self._resource_manager.unregister(self._timer_resource_id)
            self._logger.debug("WindowMonitor cleanup complete")
        except Exception as e:
            self._logger.debug(f"WindowMonitor cleanup error: {e}")
    
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


class ClosedWindowSwitchManager(QObject):
    """Manages auto-switching of overlay sources when windows close.
    
    NOTE: This is NOT related to foreground-focus autoswitch (see ForegroundAutoswitchController).
    Monitors registered overlay windows and automatically switches to next valid window
    when a window closes.
    """
    
    def __init__(self, overlay_manager):
        super().__init__()
        self._logger = get_logger("ClosedWindowSwitchManager")
        self._overlay_manager = overlay_manager
        
        # Window monitor for detecting closed windows
        self._window_monitor = WindowMonitor()
        self._window_monitor.window_closed.connect(self._handle_window_closed)
        
        # Track overlay to window mappings
        self._overlay_windows: dict[str, int] = {}
        
        # Auto-switch enabled flag
        self._auto_switch_enabled = True
        
        self._logger.debug("ClosedWindowSwitchManager initialized")
    
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
            
            self._logger.info(f"Window {closed_hwnd} closed, auto-switching {len(affected_overlays)} overlay(s)")
            
            # Handle docking overlays with MRU-aware logic
            docking_affected = [oid for oid in affected_overlays if oid.endswith('_docking')]
            if docking_affected:
                try:
                    from core.graphics.docking import get_docking_manager
                    from core.switching.mru_manager import get_mru_manager
                    
                    docking_manager = get_docking_manager()
                    if docking_manager and docking_manager.is_active():
                        # Get next MRU window (intelligent selection)
                        mru_manager = get_mru_manager()
                        mru_list = mru_manager.get_recent(limit=10)
                        
                        # Get current overlay assignments to avoid duplicates
                        current_assignments = set()
                        for oid, hwnd in self._overlay_windows.items():
                            if oid.endswith('_docking') and hwnd != closed_hwnd:
                                current_assignments.add(hwnd)
                        
                        # Find next MRU window not currently assigned
                        next_mru = None
                        for hwnd in mru_list:
                            if hwnd != closed_hwnd and hwnd not in current_assignments and winval.is_valid_window(hwnd):
                                next_mru = hwnd
                                break
                        
                        if next_mru:
                            # Switch each affected docking overlay individually
                            for overlay_id in docking_affected:
                                actual_id = overlay_id[:-8]  # Remove "_docking"
                                success = docking_manager.update_source(actual_id, next_mru)
                                if success:
                                    self._overlay_windows[overlay_id] = next_mru
                                    self._window_monitor.add_window(next_mru)
                                    self._logger.info(f"Auto-switched docking overlay {actual_id} to MRU window {next_mru}")
                                else:
                                    # Remove from tracking if switch failed
                                    self._overlay_windows.pop(overlay_id, None)
                        else:
                            # No valid MRU window, fall back to enumeration
                            exclude_hwnds = {closed_hwnd}
                            exclude_hwnds.update(current_assignments)
                            fallback_hwnd = self._window_monitor.get_next_valid_window(exclude_hwnds)
                            
                            if fallback_hwnd:
                                for overlay_id in docking_affected:
                                    actual_id = overlay_id[:-8]
                                    success = docking_manager.update_source(actual_id, fallback_hwnd)
                                    if success:
                                        self._overlay_windows[overlay_id] = fallback_hwnd
                                        self._window_monitor.add_window(fallback_hwnd)
                                        self._logger.info(f"Auto-switched docking overlay {actual_id} to fallback window {fallback_hwnd}")
                                    else:
                                        self._overlay_windows.pop(overlay_id, None)
                            else:
                                # No valid windows at all
                                for overlay_id in docking_affected:
                                    self._overlay_windows.pop(overlay_id, None)
                                self._logger.warning("No valid window found for docking overlays")
                        return
                except Exception as e:
                    self._logger.error(f"Docking overlay auto-switch failed: {e}")
            
            # Handle single overlays (non-docking)
            single_affected = [oid for oid in affected_overlays if not oid.endswith('_docking')]
            if single_affected:
                # Get next valid window, excluding closed one and current overlay windows
                exclude_hwnds = {closed_hwnd}
                exclude_hwnds.update(self._overlay_windows.values())
                
                # Try MRU first for single overlays too
                try:
                    from core.switching.mru_manager import get_mru_manager
                    mru_manager = get_mru_manager()
                    mru_list = mru_manager.get_recent(limit=10)
                    
                    next_hwnd = None
                    for hwnd in mru_list:
                        if hwnd not in exclude_hwnds and winval.is_valid_window(hwnd):
                            next_hwnd = hwnd
                            break
                    
                    if not next_hwnd:
                        # Fallback to enumeration
                        next_hwnd = self._window_monitor.get_next_valid_window(exclude_hwnds)
                except Exception:
                    next_hwnd = self._window_monitor.get_next_valid_window(exclude_hwnds)
                
                if not next_hwnd:
                    self._logger.warning("No valid window found for single overlay auto-switch")
                    for overlay_id in single_affected:
                        self._overlay_windows.pop(overlay_id, None)
                    return
                
                # Switch each affected single overlay
                for overlay_id in single_affected:
                    overlay = self._overlay_manager.get_overlay(overlay_id)
                    if not overlay:
                        self._logger.warning(f"Overlay {overlay_id} not found for auto-switch")
                        continue
                    
                    if hasattr(overlay, 'update_source'):
                        success = overlay.update_source(next_hwnd)
                        if success:
                            self._overlay_windows[overlay_id] = next_hwnd
                            self._window_monitor.add_window(next_hwnd)
                            self._logger.info(f"Auto-switched single overlay {overlay_id} to window {next_hwnd}")
                        else:
                            self._logger.error(f"Failed to switch overlay {overlay_id} to window {next_hwnd}")
                    else:
                        self._logger.warning(f"Overlay {overlay_id} does not support source switching")
                
        except Exception as e:
            self._logger.error(f"Auto-switch failed: {e}")


# Global instance
_closed_window_switch_manager: Optional[ClosedWindowSwitchManager] = None


def get_closed_window_switch_manager() -> Optional[ClosedWindowSwitchManager]:
    """Get the global closed-window switch manager instance."""
    return _closed_window_switch_manager


def initialize_closed_window_switch_manager(overlay_manager) -> ClosedWindowSwitchManager:
    """Initialize the global closed-window switch manager.
    
    Args:
        overlay_manager: The overlay manager instance
        
    Returns:
        The initialized closed-window switch manager
    """
    global _closed_window_switch_manager
    if _closed_window_switch_manager is None:
        _closed_window_switch_manager = ClosedWindowSwitchManager(overlay_manager)
    return _closed_window_switch_manager


def shutdown_closed_window_switch_manager() -> None:
    """Shutdown the global closed-window switch manager."""
    global _closed_window_switch_manager
    if _closed_window_switch_manager:
        _closed_window_switch_manager._window_monitor.clear_all()
        _closed_window_switch_manager = None
