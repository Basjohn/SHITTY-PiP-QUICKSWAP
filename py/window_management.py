"""
window_management.py

Centralized window management logic extracted from main.py for improved modularity and maintainability.
"""

import win32gui
import gc
import window_validation

from debug_utils import get_logger
from PySide6.QtCore import Signal, QObject

logger = get_logger(__name__)

class WindowManager(QObject):
    """
    Centralized window management for overlays, MRU, and lock state.
    All logic is migrated from main.py for modularity and maintainability.
    """
    # Define signals for external components to connect to
    lock_state_changed = Signal(bool)  # Emitted when overlay lock state changes
    
    def __init__(self, max_mru_items=50):
        super().__init__()
        self.logger = get_logger(__name__)
        
        self.mru_hwnds = []
        self._hwnd_last_focus_ts = {}
        self.MAX_MRU_ITEMS = max_mru_items
        self._overlays_locked = False
        self.active_overlays = {}
        self.keep_alive_handlers = {}

    def is_overlay_locked(self):
        return self._overlays_locked

    def set_overlay_lock(self, locked):
        locked_state = bool(locked)
        if self._overlays_locked != locked_state:
            self._overlays_locked = locked_state
            self.logger.debug(f"Overlay lock {'enabled' if locked_state else 'disabled'}")
            
            # Update all overlays with the new lock state
            for overlay in self.active_overlays.values():
                if hasattr(overlay, 'update_lock_state'):
                    overlay.update_lock_state(locked_state)
            
            # Emit signal to notify other components of the state change
            self.lock_state_changed.emit(locked_state)

    def toggle_overlay_lock(self):
        self.set_overlay_lock(not self.is_overlay_locked())
        return self._overlays_locked

    def update_mru(self, hwnd):
        if self.is_overlay_locked():
            return
        if not hwnd or not window_validation.is_valid_window(hwnd):
            return
        for overlay in self.active_overlays.values():
            if hasattr(overlay, 'winId') and overlay.winId() == hwnd:
                return
        self.clean_mru_list()
        if hwnd in self.mru_hwnds:
            self.mru_hwnds.remove(hwnd)
            self._hwnd_last_focus_ts.pop(hwnd, None)
        self.mru_hwnds.insert(0, hwnd)
        self._hwnd_last_focus_ts[hwnd] = __import__('time').time()
        if len(self.mru_hwnds) > self.MAX_MRU_ITEMS:
            self.mru_hwnds = self.mru_hwnds[:self.MAX_MRU_ITEMS]
        self.logger.debug(f"Updated MRU list. New order: {[f'HWND={h}' for h in self.mru_hwnds]}")

    def clean_mru_list(self):
        if self.is_overlay_locked():
            return
        if not hasattr(self, 'mru_hwnds'):
            self.mru_hwnds = []
            return
        our_windows = set()
        for overlay in self.active_overlays.values():
            if hasattr(overlay, 'winId'):
                our_windows.add(overlay.winId())
        cleaned_mru = []
        for hwnd in self.mru_hwnds:
            if not window_validation.is_valid_window(hwnd):
                continue
            if hwnd in our_windows:
                continue
            if not win32gui.GetWindowText(hwnd):
                continue
            cleaned_mru.append(hwnd)
        self.mru_hwnds = cleaned_mru

    def _get_widget_hwnd(self, widget):
        """Standardized method to get a window handle from a widget.
        
        Args:
            widget: The widget to get the window handle for
            
        Returns:
            int: The window handle, or None if not available
        """
        if not widget:
            return None
            
        # First check if it has an hwnd attribute
        hwnd = getattr(widget, 'hwnd', None)
        
        # Try to get it from winId() method
        if not hwnd and hasattr(widget, 'winId'):
            try:
                hwnd = int(widget.winId())
                # Cache the hwnd for future use
                widget._hwnd = hwnd
            except (AttributeError, TypeError) as e:
                self.logger.debug(f"Could not get winId: {e}")
        
        return hwnd
    
    def add_overlay(self, overlay_widget):
        """Add an overlay widget to the active overlays registry.
        
        Args:
            overlay_widget: The overlay widget to add
        """
        if not overlay_widget:
            return
        try:
            # Get the window handle using standardized method
            hwnd = self._get_widget_hwnd(overlay_widget)
            
            # Use object ID as fallback if no hwnd available
            if not hwnd:
                hwnd = id(overlay_widget)
                self.logger.warning(f"Overlay has no hwnd, using object id as fallback: {hwnd}")
                
            # Add to active overlays and log
            self.active_overlays[hwnd] = overlay_widget
            self.logger.info(f"Added overlay with ID: {hwnd}, Type: {type(overlay_widget).__name__}, Total overlays: {len(self.active_overlays)}")
        except Exception as e:
            self.logger.exception(f"Error adding overlay: {e}")
        
        # Update MRU if we got a valid hwnd
        if hwnd:
            self.update_mru(hwnd)

    def remove_overlay(self, overlay_widget):
        """Remove an overlay widget from the active overlays registry and clean it up.
        
        Args:
            overlay_widget: The overlay widget to remove
        """
        if not overlay_widget:
            return
            
        hwnd = None
        try:
            # Get the window handle using standardized method
            hwnd = self._get_widget_hwnd(overlay_widget)
            
            # If no hwnd found, search for the widget in active_overlays by reference
            if not hwnd:
                for k, v in list(self.active_overlays.items()):
                    if v == overlay_widget:
                        hwnd = k
                        self.logger.debug(f"Found overlay by reference with key: {hwnd}")
                        break
        except Exception as e:
            self.logger.exception(f"Error removing overlay: {e}")
            
        # Remove from registry if found
        if hwnd in self.active_overlays:
            del self.active_overlays[hwnd]
            self.logger.info(f"Removed overlay for HWND: {hwnd}, Remaining overlays: {len(self.active_overlays)}")
            
            # Cleanup the widget
            try:
                if hasattr(overlay_widget, 'cleanup'):
                    overlay_widget.cleanup()
                if hasattr(overlay_widget, 'deleteLater'):
                    overlay_widget.deleteLater()
            except Exception as e:
                self.logger.exception(f"Error cleaning up overlay: {e}")
                
            # Force garbage collection
            gc.collect()
