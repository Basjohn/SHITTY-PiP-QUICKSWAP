"""
Centralized opacity manager for the application.

This module provides centralized opacity management that works across all windows
and overlays, independent of any specific UI instance.
"""

from core.logging import get_logger
from core.settings.settings_manager import SettingsManager
from core.threading import ThreadManager
from PySide6.QtCore import QObject, Signal
import threading
import keyboard

class OpacityManager(QObject):
    """Centralized opacity manager for the application.
    
    This class handles opacity adjustments for all windows and overlays,
    using the settings manager for persistence and the hotkey manager for
    global hotkey registration. It's implemented as a singleton to ensure
    only one instance manages opacity system-wide.
    """
    
    # Timer adjustment interval in milliseconds
    ADJUSTMENT_INTERVAL = 30  # Adjust every 30ms for smoother animation
    
    # Opacity adjustment amount per timer tick
    ADJUSTMENT_AMOUNT = 1  # Adjust by 1% per tick for fine control
    
    # Signal emitted when opacity changes
    opacityChanged = Signal(int)  # opacity percentage
    
    # Signals for hotkey events (thread-safe)
    decreaseKeyPressed = Signal()
    increaseKeyPressed = Signal()
    decreaseKeyReleased = Signal()
    increaseKeyReleased = Signal()
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OpacityManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the opacity manager."""
        if self._initialized:
            return
            
        super().__init__()
        self._logger = get_logger(__name__)
        self._settings_manager = SettingsManager()
        # No per-instance ThreadManager; use centralized static API
        
        # Active flags for self-rescheduling ticks
        self._decrease_key_pressed = False
        self._increase_key_pressed = False
        
        # Deprecated timer placeholders removed (QTimer eliminated)
        
        # Current hotkey settings
        self._decrease_key = None
        self._increase_key = None
        
        # Hotkey handlers (to allow unregistering)
        self._decrease_press_handler = None
        self._increase_press_handler = None
        self._decrease_release_handler = None
        self._increase_release_handler = None
        
        # Load hotkey settings and register global hotkeys
        self._load_hotkey_settings()
        self._register_hotkeys()
        
        self._initialized = True
        self._logger.debug("OpacityManager initialized")
    
    def _register_hotkeys(self) -> None:
        """Register global hotkeys for opacity adjustment using keyboard module."""
        try:
            # Unregister existing hotkeys if any
            self._unregister_hotkeys()
            
            # Check if opacity hotkeys are enabled
            hotkeys_enabled = self._settings_manager.get('hotkeys.opacity_enabled', True)
            if not hotkeys_enabled:
                self._logger.debug("Opacity hotkeys are disabled, not registering hotkeys")
                return
            
            self._logger.debug(f"Registering global hotkeys for opacity adjustment: decrease='{self._decrease_key}', increase='{self._increase_key}'")
            
            # Validate hotkey values before registration
            if not self._decrease_key or not isinstance(self._decrease_key, str):
                self._logger.warning("Invalid decrease hotkey value, skipping registration")
                return
            if not self._increase_key or not isinstance(self._increase_key, str):
                self._logger.warning("Invalid increase hotkey value, skipping registration")
                return
            
            # Register hotkey press events
            if self._decrease_key:
                self._decrease_press_handler = keyboard.on_press_key(self._decrease_key, self._on_decrease_key_press, suppress=True)
            if self._increase_key:
                self._increase_press_handler = keyboard.on_press_key(self._increase_key, self._on_increase_key_press, suppress=True)
            
            # Register hotkey release events
            if self._decrease_key:
                self._decrease_release_handler = keyboard.on_release_key(self._decrease_key, self._on_decrease_key_release, suppress=True)
            if self._increase_key:
                self._increase_release_handler = keyboard.on_release_key(self._increase_key, self._on_increase_key_release, suppress=True)
            
            self._logger.debug("Registered global hotkeys for opacity adjustment using keyboard module")
            
        except Exception as e:
            self._logger.error(f"Failed to register opacity hotkeys: {e}", exc_info=True)
    
    def _on_decrease_key_press(self, event) -> None:
        """Handle decrease opacity key press event."""
        # Only start the timer if this is the first press event (not a repeat)
        if not self._decrease_key_pressed and not event.is_keypad:
            self._logger.debug("Decrease opacity key pressed")
            self._decrease_key_pressed = True
            # Suppress the key event to prevent it from affecting other applications
            event.suppress = True
            # Dispatch to UI thread
            ThreadManager.run_on_ui_thread(self._start_decrease_opacity)
        
    def _on_increase_key_press(self, event) -> None:
        """Handle increase opacity key press event."""
        # Only start the timer if this is the first press event (not a repeat)
        if not self._increase_key_pressed and not event.is_keypad:
            self._logger.debug("Increase opacity key pressed")
            self._increase_key_pressed = True
            # Suppress the key event to prevent it from affecting other applications
            event.suppress = True
            # Dispatch to UI thread
            ThreadManager.run_on_ui_thread(self._start_increase_opacity)
            
    def _on_decrease_key_release(self, event) -> None:
        """Handle decrease opacity key release event."""
        if self._decrease_key_pressed:
            self._logger.debug("Decrease opacity key released")
            self._decrease_key_pressed = False
            # Dispatch to UI thread
            ThreadManager.run_on_ui_thread(self._stop_decrease_opacity)
        
    def _on_increase_key_release(self, event) -> None:
        """Handle increase opacity key release event."""
        if self._increase_key_pressed:
            self._logger.debug("Increase opacity key released")
            self._increase_key_pressed = False
            # Dispatch to UI thread
            ThreadManager.run_on_ui_thread(self._stop_increase_opacity)
            
    def _start_decrease_opacity(self) -> None:
        """Start decreasing opacity via self-rescheduling tick."""
        self._logger.debug("_start_decrease_opacity called")
        # Kick off the periodic tick if newly pressed
        if self._decrease_key_pressed:
            ThreadManager.single_shot(self.ADJUSTMENT_INTERVAL, self._decrease_tick)
            
    def _decrease_tick(self) -> None:
        """One tick of decreasing opacity; reschedules itself while key is held."""
        if not self._decrease_key_pressed:
            return
        self._decrease_opacity()
        if self._decrease_key_pressed:
            ThreadManager.single_shot(self.ADJUSTMENT_INTERVAL, self._decrease_tick)
            
    def _start_increase_opacity(self) -> None:
        """Start increasing opacity via self-rescheduling tick."""
        self._logger.debug("_start_increase_opacity called")
        if self._increase_key_pressed:
            ThreadManager.single_shot(self.ADJUSTMENT_INTERVAL, self._increase_tick)
            
    def _increase_tick(self) -> None:
        """One tick of increasing opacity; reschedules itself while key is held."""
        if not self._increase_key_pressed:
            return
        self._increase_opacity()
        if self._increase_key_pressed:
            ThreadManager.single_shot(self.ADJUSTMENT_INTERVAL, self._increase_tick)
            
    def _stop_decrease_opacity(self) -> None:
        """Stop decreasing opacity."""
        # Simply stop rescheduling by relying on key flag
        pass
        
    # Internal timer stop methods removed (QTimer eliminated)
        
    def _stop_increase_opacity(self) -> None:
        """Stop increasing opacity."""
        # Simply stop rescheduling by relying on key flag
        pass
        
    # Internal timer stop methods removed (QTimer eliminated)
    
    def _increase_opacity(self) -> None:
        """Increase the opacity by a small amount for smooth adjustment."""
        current_opacity = self._settings_manager.get("appearance.opacity", 100)
        new_opacity = min(100, current_opacity + self.ADJUSTMENT_AMOUNT)
        self.set_opacity(new_opacity)
        
    def _decrease_opacity(self) -> None:
        """Decrease the opacity by a small amount for smooth adjustment."""
        current_opacity = self._settings_manager.get("appearance.opacity", 100)
        new_opacity = max(0, current_opacity - self.ADJUSTMENT_AMOUNT)
        self.set_opacity(new_opacity)
    
    def increase_opacity(self, amount: int = 5) -> None:
        """Increase the opacity by the specified amount."""
        current_opacity = self._settings_manager.get("appearance.opacity", 100)
        new_opacity = min(100, current_opacity + amount)
        self.set_opacity(new_opacity)
        
    def decrease_opacity(self, amount: int = 1) -> None:
        """Decrease the opacity by the specified amount."""
        current_opacity = self._settings_manager.get("appearance.opacity", 100)
        new_opacity = max(0, current_opacity - amount)
        self.set_opacity(new_opacity)
    
    def set_opacity(self, percent: int) -> None:
        """Set the opacity to the given percentage (0-100)."""
        percent = max(0, min(100, percent))
        # Read current to avoid redundant updates/logging
        current = self._settings_manager.get("appearance.opacity", 100)
        if current == percent:
            # No change; avoid emitting signals or logging to reduce noise
            return
        # Save and emit only when changed
        self._settings_manager.set("appearance.opacity", percent)
        self.opacityChanged.emit(percent)
    
    def get_opacity(self) -> int:
        """Get the current opacity value."""
        return self._settings_manager.get("appearance.opacity", 100)

    def _unregister_hotkeys(self) -> None:
        try:
            if self._decrease_press_handler:
                try:
                    keyboard.unregister_hotkey(self._decrease_press_handler)
                except KeyError:
                    pass  # Already unregistered
                except Exception as e:
                    self._logger.error(f"Failed to unregister decrease_press_handler: {e}", exc_info=True)
                else:
                    self._decrease_press_handler = None
            if self._increase_press_handler:
                try:
                    keyboard.unregister_hotkey(self._increase_press_handler)
                except KeyError:
                    pass
                except Exception as e:
                    self._logger.error(f"Failed to unregister increase_press_handler: {e}", exc_info=True)
                else:
                    self._increase_press_handler = None
            if self._decrease_release_handler:
                try:
                    keyboard.unregister_hotkey(self._decrease_release_handler)
                except KeyError:
                    pass
                except Exception as e:
                    self._logger.error(f"Failed to unregister decrease_release_handler: {e}", exc_info=True)
                else:
                    self._decrease_release_handler = None
            if self._increase_release_handler:
                try:
                    keyboard.unregister_hotkey(self._increase_release_handler)
                except KeyError:
                    pass
                except Exception as e:
                    self._logger.error(f"Failed to unregister increase_release_handler: {e}", exc_info=True)
                else:
                    self._increase_release_handler = None
            self._logger.debug("Unregistered global hotkeys for opacity adjustment")
        except Exception as e:
            self._logger.error(f"Failed to unregister opacity hotkeys: {e}", exc_info=True)
    def _load_hotkey_settings(self) -> None:
        try:
            # Use only valid defaults
            valid_decrease = '-'
            valid_increase = '='
            valid_quickswitch = '`'
            # Load from settings or use valid defaults
            self._decrease_key = self._settings_manager.get('hotkeys.opacity_decrease', valid_decrease)
            self._increase_key = self._settings_manager.get('hotkeys.opacity_increase', valid_increase)
            self._quickswitch_key = self._settings_manager.get('hotkeys.opacity_quickswitch', valid_quickswitch)
            # Overwrite settings if missing or invalid
            if not self._decrease_key or not isinstance(self._decrease_key, str) or self._decrease_key.lower() in ("", "ctrl+alt+down", "ctrl+shift+down"):
                self._decrease_key = valid_decrease
                self._settings_manager.set('hotkeys.opacity_decrease', valid_decrease)
            if not self._increase_key or not isinstance(self._increase_key, str) or self._increase_key.lower() in ("", "ctrl+alt+up", "ctrl+shift+up"):
                self._increase_key = valid_increase
                self._settings_manager.set('hotkeys.opacity_increase', valid_increase)
            if not self._quickswitch_key or not isinstance(self._quickswitch_key, str) or self._quickswitch_key.lower() in ("", ):
                self._quickswitch_key = valid_quickswitch
                self._settings_manager.set('hotkeys.opacity_quickswitch', valid_quickswitch)
            self._logger.debug(f"Loaded hotkey settings: decrease='{self._decrease_key}', increase='{self._increase_key}', quickswitch='{self._quickswitch_key}'")
        except Exception as e:
            self._logger.error(f"Failed to load hotkey settings: {e}", exc_info=True)
            # Use valid defaults if loading fails
            self._decrease_key = '-'
            self._increase_key = '='
            self._quickswitch_key = '`'
            self._settings_manager.set('hotkeys.opacity_decrease', '-')
            self._settings_manager.set('hotkeys.opacity_increase', '=')
            self._settings_manager.set('hotkeys.opacity_quickswitch', '`')
            self._logger.debug("Using valid default hotkey settings after error")
        self._decrease_key = valid_decrease
        self._increase_key = valid_increase
        self._quickswitch_key = valid_quickswitch
        self._settings_manager.set('hotkeys.opacity_decrease', valid_decrease)
        self._settings_manager.set('hotkeys.opacity_increase', valid_increase)
        self._settings_manager.set('hotkeys.opacity_quickswitch', valid_quickswitch)
        self._logger.debug("Using valid default hotkey settings after error")

    def update_hotkeys(self) -> None:
        """Update hotkeys based on current settings."""
        try:
            self._logger.debug("Updating hotkeys based on current settings")
            # Load new hotkey settings
            self._load_hotkey_settings()
            # Re-register hotkeys
            self._register_hotkeys()
            self._logger.debug("Hotkeys updated successfully")
        except Exception as e:
            self._logger.error(f"Failed to update hotkeys: {e}", exc_info=True)

# Convenience function to get the singleton instance
def get_opacity_manager() -> OpacityManager:
    """Get the singleton instance of the opacity manager."""
    return OpacityManager()
