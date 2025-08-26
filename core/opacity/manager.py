"""
Centralized opacity manager for the application.

This module provides centralized opacity management that works across all windows
and overlays, independent of any specific UI instance.
"""

from core.logging import get_logger
from core.settings.settings_manager import SettingsManager
from core.threading import ThreadManager
from core.hotkeys.manager import HotkeyManager
from utils.resource_manager import get_resource_manager, ResourceType
from PySide6.QtCore import QObject, Signal
import threading

class OpacityManager(QObject):
    """Centralized opacity manager for the application.
    
    This class handles opacity adjustments for all windows and overlays,
    using the settings manager for persistence and the hotkey manager for
    global hotkey registration. It's implemented as a singleton to ensure
    only one instance manages opacity system-wide.
    """
    
    # Timer adjustment interval in milliseconds
    # Reduced further for ~30% faster adjustment vs previous 24ms (now ~17ms)
    ADJUSTMENT_INTERVAL = 17
    
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
        # Resource registration for lifecycle management
        self._resource_id = None
        try:
            rm = get_resource_manager()
            self._resource_id = rm.register(
                self,
                ResourceType.CUSTOM,
                "OpacityManager singleton",
                cleanup_handler=lambda obj: obj.shutdown(),
                tags={"opacity", "manager"},
            )
        except Exception:
            # Best-effort registration
            self._resource_id = None
        # Normalize stored opacity to minimum 10% for self-healing of legacy values
        try:
            stored = self._settings_manager.get("appearance.opacity", 100)
            if isinstance(stored, (int, float)) and stored < 10:
                self._logger.debug(f"[OPACITY] Normalizing stored value from {stored}% to 10%")
                self._settings_manager.set("appearance.opacity", 10)
                # Save immediately to persist correction
                self._settings_manager.save()
        except Exception:
            pass
        # No per-instance ThreadManager; use centralized static API
        
        # Active flags for self-rescheduling ticks
        self._decrease_key_pressed = False
        self._increase_key_pressed = False
        
        # Deprecated timer placeholders removed (QTimer eliminated)
        
        # Current hotkey settings
        self._decrease_key = None
        self._increase_key = None
        
        # Hotkey IDs for centralized HotkeyManager
        self._hk_id_decrease = "opacity_decrease"
        self._hk_id_increase = "opacity_increase"

        # Tick indices for ~25% faster adjustment using a 1-1-1-2 schedule
        # Separate counters per direction to avoid cross-interference
        self._inc_tick_index = 0
        self._dec_tick_index = 0
        
        # Load hotkey settings and register global hotkeys
        self._load_hotkey_settings()
        self._register_hotkeys()
        
        self._initialized = True
        self._logger.debug("OpacityManager initialized")
    
    def _register_hotkeys(self) -> None:
        """Register opacity hotkeys via centralized HotkeyManager (no LL hooks)."""
        try:
            # Unregister existing hotkeys if any
            self._unregister_hotkeys()

            # Check if opacity hotkeys are enabled
            hotkeys_enabled = self._settings_manager.get('hotkeys.opacity_enabled', True)
            if not hotkeys_enabled:
                self._logger.debug("Opacity hotkeys are disabled, not registering hotkeys")
                return

            self._logger.debug(
                f"Registering hotkeys for opacity adjustment via HotkeyManager: decrease='{self._decrease_key}', increase='{self._increase_key}'"
            )

            # Validate tokens
            if not self._decrease_key or not isinstance(self._decrease_key, str):
                self._logger.warning("Invalid decrease hotkey value, skipping registration")
                return
            if not self._increase_key or not isinstance(self._increase_key, str):
                self._logger.warning("Invalid increase hotkey value, skipping registration")
                return

            hm = HotkeyManager()
            # Keyboard-backend safe single-key registrations with suppression
            hm.register_hotkey(
                self._hk_id_decrease,
                self._hk_on_decrease_press,
                sequence=self._decrease_key,
                suppress=True,
                global_hotkey=False,
                on_release=self._hk_on_decrease_release,
            )
            hm.register_hotkey(
                self._hk_id_increase,
                self._hk_on_increase_press,
                sequence=self._increase_key,
                suppress=True,
                global_hotkey=False,
                on_release=self._hk_on_increase_release,
            )

            self._logger.debug("Registered opacity hotkeys via HotkeyManager")

        except Exception as e:
            self._logger.error(f"Failed to register opacity hotkeys: {e}", exc_info=True)
    
    def _hk_on_decrease_press(self) -> None:
        """HotkeyManager press callback for decrease key."""
        if not self._decrease_key_pressed:
            self._logger.debug("Decrease opacity key pressed")
            self._decrease_key_pressed = True
            ThreadManager.run_on_ui_thread(self._start_decrease_opacity)
        
    def _hk_on_increase_press(self) -> None:
        """HotkeyManager press callback for increase key."""
        if not self._increase_key_pressed:
            self._logger.debug("Increase opacity key pressed")
            self._increase_key_pressed = True
            ThreadManager.run_on_ui_thread(self._start_increase_opacity)
            
    def _hk_on_decrease_release(self) -> None:
        """HotkeyManager release callback for decrease key."""
        if self._decrease_key_pressed:
            self._logger.debug("Decrease opacity key released")
            self._decrease_key_pressed = False
            ThreadManager.run_on_ui_thread(self._stop_decrease_opacity)
        
    def _hk_on_increase_release(self) -> None:
        """HotkeyManager release callback for increase key."""
        if self._increase_key_pressed:
            self._logger.debug("Increase opacity key released")
            self._increase_key_pressed = False
            ThreadManager.run_on_ui_thread(self._stop_increase_opacity)
            
    def _start_decrease_opacity(self) -> None:
        """Start decreasing opacity via self-rescheduling tick."""
        self._logger.debug("_start_decrease_opacity called")
        # Reset tick index for consistent 1-1-1-2 cadence
        self._dec_tick_index = 0
        # If already at lower bound, stop immediately
        try:
            if self.get_opacity() <= 10:
                self._decrease_key_pressed = False
                return
        except Exception:
            pass
        # Kick off the periodic tick if newly pressed
        if self._decrease_key_pressed:
            ThreadManager.single_shot(self.ADJUSTMENT_INTERVAL, self._decrease_tick)
            
    def _decrease_tick(self) -> None:
        """One tick of decreasing opacity; reschedules itself while key is held."""
        if not self._decrease_key_pressed:
            return
        self._decrease_opacity()
        # Stop at lower bound without looping
        try:
            if self.get_opacity() <= 10:
                self._decrease_key_pressed = False
                self._logger.debug("[OPACITY] Hit lower bound 10%; stopping decrease ticks")
                # Persist final value once when reaching bound
                try:
                    self._settings_manager.save()
                except Exception:
                    pass
                return
        except Exception:
            pass
        if self._decrease_key_pressed:
            ThreadManager.single_shot(self.ADJUSTMENT_INTERVAL, self._decrease_tick)
            
    def _start_increase_opacity(self) -> None:
        """Start increasing opacity via self-rescheduling tick."""
        self._logger.debug("_start_increase_opacity called")
        # Reset tick index for consistent 1-1-1-2 cadence
        self._inc_tick_index = 0
        # If already at upper bound, stop immediately
        try:
            if self.get_opacity() >= 100:
                self._increase_key_pressed = False
                return
        except Exception:
            pass
        if self._increase_key_pressed:
            ThreadManager.single_shot(self.ADJUSTMENT_INTERVAL, self._increase_tick)
            
    def _increase_tick(self) -> None:
        """One tick of increasing opacity; reschedules itself while key is held."""
        if not self._increase_key_pressed:
            return
        self._increase_opacity()
        # Stop at upper bound without looping
        try:
            if self.get_opacity() >= 100:
                self._increase_key_pressed = False
                self._logger.debug("[OPACITY] Hit upper bound 100%; stopping increase ticks")
                # Persist final value once when reaching bound
                try:
                    self._settings_manager.save()
                except Exception:
                    pass
                return
        except Exception:
            pass
        if self._increase_key_pressed:
            ThreadManager.single_shot(self.ADJUSTMENT_INTERVAL, self._increase_tick)
            
    def _stop_decrease_opacity(self) -> None:
        """Stop decreasing opacity."""
        # Persist the final opacity once on key release
        try:
            self._settings_manager.save()
        except Exception:
            pass
        
    # Internal timer stop methods removed (QTimer eliminated)
        
    def _stop_increase_opacity(self) -> None:
        """Stop increasing opacity."""
        # Persist the final opacity once on key release
        try:
            self._settings_manager.save()
        except Exception:
            pass
        
    # Internal timer stop methods removed (QTimer eliminated)
    
    def _increase_opacity(self) -> None:
        """Increase the opacity by a small amount for smooth adjustment."""
        current_opacity = self._settings_manager.get("appearance.opacity", 100)
        # 1-1-1-2 cadence -> average +25% speed over base step=1
        step = 2 if (self._inc_tick_index % 4 == 3) else 1
        self._inc_tick_index += 1
        new_opacity = min(100, current_opacity + step)
        self.set_opacity(new_opacity)
        
    def _decrease_opacity(self) -> None:
        """Decrease the opacity by a small amount for smooth adjustment."""
        current_opacity = self._settings_manager.get("appearance.opacity", 100)
        # 1-1-1-2 cadence -> average +25% speed over base step=1
        step = 2 if (self._dec_tick_index % 4 == 3) else 1
        self._dec_tick_index += 1
        new_opacity = max(10, current_opacity - step)
        self.set_opacity(new_opacity)
    
    def increase_opacity(self, amount: int = 5) -> None:
        """Increase the opacity by the specified amount."""
        current_opacity = self._settings_manager.get("appearance.opacity", 100)
        new_opacity = min(100, current_opacity + amount)
        self.set_opacity(new_opacity)
        
    def decrease_opacity(self, amount: int = 1) -> None:
        """Decrease the opacity by the specified amount."""
        current_opacity = self._settings_manager.get("appearance.opacity", 100)
        new_opacity = max(10, current_opacity - amount)
        self.set_opacity(new_opacity)
    
    def set_opacity(self, percent: int) -> None:
        """Set the opacity to the given percentage (10-100)."""
        percent = max(10, min(100, percent))
        # Read current to avoid redundant updates/logging
        current = self._settings_manager.get("appearance.opacity", 100)
        if current == percent:
            # No change; avoid emitting signals or logging to reduce noise
            return
        # Save and emit only when changed
        # Do NOT save immediately on each tick; batch-save on key release or bounds
        self._settings_manager.set("appearance.opacity", percent, save_immediately=False)
        # Log at bounds for validation
        if percent in (10, 100):
            self._logger.debug(f"[OPACITY] set -> {percent}%")
        self.opacityChanged.emit(percent)
    
    def get_opacity(self) -> int:
        """Get the current opacity value."""
        return self._settings_manager.get("appearance.opacity", 100)

    def _unregister_hotkeys(self) -> None:
        """Unregister opacity hotkeys via HotkeyManager."""
        try:
            hm = HotkeyManager()
            try:
                hm.unregister_hotkey(self._hk_id_decrease)
            except Exception:
                pass
            try:
                hm.unregister_hotkey(self._hk_id_increase)
            except Exception:
                pass
            self._logger.debug("Unregistered opacity hotkeys via HotkeyManager")
        except Exception as e:
            self._logger.error(f"Failed to unregister opacity hotkeys: {e}", exc_info=True)
    def _load_hotkey_settings(self) -> None:
        """Load opacity hotkey settings without clobbering user choices.

        Only set defaults when keys are missing/empty. Do not overwrite
        user selections programmatically.
        """
        try:
            valid_decrease = '-'
            valid_increase = '='

            # Load from settings or default
            dec = self._settings_manager.get('hotkeys.opacity_decrease', valid_decrease)
            inc = self._settings_manager.get('hotkeys.opacity_increase', valid_increase)

            # Backfill defaults if missing/empty
            if not isinstance(dec, str) or not dec.strip():
                dec = valid_decrease
                try:
                    self._settings_manager.set('hotkeys.opacity_decrease', valid_decrease)
                except Exception:
                    pass
            if not isinstance(inc, str) or not inc.strip():
                inc = valid_increase
                try:
                    self._settings_manager.set('hotkeys.opacity_increase', valid_increase)
                except Exception:
                    pass

            self._decrease_key = dec
            self._increase_key = inc
            self._logger.debug(
                f"Loaded opacity hotkeys: decrease='{self._decrease_key}', increase='{self._increase_key}'"
            )
        except Exception as e:
            self._logger.error(f"Failed to load opacity hotkey settings: {e}", exc_info=True)
            self._decrease_key = '-'
            self._increase_key = '='

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

    def shutdown(self) -> None:
        """Shutdown lifecycle: unregister hotkeys and best-effort resource cleanup."""
        try:
            self._decrease_key_pressed = False
            self._increase_key_pressed = False
        except Exception:
            pass
        try:
            self._unregister_hotkeys()
        except Exception:
            pass
        # Best-effort unregister from ResourceManager
        try:
            if getattr(self, "_resource_id", None) is not None:
                rm = get_resource_manager()
                rm.unregister(self._resource_id, force=True)
                self._resource_id = None
        except Exception:
            pass

# Convenience function to get the singleton instance
def get_opacity_manager() -> OpacityManager:
    """Get the singleton instance of the opacity manager."""
    return OpacityManager()
