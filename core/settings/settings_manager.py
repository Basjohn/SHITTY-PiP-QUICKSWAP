"""
Implementation of the settings management system.

This module contains the core implementation of the settings manager,
handling loading, saving, and accessing application settings with
thread safety and type checking.
"""

import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from PySide6.QtCore import QObject, Signal

from .. import logging as core_logging
from core.interfaces import ISettingsManager
from .types import SettingDefinition, SettingsCategory

get_logger = core_logging.get_logger

# Type variables
T = TypeVar('T')


class _SettingsManagerImpl(ISettingsManager):
    """Internal implementation of the settings manager.
    
    This class handles the core settings logic without Qt dependencies,
    making it easier to test and maintain.
    """
    
    def __init__(self):
        """Initialize the settings manager with default values."""
        self._settings: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._settings_definitions: Dict[str, SettingDefinition] = {}
        self._logger = get_logger(__name__)
        self._load_defaults()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value.
        
        Args:
            key: Setting key in dot notation (e.g., 'app.theme.color')
            default: Default value to return if key is not found
            
        Returns:
            The setting value or default if not found
        """
        with self._lock:
            return self._settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a setting value.
        
        Args:
            key: Setting key in dot notation
            value: Value to set
        """
        with self._lock:
            self._settings[key] = value
    
    def save(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Save settings to persistent storage.
        
        Args:
            file_path: Optional path to save settings file. If None, uses default location.
        """
        if file_path is None:
            if hasattr(self, '_settings_file') and self._settings_file is not None:
                file_path = self._settings_file
            else:
                settings_dir = Path.home() / '.spqmodular'
                settings_dir.mkdir(exist_ok=True)
                file_path = settings_dir / 'settings.json'
        else:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock, open(file_path, 'w') as f:
            json.dump(self._settings, f, indent=4)
    
    def load(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """Load settings from persistent storage.
        
        Args:
            file_path: Optional path to load settings from. If None, uses default location.
        """
        if file_path is None:
            file_path = Path.home() / '.spqmodular' / 'settings.json'
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                loaded_settings = json.load(f)
                
            with self._lock:
                # Only update existing keys to preserve defaults for missing settings
                for key, value in loaded_settings.items():
                    if key in self._settings_definitions:
                        self._settings[key] = value
                        
        except (json.JSONDecodeError, OSError) as e:
            get_logger(__name__).error(f"Failed to load settings: {e}")
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        with self._lock:
            self._load_defaults()
    
    def get_setting_definition(self, key: str) -> Optional[SettingDefinition]:
        """Get the definition for a setting.
        
        Args:
            key: Setting key
            
        Returns:
            SettingDefinition if found, None otherwise
        """
        return self._settings_definitions.get(key)
    
    def _load_defaults(self) -> None:
        """Load default settings definitions and values."""
        self._settings_definitions = {
            # Appearance
            'theme': SettingDefinition(
                default='dark',
                setting_type=str,
                options=['light', 'dark'],
                description='Application color theme (canonical key)',
                category=SettingsCategory.APPEARANCE
            ),
            'appearance.opacity': SettingDefinition(
                default=100,
                setting_type=int,
                validator=lambda x: 0 <= x <= 100,
                description='Opacity of the overlay windows (0-100%)',
                category=SettingsCategory.APPEARANCE
            ),
            # Overlay visual options
            'overlay.rounded_borders': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Use rounded corners for overlay border rendering',
                category=SettingsCategory.APPEARANCE
            ),
            # UI state persistence
            'ui.main_window_geometry': SettingDefinition(
                default={'x': 0, 'y': 0, 'width': 800, 'height': 500, 'maximized': False},
                setting_type=dict,
                validator=lambda g: isinstance(g, dict)
                    and isinstance(g.get('x', 0), int)
                    and isinstance(g.get('y', 0), int)
                    and isinstance(g.get('width', 0), int)
                    and isinstance(g.get('height', 0), int)
                    and isinstance(g.get('maximized', False), bool),
                description='Main window geometry and state',
                category=SettingsCategory.GENERAL
            ),
            'ui.badge_file': SettingDefinition(
                default='Badge19.png',
                setting_type=str,
                description='Selected badge image filename',
                category=SettingsCategory.GENERAL
            ),
            'ui.mode_button_state': SettingDefinition(
                default='window',
                setting_type=str,
                options=['window', 'monitor'],
                description='Selected mode button state',
                category=SettingsCategory.GENERAL
            ),
            # Feature toggles
            'features.autoswitch_enabled': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Enable Autoswitch controller',
                category=SettingsCategory.BEHAVIOR
            ),
            'features.keypassthrough_enabled': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Enable key passthrough for overlays',
                category=SettingsCategory.BEHAVIOR
            ),
            'features.keypassthrough_blocklist_enabled': SettingDefinition(
                default=True,
                setting_type=bool,
                description='Enable keypassthrough blocklist feature',
                category=SettingsCategory.BEHAVIOR
            ),
            'features.display_locked_switching': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Restrict window switching to same monitor as overlay source',
                category=SettingsCategory.BEHAVIOR
            ),
            'features.media_control_enabled': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Enable media key routing and media controller features',
                category=SettingsCategory.BEHAVIOR
            ),
            # Debug/diagnostics (verbose logging controls)
            'debug.keypassthrough_verbose': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Verbose logging for key passthrough routing decisions',
                category=SettingsCategory.GENERAL
            ),
            'debug.window_filter_verbose': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Verbose logging for window filtering exclusions (WindowFilter)',
                category=SettingsCategory.GENERAL
            ),
            'debug.volume_osd_verbose': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Verbose logging for Volume OSD widget visibility and updates',
                category=SettingsCategory.GENERAL
            ),
            # Hotkeys
            'hotkeys.opacity_enabled': SettingDefinition(
                default=True,
                setting_type=bool,
                description='Enable opacity hotkeys',
                category=SettingsCategory.HOTKEYS
            ),
            'hotkeys.opacity_decrease': SettingDefinition(
                default='-',
                setting_type=str,
                description='Opacity decrease key (single-key suppression backend)',
                category=SettingsCategory.HOTKEYS
            ),
            'hotkeys.opacity_increase': SettingDefinition(
                default='=',
                setting_type=str,
                description='Opacity increase key (single-key suppression backend)',
                category=SettingsCategory.HOTKEYS
            ),
            'hotkeys.opacity_quickswitch': SettingDefinition(
                default='`',
                setting_type=str,
                description='Quickswitch key (backtick by default)',
                category=SettingsCategory.HOTKEYS
            ),
            # Graphics pipeline selection (feature flag)
            'graphics.pipeline': SettingDefinition(
                default='dxgi',
                setting_type=str,
                options=['dxgi'],
                description='Capture/renderer pipeline selection (dxgi only)',
                requires_restart=True,
                category=SettingsCategory.EXPERIMENTAL
            ),
            # Graphics presentation selection (renderer presentation path)
            'graphics.presentation': SettingDefinition(
                default='cpu-blit',
                setting_type=str,
                options=['cpu-blit', 'd3d11-swapchain'],
                description='Renderer presentation path: CPU QImage blit or D3D11 swapchain host',
                requires_restart=False,
                category=SettingsCategory.EXPERIMENTAL
            ),
            
            # Behavior
            'behavior.auto_switch': SettingDefinition(
                default=True,
                setting_type=bool,
                description='Automatically switch to the most recently used window',
                category=SettingsCategory.BEHAVIOR
            ),
            'behavior.click_through': SettingDefinition(
                default=False,
                setting_type=bool,
                description='Allow mouse clicks to pass through the overlay',
                category=SettingsCategory.BEHAVIOR
            ),
            
            # Performance
            'performance.cache_size': SettingDefinition(
                default=100,
                setting_type=int,
                validator=lambda x: x > 0,
                description='Maximum number of items to cache',
                category=SettingsCategory.PERFORMANCE
            ),
            'performance.threads': SettingDefinition(
                default=4,
                setting_type=int,
                validator=lambda x: 1 <= x <= 32,
                description='Number of worker threads to use',
                category=SettingsCategory.PERFORMANCE
            )
        }
        
        # Capture preferences
        self._settings_definitions['capture.fps'] = SettingDefinition(
            default=60,
            setting_type=int,
            validator=lambda x: isinstance(x, int) and 1 <= x <= 165,
            description='Target monitor capture frames per second (1-165)',
            category=SettingsCategory.PERFORMANCE
        )
        
        # Set default values
        for key, definition in self._settings_definitions.items():
            self._settings[key] = definition.default


class SettingsManager(QObject):
    """
    Qt-compatible settings manager with signals for setting changes.
    
    This class provides a Qt-compatible interface to the settings system,
    with support for change notifications via signals.
    """
    
    # Signal emitted when a setting changes
    setting_changed = Signal(str, object)
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton behavior."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, parent: Optional[QObject] = None, settings_file: Optional[Union[str, Path]] = None):
        """Initialize the settings manager."""
        if SettingsManager._initialized:
            return
            
        super().__init__(parent)
        self._impl = _SettingsManagerImpl()
        self._logger = get_logger(__name__)
        self._lock = threading.RLock()
        self._handlers = {}
        self._settings_file = settings_file
        
        # Load saved settings
        self._impl.load(settings_file)
        # Apply explicit migrations, then validate
        self._apply_migrations()
        self._validate_loaded_settings()

        # Ensure keypassthrough blocklist exists alongside the settings file
        try:
            self._ensure_keypassthrough_blocklist_defaults()
        except Exception as e:
            # Non-fatal: log explicitly; creation is best-effort
            self._logger.error(f"[KEYPASS] Failed to ensure blocklist defaults: {e}", exc_info=True)
        
        SettingsManager._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value with type conversion.
        
        Args:
            key: Setting key in dot notation (e.g., 'app.theme.color')
            default: Default value to return if key is not found
            
        Returns:
            The setting value, converted to the specified type if possible
        """
        return self._impl.get(key, default)
    
    def set(self, key: str, value: Any, save_immediately: bool = True) -> bool:
        """Set a setting value.
        
        Args:
            key: Setting key in dot notation
            value: Value to set
            save_immediately: If True, save to disk immediately
            
        Returns:
            bool: True if the setting was updated, False otherwise
            
        Raises:
            ValueError: If the value is not valid for the setting
        """
        definition = self._impl.get_setting_definition(key)
        if definition is not None:
            # Validate the value
            if not isinstance(value, definition.setting_type):
                try:
                    value = definition.setting_type(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid value for setting {key}: {value}. "
                        f"Expected {definition.setting_type.__name__}"
                    ) from e
            
            if definition.validator and not definition.validator(value):
                raise ValueError(f"Invalid value for setting {key}: {value}")
            
            if definition.options and value not in definition.options:
                raise ValueError(
                    f"Invalid value for setting {key}. "
                    f"Must be one of: {', '.join(map(str, definition.options))}"
                )
        
        # Update the setting without alias mirroring (canonical only)
        with self._lock:
            changed = False
            old_value = self._impl.get(key)
            if old_value != value:
                self._impl.set(key, value)
                changed = True
                # Notify listeners for the changed key
                self.setting_changed.emit(key, value)

                # Call any registered handlers for the key
                if key in self._handlers:
                    for handler in self._handlers[key]:
                        try:
                            handler(key, value)
                        except Exception as e:
                            self._logger.error(
                                f"Error in setting change handler for {key}: {e}",
                                exc_info=True
                            )

            if changed and save_immediately:
                self.save()

            return changed
    
    def save(self) -> None:
        """Save settings to persistent storage."""
        if hasattr(self, '_settings_file') and self._settings_file is not None:
            self._impl.save(self._settings_file)
        else:
            self._impl.save()
    
    def load(self) -> None:
        """Load settings from persistent storage."""
        self._impl.load()
        # After load, migrate, validate, then reconcile without emitting extra signals
        self._apply_migrations()
        self._validate_loaded_settings()
        # No alias reconciliation; only canonical 'theme' is recognized
    
    def get_settings_dir(self) -> Path:
        """Return the directory where settings and auxiliary files live.

        This is a public wrapper over the internal resolver used for colocating
        auxiliary files such as the key passthrough blocklist.
        """
        try:
            return self._resolve_settings_dir()
        except Exception:
            # Best-effort fallback
            from pathlib import Path as _P
            return _P.home() / '.spqmodular'
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values."""
        with self._lock:
            self._impl.reset_to_defaults()
            
            # Notify listeners of all changes
            for key in self._impl._settings_definitions.keys():
                value = self._impl.get(key)
                self.setting_changed.emit(key, value)
            
            self.save()
    
    def register_change_handler(self, key: str, handler: Callable[[str, Any], None]) -> None:
        """Register a callback for when a setting changes.
        
        Args:
            key: Setting key to monitor
            handler: Callback function that takes (key, value) parameters
        """
        with self._lock:
            if key not in self._handlers:
                self._handlers[key] = []
            self._handlers[key].append(handler)
    
    def unregister_change_handler(self, key: str, handler: Callable[[str, Any], None]) -> None:
        """Unregister a previously registered change handler.
        
        Args:
            key: Setting key
            handler: Callback function to remove
        """
        with self._lock:
            if key in self._handlers and handler in self._handlers[key]:
                self._handlers[key].remove(handler)
                if not self._handlers[key]:
                    del self._handlers[key]
    
    def get_setting_definition(self, key: str) -> Optional[SettingDefinition]:
        """Get the definition for a setting.
        
        Args:
            key: Setting key
            
        Returns:
            SettingDefinition if found, None otherwise
        """
        return self._impl.get_setting_definition(key)
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary.
        
        Returns:
            Dictionary of all setting key-value pairs
        """
        return dict(self._impl._settings)
    
    def get_settings_by_category(self, category: SettingsCategory) -> Dict[str, Any]:
        """Get all settings in a specific category.
        
        Args:
            category: Settings category
            
        Returns:
            Dictionary of setting key-value pairs in the specified category
        """
        return {
            key: value for key, value in self._impl._settings.items()
            if (defn := self._impl.get_setting_definition(key)) and defn.category == category
        }

    # --- Internal helpers -------------------------------------------------
    def _apply_migrations(self) -> None:
        """Apply explicit, deterministic migrations to persisted settings.

        Current migrations:
        - Map legacy theme value 'system' to 'dark' for canonical 'theme'.
        - Map legacy graphics.pipeline value 'wgc-d3d11' to 'dxgi'.
        """
        with self._lock:
            migrated = False
            v = self._impl.get('theme')
            if v == 'system':
                self._logger.info("Migrating theme from 'system' to 'dark'")
                self._impl.set('theme', 'dark')
                migrated = True
            gp = self._impl.get('graphics.pipeline')
            if gp == 'wgc-d3d11':
                self._logger.info("Migrating graphics.pipeline from 'wgc-d3d11' to 'dxgi'")
                self._impl.set('graphics.pipeline', 'dxgi')
                migrated = True
            if migrated:
                # Do not emit signals here; reconciliation and save will follow
                pass
    def _validate_loaded_settings(self) -> None:
        """Validate persisted settings strictly against definitions.

        Raises:
            ValueError: If any persisted value violates type, validator, or options.
        """
        with self._lock:
            for k, v in list(self._impl._settings.items()):
                defn = self._impl.get_setting_definition(k)
                if not defn:
                    continue
                # Enforce exact type, no silent coercion
                if not isinstance(v, defn.setting_type):
                    raise ValueError(
                        f"Invalid type for setting {k}: {type(v).__name__}, expected {defn.setting_type.__name__}"
                    )
                if defn.validator and not defn.validator(v):
                    raise ValueError(f"Invalid value for setting {k}: {v}")
                if defn.options and v not in defn.options:
                    opts = ', '.join(map(str, defn.options))
                    raise ValueError(f"Invalid value for setting {k}. Must be one of: {opts}")



    # --- Blocklist defaults (KeyPassthrough) ---------------------------------
    def _resolve_settings_dir(self) -> Path:
        """Resolve the directory where settings and auxiliary files live.

        If a specific settings file path was provided, use its parent directory.
        Otherwise, fall back to the user configuration directory (~/.spqmodular).
        """
        try:
            if hasattr(self, '_settings_file') and self._settings_file:
                p = Path(self._settings_file)
                return p.parent
        except Exception:
            pass
        return Path.home() / '.spqmodular'

    def _ensure_keypassthrough_blocklist_defaults(self) -> None:
        """Create a default key passthrough blocklist file if it doesn't exist.

        - Location: same directory as settings.json (or ~/.spqmodular if not set)
        - Encoding: UTF-8
        - Content: documented header + ~24+ widely-used anti-cheat game executables
        """
        settings_dir = self._resolve_settings_dir()
        try:
            settings_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Directory creation failure will be surfaced when opening the file
            pass

        blocklist_path = settings_dir / 'keypassthrough_blocklist.txt'
        if blocklist_path.exists():
            self._logger.debug(f"[KEYPASS] Blocklist present at: {blocklist_path}")
            return

        header = (
            "# KeyPassthrough Blocklist\n"
            "#\n"
            "# How to use (one entry per line):\n"
            "# - You can write plain lines (no JSON) OR a JSON object per line.\n"
            "# - Plain line rules:\n"
            "#   * Ends with .exe  => match by process name (exe), case-insensitive\n"
            "#   * Otherwise       => match window title (contains); if multiple words, ALL must be present\n"
            "# - JSON object rules (advanced):\n"
            "#   {\"exe\": \"cs2.exe\"}\n"
            "#   {\"title_exact\": \"PUBERTY SIMULATOR\"}\n"
            "#   {\"title_contains\": \"genshin impact\"}\n"
            "# - Comments start with # and are ignored. Blank lines are OK. Trailing commas on a line are ignored.\n"
            "#\n"
            "# Default entries (popular online games with anti-cheat):\n"
        )

        # Keep entries conservative and process-name specific
        default_entries = [
            "cs2.exe",
            "VALORANT-Win64-Shipping.exe",
            "FortniteClient-Win64-Shipping.exe",
            "r5apex.exe",
            "TslGame.exe",
            "Overwatch.exe",
            "GenshinImpact.exe",
            "RustClient.exe",
            "DeadByDaylight-Win64-Shipping.exe",
            "EscapeFromTarkov.exe",
            "RainbowSix.exe",
            "Destiny2.exe",
            "FallGuys_client.exe",
            "BF2042.exe",
            "BFV.exe",
            "BF1.exe",
            "NewWorld.exe",
            "ELDENRING.exe",
            "Paladins.exe",
            "Warframe.x64.exe",
            "HaloInfinite.exe",
            "HuntGame.exe",
            "LOSTARK.exe",
            "Prospect-Win64-Shipping.exe",
        ]

        try:
            with open(blocklist_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(header)
                for entry in default_entries:
                    f.write(entry + "\n")
            self._logger.info(
                f"[KEYPASS] Created default key passthrough blocklist at {blocklist_path} with {len(default_entries)} entries"
            )
        except Exception as e:
            self._logger.error(f"[KEYPASS] Failed to write blocklist defaults to {blocklist_path}: {e}", exc_info=True)

