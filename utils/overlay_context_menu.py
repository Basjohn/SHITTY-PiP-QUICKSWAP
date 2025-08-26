"""
overlay_context_menu.py

Unified context menu builder for all overlay types in SPQ: window, monitor, and DWM overlays.
Handles menu creation, population, theming, action callbacks, and border preservation.

Usage:
    # Basic usage
    menu_handler = OverlayContextMenu(overlay_widget, overlay_type='window'|'monitor'|'dwm')
    menu_handler.show_menu(event.globalPos())
    
    # Or attach directly to an overlay's contextMenuEvent
    menu_handler.attach_to_overlay(overlay_widget)
    
    # For DWM overlays with border overlay
    menu_handler = OverlayContextMenu(overlay_widget, overlay_type='dwm', 
                                     border_overlay=border_overlay)

The menu handler automatically handles border visibility for DWM overlays.
"""

from __future__ import annotations

from core.logging import get_logger
from utils.debug import debug_enabled
from utils.resource_manager import get_resource_manager
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QObject, QEvent
from typing import Callable, Optional, Dict, Any
from utils.window_menu_utils import populate_window_switch_menu

class OverlayContextMenu:
    """Unified context menu handler for all overlay types.
    
    This class manages context menus for window, monitor, and DWM overlays,
    providing consistent behavior, theming, and proper border preservation.
    """
    
    def __init__(self, overlay_widget, overlay_type='window', config=None, border_overlay=None):
        """Initialize the context menu handler.
        
        Args:
            overlay_widget: The overlay widget this menu belongs to
            overlay_type: Type of overlay ('window', 'monitor', or 'dwm')
            config: Optional configuration dict
            border_overlay: Optional BorderOverlay instance (for DWM overlays)
        """
        # Keep the originally provided widget (often the host QWidget)
        self._overlay_widget = overlay_widget
        # Resolve to the real overlay instance if a host with `_parent_overlay` was provided
        self.overlay = self._resolve_overlay(overlay_widget)
        self.overlay_type = overlay_type.lower()  # 'window', 'monitor', or 'dwm'
        self.config = config or {}
        self.border_overlay = None  # Legacy field kept for API compatibility
        self.menu = None
        self.switch_to_window_menu = None
        self.switch_to_monitor_menu = None
        self.lock_action = None
        self._actions = {}
        self._logger = get_logger("OverlayContextMenu")
        # Track installed event filters so we can detach cleanly. Each entry is (target, filter).
        self._filters: list[tuple[QObject, QObject]] = []
        # Optional explicit callbacks for actions (preferred over overlay bound methods)
        # Expected keys: 'lock', 'show_settings', 'show_sub_settings', 'hide', 'reset', 'quit'
        self._callbacks: Dict[str, Callable[..., Any]] = {}
        try:
            actions = (self.config or {}).get('actions', {})
            if isinstance(actions, dict):
                # Only keep callables
                self._callbacks = {k: v for k, v in actions.items() if callable(v)}
        except Exception:
            # Ensure we never crash ctor; callbacks remain empty
            self._callbacks = {}
        # Debug log the resolution result
        if debug_enabled and self.overlay is not overlay_widget:
            try:
                self._logger.debug(f"CTX_MENU: Resolved overlay from host {type(overlay_widget).__name__} to {type(self.overlay).__name__}")
            except Exception:
                pass

    def _resolve_overlay(self, obj):
        """Return the real overlay instance.
        If a host QWidget is passed, prefer its `_parent_overlay` reference.
        """
        try:
            parent_overlay = getattr(obj, '_parent_overlay', None)
            if parent_overlay is not None:
                return parent_overlay
        except Exception:
            # Fall through to return the object itself
            pass
        return obj

    def _wire_action(self, action: QAction, key: str, overlay_attr: Optional[str] = None) -> None:
        """Wire an action to either an injected callback or an explicit overlay method.
        If neither exists, the action will be disabled and a debug message logged.
        """
        try:
            if key in self._callbacks:
                action.triggered.connect(self._callbacks[key])
                return
            if overlay_attr and hasattr(self.overlay, overlay_attr):
                action.triggered.connect(getattr(self.overlay, overlay_attr))
                return
            # Neither provided nor available on overlay: disable
            action.setEnabled(False)
            if debug_enabled:
                self._logger.debug(f"CTX_MENU: No handler for action '{key}' (attr={overlay_attr}) - disabled")
        except Exception as e:
            self._logger.error(f"CTX_MENU: Failed to wire action '{key}': {e}")

    def build_menu(self):
        """Constructs and returns a QMenu for the overlay."""
        # Use a QWidget parent if available; DWMOverlay itself is not a QWidget
        try:
            parent_widget = self.overlay if hasattr(self.overlay, 'winId') else getattr(self.overlay, '_host', None)
        except Exception:
            parent_widget = None
        
        try:
            # Create and configure the menu
            self.menu = QMenu(parent_widget)
            self.menu.setObjectName("overlayContextMenu")
            self._current_menu = self.menu  # Store reference for z-order management
            
            # Hook menu lifecycle to centralized z-order management via ResourceManager
            try:
                self.menu.aboutToShow.connect(lambda: self._ensure_border_visible(getattr(self.overlay, 'id', None), before=True))
                self.menu.aboutToHide.connect(lambda: self._ensure_border_visible(getattr(self.overlay, 'id', None), after=True))
            except Exception as e:
                self._logger.warning(f"Failed to connect menu lifecycle hooks: {e}")

            # Apply theme
            try:
                from utils.theme import get_theme_manager
                theme_manager = get_theme_manager()
                theme_manager.apply_theme_to_widget(self.menu)
            except Exception as e:
                self._logger.warning(f"Failed to apply theme to context menu: {e}")
            
            # --- Switch to Window (if supported) ---
            if self.overlay_type == 'window' or self.config.get('show_switch_to_window', True):
                if self.overlay_type in ('window', 'dwm'):
                    switch_to_window_action = QAction("Switch To Window", self.menu)
                    switch_to_window_action.setEnabled(True)
                    def _populate_window_menu():
                        try:
                            self.populate_switch_to_window_menu()
                        except Exception as e:
                            self._logger.error(f"Error populating window menu: {e}")
                    self.switch_to_window_menu = self.menu.addMenu("Switch To Window")
                    self.switch_to_window_menu.aboutToShow.connect(_populate_window_menu)
            # --- Switch to Monitor (if supported) ---
            if self.overlay_type in ('window', 'monitor') and self.config.get('show_switch_to_monitor', True):
                if self.overlay_type in ('window', 'monitor'):
                    switch_to_monitor_action = QAction("Switch To Monitor", self.menu)
                    switch_to_monitor_action.setEnabled(True)
                    def _populate_monitor_menu():
                        try:
                            self.populate_switch_to_monitor_menu()
                        except Exception as e:
                            self._logger.error(f"Error populating monitor menu: {e}")
                    self.switch_to_monitor_menu = self.menu.addMenu("Switch To Monitor")
                    self.switch_to_monitor_menu.aboutToShow.connect(_populate_monitor_menu)
            self.menu.addSeparator()
            # --- Lock Overlay (window overlays only) ---
            if self.overlay_type == 'window':
                self.lock_action = QAction("Lock Overlay", self.menu)
                self.lock_action.setCheckable(True)
                self.lock_action.setChecked(getattr(self.overlay, '_is_window_locked', False))
                self._wire_action(self.lock_action, 'lock', 'toggle_window_lock')
                self.menu.addAction(self.lock_action)
                self._actions['lock'] = self.lock_action
                self.menu.addSeparator()
            # --- Settings, Subsettings ---
            show_settings_action = QAction("Main Window", self.menu)
            # Prefer injected callback if provided, otherwise use centralized handler
            if 'show_settings' in self._callbacks:
                show_settings_action.triggered.connect(self._callbacks['show_settings'])
            else:
                show_settings_action.triggered.connect(self._handle_show_main_window)
            self.menu.addAction(show_settings_action)
            self._actions['show_settings'] = show_settings_action
            show_sub_settings_action = QAction("Subsettings", self.menu)
            if 'show_sub_settings' in self._callbacks:
                show_sub_settings_action.triggered.connect(self._callbacks['show_sub_settings'])
            else:
                show_sub_settings_action.triggered.connect(self._handle_show_sub_settings)
            self.menu.addAction(show_sub_settings_action)
            self._actions['show_sub_settings'] = show_sub_settings_action
            self.menu.addSeparator()
            # --- Hide, Reset, Quit ---
            hide_action = QAction("Hide", self.menu)
            self._wire_action(hide_action, 'hide', 'close')
            self.menu.addAction(hide_action)
            self._actions['hide'] = hide_action
            reset_action = QAction("Reset", self.menu)
            self._wire_action(reset_action, 'reset', '_handle_reset_position')
            self.menu.addAction(reset_action)
            self._actions['reset'] = reset_action
            self.menu.addSeparator()
            quit_app_action = QAction("Quit Application", self.menu)
            self._wire_action(quit_app_action, 'quit', '_handle_quit_application')
            self.menu.addAction(quit_app_action)
            self._actions['quit'] = quit_app_action
            return self.menu
        except Exception as e:
            self._logger.error(f"Failed to build context menu: {e}", exc_info=True)
            return None

    # --- Centralized handlers for Main Window and Subsettings ---------------
    def _handle_show_main_window(self) -> None:
        """Locate MainDialog and bring it to front; no silent fallbacks."""
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                raise RuntimeError("No QApplication instance available")
            try:
                from ui.main_dialog import MainDialog  # type: ignore
            except Exception as e:
                self._logger.error(f"CTX_MENU: Could not import MainDialog: {e}")
                raise
            target = None
            for w in app.topLevelWidgets():
                try:
                    if isinstance(w, MainDialog):
                        target = w
                        break
                except Exception:
                    continue
            if target is None:
                raise RuntimeError("MainDialog not found among top-level widgets")
            target.show()
            try:
                target.raise_()
                target.activateWindow()
            except Exception:
                pass
            if debug_enabled:
                self._logger.debug("CTX_MENU: Main Window shown via centralized handler")
        except Exception as e:
            self._logger.error(f"CTX_MENU: Failed to show Main Window: {e}")
            raise

    def _handle_show_sub_settings(self) -> None:
        """Locate MainDialog and open Subsettings dialog; no silent fallbacks."""
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                raise RuntimeError("No QApplication instance available")
            try:
                from ui.main_dialog import MainDialog  # type: ignore
            except Exception as e:
                self._logger.error(f"CTX_MENU: Could not import MainDialog: {e}")
                raise
            target = None
            for w in app.topLevelWidgets():
                try:
                    if isinstance(w, MainDialog):
                        target = w
                        break
                except Exception:
                    continue
            if target is None:
                raise RuntimeError("MainDialog not found; cannot open Subsettings")
            if hasattr(target, 'show_sub_settings') and callable(getattr(target, 'show_sub_settings')):
                target.show_sub_settings()
            elif hasattr(target, '_open_subsettings_dialog') and callable(getattr(target, '_open_subsettings_dialog')):
                target._open_subsettings_dialog()
            else:
                raise AttributeError("MainDialog does not expose a subsettings opener")
            if debug_enabled:
                self._logger.debug("CTX_MENU: Subsettings opened via centralized handler")
        except Exception as e:
            self._logger.error(f"CTX_MENU: Failed to open Subsettings: {e}")
            raise

    def apply_theme(self, theme=None, from_global=False):
        """Apply the current theme to the menu and submenus.
        
        Args:
            theme (str, optional): Theme name ('light' or 'dark'). If None, uses overlay's theme.
            from_global (bool): If True, indicates this is part of a global theme change.
        """
        if not self.menu:
            return
            
        # Get theme from parameter or overlay, default to 'dark'
        if theme is None:
            theme = getattr(self.overlay, 'theme', 'dark')
            
        # Normalize theme name
        theme = str(theme).lower().strip()
        if theme not in ['light', 'dark']:
            theme = 'dark'
            
        # Define colors based on theme - matching the system tray menu
        if theme == 'light':
            border_color = '#000000'  # Black border
            background_color = '#f0f0f0'  # Light gray background
            text_color = '#000000'  # Black text
            highlight_color = '#7a7a7a'  # 40% darker than original (was 20%)
            highlight_text = '#ffffff'  # White text on hover (inverted)
        else:  # Dark theme
            border_color = '#ffffff'  # White border
            background_color = '#2a2a2a'  # 10% brighter than #1a1a1a
            text_color = '#ffffff'  # White text
            highlight_color = '#3a3a3a'  # Darker than background for visibility
            highlight_text = '#ffffff'  # White text on highlight
        
        # Apply stylesheet to main menu and all submenus
        stylesheet = f"""
            /* Base menu styling */
            QMenu {{
                background-color: {background_color};
                color: {text_color};
                border: 1px solid {border_color};
                padding: 1px;
                min-width: 100px;
                font-weight: 500;
            }}
            
            /* Menu item styling */
            QMenu::item {{
                padding: 4px 15px 4px 5px;  /* Increased right padding by 2px */
                border: none;
                text-align: left;
                font-weight: 500;
                margin: 0;
                margin-left: 2px;  /* Added 1px more left margin (total 2px) */
                margin-right: -1px;  /* Pull checkmark 1px to the left */
                spacing: 3px;
                padding-left: 2px;  /* Add 2px left padding for icons */
            }}
            
            /* Checkmark circle - Qt compatible */
            QMenu::indicator {{
                width: 6px;
                height: 6px;
                border: 1px solid {text_color};
                border-radius: 3px;
                background: transparent;
                margin-right: 2px;
            }}
            
            /* Icon padding */
            QMenu::icon {{
                padding-left: 2px;
            }}
            
            QMenu::indicator:checked {{
                background: {text_color};
            }}
            
            /* Hover state */
            QMenu::item:selected {{
                background-color: {highlight_color};
                color: {highlight_text};
            }}
            
            /* Submenu arrow */
            QMenu::right-arrow {{
                right: 8px;
                width: 8px;
                height: 8px;
            }}
            
            /* Separator */
            QMenu::separator {{
                height: 1px;
                background: {border_color};
                margin: 2px 4px;
            }}
        """
        
        # Apply to main menu
        self.menu.setStyleSheet(stylesheet)
        
        # Apply to submenus with original width
        submenu_stylesheet = stylesheet.replace('min-width: 100px', 'min-width: 96px')
        
        if hasattr(self, 'switch_to_window_menu') and self.switch_to_window_menu:
            self.switch_to_window_menu.setStyleSheet(submenu_stylesheet)
        if hasattr(self, 'switch_to_monitor_menu') and self.switch_to_monitor_menu:
            self.switch_to_monitor_menu.setStyleSheet(submenu_stylesheet)
            
        # Don't override with theme colors
        if self.lock_action:
            self.lock_action.setChecked(getattr(self.overlay, '_is_window_locked', False))

    def show_menu(self, position=None):
        """Show the context menu for this overlay.
        
        Args:
            position: QPoint or (x, y) tuple where to show the menu. If None, uses cursor pos.
            
        Returns:
            True if a menu was shown, False otherwise
        """
        if not self.overlay or not self.menu:
            if debug_enabled:
                self._logger.debug(f"CTX_MENU: Cannot show menu - overlay={self.overlay is not None}, menu={self.menu is not None}")
            return False
            
        try:
            if debug_enabled:
                self._logger.debug(f"CTX_MENU: Showing menu for {self.overlay_type} overlay")
            
            # Before showing menu, notify ResourceManager to begin context menu session
            if debug_enabled:
                self._logger.debug("CTX_MENU: Begin context menu session BEFORE exec_")
            overlay_id = getattr(self.overlay, 'id', None)
            self._ensure_border_visible(overlay_id, before=True)
            
            # Show menu at position
            if position is None:
                from PySide6.QtGui import QCursor
                position = QCursor.pos()
            
            # Convert position to QPoint if it's a tuple
            if isinstance(position, tuple):
                from PySide6.QtCore import QPoint
                position = QPoint(position[0], position[1])
                
            if debug_enabled:
                self._logger.debug(f"CTX_MENU: Executing menu at position {position.x()}, {position.y()}")

            # Populate any dynamic submenus just-in-time
            self._populate_dynamic_menus()

            # Show the menu
            action = self.menu.exec_(position)
            
            if debug_enabled:
                self._logger.debug(f"CTX_MENU: Menu closed, action={action is not None}")
            
            # Post-hide enforcement is handled by menu.aboutToHide hook
            
            # Actions are already wired to their callbacks via _wire_action
            # The menu was successfully shown regardless of whether an action was selected
            return True
        except Exception as e:
            self._logger.error(f"Error showing context menu: {e}")
            return False

    def _ensure_border_visible(self, overlay_id=None, before=False, after=False):
        """Centralized context menu lifecycle delegation to ResourceManager.
        
        Args:
            overlay_id: Optional ID of the overlay (extracted from self.overlay if None)
            before: If True, begins a context menu session (priority elevation)
            after: If True, ends a context menu session (restore normal)
        """
        try:
            if not overlay_id:
                overlay_id = getattr(self.overlay, 'id', None)
                if not overlay_id:
                    return
            rm = get_resource_manager()
            menu_obj = getattr(self, '_current_menu', None) or getattr(self, 'menu', None)
            if before and menu_obj is not None:
                rm.begin_context_menu(overlay_id, menu_obj)
            elif after and menu_obj is not None:
                rm.end_context_menu(overlay_id, menu_obj)
            else:
                # Fallback: normal z-order enforcement when neither before nor after
                rm.enforce_z_order(overlay_id)
        except Exception as e:
            if debug_enabled:
                self._logger.debug(f"CTX_MENU: Lifecycle delegation failed: {e}")
            pass

    def _delayed_border_visibility_check(self, overlay_id):
        """Deprecated: explicit delayed scheduling removed per spec."""
        return
    
    # Simplified methods for integrated approach - no separate border overlay management needed

    def _populate_dynamic_menus(self):
        """Populate dynamic submenus (window/monitor lists) just before showing the menu."""
        if self.switch_to_window_menu:
            self.populate_switch_to_window_menu()
        if self.switch_to_monitor_menu:
            self.populate_switch_to_monitor_menu()

    def populate_switch_to_window_menu(self):
        """Populate the Switch To Window submenu with available windows.
        
        Uses the direct overlay method to ensure rich functionality is preserved.
        No fallbacks are provided - we either succeed or fail clearly.
        """
        menu = self.switch_to_window_menu
        menu.clear()
        if debug_enabled:
            self._logger.debug("CTX_MENU: Populating window menu")
        
        # Primary centralized path: use window_menu_utils to populate from app_instance
        try:
            app_instance = getattr(self.overlay, 'app_instance', None)
            if app_instance is None:
                self._logger.error("CTX_MENU: App instance not available for window menu population")
                disabled = menu.addAction("No application context available")
                disabled.setEnabled(False)
                return
            if not hasattr(self.overlay, '_handle_swap_window'):
                self._logger.error("CTX_MENU: Required method '_handle_swap_window' not found on overlay")
                disabled = menu.addAction("Swap handler unavailable")
                disabled.setEnabled(False)
                return
            # Use centralized utility (handles caching, icons, errors)
            # Create wrapper to pass record_mru=True for context menu swaps
            def context_menu_swap_handler(hwnd: int):
                self.overlay._handle_swap_window(hwnd, True)
            populate_window_switch_menu(menu, app_instance, context_menu_swap_handler)
            return
        except Exception as e:
            # Do not crash the whole menu; show an error item and return
            self._logger.error(f"CTX_MENU: Failed to populate window menu via utility: {e}")
            menu.clear()
            err = menu.addAction("Error loading windows")
            err.setEnabled(False)
            return
        
        windows_data = None
        
        # Legacy path (kept for uniqueness but will not be reached due to returns above)
        if hasattr(self.overlay, 'app_instance') and hasattr(self.overlay.app_instance, 'get_menu_ready_windows'):
            if debug_enabled:
                self._logger.debug("CTX_MENU: Using app_instance.get_menu_ready_windows()")
            windows_data = self.overlay.app_instance.get_menu_ready_windows()
            
        # Alternative source: overlay's own method
        elif hasattr(self.overlay, 'get_menu_ready_windows'):
            if debug_enabled:
                self._logger.debug("CTX_MENU: Using overlay.get_menu_ready_windows()")
            windows_data = self.overlay.get_menu_ready_windows()
            
        # Hard failure: required methods not available
        else:
            self._logger.error("CTX_MENU: Required method 'get_menu_ready_windows' not found")
            raise AttributeError("Overlay or its app_instance must provide 'get_menu_ready_windows' method")
        
        # Hard failure: no window data
        if not windows_data:
            self._logger.error("CTX_MENU: No windows data returned from get_menu_ready_windows")
            raise RuntimeError("No window data available for context menu")
            
        if debug_enabled:
            self._logger.debug(f"CTX_MENU: Adding {len(windows_data)} windows to menu")
        
        # Add window entries to menu
        for hwnd, title, icon in windows_data:
            display_title = title.strip() if len(title.strip()) < 60 else title[:57] + "..."
            if not display_title:
                display_title = f"[No Title] ({hwnd})"
                
            action = QAction(display_title, menu)
            if icon and hasattr(icon, 'isNull') and not icon.isNull():
                action.setIcon(icon)
            action.setData(hwnd)
            
            # Wire up the action
            if hasattr(self.overlay, '_handle_swap_window'):
                import functools
                action.triggered.connect(functools.partial(self.overlay._handle_swap_window, hwnd, True))
            else:
                # Hard failure: required handler missing
                self._logger.error("CTX_MENU: Required method '_handle_swap_window' not found on overlay")
                raise AttributeError("Overlay must provide '_handle_swap_window' method")
                
            menu.addAction(action)

    def _get_display_name(self, screen_obj, idx: int) -> str:
        """Return a concise display name for a QScreen for menu labeling.

        Format: "Display {n} — {name} ({w}x{h})"
        Falls back gracefully if attributes are missing.
        """
        try:
            disp_num = int(idx) + 1
        except Exception:
            disp_num = 1

        # Name
        try:
            name = screen_obj.name() if hasattr(screen_obj, 'name') else None
        except Exception:
            name = None
        if not name:
            name = f"Display {disp_num}"

        # Geometry
        try:
            g = screen_obj.geometry() if hasattr(screen_obj, 'geometry') else None
            if g is not None:
                w, h = g.width(), g.height()
                return f"Display {disp_num} — {name} ({w}x{h})"
        except Exception:
            pass

        return f"Display {disp_num} — {name}"

    def populate_switch_to_monitor_menu(self):
        """Populate the Switch To Monitor submenu with available monitors and mode switch option.
        
        No fallbacks are provided - we either succeed or fail clearly.
        """
        menu = self.switch_to_monitor_menu
        menu.clear()
        if debug_enabled:
            self._logger.debug("CTX_MENU: Populating monitor menu")
        
        # Add "Switch to Monitor Overlay" action if needed
        if not hasattr(self.overlay, '_handle_switch_to_monitor_overlay'):
            if debug_enabled:
                self._logger.debug("CTX_MENU: No monitor overlay switch handler available")
        else:
            if debug_enabled:
                self._logger.debug("CTX_MENU: Adding monitor overlay switch option")
            switch_action = QAction("Switch to Monitor Overlay", menu)
            switch_action.triggered.connect(self.overlay._handle_switch_to_monitor_overlay)
            menu.addAction(switch_action)
            menu.addSeparator()
        
        # Get monitors using centralized utility
        from utils.monitor_utils import get_all_monitors
        monitors = get_all_monitors()
        
        if not monitors:
            self._logger.error("CTX_MENU: No monitors returned from get_all_monitors")
            raise RuntimeError("No monitors available for context menu")
            
        # Filter and extract screen objects
        screen_objects = []
        for monitor in monitors:
            screen_obj = monitor.get("screen_object")
            if not screen_obj:
                continue
                
            # Skip current target screen if applicable
            current_screen = None
            if hasattr(self.overlay, "capture_target_screen"):
                current_screen = self.overlay.capture_target_screen
                
            if current_screen and screen_obj == current_screen:
                continue
                
            screen_objects.append(screen_obj)
        
        # Hard failure if no valid screens to switch to
        if not screen_objects:
            self._logger.error("CTX_MENU: No valid screens available for switching")
            raise RuntimeError("No valid screens available for monitor menu")
            
        # Add each screen to the menu
        seen_names = set()
        for idx, screen_obj in enumerate(screen_objects):
            if not hasattr(screen_obj, "name"):
                self._logger.error(f"CTX_MENU: Screen object {idx} has no name attribute")
                raise AttributeError(f"Screen object {idx} missing required 'name' attribute")
                
            name = screen_obj.name()
            if name in seen_names:
                continue
                
            seen_names.add(name)
            display_name = self._get_display_name(screen_obj, len(seen_names) - 1)
            
            # Create and wire the action
            action = QAction(display_name, menu)
            action.setData(screen_obj)
            
            # Wire up the action
            if hasattr(self.overlay, '_handle_switch_monitor'):
                import functools
                action.triggered.connect(functools.partial(self.overlay._handle_switch_monitor, screen_obj))
            else:
                # Hard failure: required handler missing
                self._logger.error("CTX_MENU: Required method '_handle_switch_monitor' not found on overlay")
                raise AttributeError("Overlay must provide '_handle_switch_monitor' method")
                
            menu.addAction(action)

    def _handle_monitor_selection(self, screen_obj):
        """Handle selection of a monitor from the context menu."""
        try:
            from PySide6.QtGui import QGuiApplication
            screens = QGuiApplication.screens()
            if not screen_obj or screen_obj not in screens:
                if hasattr(self.overlay, 'show_error_message'):
                    self.overlay.show_error_message("Selected monitor is no longer available.")
                return
            # Clean up focus indicator before closing overlay
            if hasattr(self.overlay, '_focus_indicator'):
                try:
                    self.overlay._focus_indicator.hide()
                    self.overlay._focus_indicator.deleteLater()
                    self.overlay._focus_indicator = None
                except Exception as e:
                    self._logger.warning(f"Error cleaning up focus indicator: {e}")
            # Store app instance reference before closing overlay
            app_instance = getattr(self.overlay, 'app_instance', None)
            # Close current overlay cleanly
            if hasattr(self.overlay, 'close') and callable(self.overlay.close):
                self.overlay.close()
            # Launch new monitor overlay using the stored app instance
            if app_instance is not None:
                if hasattr(app_instance, 'prepare_to_create_monitor_overlay'):
                    app_instance.prepare_to_create_monitor_overlay(screen_obj)
                elif hasattr(app_instance, 'create_monitor_overlay'):
                    screen_idx = screens.index(screen_obj)
                    app_instance.create_monitor_overlay(screen_idx)
                else:
                    self._logger.error("No valid overlay creation method found in app instance")
            else:
                self._logger.error("App instance not available for monitor overlay creation")
        except Exception as e:
            self._logger.error(f"Error in monitor overlay swap: {e}", exc_info=True)
            if hasattr(self.overlay, 'show_error_message'):
                self.overlay.show_error_message(f"Failed to switch to monitor overlay: {str(e)}")

    def attach_to_overlay(self, overlay=None):
        """Attach this context menu handler using a robust Qt event filter.
        
        Installs event filters on the host (and canvas/border if available) to
        intercept QEvent.ContextMenu and right-click presses, showing the unified
        menu and preserving BorderOverlay visibility.
        """
        # Prefer an explicit target; otherwise use the host for the resolved overlay
        target = overlay or getattr(self.overlay, '_host', None) or self._overlay_widget or self.overlay
        if not target:
            return False

        try:
            # Ensure a menu exists and is themed
            if not self.menu:
                self.build_menu()
                self.apply_theme()

            parent = self

            class _CtxFilter(QObject):
                def eventFilter(self, obj, event):
                    try:
                        if event.type() == QEvent.ContextMenu:
                            if debug_enabled:
                                parent._logger.debug(
                                    f"CTX_MENU: ContextMenu event on {type(obj).__name__} at {event.globalPos().x()}, {event.globalPos().y()}"
                                )
                            parent._ensure_border_visible(getattr(parent.overlay, 'id', None), before=True)
                            parent.show_menu(event.globalPos())
                            event.accept()
                            return True
                        if event.type() == QEvent.MouseButtonPress and hasattr(event, 'button') and event.button() == Qt.RightButton:
                            gp = event.globalPos() if hasattr(event, 'globalPos') else None
                            if debug_enabled:
                                parent._logger.debug(
                                    f"CTX_MENU: Right-click MouseButtonPress on {type(obj).__name__} at {gp.x() if gp else 'NA'}, {gp.y() if gp else 'NA'}"
                                )
                            parent._ensure_border_visible(getattr(parent.overlay, 'id', None), before=True)
                            parent.show_menu(gp)
                            event.accept()
                            return True
                        # Also consume release/double-click for right button to avoid any other handlers reacting
                        if event.type() == QEvent.MouseButtonRelease and hasattr(event, 'button') and event.button() == Qt.RightButton:
                            if debug_enabled:
                                parent._logger.debug(
                                    f"CTX_MENU: Right-click MouseButtonRelease on {type(obj).__name__}"
                                )
                            event.accept()
                            return True
                        if event.type() == QEvent.MouseButtonDblClick and hasattr(event, 'button') and event.button() == Qt.RightButton:
                            if debug_enabled:
                                parent._logger.debug(
                                    f"CTX_MENU: Right-click MouseButtonDblClick on {type(obj).__name__}"
                                )
                            event.accept()
                            return True
                    except AttributeError as e:
                        parent._logger.error(f"CTX_MENU: Event filter attribute error: {e}")
                    except RuntimeError as e:
                        parent._logger.error(f"CTX_MENU: Event filter runtime error: {e}")
                    except Exception as e:
                        parent._logger.error(f"CTX_MENU: Unexpected event filter error: {e}", exc_info=True)
                    return False

            # Install on host
            filt = _CtxFilter(target)
            target.installEventFilter(filt)
            self._filters.append((target, filt))
            if debug_enabled:
                self._logger.debug("CTX_MENU: Installed event filter on host")

            # Also install on canvas if present
            try:
                canvas = getattr(target, 'canvas', None)
                if canvas is not None:
                    filt_c = _CtxFilter(canvas)
                    canvas.installEventFilter(filt_c)
                    self._filters.append((canvas, filt_c))
                    if debug_enabled:
                        self._logger.debug("CTX_MENU: Installed event filter on canvas")
                    # Prevent default context menu on canvas
                    try:
                        canvas.setContextMenuPolicy(Qt.PreventContextMenu)
                    except Exception as e:
                        self._logger.warning(f"CTX_MENU: Failed to set PreventContextMenu on canvas: {e}")
                    # For integrated borders, the border frame is now part of the canvas
                    # but we still support it for backward compatibility
                    try:
                        # First try the standardized property name
                        border = getattr(canvas, 'border_frame', None)
                        if border is None:
                            # Fallback to legacy name
                            border = getattr(canvas, '_border_frame', None)
                        
                        if border is not None:
                            filt_b = _CtxFilter(border)
                            border.installEventFilter(filt_b)
                            self._filters.append((border, filt_b))
                            if debug_enabled:
                                self._logger.debug("CTX_MENU: Installed event filter on border frame")
                            try:
                                border.setContextMenuPolicy(Qt.PreventContextMenu)
                            except Exception as e:
                                self._logger.warning(f"CTX_MENU: Failed to set PreventContextMenu on border frame: {e}")
                                
                        # Also handle backdrop frame if present (integrated approach)
                        backdrop = getattr(canvas, 'backdrop', None) or getattr(canvas, '_backdrop', None)
                        if backdrop is not None:
                            filt_bd = _CtxFilter(backdrop)
                            backdrop.installEventFilter(filt_bd)
                            self._filters.append((backdrop, filt_bd))
                            if debug_enabled:
                                self._logger.debug("CTX_MENU: Installed event filter on backdrop frame")
                            try:
                                backdrop.setContextMenuPolicy(Qt.PreventContextMenu)
                            except Exception as e:
                                self._logger.warning(f"CTX_MENU: Failed to set PreventContextMenu on backdrop: {e}")
                                
                    except Exception as e:
                        self._logger.error(f"CTX_MENU: Error installing event filter on border frame: {e}")
            except Exception as e:
                self._logger.error(f"CTX_MENU: Error installing event filter on canvas: {e}")

            # Prevent default widget context menu handling from leaking through on host
            try:
                target.setContextMenuPolicy(Qt.PreventContextMenu)
            except Exception as e:
                self._logger.warning(f"CTX_MENU: Failed to set PreventContextMenu on host: {e}")

            return True
        except Exception as e:
            self._logger.error(f"Failed to attach context menu handler (filter): {e}")
            return False

    def detach_from_overlay(self, overlay=None):
        """Detach previously installed context menu event filters.
        
        If an overlay (host) is provided, only filters attached to that host (and its
        known children: canvas and border frame) will be removed. If overlay is None,
        all tracked filters will be removed.
        
        Returns:
            bool: True if at least one filter was removed, False otherwise.
        """
        removed_any = False
        try:
            # Determine which targets are allowed when overlay is specified
            allowed_targets = None
            if overlay is not None:
                try:
                    canvas = getattr(overlay, 'canvas', None)
                except Exception:
                    canvas = None
                # Handle both standard and legacy property names
                try:
                    # First try standardized property names
                    border = getattr(canvas, 'border_frame', None) if canvas is not None else None
                    if border is None and canvas is not None:
                        # Fallback to legacy name
                        border = getattr(canvas, '_border_frame', None)
                    
                    # Also check for backdrop frame (integrated approach)
                    backdrop = None
                    if canvas is not None:
                        backdrop = getattr(canvas, 'backdrop', None) or getattr(canvas, '_backdrop', None)
                except Exception as e:
                    self._logger.debug(f"CTX_MENU: Error accessing frame properties: {e}")
                    border = None
                    backdrop = None
                allowed_targets = {overlay}
                if canvas is not None:
                    allowed_targets.add(canvas)
                if border is not None:
                    allowed_targets.add(border)
                if backdrop is not None:
                    allowed_targets.add(backdrop)

            # Iterate over a copy since we'll modify the list
            for (target, filt) in list(self._filters):
                try:
                    if allowed_targets is not None and target not in allowed_targets:
                        continue
                    # Remove event filter
                    try:
                        target.removeEventFilter(filt)
                    except Exception as e:
                        self._logger.warning(f"CTX_MENU: Failed to remove event filter from {type(target).__name__}: {e}")
                    # Best-effort restore default context menu policy on host-like widgets
                    try:
                        target.setContextMenuPolicy(Qt.DefaultContextMenu)
                    except Exception as e:
                        self._logger.warning(f"CTX_MENU: Failed to restore DefaultContextMenu on {type(target).__name__}: {e}")
                    # Remove from tracking
                    self._filters.remove((target, filt))
                    removed_any = True
                except Exception as e:
                    self._logger.error(f"CTX_MENU: Failed to detach filter from {type(target).__name__}: {e}")
            return removed_any
        except Exception as e:
            self._logger.error(f"CTX_MENU: Error during detach_from_overlay: {e}")
            return False

 

 

# Remove erroneously inserted free-function duplicates; class methods already exist above.