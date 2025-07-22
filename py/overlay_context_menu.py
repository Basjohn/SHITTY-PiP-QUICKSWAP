"""
overlay_context_menu.py

Unified context menu builder for both window and monitor overlays in SPQ.
Handles menu creation, population, theming, and action callbacks for overlays.

Usage:
    menu_builder = OverlayContextMenu(overlay_widget, overlay_type='window' or 'monitor')
    menu = menu_builder.build_menu()
    menu_builder.apply_theme()
    menu.exec(pos)

See Cleanup2.md for migration rationale and integration details.
"""
from PySide6.QtWidgets import QMenu
from PySide6.QtGui import QAction

class OverlayContextMenu:
    def __init__(self, overlay_widget, overlay_type='window', config=None):
        self.overlay = overlay_widget
        self.overlay_type = overlay_type  # 'window' or 'monitor'
        self.config = config or {}
        self.menu = None
        self.switch_to_window_menu = None
        self.switch_to_monitor_menu = None
        self.lock_action = None
        self._actions = {}

    def build_menu(self):
        """Constructs and returns a QMenu for the overlay."""
        self.menu = QMenu(self.overlay)
        # --- Switch to Window (if supported) ---
        if self.overlay_type == 'window' or self.config.get('show_switch_to_window', True):
            self.switch_to_window_menu = QMenu("Switch To Window", self.menu)
            switch_to_window_action = QAction("Switch To Window", self.overlay)
            switch_to_window_action.setMenu(self.switch_to_window_menu)
            self.menu.addAction(switch_to_window_action)
            self._actions['switch_to_window'] = switch_to_window_action
        # --- Switch to Monitor (if supported) ---
        if self.overlay_type in ('window', 'monitor') and self.config.get('show_switch_to_monitor', True):
            self.switch_to_monitor_menu = QMenu("Switch To Monitor", self.menu)
            switch_to_monitor_action = QAction("Switch To Monitor", self.overlay)
            switch_to_monitor_action.setMenu(self.switch_to_monitor_menu)
            self.menu.addAction(switch_to_monitor_action)
            self._actions['switch_to_monitor'] = switch_to_monitor_action
        self.menu.addSeparator()
        # --- Lock Overlay (window overlays only) ---
        if self.overlay_type == 'window':
            self.lock_action = QAction("Lock Overlay", self.overlay)
            self.lock_action.setCheckable(True)
            self.lock_action.setChecked(getattr(self.overlay, '_is_window_locked', False))
            self.lock_action.triggered.connect(self.overlay.toggle_window_lock)
            self.menu.addAction(self.lock_action)
            self._actions['lock'] = self.lock_action
            self.menu.addSeparator()
        # --- Settings, Subsettings ---
        show_settings_action = QAction("Main Window", self.overlay)
        show_settings_action.triggered.connect(self.overlay._handle_show_settings)
        self.menu.addAction(show_settings_action)
        self._actions['show_settings'] = show_settings_action
        show_sub_settings_action = QAction("Subsettings", self.overlay)
        show_sub_settings_action.triggered.connect(self.overlay._handle_show_sub_settings)
        self.menu.addAction(show_sub_settings_action)
        self._actions['show_sub_settings'] = show_sub_settings_action
        self.menu.addSeparator()
        # --- Hide, Reset, Quit ---
        hide_action = QAction("Hide", self.overlay)
        hide_action.triggered.connect(self.overlay.close)
        self.menu.addAction(hide_action)
        self._actions['hide'] = hide_action
        reset_action = QAction("Reset", self.overlay)
        reset_action.triggered.connect(self.overlay._handle_reset_position)
        self.menu.addAction(reset_action)
        self._actions['reset'] = reset_action
        self.menu.addSeparator()
        quit_app_action = QAction("Quit Application", self.overlay)
        quit_app_action.triggered.connect(self.overlay._handle_quit_application)
        self.menu.addAction(quit_app_action)
        self._actions['quit'] = quit_app_action
        return self.menu

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
        # Increase submenu width to 124px to maintain proportion
        submenu_stylesheet = stylesheet.replace('min-width: 96px', 'min-width: 124px')
        submenu_stylesheet = submenu_stylesheet.replace('min-width: 98px', 'min-width: 124px')
        submenu_stylesheet = submenu_stylesheet.replace('min-width: 100px', 'min-width: 124px')
        
        if hasattr(self, 'switch_to_window_menu') and self.switch_to_window_menu:
            self.switch_to_window_menu.setStyleSheet(submenu_stylesheet)
        if hasattr(self, 'switch_to_monitor_menu') and self.switch_to_monitor_menu:
            self.switch_to_monitor_menu.setStyleSheet(submenu_stylesheet)
            
        # Don't override with theme colors
        if self.lock_action:
            self.lock_action.setChecked(getattr(self.overlay, '_is_window_locked', False))

    def show_menu(self, pos):
        """Show the context menu at the given position (widget coordinates).
        Dynamically (re)populate submenus before display."""
        if not self.menu:
            self.build_menu()
        self._populate_dynamic_menus()
        self.apply_theme()
        self.menu.exec(self.overlay.mapToGlobal(pos))

    def _populate_dynamic_menus(self):
        """Populate dynamic submenus (window/monitor lists) just before showing the menu."""
        if self.switch_to_window_menu:
            self.populate_switch_to_window_menu()
        if self.switch_to_monitor_menu:
            self.populate_switch_to_monitor_menu()

    def populate_switch_to_window_menu(self):
        """Populate the Switch To Window submenu with available windows."""
        menu = self.switch_to_window_menu
        menu.clear()
        windows_data = []
        # Try overlay-provided method first
        if hasattr(self.overlay, 'app_instance') and hasattr(self.overlay.app_instance, 'get_menu_ready_windows'):
            windows_data = self.overlay.app_instance.get_menu_ready_windows()
        elif hasattr(self.overlay, 'get_menu_ready_windows'):
            windows_data = self.overlay.get_menu_ready_windows()
        # Fallback: empty
        if not windows_data:
            action = QAction("No other windows found", menu)
            action.setEnabled(False)
            menu.addAction(action)
            return
        for hwnd, title, icon in windows_data:
            display_title = title.strip() if len(title.strip()) < 60 else title[:57] + "..."
            if not display_title:
                display_title = f"[No Title] ({hwnd})"
            action = QAction(display_title, menu)
            if icon and hasattr(icon, 'isNull') and not icon.isNull():
                action.setIcon(icon)
            action.setData(hwnd)
            # Use overlay's handler
            if hasattr(self.overlay, '_handle_swap_window'):
                import functools
                action.triggered.connect(functools.partial(self.overlay._handle_swap_window, hwnd))
            menu.addAction(action)

    def populate_switch_to_monitor_menu(self):
        """Populate the Switch To Monitor submenu with available monitors and mode switch option."""
        menu = self.switch_to_monitor_menu
        menu.clear()
        # Add "Switch to Monitor Overlay" action if supported
        if hasattr(self.overlay, '_handle_switch_to_monitor_overlay'):
            switch_action = QAction("Switch to Monitor Overlay", menu)
            switch_action.triggered.connect(self.overlay._handle_switch_to_monitor_overlay)
            menu.addAction(switch_action)
            menu.addSeparator()
        # Get monitors with detailed info
        try:
            from monitor_utils import get_all_monitors
            monitors = get_all_monitors()
            screen_objects = []
            for monitor in monitors:
                screen_obj = monitor.get("screen_object")
                if not screen_obj:
                    continue
                # Skip current target screen if overlay provides it
                if hasattr(self.overlay, "capture_target_screen") and screen_obj == getattr(self.overlay, "capture_target_screen", None):
                    continue
                screen_objects.append(screen_obj)
        except Exception as e:
            from PySide6.QtGui import QGuiApplication
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting monitor list: {e}")
            screen_objects = QGuiApplication.screens()
        if not screen_objects:
            action = QAction("No screens available", menu)
            action.setEnabled(False)
            menu.addAction(action)
            return
        seen_names = set()
        for idx, screen_obj in enumerate(screen_objects):
            name = screen_obj.name() if hasattr(screen_obj, "name") else str(idx)
            if name in seen_names:
                continue
            seen_names.add(name)
            display_name = self._get_display_name(screen_obj, len(seen_names) - 1)
            action = QAction(display_name, menu)
            action.triggered.connect(lambda checked=False, s=screen_obj: self._handle_monitor_selection(s))
            menu.addAction(action)

    def _handle_monitor_selection(self, screen_obj):
        """Handle monitor selection: close overlay and create monitor overlay on target screen."""
        import logging
        from PySide6.QtGui import QGuiApplication
        logger = logging.getLogger(__name__)
        try:
            # Validate screen is still available
            screens = QGuiApplication.screens()
            if screen_obj not in screens:
                logger.warning(f"Selected screen {getattr(screen_obj, 'name', lambda: 'unknown')()} no longer available.")
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
                    logger.warning(f"Error cleaning up focus indicator: {e}")
                    
            # Store app instance reference before closing overlay
            app_instance = getattr(self.overlay, 'app_instance', None)
            
            # Close current overlay cleanly
            if hasattr(self.overlay, 'close') and callable(self.overlay.close):
                self.overlay.close()
                
            # Launch new monitor overlay using the stored app instance
            if app_instance is not None:
                if hasattr(app_instance, 'prepare_to_create_monitor_overlay'):
                    app_instance.prepare_to_create_monitor_overlay(screen_obj)
                elif hasattr(app_instance, 'create_monitor_overlay'):  # Fallback to legacy method
                    screen_idx = screens.index(screen_obj) if screen_obj in screens else 0
                    app_instance.create_monitor_overlay(screen_idx)
                else:
                    logger.error("No valid overlay creation method found in app instance")
            else:
                logger.error("App instance not available for monitor overlay creation")
                
        except Exception as e:
            logger.error(f"Error in monitor overlay swap: {e}", exc_info=True)
            if hasattr(self.overlay, 'show_error_message'):
                self.overlay.show_error_message(f"Failed to switch to monitor overlay: {str(e)}")

    def _get_display_name(self, screen, idx):
        """Generate a detailed display name for the screen."""
        try:
            name = screen.name() if hasattr(screen, 'name') else f"Screen {idx+1}"
            geo = screen.geometry() if hasattr(screen, 'geometry') else None
            extra_info = []
            if hasattr(screen, 'manufacturer'):
                manufacturer = screen.manufacturer()
                if manufacturer and manufacturer != "unknown":
                    extra_info.append(manufacturer.strip())
            if hasattr(screen, 'model'):
                model = screen.model()
                if model and model != "unknown":
                    extra_info.append(model.strip())
            if geo and geo.isValid():
                extra_info.append(f"{geo.width()}x{geo.height()}")
            from PySide6.QtGui import QGuiApplication
            if screen == QGuiApplication.primaryScreen():
                extra_info.append("(Primary)")
            if extra_info:
                return f"{name} - {' '.join(extra_info)}"
            return name
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting display name: {e}")
            return f"Screen {idx+1}"
