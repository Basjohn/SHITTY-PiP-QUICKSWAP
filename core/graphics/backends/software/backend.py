from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt

from ...overlay import Overlay as OverlayBase
from ...types import OverlayConfig
from ...overlay_host import OverlayHost


class SoftwareOverlay(OverlayBase):
    """Software backend overlay hosting IntegratedBorderCanvas in a simple top-level QWidget.
    This is the initial Phase 1 visual host; rendering backends (DWM/OpenGL) will attach
    to IntegratedBorderCanvas.contentRectChanged for content drawing in subsequent phases."""

    def __init__(self, config: OverlayConfig):
        super().__init__(config)
        self._host: Optional[OverlayHost] = None
        self._context_menu_handler = None
        # No separate border overlay with integrated approach

    # Implementation hooks
    def _initialize_impl(self) -> None:
        self._host = OverlayHost(self._config)
        # Store a reference to this overlay instance in the host for proper cleanup
        self._host._parent_overlay = self
        self._host.set_host_geometry(self._config.position, self._config.size)
        
        # Set up centralized context menu
        self._add_context_menu_to_host()
        
        # Apply initial canvas/backdrop opacity from config
        try:
            self._host.set_canvas_opacity(self._config.opacity)
        except Exception as e:
            from core.logging import get_logger
            get_logger("SoftwareOverlay").warning(f"Failed to set initial canvas opacity: {e}")

        # Aspect ratio binding if provided
        ar = self._config.properties.get("content_aspect")
        if ar and isinstance(ar, (tuple, list)) and len(ar) == 2:
            self._host.canvas.set_content_aspect(int(ar[0]), int(ar[1]))

    def _show_impl(self) -> None:
        if self._host:
            self._host.show()

    def _hide_impl(self) -> None:
        if self._host:
            self._host.hide()

    def _close_impl(self) -> None:
        # Clean up context menu handler first
        if hasattr(self, '_context_menu_handler') and self._context_menu_handler:
            try:
                if hasattr(self._context_menu_handler, 'detach_from_overlay') and self._host:
                    result = self._context_menu_handler.detach_from_overlay(self._host)
                    from core.logging import get_logger
                    get_logger("SoftwareOverlay").debug(f"CONTEXT_MENU: Detached with result: {result}")
                self._context_menu_handler = None
            except Exception as e:
                from core.logging import get_logger
                get_logger("SoftwareOverlay").error(f"CONTEXT_MENU: Error detaching: {e}")
        
        # No border overlay cleanup needed with integrated approach
        
        if self._host:
            self._host.close()
            self._host.deleteLater()
            self._host = None

    def _render_impl(self) -> None:
        # No continuous rendering required for the software host.
        pass

    def _add_context_menu_to_host(self):
        """Add context menu handler to host window using the centralized OverlayContextMenu.
        
        This uses the unified OverlayContextMenu class to ensure consistent menu behavior
        and proper border preservation across all overlay types.
        """
        from core.logging import get_logger
        logger = get_logger("SoftwareOverlay")
        
        if not self._host:
            logger.debug("CONTEXT_MENU: Cannot add context menu - host window does not exist")
            return
            
        try:
            logger.debug("CONTEXT_MENU: Attaching centralized handler via event filter (PreventContextMenu)")

            # Import centralized menu builder
            from utils.overlay_context_menu import OverlayContextMenu
            from PySide6.QtCore import Qt

            # Create and theme handler (use overlay_widget=self so action methods resolve)
            self._context_menu_handler = OverlayContextMenu(
                overlay_widget=self,
                overlay_type='software',
                config={
                    # Software overlay does not support window/monitor swap menus
                    'show_switch_to_window': False,
                    'show_switch_to_monitor': False,
                }
                # No border_overlay parameter with integrated approach
            )
            self._context_menu_handler.build_menu()
            self._context_menu_handler.apply_theme()

            # Attach robust Qt event filters that intercept QEvent.ContextMenu and right-clicks.
            # The handler will set Qt.PreventContextMenu on host/canvas/border to avoid leaks.
            try:
                attached = self._context_menu_handler.attach_to_overlay(self._host)
                logger.debug(f"CONTEXT_MENU: Event filter attach result: {attached}")
            except Exception as e:
                logger.error(f"CONTEXT_MENU: Failed to attach handler via event filter: {e}")

        except Exception as e:
            logger.error(f"CONTEXT_MENU: Error setting up explicit menu: {e}")
            # No fallbacks
    
    
    def _on_host_context_menu(self, pos):
        """Handle host CustomContextMenu by showing the centralized menu.
        Ensures border overlay remains visible and correctly stacked.
        """
        from core.logging import get_logger
        logger = get_logger("SoftwareOverlay")
        
        try:
            if not self._host or not self._context_menu_handler:
                logger.error("CONTEXT_MENU: No host or handler; cannot open menu")
                return

            # Map local pos to global
            try:
                global_pos = self._host.mapToGlobal(pos)
            except Exception as e:
                logger.error(f"CONTEXT_MENU: mapToGlobal failed: {e}")
                return

            logger.debug(
                "CONTEXT_MENU: customContextMenuRequested at %s, %s",
                getattr(global_pos, 'x', lambda: 'NA')(), getattr(global_pos, 'y', lambda: 'NA')()
            )

            # Ensure integrated overlay is visible during context menu interaction
            # The centralized context menu handler now manages this internally
            try:
                # Get overlay ID for z-order enforcement
                overlay_id = getattr(self, 'id', None)
                if overlay_id:
                    # Route via ResourceManager centralized delegation
                    from utils.resource_manager import get_resource_manager
                    rm = get_resource_manager()
                    rm.enforce_z_order(overlay_id)
            except Exception as e:
                logger.warning(f"CONTEXT_MENU: Failed z-order enforcement before menu: {e}")

            # Show the context menu which handles border visibility internally
            self._context_menu_handler.show_menu(global_pos)
            
            # No need for post-menu handling as context menu now manages this
        except Exception as e:
            logger.error(f"CONTEXT_MENU: Error in host menu handler: {e}")
            
    def _config_updated(self, old_config, new_config) -> None:
        if not self._host:
            return
        # Geometry
        if old_config.get("position") != new_config.get("position") or \
           old_config.get("size") != new_config.get("size"):
            self._host.set_host_geometry(self._config.position, self._config.size)
        # Title
        if old_config.get("title") != new_config.get("title"):
            self._host.setWindowTitle(self._config.title)
        # Opacity: apply to both window and canvas backdrop for complete transparency
        if old_config.get("opacity") != new_config.get("opacity"):
            try:
                # Set window-level opacity for the entire overlay (including pillarbox/letterbox bars)
                self._host.set_window_opacity(self._config.opacity)
                # Also update canvas backdrop opacity for consistent visuals
                self._host.set_canvas_opacity(self._config.opacity)
            except Exception as e:
                from core.logging import get_logger
                get_logger("SoftwareOverlay").warning(f"Failed to update opacity on host/canvas: {e}")
        # Click-through
        if old_config.get("click_through") != new_config.get("click_through"):
            self._host.setAttribute(Qt.WA_TransparentForMouseEvents, bool(self._config.click_through))
