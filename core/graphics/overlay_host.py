from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal, QEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout

from core.logging import get_logger
from ui.overlays.integrated_border_canvas import IntegratedBorderCanvas
from ui.overlays.geometry.focus_indicator import FocusIndicatorWindow
from utils.window.behavior import WindowBehaviorManager
from utils.theme.theme_manager import ThemeManager
from utils.cursor_manager import get_cursor_manager, CursorPriority
from core.input.key_passthrough_controller import get_key_passthrough_controller
from core.graphics import get_overlay_manager
from .types import OverlayConfig
from ui.components.volume_osd import VolumeOSDWidget, VolumeOSDWindow


class OverlayHost(QWidget):
    """Unified top-level host window for overlays (Software/DWM/GL).

    - Frameless, always-on-top tool window without taskbar entry
    - Hosts the IntegratedBorderCanvas and centralizes drag/resize/snap via WindowBehaviorManager
    - Centralizes right-click context menu forwarding to the overlay's injected handler
    - Emits geometryChanged signal on move/resize for integrated rendering coordination
    """

    geometryChanged = Signal()

    def __init__(self, config: OverlayConfig):
        super().__init__(None)
        self.setObjectName("overlayHostWindow")

        # Top-level tool window, always on top; Qt.Tool avoids a taskbar entry on Windows
        flags = Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

        # Do not take focus or activate the application when shown
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        # Avoid layered (translucent) parent windows: layered parents can prevent native child
        # frames (our border) from painting on Windows. Keep host non-layered but styled.
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        # Get centralized theme manager
        self._theme_manager = ThemeManager.instance()

        # Removed click-through functionality

        self.setWindowTitle(getattr(config, "title", "Overlay"))

        # Layout with overlay canvas
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = IntegratedBorderCanvas(self)
        # Ensure QSS background/border are painted on the canvas
        self.canvas.setAttribute(Qt.WA_StyledBackground, True)
        # Canvas styling setup
        # Intercept context menu events from the canvas so the host owns the menu
        try:
            self.canvas.installEventFilter(self)
        except Exception:
            pass
        layout.addWidget(self.canvas)
        
        # Apply theme styling via centralized manager
        self._apply_theme()

        # Enforce minimum size via canvas
        self.canvas.setMinimumSize(IntegratedBorderCanvas.MIN_WIDTH, IntegratedBorderCanvas.MIN_HEIGHT)

        # Volume OSD as a top-level tool window placed above the host. Hidden by default; shows itself on events.
        self._volume_osd: Optional[VolumeOSDWidget] = None
        try:
            # Use top-level OSD window to avoid DWM/native child compositing and z-order issues
            self._volume_osd = VolumeOSDWindow(self)
            # Ensure initial position and theming
            try:
                self._volume_osd.update_position()
            except Exception:
                pass
            try:
                self._theme_manager.apply_theme_to_widget(self._volume_osd)
            except Exception:
                pass
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.debug(f"Failed to create VolumeOSDWidget: {e}")

        # Centralized window behavior for drag/resize/snap on the host window
        # IntegratedBorderCanvas forwards mouse events to this manager to keep border and background coupled
        self._behavior = WindowBehaviorManager(
            self,
            min_width=IntegratedBorderCanvas.MIN_WIDTH,
            min_height=IntegratedBorderCanvas.MIN_HEIGHT,
        )

        # Parent overlay will be injected by the backend after creation
        self._parent_overlay: Optional[object] = None
        
        # Connect to theme change signal
        self._theme_manager.theme_changed.connect(self._on_theme_changed)

        # Focus indicator widget (bottom-right), non-intrusive overlay status UI
        try:
            # Use top-level indicator window prototype to avoid DWM child compositing issues
            self._focus_indicator = FocusIndicatorWindow(self)
            # Focus indicator setup
            logger = get_logger("OverlayHost")
            logger.debug(f"Created focus indicator: {self._focus_indicator}")
            # Initial attach and position
            try:
                self._focus_indicator.update_position()
            except Exception:
                pass
            # Initialize locked state from parent overlay if available
            try:
                ov0 = getattr(self, "_parent_overlay", None)
                if ov0 is not None and hasattr(ov0, "_is_window_locked"):
                    self._focus_indicator.set_locked(bool(getattr(ov0, "_is_window_locked", False)))
            except Exception:
                pass
            # Wire passthrough state changes
            try:
                kp = get_key_passthrough_controller()
                kp.enabled_changed.connect(self._focus_indicator.set_passthrough_enabled)
                # Set initial state
                self._focus_indicator.set_passthrough_enabled(kp.is_enabled())
            except Exception:
                pass
            # Forward click to parent overlay's lock toggle if available
            def _on_lock_toggle():
                try:
                    ov = getattr(self, "_parent_overlay", None)
                    if ov and hasattr(ov, "toggle_window_lock"):
                        ov.toggle_window_lock()
                        # Reflect state if readable
                        if hasattr(ov, "_is_window_locked"):
                            self._focus_indicator.set_locked(bool(getattr(ov, "_is_window_locked", False)))
                except Exception as e:
                    logger = get_logger("OverlayHost")
                    logger.debug(f"Lock toggle failed: {e}")
            self._focus_indicator.lock_toggled.connect(_on_lock_toggle)
            
            # Force initial focus check after a short delay to ensure proper initialization
            from core.threading import ThreadManager
            ThreadManager.single_shot(100, lambda: self._check_initial_focus())
            
            # Also check passthrough state after initialization to ensure correct color
            from core.threading import ThreadManager
            ThreadManager.single_shot(150, lambda: self._sync_passthrough_state())
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.error(f"Failed to create focus indicator: {e}")
            self._focus_indicator = None

        # Initialize geometry coalescer state even if focus indicator fails
        try:
            self._geom_coalescer = None
        except Exception:
            pass

        # Lazily create a UI coalescer for geometry verification (7ms window, cap 128)
        try:
            self._geom_coalescer = None
            # Get ThreadManager directly from singleton to access factory methods
            from core.threading import get_thread_manager
            tm = get_thread_manager()
            logger = get_logger("OverlayHost")
            # Log ThreadManager instance state to diagnose missing attribute issues
            try:
                tm_type = type(tm).__name__
                tm_id = id(tm)
                has_coalescer = hasattr(tm, 'create_ui_coalescer')
                shutdown_flag = getattr(tm, '_shutdown', None)
                logger.debug(
                    f"ThreadManager for geom coalescer: type={tm_type}, id={tm_id}, "
                    f"has_create_ui_coalescer={has_coalescer}, shutdown={shutdown_flag}"
                )
            except Exception as _log_e:
                logger.debug(f"Failed to introspect ThreadManager (geom coalescer): {_log_e}")

            if hasattr(tm, 'create_ui_coalescer'):
                try:
                    self._geom_coalescer = tm.create_ui_coalescer(
                        name=f"overlay_geom_{id(self)}",
                        capacity=128,
                        window_ms=7,
                    )
                    logger.info("OverlayHost geometry UI coalescer initialized (window=7ms, cap=128)")
                except Exception as e:
                    logger.debug(f"Failed creating geometry coalescer: {e}")
                    self._geom_coalescer = None
            else:
                # Explicit diagnostic when attribute is missing
                try:
                    attrs = [a for a in dir(tm) if not a.startswith('_')]
                    logger.error(
                        f"ThreadManager missing create_ui_coalescer; available attrs (partial): {attrs[:50]}"
                    )
                except Exception:
                    pass
            self._geom_coalescer = None
        except Exception:
            # Leave coalescer as None; we'll fallback to UI thread scheduling
            self._geom_coalescer = None

        # Connect geometryChanged to coalesced verifier
        try:
            self.geometryChanged.connect(self._on_geometry_changed)
        except Exception:
            pass

        # Keep OSD positioned on geometry changes
        try:
            self.geometryChanged.connect(self._safe_update_volume_osd_position)
        except Exception:
            pass

    def set_canvas(self, widget: QWidget) -> None:
        """Replace the default IntegratedBorderCanvas with a provided widget.

        - Preserves layout, styling, and behavior manager assumptions.
        - Ensures the new canvas has styled background enabled for QSS.
        """
        try:
            if widget is None or widget is self.canvas:
                return

            # Remove old canvas from layout
            layout = self.layout()
            if layout is not None and self.canvas is not None:
                try:
                    layout.removeWidget(self.canvas)
                except Exception:
                    pass
                self.canvas.setParent(None)

            # Install new canvas
            self.canvas = widget
            try:
                self.canvas.setParent(self)
            except Exception:
                pass

            # Ensure styled background so QSS applies
            try:
                self.canvas.setAttribute(Qt.WA_StyledBackground, True)
            except Exception:
                pass

            # Add to layout
            if layout is not None:
                layout.addWidget(self.canvas)

            # Re-apply theme to host and new canvas
            self._apply_theme()
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.debug(f"set_canvas failed: {e}")
    
    def _apply_theme(self) -> None:
        """Apply current theme styling to the overlay host and canvas.
        Uses the centralized ThemeManager to ensure consistent styling.
        """
        try:
            # Apply theme to this widget and canvas via centralized manager
            self._theme_manager.apply_theme_to_widget(self)
            self._theme_manager.apply_theme_to_widget(self.canvas)
            # Apply theme to OSD if present
            if getattr(self, "_volume_osd", None) is not None:
                self._theme_manager.apply_theme_to_widget(self._volume_osd)
        except Exception as e:
            logger = get_logger(__name__)
            logger.debug(f"Theme application error: {e}")

    def _safe_update_volume_osd_position(self) -> None:
        """Safely update Volume OSD position within the host."""
        try:
            if getattr(self, "_volume_osd", None) is not None:
                self._volume_osd.update_position()
        except Exception:
            pass
    
    def _on_theme_changed(self, theme_name: str) -> None:
        """Handle theme changes from the centralized ThemeManager."""
        self._apply_theme()

    def set_canvas_opacity(self, opacity: float) -> None:
        """Route opacity to the backdrop only so the border stays fully opaque."""
        try:
            o = max(0.0, min(1.0, float(opacity)))
            if hasattr(self.canvas, "set_backdrop_opacity"):
                self.canvas.set_backdrop_opacity(o)
        except Exception:
            pass

    # Forward context menu events to the centralized context menu system
    def contextMenuEvent(self, event):
        logger = get_logger("OverlayHost")
        parent_overlay = getattr(self, "_parent_overlay", None)

        # Check if centralized context menu handler is available
        if parent_overlay and hasattr(parent_overlay, "_context_menu_handler") and parent_overlay._context_menu_handler:
            try:
                parent_overlay._context_menu_handler.show_menu(event.globalPos())
                event.accept()
                return
            except Exception as e:
                logger.error(f"Context menu error: {e}")
                event.accept()
                return

        # If we reach here, something is wrong with the context menu system
        logger.error("No context menu handler available")
        event.accept()

    def eventFilter(self, obj, event):
        """Intercept canvas events and route to appropriate handlers."""
        try:
            if obj is self.canvas and event:
                # Handle context menu events
                if event.type() == QEvent.ContextMenu:
                    self.contextMenuEvent(event)
                    return True
                # Handle key events for passthrough
                elif event.type() in (QEvent.KeyPress, QEvent.KeyRelease):
                    if self._handle_key_event(event):
                        return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Handle key press events for passthrough."""
        try:
            if self._handle_key_event(event):
                event.accept()
                return
        except Exception:
            pass
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key release events for passthrough."""
        try:
            if self._handle_key_event(event):
                event.accept()
                return
        except Exception:
            pass
        super().keyReleaseEvent(event)

    def _handle_key_event(self, event) -> bool:
        """Handle key events for passthrough when overlay is active."""
        try:
            # Only handle key events when overlay is active and has focus
            if not self.isActiveWindow():
                return False
                
            from PySide6.QtCore import QEvent
            from PySide6.QtGui import QKeyEvent
            
            if not isinstance(event, QKeyEvent):
                return False
            
            # Get the virtual key code
            vk = event.nativeVirtualKey()
            if not vk:
                return False

            logger = get_logger("OverlayHost")
            logger.debug(f"Key event: type={event.type()} auto_repeat={getattr(event, 'isAutoRepeat', lambda: False)()} VK={vk:02X} key={event.key()} text='{event.text()}'")

            # Intercept ESC on key press to close the active overlay instead of passing through
            try:
                if event.type() == QEvent.KeyPress and int(vk) == 0x1B:  # VK_ESCAPE
                    logger.info("ESC detected on overlay host; closing active overlay")
                    try:
                        get_overlay_manager().close_active()
                    except Exception as ce:
                        logger.error(f"Failed closing active overlay on ESC: {ce}")
                    return True
            except Exception:
                pass

            # Forward to KeyPassthroughController with hold/release wiring for Up/Down
            try:
                kp = get_key_passthrough_controller()
                # Force-sync target to current overlay source hwnd on every key event to avoid stale routing
                try:
                    ov = getattr(self, "_parent_overlay", None)
                    if ov is not None and hasattr(ov, "_source_hwnd"):
                        try:
                            src_hwnd = int(getattr(ov, "_source_hwnd") or 0)
                        except Exception:
                            src_hwnd = 0
                        kp.set_target_hwnd(src_hwnd if src_hwnd else None)
                except Exception:
                    pass

                VK_UP = 0x26
                VK_DOWN = 0x28
                VK_VOLUME_UP = 0xAF
                VK_VOLUME_DOWN = 0xAE
                is_auto = False
                try:
                    is_auto = bool(event.isAutoRepeat())
                except Exception:
                    is_auto = False

                # Hardware volume keys: use press/release semantics to enable hold timers
                if int(vk) in (VK_VOLUME_UP, VK_VOLUME_DOWN):
                    if event.type() == QEvent.KeyPress:
                        # Ignore auto-repeats; controller manages its own repeat loop
                        if is_auto:
                            return True
                        handled = kp.press_passthrough_key(int(vk))
                        logger.debug(f"Press volume passthrough {vk:02X} -> {handled}")
                        return handled
                    elif event.type() == QEvent.KeyRelease:
                        # Always release; safe if already released
                        kp.release_passthrough_key(int(vk))
                        logger.debug(f"Release volume passthrough {vk:02X}")
                        return True
                    else:
                        return False

                # Arrow keys: retain existing behavior (no volume mapping here)
                if int(vk) in (VK_UP, VK_DOWN):
                    if event.type() == QEvent.KeyPress:
                        # Ignore auto-repeats; controller manages its own repeat loop
                        if is_auto:
                            return True
                        handled = kp.press_passthrough_key(int(vk))
                        logger.debug(f"Press passthrough {vk:02X} -> {handled}")
                        return handled
                    elif event.type() == QEvent.KeyRelease:
                        # Always release; safe if already released
                        kp.release_passthrough_key(int(vk))
                        logger.debug(f"Release passthrough {vk:02X}")
                        return True
                    else:
                        return False
                else:
                    # Other keys: single-tap passthrough on KeyPress only
                    if event.type() != QEvent.KeyPress:
                        return False
                    result = kp.passthrough_key(int(vk))
                    logger.debug(f"Passthrough result: VK={vk:02X} -> {result}")
                    if result:
                        return True
            except Exception as e:
                logger.debug(f"Key passthrough failed: {e}")
                
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.error(f"Key event handling failed: {e}")
            
        return False

    # Forward our own mouse events to the behavior manager as well. This keeps dragging active
    # after grabMouse(), when subsequent events are delivered to the host instead of the canvas.
    def mousePressEvent(self, event):
        # Check if click is on focus indicator first
        if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
            # Forward only if indicator is a child widget; for a separate window, it handles clicks itself
            try:
                if self._focus_indicator.parentWidget() is self and self._focus_indicator.isVisible():
                    indicator_rect = QRect(self._focus_indicator.pos(), self._focus_indicator.size())
                    if indicator_rect.contains(event.pos()):
                        # Let the focus indicator handle the click
                        self._focus_indicator.mousePressEvent(event)
                        return
            except Exception:
                pass
        
        if hasattr(self, "_behavior"):
            try:
                # Entire window is draggable except resize margins handled by manager
                self._behavior.handle_mouse_press(event, is_draggable_region=lambda p: True)
                event.accept()
                return
            except Exception:
                pass
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self, "_behavior"):
            try:
                self._behavior.handle_mouse_move(event)
                event.accept()
                return
            except Exception:
                pass
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self, "_behavior"):
            try:
                self._behavior.handle_mouse_release(event)
                event.accept()
                return
            except Exception:
                pass
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        logger = get_logger("OverlayHost")
        
        # Ensure we're only processing left-button double-clicks for quickswitch
        # Right button is handled by context menu system
        try:
            is_left_button = hasattr(event, 'button') and event.button() == Qt.LeftButton
            if not is_left_button:
                super().mouseDoubleClickEvent(event)
                return
            
            # Check if behavior manager can handle double-click first
            if hasattr(self, '_behavior') and self._behavior and hasattr(self._behavior, 'handle_double_click'):
                if self._behavior.handle_double_click(event):
                    event.accept()
                    return
            
            # Check if overlay is locked - check both global and individual lock states
            try:
                from core.graphics.overlay_manager import OverlayManager
                om = OverlayManager()
                is_globally_locked = om and om.is_overlay_locked()
                is_individually_locked = hasattr(self, '_parent_overlay') and self._parent_overlay and getattr(self._parent_overlay, '_is_window_locked', False)
                
                if is_globally_locked or is_individually_locked:
                    logger.info(f"Overlay locked (global={is_globally_locked}, individual={is_individually_locked}) - focusing current window instead of switching")
                    # Focus the currently captured window without switching
                    if hasattr(self, '_parent_overlay') and self._parent_overlay:
                        current_hwnd = getattr(self._parent_overlay, '_captured_hwnd', None)
                        if current_hwnd:
                            try:
                                import win32gui
                                win32gui.SetForegroundWindow(current_hwnd)
                                logger.info(f"Focused locked window: {current_hwnd}")
                                event.accept()
                                return
                            except Exception as focus_err:
                                logger.debug(f"Failed to focus window {current_hwnd}: {focus_err}")
                    
                    # If we can't focus the specific window, just accept the event to prevent switching
                    event.accept()
                    return
            except Exception as lock_err:
                logger.debug(f"Error checking lock state: {lock_err}")
            
            # Proceed with quickswitch for left-button double-clicks when not locked
            try:
                from core.switching.quickswitch_controller import get_quickswitch_controller
                ctrl = get_quickswitch_controller()
                ctrl.quickswitch("overlay_host.double_click")
                event.accept()
                return
            except Exception as e:
                logger.error(f"Error during quickswitch: {e}")
        except Exception as e:
            logger.error(f"Error in mouseDoubleClickEvent: {e}")
            
        # Default handling if we couldn't process the event
        super().mouseDoubleClickEvent(event)

    def set_host_geometry(self, pos: QPoint, size: QSize) -> None:
        # Honor minimums
        w = max(size.width(), IntegratedBorderCanvas.MIN_WIDTH)
        h = max(size.height(), IntegratedBorderCanvas.MIN_HEIGHT)
        self.setGeometry(QRect(pos, QSize(w, h)))

    def set_window_opacity(self, opacity: float) -> None:
        """Set the window-level opacity for the entire overlay.
        This ensures the entire overlay (including pillarbox/letterbox bars) is transparent.
        """
        try:
            # Apply opacity to the entire window
            self.setWindowOpacity(max(0.0, min(1.0, float(opacity))))
        except Exception:
            pass

    def _reset_cursor_safely(self) -> None:
        """Ensure the cursor is reset back to Arrow when the host deactivates/hides.
        Uses centralized CursorManager and WindowBehaviorManager for proper coordination.
        """
        try:
            # First use window behavior manager for consistent state handling
            if hasattr(self, "_behavior") and self._behavior is not None:
                self._behavior.handle_leave()
                
            # Then use cursor manager to reset cursor with proper priority
            cursor_manager = get_cursor_manager()
            cursor_manager.set_cursor(
                requester="OverlayHost",
                widget=self,
                cursor=Qt.ArrowCursor,
                priority=CursorPriority.DEFAULT,
                reason="Host deactivation/hide cursor reset"
            )
        except Exception as e:
            # Never raise from UI event hooks; log explicitly
            logger = get_logger("OverlayHost")
            logger.debug(f"Cursor reset via CursorManager failed: {e}")
            
            # Fallback to direct cursor reset
            try:
                self.setCursor(Qt.ArrowCursor)
            except Exception:
                pass

    def changeEvent(self, event):
        # Reset cursor when the window loses activation
        try:
            if event and event.type() == QEvent.ActivationChange:
                is_active = self.isActiveWindow()
                logger = get_logger("OverlayHost")
                logger.info(f"FOCUS_AUDIT: ActivationChange: is_active={is_active}")
                self._on_window_activation_changed(is_active)
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.error(f"changeEvent failed: {e}")
        super().changeEvent(event)

    def _on_window_activation_changed(self, is_active: bool) -> None:
        """Handle window activation state changes."""
        try:
            logger = get_logger("OverlayHost")
            logger.info(f"FOCUS_AUDIT: Window activation changed: is_active={is_active}")
            if not is_active:
                self._reset_cursor_safely()
                # Safety: ensure any volume hold is released on deactivation
                try:
                    kp = get_key_passthrough_controller()
                    VK_VOLUME_UP = 0xAF
                    VK_VOLUME_DOWN = 0xAE
                    kp.release_passthrough_key(VK_VOLUME_UP)
                    kp.release_passthrough_key(VK_VOLUME_DOWN)
                except Exception:
                    pass
            # Update focus indicator visibility for both active and inactive states
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                logger.info(f"FOCUS_AUDIT: Setting focus indicator visibility: {is_active}")
                self._focus_indicator.set_has_focus(is_active)
            else:
                logger.info("FOCUS_AUDIT: No focus indicator to update")

            # Update KeyPassthroughController target HWND based on focus state
            try:
                kp = get_key_passthrough_controller()
                ov = getattr(self, "_parent_overlay", None)
                # Route only when host is active and we have a valid captured source hwnd
                if is_active and ov is not None and hasattr(ov, "_source_hwnd"):
                    try:
                        src_hwnd = int(getattr(ov, "_source_hwnd") or 0)
                    except Exception:
                        src_hwnd = 0
                    kp.set_target_hwnd(src_hwnd if src_hwnd else None)
                else:
                    # Clear target on deactivation or when overlay lacks a source hwnd
                    kp.set_target_hwnd(None)
            except Exception as e:
                logger.debug(f"Passthrough target update failed: {e}")
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.error(f"_on_window_activation_changed failed: {e}")

    def showEvent(self, event):
        """Handle show event to trigger initial focus indicator visibility."""
        try:
            # Show focus indicator when overlay becomes visible and is active
            is_active = self.isActiveWindow()
            logger = get_logger("OverlayHost")
            logger.info(f"FOCUS_AUDIT: showEvent: is_active={is_active}")
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                logger.info(f"FOCUS_AUDIT: Setting focus indicator on show: {is_active}")
                self._focus_indicator.set_has_focus(is_active)
            else:
                logger.debug("No focus indicator on show")
            # Sync passthrough target on show as well
            try:
                kp = get_key_passthrough_controller()
                ov = getattr(self, "_parent_overlay", None)
                if is_active and ov is not None and hasattr(ov, "_source_hwnd"):
                    try:
                        src_hwnd = int(getattr(ov, "_source_hwnd") or 0)
                    except Exception:
                        src_hwnd = 0
                    kp.set_target_hwnd(src_hwnd if src_hwnd else None)
                else:
                    kp.set_target_hwnd(None)
            except Exception as e:
                logger.debug(f"Passthrough target show sync failed: {e}")
            # Refresh OSD position on show
            try:
                if getattr(self, "_volume_osd", None) is not None:
                    self._volume_osd.update_position()
            except Exception:
                pass
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.error(f"showEvent failed: {e}")
        super().showEvent(event)

    def focusOutEvent(self, event):
        # On focus out, make sure any resize cursor is cleared
        try:
            self._reset_cursor_safely()
            # Hide focus indicator
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                self._focus_indicator.set_has_focus(False)
            # Safety: release any volume holds if focus is lost
            try:
                kp = get_key_passthrough_controller()
                VK_VOLUME_UP = 0xAF
                VK_VOLUME_DOWN = 0xAE
                kp.release_passthrough_key(VK_VOLUME_UP)
                kp.release_passthrough_key(VK_VOLUME_DOWN)
            except Exception:
                pass
            # Clear passthrough target on focus out
            try:
                kp = get_key_passthrough_controller()
                kp.set_target_hwnd(None)
            except Exception:
                pass
        except Exception:
            pass
        super().focusOutEvent(event)

    def hideEvent(self, event):
        # On hide, ensure we never leave an overridden cursor active
        try:
            self._reset_cursor_safely()
            # Hide and cleanup focus indicator
            self._cleanup_focus_indicator()
            # Safety: release any volume holds on hide
            try:
                kp = get_key_passthrough_controller()
                VK_VOLUME_UP = 0xAF
                VK_VOLUME_DOWN = 0xAE
                kp.release_passthrough_key(VK_VOLUME_UP)
                kp.release_passthrough_key(VK_VOLUME_DOWN)
            except Exception:
                pass
            # Clear passthrough target on hide
            try:
                kp = get_key_passthrough_controller()
                kp.set_target_hwnd(None)
            except Exception:
                pass
            # Hide Volume OSD window if present
            try:
                if getattr(self, "_volume_osd", None) is not None:
                    self._volume_osd.hide()
            except Exception:
                pass
        except Exception:
            pass
        super().hideEvent(event)

    def _cleanup_focus_indicator(self) -> None:
        """Clean up focus indicator to prevent it from remaining visible after overlay destruction."""
        try:
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                from core.logging import get_logger
                logger = get_logger("OverlayHost")
                logger.info("Cleaning up focus indicator")
                
                # Hide immediately
                self._focus_indicator.hide()
                
                # Set focus state to false
                self._focus_indicator.set_has_focus(False)
                
                # Schedule deletion via ThreadManager to avoid Qt lifecycle issues
                from core.threading import ThreadManager
                indicator = self._focus_indicator
                self._focus_indicator = None
                
                # Defer deletion to avoid Qt parent-child cleanup conflicts
                ThreadManager.single_shot(50, lambda: self._safe_delete_indicator(indicator))
                
        except Exception as e:
            from core.logging import get_logger
            logger = get_logger("OverlayHost")
            logger.debug(f"Focus indicator cleanup failed: {e}")

    def _safe_delete_indicator(self, indicator) -> None:
        """Safely delete focus indicator widget."""
        try:
            if indicator is not None:
                indicator.deleteLater()
        except Exception:
            pass

    # Emit geometryChanged for integrated rendering coordination
    def moveEvent(self, event):
        try:
            # Keep focus indicator anchored
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                try:
                    self._focus_indicator.update_position()
                    # Sync locked state from parent overlay if available
                    ov = getattr(self, "_parent_overlay", None)
                    if ov is not None and hasattr(ov, "_is_window_locked"):
                        self._focus_indicator.set_locked(bool(getattr(ov, "_is_window_locked", False)))
                except Exception:
                    pass
            self.geometryChanged.emit()
        except Exception:
            pass
        super().moveEvent(event)

    def resizeEvent(self, event):
        try:
            # Keep focus indicator anchored and properly sized
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                try:
                    self._focus_indicator.update_position()
                except Exception:
                    pass
            self.geometryChanged.emit()
        except Exception:
            pass
        super().resizeEvent(event)
    
    def _check_initial_focus(self) -> None:
        """Check initial focus state and update focus indicator accordingly."""
        try:
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                is_active = self.isActiveWindow()
                logger = get_logger("OverlayHost")
                logger.info(f"FOCUS_AUDIT: _check_initial_focus: is_active={is_active}")
                self._focus_indicator.set_has_focus(is_active)
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.debug(f"Initial focus check failed: {e}")
    
    def _sync_passthrough_state(self) -> None:
        """Sync passthrough state to focus indicator to ensure correct color on startup."""
        try:
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                kp = get_key_passthrough_controller()
                current_state = kp.is_enabled()
                logger = get_logger("OverlayHost")
                logger.debug(f"FOCUS_AUDIT: Syncing passthrough state: {current_state}")
                self._focus_indicator.set_passthrough_enabled(current_state)
        except Exception as e:
            logger = get_logger("OverlayHost")
            logger.debug(f"Passthrough state sync failed: {e}")

    def flash_focus_indicator(self, duration_ms: int = 300) -> None:
        """Briefly flash the focus indicator black for the given duration.

        Safe to call from any thread. Uses ThreadManager to route work to the UI thread.
        """
        try:
            from core.threading import ThreadManager

            def _do_flash():
                try:
                    if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                        self._focus_indicator.flash_block(duration_ms)
                except Exception:
                    pass

            ThreadManager.run_on_ui_thread(_do_flash)
        except Exception:
            # Fallback: call directly
            if hasattr(self, "_focus_indicator") and self._focus_indicator is not None:
                self._focus_indicator.flash_block(duration_ms)
