from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPaintEvent, QResizeEvent, QColor, QPainterPath, QRegion

from .rendering.border_renderer import BorderRenderer
from .geometry.border_geometry import BorderGeometry, BorderMetrics
from .theming.border_theme import BorderTheme
from core.logging import get_logger
from utils.debug import debug_enabled
from utils.window.overlay_constants import OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT


class IntegratedBorderCanvas(QWidget):
    """
    Unified overlay canvas with integrated border rendering.
    
    This eliminates the need for a separate BorderOverlay window by rendering
    borders directly on the canvas, avoiding all z-order and transparency issues.
    
    Features:
    - Direct border rendering in paintEvent (no separate window)
    - No mouse transparency conflicts
    - Simplified event handling
    - Better performance (single paint cycle)
    - Maintains all existing functionality
    """

    # Minimum size constants (centralized)
    MIN_WIDTH = OVERLAY_MIN_WIDTH
    MIN_HEIGHT = OVERLAY_MIN_HEIGHT

    contentRectChanged = Signal(QRect)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("integratedBorderCanvas")
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        
        # Border rendering components (integrated from BorderOverlay)
        self._border_renderer = BorderRenderer()
        self._border_geometry = BorderGeometry()
        
        # Centralized theme integration with minimal logging
        try:
            from utils.theme.theme_manager import ThemeManager
            self._theme_manager = ThemeManager.instance()
            self._border_theme = BorderTheme(self._theme_manager)
            
            if not self._border_theme.validate_required_tokens():
                raise ValueError("Required theme tokens missing")
                
            # Connect to theme changes for automatic updates
            self._theme_manager.theme_changed.connect(self._on_theme_changed)
        except Exception as e:
            logger = get_logger("IntegratedBorderCanvas")
            logger.error(f"Theme initialization failed: {e}")
            raise
        
        # Border state
        self._border_metrics: Optional[BorderMetrics] = None
        self._rounded_enabled: bool = False
        
        # Wheel handling delegated to WindowBehaviorManager (no local accumulator)
        
        # Content rect caching and aspect ratio
        self._cached_content_rect = QRect()
        self._last_size = QRect()
        self._content_aspect: Optional[tuple[int, int]] = None
        
        # Remove backdrop frame approach - it causes stacking issues
        # The canvas itself will handle all rendering with proper clipping
        self._backdrop_frame = None
        self._backdrop_opacity_effect = None
        
        # Initialize settings monitoring
        self._setup_settings_monitoring()
        
        # Set minimum size
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        
        # Initial geometry calculation
        self._update_layout()

    def _setup_settings_monitoring(self) -> None:
        """Monitor settings for live border updates."""
        try:
            from core.settings.settings_manager import SettingsManager
            settings = SettingsManager()
            
            # Load initial rounded setting
            self._rounded_enabled = bool(settings.get('overlay.rounded_borders', False))
            
            # Register for live updates
            settings.register_change_handler('overlay.rounded_borders', self._on_border_settings_changed)
            settings.register_change_handler('overlay.larger_borders', self._on_border_settings_changed)
            settings.register_change_handler('theme', self._on_theme_changed)
            
        except Exception as e:
            logger = get_logger("IntegratedBorderCanvas")
            logger.warning(f"Settings monitoring setup failed: {e}")

    def _on_border_settings_changed(self, key: str, value: object) -> None:
        """Handle border settings changes (rounded, larger, etc)."""
        if key == 'overlay.rounded_borders':
            self._rounded_enabled = bool(value)
        # For larger_borders or any border-affecting setting, invalidate cached metrics
        self._border_geometry.clear_cache()
        self._border_metrics = None
        self.update()

    def _on_theme_changed(self, *args) -> None:
        """Handle theme changes from ThemeManager and SettingsManager.
        
        Supports both signatures:
        - ThemeManager: (theme_name)
        - SettingsManager: (key, value)
        """
        try:
            # Determine theme name from args
            theme_name = None
            if len(args) == 1:
                theme_name = str(args[0])
            elif len(args) >= 2:
                # Expect (key, value)
                theme_name = str(args[1])

            # Log border enforcement once per theme change (avoid per-paint spam)
            if theme_name is not None and 'dark' in theme_name.lower():
                get_logger("BorderTheme").debug(f"Dark theme detected '{theme_name}': enforcing white border")

            if not self._border_theme.validate_required_tokens():
                return
            
            # Force theme update and redraw
            self._border_geometry.clear_cache()
            self._border_metrics = None
            
            # Ensure styling is also updated
            self._theme_manager.apply_theme_to_widget(self)
            
            # Reapply window masking for new theme
            try:
                parent = self.parent()
                if parent and hasattr(parent, '_parent_overlay'):
                    overlay = parent._parent_overlay
                    if hasattr(overlay, '_apply_window_masking'):
                        overlay._apply_window_masking()
            except Exception as e:
                logger = get_logger("IntegratedBorderCanvas")
                logger.debug(f"Window mask update after theme change failed: {e}")
            
            # Trigger repaint
            self.update()
            
            logger = get_logger("IntegratedBorderCanvas")
            if debug_enabled:
                if theme_name is not None:
                    logger.debug(f"Applied theme '{theme_name}' to integrated border canvas")
                else:
                    logger.debug("Applied theme update to integrated border canvas")
            
        except Exception as e:
            logger = get_logger("IntegratedBorderCanvas")
            logger.error(f"Theme change handling failed: {e}")

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the canvas with integrated border rendering and bleed prevention."""
        painter = QPainter(self)
        if not painter.isActive():
            return
            
        try:
            # Calculate border metrics if needed
            if not self._border_metrics:
                dpi_scale = self.devicePixelRatioF()
                size = self.size()
                thickness_base = self._border_theme.get_border_thickness_base()
                self._border_metrics = self._border_geometry.calculate_border_metrics(
                    size, dpi_scale, self._rounded_enabled, thickness_base
                )
            
            # Skip border rendering if metrics are invalid
            if not self._border_metrics or not self._border_metrics.is_valid():
                return
                
            # Get theme and metrics
            theme = self._border_theme
            metrics = self._border_metrics
            
            # Enable antialiasing for smooth borders
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            
            # Clear background within widget bounds to avoid bleed-through
            painter.fillRect(self.rect(), QColor(0, 0, 0, 0))

            # Apply coordinated masking to both this widget and parent if we have rounded corners
            dpi_scale = self.devicePixelRatioF()
            logical_radius = metrics.corner_radius
            if metrics.corner_radius > 0:
                # Create mask path with conservative inset to prevent bleed
                mask_path = QPainterPath()
                mask_inset = max(0.5, 1.0 / dpi_scale)  # At least 0.5px, or 1 logical pixel at current DPI
                mask_rect = self.rect().adjusted(mask_inset, mask_inset, -mask_inset, -mask_inset)
                mask_path.addRoundedRect(mask_rect, logical_radius, logical_radius)
                
                # Convert to region and apply as widget mask
                mask_region = QRegion(mask_path.toFillPolygon().toPolygon())
                self.setMask(mask_region)
                
                # Also apply mask to parent widget to prevent background bleed
                if self.parent() and hasattr(self.parent(), 'setMask'):
                    try:
                        parent_mask_path = QPainterPath()
                        parent_rect = self.parent().rect()
                        parent_mask_path.addRoundedRect(parent_rect, logical_radius, logical_radius)
                        parent_mask_region = QRegion(parent_mask_path.toFillPolygon().toPolygon())
                        self.parent().setMask(parent_mask_region)
                    except Exception:
                        pass  # Silently continue if parent masking fails
            else:
                self.clearMask()
                # Clear parent mask too
                if self.parent() and hasattr(self.parent(), 'clearMask'):
                    try:
                        self.parent().clearMask()
                    except Exception:
                        pass

            # Set up clipping path for antialiased painting
            if logical_radius > 0:
                clip_path = QPainterPath()
                clip_path.addRoundedRect(self.rect(), logical_radius, logical_radius)
                painter.setClipPath(clip_path)
            
            # Render main border
            self._border_renderer.render_border(
                painter, self.rect(), metrics.thickness, 
                theme.get_border_color(), metrics.corner_radius
            )
            
            # Render inner accent for depth effect using unified calculator
            try:
                from .accent_calculator import get_accent_calculator
                
                # Get theme accent properties
                base_accent_thickness = theme.get_accent_thickness()  # 1.0 from theme
                base_accent_inset = theme.get_accent_inset()  # 3.0 from theme
                
                # Use unified accent calculator
                calculator = get_accent_calculator()
                accent_props = calculator.calculate_accent_properties(
                    widget_rect=self.rect(),
                    border_thickness=metrics.thickness,
                    corner_radius=metrics.corner_radius,
                    dpi_scale=dpi_scale,
                    theme_base_thickness=base_accent_thickness,
                    theme_base_inset=base_accent_inset
                )
                
                self._border_renderer.render_inner_accent(
                    painter, self.rect(), theme.get_accent_color(), 
                    accent_props.thickness, accent_props.inset, accent_props.inner_radius
                )
            except Exception:
                pass  # Skip inner accent if theme methods fail
        finally:
            painter.end()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle resize events and update layout."""
        super().resizeEvent(event)
        
        # Clear cached metrics on resize
        self._border_metrics = None
        
        # Update layout components
        self._update_layout()
        
        # Reapply window masking after resize
        try:
            parent = self.parent()
            if parent and hasattr(parent, '_parent_overlay'):
                overlay = parent._parent_overlay
                if hasattr(overlay, '_apply_window_masking'):
                    overlay._apply_window_masking()
        except Exception as e:
            logger = get_logger("IntegratedBorderCanvas")
            logger.debug(f"Window mask update after resize failed: {e}")

        # Trigger repaint
        self.update()
            
        logger = get_logger("IntegratedBorderCanvas")
        if debug_enabled:
            logger.debug("Applied resize to integrated border canvas")

    def _update_layout(self) -> None:
        """Update internal layout and content rect."""
        try:
            # Update content rect
            self._update_content_rect()
            
        except Exception as e:
            logger = get_logger("IntegratedBorderCanvas")
            logger.debug(f"Layout update failed: {e}")

    def _apply_backdrop_clipping(self) -> None:
        """Deprecated - backdrop frame removed to fix stacking issues."""
        pass

    def _calc_content_rect(self) -> QRect:
        """Calculate the content rect within the border area with precise aspect ratio support."""
        import math
        
        # Get base inner area after accounting for border thickness
        if not self._border_metrics:
            border_thickness = 2.0  # fallback
            corner_radius = 0.0
        else:
            border_thickness = self._border_metrics.thickness
            corner_radius = self._border_metrics.corner_radius
        
        # Base content area
        outer = self.rect().adjusted(
            int(border_thickness), int(border_thickness),
            -int(border_thickness), -int(border_thickness)
        )
        
        # Additional corner radius accommodation for rounded borders
        if corner_radius > 0 and self._rounded_enabled:
            corner_margin = max(1, int(corner_radius * 0.15))  # Conservative margin
            outer = outer.adjusted(
                corner_margin, corner_margin,
                -corner_margin, -corner_margin
            )
        
        # Ensure inner accent remains visible by shrinking content rect by accent inset
        try:
            # Read accent metrics from theme and apply conservative DPI-aware inset
            accent_thickness = self._border_theme.get_accent_thickness()
            accent_inset = self._border_theme.get_accent_inset()
            dpi_scale = self.devicePixelRatioF()
            dpi_corrected_inset = max(accent_inset, accent_inset * dpi_scale / 1.5)
            # Include thickness to avoid any overlap when scaling; ceil for safety
            accent_margin = max(0, int(math.ceil(dpi_corrected_inset + accent_thickness)))
            if accent_margin > 0:
                outer = outer.adjusted(
                    accent_margin, accent_margin,
                    -accent_margin, -accent_margin
                )
        except Exception:
            # If theme tokens are unavailable, proceed without extra inset
            pass
        
        # If no aspect ratio is set, return the full inner area
        if not self._content_aspect:
            return outer
        
        # Apply aspect ratio fitting prioritizing exact aspect without overshoot
        ar_w, ar_h = self._content_aspect
        if ar_w <= 0 or ar_h <= 0 or outer.width() <= 0 or outer.height() <= 0:
            return outer
        
        target_ratio = ar_w / ar_h
        outer_ratio = outer.width() / outer.height() if outer.height() else target_ratio
        
        if outer_ratio > target_ratio:
            # Outer is wider -> pillarbox (vertical bars on sides)
            h = outer.height()
            w = math.floor(h * target_ratio)  # Use floor to avoid overshoot
            x = outer.x() + (outer.width() - w) // 2  # Integer division for pixel alignment
            y = outer.y()
        else:
            # Outer is taller -> letterbox (horizontal bars on top/bottom)
            w = outer.width()
            h = math.floor(w / target_ratio)  # Use floor to avoid overshoot
            x = outer.x()
            y = outer.y() + (outer.height() - h) // 2  # Integer division for pixel alignment
        
        # Ensure minimum size and bounds checking
        w = max(1, min(w, outer.width()))
        h = max(1, min(h, outer.height()))
        
        return QRect(x, y, w, h)

    def _update_content_rect(self) -> None:
        """Update the cached content rect and emit signal if changed."""
        rect = self._calc_content_rect()
        if rect != self._cached_content_rect:
            self._cached_content_rect = rect
            self.contentRectChanged.emit(QRect(rect))
            self.update()

    def content_rect(self) -> QRect:
        """Return the current content rect in widget coordinates."""
        return QRect(self._cached_content_rect)

    def get_content_insets(self) -> tuple[int, int]:
        """Return per-side horizontal and vertical insets used to derive content_rect().
        
        The returned tuple is (ix, iy) where:
        - Total horizontal reduction = 2 * ix
        - Total vertical reduction   = 2 * iy
        Values are DPI-aware and mirror the logic in `_calc_content_rect()` so the
        behavior manager can preserve AR on the inner content area.
        """
        import math
        # Base from current metrics if available
        thickness = 2.0
        radius = 0.0
        if self._border_metrics:
            thickness = self._border_metrics.thickness
            radius = self._border_metrics.corner_radius
        ix = int(thickness)
        iy = int(thickness)
        # Corner margin when rounded borders are enabled
        if radius > 0 and self._rounded_enabled:
            corner_margin = max(1, int(radius * 0.15))
            ix += corner_margin
            iy += corner_margin
        # Inner accent margin (DPI corrected)
        try:
            accent_thickness = self._border_theme.get_accent_thickness()
            accent_inset = self._border_theme.get_accent_inset()
            dpi_scale = self.devicePixelRatioF()
            dpi_corrected_inset = max(accent_inset, accent_inset * dpi_scale / 1.5)
            accent_margin = max(0, int(math.ceil(dpi_corrected_inset + accent_thickness)))
            ix += accent_margin
            iy += accent_margin
        except Exception:
            pass
        return ix, iy

    def set_content_aspect(self, width: int, height: int) -> None:
        """Set the target content aspect ratio for DWM thumbnail scaling."""
        if width <= 0 or height <= 0:
            self._content_aspect = None
        else:
            self._content_aspect = (width, height)
        self._update_content_rect()

    def set_opacity(self, opacity: float) -> None:
        """Set window opacity for desktop blending (not just backdrop)."""
        try:
            opacity = max(0.0, min(1.0, float(opacity)))
            
            # Set window-level opacity for proper desktop blending
            if self.parent():
                self.parent().setWindowOpacity(opacity)
            
            # Also set backdrop opacity for consistency
            if self._backdrop_opacity_effect:
                self._backdrop_opacity_effect.setOpacity(opacity)
                
        except Exception as e:
            logger = get_logger("IntegratedBorderCanvas")
            logger.warning(f"Set opacity failed: {e}")

    def mouseDoubleClickEvent(self, event) -> None:
        """Handle double-click events for quickswitch."""
        try:
            logger = get_logger("IntegratedBorderCanvas")
            if debug_enabled:
                logger.debug("Forwarding double-click to parent overlay")
            
            # Forward to parent overlay if it has quickswitch handling
            parent = self.parent()
            if parent and hasattr(parent, '_parent_overlay'):
                overlay = parent._parent_overlay
                if hasattr(overlay, '_handle_double_click'):
                    overlay._handle_double_click(event)
                    return
            
            # Fallback: trigger quickswitch directly
            try:
                from core.switching.quickswitch_controller import get_quickswitch_controller
                logger = get_logger("IntegratedBorderCanvas")
                if debug_enabled:
                    logger.debug("Triggering quick switch from canvas double-click")
                controller = get_quickswitch_controller()
                controller.quickswitch("overlay.double_click")
                event.accept()
            except Exception as e:
                logger.error(f"Quick switch trigger failed: {e}")
                event.ignore()
                
        except Exception as e:
            logger = get_logger("IntegratedBorderCanvas")
            logger.error(f"Double-click handling failed: {e}")
            event.ignore()

    def wheelEvent(self, event) -> None:
        """Delegate wheel-based resize to WindowBehaviorManager with proper aspect ratio."""
        host = self.parent()
        if host and hasattr(host, '_behavior') and host._behavior is not None:
            try:
                # Pass current aspect ratio to ensure resize maintains proportions
                aspect_ratio = self._content_aspect if self._content_aspect else None
                insets = self.get_content_insets()
                host._behavior.handle_wheel(event, aspect_ratio, insets)
                event.accept()
                return
            except Exception as e:
                logger = get_logger("IntegratedBorderCanvas")
                logger.error(f"Wheel delegation failed: {e}")
        super().wheelEvent(event)
    
    

    def mousePressEvent(self, event) -> None:
        """Handle mouse press events."""
        try:
            # Skip right-click to allow context menu
            if hasattr(event, 'button') and event.button() == Qt.RightButton:
                pass
        except Exception:
            pass
        
        super().mousePressEvent(event)
        
        # Forward to host behavior for drag/resize
        self._forward_mouse_to_host(event, 'press')
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move events."""
        super().mouseMoveEvent(event)
        self._forward_mouse_to_host(event, 'move')

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release events."""
        super().mouseReleaseEvent(event)
        self._forward_mouse_to_host(event, 'release')

    def _forward_mouse_to_host(self, event, kind: str) -> None:
        """Forward mouse events to host window behavior manager."""
        try:
            host = self.parent()
            if not host or not hasattr(host, '_behavior'):
                return
            if kind == 'press':
                host._behavior.handle_mouse_press(event, is_draggable_region=lambda p: True)
            elif kind == 'move':
                host._behavior.handle_mouse_move(event)
            elif kind == 'release':
                host._behavior.handle_mouse_release(event)
        except Exception as e:
            logger = get_logger("IntegratedBorderCanvas")
            if debug_enabled:
                logger.debug(f"Mouse forwarding failed: {e}")

    def contextMenuEvent(self, event) -> None:
        """Context menu is owned by OverlayHost; ignore to avoid double dispatch."""
        event.ignore()
