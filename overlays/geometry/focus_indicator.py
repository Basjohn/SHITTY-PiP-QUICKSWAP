from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPainterPath, QPen, QBrush, QPixmap
from PySide6.QtWidgets import QWidget

from utils.theme.theme_manager import ThemeManager
from utils.state.focus_state import get_focus_state



class FocusIndicatorWidget(QWidget):
    """
    A small indicator shown at the bottom-right of the overlay host to convey:
    - Focus/active hint (subtle circle)
    - Key passthrough enabled (red circle)
    - Locked (padlock glyph)

    Interaction: Left-click toggles lock via a signal; the parent overlay/host owns the state.
    """

    lock_toggled = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("focusIndicator")
        self.setWindowFlags(Qt.Widget)  # Child widget, not a window
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # Accept mouse events for lock toggle
        self.setAttribute(Qt.WA_NoSystemBackground, True)  # Transparent background
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)  # Allow transparent painting
        # Ensure this child stays above siblings and DWM thumbnail content without changing global z-order
        self.setAttribute(Qt.WA_AlwaysStackOnTop, True)
        # Create a native child window (own HWND) so it composites above DWM thumbnail drawn in parent
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setAutoFillBackground(False)

        # Theme manager for colors (fallbacks provided)
        self._theme = ThemeManager.instance()

        # State
        self._locked: bool = False
        self._passthrough_enabled: bool = False
        self._has_focus: bool = False
        # Temporary flash state (e.g., on block) – paints indicator black briefly
        self._flash_active: bool = False

        # Metrics (scale with overlay size)
        self._base_size: int = 16  # Base size for small overlays
        self._max_size: int = 24   # Max size for large overlays
        self._margin_px: int = 6

        # Hint cursor on hover
        self.setCursor(Qt.PointingHandCursor)

        # Start hidden - only show when overlay has focus
        self.hide()

        # Initial size
        self.resize(self.sizeHint())

        # Lazy-loaded pixmap for lock icon (only shown when locked)
        self._icon_lock: Optional[QPixmap] = None

    # --- Public API -----------------------------------------------------
    def set_locked(self, locked: bool) -> None:
        if bool(locked) != self._locked:
            self._locked = bool(locked)
            self.update()

    def set_passthrough_enabled(self, enabled: bool) -> None:
        if bool(enabled) != self._passthrough_enabled:
            self._passthrough_enabled = bool(enabled)
            self.update()

    def set_has_focus(self, has_focus: bool) -> None:
        """Show/hide indicator based on overlay focus state."""
        from core.logging import get_logger
        logger = get_logger("FocusIndicator")
        logger.info(f"FOCUS_AUDIT: set_has_focus called: {has_focus}, current: {self._has_focus}")
        logger.info(f"FOCUS_AUDIT: Widget state - parent: {self.parentWidget()}, isVisible: {self.isVisible()}, geometry: {self.geometry()}")
        if bool(has_focus) != self._has_focus:
            self._has_focus = bool(has_focus)
            # Update centralized focus state (thread-safe)
            try:
                get_focus_state().set_overlay_focused(self._has_focus)
            except Exception:
                pass
            if self._has_focus:
                logger.info("FOCUS_AUDIT: Showing focus indicator")
                self.update_position()
                self.raise_()  # Ensure Qt stacking
                self.show()
                self.repaint()  # Force immediate repaint
                # Ensure native z-order is above DWM thumbnail (non-topmost)
                try:
                    from utils.resource_manager import get_resource_manager
                    rm = get_resource_manager()
                    success = rm.bring_child_to_front(self)
                    logger.info(f"FOCUS_AUDIT: bring_child_to_front result: {success}")
                except Exception as e:
                    logger.warning(f"FOCUS_AUDIT: bring_child_to_front failed: {e}")
                logger.info(f"FOCUS_AUDIT: Focus indicator shown at {self.pos()}, size {self.size()}, isVisible: {self.isVisible()}")
                logger.info(f"FOCUS_AUDIT: Parent geometry: {self.parentWidget().geometry() if self.parentWidget() else 'No parent'}")
            else:
                logger.info("FOCUS_AUDIT: Hiding focus indicator")
                self.hide()

    def flash_block(self, duration_ms: int = 300) -> None:
        """Temporarily force the indicator to render black for the given duration.

        Safe to call from any thread; scheduling is routed via ThreadManager.
        """
        try:
            from core.threading import ThreadManager

            def _start_flash():
                try:
                    self._flash_active = True
                    self.update()
                finally:
                    # Schedule end of flash
                    try:
                        ThreadManager.single_shot(max(0, int(duration_ms)), _end_flash)
                    except Exception:
                        # Best-effort immediate end if scheduler fails
                        _end_flash()

            def _end_flash():
                self._flash_active = False
                self.update()

            # Ensure we start on the UI thread
            ThreadManager.run_on_ui_thread(_start_flash)
        except Exception:
            # Fallback (best-effort) without ThreadManager – execute inline and clear immediately
            self._flash_active = True
            self.update()
            # Final fallback: immediately clear without using QTimer
            self._flash_active = False
            self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        size = self._calculate_size()
        return QSize(size, size)

    def _calculate_size(self) -> int:
        """Calculate size based on parent overlay dimensions."""
        if self.parentWidget() is None:
            return self._base_size
        
        parent_rect = self.parentWidget().rect()
        # Scale based on smaller dimension to handle both wide and tall overlays
        min_dim = min(parent_rect.width(), parent_rect.height())
        
        if min_dim < 200:
            return self._base_size
        elif min_dim > 800:
            return self._max_size
        else:
            # Linear interpolation between base and max size
            ratio = (min_dim - 200) / (800 - 200)
            return int(self._base_size + ratio * (self._max_size - self._base_size))

    def update_position(self, parent_rect: Optional[QRect] = None) -> None:
        """Update position to bottom-right corner of parent with proper bounds checking."""
        from core.logging import get_logger
        logger = get_logger("FocusIndicator")
        
        p = self.parentWidget()
        if p is None:
            logger.info("FOCUS_AUDIT: update_position - no parent widget")
            return
        
        # Recalculate size based on current parent size
        new_size = self._calculate_size()
        self.resize(new_size, new_size)
        logger.info(f"FOCUS_AUDIT: update_position - calculated size: {new_size}")
        
        # Use parent's actual geometry, not rect() which might be relative
        if parent_rect is not None:
            rect = parent_rect
        else:
            rect = p.rect()
        
        logger.info(f"FOCUS_AUDIT: update_position - parent rect: {rect}")
        
        w = self.width()
        h = self.height()
        
        # Position in bottom-right corner with margin
        x = rect.width() - w - self._margin_px
        y = rect.height() - h - self._margin_px
        
        # Ensure coordinates stay within parent bounds
        x = max(0, min(x, rect.width() - w))
        y = max(0, min(y, rect.height() - h))
        
        logger.info(f"FOCUS_AUDIT: update_position - calculated position: ({x}, {y})")
        self.move(QPoint(x, y))
        logger.info(f"FOCUS_AUDIT: update_position - actual position after move: {self.pos()}")
        
        # Ensure widget is raised to top and visible
        self.raise_()
        if self._has_focus:
            self.show()
            logger.info(f"FOCUS_AUDIT: update_position - widget shown, isVisible: {self.isVisible()}")
            # Ensure native z-order is enforced each move/resize while visible
            try:
                from utils.resource_manager import get_resource_manager
                rm = get_resource_manager()
                success = rm.bring_child_to_front(self)
                logger.debug(f"FOCUS_AUDIT: bring_child_to_front (update_position) result: {success}")
            except Exception as e:
                logger.warning(f"FOCUS_AUDIT: bring_child_to_front (update_position) failed: {e}")

    def attach(self, parent: QWidget) -> None:
        from core.logging import get_logger
        logger = get_logger("FocusIndicator")
        logger.debug(f"Attaching to parent: {parent}")
        self.setParent(parent)
        # Don't show immediately - wait for focus
        self.update_position()
        logger.debug(f"Attached at position {self.pos()}, size {self.size()}, visible: {self.isVisible()}")

    # --- Events ---------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event and event.button() == Qt.LeftButton:
            self.lock_toggled.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Paint the focus indicator with state-dependent visuals."""
        from core.logging import get_logger
        logger = get_logger("FocusIndicator")
        logger.debug(f"paintEvent: locked={self._locked}, passthrough={self._passthrough_enabled}, visible={self.isVisible()}")

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Visuals: default white with mild black border; red when passthrough enabled,
        # and black override during a temporary flash state (e.g., on block)
        normal_color = "#FFFFFF"
        passthrough_color = "#FF4444"

        if self._flash_active:
            base_color = QColor("#000000")
            base_color.setAlpha(220)
        else:
            base_color = QColor(passthrough_color if self._passthrough_enabled else normal_color)
            base_color.setAlpha(220)  # mildly translucent

        rect = self.rect().adjusted(2, 2, -2, -2)  # Small margin
        painter.setBrush(QBrush(base_color))
        painter.setPen(QPen(QColor("#000000"), 1))  # mild black border
        painter.drawEllipse(rect)

        # Draw lock icon overlay (only when locked)
        try:
            icon = self._get_lock_icon_pixmap(locked=self._locked)
            if icon and not icon.isNull():
                # Fit as large as possible within the inner circle, keeping aspect ratio
                pad = max(2, int(min(rect.width(), rect.height()) * 0.12))
                target = QRect(rect.x() + pad, rect.y() + pad, rect.width() - 2 * pad, rect.height() - 2 * pad)
                scaled = icon.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = target.x() + (target.width() - scaled.width()) // 2
                y = target.y() + (target.height() - scaled.height()) // 2
                painter.drawPixmap(x, y, scaled)
        except Exception as e:
            logger = get_logger("FocusIndicator")
            logger.debug(f"Lock icon draw failed: {e}")

    def _get_color_token(self, key: str, fallback: QColor) -> QColor:
        try:
            # ThemeManager.apply_theme_to_widget works via QSS; for painting, use token-based fetch if available
            # Here we fall back to provided color as ThemeManager does not expose direct token API yet.
            _ = self._theme  # keep reference
            return QColor(fallback)
        except Exception:
            return QColor(fallback)

    def _draw_lock(self, painter: QPainter, color: QColor, size: int) -> None:
        # Draw a simple padlock centered inside the circle
        body_w = max(6, int(size * 0.5))
        body_h = max(6, int(size * 0.45))
        body_x = (self.width() - body_w) // 2
        body_y = (self.height() - body_h) // 2 + int(size * 0.05)

        pen = QPen(color)
        pen.setWidthF(max(1.0, size * 0.06))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        # Shackle
        shackle_r = body_w / 2.0
        shackle_center = QPoint(self.width() // 2, body_y)
        rect = QRect(shackle_center.x() - shackle_r, body_y - shackle_r, int(2 * shackle_r), int(2 * shackle_r))
        path = QPainterPath()
        path.arcMoveTo(rect, 0)
        path.arcTo(rect, 0, 180)
        painter.drawPath(path)

        # Body
        painter.drawRoundedRect(QRect(body_x, body_y, body_w, body_h), 2, 2)

    # --- Internal helpers -------------------------------------------------
    def _get_lock_icon_pixmap(self, locked: bool) -> Optional[QPixmap]:
        """Return lock icon only when locked; blank when unlocked.
        Loads `resources/lock.png` lazily. Returns None if unlocked or load fails.
        """
        if not locked:
            return None
        if self._icon_lock is None:
            self._icon_lock = self._load_pixmap_from_resources("lock.png")
        return self._icon_lock

    def _load_pixmap_from_resources(self, filename: str) -> Optional[QPixmap]:
        """Resolve and load a pixmap from the project's resources directory.
        Robustly computes the project root relative to this file.
        """
        try:
            import os
            # This file: ui/overlays/geometry/focus_indicator.py
            here = os.path.abspath(os.path.dirname(__file__))
            # project_root = here/../../.. (geometry -> overlays -> ui -> project root)
            project_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))
            res_path = os.path.join(project_root, "resources", filename)
            if not os.path.exists(res_path):
                # Fallback: try relative to project_root/../resources if structure slightly differs
                alt = os.path.abspath(os.path.join(project_root, os.pardir, "resources", filename))
                res_path = alt if os.path.exists(alt) else res_path
            pm = QPixmap(res_path)
            return pm if not pm.isNull() else None
        except Exception:
            return None


class FocusIndicatorWindow(FocusIndicatorWidget):
    """A top-level, non-activating indicator window that tracks an OverlayHost.

    This avoids child compositing issues with DWM thumbnails by giving the indicator
    its own HWND. It is kept above the host via native z-order operations without
    using global topmost.
    """

    def __init__(self, host_widget: QWidget) -> None:
        super().__init__(parent=None)
        self._host_ref: Optional[QWidget] = host_widget
        # Convert to top-level tool window with no activation and frameless
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowDoesNotAcceptFocus
            | Qt.WindowStaysOnTopHint  # Match host's topmost band so we can sit above it
        )
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        # Enable per-pixel transparency for a frameless top-level indicator
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NativeWindow, True)
        # Start hidden until host says it has focus
        self.hide()

    def sizeHint(self) -> QSize:  # noqa: N802
        # Use base calculation but cap aggressively; this is a tiny window
        return super().sizeHint()

    def update_position(self, parent_rect: Optional[QRect] = None) -> None:
        """Position relative to the host in screen coordinates and enforce z-order."""
        from core.logging import get_logger
        logger = get_logger("FocusIndicator")

        host = self._host_ref
        if host is None:
            logger.info("FOCUS_AUDIT: window.update_position - no host")
            return

        # Compute size using host dimensions
        new_size = self._calculate_size()
        self.resize(new_size, new_size)

        # Determine host content rect
        rect = parent_rect if parent_rect is not None else host.rect()

        # Bottom-right with margin
        w = self.width()
        h = self.height()
        x_local = rect.width() - w - self._margin_px
        y_local = rect.height() - h - self._margin_px
        x_local = max(0, min(x_local, rect.width() - w))
        y_local = max(0, min(y_local, rect.height() - h))

        # Map to global (screen) coords
        top_left = host.mapToGlobal(QPoint(0, 0))
        self.move(top_left + QPoint(x_local, y_local))

        # Show/stack if focused
        self.raise_()
        if self._has_focus:
            self.show()
            try:
                # Keep directly above host (same z-band)
                from utils.resource_manager import get_resource_manager
                rm = get_resource_manager()
                rm.place_window_above(self, host)
            except Exception as e:
                logger.debug(f"FOCUS_AUDIT: window.place_above failed: {e}")

    def set_has_focus(self, has_focus: bool) -> None:
        """Override to avoid parenting assumptions and operate as a window."""
        from core.logging import get_logger
        logger = get_logger("FocusIndicator")
        if bool(has_focus) != self._has_focus:
            self._has_focus = bool(has_focus)
            # Update centralized focus state (thread-safe)
            try:
                get_focus_state().set_overlay_focused(self._has_focus)
            except Exception:
                pass
            if self._has_focus:
                self.update_position()
                self.raise_()
                self.show()
                try:
                    host = self._host_ref
                    if host is not None:
                        from utils.resource_manager import get_resource_manager
                        rm = get_resource_manager()
                        rm.place_window_above(self, host)
                except Exception as e:
                    logger.debug(f"FOCUS_AUDIT: window.place_above on focus failed: {e}")
            else:
                self.hide()
