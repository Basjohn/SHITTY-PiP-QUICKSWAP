from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath
from PySide6.QtCore import QRectF
from core.logging import get_logger
from utils.debug import debug_enabled
import time


class BorderRenderer:
    """Pure rendering engine for border drawing with zero artifacts."""
    
    def __init__(self):
        self._render_hints = QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        # Lightweight throttling for debug logs to avoid paint-cycle spam
        self._dbg_last: dict[str, float] = {}
        self._dbg_interval: float = 0.25  # seconds; ~4 logs/sec per path
        
    def render_border(self, painter: QPainter, rect: QRectF, 
                     thickness: float, color: QColor, 
                     corner_radius: float = 0.0) -> None:
        """Render border with pixel-perfect alignment and zero artifacts."""
        # Enhanced validation for small sizes
        if thickness <= 0 or rect.width() <= 1 or rect.height() <= 1:
            return
        
        # Save painter state
        painter.save()
        
        # Set optimal render hints for crisp borders
        painter.setRenderHints(self._render_hints)
        
        # For very small overlays, use a direct rectangle fill approach
        # This avoids precision issues with insets at small sizes
        if rect.width() < 20 or rect.height() < 20:
            if debug_enabled and self._should_debug('tiny'):
                try:
                    get_logger("BorderRenderer").debug(
                        "path: tiny-outline rect=%.1fx%.1f thickness=%.2f",
                        rect.width(), rect.height(), thickness
                    )
                except Exception:
                    pass
            # Use a simplified approach for tiny overlays: outline-only
            pen = QPen(color)
            pen.setWidthF(1.0)  # Fixed 1px
            pen.setJoinStyle(Qt.MiterJoin)
            pen.setCapStyle(Qt.FlatCap)

            # Pixel-align for crisp 1px stroke
            aligned = QRectF(int(rect.x()) + 0.5, int(rect.y()) + 0.5,
                             max(0.0, int(rect.width()) - 1),
                             max(0.0, int(rect.height()) - 1))

            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(aligned)
            painter.restore()
            return
        
        # For normal sized overlays, use the standard approach with insets
        # Calculate inset rect with pixel-aligned values to avoid artifacts
        inset = max(1.0, thickness / 2.0)
        draw_rect = rect.adjusted(inset, inset, -inset, -inset)
        
        # Enhanced size validation
        if draw_rect.width() < 2 or draw_rect.height() < 2:
            if debug_enabled and self._should_debug('fallback'):
                try:
                    get_logger("BorderRenderer").debug(
                        "path: fallback-outline draw_rect=%.1fx%.1f inset=%.2f thickness=%.2f",
                        draw_rect.width(), draw_rect.height(), inset, thickness
                    )
                except Exception:
                    pass
            # Fall back to simple rect outline for very small areas
            pen = QPen(color)
            pen.setWidth(1)
            pen.setJoinStyle(Qt.MiterJoin)
            pen.setCapStyle(Qt.FlatCap)

            aligned = QRectF(int(rect.x()) + 0.5, int(rect.y()) + 0.5,
                             max(0.0, int(rect.width()) - 1),
                             max(0.0, int(rect.height()) - 1))

            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(aligned)
            painter.restore()
            return
        
        # Configure pen for sharp or rounded corners
        pen = QPen(color)
        pen.setWidthF(thickness)
        pen.setCapStyle(Qt.FlatCap)
        pen.setJoinStyle(Qt.MiterJoin if corner_radius == 0 else Qt.RoundJoin)
        
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.setOpacity(1.0)  # Always full opacity for crisp borders
        
        if corner_radius > 0:
            if debug_enabled and self._should_debug('rounded'):
                try:
                    get_logger("BorderRenderer").debug(
                        "path: rounded draw draw_rect=%.1fx%.1f radius=%.2f thickness=%.2f",
                        draw_rect.width(), draw_rect.height(), corner_radius, thickness
                    )
                except Exception:
                    pass
            # Ensure radius doesn't exceed drawable area
            max_radius = min(draw_rect.width(), draw_rect.height()) / 2.0
            safe_radius = min(corner_radius, max_radius)
            
            # Create rounded rectangle path
            path = QPainterPath()
            path.addRoundedRect(draw_rect, safe_radius, safe_radius)
            painter.drawPath(path)
        else:
            if debug_enabled and self._should_debug('sharp'):
                try:
                    get_logger("BorderRenderer").debug(
                        "path: sharp draw draw_rect=%.1fx%.1f thickness=%.2f",
                        draw_rect.width(), draw_rect.height(), thickness
                    )
                except Exception:
                    pass
            # Sharp rectangle - most common case, most performant
            painter.drawRect(draw_rect)
            
        painter.restore()
        
    def render_inner_accent(self, painter: QPainter, rect: QRectF,
                           accent_color: QColor, thickness: float,
                           inset: float, corner_radius: float = 0.0) -> None:
        """Render subtle inner accent line for depth effect."""
        if thickness <= 0 or accent_color.alpha() == 0:
            return
            
        # Calculate inner accent rectangle
        accent_rect = rect.adjusted(inset, inset, -inset, -inset)
        
        # Adjust corner radius for inner accent
        inner_radius = max(0.0, corner_radius - inset) if corner_radius > 0 else 0.0
        
        # Render the accent border
        self.render_border(painter, accent_rect, thickness, accent_color, inner_radius)

    def _should_debug(self, key: str) -> bool:
        """Return True if we should emit a debug log for the given key now.
        Simple per-path rate limiting to reduce spam when debug is enabled.
        """
        try:
            now = time.monotonic()
            last = self._dbg_last.get(key, 0.0)
            if (now - last) >= self._dbg_interval:
                self._dbg_last[key] = now
                return True
            return False
        except Exception:
            # Fail-open: if timing fails for any reason, allow the log
            return True
