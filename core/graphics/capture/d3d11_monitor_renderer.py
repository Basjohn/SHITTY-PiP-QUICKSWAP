"""
D3D11MonitorRenderer - QWidget-based renderer for monitor frames.

This renderer avoids OpenGL and uses Qt's QImage/QPainter to blit frames.
It preserves the public interface used by the previous OpenGL renderer so
UI integrations (e.g., CaptureDisplayWidget) remain compatible.

Notes:
- Expects BGRA8888 (4 bytes per pixel) by default. If a 3-byte RGB buffer is
  provided, it will be converted to 4-byte format on the fly.
- Supports a hall-of-mirrors warning overlay via set_hall_of_mirrors_warning.
- Provides set_monitor_index for future integration with a polling exchange.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPainter, QColor, QFont, QFontMetrics
from PySide6.QtWidgets import QWidget

from core.logging import get_logger

logger = get_logger(__name__)


class D3D11MonitorRenderer(QWidget):
    """
    QWidget renderer that blits incoming frames as QImage without OpenGL.

    Public API:
    - render_error: Signal(str)
    - update_frame(frame): accepts CaptureFrame
    - set_monitor_index(int): hint for which exchange to read (unused for now)
    - set_hall_of_mirrors_warning(bool, str|None): overlay warning text
    - cleanup(): release retained resources
    """

    render_error = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._last_image: Optional[QImage] = None
        self._monitor_index: int = 0
        self._warning_active: bool = False
        self._warning_text: str = ""

        # Use a neutral background; actual frame will fully cover
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        logger.debug("D3D11MonitorRenderer initialized (QWidget blitter)")

    # --- Public API -----------------------------------------------------
    def update_frame(self, frame) -> None:
        try:
            w, h = int(frame.width), int(frame.height)
            data = frame.image_data
            # Determine pixel format by data length
            expected_bgra = w * h * 4
            expected_rgb = w * h * 3
            if isinstance(data, (bytes, bytearray, memoryview)):
                blit_img: Optional[QImage] = None
                if len(data) == expected_bgra:
                    # BGRA8888
                    blit_img = QImage(
                        data, w, h, w * 4, QImage.Format.Format_ARGB32
                    )
                elif len(data) == expected_rgb:
                    # Convert RGB -> BGRA (opaque)
                    try:
                        # Expand to 4 bytes per pixel
                        rgb = memoryview(data)
                        bgra = bytearray(expected_bgra)
                        # Fill
                        di = 0
                        si = 0
                        for _ in range(w * h):
                            r = rgb[si]
                            g = rgb[si + 1]
                            b = rgb[si + 2]
                            # BGRA order
                            bgra[di] = b
                            bgra[di + 1] = g
                            bgra[di + 2] = r
                            bgra[di + 3] = 255
                            si += 3
                            di += 4
                        blit_img = QImage(
                            bytes(bgra), w, h, w * 4, QImage.Format.Format_ARGB32
                        )
                    except Exception as e:
                        logger.error(f"RGB->BGRA conversion failed: {e}", exc_info=True)
                        self.render_error.emit(str(e))
                        return
                else:
                    msg = (
                        f"Unexpected frame buffer size {len(data)} for {w}x{h}. "
                        f"Expected {expected_bgra} (BGRA) or {expected_rgb} (RGB)"
                    )
                    logger.warning(msg)
                    self.render_error.emit(msg)
                    return

                # Detach to ensure data is owned, then store
                self._last_image = blit_img.copy() if blit_img is not None else None
                self.update()
            else:
                msg = "frame.image_data is not a bytes-like object"
                logger.error(msg)
                self.render_error.emit(msg)
        except Exception as e:
            logger.error(f"Frame update error: {e}", exc_info=True)
            self.render_error.emit(str(e))

    def set_monitor_index(self, index: int) -> None:
        try:
            self._monitor_index = int(index)
            # Placeholder: future integration may poll an exchange by index
        except Exception:
            pass

    def set_hall_of_mirrors_warning(self, active: bool, text: Optional[str] = None) -> None:
        self._warning_active = bool(active)
        if text is not None:
            self._warning_text = str(text)
        self.update()

    def cleanup(self) -> None:
        # Release retained image
        self._last_image = None
        logger.debug("D3D11MonitorRenderer cleaned up")

    # --- QWidget overrides ----------------------------------------------
    def sizeHint(self) -> QSize:  # type: ignore[override]
        if self._last_image is not None:
            return self._last_image.size()
        return QSize(320, 180)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(64, 64)

    def paintEvent(self, event):  # type: ignore[override]
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor(0, 0, 0))
            if self._last_image is not None:
                # Keep aspect ratio while filling available area
                target = self.rect()
                p.drawImage(target, self._last_image)

            if self._warning_active:
                p.setOpacity(0.8)
                p.fillRect(self.rect(), QColor(0, 0, 0, 180))
                p.setOpacity(1.0)
                p.setPen(QColor(255, 200, 0))
                font = QFont("Segoe UI", 16, QFont.Weight.Bold)
                p.setFont(font)
                text = self._warning_text or "CAPTURE IS CURRENT DISPLAY"
                fm = QFontMetrics(font)
                lines = text.split("\n")
                total_h = sum(fm.height() for _ in lines)
                y = (self.height() - total_h) // 2
                for line in lines:
                    w = fm.horizontalAdvance(line)
                    x = (self.width() - w) // 2
                    p.drawText(x, y + fm.ascent(), line)
                    y += fm.height()
        finally:
            p.end()
