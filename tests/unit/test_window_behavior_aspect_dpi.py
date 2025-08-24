import math
from typing import Optional

import pytest
from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtWidgets import QWidget

from utils.window.behavior import WindowBehaviorManager


class _WheelEventStub:
    """Minimal stub for QWheelEvent-like interface used by handle_wheel."""

    def __init__(self, delta: int) -> None:
        # delta is in Qt wheel steps units (multiples of 120)
        self._delta = int(delta)

    def angleDelta(self) -> QPoint:  # noqa: N802 (Qt naming)
        return QPoint(0, self._delta)


@pytest.mark.qt_no_exception_capture
class TestWindowBehaviorAspectDPI:
    def _apply_and_get_inner_size(
        self,
        manager: WindowBehaviorManager,
        insets: Optional[tuple[int, int]],
    ) -> tuple[int, int]:
        # Force apply the pending geometry that handle_wheel schedules
        manager._apply_pending_wheel_geo()  # intentionally calling private method for deterministic test
        geo: QRect = manager._widget.geometry()
        iw = geo.width()
        ih = geo.height()
        if insets and len(insets) == 2:
            ix, iy = insets
            iw = max(1, iw - 2 * int(ix))
            ih = max(1, ih - 2 * int(iy))
        return iw, ih

    def test_wheel_resize_preserves_aspect_with_insets(self, qtbot):
        # Given: a widget with known starting geometry and 16:9 content AR with DPI insets
        w = QWidget()
        qtbot.addWidget(w)
        w.setGeometry(QRect(100, 100, 320, 240))  # outer 320x240

        manager = WindowBehaviorManager(w)
        aspect = (16, 9)
        insets = (8, 8)  # simulate DPI-aware insets (e.g., borders/margins)

        # When: apply a single wheel step to grow
        manager.handle_wheel(_WheelEventStub(120), content_aspect=aspect, content_insets=insets)
        inner_w, inner_h = self._apply_and_get_inner_size(manager, insets)

        # Then: inner area should be approximately 16:9 within 1px tolerance
        if inner_h > 0:
            ar = inner_w / inner_h
            assert math.isclose(ar, 16 / 9, rel_tol=0, abs_tol=1 / max(1, inner_h))

    def test_wheel_resize_obeys_min_size_with_ar(self, qtbot):
        # Given: a widget with small min size and AR 4:3
        w = QWidget()
        qtbot.addWidget(w)
        w.setGeometry(QRect(200, 150, 160, 120))  # start small

        # enforce a larger minimum so shrinking tries to go below and gets clamped
        manager = WindowBehaviorManager(w, min_width=200, min_height=150)
        aspect = (4, 3)
        insets = (6, 4)

        # When: apply a negative wheel step to shrink
        manager.handle_wheel(_WheelEventStub(-120), content_aspect=aspect, content_insets=insets)
        inner_w, inner_h = self._apply_and_get_inner_size(manager, insets)

        # Then: outer size is at least min, and inner maintains AR within 1px tolerance
        outer = w.geometry().size()
        assert outer.width() >= 200
        assert outer.height() >= 150

        if inner_h > 0:
            ar = inner_w / inner_h
            assert math.isclose(ar, 4 / 3, rel_tol=0, abs_tol=1 / max(1, inner_h))
