import os
import sys
import pytest

# Ensure offscreen to avoid GUI requirement
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QRectF
    from ui.overlays.integrated_border_canvas import IntegratedBorderCanvas
    from ui.overlays.geometry.border_geometry import BorderMetrics
except Exception as e:  # pragma: no cover - environment import guard
    pytest.skip(f"PySide6/Qt environment not available: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv[:1])
    return app


class _DummyTheme:
    def get_accent_thickness(self) -> float:
        return 0.0

    def get_accent_inset(self) -> float:
        return 0.0


def _set_fixed_metrics(widget: IntegratedBorderCanvas, w: int, h: int, thickness: float = 4.0, radius: float = 0.0) -> None:
    # Disable rounded and accents influence
    widget._rounded_enabled = False
    widget._border_theme = _DummyTheme()  # type: ignore[attr-defined]
    # Install deterministic metrics to avoid theme/dpi variability
    widget._border_metrics = BorderMetrics(
        thickness=thickness,
        corner_radius=radius,
        inner_accent_thickness=0.0,
        render_rect=QRectF(0, 0, w, h),
        accent_inset=0.0,
    )


def test_letterbox_floor_fit_no_overshoot(qapp):
    w, h = 800, 600
    widget = IntegratedBorderCanvas()
    widget.setFixedSize(w, h)
    _set_fixed_metrics(widget, w, h, thickness=4.0, radius=0.0)

    # Target wider AR than outer -> letterbox path (outer_ratio < target_ratio)
    widget.set_content_aspect(16, 9)
    rect = widget._calc_content_rect()

    # Outer after thickness inset: 792x592 at (4,4)
    assert rect.width() == 792
    # floor(792 / (16/9)) = floor(445.5) = 445
    assert rect.height() == 445
    assert rect.x() == 4
    # y = 4 + (592 - 445)//2 = 4 + 73 = 77
    assert rect.y() == 77

    widget.deleteLater()


def test_pillarbox_floor_fit_no_overshoot(qapp):
    w, h = 800, 600
    widget = IntegratedBorderCanvas()
    widget.setFixedSize(w, h)
    _set_fixed_metrics(widget, w, h, thickness=4.0, radius=0.0)

    # Target squarer AR than outer -> pillarbox path (outer_ratio > target_ratio)
    widget.set_content_aspect(1, 1)
    rect = widget._calc_content_rect()

    # Outer after thickness inset: 792x592 at (4,4)
    # floor(592 * 1.0) = 592
    assert rect.width() == 592
    assert rect.height() == 592
    # x = 4 + (792 - 592)//2
    assert rect.x() == 104
    assert rect.y() == 4

    widget.deleteLater()
