import os
import sys
import pytest

# Ensure offscreen to avoid GUI requirement
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPoint, QSize
    from core.graphics.overlay_host import OverlayHost
    from core.graphics.types import OverlayConfig
    from ui.components.volume_osd import VolumeOSDWidget
    from core.application.core import get_app_core
    from PySide6.QtTest import QTest
except Exception as e:  # pragma: no cover - environment import guard
    pytest.skip(f"PySide6/Qt environment not available: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv[:1])
    return app


def _expected_osd_pos(host: OverlayHost, osd: VolumeOSDWidget) -> QPoint:
    rect = host.rect()
    w, h = osd.width(), osd.height()
    x_local = rect.x() + (rect.width() - w) // 2
    y_local = rect.y() + rect.height() - h - 12
    base = QPoint(max(0, x_local), max(0, y_local))
    # If OSD is a top-level window, its position is in global coordinates
    if osd.parentWidget() is host:
        return base
    else:
        top_left = host.mapToGlobal(QPoint(0, 0))
        return top_left + base


def test_volume_osd_integration_instantiates_and_positions(qapp):
    config = OverlayConfig()
    host = OverlayHost(config)
    host.resize(640, 400)
    host.show()

    # Allow event loop to settle
    QTest.qWait(50)

    # Find the OSD by objectName and type. It may be a top-level window.
    osd = host.findChild(VolumeOSDWidget, "volumeOSD")
    if osd is None:
        # Search top-level widgets for the OSD window
        for w in QApplication.topLevelWidgets():
            if isinstance(w, VolumeOSDWidget) and getattr(w, "objectName", lambda: "")() == "volumeOSD":
                osd = w
                break
    assert osd is not None, "VolumeOSDWidget should exist (child or top-level window)"

    # On creation it should be hidden; position should still be set deterministically
    osd.update_position()
    exp = _expected_osd_pos(host, osd)
    assert osd.pos() == exp

    # Publish a media.volume.changed event and ensure the OSD becomes visible
    app_core = get_app_core()
    evt_data = {"app_name": "pytest", "volume": 0.42, "source": "test"}
    app_core.events.publish("media.volume.changed", evt_data, source="testcase")

    # Give UI time to coalesce and show
    QTest.qWait(120)
    assert osd.isVisible() is True

    # Resize host and ensure OSD repositions
    host.resize(800, 500)
    QTest.qWait(50)
    osd.update_position()  # also triggered via geometryChanged, but ensure deterministic
    exp2 = _expected_osd_pos(host, osd)
    assert osd.pos() == exp2

    host.hide()
    host.deleteLater()
