
from core.graphics.backend_manager import BackendManager, BackendInfo, BackendPriority
from core.graphics.backends import BackendType
from core.graphics.types import OverlayType
from core.graphics.overlay import Overlay as OverlayBase


class DummyOverlay(OverlayBase):
    def initialize(self) -> bool:  # pragma: no cover - not used
        return True


def make_backend_info(bt: BackendType, supported: bool, priority: BackendPriority = BackendPriority.PREFERRED):
    info = BackendInfo(bt, priority=priority, supported=supported, reason=("unavailable" if not supported else ""))
    # Prevent BackendInfo from trying to resolve real backend_class
    if supported:
        info.backend_class = DummyOverlay
    return info


def test_preferred_backend_unavailable_returns_none():
    mgr = BackendManager()
    # Force states: DWM unavailable, SOFTWARE available
    mgr._backends[BackendType.DWM] = make_backend_info(BackendType.DWM, supported=False)
    mgr._backends[BackendType.SOFTWARE] = make_backend_info(BackendType.SOFTWARE, supported=True)

    selected = mgr.select_backend(preferred=BackendType.DWM, overlay_type=OverlayType.WINDOW)
    assert selected is None, "Explicit preferred backend must fail-fast when unavailable"


def test_auto_selects_highest_priority_available():
    mgr = BackendManager()
    # Make SOFTWARE available with PREFERRED, DWM available with REQUIRED
    mgr._backends[BackendType.SOFTWARE] = make_backend_info(BackendType.SOFTWARE, supported=True, priority=BackendPriority.PREFERRED)
    mgr._backends[BackendType.DWM] = make_backend_info(BackendType.DWM, supported=True, priority=BackendPriority.REQUIRED)

    selected = mgr.select_backend(preferred=BackendType.AUTO, overlay_type=OverlayType.WINDOW)
    # With lower numeric priority (REQUIRED=0) DWM should win
    assert selected is DummyOverlay
