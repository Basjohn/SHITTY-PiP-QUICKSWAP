import os
import pytest

from core.graphics.capture.monitor_capture_manager import MonitorCaptureManager


@pytest.fixture(autouse=True)
def clear_backend_env(monkeypatch):
    # Ensure env doesn't force a specific backend unless a test sets it
    monkeypatch.delenv("SPQ_CAPTURE_BACKEND", raising=False)


def test_probe_auto_prefers_dxgi_when_available(monkeypatch):
    # Simulate both backends available; auto should pick dxgi
    import core.graphics.capture.monitor_capture_manager as mod
    monkeypatch.setattr(mod, "DXCAM_AVAILABLE", True, raising=False)

    m = MonitorCaptureManager()
    res = m.probe_backend_availability("auto")
    assert res["requested"] == "auto"
    assert res["effective"] == "dxgi"
    assert "dxcam" in res["reason"].lower()


def test_stats_include_backend_fields():
    m = MonitorCaptureManager()
    stats = m.get_capture_stats()
    # Ensure diagnostic keys exist, values may vary before start
    assert "backend_requested" in stats
    assert "backend_effective" in stats


def test_probe_runtime_environment_matches_dxcam_presence():
    """Smoke-test the real environment: if dxcam is importable, probe should pick dxgi when requested.

    This does not start capture; it only validates selection logic against actual module presence.
    """
    import importlib.util
    m = MonitorCaptureManager()
    res = m.probe_backend_availability("dxgi")
    dxcam_spec = importlib.util.find_spec("dxcam")
    if dxcam_spec is not None:
        assert res["effective"] == "dxgi"
    else:
        # No backend available if dxcam is missing
        assert res["effective"] is None
