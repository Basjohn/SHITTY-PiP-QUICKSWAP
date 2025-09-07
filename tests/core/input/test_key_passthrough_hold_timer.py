import os
import pytest

# Ensure offscreen to avoid GUI requirement if Qt widgets get involved indirectly
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# NOTE: These tests isolate KeyPassthroughController's volume hold timing logic by
# monkeypatching ThreadManager.single_shot to record scheduled delays and by
# providing fakes for focus state, window validity, and media controller.

@pytest.fixture()
def controller(monkeypatch):
    # Import here to ensure environment is set
    from core.input.key_passthrough_controller import KeyPassthroughController

    # Reset singleton between tests
    KeyPassthroughController._instance = None  # type: ignore[attr-defined]
    ctrl = KeyPassthroughController()

    # Force enabled and media routing enabled
    ctrl._enabled = True
    ctrl._media_routing_enabled = True

    # Provide deterministic hold timings for test
    ctrl._hold_initial_delay_ms = 123
    ctrl._hold_interval_ms = 45

    # Fake focus state: overlay NOT focused so volume routing/holds are active
    class _FS:
        def is_overlay_focused(self):
            return False
    monkeypatch.setattr("core.input.key_passthrough_controller.get_focus_state", lambda: _FS())

    # Fake window validity and set target hwnd
    monkeypatch.setattr("core.input.key_passthrough_controller._is_window", lambda h: True)
    ctrl._target_hwnd = 0x123456

    # Fake media controller
    class _MC:
        def __init__(self):
            self.up_calls = 0
            self.down_calls = 0
            self.mute_calls = 0
        def volume_up_for_hwnd(self, h):
            self.up_calls += 1
            return True, "ok"
        def volume_down_for_hwnd(self, h):
            self.down_calls += 1
            return True, "ok"
        def _send_command_for_hwnd(self, h, cmd):
            self.mute_calls += 1
            return True, "ok"
    mc = _MC()
    monkeypatch.setattr("core.input.key_passthrough_controller.get_media_controller", lambda: mc)

    # Stub ThreadManager.single_shot to capture scheduled calls and invoke under test control
    scheduled = []  # list of tuples (delay, callback)
    def _single_shot(delay, cb):
        scheduled.append((delay, cb))
    monkeypatch.setattr("core.input.key_passthrough_controller.ThreadManager.single_shot", _single_shot)

    # Attach helpers for the test to access
    ctrl._test_scheduled = scheduled  # type: ignore[attr-defined]
    ctrl._test_media = mc            # type: ignore[attr-defined]

    return ctrl


def _run_next(ctrl):
    # Pop and run the next scheduled callback (FIFO)
    scheduled = ctrl._test_scheduled  # type: ignore[attr-defined]
    assert scheduled, "Expected a scheduled callback"
    delay, cb = scheduled.pop(0)
    cb()
    return delay


def test_volume_up_hold_timer_schedules_and_stops_on_release(controller):
    from core.input.key_passthrough_controller import _VK_VOLUME_UP

    ctrl = controller

    # Press should trigger an immediate step via passthrough_key (media path) and schedule first hold tick
    pre_calls = ctrl._test_media.up_calls  # type: ignore[attr-defined]
    assert ctrl.press_passthrough_key(_VK_VOLUME_UP) is True

    # First schedule: initial delay
    scheduled = ctrl._test_scheduled  # type: ignore[attr-defined]
    assert len(scheduled) == 1
    assert scheduled[0][0] == 123  # initial delay ms

    # Run first tick; should perform another volume up and schedule next at interval
    d0 = _run_next(ctrl)
    assert d0 == 123
    assert ctrl._test_media.up_calls >= pre_calls + 1  # tick step occurred
    assert len(ctrl._test_scheduled) == 1
    assert ctrl._test_scheduled[0][0] == 45

    # Release should invalidate further repeats
    ctrl.release_passthrough_key(_VK_VOLUME_UP)

    # Run the interval callback that was already scheduled; since token invalidated, it should not reschedule
    d1 = _run_next(ctrl)
    assert d1 == 45
    assert len(ctrl._test_scheduled) == 0  # no further scheduling


def test_volume_down_hold_timer_schedules_and_stops_on_release(controller):
    from core.input.key_passthrough_controller import _VK_VOLUME_DOWN

    ctrl = controller

    pre_calls = ctrl._test_media.down_calls  # type: ignore[attr-defined]
    assert ctrl.press_passthrough_key(_VK_VOLUME_DOWN) is True

    # First schedule: initial delay
    scheduled = ctrl._test_scheduled  # type: ignore[attr-defined]
    assert len(scheduled) == 1
    assert scheduled[0][0] == 123

    # Run first tick; should perform another volume down and schedule next at interval
    d0 = _run_next(ctrl)
    assert d0 == 123
    assert ctrl._test_media.down_calls >= pre_calls + 1
    assert len(ctrl._test_scheduled) == 1
    assert ctrl._test_scheduled[0][0] == 45

    # Release should invalidate further repeats
    ctrl.release_passthrough_key(_VK_VOLUME_DOWN)

    # Run the interval callback that was already scheduled; since token invalidated, it should not reschedule
    d1 = _run_next(ctrl)
    assert d1 == 45
    assert len(ctrl._test_scheduled) == 0
