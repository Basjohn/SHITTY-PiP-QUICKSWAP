import pytest

# Import HotkeyManager and ThreadManager symbol to monkeypatch single_shot
from core.hotkeys.manager import HotkeyManager
import core.threading as threading_core


class FakeKeyboard:
    def __init__(self):
        self._press_cb = None
        self._release_cb = None
        # Sequence of physical state reads for watchdog: True -> False
        self._pressed_sequence = [True, False]
        self._target_key = "space"

    def on_press_key(self, key, callback, suppress=True):
        self._press_cb = callback
        return ("press", key)

    def on_release_key(self, key, callback, suppress=True):
        self._release_cb = callback
        return ("release", key)

    def is_pressed(self, key):
        # Modifiers should report not pressed in this test
        if key in ("ctrl", "shift"):
            return False
        # Only the target key follows the programmed sequence
        if key == self._target_key:
            if self._pressed_sequence:
                return self._pressed_sequence.pop(0)
            return False
        return False

    def unhook(self, handle):
        # no-op for tests
        pass


class DummyEvent:
    def __init__(self):
        self.suppress = False


@pytest.fixture
def monkeypatch_single_shot(monkeypatch):
    # Replace ThreadManager.single_shot to call immediately (no Qt dependency)
    def _immediate_single_shot(delay_ms, func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr(threading_core.ThreadManager, "single_shot", staticmethod(_immediate_single_shot))
    return _immediate_single_shot


def test_watchdog_resets_gate_without_release(monkeypatch, monkeypatch_single_shot):
    hm = HotkeyManager()

    # Force keyboard backend to our fake
    fake_kb = FakeKeyboard()
    hm._keyboard_lib = fake_kb  # type: ignore[attr-defined]
    hm._kb_available = True

    triggered = []

    def cb():
        triggered.append(True)

    # Register a single key (e.g., space) with suppression to route to keyboard backend
    ok = hm.register_hotkey("hk_space", cb, sequence="space", suppress=True, global_hotkey=True)
    assert ok, "registration should succeed"

    # Simulate press event (no subsequent release)
    assert fake_kb._press_cb is not None, "press callback should be set"
    # Ensure no modifiers are reported as pressed regardless of environment
    fake_kb.is_pressed = lambda key: (key == fake_kb._target_key and (fake_kb._pressed_sequence.pop(0) if fake_kb._pressed_sequence else False))
    fake_kb._press_cb(DummyEvent())

    # After press, callback fired once. With immediate single_shot, watchdog may already have reset the gate.
    assert len(triggered) == 1

    # Watchdog runs via immediate single_shot, consuming True then False and resets gate
    # Since our single_shot is immediate, by this point gate should be reset
    assert hm._kb_press_state.get("hk_space", True) is False, "watchdog should reset gate when physical key goes up"

    # Subsequent press should trigger again (gate reopened)
    fake_kb._pressed_sequence = [True, False]
    fake_kb.is_pressed = lambda key: (key == fake_kb._target_key and (fake_kb._pressed_sequence.pop(0) if fake_kb._pressed_sequence else False))
    fake_kb._press_cb(DummyEvent())
    assert len(triggered) == 2

    # Cleanup
    hm.unregister_hotkey("hk_space")
