import types
import time
import pytest


class FakeSettings:
    def __init__(self, store=None):
        self._store = store if store is not None else {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value

    def register_change_handler(self, key, cb):
        # Not used in tests
        return None


class FakeMediaController:
    def __init__(self, apps=None):
        # list of (name, hwnd)
        self._apps = apps if apps is not None else [('chrome', 1001)]

    def get_running_media_apps(self):
        return list(self._apps)


@pytest.fixture()
def keepalive_env(monkeypatch):
    # Shared store enabling the feature by default
    store = {
        'features.media_control_enabled': True,
        'media.subtle_activation_enabled': True,
    }

    fake_settings = FakeSettings(store)
    fake_mc = FakeMediaController()

    import core.media.keepalive as ka

    # Patch constructors and dependencies
    monkeypatch.setattr(ka, 'SettingsManager', lambda: fake_settings)
    monkeypatch.setattr(ka, 'get_media_controller', lambda: fake_mc)

    # Patch responsiveness and window validity helpers to be safe, deterministic
    monkeypatch.setattr(ka, 'is_process_responsive', lambda hwnd, timeout_ms=1000: True)

    # Intercept ThreadManager.single_shot to avoid real timers
    calls = []
    def _fake_single_shot(delay_ms, func):
        calls.append((delay_ms, getattr(func, '__name__', str(func))))
        # Do not execute func; just return a token
        return f"timer-{len(calls)}"

    from core import threading as core_threading
    monkeypatch.setattr(core_threading.ThreadManager, 'single_shot', staticmethod(_fake_single_shot))

    return types.SimpleNamespace(store=store, settings=fake_settings, mc=fake_mc, ka=ka, timer_calls=calls)


def test_keepalive_start_schedules_tasks_when_enabled(keepalive_env):
    KA = keepalive_env.ka.MediaPlayerKeepAlive
    ka = KA()
    ka.start()
    # Expect two scheduled timers: monitor cycle and keepalive sweep
    assert len(keepalive_env.timer_calls) >= 2


def test_request_subtle_activation_heuristic_path(keepalive_env, monkeypatch):
    KA = keepalive_env.ka.MediaPlayerKeepAlive
    ka = KA()

    # Force heuristic True and successful activation
    monkeypatch.setattr(ka, '_detect_media_needs_activation', lambda hwnd, name: True)
    monkeypatch.setattr(ka, '_is_media_likely_inactive', lambda hwnd, name: True)
    monkeypatch.setattr(ka, '_perform_subtle_activation', lambda hwnd, name: True)

    res = ka.request_subtle_activation(12345, app_name='chrome')
    assert res is True


def test_hint_media_activity_sets_flags(keepalive_env):
    KA = keepalive_env.ka.MediaPlayerKeepAlive
    ka = KA()

    # Insert a monitored app entry directly
    from core.media.keepalive import AppStatus
    hwnd = 4242
    ka._monitored_apps[hwnd] = AppStatus(
        hwnd=hwnd,
        process_name='chrome',
        pid=999,
        last_check=time.monotonic(),
        responsive=True,
        consecutive_failures=0,
        last_seen=time.monotonic(),
    )

    ka.hint_media_activity(hwnd)
    st = ka._monitored_apps[hwnd]
    assert st.media_activity_detected is True
    assert st.needs_background_keepalive is True
