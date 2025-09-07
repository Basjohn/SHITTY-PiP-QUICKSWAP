import types
import pytest


class FakeSettings:
    def __init__(self, store=None):
        self._store = store if store is not None else {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value

    def register_change_handler(self, key, cb):
        return None


class FakeKeyboard:
    def __init__(self):
        self.add_calls = []  # list of dicts: {seq, suppress}
        self.remove_calls = []  # list of handles
        self._handles = {}
        self._next_id = 1

    def add_hotkey(self, sequence, callback, suppress=False):  # signature subset
        handle = f"h{self._next_id}"
        self._next_id += 1
        self._handles[handle] = (sequence, callback, suppress)
        self.add_calls.append({"sequence": sequence, "suppress": suppress})
        return handle

    def remove_hotkey(self, handle):
        self.remove_calls.append(handle)
        self._handles.pop(handle, None)


class FakeRM:
    def __init__(self):
        self._next = 1
        self._regs = {}  # rid -> (resource, cleanup_handler, description)
        self.register_calls = []
        self.unregister_calls = []

    def register(self, resource, *, resource_type=None, description="", cleanup_handler=None, cleanup_priority=None, **metadata):
        rid = f"r{self._next}"
        self._next += 1
        self._regs[rid] = (resource, cleanup_handler, description)
        self.register_calls.append({"rid": rid, "description": description})
        return rid

    def unregister(self, rid):
        self.unregister_calls.append(rid)
        tup = self._regs.pop(rid, None)
        if tup is not None:
            resource, cleanup_handler, _ = tup
            if callable(cleanup_handler):
                try:
                    cleanup_handler(resource)
                except Exception:
                    pass
        return True


@pytest.fixture()
def quickswitch_env(monkeypatch):
    store = {
        'hotkeys.quickswitch_enabled': False,
    }
    fake_settings = FakeSettings(store)
    fake_kb = FakeKeyboard()
    fake_rm = FakeRM()

    import core.switching.quickswitch_controller as qsc
    # Patch constructors/globals used inside QuickSwitchController
    monkeypatch.setattr(qsc, 'SettingsManager', lambda: fake_settings)
    monkeypatch.setattr(qsc, 'keyboard', fake_kb)
    monkeypatch.setattr(qsc, 'get_resource_manager', lambda: fake_rm)

    # Reset singleton between tests
    try:
        if hasattr(qsc.QuickSwitchController, '_instance'):
            qsc.QuickSwitchController._instance = None
    except Exception:
        pass

    return types.SimpleNamespace(store=store, settings=fake_settings, kb=fake_kb, rm=fake_rm, qsc=qsc)


def test_quickswitch_does_not_register_when_disabled(quickswitch_env):
    QS = quickswitch_env.qsc.QuickSwitchController
    _ = QS()
    assert quickswitch_env.kb.add_calls == []
    assert quickswitch_env.rm.register_calls == []


def test_quickswitch_registers_default_combo_when_enabled(quickswitch_env):
    quickswitch_env.store['hotkeys.quickswitch_enabled'] = True
    QS = quickswitch_env.qsc.QuickSwitchController
    _ = QS()

    assert len(quickswitch_env.kb.add_calls) == 1
    call = quickswitch_env.kb.add_calls[0]
    assert call["sequence"] == "shift+x"
    assert call["suppress"] is False
    # Resource registered
    assert len(quickswitch_env.rm.register_calls) == 1
    assert "quickswitch" in quickswitch_env.rm.register_calls[0]["description"].lower()


def test_quickswitch_uses_user_configured_combo(quickswitch_env):
    quickswitch_env.store['hotkeys.quickswitch_enabled'] = True
    quickswitch_env.store['hotkeys.opacity_quickswitch'] = 'ctrl+shift+q'

    QS = quickswitch_env.qsc.QuickSwitchController
    _ = QS()

    assert len(quickswitch_env.kb.add_calls) == 1
    assert quickswitch_env.kb.add_calls[0]["sequence"] == 'ctrl+shift+q'


def test_quickswitch_disable_triggers_resource_cleanup(quickswitch_env):
    quickswitch_env.store['hotkeys.quickswitch_enabled'] = True
    QS = quickswitch_env.qsc.QuickSwitchController
    ctrl = QS()
    assert len(quickswitch_env.rm.register_calls) == 1

    # Disable and update → should unregister resource and remove keyboard hotkey
    quickswitch_env.store['hotkeys.quickswitch_enabled'] = False
    ctrl.update_hotkeys()

    assert len(quickswitch_env.rm.unregister_calls) >= 1
    # Best-effort direct remove may also be called; allow either path, but at least one unregister executed
