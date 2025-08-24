import json
import unittest
import tempfile
from pathlib import Path

# Attempt to import the SettingsManager; if PySide6 is unavailable, skip tests
HAS_QT = True
try:
    from core.settings.settings_manager import SettingsManager
except Exception:
    HAS_QT = False


def reset_settings_singleton():
    if not HAS_QT:
        return
    # Reset the singleton state to allow fresh instances per test
    SettingsManager._instance = None  # type: ignore[attr-defined]
    SettingsManager._initialized = False  # type: ignore[attr-defined]


@unittest.skipUnless(HAS_QT, "PySide6/SettingsManager not available")
class TestSettingsManager(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.settings_path = Path(self.td.name) / "settings.json"
        reset_settings_singleton()

    def _new_manager(self, initial_json=None):
        if initial_json is not None:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(initial_json, f)
        reset_settings_singleton()
        return SettingsManager(settings_file=self.settings_path)

    def test_theme_mirroring_set_theme(self):
        mgr = self._new_manager()
        calls = []
        mgr.register_change_handler("theme", lambda k, v: calls.append((k, v)))

        changed = mgr.set("theme", "light", save_immediately=True)
        self.assertTrue(changed)
        self.assertEqual(mgr.get("theme"), "light")
        # Ensure only canonical key notified
        notified_keys = {k for (k, _) in calls}
        self.assertSetEqual(notified_keys, {"theme"})


    def test_invalid_option_rejected_on_set(self):
        mgr = self._new_manager()
        with self.assertRaises(ValueError):
            mgr.set("theme", "system", save_immediately=False)

    def test_migration_maps_system_to_dark(self):
        # Persist legacy values
        mgr = self._new_manager({"theme": "system"})
        # After init + migration + validation
        self.assertEqual(mgr.get("theme"), "dark")

    def test_invalid_type_raises_on_load(self):
        # Wrong type for performance.cache_size (expects int)
        initial = {"performance.cache_size": "not-an-int"}
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(initial, f)
        # Expect ValueError during initialization (validation after load)
        with self.assertRaises(ValueError):
            self._new_manager()

    def test_persistence_roundtrip(self):
        mgr = self._new_manager()
        mgr.set("performance.cache_size", 200, save_immediately=True)
        self.assertTrue(self.settings_path.exists())
        # Reinitialize and ensure value persisted
        mgr2 = self._new_manager()
        self.assertEqual(mgr2.get("performance.cache_size"), 200)


if __name__ == "__main__":
    unittest.main()
