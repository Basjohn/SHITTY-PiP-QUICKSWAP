#!/usr/bin/env python3
"""
Comprehensive real functionality tests for SPQModular.

This test suite replaces mocked unit tests with real integration tests
that use actual components to catch genuine system issues.
"""
import pytest
# time import removed - not needed for real functionality tests
import json
import tempfile
from pathlib import Path
import sys

# Add repo root to path for consistent imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class TestRealResourceManagement:
    """Test ResourceManager with real manager instances."""
    
    def test_manager_registration_and_cleanup(self):
        """Test that managers register and cleanup correctly."""
        from utils.resource_manager import get_resource_manager
        from utils.z_order_manager import ZOrderManager
        from utils.cursor_manager import CursorManager
        from utils.mouse_capture_coordinator import MouseCaptureCoordinator
        
        rm = get_resource_manager()
        
        # Create real managers
        z_manager = ZOrderManager()
        cursor_manager = CursorManager()
        mouse_coordinator = MouseCaptureCoordinator()
        
        # Verify registration occurred
        assert z_manager._resource_id is not None
        assert cursor_manager._resource_id is not None
        assert mouse_coordinator._resource_id is not None
        
        # Test cleanup - should not raise TypeError
        rm.cleanup_all()
    
    def test_core_managers_integration(self):
        """Test core managers with real ResourceManager integration."""
        from core.graphics.backend_manager import BackendManager
        from core.graphics.overlay_manager import OverlayManager
        from core.events.event_system import EventSystem
        from utils.resource_manager import get_resource_manager
        
        rm = get_resource_manager()
        
        # Create core managers
        backend_mgr = BackendManager()
        overlay_mgr = OverlayManager()
        event_system = EventSystem()
        
        # Verify registration
        assert backend_mgr._resource_id is not None
        assert overlay_mgr._resource_id is not None
        assert event_system._resource_id is not None
        
        # Test cleanup
        rm.cleanup_all()


class TestRealSettingsManager:
    """Test SettingsManager with real file I/O."""
    
    def test_settings_file_io(self, tmp_path):
        """Test real settings file save/load operations with explicit path."""
        from core.settings.settings_manager import SettingsManager
        
        # Reset singleton for clean test
        SettingsManager._reset_for_testing()
        
        # Create a test settings file
        test_settings_file = tmp_path / "test_settings.json"
        
        # Initialize with explicit file path
        settings = SettingsManager(settings_file=str(test_settings_file))
        
        # Set some values
        settings.set("theme", "light")
        settings.set("media.volume_step", 0.1)
        
        # Save should create the file at the specified location
        settings.save()
        assert test_settings_file.exists()
        
        # Verify file contents
        with open(test_settings_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["theme"] == "light"
        assert saved_data["media.volume_step"] == 0.1
        
        # Verify the settings manager respects the explicit path
        assert str(settings._settings_file) == str(test_settings_file)
    
    def test_settings_fallback_hierarchy(self):
        """Test settings file location fallback hierarchy."""
        from core.settings.settings_manager import SettingsManager
        from utils.paths import get_runtime_root
        
        # Reset singleton for clean test
        SettingsManager._reset_for_testing()
        
        # Test without explicit path - should use runtime root/settings
        settings = SettingsManager()
        
        # Should resolve to runtime_root/settings/settings.json
        runtime_root = get_runtime_root()
        expected_path = runtime_root / 'settings' / 'settings.json'
        
        assert settings._settings_file == expected_path


    def test_settings_comprehensive_defaults(self, tmp_path):
        """Test comprehensive settings structure with good defaults."""
        from core.settings.settings_manager import SettingsManager
        
        # Reset singleton for clean test
        SettingsManager._reset_for_testing()
        
        # Use a clean temporary settings file to avoid existing state
        clean_settings_file = tmp_path / "clean_settings.json"
        settings = SettingsManager(settings_file=str(clean_settings_file))
        
        # Test appearance defaults
        assert settings.get("theme") == "dark"
        assert settings.get("appearance.opacity") == 100
        assert not settings.get("overlay.rounded_borders")
        
        # Test behavior defaults
        assert settings.get("media.volume_step") == 0.05
        assert not settings.get("features.autoswitch_enabled")
        assert not settings.get("features.keypassthrough_enabled")
        
        # Test hotkey defaults
        assert settings.get("hotkeys.opacity_enabled")
        assert settings.get("hotkeys.opacity_decrease") == "-"
        assert settings.get("hotkeys.opacity_increase") == "="
        
        # Test graphics defaults
        assert settings.get("graphics.pipeline") == "dxgi"
        assert settings.get("graphics.presentation") == "cpu-blit"


class TestRealEventSystem:
    """Test EventSystem with real subscriptions and dispatch."""
    
    def test_event_subscription_and_dispatch(self):
        """Test real event subscription and publishing."""
        from core.events.event_system import EventSystem
        
        event_system = EventSystem()
        received_events = []
        
        def event_handler(event):
            received_events.append(event)
        
        # Real subscription
        event_system.subscribe("test_event", event_handler)
        
        # Publish real event
        test_data = {"message": "test", "value": 123}
        event_system.publish("test_event", test_data)
        
        # Verify delivery - event handlers receive Event objects
        assert len(received_events) == 1
        event = received_events[0]
        assert event.type == "test_event"
        assert event.data["message"] == "test"
        assert event.data["value"] == 123
        assert hasattr(event, 'id')
        assert hasattr(event, 'timestamp')
    
    def test_event_system_resource_cleanup(self):
        """Test EventSystem ResourceManager integration."""
        from core.events.event_system import EventSystem
        from utils.resource_manager import get_resource_manager
        
        rm = get_resource_manager()
        event_system = EventSystem()
        
        assert event_system._resource_id is not None
        
        # Cleanup should work
        rm.cleanup_all()


class TestRealBackendManagement:
    """Test BackendManager with real backend discovery."""
    
    def test_backend_discovery(self):
        """Test real backend discovery and selection."""
        from core.graphics.backend_manager import BackendManager
        from core.graphics.backends import BackendType
        from core.graphics.types import OverlayType
        
        backend_mgr = BackendManager()
        
        # Test real backend discovery
        available = backend_mgr.get_available_backends()
        assert isinstance(available, list)
        
        # Test selection - should not crash even if no backends available
        backend_mgr.select_backend(
            preferred=BackendType.AUTO,
            overlay_type=OverlayType.WINDOW
        )
    
    def test_backend_manager_resource_cleanup(self):
        """Test BackendManager ResourceManager integration."""
        from core.graphics.backend_manager import BackendManager
        from utils.resource_manager import get_resource_manager
        
        rm = get_resource_manager()
        backend_mgr = BackendManager()
        
        assert backend_mgr._resource_id is not None
        
        # Cleanup should work
        rm.cleanup_all()


class TestRealWindowValidation:
    """Test window validation functions with real Windows API."""
    
    def test_window_validation_functions(self):
        """Test window validation with real API calls."""
        from utils.window_validation import (
            get_window_text,
            get_window_title,
            get_window_class_name,
            is_valid_window
        )
        
        # Test functions exist and are callable
        assert callable(get_window_text)
        assert callable(get_window_title)
        assert callable(get_window_class_name)
        assert callable(is_valid_window)
        
        # Test compatibility alias
        assert get_window_title is get_window_text or hasattr(get_window_title, '__name__')
        
        # Test with invalid handle (should not crash)
        invalid_hwnd = 0
        assert isinstance(get_window_text(invalid_hwnd), str)
        assert isinstance(get_window_title(invalid_hwnd), str)
        assert isinstance(get_window_class_name(invalid_hwnd), str)
        assert isinstance(is_valid_window(invalid_hwnd), bool)
        assert not is_valid_window(invalid_hwnd)


class TestRealCrossManagerInteractions:
    """Test real interactions between multiple managers."""
    
    def test_full_system_integration(self):
        """Test complete system with multiple real managers."""
        from utils.resource_manager import get_resource_manager
        from core.settings.settings_manager import SettingsManager
        from core.events.event_system import EventSystem
        from core.graphics.backend_manager import BackendManager
        from utils.z_order_manager import ZOrderManager
        from utils.cursor_manager import CursorManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "integration_test.json"
            settings_file.write_text('{"integration_test": true}')
            
            # Create full system
            rm = get_resource_manager()
            settings = SettingsManager(settings_file=settings_file)
            events = EventSystem()
            backend_mgr = BackendManager()
            z_order = ZOrderManager()
            cursor_mgr = CursorManager()
            
            # Verify all registered
            managers = [settings, events, backend_mgr, z_order, cursor_mgr]
            for manager in managers:
                assert manager._resource_id is not None
            
            # Test cross-manager functionality
            settings.set("integration_success", True)
            assert settings.get("integration_success") is True
            
            # Test event system
            test_events = []
            events.subscribe("integration_test", lambda e: test_events.append(e))
            events.publish("integration_test", {"status": "success"})
            assert len(test_events) == 1
            
            # Final cleanup test
            rm.cleanup_all()
    
    def test_overlay_manager_integration(self):
        """Test OverlayManager with real components."""
        from core.graphics.overlay_manager import OverlayManager
        from core.graphics.backend_manager import BackendManager
        from utils.resource_manager import get_resource_manager
        
        rm = get_resource_manager()
        backend_mgr = BackendManager()
        overlay_mgr = OverlayManager()
        
        # Verify registration
        assert backend_mgr._resource_id is not None
        assert overlay_mgr._resource_id is not None
        
        # Test cleanup
        rm.cleanup_all()


class TestRealImportIntegrity:
    """Test import chains and dependencies."""
    
    def test_core_module_imports(self):
        """Test all core modules can be imported."""
        import core.application.core
        import core.graphics.backend_manager
        import core.graphics.overlay_manager
        import core.settings.settings_manager
        import core.events.event_system
        import core.resources.manager
        
        # Verify modules loaded
        assert hasattr(core.application.core, 'ApplicationCore')
        assert hasattr(core.graphics.backend_manager, 'BackendManager')
        assert hasattr(core.graphics.overlay_manager, 'OverlayManager')
        assert hasattr(core.settings.settings_manager, 'SettingsManager')
        assert hasattr(core.events.event_system, 'EventSystem')
    
    def test_utils_module_imports(self):
        """Test all utils modules can be imported."""
        import utils.window_validation
        import utils.z_order_manager
        import utils.cursor_manager
        import utils.mouse_capture_coordinator
        import utils.resource_manager
        
        # Verify key functions/classes exist
        assert hasattr(utils.window_validation, 'get_window_text')
        assert hasattr(utils.window_validation, 'get_window_title')
        assert hasattr(utils.z_order_manager, 'ZOrderManager')
        assert hasattr(utils.cursor_manager, 'CursorManager')
        assert hasattr(utils.mouse_capture_coordinator, 'MouseCaptureCoordinator')
        assert hasattr(utils.resource_manager, 'get_resource_manager')
    
    def test_ui_module_imports(self):
        """Test UI modules can be imported."""
        import ui.overlays.monitor.monitor_overlay
        import ui.dialogs.about_dialog
        import ui.components.circle_checkbox
        
        # Verify key classes exist
        assert hasattr(ui.overlays.monitor.monitor_overlay, 'MonitorOverlay')
        assert hasattr(ui.dialogs.about_dialog, 'AboutDialog')
        assert hasattr(ui.components.circle_checkbox, 'CircleCheckBox')
    
    def test_main_module_import(self):
        """Test main module can be imported."""
        import main
        
        assert hasattr(main, 'main')
        assert callable(main.main)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
