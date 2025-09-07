"""
Smoke tests for application startup and basic functionality.

These tests verify that the application can start without critical errors
and that basic systems are functional.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add repo root to path for consistent imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class TestApplicationStartup:
    """Smoke tests for application startup."""
    
    def test_main_module_can_be_imported(self):
        """Test that main.py can be imported without errors."""
        try:
            import main
            assert hasattr(main, 'PiPApplication')
            assert hasattr(main, 'main')
        except ImportError as e:
            pytest.fail(f"Failed to import main module: {e}")
    
    @patch('PySide6.QtWidgets.QApplication')
    @patch('main.get_resource_manager')
    def test_pip_application_can_be_instantiated(self, mock_rm, mock_qapp):
        """Test that PiPApplication can be created without errors."""
        # Mock QApplication instance
        mock_app_instance = MagicMock()
        mock_qapp.return_value = mock_app_instance
        mock_qapp.instance.return_value = mock_app_instance
        
        # Mock ResourceManager
        mock_resource_manager = MagicMock()
        mock_rm.return_value = mock_resource_manager
        
        try:
            from main import PiPApplication
            
            # Mock sys.argv to avoid Qt argument parsing issues
            with patch('sys.argv', ['test']):
                app = PiPApplication()
                assert app is not None
                
        except Exception as e:
            pytest.fail(f"Failed to instantiate PiPApplication: {e}")
    
    def test_core_services_can_be_imported(self):
        """Test that all core services can be imported."""
        core_services = [
            'core.application.core',
            'core.settings.settings_manager', 
            'core.events.event_system',
            'core.resources.manager',
            'core.graphics.backend_manager',
            'core.graphics.overlay_manager',
        ]
        
        for service in core_services:
            try:
                __import__(service)
            except ImportError as e:
                pytest.fail(f"Failed to import core service {service}: {e}")
    
    def test_utility_managers_can_be_imported(self):
        """Test that utility managers can be imported."""
        utility_managers = [
            'utils.z_order_manager',
            'utils.cursor_manager', 
            'utils.mouse_capture_coordinator',
            'utils.resource_manager',
        ]
        
        for manager in utility_managers:
            try:
                __import__(manager)
            except ImportError as e:
                pytest.fail(f"Failed to import utility manager {manager}: {e}")
    
    def test_window_validation_functions_available(self):
        """Test that window validation functions are available."""
        try:
            from utils.window_validation import (
                get_window_text,
                get_window_title,
                get_window_class_name,
                is_valid_window
            )
            
            # Verify functions are callable
            assert callable(get_window_text)
            assert callable(get_window_title) 
            assert callable(get_window_class_name)
            assert callable(is_valid_window)
            
        except ImportError as e:
            pytest.fail(f"Failed to import window validation functions: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
