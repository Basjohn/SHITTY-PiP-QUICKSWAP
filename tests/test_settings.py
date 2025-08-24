"""
Tests for the core settings module.
"""

import tempfile
import unittest
from pathlib import Path

from core.settings.settings_manager import SettingsManager
from core.settings.types import SettingsCategory


class TestSettingsManager(unittest.TestCase):
    """Test cases for the SettingsManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test settings
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_file = Path(self.temp_dir.name) / 'settings.json'
        
        # Initialize settings manager with test file
        self.settings = SettingsManager()
        self.settings._impl._settings_file = self.settings_file  # Override settings file path
        # Ensure a consistent baseline for each test (singleton instance persists across tests)
        self.settings.reset_to_defaults()
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_get_set_value(self):
        """Test getting and setting values."""
        # Test with a known setting
        self.assertEqual(self.settings.get('theme'), 'dark')
        
        # Test setting a new value
        self.settings.set('theme', 'light')
        self.assertEqual(self.settings.get('theme'), 'light')
        
        # Test with default value
        self.assertEqual(self.settings.get('nonexistent.setting', 'default'), 'default')
    
    def test_validation(self):
        """Test setting validation."""
        # Test valid value
        self.settings.set('appearance.opacity', 50)
        self.assertEqual(self.settings.get('appearance.opacity'), 50)
        
        # Test invalid value
        with self.assertRaises(ValueError):
            self.settings.set('appearance.opacity', 150)  # Out of range
    
    def test_save_load(self):
        """Test saving and loading settings to/from disk."""
        # Change a setting
        self.settings.set('theme', 'light')
        self.settings.save()
        
        # Create a new settings instance that should load the saved settings
        new_settings = SettingsManager()
        new_settings._impl._settings_file = self.settings_file
        new_settings.load()
        
        self.assertEqual(new_settings.get('theme'), 'light')
    
    def test_reset_to_defaults(self):
        """Test resetting settings to default values."""
        # Change a setting
        self.settings.set('theme', 'light')
        self.assertEqual(self.settings.get('theme'), 'light')
        
        # Reset to defaults
        self.settings.reset_to_defaults()
        self.assertEqual(self.settings.get('theme'), 'dark')
    
    def test_change_handlers(self):
        """Test setting change handlers."""
        changes = []
        
        def handler(key: str, value: str) -> None:
            changes.append((key, value))
        
        # Register handler
        self.settings.register_change_handler('theme', handler)
        
        # Change the setting
        self.settings.set('theme', 'light')
        
        # Check that handler was called
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0], ('theme', 'light'))
        
        # Unregister handler
        self.settings.unregister_change_handler('theme', handler)
        
        # Change again - handler should not be called
        self.settings.set('theme', 'dark')
        self.assertEqual(len(changes), 1)  # Still only one change recorded
    
    def test_get_settings_by_category(self):
        """Test getting settings by category."""
        # Get all appearance settings
        appearance_settings = self.settings.get_settings_by_category(SettingsCategory.APPEARANCE)
        
        # Should include theme and opacity
        self.assertIn('theme', appearance_settings)
        self.assertIn('appearance.opacity', appearance_settings)
        
        # Should not include settings from other categories
        self.assertNotIn('behavior.auto_switch', appearance_settings)


if __name__ == '__main__':
    unittest.main()
