"""
Unit tests for the ThemeManager implementation.
"""
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from utils.theme.theme_manager import ThemeManager, ThemeVariant, ThemeColorRole
from utils.theme.validation import validate_theme
from utils.theme.consumer import ThemeConsumer


class MockThemeConsumer(ThemeConsumer):
    """Mock implementation of ThemeConsumer for testing."""
    
    def __init__(self):
        self.theme_applied = None
        self.required_tokens = ["base", "text", "overlay.border.stroke"]
    
    def apply_theme(self, theme_name: str) -> None:
        self.theme_applied = theme_name
    
    def get_required_tokens(self):
        return self.required_tokens


class TestThemeManager(unittest.TestCase):
    """Test case for ThemeManager."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        # Create a test directory structure
        project_root = Path(__file__).resolve().parents[3]
        cls.test_dir = project_root / "tests" / "temp" / "themes"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a test theme file
        test_theme = {
            "name": "test_theme",
            "base": "#222222",
            "text": "#eeeeee",
            "border": "#444444",
            "highlight": "#0078d7",
            "highlight_text": "#ffffff",
            "overlay.border.stroke": "#ffffff",
            "overlay.border.thickness.base": "1.5",
            "overlay.border.accent": "#888888",
            "overlay.border.rounded.enabled": "true"
        }
        
        import json
        test_theme_path = cls.test_dir / "test_theme.json"
        with open(test_theme_path, "w") as f:
            json.dump(test_theme, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        import shutil
        # Clean up test directory
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir.parent)
    
    def setUp(self):
        """Set up test environment."""
        # Mock dependencies
        self.mock_app = MagicMock()
        
        # Patch settings_manager
        self.mock_settings = MagicMock()
        self.mock_settings.get.return_value = "dark"
        
        # Create instance with mocks
        with patch("utils.theme.theme_manager.settings_manager", self.mock_settings):
            # Reset singleton for each test
            ThemeManager._instance = None
            self.theme_manager = ThemeManager(self.mock_app)
    
    def test_instance_singleton(self):
        """Test singleton pattern implementation."""
        # Establish the singleton via the public API
        first = ThemeManager.instance(self.mock_app)
        # Subsequent calls should return the same object
        second = ThemeManager.instance()
        self.assertIs(first, second)
    
    def test_default_themes(self):
        """Test default themes are initialized."""
        # Verify required default themes are accessible via public API
        self.assertIsNotNone(self.theme_manager.get_theme_colors("dark"))
        self.assertIsNotNone(self.theme_manager.get_theme_colors("light"))
    
    def test_get_token(self):
        """Test token retrieval."""
        # Get token from default theme
        token = self.theme_manager.get_token("base")
        self.assertIsNotNone(token)
        
        # Get token with explicit theme
        token = self.theme_manager.get_token("text", "light")
        self.assertIsNotNone(token)
        
        # Nonexistent token should raise ValueError (no fallback)
        with self.assertRaises(ValueError):
            self.theme_manager.get_token("nonexistent_token")
    
    def test_theme_validation(self):
        """Test theme validation."""
        # Mock consumer that requires certain tokens
        consumer = MockThemeConsumer()
        
        # Register the consumer
        with patch("utils.theme.validation._REQUIRED_TOKENS", {"test": set(consumer.get_required_tokens())}):
            # Validate with required tokens present
            self.assertTrue(validate_theme("dark", "test"))
            
            # Create invalid theme
            from utils.theme.theme_manager import ThemeColors
            dark_colors = dict(self.theme_manager.get_theme_colors("dark").colors)
            # Remove a required token to force invalidation
            dark_colors.pop("overlay.border.stroke", None)
            # Inject as ThemeColors into the manager's theme registry
            self.theme_manager._theme_colors["invalid"] = ThemeColors(dark_colors)
            
            # Validation should fail for invalid theme
            self.assertFalse(validate_theme("invalid", "test"))
    
    def test_theme_switching(self):
        """Test theme switching functionality."""
        # Apply theme
        self.theme_manager.apply_theme("light")
        self.assertEqual(self.theme_manager._current_theme, "light")
        
        # Apply theme with enum
        self.theme_manager.apply_theme(ThemeVariant.DARK)
        self.assertEqual(self.theme_manager._current_theme, "dark")
        
        # Invalid theme should raise ValueError
        with self.assertRaises(ValueError):
            self.theme_manager.apply_theme("nonexistent_theme")
    
    def test_color_tokens(self):
        """Test color token access."""
        # Get standard color
        color = self.theme_manager.get_token(ThemeColorRole.BASE)
        self.assertTrue(color.startswith("#"))
        
        # Get overlay color
        color = self.theme_manager.get_token("overlay.border.stroke")
        self.assertTrue(color.startswith("#"))


if __name__ == "__main__":
    unittest.main()
