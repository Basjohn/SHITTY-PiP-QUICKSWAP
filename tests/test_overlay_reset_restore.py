"""
Tests for overlay reset and restore functionality.

These tests verify that:
1. Reset Overlay preserves the current DWM source window
2. Restore Hidden Overlays creates overlays with valid application windows (not system tray)
"""

import pytest
from unittest.mock import Mock, patch
from PySide6.QtCore import QPoint, QSize

from core.graphics.overlay_manager import OverlayManager
from core.graphics.types import OverlayType
from core.switching.mru_manager import MRUManager
from utils.overlay_context_menu import OverlayContextMenu


class TestOverlayReset:
    """Test overlay reset functionality."""
    
    @pytest.fixture
    def mock_overlay_manager(self):
        """Mock overlay manager."""
        om = Mock(spec=OverlayManager)
        om.get_overlay.return_value = Mock()
        om.destroy_overlay.return_value = True
        om.create_overlay.return_value = "new-overlay-id"
        return om
    
    @pytest.fixture
    def mock_overlay(self):
        """Mock overlay with DWM source."""
        overlay = Mock()
        overlay.get_config.return_value = Mock(
            overlay_type=OverlayType.WINDOW,
            position=QPoint(100, 100),
            size=QSize(640, 360),
            opacity=1.0,
            title="Test Overlay",
            properties={'hwnd': 12345}
        )
        overlay.get_source_hwnd.return_value = 12345  # Current DWM source
        return overlay
    
    @patch('core.threading.ThreadManager')
    def test_reset_preserves_current_dwm_source(self, mock_thread_manager, mock_overlay_manager, mock_overlay):
        """Test that reset overlay preserves the current DWM source window."""
        # Setup
        mock_thread_manager.single_shot = Mock()
        
        # Create context menu instance with mocked overlay manager
        context_menu = OverlayContextMenu(Mock(), Mock())
        context_menu.overlay = Mock()
        context_menu.overlay.identifier = "test-overlay-id"
        context_menu.overlay.id = "test-overlay-id"  # Also set id attribute
        
        # Mock the OverlayManager import and instance
        with patch('core.graphics.overlay_manager.OverlayManager') as mock_om_class:
            mock_om_class.return_value = mock_overlay_manager
            mock_overlay_manager.get_overlay.return_value = mock_overlay
            
            # Execute reset
            context_menu._handle_recreate_overlay()
        
            # Verify overlay was queried for current source
            mock_overlay_manager.get_overlay.assert_called_once_with("test-overlay-id")
            mock_overlay.get_source_hwnd.assert_called_once()
            
            # Verify destroy was called
            mock_overlay_manager.destroy_overlay.assert_called_once_with("test-overlay-id")
            
            # Verify delayed recreation was scheduled
            mock_thread_manager.single_shot.assert_called_once()
            args = mock_thread_manager.single_shot.call_args
            assert args[0][0] == 100  # 100ms delay
            
            # Execute the delayed recreation callback
            delayed_callback = args[0][1]
            delayed_callback()
            
            # Verify new overlay was created with preserved hwnd
            mock_overlay_manager.create_overlay.assert_called_once()
            create_args = mock_overlay_manager.create_overlay.call_args
            assert 'properties' in create_args[1]
            assert create_args[1]['properties']['hwnd'] == 12345


class TestOverlayRestore:
    """Test overlay restore functionality."""
    
    @pytest.fixture
    def mock_mru_manager(self):
        """Mock MRU manager with valid windows."""
        mru = Mock(spec=MRUManager)
        # Return a valid application window, not system tray
        mru.get_most_recent.return_value = 67340  # Firefox window from logs
        return mru
    
    @pytest.fixture
    def mock_overlay_manager(self):
        """Mock overlay manager."""
        om = Mock(spec=OverlayManager)
        om.create_overlay.return_value = "restored-overlay-id"
        return om
    
    @patch('utils.window_validation.is_valid_window')
    def test_restore_creates_overlay_with_valid_window(self, mock_is_valid, mock_mru_manager, mock_overlay_manager):
        """Test that restore creates overlay with a valid application window."""
        # Setup
        mock_is_valid.return_value = True
        
        # Mock the imports and managers
        with patch('core.switching.mru_manager.get_mru_manager') as mock_get_mru, \
             patch('core.graphics.overlay_manager.OverlayManager') as mock_om_class:
            mock_get_mru.return_value = mock_mru_manager
            mock_om_class.return_value = mock_overlay_manager
            
            # Test the method directly without creating SystemTrayManager
            from core.ui.tray import SystemTrayManager
            logger = Mock()
            
            # Create a mock tray instance and call the method directly
            tray_instance = Mock(spec=SystemTrayManager)
            SystemTrayManager._recreate_most_recent_overlay(tray_instance, logger)
        
            # Verify MRU was queried
            mock_mru_manager.get_most_recent.assert_called_once()
            
            # Verify overlay was created with the MRU window
            mock_overlay_manager.create_overlay.assert_called_once()
            create_args = mock_overlay_manager.create_overlay.call_args
            assert 'properties' in create_args[1]
            assert create_args[1]['properties']['hwnd'] == 67340
            
            # Verify success was logged
            logger.info.assert_called_with("Recreated overlay for most recent window: 67340")
    
    def test_restore_handles_no_recent_windows(self):
        """Test that restore handles case with no recent windows gracefully."""
        # Setup - MRU returns None
        mock_mru = Mock(spec=MRUManager)
        mock_mru.get_most_recent.return_value = None
        
        # Mock the get_mru_manager import
        with patch('core.switching.mru_manager.get_mru_manager') as mock_get_mru:
            mock_get_mru.return_value = mock_mru
            
            # Test the method directly without creating SystemTrayManager
            from core.ui.tray import SystemTrayManager
            logger = Mock()
            
            # Create a mock tray instance and call the method directly
            tray_instance = Mock(spec=SystemTrayManager)
            SystemTrayManager._recreate_most_recent_overlay(tray_instance, logger)
            
            # Verify it logged no recent windows and returned early
            logger.debug.assert_called_with("No recent windows in MRU")
    
    @patch('utils.window_validation.is_system_window')
    def test_system_tray_window_filtered_out(self, mock_is_system):
        """Test that system tray overflow windows are filtered out."""
        # System tray overflow window should be filtered
        mock_is_system.return_value = True
        
        from utils.window_validation import is_valid_window
        
        # Test with system tray overflow window ID from logs
        result = is_valid_window(2165032, our_pid=12345)
        
        # Should be filtered out
        assert result is False


class TestMRUManager:
    """Test MRU manager functionality."""
    
    def test_get_most_recent_returns_valid_window(self):
        """Test that get_most_recent returns a valid window."""
        # Setup
        with patch('core.switching.mru_manager.is_valid_window') as mock_is_valid:
            mock_is_valid.return_value = True
            
            mru = MRUManager()
            mru._mru = [67340, 12345, 54321]  # Simulate MRU list
            
            # Execute
            result = mru.get_most_recent()
            
            # Verify - should return first valid window
            assert result == 67340
            # Should have called is_valid_window for the first window only (since it's valid)
            assert mock_is_valid.call_count >= 1
            mock_is_valid.assert_any_call(67340, our_pid=mru._pid)
    
    def test_get_most_recent_filters_invalid_windows(self):
        """Test that get_most_recent filters out invalid windows."""
        # Setup - first window invalid, second valid
        with patch('core.switching.mru_manager.is_valid_window') as mock_is_valid:
            mock_is_valid.side_effect = [False, True]
            
            mru = MRUManager()
            mru._mru = [2165032, 67340]  # System tray first, Firefox second
            
            # Execute
            result = mru.get_most_recent()
            
            # Verify it skipped invalid window and returned valid one
            assert result == 67340
            assert mock_is_valid.call_count == 2
    
    def test_get_most_recent_returns_none_when_empty(self):
        """Test that get_most_recent returns None when MRU is empty."""
        mru = MRUManager()
        mru._mru = []
        
        result = mru.get_most_recent()
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
