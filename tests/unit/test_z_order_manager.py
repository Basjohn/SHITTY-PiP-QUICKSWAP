"""
Unit tests for canonical utils.z_order_manager.ZOrderManager
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from PySide6.QtWidgets import QWidget, QApplication
import sys

import utils.z_order_manager as zomod
from utils.z_order_manager import ZOrderManager, ZOrderPriority


class TestZOrderManager(unittest.TestCase):
    """Tests for canonical ZOrderManager behavior and APIs."""

    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.zm = ZOrderManager()
        self.overlay_id = "test_overlay"

        # Mock widgets
        self.main_widget = Mock(spec=QWidget)
        self.border_widget = Mock(spec=QWidget)

        # Mock window handles
        self.main_hwnd = 12345
        self.border_hwnd = 67890

        # Configure mock widgets
        self.main_widget.winId.return_value = self.main_hwnd
        self.border_widget.winId.return_value = self.border_hwnd

    def _patch_win32(self, is_window=True, setpos_return=True, last_error=0):
        """Patch utils.z_order_manager Win32 symbols for deterministic tests."""
        patches = [
            patch.object(zomod, 'IS_WINDOWS', True, create=True),
            patch.object(zomod, 'IsWindow', MagicMock(return_value=is_window), create=True),
            patch.object(zomod, 'SetWindowPos', MagicMock(return_value=setpos_return), create=True),
            patch.object(zomod, 'GetLastError', MagicMock(return_value=last_error), create=True),
            patch.object(zomod, 'HWND_TOPMOST', 1, create=True),
            patch.object(zomod, 'HWND_TOP', 2, create=True),
            patch.object(zomod, 'SWP_NOSIZE', 0x1, create=True),
            patch.object(zomod, 'SWP_NOMOVE', 0x2, create=True),
            patch.object(zomod, 'SWP_NOACTIVATE', 0x10, create=True),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        return zomod.SetWindowPos
    
    def test_register_overlay_basic(self):
        ok = self.zm.register_overlay(self.overlay_id, self.main_widget)
        self.assertTrue(ok)
        self.assertIn(self.overlay_id, self.zm._overlays)
        self.assertEqual(self.zm.get_overlay_count(), 1)
    
    def test_register_overlay_with_border(self):
        ok = self.zm.register_overlay(self.overlay_id, self.main_widget, border_widget=self.border_widget)
        self.assertTrue(ok)
        info = self.zm._overlays[self.overlay_id]
        self.assertIsNotNone(info.border_widget)
        self.assertIs(info.border_widget(), self.border_widget)
    
    def test_unregister_overlay(self):
        self.zm.register_overlay(self.overlay_id, self.main_widget)
        ok = self.zm.unregister_overlay(self.overlay_id)
        self.assertTrue(ok)
        self.assertNotIn(self.overlay_id, self.zm._overlays)
        self.assertNotIn(self.overlay_id, self.zm._pending_enforcements)
    
    def test_enforce_z_order_immediate_success(self):
        self.zm.register_overlay(self.overlay_id, self.main_widget)
        setpos = self._patch_win32(is_window=True, setpos_return=True)
        result = self.zm._enforce_z_order_immediate(self.overlay_id, ZOrderPriority.NORMAL)
        self.assertTrue(result)
        self.assertEqual(setpos.call_count, 1)
    
    def test_enforce_with_border_calls_twice(self):
        self.zm.register_overlay(self.overlay_id, self.main_widget, border_widget=self.border_widget)
        setpos = self._patch_win32(is_window=True, setpos_return=True)
        result = self.zm._enforce_z_order_immediate(self.overlay_id, ZOrderPriority.NORMAL)
        self.assertTrue(result)
        self.assertEqual(setpos.call_count, 2)
    
    def test_enforce_z_order_invalid_handle(self):
        self.zm.register_overlay(self.overlay_id, self.main_widget)
        setpos = self._patch_win32(is_window=False)
        result = self.zm._enforce_z_order_immediate(self.overlay_id, ZOrderPriority.NORMAL)
        self.assertFalse(result)
        setpos.assert_not_called()
    
    def test_enforce_z_order_destroyed_handle(self):
        self.zm.register_overlay(self.overlay_id, self.main_widget)
        self._patch_win32(is_window=True, setpos_return=False, last_error=1400)
        result = self.zm._enforce_z_order_immediate(self.overlay_id, ZOrderPriority.NORMAL)
        self.assertFalse(result)
        # Overlay should be unregistered on destroyed handle
        self.assertNotIn(self.overlay_id, self.zm._overlays)
    
    def test_context_menu_priority_uses_hwnd_top(self):
        self.zm.register_overlay(self.overlay_id, self.main_widget)
        setpos = self._patch_win32(is_window=True, setpos_return=True)
        result = self.zm._enforce_z_order_immediate(self.overlay_id, ZOrderPriority.CONTEXT_MENU)
        self.assertTrue(result)
        # Inspect hWndInsertAfter argument
        call = setpos.mock_calls[0]
        args = call[1]
        self.assertEqual(args[1], zomod.HWND_TOP)
    
    def test_debounced_enforcement_single_shot(self):
        self.zm.register_overlay(self.overlay_id, self.main_widget)
        setpos = self._patch_win32(is_window=True, setpos_return=True)
        # Make single_shot invoke callback immediately
        with patch('core.threading.manager.ThreadManager.single_shot', side_effect=lambda ms, cb: cb()):
            ok = self.zm.enforce_z_order(self.overlay_id)
            self.assertTrue(ok)
        self.assertEqual(setpos.call_count, 1)
    
    # Legacy internal destroyed-handles tracking is not part of canonical manager; tests removed.


if __name__ == '__main__':
    unittest.main()
