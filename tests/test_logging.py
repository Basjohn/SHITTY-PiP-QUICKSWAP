"""
Tests for the core logging functionality.
"""

import logging
import tempfile
import unittest
from pathlib import Path

from core.logging.logger_impl import AppLogger, get_logger


class TestLogging(unittest.TestCase):
    """Test cases for the logging module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test logs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        
        # Initialize logger with test directory
        self.logger = AppLogger(
            name="test_logger",
            log_dir=self.log_dir,
            console_level=logging.DEBUG,
            file_level=logging.DEBUG
        )
        
        # Verify at least one log file was created with the expected pattern
        log_files = list(self.log_dir.glob("app_*.log"))
        self.assertGreater(len(log_files), 0, "No log files were created")
    
    def tearDown(self):
        """Clean up test environment."""
        # Ensure we release file handles before removing the temp directory (Windows)
        try:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.shutdown()
        finally:
            self.temp_dir.cleanup()
    
    def test_log_messages(self):
        """Test logging messages at different levels."""
        # Test different log levels
        self.logger.debug("Debug message", extra={"key": "value"})
        self.logger.info("Info message", extra={"key": "value"})
        self.logger.warning("Warning message", extra={"key": "value"})
        self.logger.error("Error message", extra={"key": "value"})
        self.logger.critical("Critical message", extra={"key": "value"})
        
        # Test exception logging
        try:
            1 / 0
        except ZeroDivisionError:
            self.logger.exception("Exception occurred")
        
        # Check that log files were created
        log_files = list(self.log_dir.glob("*.log"))
        self.assertGreater(len(log_files), 0, "No log files were created")
    
    def test_get_logger(self):
        """Test the get_logger convenience function."""
        logger = get_logger("test_module")
        logger.info("Test message from get_logger")
        
        # Check that the logger has the expected name
        self.assertEqual(logger.name, "test_module")
        
        # Check that it has at least one handler
        self.assertGreater(len(logger.handlers), 0, "Logger has no handlers")


if __name__ == "__main__":
    unittest.main()
