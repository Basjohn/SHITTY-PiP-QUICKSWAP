"""
Application instance management.

This module provides functionality for managing a single application instance
using a system-wide mutex to prevent multiple instances of the application
from running simultaneously.
"""

import time
from typing import Optional

import win32api
import win32con
import win32event
import win32gui
import winerror

# Local imports
from core.logging import get_logger
from core.threading.manager import ThreadManager
from core.threading.priority import TaskPriority
from core.threading import get_thread_manager

logger = get_logger(__name__)

class ApplicationInstanceManager:
    """Manages application instance using a system mutex with ThreadManager integration."""
    
    def __init__(self, app_name: str = "ShittyPiP", thread_manager: Optional[ThreadManager] = None):
        """Initialize the instance manager.
        
        Args:
            app_name: The name of the application, used to create a unique mutex name.
            thread_manager: Optional ThreadManager instance. If None, a new one will be created.
        """
        self.app_name = app_name
        self.mutex_name = f"Global\\{app_name.replace(' ', '_')}_SingleInstance_Mutex"
        self.mutex_handle = None
        self.is_primary = False
        # Use centralized singleton to avoid accidental multiple manager instances
        self.thread_manager = thread_manager or get_thread_manager()
    
    def is_another_instance_running(self) -> bool:
        """Check if another instance of the application is already running.
        
        This method uses a system-wide mutex to ensure thread-safe instance checking.
        
        Returns:
            bool: True if another instance is running, False otherwise.
        """
        try:
            # Try to create a named mutex
            self.mutex_handle = win32event.CreateMutex(
                None,  # Default security attributes
                False,  # Initially not owned
                self.mutex_name
            )
            
            # Check if the mutex already exists
            last_error = win32api.GetLastError()
            if last_error == winerror.ERROR_ALREADY_EXISTS:
                logger.debug("Another instance is already running")
                if self.mutex_handle:
                    win32api.CloseHandle(self.mutex_handle)
                    self.mutex_handle = None
                return True
                
            self.is_primary = True
            logger.debug("This is the primary application instance")
            return False
            
        except Exception as e:
            logger.error(f"Error checking for existing instance: {e}")
            # If we can't check, assume another instance is running to be safe
            return True
    
    def close_existing_instance(self) -> bool:
        """Attempt to close an existing instance of the application.
        
        This method sends a close message to the existing instance's main window
        and waits for it to close. Uses ThreadManager for thread safety.
        
        Returns:
            bool: True if the existing instance was closed successfully, False otherwise.
        """
        def find_and_close_window() -> bool:
            try:
                # Find the main window of the existing instance
                hwnd = win32gui.FindWindow("Qt5QWindowIcon", None)
                while hwnd:
                    window_title = win32gui.GetWindowText(hwnd)
                    if "PiP Overlay" in window_title:
                        logger.debug(f"Found existing instance window: {window_title} (HWND: {hwnd})")
                        # Send close message
                        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                        
                        # Wait for the window to close with timeout
                        start_time = time.time()
                        timeout = 5  # 5 second timeout
                        
                        while win32gui.IsWindow(hwnd) and (time.time() - start_time) < timeout:
                            win32gui.PumpWaitingMessages()
                            time.sleep(0.1)
                        
                        if win32gui.IsWindow(hwnd):
                            logger.warning("Existing instance did not close within timeout")
                            return False
                            
                        logger.debug("Existing instance closed successfully")
                        return True
                    hwnd = win32gui.GetWindow(hwnd, win32con.GW_HWNDNEXT)
                
                logger.debug("No existing instance window found")
                return False
                
            except Exception as e:
                logger.error(f"Error in find_and_close_window: {e}")
                return False
        
        try:
            # Submit the task to the thread manager with a descriptive name
            future = self.thread_manager.submit(
                find_and_close_window,
                task_id="close_existing_instance",
                priority=TaskPriority.HIGH
            )
            
            # Wait for the task to complete with a timeout
            return future.result(timeout=10)  # 10 second overall timeout
            
        except Exception as e:
            logger.error(f"Error in close_existing_instance: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by the instance manager."""
        if self.mutex_handle and self.is_primary:
            try:
                win32api.CloseHandle(self.mutex_handle)
                self.mutex_handle = None
                self.is_primary = False
                logger.debug("Released instance mutex")
            except Exception as e:
                logger.error(f"Error releasing instance mutex: {e}")
    
    def __del__(self):
        """Ensure resources are cleaned up when the manager is garbage collected."""
        self.cleanup()
