"""
Media-related utilities for the application.

This module provides utilities for managing media players and related functionality.
"""

from core.logging import get_logger
import time
from typing import Optional

import win32api
import win32gui
from PySide6.QtCore import QObject
from core.threading import ThreadManager

logger = get_logger("MEDIA")

class MediaPlayerKeepAlive(QObject):
    """
    A class to keep media players active by periodically sending fake input events.
    This is a best-effort implementation and will fail gracefully if anything goes wrong.
    """
    
    def __init__(self, hwnd: int, parent: Optional[QObject] = None):
        """
        Initialize the MediaPlayerKeepAlive.
        
        Args:
            hwnd: The window handle of the media player to keep alive
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.hwnd = hwnd
        self.is_active = False
        self.last_keepalive_time = 0
        self.keepalive_interval = 30  # seconds
        self.window_class = ""
        self.window_title = ""
        self._update_window_info()

    def _schedule_tick(self) -> None:
        """Schedule the next keep-alive check via ThreadManager."""
        try:
            ThreadManager.single_shot(1000, self._tick_dispatch)
        except Exception:
            # If scheduling fails, try again later when start() is called next
            pass

    def _tick_dispatch(self) -> None:
        """Dispatch a tick if active, then reschedule."""
        if not self.is_active:
            return
        try:
            self._keep_alive_tick()
        finally:
            # Re-schedule the next tick only if still active
            if self.is_active:
                self._schedule_tick()
    
    def _update_window_info(self) -> bool:
        """
        Update window class and title information.
        
        Returns:
            bool: True if window info was updated successfully, False otherwise
        """
        try:
            if win32gui.IsWindow(self.hwnd):
                self.window_class = win32gui.GetClassName(self.hwnd)
                self.window_title = win32gui.GetWindowText(self.hwnd)
                return True
        except Exception as e:
            logger.debug(f"Error updating window info: {e}")
        return False
    
    def _keep_alive_tick(self):
        """Send a fake input event to keep the media player active."""
        try:
            current_time = time.time()
            
            # Only send keep-alive if it's been long enough since the last one
            if current_time - self.last_keepalive_time >= self.keepalive_interval:
                if not self._update_window_info() or not win32gui.IsWindow(self.hwnd):
                    logger.debug("Window no longer exists, stopping keep-alive")
                    self.stop()
                    return
                
                # Get window rect for calculating center position
                try:
                    left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
                    center_x = left + (right - left) // 2
                    center_y = top + (bottom - top) // 2
                    
                    # Convert to screen coordinates
                    point = win32gui.ClientToScreen(self.hwnd, (center_x, center_y))
                    
                    # Send a mouse move event to the center of the window
                    win32api.SetCursorPos(point)
                    
                    self.last_keepalive_time = current_time
                    logger.debug(f"Sent keep-alive to window {self.hwnd} ({self.window_title})")
                    
                except Exception as e:
                    logger.debug(f"Error sending keep-alive to window {self.hwnd}: {e}")
                    self.stop()
        except Exception as e:
            logger.error(f"Unexpected error in keep-alive tick: {e}")
            self.stop()
    
    def start(self) -> bool:
        """
        Start the keep-alive timer.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.is_active and win32gui.IsWindow(self.hwnd):
            self.is_active = True
            self.last_keepalive_time = time.time()
            self._schedule_tick()
            logger.debug(f"Started keep-alive for window {self.hwnd} (checking every {self.keepalive_interval} seconds)")
            return True
        return False
    
    def stop(self):
        """Stop the keep-alive timer."""
        if self.is_active:
            self.is_active = False
            logger.debug(f"Stopped keep-alive for window {self.hwnd}")
