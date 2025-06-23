"""
CaptureWorker - A QThread-based worker for high-performance screen capture.

This module provides a worker thread that handles screen capture using MSS,
frame comparison, and adaptive FPS control to optimize performance.
"""

import time
import numpy as np
import mss
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, QRect

class CaptureWorker(QThread):
    """
    A worker thread that handles screen capture and frame comparison.
    
    This worker runs in a separate thread to perform screen captures using MSS,
    compare frames to detect changes, and emit signals when new frames are available.
    It also implements adaptive FPS control to optimize performance.
    """
    
    # Signal emitted when a new frame is ready
    # Parameters: frame_data (memoryview), width (int), height (int), dpr (float)
    frame_ready = Signal(object, int, int, float)
    
    # Signal emitted when the FPS changes
    fps_changed = Signal(float)
    
    def __init__(self, mss_instance, parent=None):
        """
        Initialize the CaptureWorker.
        
        Args:
            mss_instance: An instance of mss.mss() for screen capture
            parent: Parent QObject
        """
        super().__init__(parent)
        self.mss_instance = mss_instance
        self._mutex = QMutex()
        self._running = True
        self._capture_params = {
            'monitor_idx': 1,  # Default to first monitor (0 is virtual screen)
            'width': 1920,     # Default width
            'height': 1080,    # Default height
        }
        self._target_fps = 30.0
        self._min_fps = 15.0
        self._max_fps = 60.0
        self._current_fps = 30.0
        self._last_frame_time = 0
        self._last_frame = None
        self._frame_count = 0
        self._last_fps_update = 0
        
    def run(self):
        """
        Main capture loop.
        
        This method runs in a separate thread and continuously captures
        screenshots, compares them to the previous frame, and emits
        signals when a new frame is available.
        """
        last_frame_time = time.monotonic()
        frame_count = 0
        last_fps_update = last_frame_time
        
        while self._running:
            try:
                frame_start = time.monotonic()
                
                # Capture frame
                frame_data, width, height, dpr = self._capture_frame()
                if frame_data is None:
                    time.sleep(1.0 / self._target_fps)
                    continue
                
                # Check if frame has changed
                if self._has_frame_changed(frame_data):
                    self._last_frame = frame_data
                    self.frame_ready.emit(frame_data, width, height, dpr)
                
                # Calculate actual FPS
                frame_count += 1
                now = time.monotonic()
                elapsed = now - last_frame_time
                
                # Update FPS every second
                if now - last_fps_update >= 1.0:
                    self._current_fps = frame_count / (now - last_fps_update)
                    self.fps_changed.emit(self._current_fps)
                    frame_count = 0
                    last_fps_update = now
                    
                    # Adjust target FPS based on performance
                    self._adjust_fps(elapsed)
                
                # Calculate sleep time to maintain target FPS
                frame_time = time.monotonic() - frame_start
                target_frame_time = 1.0 / self._target_fps
                sleep_time = max(0, target_frame_time - frame_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_frame_time = time.monotonic()
                
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def stop(self):
        """Stop the capture thread."""
        with QMutexLocker(self._mutex):
            self._running = False
    
    def update_capture_params(self, monitor_idx, width, height):
        """
        Update capture parameters in a thread-safe manner.
        
        Args:
            monitor_idx: Index of the monitor to capture
            width: Width of the capture area
            height: Height of the capture area
        """
        with QMutexLocker(self._mutex):
            self._capture_params = {
                'monitor_idx': monitor_idx,
                'width': width,
                'height': height
            }
    
    def _capture_frame(self):
        """
        Capture a single frame using MSS.
        
        Returns:
            tuple: (frame_data, width, height, dpr) or (None, 0, 0, 1.0) on error
        """
        try:
            with QMutexLocker(self._mutex):
                monitor_idx = self._capture_params['monitor_idx']
                
            # Get the monitor info
            if monitor_idx >= len(self.mss_instance.monitors):
                return None, 0, 0, 1.0
                
            monitor = self.mss_instance.monitors[monitor_idx]
            
            # Capture the screen
            screenshot = self.mss_instance.grab(monitor)
            
            # Convert to numpy array (BGRA format)
            img_array = np.frombuffer(screenshot.bgra, dtype=np.uint8)
            img_array = img_array.reshape((screenshot.height, screenshot.width, 4))
            
            # Convert BGRA to RGB and ensure C-contiguous array
            rgb_array = np.ascontiguousarray(img_array[:, :, [2, 1, 0]])  # BGR to RGB
            
            return rgb_array, screenshot.width, screenshot.height, 1.0
            
        except Exception as e:
            print(f"Capture error: {e}")
            return None, 0, 0, 1.0
    
    def _has_frame_changed(self, new_frame):
        """
        Check if the frame has changed significantly from the last frame.
        
        Args:
            new_frame: New frame data (numpy array)
            
        Returns:
            bool: True if the frame has changed significantly
        """
        if self._last_frame is None:
            return True
            
        if new_frame.shape != self._last_frame.shape:
            return True
            
        # Simple pixel difference comparison
        # For better performance, we can use a more sophisticated method
        diff = np.abs(new_frame.astype(np.int16) - self._last_frame.astype(np.int16))
        mean_diff = np.mean(diff)
        
        # Consider frame changed if mean difference is above threshold
        return mean_diff > 5.0  # Adjust threshold as needed
    
    def _adjust_fps(self, frame_time):
        """
        Adjust target FPS based on performance.
        
        Args:
            frame_time: Time taken to process the last frame
        """
        target_frame_time = 1.0 / self._target_fps
        
        # If we're consistently faster than target, try increasing FPS
        if frame_time < target_frame_time * 0.8 and self._target_fps < self._max_fps:
            self._target_fps = min(self._max_fps, self._target_fps + 5.0)
        # If we're consistently slower than target, decrease FPS
        elif frame_time > target_frame_time * 1.2 and self._target_fps > self._min_fps:
            self._target_fps = max(self._min_fps, self._target_fps - 5.0)
        
        # Ensure FPS is within bounds
        self._target_fps = max(self._min_fps, min(self._max_fps, self._target_fps))
