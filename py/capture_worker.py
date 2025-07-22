"""
CaptureWorker - A QThread-based worker for high-performance screen capture.

This module provides a worker thread that handles screen capture using MSS,
frame comparison, and adaptive FPS control to optimize performance.
"""

import time
import numpy as np
import mss
import psutil
import os
import gc
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, QRect, QTimer

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
        Initialize the CaptureWorker with memory management optimizations.
        
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
        
        # Reusable buffers to avoid memory allocations
        self._frame_buffer = None
        self._gray_buffer = None
        self._diff_buffer = None
        self._last_frame = None
        self._last_downsampled = None
        
        # Frame comparison settings
        self._frame_change_threshold = 5.0  # Initial threshold for frame change
        self._min_change_threshold = 3.0    # Minimum threshold
        self._max_change_threshold = 20.0   # Maximum threshold
        self._change_sensitivity = 0.9      # How quickly to adapt the threshold (0-1)
        
        # Performance tracking
        self._target_fps = 30.0
        self._min_fps = 15.0
        self._max_fps = 60.0
        self._current_fps = 30.0
        self._last_frame_time = 0
        self._frame_count = 0
        self._last_fps_update = 0
        self._last_process_time = 0
        
        # Memory management
        self._last_mem_check = 0
        self._mem_check_interval = 5.0  # Check memory every 5 seconds
        self._max_memory_mb = 200       # Max memory in MB before cleanup
        
        # Statistics
        self._stats_window = 10  # Number of frames to track for statistics
        self._frame_times = []
        self._frame_diffs = []
        self._last_stats_update = 0
        
        # Setup memory monitoring timer
        self._mem_timer = QTimer()
        self._mem_timer.timeout.connect(self._check_memory_usage)
        self._mem_timer.start(5000)  # Check every 5 seconds
        
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
        """
        Stop the capture thread and clean up resources.
        """
        with QMutexLocker(self._mutex):
            self._running = False
            
        # Stop memory monitoring timer
        if hasattr(self, '_mem_timer') and self._mem_timer.isActive():
            self._mem_timer.stop()
            
        # Explicitly release large objects
        self._release_resources()
        
        # Force garbage collection
        gc.collect()
    
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
    
    def _check_memory_usage(self):
        """Check memory usage and trigger cleanup if needed."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
        
        if mem_mb > self._max_memory_mb:
            logger.warning(f"High memory usage: {mem_mb:.1f}MB, performing cleanup...")
            self._release_resources()
            gc.collect()
            
            # Log memory after cleanup
            mem_after = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Memory after cleanup: {mem_after:.1f}MB")
    
    def _release_resources(self):
        """Release large objects and reset buffers."""
        self._frame_buffer = None
        self._gray_buffer = None
        self._diff_buffer = None
        self._last_frame = None
        self._last_downsampled = None
        
        # Force garbage collection
        gc.collect()
    
    def _capture_frame(self):
        """
        Capture a single frame using MSS with memory optimizations.
        
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
            
            # Initialize or reuse frame buffer
            if (self._frame_buffer is None or 
                self._frame_buffer.shape != (screenshot.height, screenshot.width, 3)):
                self._frame_buffer = np.empty((screenshot.height, screenshot.width, 3), dtype=np.uint8)
            
            # Convert BGRA to RGB directly into the buffer
            # This avoids creating temporary arrays
            bgra_data = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
                screenshot.height, screenshot.width, 4)
            
            # Copy RGB channels (excluding alpha) - this is faster than array indexing
            # and uses less memory than creating intermediate arrays
            self._frame_buffer[..., 0] = bgra_data[..., 2]  # R
            self._frame_buffer[..., 1] = bgra_data[..., 1]  # G
            self._frame_buffer[..., 2] = bgra_data[..., 0]  # B
            
            return self._frame_buffer, screenshot.width, screenshot.height, 1.0
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None, 0, 0, 1.0
    
    def _downsample_frame(self, frame, scale=0.25):
        """
        Downsample the frame to a smaller size for faster comparison.
        
        Args:
            frame: Input frame (numpy array)
            scale: Scaling factor (0-1)
            
        Returns:
            Downsampled frame
        """
        if scale >= 1.0:
            return frame
            
        height, width = frame.shape[:2]
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        
        # Use array slicing for downsampling (faster than resize for this use case)
        step_x = width // new_width
        step_y = height // new_height
        
        return frame[::step_y, ::step_x].copy()

    def _capture_frame(self):
        """
        Capture a single frame using MSS with memory optimizations.
        
        Returns:
            tuple: (frame_data, width, height, dpr) or (None, 0, 0, 1.0) on error
        """
        try:
            # Get the monitor to capture
            monitor = self.mss_instance.monitors[self._capture_params['monitor_idx']]
            
            # Adjust monitor dimensions if needed
            if (self._capture_params['width'] > 0 and 
                self._capture_params['height'] > 0):
                monitor.update({
                    'width': self._capture_params['width'],
                    'height': self._capture_params['height']
                })
            
            # Capture the screen
            screenshot = self.mss_instance.grab(monitor)
            
            # Initialize or reuse frame buffer
            if (self._frame_buffer is None or 
                self._frame_buffer.shape != (screenshot.height, screenshot.width, 3)):
                self._frame_buffer = np.empty((screenshot.height, screenshot.width, 3), dtype=np.uint8)
            
            # Convert BGRA to RGB directly into the buffer
            # This avoids creating temporary arrays
            bgra_data = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
                screenshot.height, screenshot.width, 4)
            
            # Copy RGB channels (excluding alpha) - this is faster than array indexing
            # and uses less memory than creating intermediate arrays
            self._frame_buffer[..., 0] = bgra_data[..., 2]  # R
            self._frame_buffer[..., 1] = bgra_data[..., 1]  # G
            self._frame_buffer[..., 2] = bgra_data[..., 0]  # B
            
            return self._frame_buffer, screenshot.width, screenshot.height, 1.0
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None, 0, 0, 1.0

def _downsample_frame(self, frame, scale=0.25):
    """
    Downsample the frame to a smaller size for faster comparison.
    
    Args:
        frame: Input frame (numpy array)
        scale: Scaling factor (0-1)
        
    Returns:
        Downsampled frame
    """
    if scale >= 1.0:
        return frame
        
    height, width = frame.shape[:2]
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    
    # Use array slicing for downsampling (faster than resize for this use case)
    step_x = width // new_width
    step_y = height // new_height
    return frame[::step_y, ::step_x]

def _calculate_frame_difference(self, frame1, frame2):
    """
    Calculate the difference between two frames using a perceptual metric.
    Optimized for memory efficiency by reusing buffers.
    
    Args:
        frame1: First frame (numpy array)
        frame2: Second frame (numpy array)
        
    Returns:
        float: Difference metric (0-255)
    """
    if frame1 is None or frame2 is None:
        return float('inf')
        
    if frame1.shape != frame2.shape:
        return float('inf')
    
    # Initialize or resize buffers if needed
    height, width = frame1.shape[0], frame1.shape[1]
    if (self._gray_buffer is None or 
        self._gray_buffer.shape != (height, width)):
        self._gray_buffer = np.empty((height, width), dtype=np.uint8)
        self._diff_buffer = np.empty((height, width), dtype=np.float32)
    
    # Convert to grayscale if needed (much faster for comparison)
    if len(frame1.shape) == 3:
        # Convert RGB to grayscale using luminance formula
        # Using pre-allocated buffer to avoid temporary arrays
        np.dot(frame1[..., :3], [0.2989, 0.5870, 0.1140], out=self._gray_buffer)
        gray1 = self._gray_buffer.astype(np.uint8, copy=False)
        
        # Reuse buffer for second frame
        np.dot(frame2[..., :3], [0.2989, 0.5870, 0.1140], out=self._gray_buffer)
        gray2 = self._gray_buffer.astype(np.uint8, copy=False)
    else:
        gray1 = frame1
        gray2 = frame2
    
    # Calculate absolute difference using pre-allocated buffer
    np.abs(gray1.astype(np.int16, copy=False) - 
           gray2.astype(np.int16, copy=False), 
           out=self._diff_buffer)
    
    # Use a weighted metric that's more sensitive to changes in bright areas
    # This helps with detecting UI changes which are often bright
    # Reuse the diff buffer for weights calculation
    np.power(gray1.astype(np.float32) / 255.0, 0.5, out=self._diff_buffer)
    weighted_diff = self._diff_buffer * (1.0 + self._diff_buffer)
    
    return np.mean(weighted_diff)

def _has_frame_changed(self, new_frame):
    """
    Check if the frame has changed significantly from the last frame.
    This optimized version uses downsampling and adaptive thresholding
    to improve performance and accuracy, with memory optimizations.
    
    Args:
        new_frame: New frame data (numpy array)
        
    Returns:
        bool: True if the frame has changed significantly
    """
    start_time = time.monotonic()
    
    # First frame is always considered changed
    if self._last_frame is None:
        # Store a reference to avoid copying if possible
        self._last_frame = new_frame
        self._last_downsampled = self._downsample_frame(new_frame)
        return True
        
    # Check if dimensions match
    if new_frame.shape != self._last_frame.shape:
        # Update stored frame
        self._last_frame = new_frame
        self._last_downsampled = self._downsample_frame(new_frame)
        return True
    
    # Downsample the new frame for faster comparison
    downsampled = self._downsample_frame(new_frame)
    
    try:
        # Calculate frame difference using the downsampled frames
        diff = self._calculate_frame_difference(self._last_downsampled, downsampled)
        
        # Update frame statistics
        current_time = time.monotonic()
        self._frame_diffs.append(diff)
        self._frame_times.append(current_time - start_time)
        
        # Keep only the last N frames in the statistics
        if len(self._frame_diffs) > self._stats_window:
            self._frame_diffs.pop(0)
            self._frame_times.pop(0)
        
        # Update threshold based on recent frame differences
        if len(self._frame_diffs) >= 3:  # Need at least 3 samples
            avg_diff = np.mean(self._frame_diffs)
            # Adjust threshold based on recent activity
            self._frame_change_threshold = np.clip(
                avg_diff * 1.5,  # 1.5x the average difference
                self._min_change_threshold,
                self._max_change_threshold
            )
        
        # Update the stored frames if we're keeping this one
        if diff > self._frame_change_threshold:
            # Only update if the frame has changed significantly
            self._last_frame = new_frame
            self._last_downsampled = downsampled
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error in frame comparison: {e}")
        # On error, assume the frame has changed to be safe
        self._last_frame = new_frame
        self._last_downsampled = downsampled
        return True

def _adjust_fps(self, frame_time):
    """
    Adjust target FPS based on performance and frame differences.
    
    This version uses more sophisticated logic to balance performance
    and responsiveness based on frame differences and processing times.
    
    Args:
        frame_time: Time taken to process the last frame
    """
    if not self._frame_times or not self._frame_diffs:
        return
            
    # Calculate average frame time and difference
    avg_frame_time = np.mean(self._frame_times)
    avg_diff = np.mean(self._frame_diffs)
    
    # Calculate target frame time based on current FPS
    target_frame_time = 1.0 / self._target_fps
    
    # Adjust FPS based on performance and content changes
    if len(self._frame_times) >= 3:  # Need some history
        # If we're processing frames very quickly and seeing lots of changes,
        # we might want to increase FPS to capture more detail
        if (avg_frame_time < target_frame_time * 0.7 and 
            avg_diff > self._frame_change_threshold * 0.8 and
            self._target_fps < self._max_fps):
            # Increase FPS more aggressively if we're well below target
            increment = 10.0 if avg_frame_time < target_frame_time * 0.5 else 5.0
            self._target_fps = min(self._max_fps, self._target_fps + increment)
        
        # If we're struggling to keep up, decrease FPS more aggressively
        elif avg_frame_time > target_frame_time * 1.3 and self._target_fps > self._min_fps:
            # Decrease more aggressively if we're really struggling
            decrement = 10.0 if avg_frame_time > target_frame_time * 2.0 else 5.0
            self._target_fps = max(self._min_fps, self._target_fps - decrement)
        
        # If content is very static, we can reduce FPS to save resources
        elif (avg_diff < self._frame_change_threshold * 0.3 and 
              self._target_fps > self._min_fps + 5.0):
            self._target_fps = max(self._min_fps, self._target_fps - 1.0)
    
    # Ensure FPS stays within bounds
    self._target_fps = np.clip(self._target_fps, self._min_fps, self._max_fps)
    self._target_fps = max(self._min_fps, min(self._max_fps, self._target_fps))
