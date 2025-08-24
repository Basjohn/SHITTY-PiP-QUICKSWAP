"""
MediaPlayerKeepAlive - Continuous monitoring and responsiveness checking for media applications.

Provides background monitoring of media applications with ThreadManager integration,
crash detection, and automatic app catalog updates.
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Dict, Set, List, Tuple
from dataclasses import dataclass

from core.logging import get_logger
from core.settings.settings_manager import SettingsManager
from core.threading.manager import ThreadManager
from core.media.media_controller import MediaController, get_media_controller
from utils.win.winmsg import is_process_responsive

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


@dataclass
class AppStatus:
    """Status tracking for a monitored media application."""
    hwnd: int
    process_name: str
    pid: int
    last_check: float
    responsive: bool
    consecutive_failures: int
    last_seen: float


class MediaPlayerKeepAlive:
    """
    Background monitoring service for media applications.
    
    Features:
    - Continuous responsiveness checking with configurable intervals
    - Automatic detection of new/closed media applications
    - Crash-prone app flagging and recovery
    - ThreadManager integration for all async operations
    - Settings-driven monitoring intervals and thresholds
    """
    
    _instance: Optional["MediaPlayerKeepAlive"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
            
        self._logger = get_logger("MEDIA_KEEPALIVE")
        self._settings = SettingsManager()
        self._media_controller: Optional[MediaController] = None
        
        # Monitoring state
        self._monitored_apps: Dict[int, AppStatus] = {}  # hwnd -> AppStatus
        self._running = False
        self._monitor_timer_id: Optional[str] = None
        
        # Configuration (CPU-efficient settings)
        self._check_interval_ms = 10000  # 10 seconds - reduced frequency for efficiency
        self._responsiveness_timeout_ms = 1000  # 1 second - faster timeout
        self._max_consecutive_failures = 2  # Faster detection of issues
        self._discovery_interval_ms = 30000  # 30 seconds - less frequent discovery
        self._last_discovery = 0.0
        
        # Adaptive polling - slow down when all apps are responsive
        self._adaptive_polling = True
        self._slow_interval_ms = 20000  # 20 seconds when all apps responsive
        self._fast_interval_ms = 5000   # 5 seconds when issues detected
        self._current_interval_ms = self._check_interval_ms
        
        # Settings integration
        self._enabled = bool(self._settings.get("features.media_control_enabled", False))
        try:
            self._settings.register_change_handler(
                "features.media_control_enabled", self._on_setting_changed
            )
        except Exception as e:
            self._logger.error(f"Failed to register settings handler: {e}")
        
        self._initialized = True
        self._logger.debug(f"Initialized MediaPlayerKeepAlive enabled={self._enabled}")
    
    def start(self) -> None:
        """Start the keepalive monitoring service."""
        if self._running or not self._enabled:
            return
            
        try:
            self._media_controller = get_media_controller()
            if not self._media_controller:
                self._logger.warning("MediaController not available, keepalive disabled")
                return
                
            self._running = True
            self._schedule_monitor_cycle()
            self._logger.info("MediaPlayerKeepAlive started")
            
        except Exception as e:
            self._logger.error(f"Failed to start MediaPlayerKeepAlive: {e}")
    
    def stop(self) -> None:
        """Stop the keepalive monitoring service."""
        if not self._running:
            return
            
        self._running = False
        if self._monitor_timer_id:
            try:
                # Cancel the scheduled timer if possible
                pass  # ThreadManager doesn't expose timer cancellation
            except Exception:
                pass
            self._monitor_timer_id = None
            
        self._monitored_apps.clear()
        self._logger.info("MediaPlayerKeepAlive stopped")
    
    def get_monitored_apps(self) -> List[Tuple[str, int, bool]]:
        """Get list of currently monitored apps as (process_name, hwnd, responsive) tuples."""
        return [
            (status.process_name, status.hwnd, status.responsive)
            for status in self._monitored_apps.values()
        ]
    
    def force_check(self, hwnd: int) -> Optional[bool]:
        """Force an immediate responsiveness check for a specific window.
        
        Returns True if responsive, False if unresponsive, None if not monitored.
        """
        if hwnd not in self._monitored_apps:
            return None
            
        status = self._monitored_apps[hwnd]
        responsive = self._check_window_responsive(hwnd, status.process_name)
        
        if responsive != status.responsive:
            self._logger.debug(f"Responsiveness changed for {status.process_name} (hwnd={hwnd}): {responsive}")
            status.responsive = responsive
            status.consecutive_failures = 0 if responsive else status.consecutive_failures + 1
            
        status.last_check = time.monotonic()
        return responsive
    
    def _on_setting_changed(self, key: str, value) -> None:
        """Handle media control setting changes."""
        if key != "features.media_control_enabled":
            return
            
        new_enabled = bool(value)
        if new_enabled == self._enabled:
            return
            
        self._enabled = new_enabled
        if self._enabled:
            self.start()
        else:
            self.stop()
            
        self._logger.debug(f"MediaPlayerKeepAlive enabled set to {new_enabled}")
    
    def _schedule_monitor_cycle(self) -> None:
        """Schedule the next monitoring cycle."""
        if not self._running:
            return
            
        try:
            self._monitor_timer_id = ThreadManager.single_shot(
                self._current_interval_ms, self._monitor_cycle
            )
        except Exception as e:
            self._logger.error(f"Failed to schedule monitor cycle: {e}")
            # Fallback: try again in a longer interval
            try:
                ThreadManager.single_shot(self._current_interval_ms * 2, self._schedule_monitor_cycle)
            except Exception:
                self._logger.error("Failed to schedule fallback monitor cycle, stopping")
                self._running = False
    
    def _monitor_cycle(self) -> None:
        """Main monitoring cycle - check responsiveness and discover new apps."""
        if not self._running:
            return
            
        try:
            current_time = time.monotonic()
            
            # Periodic app discovery
            if current_time - self._last_discovery > (self._discovery_interval_ms / 1000.0):
                self._discover_media_apps()
                self._last_discovery = current_time
            
            # Check responsiveness of monitored apps
            issues_detected = self._check_monitored_apps()
            
            # Adaptive polling: adjust interval based on app health
            if self._adaptive_polling:
                self._adjust_polling_interval(issues_detected)
            
            # Clean up dead/closed windows
            self._cleanup_dead_apps()
            
        except Exception as e:
            self._logger.error(f"Error in monitor cycle: {e}")
        finally:
            # Schedule next cycle
            self._schedule_monitor_cycle()
    
    def _discover_media_apps(self) -> None:
        """Discover new media applications that should be monitored."""
        if not self._media_controller:
            return
            
        try:
            # Get current media apps from MediaController
            media_apps = self._media_controller.get_running_media_apps()
            
            for app_name, hwnd in media_apps:
                if hwnd not in self._monitored_apps:
                    # New app discovered
                    try:
                        pid = self._get_window_pid(hwnd)
                        if pid:
                            status = AppStatus(
                                hwnd=hwnd,
                                process_name=app_name,
                                pid=pid,
                                last_check=time.monotonic(),
                                responsive=True,  # Assume responsive initially
                                consecutive_failures=0,
                                last_seen=time.monotonic()
                            )
                            self._monitored_apps[hwnd] = status
                            self._logger.debug(f"Started monitoring {app_name} (hwnd={hwnd}, pid={pid})")
                    except Exception as e:
                        self._logger.warning(f"Failed to start monitoring {app_name}: {e}")
                        
        except Exception as e:
            self._logger.error(f"Error discovering media apps: {e}")
    
    def _check_monitored_apps(self) -> bool:
        """Check responsiveness of all monitored applications.
        
        Returns True if any issues were detected, False if all apps are healthy.
        """
        current_time = time.monotonic()
        issues_detected = False
        
        for hwnd, status in list(self._monitored_apps.items()):
            try:
                # Check if window still exists
                if not self._is_window_valid(hwnd):
                    self._logger.debug(f"Window {status.process_name} (hwnd={hwnd}) no longer valid")
                    del self._monitored_apps[hwnd]
                    continue
                
                # Check responsiveness
                responsive = self._check_window_responsive(hwnd, status.process_name)
                
                if responsive != status.responsive:
                    if responsive:
                        self._logger.info(f"{status.process_name} (hwnd={hwnd}) recovered, now responsive")
                        status.consecutive_failures = 0
                    else:
                        status.consecutive_failures += 1
                        issues_detected = True  # Mark that we found issues
                        self._logger.warning(
                            f"{status.process_name} (hwnd={hwnd}) unresponsive "
                            f"(failures: {status.consecutive_failures}/{self._max_consecutive_failures})"
                        )
                        
                        # Mark as crash-prone if too many failures
                        if status.consecutive_failures >= self._max_consecutive_failures:
                            self._mark_app_crash_prone(status.process_name)

            except Exception as e:
                self._logger.error(f"Error checking {status.process_name}: {e}")
        
        return issues_detected
    
    def _cleanup_dead_apps(self) -> None:
        """Remove monitoring for applications that are no longer running."""
        current_time = time.monotonic()
        dead_hwnds = []
        
        for hwnd, status in self._monitored_apps.items():
            # Remove apps not seen for a while or with invalid windows
            if (current_time - status.last_seen > 30.0 or  # 30 seconds timeout
                not self._is_window_valid(hwnd)):
                dead_hwnds.append(hwnd)
        
        for hwnd in dead_hwnds:
            status = self._monitored_apps.pop(hwnd, None)
            if status:
                self._logger.debug(f"Stopped monitoring {status.process_name} (hwnd={hwnd})")
    
    def _check_window_responsive(self, hwnd: int, app_name: str) -> bool:
        """Check if a window is responsive using safe messaging."""
        try:
            return is_process_responsive(hwnd, timeout_ms=self._responsiveness_timeout_ms)
        except Exception as e:
            self._logger.debug(f"Responsiveness check failed for {app_name}: {e}")
            return False
    
    def _is_window_valid(self, hwnd: int) -> bool:
        """Check if a window handle is still valid."""
        try:
            from utils.win.winmsg import is_window
            return is_window(hwnd)
        except Exception:
            return False
    
    def _get_window_pid(self, hwnd: int) -> Optional[int]:
        """Get process ID for a window handle."""
        if not _PSUTIL_AVAILABLE:
            return None
            
        try:
            import win32gui
            import win32process
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            return pid
        except Exception:
            return None
    
    def _mark_app_crash_prone(self, app_name: str) -> None:
        """Mark an application as crash-prone in the MediaController."""
        if not self._media_controller:
            return
            
        try:
            # Update MediaController's app catalog to mark as crash-prone
            if hasattr(self._media_controller, '_apps') and app_name in self._media_controller._apps:
                app_info = self._media_controller._apps[app_name]
                if not app_info.crash_prone:
                    app_info.crash_prone = True
                    self._logger.warning(f"Marked {app_name} as crash-prone due to repeated unresponsiveness")
        except Exception as e:
            self._logger.error(f"Failed to mark {app_name} as crash-prone: {e}")

    def _adjust_polling_interval(self, issues_detected: bool) -> None:
        """Adjust polling interval based on app health to minimize CPU usage.
        
        Uses fast polling when issues are detected, slow polling when all apps are healthy.
        """
        if issues_detected:
            # Switch to fast polling when problems detected
            new_interval = self._fast_interval_ms
            if self._current_interval_ms != new_interval:
                self._current_interval_ms = new_interval
                self._logger.debug(f"Switched to fast polling ({new_interval}ms) due to detected issues")
        else:
            # Switch to slow polling when all apps are healthy
            new_interval = self._slow_interval_ms
            if self._current_interval_ms != new_interval:
                self._current_interval_ms = new_interval
                self._logger.debug(f"Switched to slow polling ({new_interval}ms) - all apps healthy")


# Singleton accessor
_keepalive_instance: Optional[MediaPlayerKeepAlive] = None

def get_media_keepalive() -> MediaPlayerKeepAlive:
    """Get the global MediaPlayerKeepAlive instance."""
    global _keepalive_instance
    if _keepalive_instance is None:
        _keepalive_instance = MediaPlayerKeepAlive()
    return _keepalive_instance
