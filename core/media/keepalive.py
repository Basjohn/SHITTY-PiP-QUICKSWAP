"""
MediaPlayerKeepAlive - Continuous monitoring and responsiveness checking for media applications.

Provides background monitoring of media applications with ThreadManager integration,
crash detection, and automatic app catalog updates.
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from core.logging import get_logger
from core.settings.settings_manager import SettingsManager
from core.threading import ThreadManager
from core.media.media_controller import MediaController, get_media_controller
from utils.win.winmsg import is_process_responsive

try:
    from importlib.util import find_spec as _find_spec
    _PSUTIL_AVAILABLE = _find_spec('psutil') is not None
except Exception:
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
    # Media tracking fields
    media_activity_detected: bool = False
    last_media_activity: float = 0.0
    needs_background_keepalive: bool = False
    z_order_position: Optional[int] = None
    is_minimized: bool = False


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
        # Set up throttled and deduped log emitters for hot-path logs
        try:
            from core.logging.logger_impl import throttled, log_dedupe
            self._tdebug_init = throttled(self._logger.debug, "media:init", 1000)
            self._tdebug_disc = throttled(self._logger.debug, "media:discover", 2000)
            self._tdebug_check = throttled(self._logger.debug, "media:check", 500)
            self._tdebug_keepalive = throttled(self._logger.debug, "media:keepalive", 1000)
            self._tdebug_act = throttled(self._logger.debug, "media:activate", 2000)
            self._dwarn_unresp = log_dedupe(self._logger.warning, "media:unresponsive", 5000)
        except Exception:
            # Fallbacks if helpers are not available during early startup
            self._tdebug_init = self._logger.debug
            self._tdebug_disc = self._logger.debug
            self._tdebug_check = self._logger.debug
            self._tdebug_keepalive = self._logger.debug
            self._tdebug_act = self._logger.debug
            self._dwarn_unresp = self._logger.warning
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
        
        # Focus and activation management (subtle, no focus steal)
        self._focus_candidates: set[int] = set()
        self._last_focus_attempt: Dict[int, float] = {}
        self._focus_cooldown_seconds = 30.0
        self._media_state_cache: Dict[int, str] = {}
        self._browser_activation_tracking: Dict[int, Dict] = {}
        # Setting-driven toggle, default True
        try:
            self._subtle_activation_enabled = bool(
                self._settings.get("media.subtle_activation_enabled", True)
            )
        except Exception:
            self._subtle_activation_enabled = True
        
        # Media keepalive configuration
        self._media_keepalive_interval_ms = 15000
        self._media_activity_timeout_ms = 300000
        self._last_keepalive_sweep = 0.0
        # Browser-specific tolerance
        self._browser_media_tolerance = 4
        # Enable/disable media activity detection heuristic (do not shadow method name)
        self._media_activity_detection_enabled = True
        
        # Settings integration
        self._enabled = bool(self._settings.get("features.media_control_enabled", False))
        try:
            self._settings.register_change_handler(
                "features.media_control_enabled", self._on_setting_changed
            )
        except Exception as e:
            self._logger.error(f"Failed to register settings handler: {e}")
        
        self._initialized = True
        self._tdebug_init(f"Initialized MediaPlayerKeepAlive enabled={self._enabled}")
    
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
            self._schedule_keepalive_sweep()
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
            self._tdebug_check(f"Responsiveness changed for {status.process_name} (hwnd={hwnd}): {responsive}")
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
                            self._tdebug_disc(f"Started monitoring {app_name} (hwnd={hwnd}, pid={pid})")
                    except Exception as e:
                        self._logger.warning(f"Failed to start monitoring {app_name}: {e}")
                        
        except Exception as e:
            self._logger.error(f"Error discovering media apps: {e}")
    
    def _check_monitored_apps(self) -> bool:
        """Enhanced app monitoring with media activation detection and background keepalive."""
        current_time = time.monotonic()
        issues_detected = False
        
        for hwnd, status in list(self._monitored_apps.items()):
            try:
                # Check if window still exists
                if not self._is_window_valid(hwnd):
                    self._tdebug_check(f"Window {status.process_name} (hwnd={hwnd}) no longer valid")
                    del self._monitored_apps[hwnd]
                    continue
                
                # Update window state tracking
                status.is_minimized = self._is_window_minimized(hwnd)
                current_z_order = self._get_window_z_order(hwnd)
                z_order_changed = (status.z_order_position != current_z_order)
                status.z_order_position = current_z_order
                
                # Check for media activity
                if self._media_activity_detection_enabled:
                    has_media = self._detect_media_activity(hwnd, status.process_name)
                else:
                    has_media = False
                if has_media:
                    status.media_activity_detected = True
                    status.last_media_activity = current_time
                    # Mark browsers/discord with media for background keepalive
                    if status.process_name in ['chrome', 'firefox', 'edge', 'discord']:
                        status.needs_background_keepalive = True
                
                # Detect if this might be media that needs activation
                if self._detect_media_needs_activation(hwnd, status.process_name):
                    self._focus_candidates.add(hwnd)
                    # Try to detect if media is paused/inactive
                    if self._is_media_likely_inactive(hwnd, status.process_name):
                        # Attempt subtle activation (no focus steal)
                        if self._perform_subtle_activation(hwnd, status.process_name):
                            self._tdebug_act(f"Performed activation for {status.process_name}")
                
                # Perform background keepalive for media windows that need it
                if (
                    status.needs_background_keepalive and 
                    (current_time - status.last_media_activity) < (self._media_activity_timeout_ms / 1000.0)
                ):
                    should_keepalive = (
                        status.is_minimized or 
                        (current_z_order is not None and current_z_order > 5) or
                        z_order_changed
                    )
                    if should_keepalive:
                        if self._perform_background_keepalive(hwnd, status.process_name):
                            self._tdebug_keepalive(f"Background keepalive sent to {status.process_name}")
                
                # Regular responsiveness check
                responsive = self._check_window_responsive(hwnd, status.process_name)
                if responsive != status.responsive:
                    if responsive:
                        self._logger.info(f"{status.process_name} (hwnd={hwnd}) recovered, now responsive")
                        status.consecutive_failures = 0
                    else:
                        status.consecutive_failures += 1
                        issues_detected = True
                        # Be more lenient for media-active apps
                        max_failures = self._max_consecutive_failures
                        if status.media_activity_detected:
                            max_failures *= 2
                        self._logger.warning(
                            self._dwarn_unresp(
                                f"{status.process_name} (hwnd={hwnd}) unresponsive "
                                f"(failures: {status.consecutive_failures}/{max_failures})"
                            )
                        )
                        if status.consecutive_failures >= max_failures:
                            self._mark_app_crash_prone(status.process_name)
                
                status.responsive = responsive
                status.last_check = current_time
                status.last_seen = current_time
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
            self._tdebug_check(f"Responsiveness check failed for {app_name}: {e}")
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
            import win32process
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            return pid
        except Exception:
            return None
    
    def _detect_media_needs_activation(self, hwnd: int, app_name: str) -> bool:
        """Heuristic detection if media likely needs focus activation to play."""
        if not app_name or app_name.lower() not in ['chrome', 'firefox', 'edge', 'discord']:
            return False
        try:
            import win32gui
            title = (win32gui.GetWindowText(hwnd) or '').lower()
            indicators = ['youtube', 'netflix', 'spotify', 'twitch', 'vimeo', 'soundcloud', 'bandcamp', 'disney', 'hulu', 'prime video', 'plex', 'jellyfin', 'emby', 'crunchyroll']
            return any(ind in title for ind in indicators)
        except Exception:
            return False
    
    def _detect_media_activity(self, hwnd: int, app_name: str) -> bool:
        """Detect if there's active media without affecting focus."""
        try:
            if hasattr(self._media_controller, 'get_session_volume_for_hwnd'):
                volume = self._media_controller.get_session_volume_for_hwnd(hwnd)
                if volume is not None and volume > 0:
                    return True
            if app_name in ['chrome', 'firefox', 'edge', 'discord']:
                try:
                    import win32gui
                    title = (win32gui.GetWindowText(hwnd) or '').lower()
                    media_indicators = ['▶', '⏸', '🔊', '🔇', 'playing', 'paused', 'youtube', 'netflix', 'spotify', 'twitch', 'prime video', 'disney', 'hulu', 'plex']
                    if any(ind in title for ind in media_indicators):
                        return True
                except Exception:
                    pass
                try:
                    import win32gui
                    media_child_found = False
                    def enum_child_callback(child_hwnd, _):
                        nonlocal media_child_found
                        try:
                            child_class = (win32gui.GetClassName(child_hwnd) or '').lower()
                            if any(cls in child_class for cls in ['video', 'media', 'player', 'canvas']):
                                media_child_found = True
                                return False
                        except Exception:
                            pass
                        return True
                    win32gui.EnumChildWindows(hwnd, enum_child_callback, None)
                    if media_child_found:
                        return True
                except Exception:
                    pass
            return False
        except Exception as e:
            self._tdebug_check(f"Media activity detection failed for {app_name}: {e}")
            return False
    
    def _perform_subtle_activation(self, hwnd: int, app_name: str) -> bool:
        """Perform subtle activation without stealing user focus."""
        if not self._subtle_activation_enabled:
            return False
        now = time.monotonic()
        last = self._last_focus_attempt.get(hwnd, 0.0)
        if (now - last) < self._focus_cooldown_seconds:
            return False
        try:
            import win32gui
            import win32con
            current_fg = win32gui.GetForegroundWindow()
            if app_name in ['chrome', 'edge']:
                if not win32gui.IsIconic(hwnd):
                    placement = win32gui.GetWindowPlacement(hwnd)
                    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                    time.sleep(0.05)
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    try:
                        win32gui.SetWindowPlacement(hwnd, placement)
                    except Exception:
                        pass
            elif app_name == 'firefox':
                try:
                    win32gui.SetWindowPos(
                        hwnd, win32con.HWND_TOP, 0, 0, 0, 0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
                    )
                    time.sleep(0.1)
                    win32gui.SetWindowPos(
                        hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE
                    )
                except Exception:
                    pass
            else:
                try:
                    win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
                except Exception:
                    pass
            # Restore original foreground window if changed
            try:
                if current_fg and current_fg != win32gui.GetForegroundWindow():
                    win32gui.SetForegroundWindow(current_fg)
            except Exception:
                pass
            self._last_focus_attempt[hwnd] = now
            return True
        except Exception as e:
            self._tdebug_act(f"Subtle activation failed for {app_name}: {e}")
            return False
    
    def _get_window_z_order(self, hwnd: int) -> Optional[int]:
        """Get approximate Z-order position of window (lower = more foreground)."""
        try:
            import win32gui
            z_order = 0
            def enum_callback(enum_hwnd, _):
                nonlocal z_order
                if enum_hwnd == hwnd:
                    return False
                if win32gui.IsWindowVisible(enum_hwnd):
                    z_order += 1
                return True
            win32gui.EnumWindows(enum_callback, None)
            return z_order
        except Exception:
            return None
    
    def _is_window_minimized(self, hwnd: int) -> bool:
        try:
            import win32gui
            return bool(win32gui.IsIconic(hwnd))
        except Exception:
            return False
    
    def _perform_background_keepalive(self, hwnd: int, app_name: str) -> bool:
        """Send lightweight keepalive messages that do not affect focus or UI."""
        try:
            if app_name in ['chrome', 'firefox', 'edge']:
                import win32gui
                import win32con
                try:
                    win32gui.PostMessage(hwnd, win32con.WM_NULL, 0, 0)
                except Exception:
                    pass
                children: list[int] = []
                try:
                    win32gui.EnumChildWindows(hwnd, lambda ch, _: (children.append(ch), True)[1], None)
                except Exception:
                    pass
                for ch in children[:5]:
                    try:
                        win32gui.PostMessage(ch, win32con.WM_NULL, 0, 0)
                    except Exception:
                        pass
                return True
            elif app_name in ['vlc', 'potplayer', 'kmplayer']:
                import win32gui
                import win32con
                try:
                    win32gui.PostMessage(hwnd, win32con.WM_TIMER, 9999, 0)
                    return True
                except Exception:
                    return False
            return False
        except Exception as e:
            self._tdebug_keepalive(f"Background keepalive failed for {app_name}: {e}")
            return False
    
    def _is_media_likely_inactive(self, hwnd: int, app_name: str) -> bool:
        try:
            import win32gui
            if app_name in ['chrome', 'firefox', 'edge']:
                title = (win32gui.GetWindowText(hwnd) or '').lower()
                if any(ind in title for ind in ['paused', '⏸', 'stopped']):
                    return True
                if 'youtube' in title and '▶' not in title:
                    return True
        except Exception:
            pass
        return False
    
    def _schedule_keepalive_sweep(self) -> None:
        if not self._running:
            return
        try:
            ThreadManager.single_shot(self._media_keepalive_interval_ms, self._keepalive_sweep)
        except Exception as e:
            self._logger.error(f"Failed to schedule keepalive sweep: {e}")
    
    def _keepalive_sweep(self) -> None:
        if not self._running:
            return
        try:
            now = time.monotonic()
            for hwnd, status in list(self._monitored_apps.items()):
                if (
                    status.needs_background_keepalive and 
                    (now - status.last_media_activity) < (self._media_activity_timeout_ms / 1000.0)
                ):
                    self._perform_background_keepalive(hwnd, status.process_name)
        except Exception as e:
            self._logger.error(f"Error in keepalive sweep: {e}")
        finally:
            self._schedule_keepalive_sweep()
    
    def hint_media_activity(self, hwnd: int) -> None:
        """External hint that media activity was detected in a window."""
        if hwnd in self._monitored_apps:
            status = self._monitored_apps[hwnd]
            status.media_activity_detected = True
            status.last_media_activity = time.monotonic()
            status.needs_background_keepalive = True
            try:
                self._tdebug_check(f"Media activity hinted for {status.process_name}")
            except Exception:
                pass
    
    def request_subtle_activation(self, hwnd: int, app_name: Optional[str] = None) -> bool:
        """Public API: attempt subtle activation if heuristics suggest it.
        
        Does not steal focus; rate-limited per hwnd.
        Returns True if an activation attempt was performed.
        """
        try:
            if hwnd not in self._monitored_apps:
                # Try to detect app_name if not provided; leave None if unknown
                pname = (app_name or "").lower()
            else:
                pname = (self._monitored_apps[hwnd].process_name or app_name or "").lower()
        except Exception:
            pname = (app_name or "").lower()
        
        try:
            if not pname:
                # Best effort: attempt a conservative activation
                return self._perform_subtle_activation(hwnd, "")
            if self._detect_media_needs_activation(hwnd, pname) or self._is_media_likely_inactive(hwnd, pname):
                return self._perform_subtle_activation(hwnd, pname)
            return False
        except Exception:
            return False
    
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
                self._tdebug_check(f"Switched to fast polling ({new_interval}ms) due to detected issues")
        else:
            # Switch to slow polling when all apps are healthy
            new_interval = self._slow_interval_ms
            if self._current_interval_ms != new_interval:
                self._current_interval_ms = new_interval
                self._tdebug_check(f"Switched to slow polling ({new_interval}ms) - all apps healthy")


# Singleton accessor
_keepalive_instance: Optional[MediaPlayerKeepAlive] = None

def get_media_keepalive() -> MediaPlayerKeepAlive:
    """Get the global MediaPlayerKeepAlive instance."""
    global _keepalive_instance
    if _keepalive_instance is None:
        _keepalive_instance = MediaPlayerKeepAlive()
    return _keepalive_instance
