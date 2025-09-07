"""Centralized media controller with app-specific routing and crash protection."""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from PySide6.QtCore import QTimer
from core.logging import get_logger
from utils.win.winmsg import (
    is_window,
    key_press,
    safe_send_appcommand,
    is_process_responsive,
    WIN_AVAILABLE,
    key_press_with_char,
    send_wm_command,
)
from utils.audio import session_volume

if WIN_AVAILABLE:
    import win32gui
    import win32process
    import win32con
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
else:
    PSUTIL_AVAILABLE = False


@dataclass
class MediaApp:
    """Media application definition with safe methods and crash protection."""
    process: str
    class_name: Optional[str] = None
    safe_methods: List[str] = field(default_factory=lambda: ['media_command'])
    crash_prone: bool = False
    hotkeys: Dict[str, int] = field(default_factory=dict)
    commands: Dict[str, int] = field(default_factory=dict)


class MediaController:
    """Centralized media player controller with app-specific routing."""
    
    # Media command constants
    APPCOMMAND_MEDIA_PLAY_PAUSE = 14
    APPCOMMAND_MEDIA_NEXTTRACK = 11
    APPCOMMAND_MEDIA_PREVIOUSTRACK = 12
    APPCOMMAND_MEDIA_STOP = 13
    APPCOMMAND_VOLUME_MUTE = 8
    APPCOMMAND_VOLUME_DOWN = 9
    APPCOMMAND_VOLUME_UP = 10
    
    def __init__(self, settings_manager):
        """Initialize MediaController with settings manager."""
        self._logger = get_logger(__name__)
        self._settings = settings_manager
        # Window enumeration cache
        self._window_cache = {}  # Cache for window enumeration
        self._cache_ttl = 5.0  # Cache TTL in seconds
        self._last_cache_update = 0.0
        # Initialize state
        self._running_media_apps = {}  # hwnd -> app_name
        self._app_detection_cache = {}  # hwnd -> (app_name, timestamp)
        self._app_detection_ttl = 2.0  # seconds
        
        # Continuous volume adjustment state
        self._volume_hold_state = {}  # hwnd -> {'direction': 'up'/'down', 'start_time': float, 'timer': QTimer}
        self._volume_hold_delay = 0.5  # seconds before continuous adjustment starts
        self._volume_continuous_interval = 0.05  # seconds between continuous steps
        
        # App catalog with comprehensive definitions from SPQStruggle
        self._default_apps = {
            # Audio players
            'spotify': MediaApp(
                process='spotify.exe',
                safe_methods=['media_command', 'spacebar']
            ),
            'foobar': MediaApp(
                process='foobar2000.exe',
                class_name='{E7076D1C-A7BF-4f39-B771-BCBE88F2A2A8}',
                safe_methods=['media_command', 'hotkeys']
            ),
            'aimp': MediaApp(
                process='AIMP.exe',
                class_name='AIMP2_MainForm',
                safe_methods=['media_command', 'hotkeys']
            ),
            'winamp': MediaApp(
                process='winamp.exe',
                class_name='Winamp v1.x',
                safe_methods=['media_command', 'hotkeys']
            ),
            'musicbee': MediaApp(
                process='MusicBee.exe',
                class_name='WindowsForms10.Window.8.app.0.2bf8098_r11_ad1',
                safe_methods=['media_command']
            ),
            'mediamonkey': MediaApp(
                process='MediaMonkey.exe',
                class_name='TMainForm',
                safe_methods=['media_command']
            ),
            'itunes': MediaApp(
                process='iTunes.exe',
                class_name='iTunes',
                safe_methods=['media_command']
            ),
            
            # Browsers
            'chrome': MediaApp(
                process='chrome.exe',
                class_name='Chrome_WidgetWin_1',
                safe_methods=['spacebar', 'media_command']
            ),
            'firefox': MediaApp(
                process='firefox.exe',
                class_name='MozillaWindowClass',
                safe_methods=['spacebar', 'media_command']
            ),
            'edge': MediaApp(
                process='msedge.exe',
                class_name='Chrome_WidgetWin_1',
                safe_methods=['spacebar', 'media_command']
            ),
            'discord': MediaApp(
                process='Discord.exe',
                class_name='Chrome_WidgetWin_1',
                safe_methods=['spacebar', 'media_command']
            ),
            
            # Video players - safe
            'vlc': MediaApp(
                process='vlc.exe',
                class_name='Qt5QWindowIcon',
                safe_methods=['media_command', 'hotkeys'],
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': 0x4E,  # N key
                    'previous': 0x50,  # P key
                    'stop': 0x53,  # S key
                    'volume_up': win32con.VK_UP if WIN_AVAILABLE else 0x26,
                    'volume_down': win32con.VK_DOWN if WIN_AVAILABLE else 0x28,
                }
            ),
            'potplayer': MediaApp(
                process='PotPlayerMini.exe',
                class_name='PotPlayer',
                safe_methods=['media_command', 'hotkeys'],
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
                    'previous': win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
                }
            ),
            'potplayer64': MediaApp(
                process='PotPlayerMini64.exe',
                class_name='PotPlayer64',
                safe_methods=['media_command', 'hotkeys']
            ),
            'kmplayer': MediaApp(
                process='KMPlayer.exe',
                class_name='KMPWnd',
                safe_methods=['media_command', 'hotkeys']
            ),
            'gom': MediaApp(
                process='GOM.exe',
                class_name='GomPlayerWndClass',
                safe_methods=['media_command', 'hotkeys']
            ),
            'bsplayer': MediaApp(
                process='bsplayer.exe',
                class_name='BSPlayer',
                safe_methods=['media_command', 'hotkeys']
            ),
            'media_player': MediaApp(
                process='wmplayer.exe',
                class_name='WMPlayerApp',
                safe_methods=['media_command']
            ),
            'plex': MediaApp(
                process='PlexMediaPlayer.exe',
                class_name='Qt5QWindowIcon',
                safe_methods=['media_command', 'spacebar']
            ),
            'jellyfin': MediaApp(
                process='JellyfinMediaPlayer.exe',
                class_name='Qt5QWindowIcon',
                safe_methods=['media_command', 'spacebar']
            ),
            
            # Video players - crash prone (hotkeys only)
            'mpc_hc': MediaApp(
                process='mpc-hc.exe',
                class_name='MediaPlayerClassicW',
                safe_methods=['wm_command', 'hotkeys'],
                crash_prone=False,
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
                    'previous': win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
                    'next_alt': 0xDD,  # ] key fallback
                    'previous_alt': 0xDB,  # [ key fallback
                    'stop': 0x53,  # S key
                    'volume_up': win32con.VK_UP if WIN_AVAILABLE else 0x26,
                    'volume_down': win32con.VK_DOWN if WIN_AVAILABLE else 0x28,
                }
            ),
            'mpc_hc64': MediaApp(
                process='mpc-hc64.exe',
                class_name='MediaPlayerClassicW',
                safe_methods=['wm_command', 'hotkeys'],
                crash_prone=False,
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
                    'previous': win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
                    'next_alt': 0xDD,  # ] key fallback
                    'previous_alt': 0xDB,  # [ key fallback
                    'stop': 0x53,  # S key
                }
            ),
            'mpc_be': MediaApp(
                process='mpc-be.exe',
                class_name='MediaPlayerClassicW',
                safe_methods=['wm_command', 'hotkeys'],
                crash_prone=False,
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
                    'previous': win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
                    'next_alt': 0xDD,  # ] key fallback
                    'previous_alt': 0xDB,  # [ key fallback
                    'stop': 0x53,  # S key
                }
            ),
            'mpc_be64': MediaApp(
                process='mpc-be64.exe',
                class_name='MediaPlayerClassicW',
                safe_methods=['wm_command', 'hotkeys'],
                crash_prone=False,
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
                    'previous': win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
                    'next_alt': 0xDD,  # ] key fallback
                    'previous_alt': 0xDB,  # [ key fallback
                    'stop': 0x53,  # S key
                }
            ),
            'mpv': MediaApp(
                process='mpv.exe',
                class_name='mpv',
                safe_methods=['hotkeys_only'],
                crash_prone=True,
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
                    'previous': win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
                    'stop': 0x53,  # S key
                    'volume_up': win32con.VK_UP if WIN_AVAILABLE else 0x26,
                    'volume_down': win32con.VK_DOWN if WIN_AVAILABLE else 0x28,
                }
            ),
            'mpv_net': MediaApp(
                process='mpvnet.exe',
                class_name='mpv.net',
                safe_methods=['hotkeys_only'],
                crash_prone=True,
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
                    'previous': win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
                    'stop': 0x53,  # S key
                }
            ),
            'kodi': MediaApp(
                process='Kodi.exe',
                class_name='Kodi',
                safe_methods=['hotkeys_only'],
                crash_prone=True,
                hotkeys={
                    'play_pause': win32con.VK_SPACE if WIN_AVAILABLE else 0x20,
                    'next': win32con.VK_RIGHT if WIN_AVAILABLE else 0x27,
                    'previous': win32con.VK_LEFT if WIN_AVAILABLE else 0x25,
                    'stop': 0x53,  # S key
                }
            ),
        }
        
        # Load app catalog from settings or use defaults (after defaults are defined)
        self._load_app_catalog()
        # Cache validity timestamp for window enumeration
        self._cache_valid_until = 0

    def _publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event via the centralized ApplicationCore EventSystem.

        This is a no-op if the ApplicationCore or EventSystem are not yet
        initialized.
        """
        try:
            from core.application.core import get_app_core
            core = get_app_core()
            if core and getattr(core, "events", None):
                core.events.publish(event_type, data=data, source=self)
        except Exception:
            # Event system may not be initialized (e.g., tests), ignore
            pass
        
    def _load_app_catalog(self) -> None:
        """Load app catalog from settings with defaults."""
        try:
            catalog_data = self._settings.get('media.app_catalog', {})
            self._apps = self._default_apps.copy()
            
            # Override with user settings if present
            for app_name, app_data in catalog_data.items():
                if isinstance(app_data, dict):
                    self._apps[app_name] = MediaApp(**app_data)
                    
        except Exception as e:
            self._logger.warning(f"Failed to load app catalog, using defaults: {e}")
            self._apps = self._default_apps.copy()
    
    def _is_cache_valid(self) -> bool:
        """Check if window enumeration cache is still valid."""
        import time
        return time.time() < self._cache_valid_until
    
    def _invalidate_cache(self) -> None:
        """Invalidate window enumeration cache."""
        self._window_cache.clear()
        self._cache_valid_until = 0
    
    def _find_windows_by_process(self, process_name: str) -> List[int]:
        """Find all windows belonging to a specific process."""
        if not WIN_AVAILABLE or not PSUTIL_AVAILABLE:
            return []
            
        # Check cache first
        if self._is_cache_valid() and process_name in self._window_cache:
            return self._window_cache[process_name]
        
        windows = []
        
        def enum_callback(hwnd, _):
            # Only include visible top-level windows. Minimized windows typically remain visible
            # (WS_VISIBLE stays set), but truly hidden windows should be excluded.
            if win32gui.IsWindowVisible(hwnd):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    process = psutil.Process(pid)
                    if process.name().lower() == process_name.lower():
                        windows.append(hwnd)
                except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                    pass
            return True
        
        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception as e:
            self._logger.debug(f"Window enumeration failed for {process_name}: {e}")
        
        # Cache results for 5 seconds
        self._window_cache[process_name] = windows
        import time
        self._cache_valid_until = time.time() + 5.0
        
        return windows
    
    def _find_window_by_app(self, app_name: str) -> Optional[int]:
        """Find the main window for a specific app."""
        app_name = app_name.lower()
        
        if app_name not in self._apps:
            return None

        app_info = self._apps[app_name]
        windows = self._find_windows_by_process(app_info.process)
        if windows:
            try:
                self._logger.debug(f"Found {len(windows)} window(s) for {app_name}: {windows[:5]}")
            except Exception:
                pass
        
        if not windows:
            return None
        
        # PRIORITY: Check if overlay is targeting a specific window of this app
        try:
            from core.input import get_key_passthrough_controller
            kp = get_key_passthrough_controller()
            target_hwnd = getattr(kp, '_target_hwnd', None)
            
            if target_hwnd and target_hwnd in windows:
                try:
                    title = win32gui.GetWindowText(target_hwnd) if WIN_AVAILABLE else ""
                    self._logger.debug(f"Using overlay target window for {app_name}: hwnd={target_hwnd} title='{title}'")
                except Exception:
                    pass
                return target_hwnd
        except Exception:
            pass
        
        # For apps with multiple windows, try to find the main one
        for hwnd in windows:
            try:
                if not is_window(hwnd):
                    continue
                    
                window_text = win32gui.GetWindowText(hwnd) if WIN_AVAILABLE else ""
                
                # Skip empty titles or very short titles
                if not window_text or len(window_text) < 3:
                    continue
                
                try:
                    self._logger.debug(f"Selected window for {app_name}: hwnd={hwnd} title='{window_text}'")
                except Exception:
                    pass
                return hwnd
            except Exception:
                continue
        
        # If no specific match, return the first valid window
        for hwnd in windows:
            if is_window(hwnd):
                try:
                    title = win32gui.GetWindowText(hwnd) if WIN_AVAILABLE else ""
                    self._logger.debug(f"Fallback select window for {app_name}: hwnd={hwnd} title='{title}'")
                except Exception:
                    pass
                return hwnd
                
        return None

    def _detect_app_for_hwnd(self, hwnd: int) -> Optional[str]:
        """Detect catalog app name for a specific hwnd by matching process name.

        Returns the app key in `self._apps` if the hwnd belongs to a known media app.
        Uses caching to avoid repeated process lookups for the same HWND.
        """
        if not (WIN_AVAILABLE and hwnd and is_window(hwnd)):
            return None
            
        # Check cache first - optimized for frequent lookups
        current_time = time.time()
        if hwnd in self._app_detection_cache:
            app_name, timestamp = self._app_detection_cache[hwnd]
            # Use cached result if within TTL
            if current_time - timestamp < self._app_detection_ttl:
                return app_name
            # Remove expired entry
            del self._app_detection_cache[hwnd]
                
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if not PSUTIL_AVAILABLE:
                return None
            proc = psutil.Process(pid)
            pname = proc.name().lower()
            self._logger.debug(f"_detect_app_for_hwnd: hwnd={hwnd} pid={pid} process='{pname}'")
            
            app_name = None
            for candidate_name, app in self._apps.items():
                if app.process.lower() == pname:
                    self._logger.debug(f"_detect_app_for_hwnd: Matched {pname} -> {candidate_name}")
                    app_name = candidate_name
                    break
            
            # Enhanced fallback: try to derive app name from window title or process name
            if app_name is None:
                app_name = self._derive_app_name_from_context(hwnd, pname)
                
            # Cache the result (even if None)
            self._app_detection_cache[hwnd] = (app_name, current_time)
            
            if app_name is None:
                self._logger.debug(f"_detect_app_for_hwnd: No match for process '{pname}' among {list(self._apps.keys())}")
            return app_name
            
        except Exception as e:
            self._logger.debug(f"_detect_app_for_hwnd: Exception for hwnd {hwnd}: {e}")
            # Cache the failure to avoid repeated exceptions
            self._app_detection_cache[hwnd] = (None, current_time)
            return None

    def _derive_app_name_from_context(self, hwnd: int, process_name: str) -> Optional[str]:
        """Derive application name from window context when not in catalog.
        
        This helps resolve UNKNOWN application names by using window title,
        class name, and process characteristics to identify media applications.
        """
        try:
            # Get window title and class name for analysis
            window_title = ""
            class_name = ""
            try:
                window_title = win32gui.GetWindowText(hwnd).lower()
                class_name = win32gui.GetClassName(hwnd).lower()
            except Exception:
                pass
            
            # Common media application patterns
            media_patterns = {
                # Process name patterns
                'spotify.exe': 'spotify',
                'vlc.exe': 'vlc',
                'chrome.exe': 'chrome',
                'firefox.exe': 'firefox',
                'msedge.exe': 'edge',
                'discord.exe': 'discord',
                'potplayer.exe': 'potplayer',
                'potplayermini.exe': 'potplayer',
                'potplayermini64.exe': 'potplayer64',
                'mpc-hc.exe': 'mpc_hc',
                'mpc-hc64.exe': 'mpc_hc64',
                'mpc-be.exe': 'mpc_be',
                'mpc-be64.exe': 'mpc_be64',
                'mpv.exe': 'mpv',
                'mpvnet.exe': 'mpv_net',
                'kodi.exe': 'kodi',
                'foobar2000.exe': 'foobar',
                'aimp.exe': 'aimp',
                'winamp.exe': 'winamp',
                'musicbee.exe': 'musicbee',
                'mediamonkey.exe': 'mediamonkey',
                'itunes.exe': 'itunes',
                'kmplayer.exe': 'kmplayer',
                'gom.exe': 'gom',
                'bsplayer.exe': 'bsplayer',
                'wmplayer.exe': 'media_player',
                'plexmediaplayer.exe': 'plex',
                'jellyfinmediaplayer.exe': 'jellyfin',
            }
            
            # Check direct process name match
            if process_name in media_patterns:
                derived_name = media_patterns[process_name]
                self._logger.debug(f"Derived app name from process: {process_name} -> {derived_name}")
                return derived_name
            
            # Check window title patterns for browsers and media apps
            title_patterns = {
                'spotify': 'spotify',
                'youtube': 'chrome',  # Default to chrome for YouTube
                'netflix': 'chrome',
                'twitch': 'chrome',
                'discord': 'discord',
                'vlc media player': 'vlc',
                'potplayer': 'potplayer',
                'media player classic': 'mpc_hc',
                'foobar2000': 'foobar',
                'aimp': 'aimp',
                'winamp': 'winamp',
                'musicbee': 'musicbee',
                'mediamonkey': 'mediamonkey',
                'itunes': 'itunes',
                'kmplayer': 'kmplayer',
                'gom player': 'gom',
                'bs.player': 'bsplayer',
                'windows media player': 'media_player',
                'plex': 'plex',
                'jellyfin': 'jellyfin',
                'kodi': 'kodi',
                'mpv': 'mpv',
            }
            
            for pattern, app_name in title_patterns.items():
                if pattern in window_title:
                    self._logger.debug(f"Derived app name from title: '{window_title}' -> {app_name}")
                    return app_name
            
            # Check class name patterns
            class_patterns = {
                'chrome_widgetwin_1': 'chrome',
                'mozillawindowclass': 'firefox',
                'qt5qwindowicon': 'vlc',  # Common for Qt-based media players
                'potplayer': 'potplayer',
                'mediaplayerclassicw': 'mpc_hc',
                'aimp2_mainform': 'aimp',
                'winamp v1.x': 'winamp',
                'kmplayer': 'kmplayer',
                'gomplayerwndclass': 'gom',
                'bsplayer': 'bsplayer',
                'wmplayerapp': 'media_player',
                'kodi': 'kodi',
                'mpv': 'mpv',
            }
            
            for pattern, app_name in class_patterns.items():
                if pattern in class_name:
                    self._logger.debug(f"Derived app name from class: '{class_name}' -> {app_name}")
                    return app_name
            
            # If we still can't identify it, but it has media-related characteristics, 
            # return a generic identifier based on process name
            if any(keyword in window_title for keyword in ['play', 'music', 'video', 'media', 'player']):
                # Use process name without extension as fallback
                base_name = process_name.replace('.exe', '').replace('.', '_').lower()
                self._logger.debug(f"Generic media app derived: {process_name} -> {base_name}")
                return base_name
            
            return None
            
        except Exception as e:
            self._logger.debug(f"_derive_app_name_from_context failed: {e}")
            return None

    def _start_continuous_volume_adjustment(self, hwnd: int, direction: str) -> None:
        """Start continuous volume adjustment for key hold behavior."""
        try:
            # Stop any existing continuous adjustment for this window
            self._stop_continuous_volume_adjustment(hwnd)
            
            # Create timer for continuous adjustment
            timer = QTimer()
            timer.timeout.connect(lambda: self._perform_continuous_volume_step(hwnd, direction))
            
            # Store state
            self._volume_hold_state[hwnd] = {
                'direction': direction,
                'start_time': time.time(),
                'timer': timer
            }
            
            # Start timer with initial delay, then continuous intervals
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self._switch_to_continuous_mode(hwnd))
            timer.start(int(self._volume_hold_delay * 1000))  # Convert to milliseconds
            
        except Exception as e:
            self._logger.debug(f"Failed to start continuous volume adjustment: {e}")

    def _switch_to_continuous_mode(self, hwnd: int) -> None:
        """Switch from initial delay to continuous adjustment mode."""
        try:
            if hwnd not in self._volume_hold_state:
                return
                
            state = self._volume_hold_state[hwnd]
            timer = state['timer']
            direction = state['direction']
            
            # Disconnect old signal and connect new one for continuous mode
            timer.disconnect()
            timer.timeout.connect(lambda: self._perform_continuous_volume_step(hwnd, direction))
            timer.setSingleShot(False)
            timer.start(int(self._volume_continuous_interval * 1000))
            
        except Exception as e:
            self._logger.debug(f"Failed to switch to continuous volume mode: {e}")

    def _perform_continuous_volume_step(self, hwnd: int, direction: str) -> None:
        """Perform a single step of continuous volume adjustment."""
        try:
            if hwnd not in self._volume_hold_state:
                return
                
            # Get current volume to check bounds
            try:
                current_volume = session_volume.get_session_volume_for_hwnd(hwnd)
                if current_volume is None:
                    return
                    
                # Stop if we've reached the bounds
                if direction == 'up' and current_volume >= 0.99:
                    self._stop_continuous_volume_adjustment(hwnd)
                    return
                elif direction == 'down' and current_volume <= 0.01:
                    self._stop_continuous_volume_adjustment(hwnd)
                    return
                    
            except Exception:
                pass  # Continue with adjustment even if we can't check bounds
            
            # Use progressive steps for smooth continuous adjustment
            # Start with 1% steps, increase to 2% after 1 second for faster traversal
            hold_duration = time.time() - self._volume_hold_state[hwnd]['start_time']
            if hold_duration > 1.0:
                continuous_step = 0.02 if direction == 'up' else -0.02
            else:
                continuous_step = 0.01 if direction == 'up' else -0.01
            
            # Perform the volume adjustment
            try:
                if session_volume.adjust_session_volume_for_hwnd(hwnd, continuous_step):
                    pass  # Success, continue
                else:
                    # If session volume fails, stop continuous adjustment
                    self._stop_continuous_volume_adjustment(hwnd)
            except Exception:
                self._stop_continuous_volume_adjustment(hwnd)
                
        except Exception as e:
            self._logger.debug(f"Continuous volume step failed: {e}")
            self._stop_continuous_volume_adjustment(hwnd)

    def _stop_continuous_volume_adjustment(self, hwnd: int) -> None:
        """Stop continuous volume adjustment for a window."""
        try:
            if hwnd in self._volume_hold_state:
                state = self._volume_hold_state[hwnd]
                timer = state['timer']
                timer.stop()
                timer.deleteLater()
                del self._volume_hold_state[hwnd]
        except Exception as e:
            self._logger.debug(f"Failed to stop continuous volume adjustment: {e}")

    def stop_all_continuous_volume_adjustments(self) -> None:
        """Stop all continuous volume adjustments (cleanup method)."""
        try:
            for hwnd in list(self._volume_hold_state.keys()):
                self._stop_continuous_volume_adjustment(hwnd)
        except Exception as e:
            self._logger.debug(f"Failed to stop all continuous volume adjustments: {e}")

    def handle_volume_key_press(self, hwnd: int, direction: str) -> bool:
        """Handle volume key press with continuous adjustment support.
        
        Args:
            hwnd: Target window handle
            direction: 'up' or 'down'
            
        Returns:
            True if handled successfully
        """
        try:
            # Perform immediate single step
            command = self.APPCOMMAND_VOLUME_UP if direction == 'up' else self.APPCOMMAND_VOLUME_DOWN
            success, _ = self.send_media_command(hwnd, command)
            
            # Start continuous adjustment for key hold
            self._start_continuous_volume_adjustment(hwnd, direction)
            
            return success
        except Exception as e:
            self._logger.debug(f"Volume key press handling failed: {e}")
            return False

    def handle_volume_key_release(self, hwnd: int) -> None:
        """Handle volume key release to stop continuous adjustment."""
        try:
            self._stop_continuous_volume_adjustment(hwnd)
        except Exception as e:
            self._logger.debug(f"Volume key release handling failed: {e}")

    # --- Public helpers ----------------------------------------------------
    def detect_app_for_hwnd(self, hwnd: int) -> Optional[str]:
        """Public wrapper to detect the catalog app name for a given hwnd.

        Returns a lowercase app key (e.g., 'firefox', 'chrome') when recognized,
        otherwise None.
        """
        try:
            return self._detect_app_for_hwnd(hwnd)
        except Exception:
            return None

    def _send_command_for_hwnd(self, hwnd: int, command: int) -> Tuple[bool, str]:
        """Resolve app by hwnd and send the given media command to that exact window."""
        app_name = self._detect_app_for_hwnd(hwnd)
        if not app_name:
            # Permit session-only volume adjustments for unknown apps.
            # This enables per-HWND volume control even when the app is not in the catalog.
            if command in (self.APPCOMMAND_VOLUME_UP, self.APPCOMMAND_VOLUME_DOWN):
                app_name = "unknown"
            else:
                return False, "Target window is not a recognized media app"
        return self._send_media_command_safe(hwnd, command, app_name)
    
    def _send_browser_media_command(self, hwnd: int, command: int) -> bool:
        """Send media command to browser with enhanced child window enumeration."""
        if not WIN_AVAILABLE:
            return False
            
        try:
            # Method 1: Send to all child windows (finds embedded video players)
            child_windows = []
            
            def enum_child_callback(child_hwnd, _):
                child_windows.append(child_hwnd)
                return True
            
            win32gui.EnumChildWindows(hwnd, enum_child_callback, None)
            try:
                self._logger.debug(f"Browser hwnd={hwnd}: enumerated {len(child_windows)} child windows")
            except Exception:
                pass
            
            # Try sending to child windows first (embedded video players)
            for child_hwnd in child_windows:
                if is_process_responsive(child_hwnd, timeout_ms=500):
                    if safe_send_appcommand(child_hwnd, command, timeout_ms=500):
                        try:
                            self._logger.debug(f"Browser hwnd={hwnd}: child hwnd={child_hwnd} accepted appcommand {command}")
                        except Exception:
                            pass
                        return True
                else:
                    try:
                        self._logger.debug(f"Browser hwnd={hwnd}: child hwnd={child_hwnd} unresponsive; skipping")
                    except Exception:
                        pass
            
            return False
        except Exception as e:
            self._logger.debug(f"Browser media command failed: {e}")
            return False
    
    def _send_browser_media_command_targeted(self, target_hwnd: int, command: int) -> bool:
        """Send media command to browser targeting specific window matching DWM overlay content.
        
        This method attempts to match the target_hwnd (from overlay) to the correct browser
        child window, solving the multi-window browser media control issue.
        """
        if not WIN_AVAILABLE:
            return False
            
        try:
            # First, try to send command directly to target_hwnd if it's responsive
            if is_process_responsive(target_hwnd, timeout_ms=500):
                if safe_send_appcommand(target_hwnd, command, timeout_ms=500):
                    try:
                        self._logger.debug(f"Direct command to target hwnd={target_hwnd} succeeded")
                    except Exception:
                        pass
                    return True
            
            # If direct approach fails, find parent browser window and enumerate children
            try:
                # Get the top-level browser window that contains this target
                parent_hwnd = target_hwnd
                while parent_hwnd:
                    parent = win32gui.GetParent(parent_hwnd)
                    if not parent:
                        break
                    parent_hwnd = parent
                
                # Verify this is a browser window
                app_name = self._detect_app_for_hwnd(parent_hwnd)
                if app_name not in ['chrome', 'firefox', 'edge', 'discord']:
                    # Fall back to original method if not a browser
                    return self._send_browser_media_command(target_hwnd, command)
                
                # Enumerate child windows and prioritize those matching target characteristics
                child_windows = []
                target_title = ""
                target_class = ""
                
                try:
                    target_title = win32gui.GetWindowText(target_hwnd).lower()
                    target_class = win32gui.GetClassName(target_hwnd).lower()
                except Exception:
                    pass
                
                def enum_child_callback(child_hwnd, _):
                    child_windows.append(child_hwnd)
                    return True
                
                win32gui.EnumChildWindows(parent_hwnd, enum_child_callback, None)
                
                # Prioritize child windows that match target characteristics
                prioritized_children = []
                other_children = []
                
                for child_hwnd in child_windows:
                    try:
                        child_title = win32gui.GetWindowText(child_hwnd).lower()
                        child_class = win32gui.GetClassName(child_hwnd).lower()
                        
                        # Score based on similarity to target
                        score = 0
                        if target_title and child_title and target_title in child_title:
                            score += 10
                        if target_class and child_class and target_class == child_class:
                            score += 5
                        
                        # Check for media-related content
                        media_keywords = ['youtube', 'netflix', 'spotify', 'twitch', 'vimeo', 'disney', 'prime video', 'hulu', 'plex', 'jellyfin']
                        if any(keyword in child_title for keyword in media_keywords):
                            score += 3
                        
                        if score > 0:
                            prioritized_children.append((score, child_hwnd))
                        else:
                            other_children.append(child_hwnd)
                    except Exception:
                        other_children.append(child_hwnd)
                
                # Sort prioritized children by score (highest first)
                prioritized_children.sort(key=lambda x: x[0], reverse=True)
                ordered_children = [hwnd for _, hwnd in prioritized_children] + other_children
                
                # Try sending command to prioritized children first
                for child_hwnd in ordered_children:
                    if is_process_responsive(child_hwnd, timeout_ms=500):
                        if safe_send_appcommand(child_hwnd, command, timeout_ms=500):
                            try:
                                self._logger.debug(f"Targeted browser command: parent={parent_hwnd} child={child_hwnd} succeeded")
                            except Exception:
                                pass
                            return True
                
                return False
                
            except Exception as e:
                self._logger.debug(f"Targeted browser command failed, falling back: {e}")
                return self._send_browser_media_command(target_hwnd, command)
            
        except Exception as e:
            self._logger.debug(f"Targeted browser media command failed: {e}")
            return False
    
    def _send_browser_media_command_enhanced(self, hwnd: int, command: int, app_name: str) -> bool:
        """Enhanced browser media dispatch with subtle activation & retry.

        Steps:
        - Try child-targeted WM_APPCOMMAND first (avoids global routing)
        - If it fails, request a subtle activation via KeepAlive (no focus steal)
        - Retry child-targeted command after activation
        """
        # First attempt using targeted child routing
        if self._send_browser_media_command_targeted(hwnd, command):
            return True
        # Try subtle activation via KeepAlive and retry
        try:
            from .keepalive import get_media_keepalive  # local import to avoid cycles
            ka = get_media_keepalive()
            if ka and hasattr(ka, 'request_subtle_activation'):
                if ka.request_subtle_activation(hwnd, app_name):
                    # brief delay not to block; rely on async where possible, but a tiny sleep helps
                    time.sleep(0.03)
                    if self._send_browser_media_command(hwnd, command):
                        return True
        except Exception:
            pass
        return False
    
    def _send_hotkey(self, hwnd: int, vk: int) -> bool:
        """Send hotkey to window using PostMessage."""
        return key_press(hwnd, vk, delay_ms=10)
    
    def _send_browser_hotkey(self, hwnd: int, vk: int, include_char: bool = False, char_code: Optional[int] = None) -> bool:
        """Send a hotkey to browser child content windows first, then top-level.

        - If include_char is True, dispatch WM_KEYDOWN + WM_CHAR + WM_KEYUP using
          `key_press_with_char` for printable keys (Firefox-friendly).
        - Otherwise dispatch WM_KEYDOWN/UP only via `key_press` (Chrome/Edge-friendly).
        - This avoids WM_APPCOMMAND default OS handling that can route to MRU apps.
        """
        if not WIN_AVAILABLE:
            return False
        try:
            child_windows: List[int] = []
            try:
                win32gui.EnumChildWindows(hwnd, lambda ch, _: (child_windows.append(ch), True)[1], None)
            except Exception:
                pass
            
            # Prioritize likely media content windows by title keywords
            prioritized: List[int] = []
            others: List[int] = []
            keywords = ['youtube', 'netflix', 'twitch', 'spotify', 'vimeo', 'disney', 'prime video', 'hulu', 'plex', 'jellyfin']
            for ch in child_windows:
                try:
                    title = (win32gui.GetWindowText(ch) or '').lower()
                except Exception:
                    title = ''
                matched = any(k in title for k in keywords) if title else False
                if matched:
                    prioritized.append(ch)
                else:
                    others.append(ch)
            ordered_children = prioritized + others

            # Prefer children
            for ch in ordered_children:
                if is_process_responsive(ch, timeout_ms=250):
                    if include_char:
                        if key_press_with_char(ch, vk, char_code=(char_code if char_code is not None else (vk & 0xFF)), delay_ms=10):
                            try:
                                self._logger.debug(f"Browser hwnd={hwnd}: child hwnd={ch} accepted vk={vk} with WM_CHAR")
                            except Exception:
                                pass
                            return True
                    elif self._send_hotkey(ch, vk):
                        try:
                            self._logger.debug(f"Browser hwnd={hwnd}: child hwnd={ch} accepted hotkey vk={vk}")
                        except Exception:
                            pass
                        return True
            # Fallback to top-level hotkey
            if is_process_responsive(hwnd, timeout_ms=250):
                if include_char:
                    if key_press_with_char(hwnd, vk, char_code=(char_code if char_code is not None else (vk & 0xFF)), delay_ms=10):
                        try:
                            self._logger.debug(f"Browser hwnd={hwnd}: top-level accepted vk={vk} with WM_CHAR")
                        except Exception:
                            pass
                        return True
                elif self._send_hotkey(hwnd, vk):
                    try:
                        self._logger.debug(f"Browser hwnd={hwnd}: top-level accepted hotkey vk={vk}")
                    except Exception:
                        pass
                    return True
        except Exception as e:
            self._logger.debug(f"Browser hotkey failed: {e}")
        return False
    
    def _send_media_command_safe(self, hwnd: int, command: int, app_name: str) -> Tuple[bool, str]:
        """Safely send media command based on app type with enhanced crash protection."""
        app_info = self._apps.get(app_name, MediaApp(process="unknown"))
        safe_methods = app_info.safe_methods
        is_crash_prone = app_info.crash_prone
        # Removed noisy debug logging for media dispatch
        
        # First check if the window is responsive (crash protection)
        if not is_process_responsive(hwnd, timeout_ms=1000):
            return False, f"{app_name} is not responsive"
        
        # Per-app session volume (preferred) — handle volume first regardless of crash-prone
        if command in (self.APPCOMMAND_VOLUME_UP, self.APPCOMMAND_VOLUME_DOWN):
            # Configurable per-step volume delta (default 2% for finer control)
            try:
                cfg_step = float(self._settings.get("media.volume_step", 0.02))
            except Exception:
                cfg_step = 0.02
            # Clamp to safe range [0.01, 0.25]
            if cfg_step < 0.01:
                cfg_step = 0.01
            elif cfg_step > 0.25:
                cfg_step = 0.25
            step = cfg_step if command == self.APPCOMMAND_VOLUME_UP else -cfg_step
            reason = 'up' if step > 0 else 'down'
            # Try per-app session volume via PyCAW (if available)
            try:
                if session_volume.adjust_session_volume_for_hwnd(hwnd, step):
                    # Get exact level when available
                    try:
                        level = session_volume.get_session_volume_for_hwnd(hwnd)
                    except Exception:
                        level = None
                    # Get window title for better OSD display
                    window_title = None
                    try:
                        if WIN_AVAILABLE:
                            from utils.window_validation import get_window_title
                            window_title = get_window_title(hwnd)
                    except Exception:
                        pass
                    
                    # Use window title if app_name is unknown
                    display_name = app_name
                    if app_name == "unknown" and window_title:
                        display_name = window_title
                    
                    self._publish(
                        'media.volume.changed',
                        {
                            'hwnd': hwnd,
                            'app_name': display_name,
                            'window_title': window_title,
                            'level': level,
                            'volume': level,
                            'source': 'session',
                            'reason': reason,
                        },
                    )
                    return True, f"Adjusted session volume for {app_name} ({reason})"
            except Exception:
                # Already logged in session_volume
                pass
            # Simplified policy: session-only routing; no hotkey or global mixer fallback
            try:
                self._logger.debug(f"Volume routing policy: session-only; no fallback for {app_name} ({reason})")
            except Exception:
                pass
            return False, f"Session volume unavailable for {app_name}"

        # For crash-prone apps, use hotkeys only (non-volume commands)
        if is_crash_prone or 'hotkeys_only' in safe_methods:
            action_map = {
                self.APPCOMMAND_MEDIA_PLAY_PAUSE: 'play_pause',
                self.APPCOMMAND_MEDIA_NEXTTRACK: 'next',
                self.APPCOMMAND_MEDIA_PREVIOUSTRACK: 'previous',
                self.APPCOMMAND_MEDIA_STOP: 'stop',
            }
            
            action = action_map.get(command)
            if not action:
                return False, f"No hotkey mapping for command {command}"
            
            # Try primary hotkey
            if action in app_info.hotkeys:
                vk = app_info.hotkeys[action]
                if self._send_hotkey(hwnd, vk):
                    return True, f"Sent {action} hotkey to {app_name}"
            
            # Try alternative hotkey (for MPC players)
            alt_action = f"{action}_alt"
            if alt_action in app_info.hotkeys:
                vk = app_info.hotkeys[alt_action]
                if self._send_hotkey(hwnd, vk):
                    return True, f"Sent {action} alt hotkey to {app_name}"
            
            return False, f"Hotkey method failed for {app_name}"
        
        # Browser-specific routing for play/pause and next/previous per app behavior
        if app_name in ['chrome', 'edge', 'firefox']:
            try:
                VK_SPACE = win32con.VK_SPACE if WIN_AVAILABLE else 0x20
                VK_LEFT = win32con.VK_LEFT if WIN_AVAILABLE else 0x25
                VK_RIGHT = win32con.VK_RIGHT if WIN_AVAILABLE else 0x27
                VK_K = 0x4B  # 'K' key
                if command == self.APPCOMMAND_MEDIA_PLAY_PAUSE:
                    if app_name in ['chrome', 'edge']:
                        # Prefer 'K' via WM_KEYDOWN/UP only; fallback to SPACE without WM_CHAR
                        if (
                            self._send_browser_hotkey(hwnd, VK_K, include_char=False)
                            or self._send_browser_hotkey(hwnd, VK_SPACE, include_char=False)
                        ):
                            return True, f"Browser {app_name}: play/pause via key"
                    else:  # firefox
                        # Include WM_CHAR for printable keys
                        if (
                            self._send_browser_hotkey(hwnd, VK_K, include_char=True, char_code=ord('k'))
                            or self._send_browser_hotkey(hwnd, VK_SPACE, include_char=True, char_code=0x20)
                        ):
                            return True, "Firefox: play/pause via key (WM_CHAR)"
                elif command == self.APPCOMMAND_MEDIA_NEXTTRACK:
                    if self._send_browser_hotkey(hwnd, VK_RIGHT, include_char=False):
                        return True, f"{app_name}: next via RightArrow"
                elif command == self.APPCOMMAND_MEDIA_PREVIOUSTRACK:
                    if self._send_browser_hotkey(hwnd, VK_LEFT, include_char=False):
                        return True, f"{app_name}: previous via LeftArrow"
            except Exception:
                pass

        # For browsers, use enhanced child-only routing + subtle activation on failure
        if app_name in ['chrome', 'firefox', 'edge', 'discord']:
            if self._send_browser_media_command_enhanced(hwnd, command, app_name):
                # Hint media activity to KeepAlive to keep background media responsive
                try:
                    from .keepalive import get_media_keepalive  # lazy import to avoid cycles
                    ka = get_media_keepalive()
                    if ka and hasattr(ka, 'hint_media_activity'):
                        ka.hint_media_activity(hwnd)
                except Exception:
                    pass
                return True, f"Browser media command sent to {app_name}"
        
        # WM_COMMAND support for MPC variants when IDs are provided via settings
        if 'wm_command' in safe_methods and command in (
            self.APPCOMMAND_MEDIA_PLAY_PAUSE,
            self.APPCOMMAND_MEDIA_NEXTTRACK,
            self.APPCOMMAND_MEDIA_PREVIOUSTRACK,
            self.APPCOMMAND_MEDIA_STOP,
        ):
            action_map = {
                self.APPCOMMAND_MEDIA_PLAY_PAUSE: 'play_pause',
                self.APPCOMMAND_MEDIA_NEXTTRACK: 'next',
                self.APPCOMMAND_MEDIA_PREVIOUSTRACK: 'previous',
                self.APPCOMMAND_MEDIA_STOP: 'stop',
            }
            action = action_map.get(command)
            # Settings override first, then per-app default commands map
            ids = self._settings.get(f'media.wm_command_ids.{app_name}', {}) or {}
            cmd_id = ids.get(action) or app_info.commands.get(action)
            if isinstance(cmd_id, int) and cmd_id > 0:
                if send_wm_command(hwnd, cmd_id, timeout_ms=750):
                    return True, f"WM_COMMAND {cmd_id} ({action}) sent to {app_name}"
            # If no command id available, continue to other methods

        # Hotkeys for apps that support them (non-crash-prone or as a fallback)
        # Try for play/pause/next/previous/stop when declared in app hotkeys
        if 'hotkeys' in safe_methods and command in (
            self.APPCOMMAND_MEDIA_PLAY_PAUSE,
            self.APPCOMMAND_MEDIA_NEXTTRACK,
            self.APPCOMMAND_MEDIA_PREVIOUSTRACK,
            self.APPCOMMAND_MEDIA_STOP,
        ):
            action_map = {
                self.APPCOMMAND_MEDIA_PLAY_PAUSE: 'play_pause',
                self.APPCOMMAND_MEDIA_NEXTTRACK: 'next',
                self.APPCOMMAND_MEDIA_PREVIOUSTRACK: 'previous',
                self.APPCOMMAND_MEDIA_STOP: 'stop',
            }
            action = action_map.get(command)
            if action:
                # Try primary hotkey
                if action in app_info.hotkeys:
                    vk = app_info.hotkeys[action]
                    if self._send_hotkey(hwnd, vk):
                        return True, f"Sent {action} hotkey to {app_name}"
                # Try alternative hotkey (for MPC players)
                alt_action = f"{action}_alt"
                if alt_action in app_info.hotkeys:
                    vk = app_info.hotkeys[alt_action]
                    if self._send_hotkey(hwnd, vk):
                        return True, f"Sent {action} alt hotkey to {app_name}"

        # Standard media command for safe apps (except volume; see above)
        if 'media_command' in safe_methods:
            if command not in (self.APPCOMMAND_VOLUME_UP, self.APPCOMMAND_VOLUME_DOWN):
                if safe_send_appcommand(hwnd, command, timeout_ms=2000):
                    # Hint media activity on successful command
                    try:
                        from .keepalive import get_media_keepalive
                        ka = get_media_keepalive()
                        if ka and hasattr(ka, 'hint_media_activity'):
                            ka.hint_media_activity(hwnd)
                    except Exception:
                        pass
                    return True, f"Media command sent to {app_name}"
        
        # Spacebar fallback (non-browser apps) for play/pause only
        if ('spacebar' in safe_methods and 
            command == self.APPCOMMAND_MEDIA_PLAY_PAUSE):
            # Send WM_CHAR-aware spacebar for better compatibility
            vk_space = win32con.VK_SPACE if WIN_AVAILABLE else 0x20
            if key_press_with_char(hwnd, vk_space, char_code=0x20, delay_ms=10):
                try:
                    from .keepalive import get_media_keepalive
                    ka = get_media_keepalive()
                    if ka and hasattr(ka, 'hint_media_activity'):
                        ka.hint_media_activity(hwnd)
                except Exception:
                    pass
                return True, f"Spacebar sent to {app_name}"
        
        return False, f"All methods failed for {app_name}"
    
    
    def play_pause(self, app: Optional[str] = None) -> Tuple[bool, str]:
        """Play/pause media in specified app or any available app."""
        if app:
            hwnd = self._find_window_by_app(app.lower())
            if not hwnd:
                return False, f"Could not find {app}"
            
            return self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PLAY_PAUSE, app.lower())
        
        # Try to get preferred app first (based on overlay context)
        preferred_app = self.get_preferred_app()
        if preferred_app:
            hwnd = self._find_window_by_app(preferred_app)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PLAY_PAUSE, preferred_app)
                if success:
                    return True, msg
        
        # Try all available media apps as fallback
        running_apps = self.list_running_apps()
        if not running_apps:
            return False, "No media applications found"
        
        for app_name in running_apps:
            if app_name == preferred_app:
                continue  # Already tried above
            hwnd = self._find_window_by_app(app_name)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PLAY_PAUSE, app_name)
                if success:
                    return True, msg
        
        return False, "Failed to control any media application"

    # HWND-targeted control APIs (preferred when a capture target exists)
    def play_pause_for_hwnd(self, hwnd: int) -> Tuple[bool, str]:
        return self._send_command_for_hwnd(hwnd, self.APPCOMMAND_MEDIA_PLAY_PAUSE)
    
    def next_track(self, app: Optional[str] = None) -> Tuple[bool, str]:
        """Skip to next track."""
        if app:
            hwnd = self._find_window_by_app(app.lower())
            if not hwnd:
                return False, f"Could not find {app}"
            
            return self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_NEXTTRACK, app.lower())
        
        # Try preferred app first
        preferred_app = self.get_preferred_app()
        if preferred_app:
            hwnd = self._find_window_by_app(preferred_app)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_NEXTTRACK, preferred_app)
                if success:
                    return True, msg
        
        running_apps = self.list_running_apps()
        for app_name in running_apps:
            if app_name == preferred_app:
                continue
            hwnd = self._find_window_by_app(app_name)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_NEXTTRACK, app_name)
                if success:
                    return True, msg
        
        return False, "Failed to skip in any application"

    def next_for_hwnd(self, hwnd: int) -> Tuple[bool, str]:
        return self._send_command_for_hwnd(hwnd, self.APPCOMMAND_MEDIA_NEXTTRACK)
    
    def previous_track(self, app: Optional[str] = None) -> Tuple[bool, str]:
        """Skip to previous track."""
        if app:
            hwnd = self._find_window_by_app(app.lower())
            if not hwnd:
                return False, f"Could not find {app}"
            
            return self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PREVIOUSTRACK, app.lower())
        
        # Try preferred app first
        preferred_app = self.get_preferred_app()
        if preferred_app:
            hwnd = self._find_window_by_app(preferred_app)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PREVIOUSTRACK, preferred_app)
                if success:
                    return True, msg
        
        running_apps = self.list_running_apps()
        for app_name in running_apps:
            if app_name == preferred_app:
                continue
            hwnd = self._find_window_by_app(app_name)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PREVIOUSTRACK, app_name)
                if success:
                    return True, msg
        
        return False, "Failed to skip back in any application"

    def previous_for_hwnd(self, hwnd: int) -> Tuple[bool, str]:
        return self._send_command_for_hwnd(hwnd, self.APPCOMMAND_MEDIA_PREVIOUSTRACK)

    # Aliases expected by KeyPassthroughController
    def next(self, app: Optional[str] = None) -> Tuple[bool, str]:
        return self.next_track(app)

    def previous(self, app: Optional[str] = None) -> Tuple[bool, str]:
        return self.previous_track(app)
    
    def stop(self, app: Optional[str] = None) -> Tuple[bool, str]:
        """Stop media playback."""
        if app:
            hwnd = self._find_window_by_app(app.lower())
            if not hwnd:
                return False, f"Could not find {app}"
            
            return self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_STOP, app.lower())
        
        running_apps = self.list_running_apps()
        for app_name in running_apps:
            hwnd = self._find_window_by_app(app_name)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_STOP, app_name)
                if success:
                    return True, msg
        
        return False, "Failed to stop any application"

    def stop_for_hwnd(self, hwnd: int) -> Tuple[bool, str]:
        return self._send_command_for_hwnd(hwnd, self.APPCOMMAND_MEDIA_STOP)

    def volume_up(self, app: Optional[str] = None) -> Tuple[bool, str]:
        """Increase volume in the media application (prefers app-local handling)."""
        if app:
            hwnd = self._find_window_by_app(app.lower())
            if not hwnd:
                return False, f"Could not find {app}"
            return self._send_media_command_safe(hwnd, self.APPCOMMAND_VOLUME_UP, app.lower())
        
        # Try preferred app first
        preferred_app = self.get_preferred_app()
        if preferred_app:
            hwnd = self._find_window_by_app(preferred_app)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_VOLUME_UP, preferred_app)
                if success:
                    return True, msg
        
        running_apps = self.list_running_apps()
        for app_name in running_apps:
            if app_name == preferred_app:
                continue
            hwnd = self._find_window_by_app(app_name)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_VOLUME_UP, app_name)
                if success:
                    return True, msg
        return False, "Failed to adjust volume up in any application"

    def volume_up_for_hwnd(self, hwnd: int) -> Tuple[bool, str]:
        return self._send_command_for_hwnd(hwnd, self.APPCOMMAND_VOLUME_UP)

    def volume_down(self, app: Optional[str] = None) -> Tuple[bool, str]:
        """Decrease volume in the media application (prefers app-local handling)."""
        if app:
            hwnd = self._find_window_by_app(app.lower())
            if not hwnd:
                return False, f"Could not find {app}"
            return self._send_media_command_safe(hwnd, self.APPCOMMAND_VOLUME_DOWN, app.lower())
        
        # Try preferred app first
        preferred_app = self.get_preferred_app()
        if preferred_app:
            hwnd = self._find_window_by_app(preferred_app)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_VOLUME_DOWN, preferred_app)
                if success:
                    return True, msg
        
        running_apps = self.list_running_apps()
        for app_name in running_apps:
            if app_name == preferred_app:
                continue
            hwnd = self._find_window_by_app(app_name)
            if hwnd:
                success, msg = self._send_media_command_safe(hwnd, self.APPCOMMAND_VOLUME_DOWN, app_name)
                if success:
                    return True, msg
        return False, "Failed to adjust volume down in any application"

    def volume_down_for_hwnd(self, hwnd: int) -> Tuple[bool, str]:
        return self._send_command_for_hwnd(hwnd, self.APPCOMMAND_VOLUME_DOWN)
    
    def get_session_volume_for_hwnd(self, hwnd: int) -> Optional[float]:
        """Return current per-app session volume for hwnd when available.

        Returns a float in [0.0, 1.0] or None if unavailable.
        """
        try:
            return session_volume.get_session_volume_for_hwnd(hwnd)
        except Exception:
            return None
    
    def list_running_apps(self) -> List[str]:
        """List all currently running media applications."""
        running = []
        for app_name in self._apps:
            if self._find_window_by_app(app_name):
                running.append(app_name)
        return running

    def get_running_media_apps(self) -> List[Tuple[str, int]]:
        """Return list of (app_name, hwnd) for currently running media apps.

        Used by `core.media.keepalive.MediaPlayerKeepAlive` for monitoring.
        """
        results: List[Tuple[str, int]] = []
        for app_name in self._apps:
            hwnd = self._find_window_by_app(app_name)
            if hwnd:
                results.append((app_name, hwnd))
        return results
    
    def get_preferred_app(self) -> Optional[str]:
        """Get the preferred media app based on current context.
        
        Precise window control only - no fallbacks:
        1. Overlay target window (if valid and recognized)
        2. Return None if no precise target available
        """
        try:
            # Only check overlay target - no fallbacks for precise control
            from core.input import get_key_passthrough_controller
            kp = get_key_passthrough_controller()
            target_hwnd = getattr(kp, '_target_hwnd', None)
            
            if target_hwnd and WIN_AVAILABLE and is_window(target_hwnd):
                detected_app = self._detect_app_for_hwnd(target_hwnd)
                if detected_app:
                    self._logger.debug(f"Precise target app: {detected_app} (hwnd={target_hwnd})")
                    return detected_app
            
            # No fallbacks - return None for precise control
            self._logger.debug("No precise target window available - no fallbacks")
            return None
                
        except Exception as e:
            self._logger.debug(f"Error determining target app: {e}")
            
        return None

    def _has_active_audio_session(self, app_name: str) -> bool:
        """Check if app has active audio sessions."""
        try:
            from utils.audio.session_volume import get_session_volume_for_process
            app = self._apps.get(app_name)
            if not app or not hasattr(app, 'process'):
                return False
            
            # Check if process has audio sessions with volume > 0
            volume = get_session_volume_for_process(app.process)
            return volume is not None and volume > 0.01
        except Exception:
            return False

    def _get_app_with_highest_volume(self) -> Optional[str]:
        """Get app with highest audio session volume."""
        try:
            from utils.audio.session_volume import get_session_volume_for_process
            best_app = None
            highest_volume = 0.0
            
            for app_name, app in self._apps.items():
                if not hasattr(app, 'process'):
                    continue
                try:
                    volume = get_session_volume_for_process(app.process)
                    if volume is not None and volume > highest_volume:
                        highest_volume = volume
                        best_app = app_name
                except Exception:
                    continue
                    
            return best_app if highest_volume > 0.01 else None
        except Exception:
            return None

    def is_enabled(self) -> bool:
        """Check if media control is enabled in settings."""
        return self._settings.get('features.media_control_enabled', False)


# Global instance
_media_controller: Optional[MediaController] = None


def get_media_controller() -> MediaController:
    """Get the global MediaController instance."""
    global _media_controller
    if _media_controller is None:
        from core.settings import get_settings_manager
        settings_manager = get_settings_manager()
        _media_controller = MediaController(settings_manager)
    return _media_controller
