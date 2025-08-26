import win32gui
import win32process
import win32api
import win32con
import psutil
import time
from typing import List, Dict, Optional, Tuple

class UniversalMediaController:
    def __init__(self):
        # Common media player process names and their window classes
        self.media_apps = {
            'spotify': {'process': 'spotify.exe', 'class': None, 'safe_methods': ['media_command', 'spacebar']},
            'chrome': {'process': 'chrome.exe', 'class': 'Chrome_WidgetWin_1', 'safe_methods': ['media_command']},
            'firefox': {'process': 'firefox.exe', 'class': 'MozillaWindowClass', 'safe_methods': ['media_command']},
            'edge': {'process': 'msedge.exe', 'class': 'Chrome_WidgetWin_1', 'safe_methods': ['media_command']},
            'vlc': {'process': 'vlc.exe', 'class': 'Qt5QWindowIcon', 'safe_methods': ['media_command', 'hotkeys']},
            'media_player': {'process': 'wmplayer.exe', 'class': 'WMPlayerApp', 'safe_methods': ['media_command']},
            'itunes': {'process': 'iTunes.exe', 'class': 'iTunes', 'safe_methods': ['media_command']},
            'discord': {'process': 'Discord.exe', 'class': 'Chrome_WidgetWin_1', 'safe_methods': ['media_command']},
            'foobar': {'process': 'foobar2000.exe', 'class': '{E7076D1C-A7BF-4f39-B771-BCBE88F2A2A8}', 'safe_methods': ['media_command', 'hotkeys']},
            'winamp': {'process': 'winamp.exe', 'class': 'Winamp v1.x', 'safe_methods': ['media_command', 'hotkeys']},
            'musicbee': {'process': 'MusicBee.exe', 'class': 'WindowsForms10.Window.8.app.0.2bf8098_r11_ad1', 'safe_methods': ['media_command']},
            # Video players - with special crash protection
            'mpc_hc': {'process': 'mpc-hc.exe', 'class': 'MediaPlayerClassicW', 'safe_methods': ['hotkeys_only'], 'crash_prone': True},
            'mpc_hc64': {'process': 'mpc-hc64.exe', 'class': 'MediaPlayerClassicW', 'safe_methods': ['hotkeys_only'], 'crash_prone': True},
            'mpc_be': {'process': 'mpc-be.exe', 'class': 'MediaPlayerClassicW', 'safe_methods': ['hotkeys_only'], 'crash_prone': True},
            'mpc_be64': {'process': 'mpc-be64.exe', 'class': 'MediaPlayerClassicW', 'safe_methods': ['hotkeys_only'], 'crash_prone': True},
            'mpv_net': {'process': 'mpv.net.exe', 'class': 'mpv.net', 'safe_methods': ['hotkeys_only'], 'crash_prone': True},
            'mpv': {'process': 'mpv.exe', 'class': 'mpv', 'safe_methods': ['hotkeys_only'], 'crash_prone': True},
            'potplayer': {'process': 'PotPlayerMini.exe', 'class': 'PotPlayer', 'safe_methods': ['media_command', 'hotkeys']},
            'potplayer64': {'process': 'PotPlayerMini64.exe', 'class': 'PotPlayer64', 'safe_methods': ['media_command', 'hotkeys']},
            # Additional media players
            'kmplayer': {'process': 'KMPlayer.exe', 'class': 'KMPWnd', 'safe_methods': ['media_command', 'hotkeys']},
            'gom': {'process': 'GOM.exe', 'class': 'GomPlayerWndClass', 'safe_methods': ['media_command', 'hotkeys']},
            'bsplayer': {'process': 'bsplayer.exe', 'class': 'BSPlayer', 'safe_methods': ['media_command', 'hotkeys']},
            'aimp': {'process': 'AIMP.exe', 'class': 'AIMP2_MainForm', 'safe_methods': ['media_command', 'hotkeys']},
            'mediamonkey': {'process': 'MediaMonkey.exe', 'class': 'TMainForm', 'safe_methods': ['media_command']},
            'kodi': {'process': 'Kodi.exe', 'class': 'Kodi', 'safe_methods': ['hotkeys_only'], 'crash_prone': True},
            'plex': {'process': 'PlexMediaPlayer.exe', 'class': 'Qt5QWindowIcon', 'safe_methods': ['media_command', 'spacebar']},
            'jellyfin': {'process': 'JellyfinMediaPlayer.exe', 'class': 'Qt5QWindowIcon', 'safe_methods': ['media_command', 'spacebar']},
        }
        
        # Media control constants
        self.WM_APPCOMMAND = 0x319
        self.APPCOMMAND_MEDIA_PLAY_PAUSE = 14
        self.APPCOMMAND_MEDIA_NEXTTRACK = 11
        self.APPCOMMAND_MEDIA_PREVIOUSTRACK = 12
        self.APPCOMMAND_MEDIA_STOP = 13
        
        # MPC-specific hotkeys (using arrow keys as primary, brackets as fallback)
        self.mpc_hotkeys = {
            'play_pause': win32con.VK_SPACE,
            'next': win32con.VK_RIGHT,      # Right arrow key
            'previous': win32con.VK_LEFT,   # Left arrow key
            'stop': 0x53,  # S key
        }
        
        # Alternative MPC hotkeys (if arrows don't work)
        self.mpc_hotkeys_alt = {
            'play_pause': win32con.VK_SPACE,
            'next': 0xDD,      # ] key (VK_OEM_6)
            'previous': 0xDB,  # [ key (VK_OEM_4)
            'stop': 0x53,      # S key
        }
        
        # VLC hotkeys
        self.vlc_hotkeys = {
            'play_pause': win32con.VK_SPACE,
            'next': 0x4E,  # N key
            'previous': 0x50,  # P key  
            'stop': 0x53,  # S key
        }

    def is_process_responsive(self, hwnd: int, timeout: float = 1.0) -> bool:
        """Check if a window/process is responsive"""
        try:
            # Try a simple, safe message first
            result = win32gui.SendMessageTimeout(
                hwnd, 
                win32con.WM_NULL, 
                0, 0, 
                win32con.SMTO_NORMAL, 
                int(timeout * 1000)
            )
            return result[0] != 0  # Non-zero means success
        except Exception:
            return False

    def safe_send_message(self, hwnd: int, msg: int, wparam: int, lparam: int, timeout: float = 2.0) -> bool:
        """Safely send a message with timeout protection"""
        try:
            # First check if the window is responsive
            if not self.is_process_responsive(hwnd, 0.5):
                return False
                
            # Use SendMessageTimeout for crash protection
            result = win32gui.SendMessageTimeout(
                hwnd, msg, wparam, lparam,
                win32con.SMTO_NORMAL | win32con.SMTO_ABORTIFHUNG,
                int(timeout * 1000)
            )
            return result[0] != 0
        except Exception as e:
            print(f"Safe send message failed: {e}")
            return False

    def safe_post_message(self, hwnd: int, msg: int, wparam: int, lparam: int) -> bool:
        """Safely post a message (non-blocking)"""
        try:
            # Check if window exists and is valid
            if not win32gui.IsWindow(hwnd):
                return False
            
            win32api.PostMessage(hwnd, msg, wparam, lparam)
            return True
        except Exception as e:
            print(f"Safe post message failed: {e}")
            return False

    def send_browser_media_command(self, hwnd: int, command: int) -> bool:
        """Send media command to browser window that works with unfocused video players"""
        try:
            # For browsers, we need to send the command in a way that reaches the video element
            # even when the tab/window isn't focused. This uses a different message routing.
            
            # Method 1: Send to all child windows (finds embedded video players)
            child_windows = []
            def enum_child_callback(child_hwnd, _):
                child_windows.append(child_hwnd)
                return True
            
            win32gui.EnumChildWindows(hwnd, enum_child_callback, None)
            
            # Try sending to child windows first (embedded video players)
            for child_hwnd in child_windows:
                if self.safe_send_message(child_hwnd, self.WM_APPCOMMAND, child_hwnd, command << 16, timeout=0.5):
                    return True
            
            # Method 2: Send with different wparam (sometimes needed for browsers)
            if self.safe_send_message(hwnd, self.WM_APPCOMMAND, 0, command << 16, timeout=1.0):
                return True
                
            # Method 3: Try posting instead of sending (non-blocking)
            if self.safe_post_message(hwnd, self.WM_APPCOMMAND, hwnd, command << 16):
                return True
                
            return False
        except Exception:
            return False

    def send_mpc_hotkey(self, hwnd: int, action: str, use_alt_keys: bool = False) -> bool:
        """Send hotkeys specifically for MPC players"""
        hotkey_set = self.mpc_hotkeys_alt if use_alt_keys else self.mpc_hotkeys
        
        if action not in hotkey_set:
            return False
            
        vk_code = hotkey_set[action]
        
        try:
            # Use PostMessage for hotkeys (safer than SendMessage)
            success1 = self.safe_post_message(hwnd, win32con.WM_KEYDOWN, vk_code, 0)
            time.sleep(0.01)  # Small delay
            success2 = self.safe_post_message(hwnd, win32con.WM_KEYUP, vk_code, 0)
            
            return success1 and success2
        except Exception:
            return False

    def send_vlc_hotkey(self, hwnd: int, action: str) -> bool:
        """Send hotkeys specifically for VLC"""
        if action not in self.vlc_hotkeys:
            return False
            
        vk_code = self.vlc_hotkeys[action]
        
        try:
            success1 = self.safe_post_message(hwnd, win32con.WM_KEYDOWN, vk_code, 0)
            time.sleep(0.01)
            success2 = self.safe_post_message(hwnd, win32con.WM_KEYUP, vk_code, 0)
            
            return success1 and success2
        except Exception:
            return False

    def find_windows_by_process(self, process_name: str) -> List[int]:
        """Find all windows belonging to a specific process"""
        windows = []
        
        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    process = psutil.Process(pid)
                    if process.name().lower() == process_name.lower():
                        windows.append(hwnd)
                except Exception:
                    pass
            return True
        
        win32gui.EnumWindows(enum_callback, None)
        return windows
    
    def find_window_by_app(self, app_identifier: str) -> Optional[int]:
        """Find the main window for a specific app"""
        app_identifier = app_identifier.lower()
        
        if app_identifier not in self.media_apps:
            return None
        
        process_name = self.media_apps[app_identifier]['process']
        windows = self.find_windows_by_process(process_name)
        
        if not windows:
            return None
        
        # For apps with multiple windows, try to find the main one
        for hwnd in windows:
            try:
                window_text = win32gui.GetWindowText(hwnd)
                
                # Skip empty titles or minimize/close buttons
                if not window_text or len(window_text) < 3:
                    continue
                
                return hwnd
            except Exception:
                continue
        
        # If no specific match, return the first window
        return windows[0] if windows else None
    
    def get_all_media_windows(self) -> Dict[str, int]:
        """Get all currently running media applications"""
        found_apps = {}
        
        for app_name in self.media_apps:
            try:
                hwnd = self.find_window_by_app(app_name)
                if hwnd:
                    found_apps[app_name] = hwnd
            except Exception:
                continue
        
        return found_apps
    
    def send_media_command_safe(self, hwnd: int, command: int, app_name: str) -> Tuple[bool, str]:
        """Safely send media command based on app type"""
        app_info = self.media_apps.get(app_name, {})
        safe_methods = app_info.get('safe_methods', ['media_command'])
        is_crash_prone = app_info.get('crash_prone', False)
        
        # For crash-prone apps, use hotkeys only
        if is_crash_prone or 'hotkeys_only' in safe_methods:
            if 'mpc' in app_name:
                action_map = {
                    self.APPCOMMAND_MEDIA_PLAY_PAUSE: 'play_pause',
                    self.APPCOMMAND_MEDIA_NEXTTRACK: 'next',
                    self.APPCOMMAND_MEDIA_PREVIOUSTRACK: 'previous',
                    self.APPCOMMAND_MEDIA_STOP: 'stop'
                }
                action = action_map.get(command)
                if action:
                    # Try arrow keys first
                    if self.send_mpc_hotkey(hwnd, action, use_alt_keys=False):
                        return True, f"Sent arrow key to {app_name}"
                    # Fallback to bracket keys
                    elif self.send_mpc_hotkey(hwnd, action, use_alt_keys=True):
                        return True, f"Sent bracket key to {app_name}"
                    
            elif 'vlc' in app_name:
                action_map = {
                    self.APPCOMMAND_MEDIA_PLAY_PAUSE: 'play_pause',
                    self.APPCOMMAND_MEDIA_NEXTTRACK: 'next', 
                    self.APPCOMMAND_MEDIA_PREVIOUSTRACK: 'previous',
                    self.APPCOMMAND_MEDIA_STOP: 'stop'
                }
                action = action_map.get(command)
                if action and self.send_vlc_hotkey(hwnd, action):
                    return True, f"Sent hotkey to {app_name}"
            
            return False, f"Hotkey method failed for {app_name}"
        
        # For browsers and other apps, try media commands first
        if 'media_command' in safe_methods:
            # For browsers, use special method that works with unfocused video
            if app_name in ['chrome', 'firefox', 'edge']:
                if self.send_browser_media_command(hwnd, command):
                    return True, f"Browser media command sent to {app_name}"
            else:
                # Regular apps - standard media command
                if self.safe_send_message(hwnd, self.WM_APPCOMMAND, hwnd, command << 16, timeout=1.0):
                    return True, f"Media command sent to {app_name}"
        
        # Fallback to spacebar for certain apps (only for play/pause)
        if 'spacebar' in safe_methods and command == self.APPCOMMAND_MEDIA_PLAY_PAUSE:
            if self.safe_post_message(hwnd, win32con.WM_KEYDOWN, win32con.VK_SPACE, 0):
                time.sleep(0.01)
                self.safe_post_message(hwnd, win32con.WM_KEYUP, win32con.VK_SPACE, 0)
                return True, f"Spacebar sent to {app_name}"
        
        return False, f"All methods failed for {app_name}"

    def play_pause(self, app_identifier: Optional[str] = None) -> Tuple[bool, str]:
        """Play/pause media in specified app or any available app"""
        if app_identifier:
            hwnd = self.find_window_by_app(app_identifier.lower())
            if not hwnd:
                return False, f"Could not find {app_identifier}"
            
            return self.send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PLAY_PAUSE, app_identifier.lower())
        
        else:
            # Try all available media apps
            media_windows = self.get_all_media_windows()
            
            if not media_windows:
                return False, "No media applications found"
            
            for app_name, hwnd in media_windows.items():
                success, msg = self.send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PLAY_PAUSE, app_name)
                if success:
                    return True, msg
            
            return False, "Failed to control any media application"
    
    def next_track(self, app_identifier: Optional[str] = None) -> Tuple[bool, str]:
        """Skip to next track"""
        if app_identifier:
            hwnd = self.find_window_by_app(app_identifier.lower())
            if not hwnd:
                return False, f"Could not find {app_identifier}"
            
            return self.send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_NEXTTRACK, app_identifier.lower())
        
        else:
            media_windows = self.get_all_media_windows()
            for app_name, hwnd in media_windows.items():
                success, msg = self.send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_NEXTTRACK, app_name)
                if success:
                    return True, msg
            
            return False, "Failed to skip in any application"
    
    def previous_track(self, app_identifier: Optional[str] = None) -> Tuple[bool, str]:
        """Skip to previous track"""
        if app_identifier:
            hwnd = self.find_window_by_app(app_identifier.lower())
            if not hwnd:
                return False, f"Could not find {app_identifier}"
            
            return self.send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PREVIOUSTRACK, app_identifier.lower())
        
        else:
            media_windows = self.get_all_media_windows()
            for app_name, hwnd in media_windows.items():
                success, msg = self.send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_PREVIOUSTRACK, app_name)
                if success:
                    return True, msg
            
            return False, "Failed to skip back in any application"
    
    def stop(self, app_identifier: Optional[str] = None) -> Tuple[bool, str]:
        """Stop media playback"""
        if app_identifier:
            hwnd = self.find_window_by_app(app_identifier.lower())
            if not hwnd:
                return False, f"Could not find {app_identifier}"
            
            return self.send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_STOP, app_identifier.lower())
        
        else:
            media_windows = self.get_all_media_windows()
            for app_name, hwnd in media_windows.items():
                success, msg = self.send_media_command_safe(hwnd, self.APPCOMMAND_MEDIA_STOP, app_name)
                if success:
                    return True, msg
            
            return False, "Failed to stop any application"
    
    def list_running_media_apps(self) -> List[str]:
        """List all currently running media applications"""
        return list(self.get_all_media_windows().keys())


# Global instance for easy access
_controller = UniversalMediaController()

# Simple wrapper functions for easy integration
def toggle_play_pause(app: str = None) -> Tuple[bool, str]:
    """Toggle play/pause for specified app or any running media app"""
    return _controller.play_pause(app)

def next_track(app: str = None) -> Tuple[bool, str]:
    """Skip to next track"""
    return _controller.next_track(app)

def previous_track(app: str = None) -> Tuple[bool, str]:
    """Skip to previous track"""
    return _controller.previous_track(app)

def stop_media(app: str = None) -> Tuple[bool, str]:
    """Stop media playback"""
    return _controller.stop(app)

def list_media_apps() -> List[str]:
    """List all running media applications"""
    return _controller.list_running_media_apps()


# Example usage
if __name__ == "__main__":
    print("=== Enhanced Universal Media Controller ===")
    
    # List running media apps
    running_apps = list_media_apps()
    print(f"Running media apps: {running_apps}")
    
    if running_apps:
        # Test play/pause on any app
        success, msg = toggle_play_pause()
        print(f"Toggle any: {msg} ({'✓' if success else '✗'})")
        
        # Test specific apps
        for app in ['spotify', 'chrome', 'mpc_be', 'vlc']:
            success, msg = toggle_play_pause(app)
            print(f"Toggle {app}: {msg} ({'✓' if success else '✗'})")
    else:
        print("No media applications running")
    
    print(f"\nCrash-protected apps: {[app for app, info in _controller.media_apps.items() if info.get('crash_prone')]}")
    print("MPC hotkeys: Arrow keys (Left/Right) with bracket fallback ([/])")
    print("Browser support: Direct window messaging for unfocused video control")