#!/usr/bin/env python3
"""
Common constants used across multiple files in Shitty PiP
"""
from PySide6.QtGui import QColor


# --- Global Default Constants ---
DEFAULT_POSITION_PRESET = "Top Left"
DEFAULT_WINDOW_OVERLAY_WIDTH = 640
DEFAULT_WINDOW_OVERLAY_HEIGHT = 360
DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR = 0.30  # 30% of screen width
DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR = 0.30 # 30% of screen height
# --- End Global Default Constants ---

# --- UI Constants ---
# FocusIndicatorWidget constants
FOCUS_INDICATOR_SIZE = 20
FOCUS_INDICATOR_MARGIN = 10
FOCUS_INDICATOR_OPACITY = 0.75
FOCUS_INDICATOR_SIZE_RATIO = 0.05

# BorderWidget constants
BORDER_WIDTH = 2
EDGE_MARGIN = 10
DOUBLE_CLICK_INTERVAL_MS = 300  # ms between clicks for double-click
DEFAULT_BORDER_OPACITY = 1.0  # Fully opaque

# --- End UI Constants ---

# --- Windows API Constants ---
class Win32Constants:
    """Windows API constants organized by category."""
    
    class Monitor:
        """Monitor-related constants."""
        MONITOR_DEFAULTTONEAREST = 0x00000002
        MONITOR_DEFAULTTOPRIMARY = 0x00000001
        MONITOR_DEFAULTTONULL = 0x00000000
        MONITORINFOF_PRIMARY = 0x00000001
        
    class SystemMetrics:
        """System metrics constants."""
        SM_XVIRTUALSCREEN = 76
        SM_YVIRTUALSCREEN = 77
        SM_CXVIRTUALSCREEN = 78
        SM_CYVIRTUALSCREEN = 79
        SM_CXSCREEN = 0       # Screen width in pixels
        SM_CYSCREEN = 1       # Screen height in pixels
        SM_SWAPBUTTON = 23    # Non-zero if mouse buttons are swapped
        
    class DPI:
        """DPI-related constants."""
        LOGPIXELSX = 88
        LOGPIXELSY = 90
        
    class WindowLong:
        """Window Long index constants for Get/SetWindowLong."""
        GWL_WNDPROC = -4
        GWL_HINSTANCE = -6
        GWL_HWNDPARENT = -8
        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        GWL_USERDATA = -21
        GWL_ID = -12
        
    class WindowStyle:
        """Window style constants."""
        WS_OVERLAPPED = 0x00000000
        WS_POPUP = 0x80000000
        WS_CHILD = 0x40000000
        WS_MINIMIZE = 0x20000000
        WS_VISIBLE = 0x10000000
        WS_DISABLED = 0x08000000
        WS_CLIPSIBLINGS = 0x04000000
        WS_CLIPCHILDREN = 0x02000000
        WS_MAXIMIZE = 0x01000000
        WS_CAPTION = 0x00C00000
        WS_BORDER = 0x00800000
        WS_DLGFRAME = 0x00400000
        WS_VSCROLL = 0x00200000
        WS_HSCROLL = 0x00100000
        WS_SYSMENU = 0x00080000
        WS_THICKFRAME = 0x00040000
        WS_GROUP = 0x00020000
        WS_TABSTOP = 0x00010000
        
    class ExtendedWindowStyle:
        """Extended window style constants."""
        WS_EX_DLGMODALFRAME = 0x00000001
        WS_EX_NOPARENTNOTIFY = 0x00000004
        WS_EX_TOPMOST = 0x00000008
        WS_EX_ACCEPTFILES = 0x00000010
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_MDICHILD = 0x00000040
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_WINDOWEDGE = 0x00000100
        WS_EX_CLIENTEDGE = 0x00000200
        WS_EX_CONTEXTHELP = 0x00000400
        WS_EX_RIGHT = 0x00001000
        WS_EX_RTLREADING = 0x00002000
        WS_EX_LEFTSCROLLBAR = 0x00004000
        WS_EX_CONTROLPARENT = 0x00010000
        WS_EX_APPWINDOW = 0x00040000
        WS_EX_LAYERED = 0x00080000
        WS_EX_NOINHERITLAYOUT = 0x00100000
        WS_EX_LAYOUTRTL = 0x00400000
        WS_EX_NOACTIVATE = 0x08000000
        
    class ShowWindowCommands:
        """ShowWindow command constants."""
        SW_HIDE = 0
        SW_SHOWNORMAL = 1
        SW_NORMAL = 1
        SW_SHOWMINIMIZED = 2
        SW_SHOWMAXIMIZED = 3
        SW_MAXIMIZE = 3
        SW_SHOWNOACTIVATE = 4
        SW_SHOW = 5
        SW_MINIMIZE = 6
        SW_SHOWMINNOACTIVE = 7
        SW_SHOWNA = 8
        SW_RESTORE = 9
        SW_SHOWDEFAULT = 10
        SW_FORCEMINIMIZE = 11
        SW_MAX = 11
        
    class RedrawWindowFlags:
        """RedrawWindow flags."""
        RDW_INVALIDATE = 0x0001
        RDW_INTERNALPAINT = 0x0002
        RDW_ERASE = 0x0004
        RDW_VALIDATE = 0x0008
        RDW_NOINTERNALPAINT = 0x0010
        RDW_NOERASE = 0x0020
        RDW_NOCHILDREN = 0x0040
        RDW_ALLCHILDREN = 0x0080
        RDW_UPDATENOW = 0x0100
        RDW_ERASENOW = 0x0200
        RDW_FRAME = 0x0400
        RDW_NOFRAME = 0x0800

# Backward compatibility aliases
MONITOR_DEFAULTTONEAREST = Win32Constants.Monitor.MONITOR_DEFAULTTONEAREST
MONITOR_DEFAULTTOPRIMARY = Win32Constants.Monitor.MONITOR_DEFAULTTOPRIMARY
MONITOR_DEFAULTTONULL = Win32Constants.Monitor.MONITOR_DEFAULTTONULL
MONITORINFOF_PRIMARY = Win32Constants.Monitor.MONITORINFOF_PRIMARY

SM_XVIRTUALSCREEN = Win32Constants.SystemMetrics.SM_XVIRTUALSCREEN
SM_YVIRTUALSCREEN = Win32Constants.SystemMetrics.SM_YVIRTUALSCREEN
SM_CXVIRTUALSCREEN = Win32Constants.SystemMetrics.SM_CXVIRTUALSCREEN
SM_CYVIRTUALSCREEN = Win32Constants.SystemMetrics.SM_CYVIRTUALSCREEN
SM_CXSCREEN = Win32Constants.SystemMetrics.SM_CXSCREEN
SM_CYSCREEN = Win32Constants.SystemMetrics.SM_CYSCREEN
SM_SWAPBUTTON = Win32Constants.SystemMetrics.SM_SWAPBUTTON

LOGPIXELSX = Win32Constants.DPI.LOGPIXELSX
LOGPIXELSY = Win32Constants.DPI.LOGPIXELSY

GWL_STYLE = Win32Constants.WindowLong.GWL_STYLE
GWL_EXSTYLE = Win32Constants.WindowLong.GWL_EXSTYLE

WS_DISABLED = Win32Constants.WindowStyle.WS_DISABLED
WS_VISIBLE = Win32Constants.WindowStyle.WS_VISIBLE
WS_EX_TOOLWINDOW = Win32Constants.ExtendedWindowStyle.WS_EX_TOOLWINDOW
WS_EX_NOACTIVATE = Win32Constants.ExtendedWindowStyle.WS_EX_NOACTIVATE

SW_RESTORE = Win32Constants.ShowWindowCommands.SW_RESTORE
SW_SHOW = Win32Constants.ShowWindowCommands.SW_SHOW

RDW_INVALIDATE = Win32Constants.RedrawWindowFlags.RDW_INVALIDATE
RDW_ERASE = Win32Constants.RedrawWindowFlags.RDW_ERASE
RDW_ALLCHILDREN = Win32Constants.RedrawWindowFlags.RDW_ALLCHILDREN
# --- End Windows API Constants ---

# --- Platform and Debug Constants ---
import sys

# Platform detection
WIN32_AVAILABLE = sys.platform == 'win32'
DEBUG_MODE = True  # Set to False in production

# --- Resource Paths ---
THEMES_DIR = ":/themes"
RESOURCES_DIR = ":/Resources"
# --- End Resource Paths ---

# --- DPI Constants ---
MDT_EFFECTIVE_DPI = 0
MDT_ANGULAR_DPI = 1
MDT_RAW_DPI = 2
# --- End DPI Constants ---

# --- Windows Input Constants ---
class WindowMessages:
    """Windows API message constants."""
    WM_ACTIVATE = 0x0006
    WA_CLICKACTIVE = 2
    WM_SETFOCUS = 0x0007
    WM_KILLFOCUS = 0x0008
    WM_SYSCOMMAND = 0x0112
    SC_HOTKEY = 0xF150
    WH_KEYBOARD_LL = 13
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_SYSKEYDOWN = 0x0104
    WM_SYSKEYUP = 0x0105

class InputType:
    """Input type constants for Windows API."""
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_SCANCODE = 0x0008

class AppType:
    """Application type constants for key passthrough handling."""
    OTHER = 0
    FIREFOX = 1
    CHROMIUM = 2
    MEDIA = 3
    GAME = 4

# Common key names for virtual key codes
KEY_NAMES = {
    0x20: 'SPACE',    # VK_SPACE
    0x0D: 'ENTER',    # VK_RETURN
    0x25: 'LEFT',     # VK_LEFT
    0x27: 'RIGHT'     # VK_RIGHT
}

# Windows virtual key constants
VK_SPACE = 0x20
VK_RETURN = 0x0D
VK_LEFT = 0x25
VK_RIGHT = 0x27
# --- End Windows Input Constants ---

# --- Application Constants ---
# Application metadata
APP_NAME = "Shitty PiP QuickSwap"
APP_VERSION = "1.0.0"
ORGANIZATION_NAME = "PiPOverlay"

# Application settings defaults
DEFAULT_THEME = 'dark'
DEFAULT_OPACITY = 100
DEFAULT_WINDOW_SORT_ORDER = 'Most Recently Active'
MAX_MRU_ITEMS = 50  # Maximum number of Most Recently Used items

# Platform-specific flags
WIN32_AVAILABLE = False  # Will be set to True on Windows platforms

# --- End Application Constants ---

# --- Theme Constants ---
class ThemeColors:
    """Centralized theme color definitions for consistent theming across the application."""
    
    # Dark Theme
    DARK = {
        'name': 'dark',
        'primary': QColor(0, 120, 215),  # Blue accent
        'background': QColor(30, 30, 30),
        'foreground': QColor(240, 240, 240),
        'border': QColor(40, 40, 40),
        'fill': QColor(20, 20, 20),
        'highlight': QColor(0, 102, 204),
        'button': {
            'background': QColor(68, 68, 68),
            'text': QColor(255, 255, 255),
            'border': QColor(119, 119, 119),
            'hover': QColor(85, 85, 85),
            'pressed': QColor(51, 51, 51)
        },
        'menu': {
            'background': QColor(30, 30, 30),
            'text': QColor(240, 240, 240),
            'border': QColor(40, 40, 40),
            'highlight': QColor(0, 120, 215)
        }
    }
    
    # Light Theme
    LIGHT = {
        'name': 'light',
        'primary': QColor(0, 102, 204),  # Slightly darker blue for better contrast
        'background': QColor(240, 240, 240),
        'foreground': QColor(30, 30, 30),
        'border': QColor(180, 180, 180),
        'fill': QColor(200, 200, 200),
        'highlight': QColor(0, 122, 204),
        'button': {
            'background': QColor(224, 224, 224),  # ~30% lighter than dark theme
            'text': QColor(0, 0, 0),  # Black text
            'border': QColor(0, 0, 0),  # Black border
            'hover': QColor(240, 240, 240),
            'pressed': QColor(208, 208, 208)
        },
        'menu': {
            'background': QColor(240, 240, 240),
            'text': QColor(30, 30, 30),
            'border': QColor(180, 180, 180),
            'highlight': QColor(0, 102, 204)
        }
    }
    
    @classmethod
    def get_theme_colors(cls, theme_name):
        """Get the color dictionary for the specified theme."""
        theme_name = theme_name.lower() if theme_name else 'dark'
        return cls.DARK if theme_name == 'dark' else cls.LIGHT
    
    @classmethod
    def get_theme_stylesheet(cls, theme_name):
        """Get a stylesheet string for the specified theme."""
        # Prevent Nuitka from removing unused parameter

        colors = cls.get_theme_colors(theme_name)
        menu = colors['menu']
        
        # Base styles for menus
        base_styles = f"""
            QMenu {{
                background-color: rgb({menu['background'].red()}, {menu['background'].green()}, {menu['background'].blue()});
                color: rgb({menu['text'].red()}, {menu['text'].green()}, {menu['text'].blue()});
                border: 1px solid rgb({menu['border'].red()}, {menu['border'].green()}, {menu['border'].blue()});
                padding: 5px;
            }}
            QMenu::item:selected {{
                background-color: rgb({menu['highlight'].red()}, {menu['highlight'].green()}, {menu['highlight'].blue()});
            }}
            QMenu::item:disabled {{
                color: gray;
            }}
            QMenu::separator {{
                height: 1px;
                background: rgb({menu['border'].red()}, {menu['border'].green()}, {menu['border'].blue()});
                margin: 4px 8px;
            }}
        """
        
        # Try to load the full theme stylesheet from resources
        try:
            from PySide6.QtCore import QFile, QTextStream, QIODevice
            
            # Map theme name to resource path
            theme_map = {
                'dark': ':/themes/dark.qss',
                'light': ':/themes/light.qss'
            }
            
            theme_path = theme_map.get(theme_name.lower())
            if theme_path:
                file = QFile(theme_path)
                if file.open(QIODevice.ReadOnly | QIODevice.Text):
                    stream = QTextStream(file)
                    theme_styles = stream.readAll()
                    file.close()
                    return base_styles + '\n' + theme_styles
                else:
                    print(f"Warning: Could not open theme file: {theme_path}")
        except Exception as e:
            print(f"Warning: Could not load theme from resources: {e}")
            import traceback
            traceback.print_exc()
        
        # Fall back to basic styles if theme loading fails
        return base_styles
# --- End Theme Constants ---