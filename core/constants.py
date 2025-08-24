"""
Core constants and shared utilities.

This module contains shared constants, type definitions, and utility functions
that can be imported by any module without causing circular imports.
"""
from typing import Any, TypeVar
from PySide6.QtGui import QColor

# Type variable for generic types
T = TypeVar('T')

def is_valid(obj: Any) -> bool:
    """Check if an object is valid (not None)."""
    return obj is not None

# --- Global Default Constants ---
DEFAULT_POSITION_PRESET = "Top Left"
DEFAULT_WINDOW_OVERLAY_WIDTH = 640
DEFAULT_WINDOW_OVERLAY_HEIGHT = 360
DEFAULT_MONITOR_OVERLAY_WIDTH_FACTOR = 0.30  # 30% of screen width
DEFAULT_MONITOR_OVERLAY_HEIGHT_FACTOR = 0.30  # 30% of screen height

# --- UI Constants ---
# FocusIndicatorWidget constants
FOCUS_INDICATOR_SIZE = 20
FOCUS_INDICATOR_MARGIN = 10

class Win32Constants:
    """Windows API constants organized by category."""
    
    class Monitor:
        """Monitor-related constants."""
        MONITOR_DEFAULTTONULL = 0
        MONITOR_DEFAULTTOPRIMARY = 1
        MONITOR_DEFAULTTONEAREST = 2
        MONITORINFOF_PRIMARY = 1
    
    class SystemMetrics:
        """System metrics constants."""
        SM_XVIRTUALSCREEN = 76
        SM_YVIRTUALSCREEN = 77
        SM_CXVIRTUALSCREEN = 78
        SM_CYVIRTUALSCREEN = 79
    
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
    0x1B: 'ESC',      # VK_ESCAPE
    0x09: 'TAB',      # VK_TAB
    0x08: 'BACKSPACE',# VK_BACK
    0x2D: 'INSERT',   # VK_INSERT
    0x2E: 'DELETE',   # VK_DELETE
    0x25: 'LEFT',     # VK_LEFT
    0x26: 'UP',       # VK_UP
    0x27: 'RIGHT',    # VK_RIGHT
    0x28: 'DOWN',     # VK_DOWN
    0x21: 'PAGE_UP',  # VK_PRIOR
    0x22: 'PAGE_DOWN',# VK_NEXT
    0x24: 'HOME',     # VK_HOME
    0x23: 'END',      # VK_END
    0x70: 'F1',       # VK_F1
    0x71: 'F2',       # VK_F2
    0x72: 'F3',       # VK_F3
    0x73: 'F4',       # VK_F4
    0x74: 'F5',       # VK_F5
    0x75: 'F6',       # VK_F6
    0x76: 'F7',       # VK_F7
    0x77: 'F8',       # VK_F8
    0x78: 'F9',       # VK_F9
    0x79: 'F10',      # VK_F10
    0x7A: 'F11',      # VK_F11
    0x7B: 'F12',      # VK_F12
}

class ThemeColors:
    """Centralized theme color definitions for consistent theming across the application."""
    
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
    def get_theme_colors(cls, theme_name: str) -> dict:
        """Get the color dictionary for the specified theme."""
        return getattr(cls, theme_name.upper(), cls.DARK)

    @classmethod
    def get_theme_stylesheet(cls, theme_name: str) -> str:
        """Get a stylesheet string for the specified theme."""
        theme = cls.get_theme_colors(theme_name)
        return f"""
            QWidget {{
                background-color: {theme['background'].name()};
                color: {theme['foreground'].name()};
                border: 1px solid {theme['border'].name()};
            }}
            QPushButton {{
                background-color: {theme['button']['background'].name()};
                color: {theme['button']['text'].name()};
                border: 1px solid {theme['button']['border'].name()};
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {theme['button']['hover'].name()};
            }}
            QPushButton:pressed {{
                background-color: {theme['button']['pressed'].name()};
            }}
            QMenu {{
                background-color: {theme['menu']['background'].name()};
                color: {theme['menu']['text'].name()};
                border: 1px solid {theme['menu']['border'].name()};
            }}
            QMenu::item:selected {{
                background-color: {theme['menu']['highlight'].name()};
            }}
        """
