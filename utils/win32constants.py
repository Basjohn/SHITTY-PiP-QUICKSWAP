"""
win32constants.py

Centralized Win32 style and flag constants for Windows-only integrations.

- Provides IntFlag enums for window styles and SetWindowPos flags.
- Keeps platform-specific constants in one place for clarity and maintainability.
- Safe to import cross-platform; on non-Windows platforms, these remain defined but unused.

Usage:
    from utils.win32constants import SetWindowPosFlags as SWP, ExtendedWindowStyles as WS_EX

    flags = SWP.NOSIZE | SWP.NOMOVE | SWP.NOZORDER | SWP.NOACTIVATE
    style_ex = WS_EX.LAYERED | WS_EX.TRANSPARENT

Note: HWND constants (e.g., HWND_TOPMOST) are not defined here because they are handles,
      not bitflags. Define or obtain them within the Windows interop layer as needed.
"""
from __future__ import annotations

from enum import IntFlag
from typing import Final
import sys

IS_WINDOWS: Final[bool] = sys.platform.startswith("win32") or sys.platform.startswith("cygwin")


class WindowStyles(IntFlag):
    """Standard window style flags (WS_*)"""
    OVERLAPPED = 0x00000000
    POPUP = 0x80000000
    CHILD = 0x40000000
    MINIMIZE = 0x20000000
    VISIBLE = 0x10000000
    DISABLED = 0x08000000
    CLIPSIBLINGS = 0x04000000
    CLIPCHILDREN = 0x02000000
    MAXIMIZE = 0x01000000
    CAPTION = 0x00C00000
    BORDER = 0x00800000
    DLGFRAME = 0x00400000
    VSCROLL = 0x00200000
    HSCROLL = 0x00100000
    SYSMENU = 0x00080000
    THICKFRAME = 0x00040000
    GROUP = 0x00020000
    TABSTOP = 0x00010000
    MINIMIZEBOX = 0x00020000
    MAXIMIZEBOX = 0x00010000
    # Common combined style for top-level overlapped windows with caption and system menu
    OVERLAPPEDWINDOW = OVERLAPPED | CAPTION | SYSMENU | THICKFRAME | MINIMIZEBOX | MAXIMIZEBOX


class ExtendedWindowStyles(IntFlag):
    """Extended window style flags (WS_EX_*)"""
    DLGMODALFRAME = 0x00000001
    NOPARENTNOTIFY = 0x00000004
    TOPMOST = 0x00000008
    ACCEPTFILES = 0x00000010
    TRANSPARENT = 0x00000020
    MDICHILD = 0x00000040
    TOOLWINDOW = 0x00000080
    WINDOWEDGE = 0x00000100
    CLIENTEDGE = 0x00000200
    CONTEXTHELP = 0x00000400
    RIGHT = 0x00001000
    LEFT = 0x00000000
    RTLREADING = 0x00002000
    LTRREADING = 0x00000000
    LEFTSCROLLBAR = 0x00004000
    RIGHTSCROLLBAR = 0x00000000
    CONTROLPARENT = 0x00010000
    STATICEDGE = 0x00020000
    APPWINDOW = 0x00040000
    LAYERED = 0x00080000
    NOINHERITLAYOUT = 0x00100000
    NOCOPYBITS = 0x010000
    NOREDIRECTIONBITMAP = 0x00200000
    LAYOUTRTL = 0x00400000
    COMPOSITED = 0x02000000
    NOACTIVATE = 0x08000000


class SetWindowPosFlags(IntFlag):
    """Flags for SetWindowPos (SWP_*)"""
    NOSIZE = 0x0001
    NOMOVE = 0x0002
    NOZORDER = 0x0004
    NOREDRAW = 0x0008
    NOACTIVATE = 0x0010
    FRAMECHANGED = 0x0020
    SHOWWINDOW = 0x0040
    HIDEWINDOW = 0x0080
    NOCOPYBITS = 0x0100
    NOOWNERZORDER = 0x0200
    NOSENDCHANGING = 0x0400
    DEFERERASE = 0x2000
    ASYNCWINDOWPOS = 0x4000


class LayeredWindowAttributes(IntFlag):
    """Flags for SetLayeredWindowAttributes (LWA_*)"""
    COLORKEY = 0x00000001
    ALPHA = 0x00000002


__all__ = [
    "IS_WINDOWS",
    "WindowStyles",
    "ExtendedWindowStyles",
    "SetWindowPosFlags",
    "LayeredWindowAttributes",
]
