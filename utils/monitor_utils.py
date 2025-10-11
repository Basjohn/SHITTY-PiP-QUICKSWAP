"""
monitor_utils.py - Consolidated monitor information and management utilities.
"""

import ctypes
import time
from ctypes import wintypes
from typing import Dict, List, Optional, Any

from PySide6.QtCore import QPoint, QRect, QSizeF
from PySide6.QtGui import QGuiApplication, QScreen

from core.logging import get_logger

logger = get_logger(__name__)

# Prevent log spam: remember last enumerated monitor signature
_last_monitors_signature: Optional[str] = None

class MONITORINFOEX(ctypes.Structure):
    _fields_ = [
        ('cbSize', wintypes.DWORD),
        ('rcMonitor', wintypes.RECT),
        ('rcWork', wintypes.RECT),
        ('dwFlags', wintypes.DWORD),
        ('szDevice', wintypes.WCHAR * 32)
    ]

user32 = ctypes.WinDLL('user32', use_last_error=True)
gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
shcore = ctypes.WinDLL('shcore', use_last_error=True)

MONITOR_DEFAULTTOPRIMARY = 0x00000001
MONITOR_DEFAULTTONEAREST = 0x00000002
MONITOR_DEFAULTTONULL = 0x00000000
MONITORINFOF_PRIMARY = 0x00000001
LOGPIXELSX = 88
LOGPIXELSY = 90

MonitorFromPoint = user32.MonitorFromPoint
MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
MonitorFromPoint.restype = wintypes.HMONITOR

GetMonitorInfo = user32.GetMonitorInfoW
GetMonitorInfo.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFOEX)]
GetMonitorInfo.restype = wintypes.BOOL

GetDpiForMonitor = shcore.GetDpiForMonitor
GetDpiForMonitor.argtypes = [wintypes.HMONITOR, ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
GetDpiForMonitor.restype = ctypes.HRESULT

CreateDCA = gdi32.CreateDCA
CreateDCA.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPVOID]
CreateDCA.restype = wintypes.HDC

GetDeviceCaps = gdi32.GetDeviceCaps
GetDeviceCaps.argtypes = [wintypes.HDC, ctypes.c_int]
GetDeviceCaps.restype = ctypes.c_int

DeleteDC = gdi32.DeleteDC
DeleteDC.argtypes = [wintypes.HDC]
DeleteDC.restype = wintypes.BOOL

_monitor_cache = {}
_cache_timestamp = 0
CACHE_TIMEOUT = 5.0

def get_physical_monitor_info(screen: QScreen) -> Dict[str, Any]:
    global _monitor_cache, _cache_timestamp
    current_time = time.time()
    if current_time - _cache_timestamp < CACHE_TIMEOUT and screen.name() in _monitor_cache:
        return _monitor_cache[screen.name()]
    
    def get_default_values():
        geo = screen.geometry()
        dpi = screen.logicalDotsPerInch()
        scale_factor = dpi / 96.0
        return {
            'width': geo.width(),
            'height': geo.height(),
            'position': QPoint(geo.x(), geo.y()),
            'work_area': screen.availableGeometry(),
            'primary': screen == QGuiApplication.primaryScreen(),
            'monitor_rect': geo,
            'dpi': QSizeF(dpi, dpi),
            'scale_factor': scale_factor,
            'scale_factor_x': scale_factor,
            'scale_factor_y': scale_factor,
            'physical_width': int(geo.width() * scale_factor),
            'physical_height': int(geo.height() * scale_factor),
            'is_primary': screen == QGuiApplication.primaryScreen(),
            'device_name': screen.name(),
            'screen_object': screen
        }
    
    try:
        point = screen.geometry().center()
        pt = wintypes.POINT(int(point.x()), int(point.y()))
        monitor = MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST)
        if not monitor:
            logger.warning(f"MonitorFromPoint failed for point {point.x()}, {point.y()}")
            return get_default_values()
        monitor_info = MONITORINFOEX()
        monitor_info.cbSize = ctypes.sizeof(MONITORINFOEX)
        if not GetMonitorInfo(monitor, ctypes.byref(monitor_info)):
            logger.warning(f"GetMonitorInfo failed for monitor {monitor}")
            return get_default_values()
        dpi_x, dpi_y = 96, 96
        try:
            x_dpi = ctypes.c_uint()
            y_dpi = ctypes.c_uint()
            result = GetDpiForMonitor(monitor, 0, ctypes.byref(x_dpi), ctypes.byref(y_dpi))
            if result == 0:
                dpi_x, dpi_y = x_dpi.value, y_dpi.value
                logger.debug(f"Got DPI from GetDpiForMonitor: {dpi_x}x{dpi_y}")
            else:
                logger.warning(f"GetDpiForMonitor failed for monitor {monitor}")
        except (AttributeError, OSError) as e:
            logger.debug(f"GetDpiForMonitor not available: {e}")
        if dpi_x == 96 or dpi_y == 96:
            try:
                device_name = monitor_info.szDevice if hasattr(monitor_info, 'szDevice') else None
                hdc = CreateDCA(device_name, None, None, None) if device_name else None
                if hdc:
                    dpi_x = GetDeviceCaps(hdc, LOGPIXELSX)
                    dpi_y = GetDeviceCaps(hdc, LOGPIXELSY)
                    DeleteDC(hdc)
                    logger.debug(f"Got DPI from GetDeviceCaps: {dpi_x}x{dpi_y}")
            except Exception as e:
                logger.debug(f"Fallback DPI detection failed: {e}")
        scale_factor_x = dpi_x / 96.0
        scale_factor_y = dpi_y / 96.0
        # Windows MONITORINFOEX rcMonitor is in physical pixels for DPI-aware apps.
        # Treat rcMonitor as physical. Do NOT multiply by DPI here to avoid double-scaling.
        logical_width = monitor_info.rcMonitor.right - monitor_info.rcMonitor.left
        logical_height = monitor_info.rcMonitor.bottom - monitor_info.rcMonitor.top
        physical_width = int(logical_width)
        physical_height = int(logical_height)
        result = {
            'width': logical_width,
            'height': logical_height,
            'position': QPoint(monitor_info.rcMonitor.left, monitor_info.rcMonitor.top),
            'work_area': QRect(
                monitor_info.rcWork.left, monitor_info.rcWork.top,
                monitor_info.rcWork.right - monitor_info.rcWork.left,
                monitor_info.rcWork.bottom - monitor_info.rcWork.top
            ),
            'primary': bool(monitor_info.dwFlags & MONITORINFOF_PRIMARY),
            'monitor_rect': QRect(
                monitor_info.rcMonitor.left, monitor_info.rcMonitor.top,
                logical_width, logical_height
            ),
            'dpi': QSizeF(dpi_x, dpi_y),
            'scale_factor': scale_factor_x,
            'scale_factor_x': scale_factor_x,
            'scale_factor_y': scale_factor_y,
            'physical_width': physical_width,
            'physical_height': physical_height,
            'is_primary': bool(monitor_info.dwFlags & MONITORINFOF_PRIMARY),
            'device_name': monitor_info.szDevice if hasattr(monitor_info, 'szDevice') else '',
            'screen_object': screen
        }
        _monitor_cache[screen.name()] = result
        _cache_timestamp = current_time
        return result
    except Exception as e:
        logger.error(f"Error getting physical monitor info: {e}", exc_info=True)
        return get_default_values()

def get_all_monitors() -> List[Dict[str, Any]]:
    try:
        qt_screens = QGuiApplication.screens()
        if not qt_screens:
            logger.warning("No screens found via QGuiApplication.screens()")
            return []
        # Reduce verbosity: only DEBUG this frequent line
        logger.debug(f"Found {len(qt_screens)} Qt screens")
        monitors = []
        for i, screen in enumerate(qt_screens):
            try:
                monitor_info = get_physical_monitor_info(screen)
                if not monitor_info:
                    logger.warning(f"Failed to get physical monitor info for screen {i}")
                    continue
                monitor_info['screen_object'] = screen
                monitor_info['qt_index'] = i
                if 'rect' not in monitor_info:
                    monitor_info['rect'] = screen.geometry()
                if 'position' not in monitor_info:
                    monitor_info['position'] = screen.geometry().topLeft()
                if 'is_primary' not in monitor_info:
                    monitor_info['is_primary'] = (screen == QGuiApplication.primaryScreen())
                monitors.append(monitor_info)
                # Reduced debug spam - only log monitor changes, not repeated detections
            except Exception as e:
                logger.error(f"Error processing screen {i}: {e}", exc_info=True)
        monitors.sort(key=lambda m: (m['position'].x(), m['position'].y()))
        # Build a signature using physical resolution and positions to suppress repeated INFO logs
        signature_parts = []
        for i, mon in enumerate(monitors):
            phys_w = int(mon.get('physical_width', mon['rect'].width()))
            phys_h = int(mon.get('physical_height', mon['rect'].height()))
            signature_parts.append(
                f"{i}:{mon.get('device_name','?')}:{phys_w}x{phys_h}@{mon['position'].x()},{mon['position'].y()}:{int(mon.get('is_primary', False))}"
            )
        signature = "|".join(signature_parts)

        global _last_monitors_signature
        if signature != _last_monitors_signature:
            _last_monitors_signature = signature
            for i, mon in enumerate(monitors):
                phys_w = int(mon.get('physical_width', mon['rect'].width()))
                phys_h = int(mon.get('physical_height', mon['rect'].height()))
                logger.info(
                    f"Monitor {i}: {mon.get('device_name', 'Unknown')} "
                    f"{phys_w}x{phys_h}+{mon['position'].x()}+{mon['position'].y()} "
                    f"(Primary: {mon.get('is_primary', False)})"
                )
        else:
            # When unchanged, keep details at DEBUG only
            for i, mon in enumerate(monitors):
                phys_w = int(mon.get('physical_width', mon['rect'].width()))
                phys_h = int(mon.get('physical_height', mon['rect'].height()))
                logger.debug(
                    f"Monitor {i}: {mon.get('device_name', 'Unknown')} "
                    f"{phys_w}x{phys_h}+{mon['position'].x()}+{mon['position'].y()} "
                    f"(Primary: {mon.get('is_primary', False)})"
                )
        return monitors
    except Exception as e:
        logger.critical(f"Critical error in get_all_monitors: {e}", exc_info=True)
        return [{
            'screen_object': screen,
            'rect': screen.geometry(),
            'position': screen.geometry().topLeft(),
            'is_primary': (screen == QGuiApplication.primaryScreen()),
            'device_name': screen.name(),
            'qt_index': i
        } for i, screen in enumerate(QGuiApplication.screens())]

def get_primary_monitor() -> Dict[str, Any]:
    primary_screen = QGuiApplication.primaryScreen()
    if not primary_screen:
        screens = QGuiApplication.screens()
        if not screens:
            raise RuntimeError("No screens found")
        primary_screen = screens[0]
    return get_physical_monitor_info(primary_screen)

def get_monitor_at(point: QPoint) -> Optional[Dict[str, Any]]:
    screen = QGuiApplication.screenAt(point)
    if not screen:
        return None
    return get_physical_monitor_info(screen)

def debug_monitor_info() -> str:
    screens = QGuiApplication.screens()
    output = ["=" * 80]
    output.append(f"MONITOR DEBUG INFORMATION - Detected {len(screens)} screens")
    output.append("=" * 80)
    for i, screen in enumerate(screens):
        geo = screen.geometry()
        available = screen.availableGeometry()
        monitor_info = get_physical_monitor_info(screen)
        output.append("\n" + "-" * 40)
        output.append(f"Screen {i}: {screen.name() if hasattr(screen, 'name') else 'N/A'}")
        output.append("-" * 40)
        output.append(f"  Geometry:        {geo.width()}x{geo.height()} @ ({geo.x()},{geo.y()})")
        output.append(f"  Available:       {available.width()}x{available.height()} @ ({available.x()},{available.y()})")
        output.append(f"  DPI:             {monitor_info.get('dpi', (0, 0))[0]:.1f}x{monitor_info.get('dpi', (0, 0))[1]:.1f}")
        output.append(f"  Scale Factor:    {monitor_info.get('scale_factor_x', 1.0):.2f}x{monitor_info.get('scale_factor_y', 1.0):.2f}")
        output.append(f"  Primary:         {monitor_info.get('is_primary', False)}")
        if 'device_name' in monitor_info:
            output.append(f"  Device Name:     {monitor_info['device_name']}")
    output.append("\n" + "=" * 80)
    output.append("END OF MONITOR DEBUG INFORMATION")
    output.append("=" * 80)
    return "\n".join(output)

def log_monitor_info():
    logger.info(debug_monitor_info())

get_physical_monitor_for_screen = get_physical_monitor_info
debug_monitor_setup = log_monitor_info