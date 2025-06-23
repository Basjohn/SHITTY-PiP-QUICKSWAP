import logging
import ctypes
import ctypes.wintypes as wintypes
from typing import Tuple, List, Optional, Dict, Any, Union
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QRectF, QObject, Signal, QSizeF
from PySide6.QtGui import QMouseEvent, QGuiApplication, QScreen, QWindow
import math
import sys

# Windows API constants
MONITOR_DEFAULTTONEAREST = 0x00000002
MONITOR_DEFAULTTOPRIMARY = 0x00000001
MONITOR_DEFAULTTONULL = 0x00000000

# Windows API types
class RECT(ctypes.Structure):
    _fields_ = [
        ('left', wintypes.LONG),
        ('top', wintypes.LONG),
        ('right', wintypes.LONG),
        ('bottom', wintypes.LONG)
    ]

class MONITORINFOEX(ctypes.Structure):
    _fields_ = [
        ('cbSize', wintypes.DWORD),
        ('rcMonitor', RECT),
        ('rcWork', RECT),
        ('dwFlags', wintypes.DWORD),
        ('szDevice', wintypes.WCHAR * 32)
    ]

# Windows API functions
user32 = ctypes.WinDLL('user32')
shcore = ctypes.OleDLL('shcore')

# Function prototypes
user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
user32.MonitorFromPoint.restype = wintypes.HMONITOR

user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.MonitorFromWindow.restype = wintypes.HMONITOR

user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFOEX)]
user32.GetMonitorInfoW.restype = wintypes.BOOL

# For GetDpiForMonitor
try:
    shcore.GetDpiForMonitor.argtypes = [wintypes.HMONITOR, ctypes.c_int, ctypes.POINTER(wintypes.UINT), ctypes.POINTER(wintypes.UINT)]
    shcore.GetDpiForMonitor.restype = wintypes.HRESULT
    HAS_GET_DPI_FOR_MONITOR = True
except (AttributeError, OSError):
    HAS_GET_DPI_FOR_MONITOR = False

# For GetDeviceCaps
gdi32 = ctypes.WinDLL('gdi32')
LOGPIXELSX = 88
LOGPIXELSY = 90
HDC = wintypes.HDC

gdi32.GetDeviceCaps.argtypes = [HDC, ctypes.c_int]
gdi32.GetDeviceCaps.restype = ctypes.c_int

gdi32.CreateDCA.argtypes = [wintypes.LPCSTR, wintypes.LPCSTR, wintypes.LPCSTR, wintypes.LPVOID]
gdi32.CreateDCA.restype = HDC

gdi32.DeleteDC.argtypes = [HDC]
gdi32.DeleteDC.restype = wintypes.BOOL
logger = logging.getLogger(__name__)

_screen_cache = None
_cache_valid = False

# Constants for Windows APIs
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

def _refresh_screen_cache():
    global _screen_cache, _cache_valid
    monitors = []
    try:
        screens = QGuiApplication.screens()
        for screen in screens:
            geometry = screen.geometry()
            monitors.append({
                'monitor': geometry,
                'work': screen.availableGeometry(),
                'primary': screen == QGuiApplication.primaryScreen(),
                'device': screen.name()
            })
        min_x = min(m['monitor'].left() for m in monitors)
        min_y = min(m['monitor'].top() for m in monitors)
        max_x = max(m['monitor'].right() for m in monitors)
        max_y = max(m['monitor'].bottom() for m in monitors)
        _screen_cache = {
            'monitors': monitors,
            'virtual_screen': QRect(min_x, min_y, max_x - min_x, max_y - min_y)
        }
        _cache_valid = True
    except Exception as e:
        logger.error(f"Failed to get screen info: {e}")
        _screen_cache = {'monitors': [], 'virtual_screen': QRect()}

def get_screen_info():
    global _cache_valid
    if not _cache_valid:
        _refresh_screen_cache()
    return _screen_cache

def get_all_monitor_rects() -> List[QRect]:
    return [m['monitor'] for m in get_screen_info()['monitors']]

def get_virtual_screen_rect() -> QRect:
    return get_screen_info()['virtual_screen']

def _find_monitor_for_window(pos: QPoint, size: QSize) -> Optional[int]:
    monitors = get_all_monitor_rects()
    window_center = QPoint(pos.x() + size.width() // 2, pos.y() + size.height() // 2)
    for i, rect in enumerate(monitors):
        if rect.contains(window_center):
            return i
    return None

def ensure_within_available_desktop(pos: QPoint, size: QSize) -> QPoint:
    """Ensure the window stays within the available desktop area."""
    # First try to get the screen at the target position
    screen = QGuiApplication.screenAt(pos)
    
    # If no screen at position, try to find which screen contains the window center
    if not screen and size.isValid():
        center_pos = pos + QPoint(size.width() // 2, size.height() // 2)
        screen = QGuiApplication.screenAt(center_pos)
    
    # Fall back to primary screen
    if not screen:
        screen = QGuiApplication.primaryScreen()
        logger.debug(f"No screen found at {pos}, falling back to primary screen: {screen.name() if screen else 'None'}")
    
    if not screen:
        logger.debug("No screens available, using virtual screen constraints")
        return _constrain_to_virtual_screen(pos, size)
    
    # Get the available geometry (excluding taskbar)
    available_rect = screen.availableGeometry()
    
    # Calculate maximum allowed position
    max_x = available_rect.right() - size.width()
    max_y = available_rect.bottom() - size.height()
    
    # Constrain position with proper edge handling
    x = max(available_rect.left(), min(pos.x(), max_x))
    y = max(available_rect.top(), min(pos.y(), max_y))
    
    # Special case: If window is larger than screen, center it
    if size.width() > available_rect.width():
        x = available_rect.left() + (available_rect.width() - size.width()) // 2
    if size.height() > available_rect.height():
        y = available_rect.top() + (available_rect.height() - size.height()) // 2
    
    # Log if position was adjusted
    if x != pos.x() or y != pos.y():
        logger.debug(f"Constrained to available desktop - From: {pos}, To: ({x}, {y}), "
                   f"Screen: {screen.name()}, Available: {available_rect}, Size: {size}")
    
    return QPoint(x, y)

def _constrain_to_virtual_screen(pos: QPoint, size: QSize) -> QPoint:
    """Constrain a position to ensure the window stays within the virtual screen."""
    rect = get_virtual_screen_rect()
    
    # Calculate maximum allowed position
    max_x = rect.right() - size.width()
    max_y = rect.bottom() - size.height()
    
    # Constrain position with proper edge handling
    x = max(rect.left(), min(pos.x(), max_x))
    y = max(rect.top(), min(pos.y(), max_y))
    
    # Special case: If window is larger than screen, ensure it's fully visible
    if size.width() > rect.width():
        x = rect.left()
    if size.height() > rect.height():
        y = rect.top()
    
    # Log if position was adjusted
    if x != pos.x() or y != pos.y():
        logger.debug(f"Constrained window position from {pos} to ({x}, {y}) - "
                   f"Screen: {rect}, Size: {size}, Max: ({max_x}, {max_y})")
    
    return QPoint(x, y)

def apply_snap(pos: QPoint, size: QSize, snap_distance: int = 30) -> Tuple[QPoint, bool]:
    window_rect = QRect(pos, size)
    snap_points = []
    for monitor in get_all_monitor_rects():
        snap_points.extend([
            QPoint(monitor.left(), window_rect.top()),
            QPoint(monitor.right() - size.width(), window_rect.top()),
            QPoint(window_rect.left(), monitor.top()),
            QPoint(window_rect.left(), monitor.bottom() - size.height()),
        ])
    best = pos
    snapped = False
    min_dist = snap_distance + 1
    for p in snap_points:
        dist = (p - pos).manhattanLength()
        if dist < min_dist:
            min_dist = dist
            best = p
            snapped = True
    return best, snapped

# Drag/Resize handlers:
def handle_overlay_mouse_press(event: QMouseEvent, window):
    screen = window.screen() if hasattr(window, 'screen') else QGuiApplication.primaryScreen()
    state = {
        'is_resizing': False,
        'resize_edge': None,
        'drag_start_global': None,
        'initial_geometry': None,
        'drag_offset': None,
        'monitor': 0
    }
    if screen:
        idx = _find_monitor_for_window(window.pos(), window.size())
        if idx is not None:
            state['monitor'] = idx
    if event.button() == Qt.LeftButton:
        resize_edge = get_resize_edge_for_pos(event.pos(), window)
        global_pos = event.globalPosition().toPoint()
        if resize_edge:
            state.update({
                'is_resizing': True,
                'resize_edge': resize_edge,
                'drag_start_global': global_pos,
                'initial_geometry': window.geometry()
            })
        else:
            state.update({
                'drag_start_global': global_pos,
                'initial_geometry': window.geometry(),
                'drag_offset': global_pos - window.pos()
            })
    return state

def handle_overlay_mouse_move(event: QMouseEvent, window, state, snap_distance=30):
    try:
        global_pos = event.globalPosition().toPoint()
        
        # Log movement data
        logger.debug(f"Mouse move - Global: {global_pos}, State: {state}")
        
        if state.get('is_resizing'):
            delta = global_pos - state['drag_start_global']
            geom = QRect(state['initial_geometry'])
            edge = state['resize_edge']
            
            # Log resize start
            logger.debug(f"Resizing - Edge: {edge}, Delta: {delta}, Initial: {geom}")
            
            # Apply resize deltas
            if 'left' in edge: 
                geom.setLeft(geom.left() + delta.x())
            if 'right' in edge: 
                geom.setRight(geom.right() + delta.x())
            if 'top' in edge: 
                geom.setTop(geom.top() + delta.y())
            if 'bottom' in edge: 
                geom.setBottom(geom.bottom() + delta.y())
                
            # Enforce minimum size
            min_size = window.minimumSizeHint()
            if min_size.isValid():
                old_size = geom.size()
                geom.setWidth(max(geom.width(), min_size.width()))
                geom.setHeight(max(geom.height(), min_size.height()))
                if old_size != geom.size():
                    logger.debug(f"Adjusted size from {old_size} to {geom.size()} (min: {min_size})")
            
            # Constrain to desktop and apply
            old_geom = geom
            geom.moveTopLeft(ensure_within_available_desktop(geom.topLeft(), geom.size()))
            if old_geom != geom:
                logger.debug(f"Constrained geometry from {old_geom} to {geom}")
                
            window.setGeometry(geom)
            return True
            
        elif state.get('drag_offset') is not None:
            # Calculate new position
            new_pos = global_pos - state['drag_offset']
            old_pos = window.pos()
            
            # Apply snapping
            new_pos, snapped = apply_snap(new_pos, window.size(), snap_distance)
            if snapped:
                logger.debug(f"Snapped to position: {new_pos}")
                
            # Constrain to desktop
            constrained_pos = ensure_within_available_desktop(new_pos, window.size())
            if constrained_pos != new_pos:
                logger.debug(f"Constrained position from {new_pos} to {constrained_pos}")
                new_pos = constrained_pos
                
            # Move window if position changed
            if new_pos != old_pos:
                window.move(new_pos)
                logger.debug(f"Moved window from {old_pos} to {new_pos}")
                
            return True
    except KeyError as ke:
        logger.error(f"Missing drag state key: {ke}")
        if str(ke) == "'monitor'":
            state['monitor'] = 0
            return True
    except Exception as e:
        logger.error(f"Error during drag/resize: {e}", exc_info=True)
    return False

def handle_overlay_mouse_release(event: QMouseEvent, window, state):
    if event.button() == Qt.LeftButton:
        state.update({
            'is_resizing': False,
            'resize_edge': None,
            'drag_start_global': None,
            'initial_geometry': None,
            'drag_offset': None
        })
        window.setCursor(Qt.ArrowCursor)
        return True
    return False

def get_resize_edge_for_pos(pos: QPoint, widget, margin: int = 8) -> Optional[str]:
    """Determine which edge is being resized based on mouse position.
    
    Args:
        pos: Mouse position relative to the widget
        widget: The widget being resized
        margin: How many pixels from the edge to consider as a resize area
        
    Returns:
        A string indicating the edge being resized, or None if not on an edge
    """
    try:
        width = widget.width()
        height = widget.height()
        
        # Calculate edge regions with a minimum margin of 1px
        margin = max(1, margin)  # Ensure at least 1px margin
        
        # Check proximity to edges
        left = pos.x() <= margin
        right = pos.x() >= width - margin
        top = pos.y() <= margin
        bottom = pos.y() >= height - margin
        
        # Check corners first
        if left and top:
            return 'top-left'
        elif right and top:
            return 'top-right'
        elif left and bottom:
            return 'bottom-left'
        elif right and bottom:
            return 'bottom-right'
        # Then check edges
        elif left:
            return 'left'
        elif right:
            return 'right'
        elif top:
            return 'top'
        elif bottom:
            return 'bottom'
            
        return None
    except Exception as e:
        logger.error(f"Error in get_resize_edge_for_pos: {e}")
        return None

def invalidate_cache():
    global _cache_valid
    _cache_valid = False

def debug_monitor_setup():
    """Debug function to print detailed information about all monitors."""
    screens = QGuiApplication.screens()
    logger.info("=" * 80)
    logger.info(f"MONITOR DEBUG INFORMATION - Detected {len(screens)} screens")
    logger.info("=" * 80)
    
    for i, screen in enumerate(screens):
        # Get basic Qt screen info
        geometry = screen.geometry()
        available = screen.availableGeometry()
        
        # Get physical monitor info
        monitor_info = get_physical_monitor_info(screen)
        
        # Format the output
        logger.info("\n" + "-" * 40)
        logger.info(f"SCREEN {i}: {screen.name()}")
        logger.info("-" * 40)
        
        # Basic Qt information
        logger.info("[Qt Information]")
        logger.info(f"  Geometry:        {geometry.x()}, {geometry.y()} {geometry.width()}x{geometry.height()}")
        logger.info(f"  Available:       {available.x()}, {available.y()} {available.width()}x{available.height()}")
        logger.info(f"  Logical DPI:     {screen.logicalDotsPerInchX():.1f}x{screen.logicalDotsPerInchY():.1f}")
        logger.info(f"  Physical DPI:    {screen.physicalDotsPerInchX():.1f}x{screen.physicalDotsPerInchY():.1f}")
        logger.info(f"  Device Pixel:    {screen.devicePixelRatio():.1f}")
        
        # Windows API information if available
        if monitor_info:
            logger.info("\n[Physical Monitor Information]")
            logger.info(f"  Physical Res:    {monitor_info.get('physical_width', 0)}x{monitor_info.get('physical_height', 0)}")
            logger.info(f"  Scaled Res:      {monitor_info.get('scaled_width', 0)}x{monitor_info.get('scaled_height', 0)}")
            logger.info(f"  Position:        {monitor_info.get('physical_position', 'N/A')}")
            logger.info(f"  DPI:             {monitor_info.get('dpi', (0, 0))[0]:.1f}x{monitor_info.get('dpi', (0, 0))[1]:.1f}")
            logger.info(f"  Scale Factor:    {monitor_info.get('scale_factor_x', 1.0):.2f}x{monitor_info.get('scale_factor_y', 1.0):.2f}")
            logger.info(f"  Primary:         {monitor_info.get('is_primary', False)}")
            if 'device_name' in monitor_info:
                logger.info(f"  Device Name:     {monitor_info['device_name']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("END OF MONITOR DEBUG INFORMATION")
    logger.info("=" * 80)

def get_physical_monitor_info(screen: QScreen) -> Dict[str, Any]:
    """
    Get physical monitor information for a screen using Windows API.
    
    This function retrieves the true physical resolution and DPI scaling information
    for the specified screen, accounting for Windows DPI scaling.
    
    Args:
        screen: The Qt screen to get information about
        
    Returns:
        Dictionary containing monitor information with both physical and logical dimensions,
        DPI, and scaling information. Always returns a dictionary with all required keys.
    """
    def get_default_values():
        """Return a dictionary with default values for all required keys."""
        default_rect = QRect(0, 0, 1920, 1080)  # Default to 1080p
        return {
            'name': screen.name() if screen else 'Unknown',
            'is_primary': screen == QGuiApplication.primaryScreen() if screen else False,
            'screen': screen,
            'logical_geometry': screen.geometry() if screen else default_rect,
            'logical_available_geometry': screen.availableGeometry() if screen else default_rect,
            'device_pixel_ratio': screen.devicePixelRatio() if screen else 1.0,
            'logical_dpi': (screen.logicalDotsPerInchX(), screen.logicalDotsPerInchY()) if screen else (96, 96),
            'physical_dpi': (screen.physicalDotsPerInchX(), screen.physicalDotsPerInchY()) if screen else (96, 96),
            'physical_width': 1920,
            'physical_height': 1080,
            'physical_position': QPoint(0, 0),
            'physical_rect': default_rect,
            'physical_work_area': default_rect,
            'scaled_width': 1920,
            'scaled_height': 1080,
            'scaled_rect': default_rect,
            'dpi': (96, 96),
            'scale_factor': 1.0,
            'scale_factor_x': 1.0,
            'scale_factor_y': 1.0,
            'monitor_handle': None,
            'device_name': '\\.\\DISPLAY1',
        }
    
    if not screen:
        logger.warning("No screen provided to get_physical_monitor_info, using defaults")
        return get_default_values()
    
    # Start with default values that will be overridden by actual values
    result = get_default_values()
    
    try:
        # Get the screen's center point in virtual desktop coordinates
        geometry = screen.geometry()
        if not geometry.isValid() or geometry.isNull():
            logger.warning(f"Invalid geometry for screen {screen.name()}, using defaults")
            return result
            
        center = geometry.center()
        
        # Get the monitor handle for this screen
        pt = wintypes.POINT(center.x(), center.y())
        monitor = user32.MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST)
        
        if not monitor:
            logger.warning(f"Could not get monitor handle for screen {screen.name()}, using defaults")
            return result
            
        # Get monitor information
        monitor_info = MONITORINFOEX()
        monitor_info.cbSize = ctypes.sizeof(MONITORINFOEX)
        
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(monitor_info)):
            logger.warning(f"Could not get monitor info for screen {screen.name()}, using defaults")
            return result
            
        # Get the monitor rectangle in virtual desktop coordinates
        rc = monitor_info.rcMonitor
        work_rc = monitor_info.rcWork
        
        # Get DPI using Windows API if available
        dpi_x, dpi_y = 96, 96
        try:
            if HAS_GET_DPI_FOR_MONITOR:
                x_dpi = wintypes.UINT()
                y_dpi = wintypes.UINT()
                # 0 = MDT_EFFECTIVE_DPI
                if shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(x_dpi), ctypes.byref(y_dpi)) == 0:  # S_OK
                    dpi_x, dpi_y = x_dpi.value, y_dpi.value
                    logger.debug(f"Got DPI from GetDpiForMonitor: {dpi_x}x{dpi_y}")
                else:
                    logger.warning(f"GetDpiForMonitor failed for monitor {monitor}")
            
            # Fallback to GetDeviceCaps if needed
            if dpi_x == 96 or dpi_y == 96:  # If we still have default values
                try:
                    device_name = monitor_info.szDevice if hasattr(monitor_info, 'szDevice') else None
                    hdc = gdi32.CreateDCA(device_name, None, None, None) if device_name else None
                    if hdc:
                        dpi_x = gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
                        dpi_y = gdi32.GetDeviceCaps(hdc, LOGPIXELSY)
                        gdi32.DeleteDC(hdc)
                        logger.debug(f"Got DPI from GetDeviceCaps: {dpi_x}x{dpi_y}")
                except Exception as e:
                    logger.debug(f"Fallback DPI detection failed: {e}")
        except Exception as e:
            logger.warning(f"Error in DPI detection: {e}", exc_info=True)
        
        # Calculate scale factors
        scale_x = dpi_x / 96.0
        scale_y = dpi_y / 96.0
        
        # Get the physical resolution (actual pixels)
        phys_width = rc.right - rc.left
        phys_height = rc.bottom - rc.top
        
        # Calculate the logical resolution (scaled pixels)
        logical_width = int(phys_width / scale_x)
        logical_height = int(phys_height / scale_y)
        
        # Update result with all the information
        result.update({
            # Physical monitor information
            'physical_width': phys_width,
            'physical_height': phys_height,
            'physical_position': QPoint(rc.left, rc.top),
            'physical_rect': QRect(rc.left, rc.top, phys_width, phys_height),
            'physical_work_area': QRect(work_rc.left, work_rc.top, 
                                      work_rc.right - work_rc.left, 
                                      work_rc.bottom - work_rc.top),
            
            # Scaled (logical) resolution
            'scaled_width': logical_width,
            'scaled_height': logical_height,
            'scaled_rect': QRect(rc.left, rc.top, logical_width, logical_height),
            
            # DPI and scaling information
            'dpi': (dpi_x, dpi_y),
            'scale_factor': scale_x,  # Primary scale factor (X axis)
            'scale_factor_x': scale_x,
            'scale_factor_y': scale_y,
            
            # Raw Windows API information
            'monitor_handle': monitor,
            'device_name': monitor_info.szDevice,
            'is_primary': bool(monitor_info.dwFlags & 1)  # MONITORINFOF_PRIMARY = 0x1
        })
        
        logger.info(f"Monitor Info - {screen.name()}:")
        logger.info(f"  Physical: {phys_width}x{phys_height} @ ({rc.left},{rc.top})")
        logger.info(f"  Logical: {logical_width}x{logical_height}")
        logger.info(f"  DPI: {dpi_x}x{dpi_y}, Scale: {scale_x:.2f}x{scale_y:.2f}")
        logger.info(f"  Primary: {result['is_primary']}")
        
    except Exception as e:
        logger.error(f"Error getting physical monitor info for {screen.name()}: {e}", exc_info=True)
    
    return result

def get_physical_monitor_for_screen(screen: QScreen) -> Dict[str, Any]:
    """
    Get physical monitor information for a specific screen.
    
    Args:
        screen: The Qt screen to get information about
        
    Returns:
        Dictionary containing monitor information
    """
    return get_physical_monitor_info(screen)

def get_screen_scale_factor(screen: QScreen) -> float:
    """
    Get the scale factor for a screen, using Windows API if possible.
    
    Args:
        screen: The Qt screen to get the scale factor for
        
    Returns:
        The screen's scale factor as a float (e.g., 1.0 for 96 DPI, 1.5 for 144 DPI)
    """
    if not screen:
        return 1.0
    
    try:
        # First try to get scale factor from physical monitor info
        monitor_info = get_physical_monitor_info(screen)
        if 'dpi' in monitor_info and isinstance(monitor_info['dpi'], (tuple, list)) and len(monitor_info['dpi']) >= 1:
            dpi = monitor_info['dpi'][0]  # Use X DPI
            scale = dpi / 96.0
            # Round to nearest 0.25 to handle standard scaling values (100%, 125%, 150%, etc.)
            return round(scale * 4) / 4.0
    except Exception as e:
        logger.warning(f"Couldn't get scale factor from Windows API: {e}")
    
    # Fall back to Qt's device pixel ratio
    return screen.devicePixelRatio()
