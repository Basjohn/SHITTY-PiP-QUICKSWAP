"""
Dialog positioning utilities with taskbar awareness and smart snapping.

This module provides intelligent positioning for dialogs that snap to screen edges
while respecting taskbar position and supporting multi-monitor setups with persistence.
"""

import ctypes
import ctypes.wintypes as wintypes
from typing import Optional, Tuple
from enum import Enum

from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtGui import QGuiApplication

from core.logging import get_logger

logger = get_logger(__name__)

# Windows API constants for taskbar detection
ABM_GETTASKBARPOS = 5
ABE_LEFT = 0
ABE_TOP = 1
ABE_RIGHT = 2
ABE_BOTTOM = 3


class TaskbarEdge(Enum):
    """Taskbar edge positions."""
    LEFT = ABE_LEFT
    TOP = ABE_TOP
    RIGHT = ABE_RIGHT
    BOTTOM = ABE_BOTTOM
    NONE = -1


class DialogSide(Enum):
    """Dialog snap side."""
    LEFT = "left"
    RIGHT = "right"


class APPBARDATA(ctypes.Structure):
    """Windows APPBARDATA structure for taskbar queries."""
    _fields_ = [
        ('cbSize', wintypes.DWORD),
        ('hWnd', wintypes.HWND),
        ('uCallbackMessage', wintypes.UINT),
        ('uEdge', wintypes.UINT),
        ('rc', wintypes.RECT),
        ('lParam', wintypes.LPARAM)
    ]


def get_taskbar_info(monitor_rect: QRect) -> Tuple[TaskbarEdge, QRect]:
    """Get taskbar edge and rectangle for a specific monitor.
    
    Args:
        monitor_rect: The monitor's rectangle to check
        
    Returns:
        Tuple of (TaskbarEdge, taskbar_rect)
    """
    try:
        shell32 = ctypes.WinDLL('shell32', use_last_error=True)
        
        # Query taskbar position
        abd = APPBARDATA()
        abd.cbSize = ctypes.sizeof(APPBARDATA)
        
        result = shell32.SHAppBarMessage(ABM_GETTASKBARPOS, ctypes.byref(abd))
        
        if result:
            taskbar_rect = QRect(
                abd.rc.left,
                abd.rc.top,
                abd.rc.right - abd.rc.left,
                abd.rc.bottom - abd.rc.top
            )
            
            # Check if taskbar intersects with our monitor
            if not monitor_rect.intersects(taskbar_rect):
                logger.debug("Taskbar not on target monitor")
                return TaskbarEdge.NONE, QRect()
            
            edge = TaskbarEdge(abd.uEdge)
            logger.debug(f"Taskbar detected: edge={edge.name}, rect={taskbar_rect}")
            return edge, taskbar_rect
        else:
            logger.debug("No taskbar detected via SHAppBarMessage")
            return TaskbarEdge.NONE, QRect()
            
    except Exception as e:
        logger.warning(f"Failed to detect taskbar: {e}")
        return TaskbarEdge.NONE, QRect()


def calculate_snap_position(
    dialog_size: QSize,
    side: DialogSide,
    monitor_index: int = 0,
    spacing: int = 10
) -> QPoint:
    """Calculate snap position for a dialog with taskbar awareness.
    
    Args:
        dialog_size: Size of the dialog window
        side: Which side to snap to (LEFT or RIGHT)
        monitor_index: Target monitor index (default: 0)
        spacing: Spacing from edges in pixels (default: 10)
        
    Returns:
        QPoint for dialog position
    """
    screens = QGuiApplication.screens()
    if not screens or monitor_index >= len(screens):
        logger.warning(f"Invalid monitor index {monitor_index}, using primary")
        monitor_index = 0
    
    screen = screens[monitor_index]
    available_geo = screen.availableGeometry()  # Excludes taskbar
    full_geo = screen.geometry()  # Full monitor rect
    
    logger.debug(f"Monitor {monitor_index}: available={available_geo}, full={full_geo}")
    
    # Detect taskbar
    taskbar_edge, taskbar_rect = get_taskbar_info(full_geo)
    
    # Calculate base position based on taskbar
    if taskbar_edge == TaskbarEdge.BOTTOM:
        # Taskbar at bottom - snap dialogs to bottom, opposite horizontal sides
        if side == DialogSide.LEFT:
            x = available_geo.left() + spacing
        else:  # RIGHT
            x = available_geo.right() - dialog_size.width() - spacing
        y = available_geo.bottom() - dialog_size.height() - spacing
        logger.debug(f"Taskbar BOTTOM: positioning {side.value} at ({x}, {y})")
        
    elif taskbar_edge == TaskbarEdge.TOP:
        # Taskbar at top - snap dialogs to top, opposite horizontal sides
        if side == DialogSide.LEFT:
            x = available_geo.left() + spacing
        else:  # RIGHT
            x = available_geo.right() - dialog_size.width() - spacing
        y = available_geo.top() + spacing
        logger.debug(f"Taskbar TOP: positioning {side.value} at ({x}, {y})")
        
    elif taskbar_edge == TaskbarEdge.LEFT:
        # Taskbar at left - snap dialogs to left side vertically
        x = available_geo.left() + spacing
        if side == DialogSide.LEFT:
            y = available_geo.top() + spacing
        else:  # RIGHT - stack below
            y = available_geo.top() + dialog_size.height() + spacing * 2
        logger.debug(f"Taskbar LEFT: positioning {side.value} at ({x}, {y})")
        
    elif taskbar_edge == TaskbarEdge.RIGHT:
        # Taskbar at right - snap dialogs to right side vertically
        x = available_geo.right() - dialog_size.width() - spacing
        if side == DialogSide.LEFT:
            y = available_geo.top() + spacing
        else:  # RIGHT - stack below
            y = available_geo.top() + dialog_size.height() + spacing * 2
        logger.debug(f"Taskbar RIGHT: positioning {side.value} at ({x}, {y})")
        
    else:
        # No taskbar detected - fallback to bottom corners
        if side == DialogSide.LEFT:
            x = available_geo.left() + spacing
        else:  # RIGHT
            x = available_geo.right() - dialog_size.width() - spacing
        y = available_geo.bottom() - dialog_size.height() - spacing
        logger.debug(f"No taskbar: fallback to bottom {side.value} at ({x}, {y})")
    
    return QPoint(x, y)


def get_centered_position(dialog_size: QSize, monitor_index: int = 0) -> QPoint:
    """Calculate centered position as fallback.
    
    Args:
        dialog_size: Size of the dialog window
        monitor_index: Target monitor index (default: 0)
        
    Returns:
        QPoint for centered dialog position
    """
    screens = QGuiApplication.screens()
    if not screens or monitor_index >= len(screens):
        monitor_index = 0
    
    screen = screens[monitor_index]
    geometry = screen.availableGeometry()
    
    x = geometry.left() + (geometry.width() - dialog_size.width()) // 2
    y = geometry.top() + (geometry.height() - dialog_size.height()) // 2
    
    logger.debug(f"Centered position on monitor {monitor_index}: ({x}, {y})")
    return QPoint(x, y)


def position_dialog_pair(
    main_size: QSize,
    sub_size: QSize,
    monitor_index: Optional[int] = None,
    spacing: int = 10
) -> Tuple[QPoint, QPoint]:
    """Calculate positions for a pair of dialogs (main and subsettings).
    
    Args:
        main_size: Size of the main dialog
        sub_size: Size of the subsettings dialog
        monitor_index: Target monitor index (None = auto-detect primary)
        spacing: Spacing from edges in pixels (default: 10)
        
    Returns:
        Tuple of (main_position, subsettings_position)
    """
    if monitor_index is None:
        # Use primary monitor
        primary = QGuiApplication.primaryScreen()
        screens = QGuiApplication.screens()
        monitor_index = screens.index(primary) if primary in screens else 0
    
    main_pos = calculate_snap_position(main_size, DialogSide.LEFT, monitor_index, spacing)
    sub_pos = calculate_snap_position(sub_size, DialogSide.RIGHT, monitor_index, spacing)
    
    logger.info(f"Dialog pair positioned: main={main_pos}, subsettings={sub_pos}, monitor={monitor_index}")
    return main_pos, sub_pos


__all__ = [
    'TaskbarEdge',
    'DialogSide',
    'get_taskbar_info',
    'calculate_snap_position',
    'get_centered_position',
    'position_dialog_pair',
]
