"""
Centralized overlay geometry persistence utilities.

Provides helpers to compute and apply nearest-corner persistence for overlays
(DWM and Docking). Callers should use these helpers to avoid duplication.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict

from PySide6.QtCore import QRect, QSize, QPoint
from PySide6.QtGui import QGuiApplication

from utils.window.monitors import (
    get_best_monitor_for_window,
    get_available_geometry_for_monitor,
    ensure_within_available_desktop,
    _constrain_to_virtual_screen,
)


def nearest_corner_state_from_rect(rect: QRect, constrain: str = "available") -> Dict[str, int | str]:
    """Compute a persisted state from the given window rect using nearest corner.

    Returns a dict with: corner (str), width, height, monitor_index (ints).
    constrain: "available" (work area) or "physical" (full screen geometry).
    
    Note: rect is assumed to be in logical pixels (Qt widget coordinates).
    When constrain="physical", we convert both rect and screen to physical pixels for comparison.
    """
    if rect is None or rect.isEmpty():
        return {}
    mon_idx = get_best_monitor_for_window(rect)
    
    # Choose anchor space based on constraint
    if constrain == "physical":
        try:
            # Suppress excessive debug spam (enable only for debugging persistence issues)
            # from core.logging import get_logger
            # logger = get_logger("PERSIST_STATE")
            # logger.debug(f"[PHYS_CONVERT] Starting physical conversion: logical_rect=({rect.x()},{rect.y()},{rect.width()}x{rect.height()})")
            
            screens = QGuiApplication.screens()
            if 0 <= mon_idx < len(screens):
                screen = screens[mon_idx]
                dpr = screen.devicePixelRatio()
            else:
                screen = QGuiApplication.primaryScreen()
                dpr = screen.devicePixelRatio()
            
            # logger.debug(f"[PHYS_CONVERT] DPR={dpr}")
            
            # Convert rect from logical to physical pixels for physical comparison
            phys_rect_x = int(rect.x() * dpr)
            phys_rect_y = int(rect.y() * dpr)
            phys_rect_w = int(rect.width() * dpr)
            phys_rect_h = int(rect.height() * dpr)
            
            # logger.debug(f"[PHYS_CONVERT] Converted to physical: ({phys_rect_x},{phys_rect_y},{phys_rect_w}x{phys_rect_h})")
            
            # Get physical screen geometry
            logical_geom = screen.geometry()
            phys_screen_x = int(logical_geom.x() * dpr)
            phys_screen_y = int(logical_geom.y() * dpr)
            phys_screen_w = int(logical_geom.width() * dpr)
            phys_screen_h = int(logical_geom.height() * dpr)
            avail = QRect(phys_screen_x, phys_screen_y, phys_screen_w, phys_screen_h)
            
            # Calculate corners in physical space
            tl = (avail.left(), avail.top())
            tr = (avail.right() - phys_rect_w + 1, avail.top())
            bl = (avail.left(), avail.bottom() - phys_rect_h + 1)
            br = (avail.right() - phys_rect_w + 1, avail.bottom() - phys_rect_h + 1)
            current = (phys_rect_x, phys_rect_y)
            
            # Store physical dimensions for later restoration
            rect_width = phys_rect_w
            rect_height = phys_rect_h
        except Exception as e:
            # Fallback to logical if physical conversion fails
            try:
                from core.logging import get_logger
                logger = get_logger("PERSIST_STATE")
                logger.warning(f"[PHYS_CONVERT] FAILED - falling back to logical: {e}")
            except Exception:
                pass
            avail = get_available_geometry_for_monitor(mon_idx)
            tl = (avail.left(), avail.top())
            tr = (avail.right() - rect.width() + 1, avail.top())
            bl = (avail.left(), avail.bottom() - rect.height() + 1)
            br = (avail.right() - rect.width() + 1, avail.bottom() - rect.height() + 1)
            current = (rect.x(), rect.y())
            rect_width = rect.width()
            rect_height = rect.height()
    else:
        # Logical space for work area
        avail = get_available_geometry_for_monitor(mon_idx)
        tl = (avail.left(), avail.top())
        tr = (avail.right() - rect.width() + 1, avail.top())
        bl = (avail.left(), avail.bottom() - rect.height() + 1)
        br = (avail.right() - rect.width() + 1, avail.bottom() - rect.height() + 1)
        current = (rect.x(), rect.y())
        rect_width = rect.width()
        rect_height = rect.height()

    def dist2(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    candidates = {
        "top_left": tl,
        "top_right": tr,
        "bottom_left": bl,
        "bottom_right": br,
    }
    nearest = min(candidates.items(), key=lambda kv: dist2(current, kv[1]))[0]
    
    # Debug log only on verbose setting
    # (Suppressed by default to reduce spam during drag/resize)
    # Uncomment for debugging persistence issues:
    # try:
    #     from core.logging import get_logger
    #     logger = get_logger("PERSIST_STATE")
    #     logger.debug(f"corner={nearest}, constrain={constrain}")
    # except Exception:
    #     pass
    
    return {
        "corner": nearest,
        "width": int(rect_width),
        "height": int(rect_height),
        "monitor_index": int(mon_idx),
    }


def geometry_from_state(
    state: dict,
    avail: QRect,
    min_size: QSize,
    aspect: Optional[float] = None,
    insets: Tuple[int, int] = (0, 0),
    constrain: str = "available",
) -> Optional[QRect]:
    """Compute a QRect to apply from a persisted state.

    - If aspect is provided, size is adjusted to preserve aspect within available area
      while staying close to the saved outer width.
    - Insets are per-side content insets (left/right = ix, top/bottom = iy).
    - min_size clamps the final outer size.
    - constrain: "available" (default) clamps within monitor work area via
      ensure_within_available_desktop. "physical" clamps within the virtual
      desktop using full monitor geometry via _constrain_to_virtual_screen.
    """
    try:
        if not isinstance(state, dict) or not state:
            return None
        corner = str(state.get("corner", "")).lower()
        w_saved = int(state.get("width", 0))
        h_saved = int(state.get("height", 0))
        if w_saved <= 0 or h_saved <= 0 or corner not in (
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ):
            return None

        ix, iy = max(0, int(insets[0])), max(0, int(insets[1]))
        # Start from saved outer width, clamp to available
        outer_w = max(min_size.width(), min(w_saved, avail.width()))
        if aspect and aspect > 0:
            # Compute inner size then rebuild outer with insets
            inner_w = max(1, outer_w - 2 * ix)
            inner_h = max(1, int(round(inner_w / aspect)))
            # Clamp inner height to available height minus insets
            inner_h = min(inner_h, max(1, avail.height() - 2 * iy))
            inner_w = max(1, int(round(inner_h * aspect)))
            out_w = inner_w + 2 * ix
            out_h = inner_h + 2 * iy
        else:
            # No aspect provided: use saved height clamped
            out_w = outer_w
            out_h = max(min_size.height(), min(h_saved, avail.height()))

        # Final safety clamps
        out_w = max(min_size.width(), min(out_w, avail.width()))
        out_h = max(min_size.height(), min(out_h, avail.height()))

        # Corner position
        if corner == "top_left":
            x = avail.left()
            y = avail.top()
        elif corner == "top_right":
            x = avail.right() - out_w + 1
            y = avail.top()
        elif corner == "bottom_left":
            x = avail.left()
            y = avail.bottom() - out_h + 1
        else:  # bottom_right
            x = avail.right() - out_w + 1
            y = avail.bottom() - out_h + 1

        # Debug logging for persistence calculation
        # Suppress excessive debug spam during normal operation
        # Uncomment for debugging corner positioning issues:
        # try:
        #     from core.logging import get_logger
        #     logger = get_logger("DOCK_PERSIST")
        #     logger.debug(f"Corner calculation: corner={corner}, avail=({avail.x()},{avail.y()},{avail.width()}x{avail.height()}), out_size={out_w}x{out_h}")
        #     logger.debug(f"Calculated position BEFORE constraint: ({x},{y})")
        # except Exception:
        #     pass

        if constrain == "physical":
            pos = _constrain_to_virtual_screen(QPoint(x, y), QSize(out_w, out_h))
            # Suppress debug spam - uncomment for debugging constraint issues:
            # try:
            #     from core.logging import get_logger
            #     logger = get_logger("DOCK_PERSIST")
            #     logger.debug(f"Position AFTER _constrain_to_virtual_screen: ({pos.x()},{pos.y()})")
            # except Exception:
            #     pass
        else:
            # Default: available/work-area clamping
            pos = ensure_within_available_desktop(QPoint(x, y), QSize(out_w, out_h))
            # Suppress debug spam - uncomment for debugging constraint issues:
            # try:
            #     from core.logging import get_logger
            #     logger = get_logger("DOCK_PERSIST")
            #     logger.debug(f"Position AFTER ensure_within_available_desktop: ({pos.x()},{pos.y()})")
            # except Exception:
            #     pass
        return QRect(pos, QSize(out_w, out_h))
    except Exception:
        return None
