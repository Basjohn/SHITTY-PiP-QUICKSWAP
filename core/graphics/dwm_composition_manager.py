"""
Centralized DWM Composition Attribute Management.

This module provides centralized management of DWM composition attributes
to ensure consistent rendering behavior across all overlay types and window sources.
"""
from __future__ import annotations

import ctypes
from ctypes import windll, c_int, byref, sizeof
from typing import Optional, Dict, Any
from enum import IntEnum

from PySide6.QtWidgets import QWidget

from core.logging import get_logger

logger = get_logger(__name__)

# DWM Window Attributes
class DWMWA(IntEnum):
    """DWM Window Attributes enumeration."""
    NCRENDERING_ENABLED = 1
    NCRENDERING_POLICY = 2
    TRANSITIONS_FORCEDISABLED = 3
    ALLOW_NCPAINT = 4
    CAPTION_BUTTON_BOUNDS = 5
    NONCLIENT_RTL_LAYOUT = 6
    FORCE_ICONIC_REPRESENTATION = 7
    FLIP3D_POLICY = 8
    EXTENDED_FRAME_BOUNDS = 9
    HAS_ICONIC_BITMAP = 10
    DISALLOW_PEEK = 11
    EXCLUDED_FROM_PEEK = 12
    CLOAK = 13
    CLOAKED = 14
    FREEZE_REPRESENTATION = 15
    PASSIVE_UPDATE_MODE = 16
    USE_HOSTBACKDROPBRUSH = 17
    USE_IMMERSIVE_DARK_MODE = 20
    WINDOW_CORNER_PREFERENCE = 33
    BORDER_COLOR = 34
    CAPTION_COLOR = 35
    TEXT_COLOR = 36
    VISIBLE_FRAME_BORDER_THICKNESS = 37
    SYSTEMBACKDROP_TYPE = 38


class DWMCompositionManager:
    """Centralized manager for DWM composition attributes."""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._applied_attributes: Dict[int, Dict[str, Any]] = {}  # hwnd -> attributes
        
        # Standard attribute sets for different overlay types
        self._overlay_attributes = {
            'border_overlay': {
                DWMWA.TRANSITIONS_FORCEDISABLED: 1,
                DWMWA.ALLOW_NCPAINT: 1,
                DWMWA.CLOAK: 0,
                DWMWA.FREEZE_REPRESENTATION: 0,
                DWMWA.USE_HOSTBACKDROPBRUSH: 1,
                DWMWA.SYSTEMBACKDROP_TYPE: 0,
                DWMWA.PASSIVE_UPDATE_MODE: 0,
            },
            'main_overlay': {
                DWMWA.TRANSITIONS_FORCEDISABLED: 1,
                DWMWA.ALLOW_NCPAINT: 1,
                DWMWA.CLOAK: 0,
                DWMWA.FREEZE_REPRESENTATION: 0,
                DWMWA.SYSTEMBACKDROP_TYPE: 0,
            },
            'thumbnail_host': {
                DWMWA.TRANSITIONS_FORCEDISABLED: 1,
                DWMWA.ALLOW_NCPAINT: 1,
                DWMWA.CLOAK: 0,
                DWMWA.FREEZE_REPRESENTATION: 0,
                DWMWA.USE_HOSTBACKDROPBRUSH: 0,
                DWMWA.SYSTEMBACKDROP_TYPE: 0,
            }
        }
    
    def apply_overlay_attributes(self, widget: QWidget, overlay_type: str = 'main_overlay') -> bool:
        """Apply DWM composition attributes for the specified overlay type.
        
        Args:
            widget: The widget to apply attributes to
            overlay_type: Type of overlay ('border_overlay', 'main_overlay', 'thumbnail_host')
            
        Returns:
            True if all attributes were applied successfully, False otherwise
        """
        if not widget:
            self._logger.error("Cannot apply DWM attributes: widget is None")
            return False
        
        try:
            hwnd = int(widget.winId())
            if hwnd == 0:
                self._logger.error("Cannot apply DWM attributes: invalid window handle")
                return False
            
            attributes = self._overlay_attributes.get(overlay_type, self._overlay_attributes['main_overlay'])
            return self._apply_attributes(hwnd, attributes, overlay_type)
            
        except Exception as e:
            self._logger.exception(f"Failed to apply DWM attributes for {overlay_type}: {e}")
            return False
    
    def apply_custom_attributes(self, widget: QWidget, attributes: Dict[DWMWA, int], context: str = "custom") -> bool:
        """Apply custom DWM composition attributes.
        
        Args:
            widget: The widget to apply attributes to
            attributes: Dictionary of DWMWA attributes and their values
            context: Context description for logging
            
        Returns:
            True if all attributes were applied successfully, False otherwise
        """
        if not widget:
            self._logger.error("Cannot apply custom DWM attributes: widget is None")
            return False
        
        try:
            hwnd = int(widget.winId())
            if hwnd == 0:
                self._logger.error("Cannot apply custom DWM attributes: invalid window handle")
                return False
            
            return self._apply_attributes(hwnd, attributes, context)
            
        except Exception as e:
            self._logger.exception(f"Failed to apply custom DWM attributes for {context}: {e}")
            return False
    
    def _apply_attributes(self, hwnd: int, attributes: Dict[DWMWA, int], context: str) -> bool:
        """Apply DWM attributes to the specified window handle.
        
        Args:
            hwnd: Window handle
            attributes: Dictionary of attributes to apply
            context: Context for logging
            
        Returns:
            True if all attributes were applied successfully, False otherwise
        """
        try:
            success_count = 0
            total_count = len(attributes)
            
            # Convert attributes to a list and handle some attributes specially
            attributes_list = list(attributes.items())
            
            for attr, value in attributes_list:
                try:
                    # FREEZE_REPRESENTATION is not supported on all Windows versions
                    # Skip warning if it fails but don't count it against success
                    is_optional = attr == DWMWA.FREEZE_REPRESENTATION
                    
                    c_value = c_int(value)
                    result = windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, int(attr), byref(c_value), sizeof(c_value)
                    )
                    
                    if result == 0:  # S_OK
                        success_count += 1
                    else:
                        if is_optional:
                            # For optional attributes, don't warn and adjust total count
                            total_count -= 1
                            self._logger.debug(f"Optional DWM attribute {attr.name} not supported, skipping")
                        else:
                            self._logger.warning(f"DWM attribute {attr.name} failed with result {result} for {context}")
                        
                except Exception as e:
                    if attr == DWMWA.FREEZE_REPRESENTATION:
                        # Silently adjust total count for FREEZE_REPRESENTATION failures
                        total_count -= 1
                        self._logger.debug("FREEZE_REPRESENTATION not supported on this Windows version, skipping")
                    else:
                        self._logger.warning(f"Failed to set DWM attribute {attr.name} for {context}: {e}")
            
            # Track applied attributes
            if hwnd not in self._applied_attributes:
                self._applied_attributes[hwnd] = {}
            self._applied_attributes[hwnd].update({attr.name: value for attr, value in attributes.items()})
            
            # Calculate success based on adjusted total count
            success = success_count == total_count and total_count > 0
            if success:
                self._logger.debug(f"Applied {success_count}/{total_count} DWM attributes for {context}")
            else:
                self._logger.warning(f"Applied only {success_count}/{total_count} DWM attributes for {context}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to apply DWM attributes for {context}: {e}")
            # No fallback: fail fast per user policy
            raise
    
    def normalize_for_window_type(self, widget: QWidget, source_hwnd: Optional[int] = None) -> bool:
        """Normalize DWM attributes based on the source window type.
        
        This method analyzes the source window and applies appropriate DWM attributes
        to ensure consistent rendering regardless of the source window's characteristics.
        
        Args:
            widget: The overlay widget to normalize
            source_hwnd: Optional source window handle for analysis
            
        Returns:
            True if normalization was successful, False otherwise
        """
        try:
            # Get window class and style information if source_hwnd is provided
            window_info = self._analyze_source_window(source_hwnd) if source_hwnd else {}
            
            # Determine appropriate attribute set based on window analysis
            if window_info.get('is_game_window', False):
                # Game windows often need special handling
                attributes = {
                    DWMWA.TRANSITIONS_FORCEDISABLED: 1,
                    DWMWA.ALLOW_NCPAINT: 1,
                    DWMWA.CLOAK: 0,
                    DWMWA.FREEZE_REPRESENTATION: 0,
                    DWMWA.USE_HOSTBACKDROPBRUSH: 0,  # Don't use host brush for games
                    DWMWA.SYSTEMBACKDROP_TYPE: 0,
                    DWMWA.PASSIVE_UPDATE_MODE: 1,  # Use passive updates for games
                }
            elif window_info.get('is_browser_window', False):
                # Browser windows need consistent backdrop handling
                attributes = {
                    DWMWA.TRANSITIONS_FORCEDISABLED: 1,
                    DWMWA.ALLOW_NCPAINT: 1,
                    DWMWA.CLOAK: 0,
                    DWMWA.FREEZE_REPRESENTATION: 0,
                    DWMWA.USE_HOSTBACKDROPBRUSH: 1,
                    DWMWA.SYSTEMBACKDROP_TYPE: 0,
                }
            else:
                # Standard windows (Notepad, etc.)
                attributes = self._overlay_attributes['main_overlay']
            
            return self.apply_custom_attributes(widget, attributes, f"normalized_for_{window_info.get('class_name', 'unknown')}")
            
        except Exception as e:
            self._logger.exception(f"Failed to normalize DWM attributes: {e}")
            return False
    
    def _analyze_source_window(self, hwnd: int) -> Dict[str, Any]:
        """Analyze source window characteristics.
        
        Args:
            hwnd: Source window handle
            
        Returns:
            Dictionary containing window analysis results
        """
        try:
            # Get window class name
            class_name_buffer = ctypes.create_unicode_buffer(256)
            windll.user32.GetClassNameW(hwnd, class_name_buffer, 256)
            class_name = class_name_buffer.value.lower()
            
            # Get window text (title)
            title_buffer = ctypes.create_unicode_buffer(512)
            windll.user32.GetWindowTextW(hwnd, title_buffer, 512)
            window_title = title_buffer.value.lower()
            
            # Analyze window characteristics
            is_game_window = any(keyword in class_name for keyword in ['unreal', 'unity', 'directx', 'opengl']) or \
                           any(keyword in window_title for keyword in ['game', 'launcher'])
            
            is_browser_window = any(keyword in class_name for keyword in ['chrome', 'firefox', 'edge', 'browser']) or \
                              any(keyword in window_title for keyword in ['chrome', 'firefox', 'edge', 'browser'])
            
            return {
                'class_name': class_name,
                'window_title': window_title,
                'is_game_window': is_game_window,
                'is_browser_window': is_browser_window,
            }
            
        except Exception as e:
            self._logger.warning(f"Failed to analyze source window {hwnd}: {e}")
            return {}
    
    def get_applied_attributes(self, widget: QWidget) -> Optional[Dict[str, Any]]:
        """Get the DWM attributes applied to the specified widget.
        
        Args:
            widget: The widget to query
            
        Returns:
            Dictionary of applied attributes or None if not found
        """
        try:
            hwnd = int(widget.winId())
            return self._applied_attributes.get(hwnd)
        except Exception:
            return None
    
    def clear_applied_attributes(self, widget: QWidget) -> None:
        """Clear tracking of applied attributes for the specified widget.
        
        Args:
            widget: The widget to clear tracking for
        """
        try:
            hwnd = int(widget.winId())
            if hwnd in self._applied_attributes:
                del self._applied_attributes[hwnd]
        except Exception as e:
            self._logger.warning(f"Failed to clear applied attributes: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DWM composition manager statistics.
        
        Returns:
            Dictionary containing statistics
        """
        return {
            'tracked_windows': len(self._applied_attributes),
            'overlay_types': list(self._overlay_attributes.keys()),
        }


# Global instance for centralized access
_dwm_composition_manager: Optional[DWMCompositionManager] = None

def get_dwm_composition_manager() -> DWMCompositionManager:
    """Get the global DWM composition manager instance."""
    global _dwm_composition_manager
    if _dwm_composition_manager is None:
        _dwm_composition_manager = DWMCompositionManager()
    return _dwm_composition_manager
