"""
Thumbnail Manager for DWM Thumbnails

Provides centralized management of Desktop Window Manager (DWM) thumbnails
with proper resource cleanup and error handling.
"""
import ctypes
import ctypes.wintypes
from core.logging import get_logger
from typing import Optional, Tuple, Dict, Any
from utils import window_validation as winval

# Import DWM API functions
dwmapi = ctypes.windll.dwmapi

# DWM thumbnail properties structure
class DWM_THUMBNAIL_PROPERTIES(ctypes.Structure):
    _fields_ = [
        ("dwFlags", ctypes.wintypes.DWORD),
        ("rcDestination", ctypes.wintypes.RECT),
        ("rcSource", ctypes.wintypes.RECT),
        ("opacity", ctypes.wintypes.BYTE),
        ("fVisible", ctypes.wintypes.BOOL),
        ("fSourceClientAreaOnly", ctypes.wintypes.BOOL)
    ]

# Constants
DWM_TNP_RECTDESTINATION = 0x00000001
DWM_TNP_RECTSOURCE = 0x00000002
DWM_TNP_OPACITY = 0x00000004
DWM_TNP_VISIBLE = 0x00000008
DWM_TNP_SOURCECLIENTAREAONLY = 0x00000010

class ThumbnailManager:
    """Manages DWM thumbnails with automatic cleanup."""
    
    def __init__(self):
        self._thumbnails: Dict[int, Any] = {}
        self._logger = get_logger(__name__)
        # Track last HRESULT for diagnostics and selective retry in callers
        self.last_hresult: Optional[int] = None
        
    def register_thumbnail(self, hwnd_dest: int, hwnd_src: int) -> Optional[int]:
        """Register a thumbnail relationship between two windows.
        
        Args:
            hwnd_dest: Handle to the destination window
            hwnd_src: Handle to the source window
            
        Returns:
            Thumbnail handle if successful, None otherwise
        """
        try:
            self.last_hresult = None
            # Pre-validate only source HWND; dest HWND may be transiently invalid right after show
            if not winval.is_valid_window(hwnd_src):
                self._logger.error(f"DwmRegisterThumbnail rejected: invalid src hwnd={hwnd_src}")
                return None

            # Validate that client rects are non-zero where possible
            try:
                src_rect = winval.get_client_rect(hwnd_src)
                if src_rect:
                    sw = max(0, int(src_rect[2] - src_rect[0]))
                    sh = max(0, int(src_rect[3] - src_rect[1]))
                    if sw == 0 or sh == 0:
                        self._logger.error(
                            f"DwmRegisterThumbnail rejected: zero-sized src client rect {src_rect} for hwnd={hwnd_src}"
                        )
                        return None
            except Exception:
                # Non-fatal; proceed if API not available
                pass

            thumbnail_id = ctypes.wintypes.HANDLE()
            hr = dwmapi.DwmRegisterThumbnail(
                ctypes.wintypes.HWND(hwnd_dest),
                ctypes.wintypes.HWND(hwnd_src),
                ctypes.byref(thumbnail_id)
            )
            
            self.last_hresult = int(hr)
            if hr == 0:  # S_OK
                self._thumbnails[hwnd_dest] = thumbnail_id
                self._logger.debug(f"Registered thumbnail for hwnd {hwnd_dest}")
                return thumbnail_id
            
            # One-shot synchronous retry on E_INVALIDARG after a short delay
            if hr == -2147024809:  # E_INVALIDARG
                try:
                    # Sleep ~15ms without importing time to avoid event loop complexities
                    ctypes.windll.kernel32.Sleep(15)
                except Exception:
                    pass
                thumbnail_id2 = ctypes.wintypes.HANDLE()
                hr2 = dwmapi.DwmRegisterThumbnail(
                    ctypes.wintypes.HWND(hwnd_dest),
                    ctypes.wintypes.HWND(hwnd_src),
                    ctypes.byref(thumbnail_id2)
                )
                self.last_hresult = int(hr2)
                if hr2 == 0:
                    self._thumbnails[hwnd_dest] = thumbnail_id2
                    self._logger.debug(
                        f"Registered thumbnail for hwnd {hwnd_dest} after retry"
                    )
                    return thumbnail_id2

            # Decode common HRESULTs for diagnostics and include window state
            try:
                dest_ok = None
                try:
                    dest_ok = winval.is_valid_window(hwnd_dest)
                except Exception:
                    dest_ok = None
                src_ok = winval.is_valid_window(hwnd_src)
                src_rect_dbg = None
                try:
                    src_rect_dbg = winval.get_client_rect(hwnd_src)
                except Exception:
                    src_rect_dbg = None
                if hr == -2147024809:
                    self._logger.error(
                        f"DwmRegisterThumbnail E_INVALIDARG. dest={hwnd_dest} valid_dest={dest_ok} src={hwnd_src} valid_src={src_ok} src_client_rect={src_rect_dbg}"
                    )
                else:
                    self._logger.error(
                        f"Failed to register thumbnail. HRESULT: {hr} dest={hwnd_dest} valid_dest={dest_ok} src={hwnd_src} valid_src={src_ok} src_client_rect={src_rect_dbg}"
                    )
            except Exception:
                self._logger.error(f"Failed to register thumbnail. HRESULT: {hr}")
            return None
            
        except Exception as e:
            self._logger.exception(f"Error registering thumbnail: {e}")
            return None
    
    def unregister_thumbnail(self, hwnd: int) -> bool:
        """Unregister a thumbnail relationship.
        
        Args:
            hwnd: Handle to the destination window
            
        Returns:
            True if successful, False otherwise
        """
        if hwnd not in self._thumbnails:
            return False
            
        try:
            handle = self._thumbnails.get(hwnd)
            if handle is None:
                return False
            hr = dwmapi.DwmUnregisterThumbnail(handle)
            del self._thumbnails[hwnd]
            self.last_hresult = int(hr)
            if hr == 0:
                self._logger.debug(f"Unregistered thumbnail for hwnd {hwnd}")
                return True
            # During shutdown the thumbnail handle may already be invalidated by DWM.
            # Treat E_INVALIDARG as benign to avoid noisy logs on close.
            if hr == -2147024809:  # E_INVALIDARG
                self._logger.debug(
                    f"DwmUnregisterThumbnail returned E_INVALIDARG during cleanup for hwnd {hwnd} (benign)"
                )
                return True
            self._logger.error(f"DwmUnregisterThumbnail failed. HRESULT: {hr} for hwnd {hwnd}")
            return False
        
        except Exception as e:
            self._logger.exception(f"Error unregistering thumbnail: {e}")
            return False
    
    def update_thumbnail_properties(self, hwnd: int, visible: Optional[bool] = None) -> bool:
        """Update only specific thumbnail properties (convenience method).
        
        Args:
            hwnd: Handle to the destination window
            visible: Whether the thumbnail is visible
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_thumbnail(hwnd, visible=visible)
        
    def update_thumbnail(
        self,
        hwnd: int,
        dest_rect: Optional[Tuple[int, int, int, int]] = None,
        src_rect: Optional[Tuple[int, int, int, int]] = None,
        opacity: Optional[float] = None,
        visible: Optional[bool] = None,
        source_client_area_only: Optional[bool] = None
    ) -> bool:
        """Update thumbnail properties.
        
        Args:
            hwnd: Handle to the destination window
            dest_rect: (left, top, right, bottom) destination rectangle
            src_rect: (left, top, right, bottom) source rectangle
            opacity: Opacity value (0.0-1.0)
            visible: Whether the thumbnail is visible
            source_client_area_only: Whether to show only the client area
            
        Returns:
            True if successful, False otherwise
        """
        if hwnd not in self._thumbnails:
            return False
            
        try:
            props = DWM_THUMBNAIL_PROPERTIES()
            props.dwFlags = 0
            
            if dest_rect:
                props.rcDestination = ctypes.wintypes.RECT(*dest_rect)
                props.dwFlags |= DWM_TNP_RECTDESTINATION
                
            if src_rect:
                props.rcSource = ctypes.wintypes.RECT(*src_rect)
                props.dwFlags |= DWM_TNP_RECTSOURCE
                
            if opacity is not None:
                props.opacity = int(opacity * 255)
                props.dwFlags |= DWM_TNP_OPACITY
                
            if visible is not None:
                props.fVisible = visible
                props.dwFlags |= DWM_TNP_VISIBLE
                
            if source_client_area_only is not None:
                props.fSourceClientAreaOnly = source_client_area_only
                props.dwFlags |= DWM_TNP_SOURCECLIENTAREAONLY
            
            hr = dwmapi.DwmUpdateThumbnailProperties(
                self._thumbnails[hwnd],
                ctypes.byref(props)
            )
            self.last_hresult = int(hr)
            if hr != 0:
                try:
                    self._logger.error(
                        "DwmUpdateThumbnailProperties failed. hr=%s flags=0x%X dest=%s src=%s opacity=%s visible=%s client_only=%s",
                        hr,
                        props.dwFlags,
                        (props.rcDestination.left, props.rcDestination.top, props.rcDestination.right, props.rcDestination.bottom),
                        (props.rcSource.left, props.rcSource.top, props.rcSource.right, props.rcSource.bottom),
                        props.opacity,
                        bool(props.fVisible),
                        bool(props.fSourceClientAreaOnly),
                    )
                except Exception:
                    pass
            
            # If successful and this was a visibility change, ensure we apply any pending composition attributes
            if hr == 0 and visible is not None and visible and hwnd in self._thumbnails:
                self._ensure_thumbnail_composition(hwnd)
            
            return hr == 0  # S_OK
            
        except Exception as e:
            self._logger.exception(f"Error updating thumbnail: {e}")
            return False
    
    def cleanup(self):
        """Clean up all registered thumbnails."""
        for hwnd in list(self._thumbnails.keys()):
            self.unregister_thumbnail(hwnd)
        self._thumbnails.clear()
        self._logger.debug("Cleaned up all thumbnails")
    
    def _ensure_thumbnail_composition(self, hwnd: int) -> bool:
        """Ensure consistent thumbnail composition attributes across all window types.
        
        This applies additional DWM composition attributes to ensure consistent rendering
        across different window types (browsers, games, etc).
        
        Args:
            hwnd: Handle to the destination window
            
        Returns:
            True if successful, False otherwise
        """
        if hwnd not in self._thumbnails:
            return False
            
        try:
            # Constants for DWM composition
            DWMWA_FORCE_ICONIC_REPRESENTATION = 7
            DWMWA_HAS_ICONIC_BITMAP = 10
            DWMWA_DISALLOW_PEEK = 11
            
            # Get thumbnail handle
            
            # Apply consistent composition attributes to normalize rendering
            # These settings help ensure consistent rendering across different window types
            value = ctypes.c_int(0)  # 0 = disabled
            
            # Ensure we're not using iconic representation (can cause rendering issues)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                ctypes.wintypes.HWND(hwnd),
                DWMWA_FORCE_ICONIC_REPRESENTATION,
                ctypes.byref(value),
                ctypes.sizeof(value)
            )
            
            # Ensure we're not using iconic bitmap (can cause rendering issues)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                ctypes.wintypes.HWND(hwnd),
                DWMWA_HAS_ICONIC_BITMAP,
                ctypes.byref(value),
                ctypes.sizeof(value)
            )
            
            # Disallow peek (can cause rendering issues)
            value = ctypes.c_int(1)  # 1 = enabled
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                ctypes.wintypes.HWND(hwnd),
                DWMWA_DISALLOW_PEEK,
                ctypes.byref(value),
                ctypes.sizeof(value)
            )
            
            self._logger.debug(f"Applied consistent composition attributes for hwnd {hwnd}")
            return True
            
        except Exception as e:
            self._logger.error(f"Error applying consistent composition attributes: {e}")
            return False
    
    def __del__(self):
        self.cleanup()
