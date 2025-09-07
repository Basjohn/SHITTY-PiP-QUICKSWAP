"""
Centralized Z-Order Management for Overlays and Border Overlays.

This module provides centralized z-order enforcement for all overlay types,
ensuring consistent stacking order and eliminating redundant z-order calls.
"""
from __future__ import annotations

from ctypes import windll
from typing import Dict, Optional, Any
from weakref import WeakValueDictionary

from PySide6.QtWidgets import QWidget
 

from core.logging import get_logger
from core.logging.logger_impl import throttled  # call-site helper for hot paths
from core.threading import ThreadManager

logger = get_logger(__name__)

# Win32 constants for SetWindowPos
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2

# SetWindowPos flags
class SWP:
    NOSIZE = 0x0001
    NOMOVE = 0x0002
    NOZORDER = 0x0004
    NOREDRAW = 0x0008
    NOACTIVATE = 0x0010
    DRAWFRAME = 0x0020
    FRAMECHANGED = 0x0020
    SHOWWINDOW = 0x0040
    HIDEWINDOW = 0x0080
    NOCOPYBITS = 0x0100
    NOOWNERZORDER = 0x0200
    NOREPOSITION = 0x0200
    NOSENDCHANGING = 0x0400
    DEFERERASE = 0x2000
    ASYNCWINDOWPOS = 0x4000


class ZOrderManager:
    """Centralized z-order management for overlays and border overlays."""
    
    def __init__(self):
        # Lock-free: All z-order operations confined to UI thread
        self._logger = get_logger(__name__)
        # Throttled emitters for high-frequency messages
        self._tdebug_scheduled = throttled(self._logger.debug, "zorder:scheduled", 200)
        self._tdebug_batch = throttled(self._logger.debug, "zorder:batch", 500)
        
        # Track overlay relationships
        self._main_overlays: WeakValueDictionary[str, QWidget] = WeakValueDictionary()
        self._border_overlays: WeakValueDictionary[str, QWidget] = WeakValueDictionary()
        self._overlay_relationships: Dict[str, str] = {}  # border_id -> main_id
        
        # Track window handles for lifecycle management
        self._valid_handles: Dict[int, str] = {}  # hwnd -> overlay_id
        self._destroyed_handles: set[int] = set()  # Track destroyed handles
        
        # Z-order enforcement state with debouncing
        self._enforcement_pending = False
        self._enforcement_token: int = 0
        self._pending_overlays: set[str] = set()  # Overlays pending enforcement
        
        # Performance tracking
        self._last_enforcement_time = 0
        self._enforcement_count = 0
        
        # Debouncing configuration
        self._debounce_delay_ms = 10  # Coalesce calls within 10ms
        
    def register_main_overlay(self, overlay_id: str, overlay_widget: QWidget) -> None:
        """Register a main overlay for z-order management.
        
        Args:
            overlay_id: Unique identifier for the overlay
            overlay_widget: The overlay widget instance
        """
        # Lock-free: UI thread only access
        self._main_overlays[overlay_id] = overlay_widget
        self._logger.debug(f"Registered main overlay: {overlay_id} (handle tracking deferred)")
    
    def register_border_overlay(self, border_id: str, border_widget: QWidget, main_overlay_id: str) -> None:
        """Register a border overlay for z-order management.
        
        Args:
            border_id: Unique identifier for the border overlay
            border_widget: The border overlay widget instance
            main_overlay_id: ID of the associated main overlay
        """
        # Lock-free: UI thread only access
        self._border_overlays[border_id] = border_widget
        self._overlay_relationships[border_id] = main_overlay_id
        self._logger.debug(f"Registered border overlay: {border_id} for main overlay: {main_overlay_id} (handle tracking deferred)")
    
    def unregister_main_overlay(self, overlay_id: str) -> None:
        """Unregister a main overlay from z-order management.
        
        Args:
            overlay_id: Unique identifier for the overlay
        """
        # Lock-free: UI thread only access
        if overlay_id in self._main_overlays:
            overlay_widget = self._main_overlays[overlay_id]
            
            # Mark window handle as destroyed
            try:
                hwnd = int(overlay_widget.winId())
                self._destroyed_handles.add(hwnd)
                if hwnd in self._valid_handles:
                    del self._valid_handles[hwnd]
            except Exception:
                pass
            
            del self._main_overlays[overlay_id]
            
            # Remove from pending enforcement
            self._pending_overlays.discard(overlay_id)
            
            self._logger.debug(f"Unregistered main overlay: {overlay_id}")
    
    def unregister_border_overlay(self, border_id: str) -> None:
        """Unregister a border overlay from z-order management.
        
        Args:
            border_id: Unique identifier for the border overlay
        """
        # Lock-free: UI thread only access
        if border_id in self._border_overlays:
            # Clean up border overlay
            border_widget = self._border_overlays[border_id]
            
            # Mark window handle as destroyed
            try:
                hwnd = int(border_widget.winId())
                self._destroyed_handles.add(hwnd)
                if hwnd in self._valid_handles:
                    del self._valid_handles[hwnd]
            except Exception:
                pass
            
            try:
                border_widget.hide()
                border_widget.close()
            except Exception as e:
                self._logger.warning(f"Error closing border overlay {border_id}: {e}")
            
            del self._border_overlays[border_id]
                
            if border_id in self._overlay_relationships:
                del self._overlay_relationships[border_id]
                
            # Remove from pending enforcement
            self._pending_overlays.discard(border_id)
                
            self._logger.debug(f"Unregistered border overlay: {border_id}")
    
    def get_border_overlay(self, border_id: str) -> Optional[QWidget]:
        """Get a border overlay by ID.
        
        Args:
            border_id: Unique identifier for the border overlay
            
        Returns:
            Border overlay widget or None if not found
        """
        # Lock-free: UI thread only access
        return self._border_overlays.get(border_id)
    
    def cleanup_destroyed_handles(self) -> None:
        """Clean up tracking for destroyed window handles.
        
        This method should be called periodically to prevent memory leaks
        from accumulating destroyed handle references.
        """
        # Lock-free: UI thread only access
        # Limit the size of destroyed handles set to prevent memory leaks
        if len(self._destroyed_handles) > 100:
            # Keep only the most recent 50 destroyed handles
            recent_handles = list(self._destroyed_handles)[-50:]
            self._destroyed_handles = set(recent_handles)
            self._logger.debug(f"Cleaned up destroyed handles, keeping {len(recent_handles)} recent entries")
    
    def is_handle_destroyed(self, hwnd: int) -> bool:
        """Check if a window handle has been marked as destroyed.
        
        Args:
            hwnd: Window handle to check
            
        Returns:
            True if handle is known to be destroyed
        """
        # Lock-free: UI thread only access
        return hwnd in self._destroyed_handles
    
    def enforce_z_order(self, overlay_id: str) -> bool:
        """Enforce z-order for the specified overlay and its border.
        
        Uses debouncing to coalesce rapid calls and prevent window handle invalidation.
        
        Args:
            overlay_id: ID of the overlay to enforce z-order for
            
        Returns:
            True if enforcement was scheduled/executed, False if overlay not found
        """
        # Lock-free: UI thread only access
        # Check if overlay exists
        main_overlay = self._main_overlays.get(overlay_id)
        if not main_overlay:
            self._logger.warning(f"Main overlay {overlay_id} not found for z-order enforcement")
            return False
        
        # Add to pending set for debounced enforcement
        self._pending_overlays.add(overlay_id)
        self._tdebug_scheduled(f"Scheduled z-order enforcement for {overlay_id} (debounced)")
        
        # Start or restart debounce using ThreadManager.single_shot with token coalescing
        self._enforcement_token += 1
        current_token = self._enforcement_token

        def _run():
            # Only execute if this invocation is the latest scheduled
            # Lock-free: UI thread only access
            if current_token != self._enforcement_token:
                return
            self._execute_pending_enforcement()

        ThreadManager.single_shot(self._debounce_delay_ms, _run)
        
        return True
    
    def enforce_z_order_immediate(self, overlay_id: str) -> bool:
        """Enforce z-order immediately without debouncing.
        
        Use this for critical operations where immediate enforcement is required.
        
        Args:
            overlay_id: ID of the overlay to enforce z-order for
            
        Returns:
            True if successful, False otherwise
        """
        # Lock-free: UI thread only access
        try:
            # Get the main overlay
            main_overlay = self._main_overlays.get(overlay_id)
            if not main_overlay:
                self._logger.warning(f"Main overlay {overlay_id} not found for immediate z-order enforcement")
                return False
            
            # Find associated border overlay
            border_overlay = None
            for bid, mid in self._overlay_relationships.items():
                if mid == overlay_id:
                    border_overlay = self._border_overlays.get(bid)
                    break
            
            result = self._enforce_z_order_native(main_overlay, border_overlay)
            self._logger.debug(f"Immediate z-order enforcement for {overlay_id}: {'success' if result else 'failed'}")
            return result
            
        except Exception as e:
            self._logger.exception(f"Failed immediate z-order enforcement for {overlay_id}: {e}")
            return False
    
    def _execute_pending_enforcement(self) -> None:
        """Execute z-order enforcement for all pending overlays.
        
        This is called by the debounce timer to batch process all pending enforcement requests.
        """
        # Lock-free: UI thread only access
        if not self._pending_overlays:
            return
        
        pending_copy = self._pending_overlays.copy()
        self._pending_overlays.clear()
        
        self._tdebug_batch(f"Executing batched z-order enforcement for {len(pending_copy)} overlays: {pending_copy}")
        
        success_count = 0
        for overlay_id in pending_copy:
            try:
                # Get the main overlay
                main_overlay = self._main_overlays.get(overlay_id)
                if not main_overlay:
                    self._logger.warning(f"Main overlay {overlay_id} no longer exists during batched enforcement")
                    continue
                
                # Find associated border overlay
                border_overlay = None
                for bid, mid in self._overlay_relationships.items():
                    if mid == overlay_id:
                        border_overlay = self._border_overlays.get(bid)
                        break
                
                # Execute native enforcement
                if self._enforce_z_order_native(main_overlay, border_overlay):
                    success_count += 1
                    self._logger.debug(f"Batched z-order enforcement successful for {overlay_id}")
                else:
                    self._logger.warning(f"Batched z-order enforcement failed for {overlay_id}")
                    # Clean up destroyed handles periodically on failures
                    self.cleanup_destroyed_handles()
                    
            except Exception as e:
                self._logger.exception(f"Error during batched z-order enforcement for {overlay_id}: {e}")
        
        self._tdebug_batch(f"Batched z-order enforcement completed: {success_count}/{len(pending_copy)} successful")
        
        # No timer cleanup necessary when using tokenized single_shot scheduling
    
    def _enforce_z_order_native(self, main_overlay: QWidget, border_overlay: Optional[QWidget] = None) -> bool:
        """Perform native z-order enforcement using Win32 APIs.
        
        Args:
            main_overlay: The main overlay widget
            border_overlay: Optional border overlay widget
            
        Returns:
            True if successful, False otherwise
        """
        try:
            user32 = windll.user32
            SetWindowPos = user32.SetWindowPos
            GetLastError = windll.kernel32.GetLastError
            IsWindow = user32.IsWindow
            
            # Flags for TOPMOST enforcement without activation or size/position changes
            flags = (
                SWP.NOMOVE |
                SWP.NOSIZE |
                SWP.NOACTIVATE |
                SWP.NOSENDCHANGING
            )
            
            # Ensure main overlay has a native window handle and is visible
            from PySide6.QtCore import Qt
            if not main_overlay.isVisible():
                self._logger.debug("Main overlay not visible, skipping z-order enforcement")
                return False
                
            if not main_overlay.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
                self._logger.debug("Main overlay not created yet, forcing native handle creation")
                main_overlay.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
                main_overlay.winId()  # Force handle creation
            
            # Get and validate main overlay window handle (fresh each time to handle Qt recreation)
            try:
                main_hwnd = int(main_overlay.winId())
            except Exception as e:
                self._logger.warning(f"Failed to get main overlay window handle: {e}")
                return False
            
            if main_hwnd == 0 or not IsWindow(main_hwnd):
                self._logger.debug(f"Main overlay has invalid window handle: {main_hwnd}")
                return False
            
            # Use Qt's raise_() first as it's safer than direct Win32 calls
            try:
                main_overlay.raise_()
            except Exception as e:
                self._logger.debug(f"Qt raise_() failed for main overlay: {e}")
            
            # Then apply TOPMOST via Win32 for stronger enforcement
            ok_main = SetWindowPos(main_hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
            if not ok_main:
                error_code = GetLastError()
                # Error 1400 means invalid window handle - widget was destroyed
                if error_code == 1400:
                    self._logger.debug(f"Main overlay window {main_hwnd} was destroyed (error 1400)")
                    return False
                else:
                    self._logger.warning(f"SetWindowPos failed for main overlay hwnd={main_hwnd}, error={error_code}")
            
            ok_border = True
            if border_overlay and border_overlay.isVisible():
                # Ensure border overlay has a native window handle
                if not border_overlay.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
                    self._logger.debug("Border overlay not created yet, forcing native handle creation")
                    border_overlay.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
                    border_overlay.winId()  # Force handle creation
                
                # Get and validate border overlay window handle (fresh each time to handle Qt recreation)
                try:
                    border_hwnd = int(border_overlay.winId())
                except Exception as e:
                    self._logger.debug(f"Failed to get border overlay window handle: {e}")
                    ok_border = False
                    border_hwnd = 0
                
                if border_hwnd == 0 or not IsWindow(border_hwnd):
                    self._logger.debug(f"Border overlay has invalid window handle: {border_hwnd}")
                    ok_border = False
                else:
                    # Use Qt's raise_() first
                    try:
                        border_overlay.raise_()
                        border_overlay.sync_with_target()
                    except Exception as e:
                        self._logger.debug(f"Qt raise_() or sync failed for border overlay: {e}")
                    
                    # Then apply TOPMOST via Win32
                    ok_border = SetWindowPos(border_hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
                    if not ok_border:
                        error_code = GetLastError()
                        if error_code == 1400:
                            self._logger.debug(f"Border overlay window {border_hwnd} was destroyed (error 1400)")
                        else:
                            self._logger.warning(f"SetWindowPos failed for border overlay hwnd={border_hwnd}, error={error_code}")
            
            success = bool(ok_main and ok_border)
            if success:
                self._enforcement_count += 1
                self._logger.debug(f"Z-order enforcement successful (count: {self._enforcement_count})")
            else:
                self._logger.debug(f"Z-order enforcement failed: main_ok={bool(ok_main)}, border_ok={bool(ok_border)}")
                
            return success
            
        except Exception as e:
            self._logger.error(f"Native z-order enforcement failed: {e}")
            return False
    
    def schedule_z_order_enforcement(self, overlay_id: str, delay_ms: int = 0) -> None:
        """Schedule z-order enforcement with optional delay.
        
        Args:
            overlay_id: ID of the overlay to enforce z-order for
            delay_ms: Delay in milliseconds before enforcement
        """
        def enforce():
            self.enforce_z_order(overlay_id)
        
        try:
            if delay_ms > 0:
                ThreadManager.single_shot(delay_ms, enforce)
            else:
                ThreadManager.run_on_ui_thread(enforce)
        except Exception as e:
            self._logger.warning(f"Failed to schedule z-order enforcement: {e}")
            # Fallback to immediate enforcement
            self.enforce_z_order(overlay_id)
    
    def enforce_all_z_orders(self) -> bool:
        """Enforce z-order for all registered overlays.
        
        Returns:
            True if all enforcements succeeded, False otherwise
        """
        # Lock-free: UI thread only access
        success = True
        for overlay_id in list(self._main_overlays.keys()):
            if not self.enforce_z_order(overlay_id):
                success = False
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get z-order management statistics.
        
        Returns:
            Dictionary containing statistics
        """
        # Lock-free: UI thread only access
        return {
            'main_overlays_count': len(self._main_overlays),
            'border_overlays_count': len(self._border_overlays),
            'enforcement_count': self._enforcement_count,
            'relationships_count': len(self._overlay_relationships)
        }
