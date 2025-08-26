"""
Unified Z-Order Management System

Centralizes all z-order enforcement with built-in context menu priority,
eliminating race conditions between ResourceManager and OverlayContextMenu.
"""

import logging
import weakref
from typing import Dict, Optional, Set
try:
    from typing import WeakSet
except ImportError:
    # Python 3.11 doesn't have WeakSet in typing, use weakref
    from weakref import WeakSet
from enum import Enum, auto
from dataclasses import dataclass
from threading import RLock

from PySide6.QtCore import QObject, QCoreApplication
from PySide6.QtWidgets import QWidget, QMenu

try:
    from win32gui import SetWindowPos, GetLastError, IsWindow
    from win32con import HWND_TOPMOST, HWND_TOP, SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE
    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False


class ZOrderPriority(Enum):
    """Z-order enforcement priority levels"""
    NORMAL = auto()
    CONTEXT_MENU = auto()
    CRITICAL = auto()


@dataclass
class OverlayInfo:
    """Information about a registered overlay with integrated border approach"""
    overlay_id: str
    main_widget: weakref.ReferenceType
    # No separate border_widget reference needed with integrated approach
    # Legacy support maintained for backward compatibility
    border_widget: Optional[weakref.ReferenceType] = None
    is_active: bool = True


class ZOrderManager(QObject):
    """
    Unified z-order management with context menu priority.
    
    Eliminates race conditions by centralizing all z-order enforcement
    and providing built-in context menu priority handling.
    """
    
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._lock = RLock()
        # Throttled emitters for high-frequency debug lines
        try:
            from core.logging.logger_impl import throttled, log_dedupe  # lightweight import
            self._tdebug_reg = throttled(self._logger.debug, "zorder:register", 200)
            self._tdebug_unreg = throttled(self._logger.debug, "zorder:unregister", 200)
            self._tdebug_ctx = throttled(self._logger.debug, "zorder:context", 200)
            self._tdebug_enf = throttled(self._logger.debug, "zorder:enforce", 200)
            self._dwarn_native = log_dedupe(self._logger.warning, "zorder:native_warn", 2000)
        except Exception:
            # Helpers unavailable early in startup; fall back to direct logger
            self._tdebug_reg = self._logger.debug
            self._tdebug_unreg = self._logger.debug
            self._tdebug_ctx = self._logger.debug
            self._tdebug_enf = self._logger.debug
            self._dwarn_native = self._logger.warning
        
        # Registry of overlays
        self._overlays: Dict[str, OverlayInfo] = {}
        
        # Context menu state
        self._active_context_menus: WeakSet[QMenu] = WeakSet()
        self._context_menu_overlay_id: Optional[str] = None
        
        # Enforcement state (debounced via centralized ThreadManager)
        self._debounce_pending: bool = False
        self._debounce_delay_ms: int = 16
        self._pending_enforcements: Set[str] = set()
        
        # Lazy-initialized UI coalescer (created when Qt is ready)
        self._coalescer = None

        self._logger.info("ZOrderManager initialized")
    
    def register_overlay(self, overlay_id: str, main_widget: QWidget, 
                        border_widget: Optional[QWidget] = None) -> bool:
        """Register an overlay for z-order management
        
        Args:
            overlay_id: Unique identifier for the overlay
            main_widget: Main widget of the overlay (with integrated borders)
            border_widget: Optional border widget (maintained for backward compatibility)
                          With integrated borders, this parameter is typically None
        
        Returns:
            bool: True if registration was successful
        """
        with self._lock:
            try:
                main_ref = weakref.ref(main_widget)
                # Support legacy border_widget parameter but not required for integrated borders
                border_ref = weakref.ref(border_widget) if border_widget else None
                
                self._overlays[overlay_id] = OverlayInfo(
                    overlay_id=overlay_id,
                    main_widget=main_ref,
                    border_widget=border_ref
                )
                
                self._tdebug_reg(f"Registered overlay {overlay_id} with border={border_widget is not None}")
                return True
                
            except Exception as e:
                self._logger.error(f"Failed to register overlay {overlay_id}: {e}")
                return False
    
    def unregister_overlay(self, overlay_id: str) -> bool:
        """Unregister an overlay from z-order management"""
        with self._lock:
            if overlay_id in self._overlays:
                del self._overlays[overlay_id]
                self._pending_enforcements.discard(overlay_id)
                
                # Clear context menu state if this overlay was active
                if self._context_menu_overlay_id == overlay_id:
                    self._context_menu_overlay_id = None
                
                self._tdebug_unreg(f"Unregistered overlay {overlay_id}")
                return True
            return False
    
    def begin_context_menu(self, overlay_id: str, menu: QMenu) -> bool:
        """
        Begin context menu display with immediate z-order priority.
        
        This gives the context menu immediate priority over normal z-order enforcement.
        """
        with self._lock:
            if overlay_id not in self._overlays:
                self._logger.error(f"Cannot begin context menu for unregistered overlay {overlay_id}")
                return False
            
            self._context_menu_overlay_id = overlay_id
            self._active_context_menus.add(menu)
            
            # Immediate enforcement with context menu priority
            success = self._enforce_z_order_immediate(overlay_id, ZOrderPriority.CONTEXT_MENU)
            
            self._tdebug_ctx(f"Context menu begun for overlay {overlay_id}, enforcement={success}")
            return success
    
    def end_context_menu(self, overlay_id: str, menu: QMenu) -> bool:
        """End context menu display and restore normal z-order"""
        with self._lock:
            self._active_context_menus.discard(menu)
            
            # Only clear context menu state if this was the active overlay
            if self._context_menu_overlay_id == overlay_id:
                self._context_menu_overlay_id = None
            
            # Restore normal z-order
            success = self._enforce_z_order_immediate(overlay_id, ZOrderPriority.NORMAL)
            
            self._tdebug_ctx(f"Context menu ended for overlay {overlay_id}, enforcement={success}")
            return success
    
    def enforce_z_order(self, overlay_id: str, priority: ZOrderPriority = ZOrderPriority.NORMAL) -> bool:
        """
        Request z-order enforcement with optional debouncing.
        
        Context menu priority bypasses debouncing for immediate enforcement.
        """
        used_coalescer = False
        with self._lock:
            if overlay_id not in self._overlays:
                self._logger.warning(f"Cannot enforce z-order for unregistered overlay {overlay_id}")
                return False
            
            # Context menu priority gets immediate enforcement
            if priority == ZOrderPriority.CONTEXT_MENU:
                return self._enforce_z_order_immediate(overlay_id, priority)
            
            # Normal priority: prefer coalesced enforcement via UI coalescer
            self._pending_enforcements.add(overlay_id)

            # Try to lazily initialize a coalescer when Qt is ready
            if self._coalescer is None:
                try:
                    if QCoreApplication.instance() is not None:
                        from core.threading import get_thread_manager
                        tm = get_thread_manager()
                        # 7ms window per project default for responsiveness
                        self._coalescer = tm.create_ui_coalescer(
                            name="z_order_enforcement",
                            capacity=128,
                            window_ms=7,
                        )
                        self._logger.info("ZOrderManager UI Coalescer initialized (window=7ms, cap=128)")
                except Exception as e:
                    # Log and continue with fallback debounced path
                    self._tdebug_enf(f"UI Coalescer initialization failed: {e}")

            if self._coalescer is not None:
                try:
                    # Submit a single drain-triggering task; actual work pulls from _pending_enforcements
                    self._coalescer.submit(self._execute_enforcement)
                    used_coalescer = True
                except Exception as e:
                    self._tdebug_enf(f"UI Coalescer submit failed, falling back to debounce: {e}")

            # Fallback to debounced enforcement when coalescer is unavailable
            if not used_coalescer:
                if not self._debounce_pending:
                    self._debounce_pending = True
                    try:
                        from core.threading import ThreadManager
                        ThreadManager.single_shot(self._debounce_delay_ms, self._execute_enforcement_wrapper)
                    except Exception as e:
                        # If centralized scheduling is unavailable, execute immediately as a fallback
                        self._tdebug_enf(f"Debounce scheduling failed, executing immediately: {e}")
                        # Release the lock before executing to avoid deadlocks
                        pass
        # Execute outside the lock if scheduling failed and we didn't use the coalescer
        if not used_coalescer and self._debounce_pending is False:
            self._execute_enforcement_wrapper()
        return True

    def _execute_enforcement_wrapper(self):
        """Wrapper to reset debounce flag and execute pending enforcements."""
        try:
            with self._lock:
                self._debounce_pending = False
        except Exception:
            # Best effort reset
            self._debounce_pending = False
        self._execute_enforcement()

    def _execute_enforcement(self):
        """Execute pending z-order enforcements"""
        with self._lock:
            pending = self._pending_enforcements.copy()
            self._pending_enforcements.clear()
            
            for overlay_id in pending:
                self._enforce_z_order_immediate(overlay_id, ZOrderPriority.NORMAL)
    
    def bring_child_to_front(self, widget: QWidget) -> bool:
        """Bring a widget's native window to the front of its parent's z-order.
        
        This is intended for small child widgets (e.g., focus indicators) that must
        render above DWM thumbnails or other composited content within the same host
        window. It does not change global topmost state.
        
        Returns True on success or when running on non-Windows platforms (no-op).
        """
        if widget is None:
            return False
        if not IS_WINDOWS:
            return True
        try:
            # Ensure the widget has a native handle
            hwnd = int(widget.winId())
            if not hwnd:
                self._logger.debug("bring_child_to_front: widget has no HWND")
                return False
            # Validate window and raise to top of parent z-order (not topmost)
            if IsWindow(hwnd):
                ok = SetWindowPos(
                    hwnd, HWND_TOP, 0, 0, 0, 0,
                    SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE
                )
                if not ok:
                    err = GetLastError()
                    self._logger.debug(f"bring_child_to_front: SetWindowPos failed (hwnd={hwnd}) error={err}")
                    return False
                self._logger.debug(f"bring_child_to_front: success for hwnd={hwnd}")
                return True
            else:
                self._logger.debug(f"bring_child_to_front: invalid HWND {hwnd}")
                return False
        except Exception as e:
            self._logger.debug(f"bring_child_to_front: exception {e}")
            return False

    def place_window_above(self, widget: QWidget, reference: QWidget) -> bool:
        """Place a top-level widget directly above a reference window within the same z-band.

        This uses SetWindowPos with the reference window as the insert-after target. It does not
        make the widget globally topmost and avoids altering the topmost band. Intended for keeping
        small tool windows (e.g., a focus indicator) stacked above their owner without affecting
        other applications.

        Returns True on success (or no-op on non-Windows).
        """
        if widget is None or reference is None:
            return False
        if not IS_WINDOWS:
            return True
        try:
            widget_hwnd = int(widget.winId())
            ref_hwnd = int(reference.winId())
            if not widget_hwnd or not ref_hwnd:
                self._logger.debug("place_window_above: missing HWND(s)")
                return False
            if not IsWindow(widget_hwnd) or not IsWindow(ref_hwnd):
                self._logger.debug(f"place_window_above: invalid HWNDs widget={widget_hwnd} ref={ref_hwnd}")
                return False
            ok = SetWindowPos(
                widget_hwnd, ref_hwnd, 0, 0, 0, 0,
                SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE
            )
            if not ok:
                err = GetLastError()
                self._logger.debug(f"place_window_above: SetWindowPos failed (widget={widget_hwnd} ref={ref_hwnd}) error={err}")
                return False
            self._logger.debug(f"place_window_above: success widget={widget_hwnd} above ref={ref_hwnd}")
            return True
        except Exception as e:
            self._logger.debug(f"place_window_above: exception {e}")
            return False
    
    def _enforce_z_order_immediate(self, overlay_id: str, priority: ZOrderPriority) -> bool:
        """Execute immediate z-order enforcement"""
        if not IS_WINDOWS:
            return True  # No-op on non-Windows platforms
        
        overlay_info = self._overlays.get(overlay_id)
        if not overlay_info:
            return False
        
        main_widget = overlay_info.main_widget()
        if not main_widget:
            self._logger.debug(f"Main widget for overlay {overlay_id} was destroyed")
            self.unregister_overlay(overlay_id)
            return False
        
        success = True
        
        # Determine z-order position based on priority and context menu state
        if priority == ZOrderPriority.CONTEXT_MENU or self._has_active_context_menu():
            hwnd_pos = HWND_TOP  # Below topmost but above normal windows
        else:
            hwnd_pos = HWND_TOPMOST  # Full topmost
        
        # Enforce main widget z-order
        main_hwnd = int(main_widget.winId())
        if IsWindow(main_hwnd):
            ok_main = SetWindowPos(
                main_hwnd, hwnd_pos, 0, 0, 0, 0,
                SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE
            )
            if not ok_main:
                error_code = GetLastError()
                if error_code == 1400:  # ERROR_INVALID_WINDOW_HANDLE
                    self._logger.debug(f"Main overlay window {main_hwnd} was destroyed")
                    self.unregister_overlay(overlay_id)
                    return False
                else:
                    self._logger.warning(f"SetWindowPos failed for main overlay {main_hwnd}: error {error_code}")
                    success = False
        else:
            self._logger.debug(f"Main overlay window {main_hwnd} is not valid")
            success = False
        
        # With integrated borders, the main widget already handles borders
        # This legacy code is maintained only for backward compatibility
        # with pre-integrated border overlays
        if overlay_info.border_widget:
            border_widget = overlay_info.border_widget()
            if border_widget:
                try:
                    border_hwnd = int(border_widget.winId())
                    if IsWindow(border_hwnd):
                        ok_border = SetWindowPos(
                            border_hwnd, hwnd_pos, 0, 0, 0, 0,
                            SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE
                        )
                        if not ok_border:
                            error_code = GetLastError()
                            if error_code == 1400:
                                self._logger.debug(f"Legacy border window {border_hwnd} was destroyed")
                                # Clear the reference since it's invalid
                                overlay_info.border_widget = None
                            else:
                                self._logger.debug(f"SetWindowPos failed for legacy border: error {error_code}")
                    else:
                        self._logger.debug(f"Legacy border window {border_hwnd} is not valid")
                        overlay_info.border_widget = None
                except Exception as e:
                    self._logger.debug(f"Error handling legacy border for {overlay_id}: {e}")
                    overlay_info.border_widget = None
            else:
                # Border widget was destroyed, clear the reference
                overlay_info.border_widget = None
        
        priority_str = priority.name.lower()
        self._logger.debug(f"Z-order enforcement for overlay {overlay_id} (priority={priority_str}): {success}")
        
        return success
    
    def _has_active_context_menu(self) -> bool:
        """Check if any context menus are currently active"""
        # Clean up dead references
        active_menus = [menu for menu in self._active_context_menus if menu is not None]
        self._active_context_menus.clear()
        self._active_context_menus.update(active_menus)
        
        return len(active_menus) > 0
    
    def get_overlay_count(self) -> int:
        """Get the number of registered overlays"""
        with self._lock:
            return len(self._overlays)
    
    def cleanup(self):
        """Clean up resources"""
        with self._lock:
            # Reset debounce state; timers scheduled via ThreadManager cannot be cancelled, but
            # clearing pending requests prevents further actions.
            self._debounce_pending = False
            self._overlays.clear()
            self._active_context_menus.clear()
            self._pending_enforcements.clear()
            self._context_menu_overlay_id = None
            self._coalescer = None
            
            self._logger.info("ZOrderManager cleaned up")


# Singleton instance
_z_order_manager: Optional[ZOrderManager] = None


def get_z_order_manager() -> ZOrderManager:
    """Get the singleton z-order manager instance"""
    global _z_order_manager
    if _z_order_manager is None:
        _z_order_manager = ZOrderManager()
    return _z_order_manager


def cleanup_z_order_manager():
    """Clean up the singleton z-order manager"""
    global _z_order_manager
    if _z_order_manager is not None:
        _z_order_manager.cleanup()
        _z_order_manager = None
