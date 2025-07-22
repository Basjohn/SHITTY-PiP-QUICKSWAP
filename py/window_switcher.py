"""
Window Switcher Module

This module provides functionality for switching between windows with different strategies.
Handles both normal window switching and auto-switching with proper window filtering.
"""
import os
import time
from typing import Optional, Tuple, Any, TYPE_CHECKING

import win32gui
import win32con
import win32process
import win32api
from debug_utils import get_logger

if TYPE_CHECKING:
    pass  # For type hints only

logger = get_logger(__name__)

# Import window filter after logger is set up
try:
    from window_filter import WindowFilter
except ImportError as e:
    logger.error(f"Failed to import WindowFilter: {e}")
    WindowFilter = None


class WindowSwitcher:
    """Manages window switching operations with support for normal and auto-switching modes.
    
    This class centralizes all window switching logic and provides validation
    through the WindowFilter class to ensure only valid windows are targeted.
    """
    
    def __init__(self, app_instance=None):
        """Initialize the WindowSwitcher with an optional reference to the main application.
        
        Args:
            app_instance: Reference to the main application instance (optional)
        """
        self.app_instance = app_instance
        self._last_switched_hwnd = None
        self._last_switch_time = 0
        self._switch_cooldown = 0.5  # 500ms cooldown between switches
        self._our_pid = os.getpid()
        self._our_windows = set()  # Cache of our application's windows
        
        # Window list caching
        self._cached_window_list = []
        self._window_list_cache_time = 0
        self._window_list_cache_ttl = 0.5  # Cache TTL in seconds
        
        # Initialize window filter
        self.window_filter = WindowFilter() if WindowFilter else None
        logger.debug("WindowSwitcher initialized")
    
    def _update_our_windows_cache(self):
        """Update the cache of windows belonging to our application."""
        self._our_windows.clear()
        
        if self.app_instance and hasattr(self.app_instance, 'overlays'):
            for overlay in self.app_instance.overlays:
                if hasattr(overlay, 'winId'):
                    try:
                        # Get the native window handle
                        hwnd = int(overlay.winId())
                        self._our_windows.add(hwnd)
                        logger.debug(f"Added overlay window to our_windows cache: {hwnd}")
                    except Exception as e:
                        logger.debug(f"Error getting winId from overlay: {e}")
        
        # Also add any windows from our process
        def enum_windows_proc(hwnd, lparam):
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == self._our_pid:
                    self._our_windows.add(hwnd)
                    logger.debug(f"Added process window to our_windows cache: {hwnd}")
            except Exception:
                pass  # Ignore errors for individual windows
            return True
        
        try:
            win32gui.EnumWindows(enum_windows_proc, 0)
        except Exception as e:
            logger.debug(f"Error enumerating windows for our process: {e}")
    
    def _is_our_window(self, hwnd: int) -> bool:
        """Check if a window belongs to our application.
        
        Args:
            hwnd: The window handle to check
            
        Returns:
            bool: True if the window belongs to our application
        """
        # Check cached windows first
        if hwnd in self._our_windows:
            return True
            
        # Check if it's a window from our process
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == self._our_pid:
                self._our_windows.add(hwnd)  # Add to cache
                return True
        except Exception:
            pass
            
        return False
    
    def validate_window(self, hwnd: int, for_auto_switch: bool = False) -> Tuple[bool, str]:
        """Validate if a window is suitable for switching.
        
        Args:
            hwnd: The window handle to validate
            for_auto_switch: Whether this is for auto-switch (stricter validation)
            
        Returns:
            Tuple[bool, str]: (is_valid, reason) where reason explains why if not valid
        """
        if not hwnd or hwnd == 0:
            return False, "Invalid window handle"
        
        # CRITICAL: Never allow switching to our own application windows
        if self._is_our_window(hwnd):
            return False, "Window belongs to our application"
            
        # Skip if we just switched to this window
        if hwnd == self._last_switched_hwnd and (time.time() - self._last_switch_time) < self._switch_cooldown:
            return False, "Skipping recently switched window"
            
        # Use WindowFilter if available, otherwise basic validation
        if self.window_filter:
            if not self.window_filter.is_valid_window(hwnd, self._our_pid):
                return False, "Window filtered out by WindowFilter"
        else:
            # Basic validation if WindowFilter is not available
            if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
                return False, "Window is not visible or invalid"
        
        # Additional validation for auto-switch
        if for_auto_switch:
            # Skip minimized windows for auto-switch
            if win32gui.IsIconic(hwnd):
                return False, "Skipping minimized window in auto-switch mode"
                
            # Skip windows that don't have a title for auto-switch
            if not win32gui.GetWindowText(hwnd).strip():
                return False, "Skipping window without title in auto-switch mode"
        
        return True, ""
    
    def switch_to_window(self, target_hwnd: int, current_overlay: Any = None) -> bool:
        """Switch to the specified window using the current switching strategy.
        
        Args:
            target_hwnd: The window handle to switch to
            current_overlay: The current overlay widget (optional)
            
        Returns:
            bool: True if the switch was successful, False otherwise
        """
        logger.debug(f"Switching to window {target_hwnd}")
        
        # Validate the window for normal switching
        is_valid, reason = self.validate_window(target_hwnd, for_auto_switch=False)
        if not is_valid:
            logger.warning(f"Cannot switch to window {target_hwnd}: {reason}")
            return False
            
        # Update last switched window and time
        self._last_switched_hwnd = target_hwnd
        self._last_switch_time = time.time()
        
        # Delegate to the standard switch implementation
        return self.standard_switch(target_hwnd, current_overlay)
        
    def auto_switch_to_window(self, target_hwnd: int, current_overlay: Any = None) -> bool:
        """Handle auto-switching to a window with additional validation.
        
        This method enforces stricter validation rules than switch_to_window(),
        such as skipping minimized windows and windows without titles.
        
        Args:
            target_hwnd: The window handle to switch to
            current_overlay: The current overlay widget (optional)
            
        Returns:
            bool: True if the auto-switch was successful, False otherwise
        """
        logger.debug(f"Attempting auto-switch to window {target_hwnd}")
        
        # Validate the window with stricter rules for auto-switch
        is_valid, reason = self.validate_window(target_hwnd, for_auto_switch=True)
        if not is_valid:
            logger.debug(f"Skipping auto-switch to window {target_hwnd}: {reason}")
            return False
            
        # Use the standard switch if validation passes
        return self.switch_to_window(target_hwnd, current_overlay)
    
    def standard_switch(self, target_hwnd: int, current_overlay: Any = None) -> bool:
        """Standard window switching implementation.
        
        This method updates the overlay to show the target window instead of
        changing the foreground window.
        
        Args:
            target_hwnd: The window handle to switch to
            current_overlay: The current overlay widget (required)
            
        Returns:
            bool: True if the switch was successful, False otherwise
        """
        logger.debug("Initiating standard window switch operation")
        
        # Handle desktop overlay case first
        if hasattr(current_overlay, 'is_desktop_overlay') and current_overlay.is_desktop_overlay:
            logger.info("Desktop overlay detected - triggering desktop double-click action")
            current_overlay._handle_desktop_double_click()
            return True
            
        # Validate the current overlay state
        if not target_hwnd or not current_overlay:
            logger.warning("Switch aborted: Missing target HWND or overlay reference")
            return False
            
        try:
            # Update the overlay to show the new window
            logger.debug(f"Updating overlay to show window {target_hwnd}")
            
            # Store the current state
            original_hwnd = current_overlay.hwnd
            
            # Update the overlay's target window
            current_overlay.hwnd = target_hwnd
            
            # If the window is minimized, restore it (but don't focus it)
            if win32gui.IsIconic(target_hwnd):
                win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                
            # Update the thumbnail
            if hasattr(current_overlay, 'register_thumbnail'):
                current_overlay.register_thumbnail()
                
            logger.info(f"Updated overlay to show window {target_hwnd}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating overlay for window {target_hwnd}: {e}", exc_info=True)
            # Try to restore the original window if possible
            if 'original_hwnd' in locals() and hasattr(current_overlay, 'hwnd'):
                try:
                    current_overlay.hwnd = original_hwnd
                    if hasattr(current_overlay, 'register_thumbnail'):
                        current_overlay.register_thumbnail()
                except Exception as restore_error:
                    logger.error(f"Failed to restore original window: {restore_error}")
            return False
        
    def auto_switch_to_window(self, target_hwnd: int, current_overlay: Any = None) -> bool:
        """Handle auto-switching to a window with additional validation.
        
        This method enforces stricter validation rules than switch_to_window(),
        such as skipping minimized windows and windows without titles.
        
        Args:
            target_hwnd: The window handle to switch to
            current_overlay: The current overlay widget (optional)
            
        Returns:
            bool: True if the auto-switch was successful, False otherwise
        """
        logger.debug(f"Attempting auto-switch to window {target_hwnd}")
        
        # Validate the window with stricter rules for auto-switch
        is_valid, reason = self.validate_window(target_hwnd, for_auto_switch=True)
        if not is_valid:
            logger.debug(f"Skipping auto-switch to window {target_hwnd}: {reason}")
            return False
            
        # Use the standard switch if validation passes
        return self.switch_to_window(target_hwnd, current_overlay)
    
    def standard_switch(self, target_hwnd: int, current_overlay: Any = None) -> bool:
        """Standard window switching implementation.
        
        This method updates the overlay to show the target window instead of
        changing the foreground window.
        
        Args:
            target_hwnd: The window handle to switch to
            current_overlay: The current overlay widget (required)
            
        Returns:
            bool: True if the switch was successful, False otherwise
        """
        logger.debug("Initiating standard window switch operation")
        
        # Handle desktop overlay case first
        if hasattr(current_overlay, 'is_desktop_overlay') and current_overlay.is_desktop_overlay:
            logger.info("Desktop overlay detected - triggering desktop double-click action")
            current_overlay._handle_desktop_double_click()
            return True
            
        # Validate the current overlay state
        if not target_hwnd or not current_overlay:
            logger.warning("Switch aborted: Missing target HWND or overlay reference")
            return False
            
        try:
            # Update the overlay to show the new window
            logger.debug(f"Updating overlay to show window {target_hwnd}")
            
            # Store the current state
            original_hwnd = current_overlay.hwnd
            
            # Update the overlay's target window
            current_overlay.hwnd = target_hwnd
            
            # If the window is minimized, restore it (but don't focus it)
            if win32gui.IsIconic(target_hwnd):
                win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                
            # Update the thumbnail
            if hasattr(current_overlay, 'register_thumbnail'):
                current_overlay.register_thumbnail()
                
            logger.info(f"Updated overlay to show window {target_hwnd}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating overlay for window {target_hwnd}: {e}", exc_info=True)
            # Try to restore the original window if possible
            if 'original_hwnd' in locals() and hasattr(current_overlay, 'hwnd'):
                try:
                    current_overlay.hwnd = original_hwnd
                    if hasattr(current_overlay, 'register_thumbnail'):
                        current_overlay.register_thumbnail()
                except Exception as restore_error:
                    logger.error(f"Failed to restore original window: {restore_error}")
            return False
            
    def quick_switch_windows(self, overlay_hwnd: int, focused_hwnd: int, overlay_widget: Any) -> bool:
        """Handle quick switching between the overlay window and the currently focused window.
        
        This is the core implementation of the quick-switch functionality where:
        1. The window in the overlay is brought to the foreground
        2. The previously focused window is shown in the overlay
        
        Args:
            overlay_hwnd: The window handle currently in the overlay
            focused_hwnd: The window handle that currently has focus
            overlay_widget: The overlay widget instance
            
        Returns:
            bool: True if the switch was successful, False otherwise
        """
        try:
            # Update our windows cache to ensure we have the latest state
            self._update_our_windows_cache()
            
            logger.info(f"Initiating quick switch - Overlay: {overlay_hwnd}, Focused: {focused_hwnd}")
            
            # CRITICAL: Ensure we're not working with our own windows
            if self._is_our_window(overlay_hwnd):
                logger.warning(f"Overlay window {overlay_hwnd} belongs to our application - aborting quick switch")
                return False
                
            if focused_hwnd and self._is_our_window(focused_hwnd):
                logger.debug(f"Focused window {focused_hwnd} belongs to our application - will find alternative")
                focused_hwnd = None  # Clear it so we'll find an alternative
            
            # First try to use the focused window if it's valid
            if (focused_hwnd and 
                focused_hwnd != overlay_hwnd and 
                win32gui.IsWindow(focused_hwnd) and
                win32gui.IsWindowVisible(focused_hwnd) and
                not self._is_our_window(focused_hwnd)):
                
                swap_target_hwnd = focused_hwnd
                logger.debug(f"Using focused window as swap target: {focused_hwnd}")
                
            # If focused window isn't valid, try to find a target window
            else:
                logger.debug("No valid focused window, searching for swap target via MRU")
                swap_target_hwnd = self._find_swap_target_window(overlay_hwnd, overlay_hwnd, overlay_widget)
                if swap_target_hwnd:
                    logger.debug(f"Found MRU swap target: {swap_target_hwnd}")
                else:
                    # If no valid MRU found, try to use the current foreground window as fallback
                    try:
                        fg_hwnd = win32gui.GetForegroundWindow()
                        # Basic validation of foreground window
                        if (fg_hwnd and fg_hwnd != overlay_hwnd and 
                            win32gui.IsWindow(fg_hwnd) and 
                            win32gui.IsWindowVisible(fg_hwnd) and 
                            not win32gui.IsIconic(fg_hwnd) and
                            not self._is_our_window(fg_hwnd)):
                            logger.debug(f"Using current foreground window as fallback swap target: {fg_hwnd}")
                            swap_target_hwnd = fg_hwnd
                        else:
                            logger.warning("No valid swap target found, including fallback")
                            return False
                    except Exception as e:
                        logger.error(f"Error getting fallback foreground window: {e}")
                        return False
            
            # Final check: ensure swap target is not our window
            if self._is_our_window(swap_target_hwnd):
                logger.warning(f"Swap target {swap_target_hwnd} belongs to our application - aborting")
                return False
            
            # Log the planned swap operation
            try:
                overlay_title = win32gui.GetWindowText(overlay_hwnd) if win32gui.IsWindow(overlay_hwnd) else "[Invalid Window]"
                swap_title = win32gui.GetWindowText(swap_target_hwnd) if win32gui.IsWindow(swap_target_hwnd) else "[Invalid Window]"
                logger.info(f"SWAP PLAN - Activate: {overlay_title} (HWND: {overlay_hwnd})")
                logger.info(f"SWAP PLAN - Put in overlay: {swap_title} (HWND: {swap_target_hwnd})")
            except Exception as e:
                logger.debug(f"Could not get window titles for logging: {e}")
            
            # Update MRU list for external windows only
            if (hasattr(overlay_widget, 'app_instance') and 
                hasattr(overlay_widget.app_instance, 'update_mru_list')):
                try:
                    # Update MRU for the overlay window if it's external
                    if not self._is_our_window(overlay_hwnd):
                        logger.debug("Updating MRU list for overlay window")
                        overlay_widget.app_instance.update_mru_list(overlay_hwnd)
                    
                    # Update MRU for the swap target if it's external
                    if not self._is_our_window(swap_target_hwnd):
                        logger.debug("Updating MRU list for swap target")
                        overlay_widget.app_instance.update_mru_list(swap_target_hwnd)
                    
                    # Ensure our windows are not in the MRU list
                    overlay_widget.app_instance.clean_mru_list()
                        
                except Exception as e:
                    logger.error(f"Error updating MRU list: {e}")
            
            # NOW perform the swap:
            # Step 1: Update the overlay to show the swap target FIRST
            swap_method = None
            
            # Enhanced diagnostic logging
            # Only log widget type at debug level
            logger.debug(f"Overlay widget type: {type(overlay_widget).__name__}")
            
            # Check for both method names (with and without underscore prefix)
            if hasattr(overlay_widget, 'handle_swap_window'):
                swap_method = overlay_widget.handle_swap_window
                logger.info("Found handle_swap_window method")
            elif hasattr(overlay_widget, '_handle_swap_window'):
                swap_method = overlay_widget._handle_swap_window
                logger.info("Found _handle_swap_window method")
                
            if swap_method:
                try:
                    logger.debug(f"Step 1: Updating overlay to show swap target: {swap_target_hwnd}")
                    result = swap_method(swap_target_hwnd)
                    if not result:
                        logger.error("Failed to update overlay with swap target")
                        return False
                    logger.debug("Overlay updated successfully")
                except Exception as e:
                    logger.error(f"Error updating overlay: {e}", exc_info=True)
                    return False
            else:
                logger.error("No handle_swap_window or _handle_swap_window method available on overlay_widget")
                return False
            
            # Step 2: Activate the original overlay window
            try:
                logger.debug(f"Step 2: Activating original overlay window: {overlay_hwnd}")
                self._activate_window(overlay_hwnd)
                logger.debug("Window activation completed")
            except Exception as e:
                # Handle activation errors more gracefully
                logger.debug(f"Window activation error: {e}")
                # This is often not a critical failure, especially with locked or background windows
                logger.warning("Window activation skipped, but overlay has been updated")
            
            logger.info(f"Quick switch completed - activated {overlay_hwnd}, overlay now shows {swap_target_hwnd}")
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in quick_switch_windows: {e}", exc_info=True)
            return False
    
    def _get_cached_window_list(self) -> list:
        """Get a cached list of valid windows, refreshing the cache if needed.
        
        Returns:
            list: List of window handles that are valid for switching
        """
        current_time = time.time()
        
        # Return cached list if it's still valid
        if (current_time - self._window_list_cache_time) < self._window_list_cache_ttl:
            return self._cached_window_list
            
        # Refresh the cache
        logger.debug("Refreshing window list cache")
        windows = []
        
        def enum_windows_callback(hwnd, _):
            try:
                # Skip our own windows
                if self._is_our_window(hwnd):
                    return True
                    
                # Basic window validation
                if (win32gui.IsWindow(hwnd) and 
                    win32gui.IsWindowVisible(hwnd) and 
                    not win32gui.IsIconic(hwnd)):
                    
                    # Additional filtering if available
                    if self.window_filter and not self.window_filter.is_valid_window(hwnd, self._our_pid):
                        return True
                        
                    windows.append(hwnd)
            except Exception as e:
                logger.debug(f"Error processing window {hwnd}: {e}")
            return True
            
        try:
            win32gui.EnumWindows(enum_windows_callback, 0)
            self._cached_window_list = windows
            self._window_list_cache_time = current_time
            logger.debug(f"Cached {len(windows)} valid windows")
            return windows
        except Exception as e:
            logger.error(f"Error enumerating windows: {e}")
            return []
            
    def _find_swap_target_window(self, overlay_hwnd: int, current_hwnd: int, overlay: Any) -> Optional[int]:
        """Find a suitable window to swap with the overlay using MRU (Most Recently Used) order.
        
        This method attempts to find the best window to swap with by checking in this order:
        1. The last external focused window (if different from current and overlay)
        2. The current foreground window (if different from current and overlay)
        3. Windows from the cached window list (skipping current and overlay)
        
        Args:
            overlay_hwnd: The window handle of the overlay
            current_hwnd: The current target window handle
            overlay: The overlay widget instance
            
        Returns:
            int: The window handle of the target window to swap with, or None if not found
        """
        # Get the app instance if available
        app = getattr(overlay, 'app_instance', None)
        our_pid = os.getpid()
        
        # Get all windows belonging to our process to exclude them from swap targets
        our_windows = []
        if app and hasattr(app, 'overlays'):
            our_windows = [overlay.winId() for overlay in app.overlays if hasattr(overlay, 'winId')]
            
        # Get cached list of valid windows
        valid_windows = self._get_cached_window_list()
        
        def is_valid_swap_target(hwnd):
            """Helper to check if a window is a valid swap target."""
            # Basic window validation
            if not hwnd or hwnd == 0 or not win32gui.IsWindow(hwnd):
                logger.debug(f"Window {hwnd} excluded: Invalid window handle")
                return False
                
            # Skip if this is our overlay window or current window
            if hwnd in (overlay_hwnd, current_hwnd):
                logger.debug(f"Window {hwnd} excluded: Overlay or current window")
                return False
                
            # Skip if this is one of our application's windows
            if hwnd in our_windows:
                logger.debug(f"Window {hwnd} excluded: Belongs to our application")
                return False
                
            # Use WindowFilter for comprehensive validation if available
            if WindowFilter and not WindowFilter.is_valid_window(hwnd, our_pid):
                logger.debug(f"Window {hwnd} excluded: Failed WindowFilter validation")
                return False
                
            # Additional validation specific to swapping
            try:
                # Get window details for logging
                title = win32gui.GetWindowText(hwnd)
                class_name = win32gui.GetClassName(hwnd)
                logger.debug(f"Evaluating window - HWND: {hwnd}, Title: {title}, Class: {class_name}")
                
                # Skip if window is minimized
                if win32gui.IsIconic(hwnd):
                    logger.debug(f"Window {hwnd} excluded: Minimized")
                    return False
                    
                # Skip if window is not visible
                if not win32gui.IsWindowVisible(hwnd):
                    logger.debug(f"Window {hwnd} excluded: Not visible")
                    return False
                    
                # Check for window styles that indicate system/utility windows
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                if style & win32con.WS_DISABLED or not (style & win32con.WS_VISIBLE):
                    logger.debug(f"Window {hwnd} excluded: Disabled or not visible (style: {style:#x})")
                    return False
                    
                # Check for window extended styles that indicate tool windows or other special types
                ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                if ex_style & win32con.WS_EX_TOOLWINDOW or ex_style & win32con.WS_EX_NOACTIVATE:
                    logger.debug(f"Window {hwnd} excluded: Tool window or no-activate window (ex_style: {ex_style:#x})")
                    return False
                    
                # Additional check for window size
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    
                    # Skip windows that are too small (but allow for edge cases like notification windows)
                    min_width, min_height = 80, 40  # Very minimal size requirements
                    if width < min_width or height < min_height:
                        logger.debug(f"Window {hwnd} excluded due to small size: {width}x{height}")
                        return False
                        
                    logger.debug(f"Window {hwnd} is a valid swap target")
                    return True
                    
                except Exception as e:
                    logger.debug(f"Error checking window {hwnd} size: {e}")
                    return False
                    
            except Exception as e:
                logger.debug(f"Error validating window {hwnd}: {e}")
                return False
        
        # 1. Try the last external focused window first
        if app and hasattr(app, 'last_external_focused_hwnd'):
            last_focused = app.last_external_focused_hwnd
            if last_focused and last_focused not in our_windows and last_focused in valid_windows:
                logger.debug("Using last external focused window for swap")
                return last_focused
        
        # 2. Try the current foreground window (this is our fallback if no MRU is available)
        fg_hwnd = None
        try:
            fg_hwnd = win32gui.GetForegroundWindow()
            if fg_hwnd and fg_hwnd not in our_windows and fg_hwnd in valid_windows:
                logger.debug("Using current foreground window for swap")
                return fg_hwnd
        except Exception as e:
            logger.warning("Failed to get foreground window: %s", e)
        
        # 3. Try windows from the MRU list (if available)
        if app and hasattr(app, 'mru_hwnds') and app.mru_hwnds:
            logger.debug("Checking MRU list for swap candidates")
            # Try up to 5 most recent windows from MRU list
            for hwnd in app.mru_hwnds[:5]:
                if hwnd and hwnd not in our_windows and hwnd in valid_windows:
                    logger.debug(f"Using window from MRU list for swap: {hwnd}")
                    return hwnd
        
        # 4. Try any window from our cached list that isn't the overlay or current window
        for hwnd in valid_windows:
            if (hwnd and 
                hwnd != overlay_hwnd and 
                hwnd != current_hwnd and 
                hwnd not in our_windows):
                logger.debug(f"Using window from cache for swap: {hwnd}")
                return hwnd
        
        # 5. If we get here, no valid window was found in cache, try a direct approach
        if fg_hwnd and fg_hwnd not in (overlay_hwnd, current_hwnd) and fg_hwnd not in our_windows:
            try:
                # Do basic validation without WindowFilter for the fallback case
                if (win32gui.IsWindow(fg_hwnd) and 
                    win32gui.IsWindowVisible(fg_hwnd) and 
                    not win32gui.IsIconic(fg_hwnd)):
                    logger.debug("Falling back to current foreground window")
                    return fg_hwnd
            except Exception as e:
                logger.warning(f"Error in fallback to foreground window: {e}")
        
        logger.debug("No suitable swap target found")
        return None
    
    def _perform_window_swap(self, hwnd_to_activate: int, swap_target_hwnd: int, overlay: Any) -> bool:
        """Perform the actual window swap operation.
        
        Args:
            hwnd_to_activate: The window handle to activate
            swap_target_hwnd: The window handle to swap with
            overlay: The overlay widget instance
            
        Returns:
            bool: True if the swap was successful, False otherwise
        """
        try:
            logger.debug(f"Performing window swap - Activating: {hwnd_to_activate}, Overlay target: {swap_target_hwnd}")
            
            # Activate the target window
            self._activate_window(hwnd_to_activate)
            
            # Update the overlay to target the new window
            if hasattr(overlay, '_handle_swap_window'):
                result = overlay._handle_swap_window(swap_target_hwnd)
                if not result:
                    logger.error("_handle_swap_window failed during fallback swap")
                    return False
            elif hasattr(overlay, 'hwnd'):
                # Direct fallback if _handle_swap_window doesn't exist
                overlay.hwnd = swap_target_hwnd
                if hasattr(overlay, 'register_thumbnail'):
                    overlay.register_thumbnail()
            else:
                logger.error("Overlay widget doesn't support window swapping")
                return False
            
            # Update the last used window in settings if available
            try:
                if hasattr(overlay, 'app_instance') and hasattr(overlay.app_instance, 'settings'):
                    overlay.app_instance.settings.setValue("LastUsedWindowHwnd", swap_target_hwnd)
            except Exception as e:
                logger.warning("Failed to update last used window in settings: %s", e)
            
            logger.info("Window swap completed successfully")
            return True
            
        except Exception as e:
            logger.error("Error during window swap: %s", e, exc_info=True)
            # Fallback activation attempt
            try:
                win32gui.ShowWindow(hwnd_to_activate, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd_to_activate)
                logger.debug("Fallback activation succeeded")
                return True
            except Exception as fallback_e:
                logger.error("Fallback switch also failed: %s", fallback_e)
                return False
                    
    def _activate_window(self, hwnd: int, max_attempts: int = 3) -> None:
        """Activate a window with special handling for different window types.
        
        Args:
            hwnd: The window handle to activate
            max_attempts: Maximum number of activation attempts
            
        Raises:
            win32gui.error: If there's an error activating the window after all attempts
        """
        if not hwnd or not win32gui.IsWindow(hwnd):
            logger.warning("Attempted to activate invalid window handle: %s", hwnd)
            return
            
        try:
            # Get window class and title for special handling
            window_class = win32gui.GetClassName(hwnd).lower()
            window_title = win32gui.GetWindowText(hwnd).lower()
            logger.debug("Activating window - Handle: %s, Class: %s, Title: %s", 
                        hwnd, window_class, window_title)
            
            # Special handling for video players and browsers
            is_video_player = any(name in window_class or name in window_title 
                                for name in ['mpc', 'mpv', 'vlc', 'potplayer', 'kodi', 'plex', 'jellyfin'])
            is_browser = any(name in window_class 
                            for name in ['chrome', 'firefox', 'msedge', 'iexplore', 'safari'])
            
            # Log window type detection
            if is_video_player:
                logger.debug("Detected video player window")
            elif is_browser:
                logger.debug("Detected browser window")
            
            # Restore window if minimized
            if win32gui.IsIconic(hwnd):
                logger.debug("Window is minimized, restoring...")
                # For video players, first restore, then activate
                if is_video_player:
                    logger.debug("Using video player specific restore method")
                    # First restore the window
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    # Small delay to let the window restore
                    time.sleep(0.1)
                    # Force a repaint
                    win32gui.RedrawWindow(hwnd, None, None, 
                                        win32con.RDW_INVALIDATE | 
                                        win32con.RDW_ERASE | 
                                        win32con.RDW_ALLCHILDREN)
                    logger.debug("Video player window restored and repainted")
                else:
                    # Standard restore for other windows
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    logger.debug("Window restored from minimized state")
            
            # Ensure window is visible
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            logger.debug("Window visibility ensured")
            
            # Get the current foreground window's thread ID
            foreground_hwnd = win32gui.GetForegroundWindow()
            foreground_thread_id = win32process.GetWindowThreadProcessId(foreground_hwnd)[0]
            current_thread_id = win32api.GetCurrentThreadId()
            
            # Try to attach our thread to the foreground thread's input queue
            attached = False
            if foreground_thread_id != current_thread_id:
                try:
                    attached = win32process.AttachThreadInput(current_thread_id, foreground_thread_id, True)
                    if attached:
                        logger.debug("Attached to foreground thread's input queue")
                except Exception as e:
                    logger.debug(f"Failed to attach to foreground thread: {e}")
            
            # Attempt activation with retries
            for attempt in range(1, max_attempts + 1):
                try:
                    # For browsers and video players, use a more aggressive approach
                    if is_browser or is_video_player:
                        logger.debug(f"Attempt {attempt}/{max_attempts} - Using aggressive activation")
                        # First, try to bring to front
                        win32gui.BringWindowToTop(hwnd)
                        # Then set as foreground
                        win32gui.SetForegroundWindow(hwnd)
                        # Small delay between attempts
                        time.sleep(0.1)
                        # Try again with SetForegroundWindow
                        win32gui.SetForegroundWindow(hwnd)
                    else:
                        logger.debug(f"Attempt {attempt}/{max_attempts} - Using standard activation")
                        win32gui.SetForegroundWindow(hwnd)
                    
                    # Verify if the window is now in the foreground
                    if win32gui.GetForegroundWindow() == hwnd:
                        logger.debug("Window successfully brought to foreground")
                        return
                        
                    # If we get here, the window wasn't activated - try a different approach
                    logger.debug(f"Window not activated, trying alternative method (attempt {attempt})")
                    
                    # Try using ShowWindow with SW_RESTORE first
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                    
                    # Try to force the window to the top
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 0, 0,
                                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | 
                                        win32con.SWP_SHOWWINDOW | win32con.SWP_ASYNCWINDOWPOS)
                    
                    # Small delay before next attempt
                    time.sleep(0.1)
                    
                except win32gui.error as e:
                    if attempt == max_attempts:
                        logger.error(f"Failed to activate window after {max_attempts} attempts: {e}")
                        raise
                    logger.debug(f"Activation attempt {attempt} failed: {e}")
                    time.sleep(0.1)  # Small delay before retry
                    
            # If we get here, all attempts failed
            logger.error(f"Failed to activate window {hwnd} after {max_attempts} attempts")
            
        except Exception as e:
            logger.error(f"Error in _activate_window: {e}", exc_info=True)
            raise
                
        except win32gui.error as e:
            logger.error("Failed to activate window %s: %s", hwnd, e, exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error while activating window %s: %s", hwnd, e, exc_info=True)
            raise
    
    def stealth_switch(self, target_hwnd: int, current_overlay: Any) -> bool:
        """Stealth window switching logic (stub).
        
        This is a placeholder for future implementation of stealthy window switching
        that won't disrupt the user's current window focus.
        
        Args:
            target_hwnd: The window handle to switch to
            current_overlay: The current overlay widget
            
        Returns:
            bool: True if the switch was successful, False otherwise
        """
        logger.debug("Stealth switch called (not implemented, falling back to standard switch)")
        # For now, fall back to standard switch
        return self.standard_switch(target_hwnd, current_overlay)


def set_window_switcher(app):
    """Helper function to create and return a WindowSwitcher instance.

    Args:
        app: Reference to the main application instance
        
    Returns:
        WindowSwitcher: A new instance of WindowSwitcher
    """
    return WindowSwitcher(app)
