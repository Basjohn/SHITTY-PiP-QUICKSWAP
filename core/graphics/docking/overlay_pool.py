"""
Overlay pooling system for performance optimization.

This module provides overlay pooling to reduce the overhead of frequent 
overlay creation and destruction in docking mode.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from core.graphics.types import OverlayConfig
from core.graphics.backend_manager import BackendManager, BackendType
from core.logging import get_logger
from utils.resource_manager import get_resource_manager


@dataclass
class PooledOverlay:
    """Represents a pooled overlay with metadata."""
    overlay: object
    created_at: float
    last_used: float
    in_use: bool = False
    overlay_id: Optional[str] = None


class DockingOverlayPool:
    """Pool manager for docking overlays to optimize performance.
    
    UI-thread confined: All pool operations must occur on the UI thread.
    No threading.Lock needed - pool state is UI-only like other Qt objects.
    """
    
    def __init__(self, max_pool_size: int = 6, max_idle_time: float = 300.0):
        """Initialize the overlay pool.
        
        Args:
            max_pool_size: Maximum number of overlays to keep in pool
            max_idle_time: Maximum time (seconds) to keep unused overlays
        """
        self._pool: List[PooledOverlay] = []
        self._active_overlays: Dict[str, PooledOverlay] = {}
        self._max_pool_size = max_pool_size
        self._max_idle_time = max_idle_time
        self._backend_manager = BackendManager()
        self._logger = get_logger(__name__)
        self._resource_manager = get_resource_manager()
        
        # Cleanup timer
        self._cleanup_timer = None
        self._timer_resource_id = None
        self._schedule_cleanup()
    
    def acquire_overlay(self, overlay_id: str, config: OverlayConfig) -> Optional[object]:
        """Acquire an overlay from the pool or create a new one.
        
        Must be called from UI thread.
        """
        try:
            # Check if overlay is already active
            if overlay_id in self._active_overlays:
                self._logger.debug(f"Overlay {overlay_id} already active")
                return self._active_overlays[overlay_id].overlay
            
            # Try to get from pool
            pooled_overlay = self._get_from_pool()
            
            if pooled_overlay:
                # Reuse existing overlay
                pooled_overlay.overlay_id = overlay_id
                pooled_overlay.in_use = True
                pooled_overlay.last_used = time.time()
                
                self._active_overlays[overlay_id] = pooled_overlay
                self._logger.debug(f"Reused pooled overlay for {overlay_id}")
                return pooled_overlay.overlay
            else:
                # Create new overlay
                overlay = self._create_new_overlay(config)
                if overlay:
                    pooled_overlay = PooledOverlay(
                        overlay=overlay,
                        created_at=time.time(),
                        last_used=time.time(),
                        in_use=True,
                        overlay_id=overlay_id
                    )
                    
                    self._active_overlays[overlay_id] = pooled_overlay
                    self._logger.debug(f"Created new overlay for {overlay_id}")
                    return overlay
                else:
                    self._logger.warning(f"Failed to create overlay for {overlay_id}")
                    return None
                    
        except Exception as e:
            self._logger.error(f"Error acquiring overlay {overlay_id}: {e}")
            return None
    
    def release_overlay(self, overlay_id: str, return_to_pool: bool = True) -> None:
        """Release an overlay back to the pool or destroy it.
        
        Must be called from UI thread.
        """
        try:
            if overlay_id not in self._active_overlays:
                self._logger.debug(f"Overlay {overlay_id} not found in active overlays")
                return
            
            pooled_overlay = self._active_overlays.pop(overlay_id)
            pooled_overlay.in_use = False
            pooled_overlay.overlay_id = None
            pooled_overlay.last_used = time.time()
            
            if return_to_pool and len(self._pool) < self._max_pool_size:
                # Return to pool for reuse
                self._pool.append(pooled_overlay)
                self._logger.debug(f"Returned overlay {overlay_id} to pool")
            else:
                # Destroy overlay
                self._destroy_overlay(pooled_overlay.overlay)
                self._logger.debug(f"Destroyed overlay {overlay_id}")
                
        except Exception as e:
            self._logger.error(f"Error releasing overlay {overlay_id}: {e}")
    
    def _get_from_pool(self) -> Optional[PooledOverlay]:
        """Get an available overlay from the pool."""
        while self._pool:
            pooled = self._pool.pop()
            
            # Validate that Qt widgets are still alive
            try:
                overlay = pooled.overlay
                if hasattr(overlay, '_host') and overlay._host is not None:
                    # Try to access a Qt property to verify it's not deleted
                    try:
                        _ = overlay._host.isVisible()
                        # Host is valid, return this overlay
                        self._logger.debug("Validated pooled overlay - Qt widgets intact")
                        return pooled
                    except RuntimeError:
                        # Qt object deleted - destroy and continue
                        self._logger.warning("Pooled overlay has deleted Qt widgets - discarding")
                        self._destroy_overlay(overlay)
                        continue
                else:
                    # No host - overlay not usable
                    self._logger.warning("Pooled overlay missing host - discarding")
                    self._destroy_overlay(overlay)
                    continue
            except Exception as e:
                self._logger.warning(f"Error validating pooled overlay: {e} - discarding")
                try:
                    self._destroy_overlay(pooled.overlay)
                except Exception:
                    pass
                continue
        
        # Pool exhausted or all overlays invalid
        return None
    
    def _create_new_overlay(self, config: OverlayConfig) -> Optional[object]:
        """Create a new DWM overlay."""
        try:
            # Force DWM backend for docking overlays
            overlay = self._backend_manager.create_overlay(config, BackendType.DWM)
            return overlay
        except Exception as e:
            self._logger.warning(f"Failed to create new overlay: {e}")
            return None
    
    def _destroy_overlay(self, overlay: object) -> None:
        """Safely destroy an overlay."""
        try:
            if hasattr(overlay, 'cleanup'):
                overlay.cleanup()
            elif hasattr(overlay, 'close'):
                overlay.close()
            elif hasattr(overlay, 'destroy'):
                overlay.destroy()
        except Exception as e:
            self._logger.debug(f"Error destroying overlay: {e}")
    
    def _schedule_cleanup(self) -> None:
        """Schedule periodic cleanup of idle overlays."""
        try:
            from PySide6.QtCore import QTimer
            
            # Unregister old timer if exists
            if self._timer_resource_id:
                self._resource_manager.unregister(self._timer_resource_id)
                self._timer_resource_id = None
            
            if self._cleanup_timer:
                self._cleanup_timer.stop()
            
            self._cleanup_timer = QTimer()
            self._cleanup_timer.setSingleShot(False)
            self._cleanup_timer.timeout.connect(self._cleanup_idle_overlays)
            self._cleanup_timer.start(60000)  # Cleanup every minute
            
            # Register with ResourceManager for proper cleanup
            self._timer_resource_id = self._resource_manager.register_qt_timer(
                self._cleanup_timer,
                description="DockingOverlayPool cleanup timer"
            )
            
        except Exception as e:
            self._logger.debug(f"Failed to schedule cleanup: {e}")
    
    def _cleanup_idle_overlays(self) -> None:
        """Clean up overlays that have been idle too long."""
        with self._lock:
            try:
                current_time = time.time()
                overlays_to_remove = []
                
                for i, pooled_overlay in enumerate(self._pool):
                    if (current_time - pooled_overlay.last_used) > self._max_idle_time:
                        overlays_to_remove.append(i)
                
                # Remove idle overlays (in reverse order to maintain indices)
                for i in reversed(overlays_to_remove):
                    pooled_overlay = self._pool.pop(i)
                    self._destroy_overlay(pooled_overlay.overlay)
                
                if overlays_to_remove:
                    self._logger.debug(f"Cleaned up {len(overlays_to_remove)} idle overlays")
                    
            except Exception as e:
                self._logger.debug(f"Error during idle overlay cleanup: {e}")
    
    def get_pool_stats(self) -> Dict[str, int]:
        """Get current pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'active_overlays': len(self._active_overlays),
                'max_pool_size': self._max_pool_size
            }
    
    def clear_pool(self) -> None:
        """Clear all pooled overlays."""
        with self._lock:
            try:
                # Destroy all pooled overlays
                for pooled_overlay in self._pool:
                    self._destroy_overlay(pooled_overlay.overlay)
                
                self._pool.clear()
                self._logger.debug("Cleared overlay pool")
                
            except Exception as e:
                self._logger.error(f"Error clearing pool: {e}")
    
    def cleanup(self) -> None:
        """Clean up the pool and all resources."""
        try:
            # Unregister and stop cleanup timer
            if self._timer_resource_id:
                self._resource_manager.unregister(self._timer_resource_id)
                self._timer_resource_id = None
            
            if self._cleanup_timer:
                self._cleanup_timer.stop()
                self._cleanup_timer = None
            
            # Release all active overlays
            active_ids = list(self._active_overlays.keys())
            for overlay_id in active_ids:
                self.release_overlay(overlay_id, return_to_pool=False)
            
            # Clear the pool
            self.clear_pool()
            
            self._logger.debug("DockingOverlayPool cleanup completed")
            
        except Exception as e:
            self._logger.error(f"Error during pool cleanup: {e}")


# Global pool instance
_overlay_pool: Optional[DockingOverlayPool] = None


def get_docking_overlay_pool() -> DockingOverlayPool:
    """Get the global docking overlay pool instance."""
    global _overlay_pool
    if _overlay_pool is None:
        _overlay_pool = DockingOverlayPool()
    return _overlay_pool


def cleanup_docking_overlay_pool() -> None:
    """Clean up the global overlay pool."""
    global _overlay_pool
    if _overlay_pool:
        _overlay_pool.cleanup()
        _overlay_pool = None
