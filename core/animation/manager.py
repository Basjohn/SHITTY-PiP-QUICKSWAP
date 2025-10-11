"""
SPQ Animation Manager - Centralized animation coordination and performance management.

Provides a unified framework for creating, managing, and monitoring animations
throughout the application while maintaining 60fps performance targets.
"""
from __future__ import annotations

from typing import Dict, Optional, Callable, Any
from enum import Enum

from PySide6.QtCore import (
    QObject, QPropertyAnimation, QEasingCurve, QAbstractAnimation,
    QParallelAnimationGroup, QTimer, Signal
)
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QColor

from core.logging import get_logger
from core.settings import get_settings_manager
from core.threading import get_thread_manager
from utils.resource_manager import get_resource_manager, ResourceType

logger = get_logger(__name__)


class AnimationType(Enum):
    """Standard animation types with predefined durations and easing."""
    OVERLAY_SHOW = "overlay_show"
    OVERLAY_HIDE = "overlay_hide"
    OVERLAY_RESIZE = "overlay_resize"
    FOCUS_CHANGE = "focus_change"
    BUTTON_HOVER = "button_hover"
    CONTEXT_MENU = "context_menu"
    THEME_TRANSITION = "theme_transition"
    TOAST_SHOW = "toast_show"
    TOAST_HIDE = "toast_hide"
    LOADING_SPINNER = "loading_spinner"


class AnimationPerformanceMonitor:
    """Monitors animation performance and provides adaptive quality control."""
    
    def __init__(self):
        self._frame_times: list[float] = []
        self._dropped_frames = 0
        self._total_animations = 0
        self._performance_degraded = False
        self._logger = get_logger(f"{__name__}.PerformanceMonitor")
        
    def record_frame_time(self, frame_time_ms: float) -> None:
        """Record a frame time for performance analysis."""
        self._frame_times.append(frame_time_ms)
        
        # Keep only last 100 frames for analysis
        if len(self._frame_times) > 100:
            self._frame_times.pop(0)
            
        # Track dropped frames (>16.67ms = <60fps)
        if frame_time_ms > 16.67:
            self._dropped_frames += 1
            
    def record_animation_completion(self, animation_type: AnimationType, duration_ms: int) -> None:
        """Record successful animation completion."""
        self._total_animations += 1
        self._logger.debug(f"Animation completed: {animation_type.value} ({duration_ms}ms)")
        
    def is_system_stressed(self) -> bool:
        """Check if system is under performance stress."""
        if len(self._frame_times) < 10:
            return False
            
        # Calculate recent dropped frame rate
        recent_frames = self._frame_times[-10:]
        dropped_recent = sum(1 for ft in recent_frames if ft > 16.67)
        
        # Consider stressed if >30% frames dropped recently
        stress_threshold = 0.3
        is_stressed = (dropped_recent / len(recent_frames)) > stress_threshold
        
        if is_stressed and not self._performance_degraded:
            self._performance_degraded = True
            self._logger.warning("Performance degradation detected, reducing animation quality")
        elif not is_stressed and self._performance_degraded:
            self._performance_degraded = False
            self._logger.info("Performance recovered, restoring animation quality")
            
        return is_stressed
        
    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        if not self._frame_times:
            return {"avg_frame_time": 0, "dropped_frames": 0, "total_animations": 0}
            
        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        return {
            "avg_frame_time": avg_frame_time,
            "dropped_frames": self._dropped_frames,
            "total_animations": self._total_animations,
            "performance_degraded": self._performance_degraded
        }


class SPQAnimationManager(QObject):
    """
    Centralized animation manager providing standardized animations
    with performance monitoring and adaptive quality control.
    """
    
    # Signals
    animation_started = Signal(str)  # animation_id
    animation_finished = Signal(str)  # animation_id
    performance_degraded = Signal(bool)  # degraded state
    
    _instance: Optional['SPQAnimationManager'] = None
    
    def __new__(cls):
        """Singleton pattern for centralized animation management."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the animation manager."""
        if self._initialized:
            return
            
        super().__init__()
        self._active_animations: Dict[str, QAbstractAnimation] = {}
        self._animation_counter = 0
        self._settings_manager = get_settings_manager()
        self._thread_manager = get_thread_manager()
        self._performance_monitor = AnimationPerformanceMonitor()
        self._logger = get_logger(__name__)
        
        # Load animation settings
        self._load_animation_settings()
        
        # Performance monitoring timer - register with ResourceManager
        self._performance_timer = QTimer()
        self._performance_timer.timeout.connect(self._check_performance)
        self._performance_timer.start(1000)  # Check every second
        
        # Register with resource manager
        try:
            rm = get_resource_manager()
            self._resource_id = rm.register(
                ResourceType.ANIMATION_MANAGER,
                self,
                cleanup_func=self.shutdown
            )
            # Register the performance timer separately for proper cleanup
            self._timer_resource_id = rm.register_qt_timer(
                self._performance_timer,
                description="AnimationManager performance timer"
            )
        except Exception as e:
            self._logger.warning(f"Failed to register with ResourceManager: {e}")
            self._resource_id = None
            self._timer_resource_id = None
            
        self._initialized = True
        self._logger.debug("SPQAnimationManager initialized")
        
    def _load_animation_settings(self) -> None:
        """Load animation preferences from settings."""
        try:
            self._animations_enabled = self._settings_manager.get('animations.enabled', True)
            self._reduce_motion = self._settings_manager.get('animations.reduce_motion', False)
            self._performance_mode = self._settings_manager.get('animations.performance_mode', 'balanced')
            self._duration_multiplier = self._settings_manager.get('animations.custom_duration_multiplier', 1.0)
            self._overlay_transitions = self._settings_manager.get('animations.overlay_transitions', True)
            self._theme_transitions = self._settings_manager.get('animations.theme_transitions', True)
            self._ui_feedback = self._settings_manager.get('animations.ui_feedback', True)
            
            # Check system accessibility settings for reduce motion
            if self._check_system_reduce_motion():
                self._reduce_motion = True
                self._logger.info("System reduce motion preference detected")
                
        except Exception as e:
            self._logger.warning(f"Failed to load animation settings: {e}")
            # Set safe defaults
            self._animations_enabled = True
            self._reduce_motion = False
            self._performance_mode = 'balanced'
            self._duration_multiplier = 1.0
            
    def _check_system_reduce_motion(self) -> bool:
        """Check Windows system preference for reduced motion."""
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, 
                r"Control Panel\Accessibility\HighContrast"
            )
            value, _ = winreg.QueryValueEx(key, "Flags")
            return bool(value & 0x80)  # HCF_ANIMATIONS flag
        except Exception:
            return False
            
    def create_animation(
        self,
        animation_type: AnimationType,
        target: QWidget,
        property_name: bytes,
        start_value: Any = None,
        end_value: Any = None,
        duration_ms: Optional[int] = None,
        easing_curve: Optional[QEasingCurve.Type] = None,
        finished_callback: Optional[Callable] = None
    ) -> Optional[QPropertyAnimation]:
        """
        Create a standardized animation with performance monitoring.
        
        Args:
            animation_type: Type of animation to create
            target: Target widget for animation
            property_name: Qt property to animate (e.g., b"geometry", b"windowOpacity")
            start_value: Starting value (None = current value)
            end_value: Ending value
            duration_ms: Duration override (None = use standard)
            easing_curve: Easing curve override (None = use standard)
            finished_callback: Optional callback on completion
            
        Returns:
            QPropertyAnimation or None if animations disabled
        """
        if not self._should_create_animation(target):
            return None
            
        # Generate unique animation ID
        animation_id = f"{animation_type.value}_{self._animation_counter}"
        self._animation_counter += 1
        
        # Create animation
        animation = QPropertyAnimation(target, property_name)
        
        # Set values
        if start_value is not None:
            animation.setStartValue(start_value)
        animation.setEndValue(end_value)
        
        # Apply standard duration and easing
        duration = duration_ms or self._get_standard_duration(animation_type)
        duration = int(duration * self._duration_multiplier)
        animation.setDuration(duration)
        
        easing = easing_curve or self._get_standard_easing(animation_type)
        animation.setEasingCurve(easing)
        
        # Setup completion handling
        def on_finished():
            self._on_animation_finished(animation_id, animation_type, duration)
            if finished_callback:
                try:
                    finished_callback()
                except Exception as e:
                    self._logger.error(f"Error in animation callback: {e}")
                    
        animation.finished.connect(on_finished)
        
        # Store and start animation
        self._active_animations[animation_id] = animation
        
        self._logger.debug(f"Created animation: {animation_id} ({duration}ms)")
        self.animation_started.emit(animation_id)
        
        return animation
        
    def create_overlay_show_animation(self, overlay_host: QWidget) -> Optional[QParallelAnimationGroup]:
        """Create smooth overlay show animation with fade and subtle scale."""
        if not self._overlay_transitions or not self._should_create_animation(overlay_host):
            return None
            
        # Create animation group
        group = QParallelAnimationGroup()
        
        # Fade animation
        fade_animation = self.create_animation(
            AnimationType.OVERLAY_SHOW,
            overlay_host,
            b"windowOpacity",
            start_value=0.0,
            end_value=1.0
        )
        
        if fade_animation:
            group.addAnimation(fade_animation)
            
        # Subtle scale animation (95% -> 100%)
        if hasattr(overlay_host, 'geometry'):
            original_rect = overlay_host.geometry()
            start_rect = self._scale_rect(original_rect, 0.95)
            
            scale_animation = self.create_animation(
                AnimationType.OVERLAY_SHOW,
                overlay_host,
                b"geometry",
                start_value=start_rect,
                end_value=original_rect
            )
            
            if scale_animation:
                group.addAnimation(scale_animation)
                
        return group if group.animationCount() > 0 else None
        
    def create_overlay_hide_animation(self, overlay_host: QWidget) -> Optional[QParallelAnimationGroup]:
        """Create smooth overlay hide animation with fade and subtle scale."""
        if not self._overlay_transitions or not self._should_create_animation(overlay_host):
            return None
            
        # Create animation group
        group = QParallelAnimationGroup()
        
        # Fade animation
        fade_animation = self.create_animation(
            AnimationType.OVERLAY_HIDE,
            overlay_host,
            b"windowOpacity",
            start_value=overlay_host.windowOpacity(),
            end_value=0.0
        )
        
        if fade_animation:
            group.addAnimation(fade_animation)
            
        return group if group.animationCount() > 0 else None
        
    def create_focus_animation(
        self, 
        target: QWidget, 
        focused: bool,
        focus_color: QColor = None,
        unfocus_color: QColor = None
    ) -> Optional[QPropertyAnimation]:
        """Create smooth focus indicator animation."""
        if not self._ui_feedback or not self._should_create_animation(target):
            return None
            
        # Default colors if not provided
        focus_color = focus_color or QColor("#007ACC")
        unfocus_color = unfocus_color or QColor("#404040")
        
        end_color = focus_color if focused else unfocus_color
        
        return self.create_animation(
            AnimationType.FOCUS_CHANGE,
            target,
            b"borderColor",
            end_value=end_color
        )
        
    def _should_create_animation(self, target: QWidget) -> bool:
        """Determine if animation should be created based on settings and performance."""
        if not self._animations_enabled:
            return False
            
        if self._reduce_motion:
            return False
            
        # Skip animations for invisible widgets
        if not target.isVisible():
            return False
            
        # Skip if system is under stress
        if self._performance_monitor.is_system_stressed():
            return False
            
        # Limit concurrent animations
        if len(self._active_animations) > 15:
            self._logger.warning("Too many concurrent animations, skipping")
            return False
            
        return True
        
    def _get_standard_duration(self, animation_type: AnimationType) -> int:
        """Get standard duration for animation type."""
        durations = {
            AnimationType.OVERLAY_SHOW: 250,
            AnimationType.OVERLAY_HIDE: 200,
            AnimationType.OVERLAY_RESIZE: 200,
            AnimationType.FOCUS_CHANGE: 150,
            AnimationType.BUTTON_HOVER: 100,
            AnimationType.CONTEXT_MENU: 150,
            AnimationType.THEME_TRANSITION: 300,
            AnimationType.TOAST_SHOW: 400,
            AnimationType.TOAST_HIDE: 200,
            AnimationType.LOADING_SPINNER: 1000
        }
        
        base_duration = durations.get(animation_type, 200)
        
        # Adjust for performance mode
        if self._performance_mode == 'fast':
            return int(base_duration * 0.7)
        elif self._performance_mode == 'smooth':
            return int(base_duration * 1.3)
        else:  # balanced
            return base_duration
            
    def _get_standard_easing(self, animation_type: AnimationType) -> QEasingCurve.Type:
        """Get standard easing curve for animation type."""
        easing_curves = {
            AnimationType.OVERLAY_SHOW: QEasingCurve.OutCubic,
            AnimationType.OVERLAY_HIDE: QEasingCurve.InCubic,
            AnimationType.OVERLAY_RESIZE: QEasingCurve.OutQuart,
            AnimationType.FOCUS_CHANGE: QEasingCurve.OutQuad,
            AnimationType.BUTTON_HOVER: QEasingCurve.OutQuad,
            AnimationType.CONTEXT_MENU: QEasingCurve.OutBack,
            AnimationType.THEME_TRANSITION: QEasingCurve.InOutQuad,
            AnimationType.TOAST_SHOW: QEasingCurve.OutBack,
            AnimationType.TOAST_HIDE: QEasingCurve.InQuad,
            AnimationType.LOADING_SPINNER: QEasingCurve.Linear
        }
        
        return easing_curves.get(animation_type, QEasingCurve.OutQuad)
        
    def _scale_rect(self, rect, scale_factor: float):
        """Scale a rectangle around its center."""
        from PySide6.QtCore import QRect
        
        center = rect.center()
        new_width = int(rect.width() * scale_factor)
        new_height = int(rect.height() * scale_factor)
        
        scaled_rect = QRect(0, 0, new_width, new_height)
        scaled_rect.moveCenter(center)
        
        return scaled_rect
        
    def _on_animation_finished(self, animation_id: str, animation_type: AnimationType, duration: int) -> None:
        """Handle animation completion."""
        if animation_id in self._active_animations:
            del self._active_animations[animation_id]
            
        self._performance_monitor.record_animation_completion(animation_type, duration)
        self.animation_finished.emit(animation_id)
        
    def _check_performance(self) -> None:
        """Periodic performance check."""
        stats = self._performance_monitor.get_performance_stats()
        
        # Emit performance signal if state changed
        if stats.get("performance_degraded", False):
            self.performance_degraded.emit(True)
        else:
            self.performance_degraded.emit(False)
            
    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        return self._performance_monitor.get_performance_stats()
        
    def stop_all_animations(self) -> None:
        """Stop all active animations."""
        for animation in list(self._active_animations.values()):
            try:
                animation.stop()
            except Exception as e:
                self._logger.warning(f"Error stopping animation: {e}")
                
        self._active_animations.clear()
        self._logger.debug("Stopped all animations")
        
    def shutdown(self) -> None:
        """Shutdown animation manager and cleanup resources."""
        try:
            self.stop_all_animations()
            
            # Unregister and stop performance timer
            if hasattr(self, '_timer_resource_id') and self._timer_resource_id is not None:
                try:
                    rm = get_resource_manager()
                    rm.unregister(self._timer_resource_id)
                except Exception as e:
                    self._logger.warning(f"Failed to unregister timer: {e}")
            
            if hasattr(self, '_performance_timer'):
                self._performance_timer.stop()
                
            # Unregister from resource manager
            if hasattr(self, '_resource_id') and self._resource_id is not None:
                try:
                    rm = get_resource_manager()
                    rm.unregister(self._resource_id, force=True)
                except Exception as e:
                    self._logger.warning(f"Failed to unregister from ResourceManager: {e}")
                    
        except Exception as e:
            self._logger.error(f"Error during animation manager shutdown: {e}")
            
        self._logger.debug("Animation manager shutdown complete")


# Singleton instance accessor
def get_animation_manager() -> SPQAnimationManager:
    """Get the singleton animation manager instance."""
    return SPQAnimationManager()
