"""
Animated Button System - Provides smooth hover/press animations for QPushButton widgets.

Integrates with SPQDocker's centralized architecture (ThreadManager, ResourceManager, etc.)
while providing professional button feedback animations.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QColor, QPalette

from core.logging import get_logger
from core.animation import get_animation_manager, AnimationType
from core.threading import get_thread_manager
from utils.resource_manager import get_resource_manager, ResourceType

logger = get_logger(__name__)


class ButtonAnimator(QObject):
    """
    Provides smooth hover/press animations for QPushButton widgets.
    
    Features:
    - Smooth color transitions on hover/press
    - Centralized animation management
    - Thread-safe operations via ThreadManager
    - Resource cleanup via ResourceManager
    - Performance-aware animation culling
    """
    
    def __init__(self):
        """Initialize the button animator."""
        super().__init__()
        self._animated_buttons: Dict[QPushButton, Dict[str, Any]] = {}
        self._animation_manager = None
        self._thread_manager = None
        self._resource_id = None
        self._logger = get_logger(__name__)
        
        # Initialize centralized managers
        try:
            self._animation_manager = get_animation_manager()
            self._thread_manager = get_thread_manager()
            
            # Register with resource manager for cleanup
            rm = get_resource_manager()
            self._resource_id = rm.register(
                ResourceType.UI_ANIMATOR,
                self,
                cleanup_func=self.cleanup
            )
            
        except Exception as e:
            self._logger.warning(f"Failed to initialize button animator: {e}")
            
    def add_button(self, button: QPushButton, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add smooth animations to a QPushButton.
        
        Args:
            button: The QPushButton to animate
            config: Animation configuration options
                - hover_color: QColor for hover state
                - press_color: QColor for press state  
                - duration_ms: Animation duration override
                - disabled: Disable animations for this button
                
        Returns:
            True if animation was successfully added
        """
        if not self._animation_manager or not self._thread_manager:
            return False
            
        if button in self._animated_buttons:
            return True  # Already animated
            
        # Default configuration
        default_config = {
            'hover_color': None,  # Auto-detect from theme
            'press_color': None,  # Auto-detect from theme
            'duration_ms': None,  # Use standard duration
            'disabled': False,
            'original_palette': button.palette()
        }
        
        if config:
            default_config.update(config)
            
        if default_config['disabled']:
            return False
            
        try:
            # Store button state
            self._animated_buttons[button] = {
                'config': default_config,
                'hover_animation': None,
                'press_animation': None,
                'current_state': 'normal',
                'original_enter_event': getattr(button, 'enterEvent', None),
                'original_leave_event': getattr(button, 'leaveEvent', None),
                'original_press_event': getattr(button, 'mousePressEvent', None),
                'original_release_event': getattr(button, 'mouseReleaseEvent', None),
            }
            
            # Install event handlers
            self._install_event_handlers(button)
            
            self._logger.debug(f"Added animations to button: {button.objectName()}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to add button animations: {e}")
            return False
            
    def remove_button(self, button: QPushButton) -> None:
        """Remove animations from a button and restore original behavior."""
        if button not in self._animated_buttons:
            return
            
        try:
            button_data = self._animated_buttons[button]
            
            # Stop any active animations
            if button_data['hover_animation']:
                button_data['hover_animation'].stop()
            if button_data['press_animation']:
                button_data['press_animation'].stop()
                
            # Restore original event handlers
            if button_data['original_enter_event']:
                button.enterEvent = button_data['original_enter_event']
            if button_data['original_leave_event']:
                button.leaveEvent = button_data['original_leave_event']
            if button_data['original_press_event']:
                button.mousePressEvent = button_data['original_press_event']
            if button_data['original_release_event']:
                button.mouseReleaseEvent = button_data['original_release_event']
                
            # Restore original palette
            button.setPalette(button_data['config']['original_palette'])
            
            del self._animated_buttons[button]
            self._logger.debug(f"Removed animations from button: {button.objectName()}")
            
        except Exception as e:
            self._logger.error(f"Failed to remove button animations: {e}")
            
    def _install_event_handlers(self, button: QPushButton) -> None:
        """Install custom event handlers for smooth animations."""
        button_data = self._animated_buttons[button]
        
        def animated_enter_event(event):
            """Handle mouse enter with smooth animation."""
            try:
                if button_data['current_state'] != 'hover':
                    self._animate_button_state(button, 'hover')
                    
                # Call original handler if it exists
                if button_data['original_enter_event']:
                    button_data['original_enter_event'](event)
            except Exception as e:
                self._logger.debug(f"Enter event error: {e}")
                
        def animated_leave_event(event):
            """Handle mouse leave with smooth animation."""
            try:
                if button_data['current_state'] != 'normal':
                    self._animate_button_state(button, 'normal')
                    
                # Call original handler if it exists
                if button_data['original_leave_event']:
                    button_data['original_leave_event'](event)
            except Exception as e:
                self._logger.debug(f"Leave event error: {e}")
                
        def animated_press_event(event):
            """Handle mouse press with smooth animation."""
            try:
                if button_data['current_state'] != 'press':
                    self._animate_button_state(button, 'press')
                    
                # Call original handler if it exists
                if button_data['original_press_event']:
                    button_data['original_press_event'](event)
            except Exception as e:
                self._logger.debug(f"Press event error: {e}")
                
        def animated_release_event(event):
            """Handle mouse release with smooth animation."""
            try:
                # Return to hover state if mouse still over button
                if button.underMouse():
                    self._animate_button_state(button, 'hover')
                else:
                    self._animate_button_state(button, 'normal')
                    
                # Call original handler if it exists
                if button_data['original_release_event']:
                    button_data['original_release_event'](event)
            except Exception as e:
                self._logger.debug(f"Release event error: {e}")
                
        # Replace event handlers
        button.enterEvent = animated_enter_event
        button.leaveEvent = animated_leave_event
        button.mousePressEvent = animated_press_event
        button.mouseReleaseEvent = animated_release_event
        
    def _animate_button_state(self, button: QPushButton, target_state: str) -> None:
        """Animate button to target state."""
        if not self._animation_manager or button not in self._animated_buttons:
            return
            
        button_data = self._animated_buttons[button]
        
        if button_data['current_state'] == target_state:
            return  # Already in target state
            
        def _do_animation():
            try:
                # Stop existing animations
                if button_data['hover_animation']:
                    button_data['hover_animation'].stop()
                    button_data['hover_animation'] = None
                if button_data['press_animation']:
                    button_data['press_animation'].stop()
                    button_data['press_animation'] = None
                    
                # Get target colors
                target_color = self._get_state_color(button, target_state)
                if not target_color:
                    return
                    
                # Create color animation
                animation = self._animation_manager.create_animation(
                    AnimationType.BUTTON_HOVER,
                    button,
                    b"palette",  # We'll animate the palette
                    end_value=self._create_palette_with_color(button, target_color)
                )
                
                if animation:
                    # Store animation reference
                    if target_state in ['hover', 'normal']:
                        button_data['hover_animation'] = animation
                    else:
                        button_data['press_animation'] = animation
                        
                    # Update state tracking
                    button_data['current_state'] = target_state
                    
                    animation.start()
                    
            except Exception as e:
                self._logger.debug(f"Button animation failed: {e}")
                
        # Run on UI thread via ThreadManager (centralized threading)
        try:
            self._thread_manager.run_on_ui_thread(_do_animation)
        except Exception:
            pass  # Fallback: no animation
            
    def _get_state_color(self, button: QPushButton, state: str) -> Optional[QColor]:
        """Get the appropriate color for a button state."""
        button_data = self._animated_buttons[button]
        config = button_data['config']
        
        if state == 'hover':
            if config['hover_color']:
                return config['hover_color']
            # Auto-generate hover color (lighter)
            return self._lighten_color(button.palette().button().color(), 0.2)
            
        elif state == 'press':
            if config['press_color']:
                return config['press_color']
            # Auto-generate press color (darker)
            return self._darken_color(button.palette().button().color(), 0.3)
            
        else:  # normal
            return config['original_palette'].button().color()
            
    def _lighten_color(self, color: QColor, factor: float) -> QColor:
        """Lighten a color by the specified factor."""
        h, s, lightness, a = color.getHslF()
        lightness = min(1.0, lightness + factor)
        return QColor.fromHslF(h, s, lightness, a)
        
    def _darken_color(self, color: QColor, factor: float) -> QColor:
        """Darken a color by the specified factor."""
        h, s, lightness, a = color.getHslF()
        lightness = max(0.0, lightness - factor)
        return QColor.fromHslF(h, s, lightness, a)
        
    def _create_palette_with_color(self, button: QPushButton, color: QColor) -> QPalette:
        """Create a palette with the specified button color."""
        palette = button.palette()
        palette.setColor(QPalette.Button, color)
        return palette
        
    def cleanup(self) -> None:
        """Cleanup all animated buttons and resources."""
        try:
            # Remove animations from all buttons
            for button in list(self._animated_buttons.keys()):
                self.remove_button(button)
                
            self._animated_buttons.clear()
            
            # Unregister from resource manager
            if self._resource_id:
                try:
                    rm = get_resource_manager()
                    rm.unregister(self._resource_id, force=True)
                except Exception as e:
                    self._logger.warning(f"Failed to unregister from ResourceManager: {e}")
                    
        except Exception as e:
            self._logger.error(f"Error during button animator cleanup: {e}")
            
        self._logger.debug("Button animator cleanup complete")


# Global button animator instance
_button_animator: Optional[ButtonAnimator] = None


def get_button_animator() -> ButtonAnimator:
    """Get the global button animator instance."""
    global _button_animator
    if _button_animator is None:
        _button_animator = ButtonAnimator()
    return _button_animator


def animate_button(button: QPushButton, config: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to add animations to a button."""
    return get_button_animator().add_button(button, config)


def remove_button_animation(button: QPushButton) -> None:
    """Convenience function to remove animations from a button."""
    get_button_animator().remove_button(button)
