"""
Context Menu Animation System - Provides smooth slide-in animations for QMenu widgets.

Integrates with existing context menu system and centralized architecture.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, QPoint
from PySide6.QtWidgets import QMenu, QGraphicsOpacityEffect
from PySide6.QtGui import QShowEvent

from core.logging import get_logger
from core.animation import get_animation_manager, AnimationType
from core.threading import get_thread_manager
from utils.resource_manager import get_resource_manager, ResourceType

logger = get_logger(__name__)


class ContextMenuAnimator(QObject):
    """
    Provides smooth slide-in animations for context menus.
    
    Features:
    - Smooth slide-in from trigger point
    - Fade animation for professional appearance
    - Integrates with existing context menu system
    - Thread-safe via ThreadManager
    - Resource cleanup via ResourceManager
    """
    
    def __init__(self):
        """Initialize the context menu animator."""
        super().__init__()
        self._animated_menus: Dict[QMenu, Dict[str, Any]] = {}
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
            self._logger.warning(f"Failed to initialize context menu animator: {e}")
            
    def animate_menu_show(self, menu: QMenu, trigger_point: QPoint) -> bool:
        """
        Add smooth slide-in animation to a context menu.
        
        Args:
            menu: The QMenu to animate
            trigger_point: The point where the menu was triggered
            
        Returns:
            True if animation was successfully set up
        """
        if not self._animation_manager or not self._thread_manager:
            return False
            
        if menu in self._animated_menus:
            return True  # Already animated
            
        try:
            # Store menu state
            self._animated_menus[menu] = {
                'trigger_point': trigger_point,
                'slide_animation': None,
                'fade_animation': None,
                'original_show_event': getattr(menu, 'showEvent', None),
            }
            
            # Install custom show event handler
            self._install_show_event_handler(menu)
            
            self._logger.debug("Added animations to context menu")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to add context menu animations: {e}")
            return False
            
    def remove_menu_animation(self, menu: QMenu) -> None:
        """Remove animations from a menu and restore original behavior."""
        if menu not in self._animated_menus:
            return
            
        try:
            menu_data = self._animated_menus[menu]
            
            # Stop any active animations
            if menu_data['slide_animation']:
                menu_data['slide_animation'].stop()
            if menu_data['fade_animation']:
                menu_data['fade_animation'].stop()
                
            # Restore original show event handler
            if menu_data['original_show_event']:
                menu.showEvent = menu_data['original_show_event']
                
            # Remove opacity effect if we added one
            menu.setGraphicsEffect(None)
            
            del self._animated_menus[menu]
            self._logger.debug("Removed animations from context menu")
            
        except Exception as e:
            self._logger.error(f"Failed to remove context menu animations: {e}")
            
    def _install_show_event_handler(self, menu: QMenu) -> None:
        """Install custom show event handler for smooth animations."""
        menu_data = self._animated_menus[menu]
        
        def animated_show_event(event: QShowEvent):
            """Handle show event with smooth animation."""
            try:
                # Call original handler first
                if menu_data['original_show_event']:
                    menu_data['original_show_event'](event)
                    
                # Start animations after the menu is visible
                self._start_menu_animations(menu)
                
            except Exception as e:
                self._logger.debug(f"Show event error: {e}")
                
        # Replace show event handler
        menu.showEvent = animated_show_event
        
    def _start_menu_animations(self, menu: QMenu) -> None:
        """Start slide-in and fade animations for the menu."""
        if not self._animation_manager or menu not in self._animated_menus:
            return
            
        menu_data = self._animated_menus[menu]
        trigger_point = menu_data['trigger_point']
        
        def _do_animations():
            try:
                # Set up opacity effect for fade animation
                opacity_effect = QGraphicsOpacityEffect()
                menu.setGraphicsEffect(opacity_effect)
                
                # Calculate start position (slightly offset from trigger point)
                current_pos = menu.pos()
                start_pos = QPoint(
                    trigger_point.x(),
                    trigger_point.y() - 10  # Start 10px above trigger point
                )
                
                # Position the menu at start position
                menu.move(start_pos)
                
                # Create slide animation
                slide_animation = self._animation_manager.create_animation(
                    AnimationType.CONTEXT_MENU,
                    menu,
                    b"pos",
                    start_value=start_pos,
                    end_value=current_pos
                )
                
                # Create fade animation
                fade_animation = self._animation_manager.create_animation(
                    AnimationType.CONTEXT_MENU,
                    opacity_effect,
                    b"opacity",
                    start_value=0.0,
                    end_value=1.0
                )
                
                if slide_animation and fade_animation:
                    # Store animation references
                    menu_data['slide_animation'] = slide_animation
                    menu_data['fade_animation'] = fade_animation
                    
                    # Start both animations
                    slide_animation.start()
                    fade_animation.start()
                    
                    # Clean up after animations complete
                    fade_animation.finished.connect(
                        lambda: self._on_animation_finished(menu)
                    )
                    
            except Exception as e:
                self._logger.debug(f"Context menu animation failed: {e}")
                
        # Run on UI thread via ThreadManager (centralized threading)
        try:
            self._thread_manager.run_on_ui_thread(_do_animations)
        except Exception:
            pass  # Fallback: no animation
            
    def _on_animation_finished(self, menu: QMenu) -> None:
        """Handle animation completion and cleanup."""
        try:
            if menu in self._animated_menus:
                menu_data = self._animated_menus[menu]
                
                # Clear animation references
                menu_data['slide_animation'] = None
                menu_data['fade_animation'] = None
                
        except Exception as e:
            self._logger.debug(f"Animation cleanup failed: {e}")
            
    def cleanup(self) -> None:
        """Cleanup all animated menus and resources."""
        try:
            # Remove animations from all menus
            for menu in list(self._animated_menus.keys()):
                self.remove_menu_animation(menu)
                
            self._animated_menus.clear()
            
            # Unregister from resource manager
            if self._resource_id:
                try:
                    rm = get_resource_manager()
                    rm.unregister(self._resource_id, force=True)
                except Exception as e:
                    self._logger.warning(f"Failed to unregister from ResourceManager: {e}")
                    
        except Exception as e:
            self._logger.error(f"Error during context menu animator cleanup: {e}")
            
        self._logger.debug("Context menu animator cleanup complete")


# Global context menu animator instance
_context_menu_animator: Optional[ContextMenuAnimator] = None


def get_context_menu_animator() -> ContextMenuAnimator:
    """Get the global context menu animator instance."""
    global _context_menu_animator
    if _context_menu_animator is None:
        _context_menu_animator = ContextMenuAnimator()
    return _context_menu_animator


def animate_context_menu(menu: QMenu, trigger_point: QPoint) -> bool:
    """Convenience function to add animations to a context menu."""
    return get_context_menu_animator().animate_menu_show(menu, trigger_point)


def remove_context_menu_animation(menu: QMenu) -> None:
    """Convenience function to remove animations from a context menu."""
    get_context_menu_animator().remove_menu_animation(menu)
