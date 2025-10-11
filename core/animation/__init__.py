"""
Animation System Module

Provides centralized animation management with performance monitoring
and standardized animation behaviors throughout the application.
"""

from .manager import (
    SPQAnimationManager,
    AnimationType,
    AnimationPerformanceMonitor,
    get_animation_manager
)
from .button_animator import (
    ButtonAnimator,
    get_button_animator,
    animate_button,
    remove_button_animation
)
from .context_menu_animator import (
    ContextMenuAnimator,
    get_context_menu_animator,
    animate_context_menu,
    remove_context_menu_animation
)

__all__ = [
    'SPQAnimationManager',
    'AnimationType', 
    'AnimationPerformanceMonitor',
    'get_animation_manager',
    'ButtonAnimator',
    'get_button_animator', 
    'animate_button',
    'remove_button_animation',
    'ContextMenuAnimator',
    'get_context_menu_animator',
    'animate_context_menu',
    'remove_context_menu_animation'
]
