"""
Theme Cache Management

Provides caching mechanisms for theme resources to improve performance.
"""
from typing import Dict, Any, Optional
import time

from PySide6.QtGui import QColor, QIcon, QPixmap
from core.logging import get_logger

logger = get_logger(__name__)


class ThemeCache:
    """
    Caching system for theme resources.
    
    This class provides:
    - QSS stylesheet caching
    - Icon/image caching with size variants
    - Color caching
    - Performance metrics for theme operations
    """
    
    def __init__(self):
        """Initialize the cache."""
        # Stylesheet cache: theme_name -> stylesheet
        self._stylesheet_cache: Dict[str, str] = {}
        
        # Icon cache: (theme_name, icon_name, size) -> QIcon
        self._icon_cache: Dict[tuple, QIcon] = {}
        
        # Pixmap cache: (theme_name, image_name, width, height) -> QPixmap
        self._pixmap_cache: Dict[tuple, QPixmap] = {}
        
        # Color cache: (theme_name, color_role) -> QColor
        self._color_cache: Dict[tuple, QColor] = {}
        
        # Performance metrics
        self._metrics = {
            'theme_load_times': {},  # theme_name -> load_time_ms
            'cache_hits': 0,
            'cache_misses': 0,
            'last_theme_switch': 0.0  # timestamp
        }
    
    def get_stylesheet(self, theme_name: str) -> Optional[str]:
        """
        Get a cached stylesheet.
        
        Args:
            theme_name: Name of the theme
            
        Returns:
            Optional[str]: The cached stylesheet, or None if not cached
        """
        stylesheet = self._stylesheet_cache.get(theme_name)
        if stylesheet is not None:
            self._metrics['cache_hits'] += 1
        else:
            self._metrics['cache_misses'] += 1
        return stylesheet
    
    def set_stylesheet(self, theme_name: str, stylesheet: str) -> None:
        """
        Cache a stylesheet.
        
        Args:
            theme_name: Name of the theme
            stylesheet: The stylesheet to cache
        """
        self._stylesheet_cache[theme_name] = stylesheet
        logger.debug(f"Cached stylesheet for theme: {theme_name}")
    
    def get_icon(self, theme_name: str, icon_name: str, size: int) -> Optional[QIcon]:
        """
        Get a cached icon.
        
        Args:
            theme_name: Name of the theme
            icon_name: Name of the icon
            size: Size of the icon
            
        Returns:
            Optional[QIcon]: The cached icon, or None if not cached
        """
        cache_key = (theme_name, icon_name, size)
        icon = self._icon_cache.get(cache_key)
        if icon is not None:
            self._metrics['cache_hits'] += 1
        else:
            self._metrics['cache_misses'] += 1
        return icon
    
    def set_icon(self, theme_name: str, icon_name: str, size: int, icon: QIcon) -> None:
        """
        Cache an icon.
        
        Args:
            theme_name: Name of the theme
            icon_name: Name of the icon
            size: Size of the icon
            icon: The icon to cache
        """
        cache_key = (theme_name, icon_name, size)
        self._icon_cache[cache_key] = icon
    
    def get_pixmap(self, theme_name: str, image_name: str, width: int, height: int) -> Optional[QPixmap]:
        """
        Get a cached pixmap.
        
        Args:
            theme_name: Name of the theme
            image_name: Name of the image
            width: Width of the image
            height: Height of the image
            
        Returns:
            Optional[QPixmap]: The cached pixmap, or None if not cached
        """
        cache_key = (theme_name, image_name, width, height)
        pixmap = self._pixmap_cache.get(cache_key)
        if pixmap is not None:
            self._metrics['cache_hits'] += 1
        else:
            self._metrics['cache_misses'] += 1
        return pixmap
    
    def set_pixmap(self, theme_name: str, image_name: str, width: int, height: int, pixmap: QPixmap) -> None:
        """
        Cache a pixmap.
        
        Args:
            theme_name: Name of the theme
            image_name: Name of the image
            width: Width of the image
            height: Height of the image
            pixmap: The pixmap to cache
        """
        cache_key = (theme_name, image_name, width, height)
        self._pixmap_cache[cache_key] = pixmap
    
    def get_color(self, theme_name: str, color_role: str) -> Optional[QColor]:
        """
        Get a cached color.
        
        Args:
            theme_name: Name of the theme
            color_role: Color role
            
        Returns:
            Optional[QColor]: The cached color, or None if not cached
        """
        cache_key = (theme_name, color_role)
        color = self._color_cache.get(cache_key)
        if color is not None:
            self._metrics['cache_hits'] += 1
        else:
            self._metrics['cache_misses'] += 1
        return color
    
    def set_color(self, theme_name: str, color_role: str, color: QColor) -> None:
        """
        Cache a color.
        
        Args:
            theme_name: Name of the theme
            color_role: Color role
            color: The color to cache
        """
        cache_key = (theme_name, color_role)
        self._color_cache[cache_key] = color
    
    def record_theme_load(self, theme_name: str, load_time_ms: float) -> None:
        """
        Record a theme load time.
        
        Args:
            theme_name: Name of the theme
            load_time_ms: Load time in milliseconds
        """
        self._metrics['theme_load_times'][theme_name] = load_time_ms
        self._metrics['last_theme_switch'] = time.time()
    
    def clear(self, theme_name: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            theme_name: Optional theme name. If provided, only clears cache for that theme.
        """
        if theme_name is None:
            self._stylesheet_cache.clear()
            self._icon_cache.clear()
            self._pixmap_cache.clear()
            self._color_cache.clear()
            logger.debug("Cleared all theme caches")
        else:
            # Clear only entries for the specified theme
            self._stylesheet_cache.pop(theme_name, None)
            
            # Clear icon cache for theme
            keys_to_remove = [k for k in self._icon_cache if k[0] == theme_name]
            for key in keys_to_remove:
                self._icon_cache.pop(key, None)
                
            # Clear pixmap cache for theme
            keys_to_remove = [k for k in self._pixmap_cache if k[0] == theme_name]
            for key in keys_to_remove:
                self._pixmap_cache.pop(key, None)
                
            # Clear color cache for theme
            keys_to_remove = [k for k in self._color_cache if k[0] == theme_name]
            for key in keys_to_remove:
                self._color_cache.pop(key, None)
                
            logger.debug(f"Cleared cache for theme: {theme_name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dict[str, Any]: Cache performance metrics
        """
        total_accesses = self._metrics['cache_hits'] + self._metrics['cache_misses']
        hit_rate = 0.0
        if total_accesses > 0:
            hit_rate = self._metrics['cache_hits'] / total_accesses * 100.0
            
        return {
            'cache_hits': self._metrics['cache_hits'],
            'cache_misses': self._metrics['cache_misses'],
            'hit_rate_percent': hit_rate,
            'theme_load_times': self._metrics['theme_load_times'].copy(),
            'last_theme_switch': self._metrics['last_theme_switch']
        }
