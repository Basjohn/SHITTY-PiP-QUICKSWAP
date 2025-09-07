from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
from PySide6.QtCore import QSize, QRectF


@dataclass
class BorderMetrics:
    """Complete border metrics for rendering."""
    thickness: float
    corner_radius: float
    inner_accent_thickness: float
    render_rect: QRectF
    accent_inset: float
    
    def is_valid(self) -> bool:
        """Check if metrics are valid for rendering."""
        return (
            self.thickness > 0 and
            self.render_rect.isValid() and
            self.render_rect.width() > 0 and
            self.render_rect.height() > 0
        )
    
    def __eq__(self, other) -> bool:
        """Compare metrics for change detection."""
        if not isinstance(other, BorderMetrics):
            return False
        return (
            abs(self.thickness - other.thickness) < 0.01 and
            abs(self.corner_radius - other.corner_radius) < 0.01 and
            abs(self.inner_accent_thickness - other.inner_accent_thickness) < 0.01 and
            self.render_rect == other.render_rect and
            abs(self.accent_inset - other.accent_inset) < 0.01
        )


class BorderGeometry:
    """Manages all border geometry calculations and DPI scaling."""
    
    def __init__(self):
        self._metrics_cache: Dict[Tuple[int, int, float, bool, float], BorderMetrics] = {}
        
    def calculate_border_metrics(self, size: QSize, dpi_scale: float, 
                               rounded_enabled: bool = False,
                               thickness_base: float = 2.0) -> BorderMetrics:
        """Calculate optimal border metrics for given size and DPI."""
        # Create cache key
        cache_key = (size.width(), size.height(), dpi_scale, rounded_enabled, thickness_base)
        if cache_key in self._metrics_cache:
            return self._metrics_cache[cache_key]
            
        min_dim = min(size.width(), size.height())
        
        # Special handling for very small overlays to prevent artifacts
        if min_dim < 20:
            # For tiny overlays, use simplified metrics
            # Fixed 1px border, no rounded corners, no accent
            metrics = BorderMetrics(
                thickness=1.0,  # Fixed 1px for tiny overlays
                corner_radius=0.0,  # No rounded corners for tiny overlays
                inner_accent_thickness=0.0,  # No accent for tiny overlays
                render_rect=QRectF(0, 0, size.width(), size.height()),
                accent_inset=0.0
            )
            self._metrics_cache[cache_key] = metrics
            return metrics
        
        # Adaptive thickness: thinner for small overlays, thicker for large
        # Scale factor based on overlay size relative to a 400px baseline
        thickness_factor = max(0.8, min(1.5, min_dim / 400.0))
        
        # Progressive scaling - more aggressive thickness reduction for smaller sizes
        if min_dim < 100:
            # Additional scaling for small overlays (20-100px)
            small_factor = max(0.5, min_dim / 100.0)
            thickness_factor *= small_factor
        
        thickness = thickness_base * thickness_factor * dpi_scale
        
        # Ensure minimum thickness for visibility but cap for small sizes
        thickness = max(1.0, min(min_dim * 0.1, thickness))  # Cap at 10% of min dimension
        
        # Corner radius: only when explicitly enabled, scaled by size
        corner_radius = 0.0
        if rounded_enabled:
            # Scale corner radius based on size
            base_radius = 4.0  # Base logical radius
            # For small overlays, reduce radius proportionally
            if min_dim < 100:
                base_radius *= (min_dim / 100.0)
            corner_radius = base_radius * dpi_scale
            
        # Inner accent calculation removed - now handled by unified AccentCalculator
        inner_accent_thickness = 0.0
        accent_inset = 0.0
        
        # Calculate render rectangle (widget bounds)
        widget_rect = QRectF(0, 0, size.width(), size.height())
        
        metrics = BorderMetrics(
            thickness=thickness,
            corner_radius=corner_radius,
            inner_accent_thickness=inner_accent_thickness,
            render_rect=widget_rect,
            accent_inset=accent_inset
        )
        
        # Cache the result
        self._metrics_cache[cache_key] = metrics
        return metrics
        
    def get_render_rect(self, widget_rect: QRectF, thickness: float) -> QRectF:
        """Get pixel-perfect rendering rectangle with proper insets."""
        inset = thickness / 2.0
        return widget_rect.adjusted(inset, inset, -inset, -inset)
        
    def clear_cache(self) -> None:
        """Clear geometry cache when settings change."""
        self._metrics_cache.clear()
        
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for debugging."""
        return {
            'cached_entries': len(self._metrics_cache),
            'memory_usage_estimate': len(self._metrics_cache) * 200  # Rough estimate in bytes
        }
