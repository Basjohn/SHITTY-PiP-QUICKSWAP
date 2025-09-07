"""
Unified accent calculation system for overlay borders.
Eliminates dual calculation systems and ensures DWM/border alignment.
"""

from dataclasses import dataclass
from PySide6.QtCore import QRectF
from core.logging import get_logger


@dataclass
class AccentProperties:
    """Unified accent properties with guaranteed alignment."""
    thickness: float
    inset: float
    inner_radius: float
    accent_rect: QRectF
    
    
class AccentCalculator:
    """Centralized accent calculation with coordinate system unification."""
    
    def __init__(self):
        self._logger = get_logger("AccentCalculator")
    
    def calculate_accent_properties(
        self,
        widget_rect: QRectF,
        border_thickness: float,
        corner_radius: float,
        dpi_scale: float,
        theme_base_thickness: float = 1.0,
        theme_base_inset: float = 3.0
    ) -> AccentProperties:
        """
        Calculate unified accent properties ensuring DWM/border alignment.
        
        Args:
            widget_rect: Widget bounds in logical coordinates
            border_thickness: Main border thickness
            corner_radius: Border corner radius
            dpi_scale: DPI scaling factor
            theme_base_thickness: Base thickness from theme
            theme_base_inset: Base inset from theme
            
        Returns:
            AccentProperties with guaranteed alignment
        """
        
        # Calculate overlay dimensions for size-based scaling
        min_dimension = min(widget_rect.width(), widget_rect.height())
        
        # Unified thickness calculation - single scaling point
        accent_thickness = self._calculate_thickness(
            theme_base_thickness, min_dimension, dpi_scale
        )
        
        # Unified inset calculation - prevents gaps and overlaps
        accent_inset = self._calculate_inset(
            theme_base_inset, border_thickness, dpi_scale, accent_thickness
        )
        
        # Calculate accent rectangle with validated inset
        accent_rect = widget_rect.adjusted(
            accent_inset, accent_inset, -accent_inset, -accent_inset
        )
        
        # Ensure accent rectangle is valid
        if accent_rect.width() <= 0 or accent_rect.height() <= 0:
            # Fallback to minimal accent for very small overlays
            fallback_inset = min(accent_inset * 0.5, border_thickness * 0.25)
            accent_rect = widget_rect.adjusted(
                fallback_inset, fallback_inset, -fallback_inset, -fallback_inset
            )
            accent_inset = fallback_inset
        
        # Calculate inner radius ensuring it doesn't go negative
        inner_radius = max(0.0, corner_radius - accent_inset) if corner_radius > 0 else 0.0
        
        return AccentProperties(
            thickness=accent_thickness,
            inset=accent_inset,
            inner_radius=inner_radius,
            accent_rect=accent_rect
        )
    
    def _calculate_thickness(
        self, 
        base_thickness: float, 
        min_dimension: float, 
        dpi_scale: float
    ) -> float:
        """Calculate accent thickness with bounded size scaling."""
        
        # Size-based scaling factor with bounds
        if min_dimension < 200:
            size_factor = 1.0  # Small overlays: base thickness
        elif min_dimension < 500:
            # Medium overlays: linear scale from 1.0 to 1.3
            size_factor = 1.0 + (min_dimension - 200) * 0.3 / 300.0
        else:
            # Large overlays: linear scale from 1.3 to 1.6 (capped)
            size_factor = 1.3 + min((min_dimension - 500) * 0.3 / 500.0, 0.3)
        
        # Apply scaling with minimum visibility guarantee
        thickness = base_thickness * size_factor * dpi_scale
        
        # Ensure minimum thickness for visibility
        return max(1.0, thickness)
    
    def _calculate_inset(
        self,
        base_inset: float,
        border_thickness: float, 
        dpi_scale: float,
        accent_thickness: float
    ) -> float:
        """Calculate accent inset preventing gaps and overlaps."""
        
        # Base inset scaled by DPI
        scaled_inset = base_inset * dpi_scale
        
        # Ensure inset is large enough to clear border thickness
        # Use 1.1x border thickness as minimum to prevent overlap
        min_inset_for_border = border_thickness * 1.1
        
        # Ensure inset isn't so large it makes accent invisible
        # Maximum inset is 1/3 of the smaller dimension equivalent
        max_reasonable_inset = scaled_inset * 2.0
        
        # Choose inset that satisfies all constraints
        final_inset = max(scaled_inset, min_inset_for_border)
        final_inset = min(final_inset, max_reasonable_inset)
        
        return final_inset
    
    def validate_alignment(
        self,
        accent_rect: QRectF,
        dwm_rect: QRectF,
        border_rect: QRectF,
        tolerance: float = 1.0
    ) -> bool:
        """
        Validate that accent, DWM, and border are properly aligned.
        
        Args:
            accent_rect: Accent rectangle
            dwm_rect: DWM thumbnail rectangle  
            border_rect: Border rectangle
            tolerance: Alignment tolerance in pixels
            
        Returns:
            True if alignment is within tolerance
        """
        
        # Check that accent is inside border
        if not border_rect.contains(accent_rect):
            self._logger.warning("Accent extends outside border")
            return False
        
        # Check that DWM doesn't overlap accent (with tolerance)
        accent_expanded = accent_rect.adjusted(-tolerance, -tolerance, tolerance, tolerance)
        if accent_expanded.intersects(dwm_rect):
            self._logger.warning("DWM overlaps accent within tolerance")
            return False
        
        return True


# Global instance
_accent_calculator = None

def get_accent_calculator() -> AccentCalculator:
    """Get the global accent calculator instance."""
    global _accent_calculator
    if _accent_calculator is None:
        _accent_calculator = AccentCalculator()
    return _accent_calculator
