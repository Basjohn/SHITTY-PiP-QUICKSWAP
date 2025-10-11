"""
Docking Configuration Module - Configuration settings for the docking overlay system.

This module defines configuration classes and settings for the 3-overlay docking system,
including positioning, sizing, and behavior parameters.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from dataclasses import dataclass
from PySide6.QtCore import QRect, QSize


@dataclass
class DockingConfig:
    """Configuration for the docking overlay system."""
    
    # Size ratios for secondary overlays
    secondary_overlay_ratio_1: float = 0.7  # 70% of main overlay
    secondary_overlay_ratio_2: float = 0.5  # 50% of main overlay
    
    # Minimum sizes to enforce hierarchy
    min_width_base: int = 120  # Much smaller base for better hierarchy
    min_height_base: int = 80   # Much smaller base for better hierarchy
    min_size_increment: int = 30  # Smaller increment for tighter hierarchy
    
    # Maximum size limits to prevent convergence
    max_size_ratio_base: float = 0.75  # 75% for first secondary
    max_size_ratio_decrement: float = 0.1  # Decrease by 10% per level
    
    # Positioning settings
    flush_positioning: bool = True  # Zero gaps between overlays
    vertical_stacking: bool = True  # Stack secondary overlays vertically
    
    # Synchronization settings
    sync_coalesce_delay_ms: int = 10  # Delay for coalescing sync events
    enable_event_filtering: bool = True  # Use event filters for synchronization
    
    # Interaction settings
    allow_individual_resize: bool = False  # Disable individual overlay resize
    allow_individual_scroll: bool = False  # Disable individual overlay scroll
    
    # Debug and logging
    debug_sync_logging: bool = True   # Enable detailed sync logging
    debug_positioning: bool = True    # Enable positioning debug logs
    
    @classmethod
    def default(cls) -> 'DockingConfig':
        """Create a default docking configuration."""
        return cls()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DockingConfig':
        """Create configuration from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'secondary_overlay_ratio_1': self.secondary_overlay_ratio_1,
            'secondary_overlay_ratio_2': self.secondary_overlay_ratio_2,
            'min_width_base': self.min_width_base,
            'min_height_base': self.min_height_base,
            'min_size_increment': self.min_size_increment,
            'max_size_ratio_base': self.max_size_ratio_base,
            'max_size_ratio_decrement': self.max_size_ratio_decrement,
            'flush_positioning': self.flush_positioning,
            'vertical_stacking': self.vertical_stacking,
            'sync_coalesce_delay_ms': self.sync_coalesce_delay_ms,
            'enable_event_filtering': self.enable_event_filtering,
            'allow_individual_resize': self.allow_individual_resize,
            'allow_individual_scroll': self.allow_individual_scroll,
            'debug_sync_logging': self.debug_sync_logging,
            'debug_positioning': self.debug_positioning,
        }
    
    def get_secondary_ratio(self, index: int) -> float:
        """Get the size ratio for a secondary overlay by index."""
        if index == 0:
            return self.secondary_overlay_ratio_1
        elif index == 1:
            return self.secondary_overlay_ratio_2
        else:
            # For additional overlays, continue decreasing by 20%
            return max(0.3, self.secondary_overlay_ratio_2 - (index - 1) * 0.2)
    
    def get_min_size(self, index: int) -> QSize:
        """Get minimum size for an overlay by index (0 = main, 1+ = secondary)."""
        if index == 0:
            return QSize(300, 200)  # Main overlay minimum
        else:
            # Secondary overlays should have SMALLER minimums, not larger
            # Decrease minimum size for each secondary overlay to maintain hierarchy
            width = max(80, self.min_width_base - (index * self.min_size_increment))
            height = max(60, self.min_height_base - (index * (self.min_size_increment // 2)))
            return QSize(width, height)
    
    def get_max_size_ratio(self, index: int) -> float:
        """Get maximum size ratio for a secondary overlay by index."""
        if index == 0:
            return self.max_size_ratio_base
        else:
            return max(0.4, self.max_size_ratio_base - (index * self.max_size_ratio_decrement))
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.secondary_overlay_ratio_1 <= 0 or self.secondary_overlay_ratio_1 >= 1:
            return False
        if self.secondary_overlay_ratio_2 <= 0 or self.secondary_overlay_ratio_2 >= 1:
            return False
        if self.secondary_overlay_ratio_2 >= self.secondary_overlay_ratio_1:
            return False
        if self.min_width_base <= 0 or self.min_height_base <= 0:
            return False
        if self.sync_coalesce_delay_ms < 0:
            return False
        return True


@dataclass
class DockingPosition:
    """Represents a saved docking position configuration."""
    
    main_rect: QRect
    secondary_rects: list[QRect]
    opacity: float = 1.0
    monitor_index: int = 0
    timestamp: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DockingPosition':
        """Create position from dictionary."""
        main_rect = QRect(
            data['main_rect']['x'],
            data['main_rect']['y'], 
            data['main_rect']['width'],
            data['main_rect']['height']
        )
        
        secondary_rects = []
        for rect_data in data.get('secondary_rects', []):
            secondary_rects.append(QRect(
                rect_data['x'],
                rect_data['y'],
                rect_data['width'], 
                rect_data['height']
            ))
        
        return cls(
            main_rect=main_rect,
            secondary_rects=secondary_rects,
            opacity=data.get('opacity', 1.0),
            monitor_index=data.get('monitor_index', 0),
            timestamp=data.get('timestamp')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'main_rect': {
                'x': self.main_rect.x(),
                'y': self.main_rect.y(),
                'width': self.main_rect.width(),
                'height': self.main_rect.height()
            },
            'secondary_rects': [
                {
                    'x': rect.x(),
                    'y': rect.y(),
                    'width': rect.width(),
                    'height': rect.height()
                }
                for rect in self.secondary_rects
            ],
            'opacity': self.opacity,
            'monitor_index': self.monitor_index,
            'timestamp': self.timestamp
        }
