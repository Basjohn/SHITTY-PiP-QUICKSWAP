"""
Type definitions and enums for the overlay system.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from PySide6.QtCore import QPoint, QSize


class OverlayType(Enum):
    """Enumeration of supported overlay types."""
    WINDOW = auto()      # Regular window overlay
    MONITOR = auto()     # Full-screen overlay on a monitor
    BORDER = auto()      # Border around a window
    DOCKING = auto()     # Three-overlay docking mode system
    CUSTOM = auto()      # Custom overlay with custom behavior


class OverlayState(Enum):
    """Possible states of an overlay."""
    CREATED = auto()     # Overlay created but not initialized
    INITIALIZING = auto()  # Initialization in progress
    READY = auto()       # Initialized and ready to use
    SHOWING = auto()     # Currently visible
    HIDDEN = auto()      # Initialized but hidden
    ERROR = auto()       # Error state
    DESTROYING = auto()  # Being destroyed


@dataclass
class OverlayConfig:
    """Configuration for an overlay instance."""
    # Basic properties
    overlay_type: OverlayType = OverlayType.WINDOW
    position: QPoint = field(default_factory=QPoint)
    size: QSize = field(default_factory=lambda: QSize(800, 600))
    opacity: float = 1.0
    visible: bool = True
    title: str = "Overlay"
    parent_hwnd: Optional[int] = None
    z_order: int = 0
    flags: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    vsync: bool = True
    fps_limit: int = 60
    
    # Behavior settings
    always_on_top: bool = True
    
    # Custom properties
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DockingConfig:
    """Configuration specific to docking mode overlays."""
    main_overlay_id: str = ""
    secondary_overlay_ids: list[str] = field(default_factory=list)
    size_ratios: list[float] = field(default_factory=lambda: [1.0, 0.7, 0.5])  # 100%, 70%, 50%
    positioning_mode: str = "auto"  # auto, manual
    spacing: int = 2  # pixels between overlays
    mru_capacity: int = 12  # expanded MRU capacity for 3 overlays


@dataclass
class RenderStats:
    """Rendering statistics for an overlay."""
    frame_count: int = 0
    fps: float = 0.0
    last_frame_time: float = 0.0
    average_frame_time: float = 0.0
    min_frame_time: float = float('inf')
    max_frame_time: float = 0.0
    
    def update(self, frame_time: float) -> None:
        """Update statistics with a new frame time."""
        self.frame_count += 1
        self.last_frame_time = frame_time
        
        # Update min/max
        self.min_frame_time = min(self.min_frame_time, frame_time)
        self.max_frame_time = max(self.max_frame_time, frame_time)
        
        # Update average (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if self.average_frame_time == 0:
            self.average_frame_time = frame_time
        else:
            self.average_frame_time = (alpha * frame_time + 
                                     (1 - alpha) * self.average_frame_time)
        
        # Update FPS if we have enough data
        if self.average_frame_time > 0:
            self.fps = 1.0 / self.average_frame_time


class RenderLayer(Enum):
    """Rendering layers for overlay composition."""
    BACKGROUND = 0    # Background elements (lowest)
    CONTENT = 1       # Main content
    OVERLAY = 2       # UI elements on top of content
    DRAG_HANDLE = 3   # Drag handles and resize controls
    DEBUG = 4         # Debug information (highest)
