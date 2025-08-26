"""
Centralized overlay-related constants to ensure consistency across canvas and backends.

Do not import UI modules here. Keep this file dependency-light.
"""

# Minimum logical size for overlays (in device-independent pixels)
OVERLAY_MIN_WIDTH: int = 200
OVERLAY_MIN_HEIGHT: int = 180

# Default content aspect ratio (width, height)
DEFAULT_ASPECT: tuple[int, int] = (16, 9)
