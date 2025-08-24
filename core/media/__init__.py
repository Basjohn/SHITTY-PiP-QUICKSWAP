"""Media control package for SPQModular.

Provides centralized media key routing, application control, and keepalive monitoring.
"""

from .media_controller import MediaController, get_media_controller
from .keepalive import MediaPlayerKeepAlive, get_media_keepalive

__all__ = ["MediaController", "get_media_controller", "MediaPlayerKeepAlive", "get_media_keepalive"]
