"""
Application core functionality.

This package contains core application components related to instance management,
startup/shutdown procedures, and other application-level concerns.
"""

from .instance import ApplicationInstanceManager
from .core import ApplicationCore, get_app_core

__all__ = [
    'ApplicationInstanceManager',
    'ApplicationCore',
    'get_app_core',
]
