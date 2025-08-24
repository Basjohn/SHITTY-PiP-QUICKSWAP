"""
Backend manager for overlay system.

This module provides functionality for managing and selecting appropriate
rendering backends for overlays.
"""
from core.logging import get_logger
from enum import Enum
from typing import Dict, List, Optional, Type

from .overlay import Overlay as OverlayBase
from .types import OverlayConfig, OverlayType
from .backends import BackendType, get_backend, register_backend

logger = get_logger(__name__)

class BackendPriority(Enum):
    """Priority levels for backend selection."""
    REQUIRED = 0
    PREFERRED = 1
    FALLBACK = 2

class BackendInfo:
    """Information about an available backend."""
    
    def __init__(self, 
                 backend_type: BackendType, 
                 priority: BackendPriority = BackendPriority.PREFERRED,
                 supported: bool = True,
                 reason: str = ""):
        """Initialize backend information.
        
        Args:
            backend_type: Type of the backend
            priority: Priority level for selection
            supported: Whether the backend is supported
            reason: Reason if not supported
        """
        self.backend_type = backend_type
        self.priority = priority
        self.supported = supported
        self.reason = reason
        self.backend_class: Optional[Type[OverlayBase]] = None
        
        if supported:
            self.backend_class = get_backend(backend_type)
            if not self.backend_class:
                self.supported = False
                self.reason = f"Backend {backend_type.name} not found"

class BackendManager:
    """Manages overlay backends and their selection."""
    
    def __init__(self):
        """Initialize the backend manager."""
        self._backends: Dict[BackendType, BackendInfo] = {
            backend_type: BackendInfo(backend_type, supported=False, reason="Not checked yet")
            for backend_type in BackendType
        }
        self._detect_available_backends()
    
    def _detect_available_backends(self) -> None:
        """Detect and initialize available backends."""
        # Check OpenGL backend
        try:
            # Correct module path: gl_backend.py provides OpenGLOverlay
            from .backends.gl_backend import OpenGLOverlay
            register_backend(BackendType.OPENGL, OpenGLOverlay)
            self._backends[BackendType.OPENGL] = BackendInfo(
                BackendType.OPENGL,
                BackendPriority.PREFERRED,
                True,
                ""
            )
        except ImportError as e:
            self._backends[BackendType.OPENGL] = BackendInfo(
                BackendType.OPENGL,
                BackendPriority.FALLBACK,
                False,
                f"OpenGL not available: {str(e)}"
            )
        
        # Check DWM backend
        try:
            from .backends.dwm.integrated_dwm_backend import IntegratedDWMOverlay as DWMOverlay
            register_backend(BackendType.DWM, DWMOverlay)
            self._backends[BackendType.DWM] = BackendInfo(
                BackendType.DWM,
                BackendPriority.PREFERRED,
                True,
                ""
            )
        except ImportError as e:
            self._backends[BackendType.DWM] = BackendInfo(
                BackendType.DWM,
                BackendPriority.FALLBACK,
                False,
                f"DWM not available: {str(e)}"
            )

        # Check Software (QWidget host) backend
        try:
            from .backends.software.backend import SoftwareOverlay
            register_backend(BackendType.SOFTWARE, SoftwareOverlay)
            self._backends[BackendType.SOFTWARE] = BackendInfo(
                BackendType.SOFTWARE,
                BackendPriority.PREFERRED,
                True,
                ""
            )
        except ImportError as e:
            self._backends[BackendType.SOFTWARE] = BackendInfo(
                BackendType.SOFTWARE,
                BackendPriority.FALLBACK,
                False,
                f"Software backend not available: {str(e)}"
            )
        
        # Check Monitor backend
        try:
            from .backends.monitor.monitor_backend import MonitorBackend
            register_backend(BackendType.MONITOR, MonitorBackend)
            self._backends[BackendType.MONITOR] = BackendInfo(
                BackendType.MONITOR,
                BackendPriority.PREFERRED,
                True,
                ""
            )
        except ImportError as e:
            self._backends[BackendType.MONITOR] = BackendInfo(
                BackendType.MONITOR,
                BackendPriority.FALLBACK,
                False,
                f"Monitor backend not available: {str(e)}"
            )
        
        # Log backend availability
        for backend_type, info in self._backends.items():
            status = "available" if info.supported else f"unavailable: {info.reason}"
            logger.debug("Backend %s is %s", backend_type.name, status)
    
    def get_available_backends(self) -> List[BackendType]:
        """Get a list of available backends.
        
        Returns:
            List of available backend types
        """
        return [
            backend_type 
            for backend_type, info in self._backends.items() 
            if info.supported
        ]
    
    def get_backend_info(self, backend_type: BackendType) -> BackendInfo:
        """Get information about a specific backend.
        
        Args:
            backend_type: Type of the backend
            
        Returns:
            BackendInfo object with details about the backend
            
        Raises:
            ValueError: If the backend type is not recognized
        """
        if backend_type not in self._backends:
            raise ValueError(f"Unknown backend type: {backend_type}")
        return self._backends[backend_type]
    
    def select_backend(self, 
                      preferred: BackendType = BackendType.AUTO,
                      overlay_type: OverlayType = OverlayType.WINDOW) -> Optional[Type[OverlayBase]]:
        """Select the most appropriate backend.
        
        Args:
            preferred: Preferred backend type
            overlay_type: Type of overlay being created
            
        Returns:
            The selected backend class, or None if no suitable backend is available
        """
        # If a specific backend is requested, try to use it
        if preferred != BackendType.AUTO:
            info = self._backends[preferred]
            if info.supported and info.backend_class:
                return info.backend_class
            # Fail-fast: do not silently substitute when an explicit backend is requested
            logger.error("Preferred backend %s is not available: %s",
                         preferred.name, info.reason)
            return None
        
        # Otherwise, select the best available backend
        available = [
            (info.priority.value, info.backend_type, info.backend_class)
            for info in self._backends.values()
            if info.supported and info.backend_class is not None
        ]
        
        if not available:
            logger.error("No suitable backends available")
            return None
        
        # Sort by priority (lower value = higher priority)
        available.sort(key=lambda x: (x[0], getattr(x[1], "name", str(x[1]))))
        
        # Return the highest priority backend
        return available[0][2]
    
    def create_overlay(self, 
                      config: OverlayConfig,
                      preferred_backend: BackendType = BackendType.AUTO) -> Optional[OverlayBase]:
        """Create a new overlay with the specified configuration.
        
        Args:
            config: Overlay configuration
            preferred_backend: Preferred backend to use
            
        Returns:
            The created overlay instance, or None if creation failed
        """
        # Select the appropriate backend
        backend_class = self.select_backend(preferred_backend, config.overlay_type)
        if not backend_class:
            logger.error("Failed to select a suitable backend for overlay")
            return None
        
        try:
            # Create the overlay; lifecycle (initialize/show/close) is managed by OverlayManager
            overlay = backend_class(config)
            return overlay
            
        except Exception as e:
            logger.exception("Error creating overlay: %s", str(e))
            return None
