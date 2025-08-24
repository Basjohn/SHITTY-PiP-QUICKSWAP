"""
Base renderer interface and common functionality.

This module defines the base renderer interface that all rendering backends
must implement, along with common functionality for managing rendering resources.
"""
from __future__ import annotations

from core.logging import get_logger
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import QSize, QPoint, QRect

logger = get_logger()

class RendererBase(ABC):
    """Base class for all renderer implementations."""
    
    def __init__(self):
        """Initialize the renderer."""
        self._is_initialized = False
        self._logger = get_logger()
    
    @property
    def is_initialized(self) -> bool:
        """Check if the renderer has been initialized."""
        return self._is_initialized
    
    def initialize(self) -> bool:
        """Initialize the renderer.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._is_initialized:
            self._logger.warning("Renderer already initialized")
            return True
            
        try:
            self._initialize_impl()
            self._is_initialized = True
            self._logger.debug("Renderer initialized successfully")
            return True
        except Exception as e:
            self._logger.exception("Failed to initialize renderer: %s", str(e))
            return False
    
    def begin_frame(self) -> bool:
        """Begin a new frame.
        
        Returns:
            bool: True if the frame was started successfully, False otherwise
        """
        if not self._is_initialized:
            self._logger.error("Cannot begin frame: Renderer not initialized")
            return False
            
        return self._begin_frame_impl()
    
    def end_frame(self) -> None:
        """End the current frame."""
        if not self._is_initialized:
            return
            
        self._end_frame_impl()
    
    def resize(self, size: QSize) -> None:
        """Resize the render target.
        
        Args:
            size: The new size of the render target
        """
        if not self._is_initialized:
            self._logger.error("Cannot resize: Renderer not initialized")
            return
            
        self._resize_impl(size)
    
    def cleanup(self) -> None:
        """Clean up renderer resources."""
        if not self._is_initialized:
            return
            
        try:
            self._cleanup_impl()
            self._is_initialized = False
            self._logger.debug("Renderer cleaned up")
        except Exception as e:
            self._logger.exception("Error during renderer cleanup: %s", str(e))
    
    @abstractmethod
    def _initialize_impl(self) -> None:
        """Implementation-specific initialization."""
        pass
    
    @abstractmethod
    def _begin_frame_impl(self) -> bool:
        """Implementation-specific frame begin."""
        pass
    
    @abstractmethod
    def _end_frame_impl(self) -> None:
        """Implementation-specific frame end."""
        pass
    
    @abstractmethod
    def _resize_impl(self, size: QSize) -> None:
        """Implementation-specific resize.
        
        Args:
            size: The new size of the render target
        """
        pass
    
    @abstractmethod
    def _cleanup_impl(self) -> None:
        """Implementation-specific cleanup."""
        pass

class RenderContext:
    """Context for rendering operations."""
    
    def __init__(self, renderer: RendererBase):
        """Initialize the render context.
        
        Args:
            renderer: The renderer to use for this context
        """
        self.renderer = renderer
        self.viewport = QRect()
        self.projection_matrix = None
        self.view_matrix = None
        self._logger = get_logger()
    
    def begin_frame(self) -> bool:
        """Begin a new frame.
        
        Returns:
            bool: True if the frame was started successfully, False otherwise
        """
        return self.renderer.begin_frame()
    
    def end_frame(self) -> None:
        """End the current frame."""
        self.renderer.end_frame()
    
    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        """Set the viewport for rendering.
        
        Args:
            x: X coordinate of the viewport
            y: Y coordinate of the viewport
            width: Width of the viewport
            height: Height of the viewport
        """
        self.viewport = QRect(x, y, width, height)
    
    def clear(self, color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """Clear the render target.
        
        Args:
            color: The clear color (RGBA, 0.0-1.0)
        """
        self._clear_impl(color)
    
    @abstractmethod
    def _clear_impl(self, color: Tuple[float, float, float, float]) -> None:
        """Implementation-specific clear.
        
        Args:
            color: The clear color (RGBA, 0.0-1.0)
        """
        pass
