"""
Rendering interface and implementations for overlays.

This module provides the base renderer interface and common implementations
for rendering overlay content.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QTransform
)

from .types import RenderLayer, RenderStats


class RenderContext:
    """Context for rendering operations.
    
    This class provides a consistent interface for rendering operations
    and manages the rendering state.
    """
    
    def __init__(self, painter: QPainter, size: QSize):
        """Initialize the render context.
        
        Args:
            painter: The QPainter to use for drawing
            size: The size of the render target
        """
        self.painter = painter
        self.size = size
        self._transform_stack = []
        self._opacity_stack = [1.0]
        self._clip_stack = []
        self._layer = RenderLayer.CONTENT
        self._stats = RenderStats()
        self._start_time = time.time()
    
    @property
    def width(self) -> int:
        """Get the width of the render target."""
        return self.size.width()
    
    @property
    def height(self) -> int:
        """Get the height of the render target."""
        return self.size.height()
    
    @property
    def center(self) -> QPoint:
        """Get the center point of the render target."""
        return QPoint(self.width // 2, self.height // 2)
    
    @property
    def current_opacity(self) -> float:
        """Get the current opacity value."""
        return self._opacity_stack[-1]
    
    @property
    def current_layer(self) -> RenderLayer:
        """Get the current render layer."""
        return self._layer
    
    @current_layer.setter
    def current_layer(self, layer: RenderLayer) -> None:
        """Set the current render layer."""
        self._layer = layer
    
    def push_opacity(self, opacity: float) -> None:
        """Push a new opacity value onto the stack.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        opacity = max(0.0, min(1.0, opacity))
        effective_opacity = self.current_opacity * opacity
        self._opacity_stack.append(effective_opacity)
        self.painter.setOpacity(effective_opacity)
    
    def pop_opacity(self) -> float:
        """Pop the current opacity from the stack.
        
        Returns:
            The previous opacity value
        """
        if len(self._opacity_stack) > 1:
            self._opacity_stack.pop()
        
        current = self.current_opacity
        self.painter.setOpacity(current)
        return current
    
    def push_transform(self, transform: QTransform) -> None:
        """Push a transformation onto the stack.
        
        Args:
            transform: The transformation to apply
        """
        self._transform_stack.append(self.painter.transform())
        self.painter.setTransform(transform, combine=True)
    
    def pop_transform(self) -> QTransform:
        """Pop the current transformation from the stack.
        
        Returns:
            The previous transformation
        """
        if self._transform_stack:
            old_transform = self._transform_stack.pop()
            self.painter.setTransform(old_transform)
            return old_transform
        return QTransform()
    
    def push_clip_rect(self, rect: QRect) -> None:
        """Push a clipping rectangle onto the stack.
        
        Args:
            rect: The rectangle to clip to
        """
        self._clip_stack.append(self.painter.clipPath())
        self.painter.setClipRect(rect, Qt.IntersectClip)
    
    def pop_clip_rect(self) -> None:
        """Pop the current clipping rectangle from the stack."""
        if self._clip_stack:
            clip_path = self._clip_stack.pop()
            self.painter.setClipPath(clip_path)
    
    def reset_transform(self) -> None:
        """Reset the transformation to the identity matrix."""
        self.painter.resetTransform()
        self._transform_stack.clear()
    
    def begin_layer(self, layer: RenderLayer) -> None:
        """Begin rendering to a specific layer.
        
        Args:
            layer: The layer to render to
        """
        self._layer = layer
    
    def end_frame(self) -> None:
        """End the current frame and update statistics."""
        frame_time = time.time() - self._start_time
        self._stats.update(frame_time)
        self._start_time = time.time()
    
    def draw_rect(self, rect: QRect, color: QColor, border_width: int = 0, 
                 border_color: QColor = None) -> None:
        """Draw a rectangle.
        
        Args:
            rect: The rectangle to draw
            color: Fill color
            border_width: Width of the border (0 for no border)
            border_color: Color of the border
        """
        if color.alpha() > 0:
            self.painter.fillRect(rect, color)
        
        if border_width > 0 and border_color and border_color.alpha() > 0:
            pen = self.painter.pen()
            old_pen = pen
            pen.setColor(border_color)
            pen.setWidth(border_width)
            self.painter.setPen(pen)
            self.painter.drawRect(rect)
            self.painter.setPen(old_pen)
    
    def draw_rounded_rect(self, rect: QRect, radius: int, color: QColor, 
                         border_width: int = 0, border_color: QColor = None) -> None:
        """Draw a rounded rectangle.
        
        Args:
            rect: The rectangle to draw
            radius: Corner radius
            color: Fill color
            border_width: Width of the border (0 for no border)
            border_color: Color of the border
        """
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        
        if color.alpha() > 0:
            self.painter.fillPath(path, color)
        
        if border_width > 0 and border_color and border_color.alpha() > 0:
            pen = self.painter.pen()
            old_pen = pen
            pen.setColor(border_color)
            pen.setWidth(border_width)
            self.painter.setPen(pen)
            self.painter.drawPath(path)
            self.painter.setPen(old_pen)
    
    def draw_text(self, text: str, position: QPoint, color: QColor, 
                 font: QFont = None, flags: int = Qt.AlignLeft | Qt.AlignTop) -> None:
        """Draw text.
        
        Args:
            text: The text to draw
            position: Position to draw at
            color: Text color
            font: Font to use (None for default)
            flags: Text alignment and formatting flags
        """
        if font:
            old_font = self.painter.font()
            self.painter.setFont(font)
        
        if color.alpha() > 0:
            self.painter.setPen(color)
            self.painter.drawText(position, text)
        
        if font:
            self.painter.setFont(old_font)
    
    def measure_text(self, text: str, font: QFont = None) -> QSize:
        """Measure the size of text.
        
        Args:
            text: The text to measure
            font: Font to use (None for default)
            
        Returns:
            The size of the text in pixels
        """
        if font:
            old_font = self.painter.font()
            self.painter.setFont(font)
        
        metrics = self.painter.fontMetrics()
        rect = metrics.boundingRect(text)
        
        if font:
            self.painter.setFont(old_font)
        
        return rect.size()


class OverlayRenderer(ABC):
    """Base class for overlay renderers.
    
    This class defines the interface that all overlay renderers must implement.
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the renderer.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def begin_frame(self) -> RenderContext:
        """Begin a new frame.
        
        Returns:
            RenderContext: The render context for this frame
        """
        pass
    
    @abstractmethod
    def end_frame(self, context: RenderContext) -> None:
        """End the current frame.
        
        Args:
            context: The render context for this frame
        """
        pass
    
    @abstractmethod
    def resize(self, size: QSize) -> None:
        """Resize the render target.
        
        Args:
            size: New size of the render target
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the renderer."""
        pass
