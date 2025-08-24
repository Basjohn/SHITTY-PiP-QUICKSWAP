"""
Base overlay implementation.

This module provides the base Overlay class that all specific overlay
implementations should inherit from.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Dict
from uuid import uuid4

from PySide6.QtCore import QObject, QPoint, QRect, QSize, Signal

from core.logging import get_logger
from .types import (
    OverlayConfig,
    OverlayState,
    RenderStats,
)


 


class OverlaySignals(QObject):
    """Signals for the Overlay class."""
    # Lifecycle signals
    initialized = Signal()
    shown = Signal()
    hidden = Signal()
    destroyed = Signal()
    
    # State change signals
    state_changed = Signal(OverlayState, OverlayState)  # old_state, new_state
    error_occurred = Signal(Exception)
    
    # Geometry signals
    moved = Signal(QPoint)  # new_position
    resized = Signal(QSize)  # new_size
    geometry_changed = Signal(QRect)  # new_geometry
    
    # Input signals
    mouse_pressed = Signal(QPoint, int)  # position, button
    mouse_released = Signal(QPoint, int)  # position, button
    mouse_moved = Signal(QPoint)  # position
    mouse_double_clicked = Signal(QPoint, int)  # position, button
    wheel_event = Signal(int)  # delta
    key_pressed = Signal(int)  # key_code
    key_released = Signal(int)  # key_code


class Overlay(ABC):
    """Base class for all overlay implementations.
    
    This class provides common functionality for all overlays, including
    lifecycle management, state tracking, and basic properties.
    """
    
    def __init__(self, config: OverlayConfig):
        """Initialize the overlay with the given configuration.
        
        Args:
            config: Configuration for the overlay
        """
        self._id = str(uuid4())
        self._config = config
        self._state = OverlayState.CREATED
        self._signals = OverlaySignals()
        self._logger = get_logger(f"Overlay[{self._id[:8]}]")
        self._stats = RenderStats()
        self._last_render_time = 0.0
        self._initialized = False
        self._visible = False
    
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def id(self) -> str:
        """Get the unique identifier for this overlay."""
        return self._id
    
    @property
    def config(self) -> OverlayConfig:
        """Get the overlay configuration."""
        return self._config
    
    @property
    def state(self) -> OverlayState:
        """Get the current state of the overlay."""
        return self._state
    
    @property
    def is_initialized(self) -> bool:
        """Check if the overlay has been initialized."""
        return self._initialized
    
    @property
    def is_visible(self) -> bool:
        """Check if the overlay is currently visible."""
        return self._visible
    
    @property
    def signals(self) -> OverlaySignals:
        """Get the signals for this overlay."""
        return self._signals
    
    @property
    def stats(self) -> RenderStats:
        """Get rendering statistics for this overlay."""
        return self._stats
    
    # Public methods
    # -------------------------------------------------------------------------
    
    def initialize(self) -> bool:
        """Initialize the overlay resources.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._initialized:
            self._logger.warning("Overlay already initialized")
            return True
            
        self._set_state(OverlayState.INITIALIZING)
        
        try:
            self._initialize_impl()
            self._initialized = True
            self._set_state(OverlayState.READY)
            self._signals.initialized.emit()
            self._logger.info("Overlay initialized successfully")
            return True
        except Exception as e:
            self._set_state(OverlayState.ERROR)
            self._signals.error_occurred.emit(e)
            self._logger.exception("Failed to initialize overlay")
            return False
    
    def show(self) -> None:
        """Show the overlay."""
        if not self._initialized and not self.initialize():
            self._logger.error("Cannot show uninitialized overlay")
            return
            
        if self._visible:
            return
            
        try:
            self._show_impl()
            self._visible = True
            self._signals.shown.emit()
            self._logger.debug("Overlay shown")
        except Exception as e:
            self._set_state(OverlayState.ERROR)
            self._signals.error_occurred.emit(e)
            self._logger.exception("Failed to show overlay")
    
    def hide(self) -> None:
        """Hide the overlay."""
        if not self._visible:
            return
            
        try:
            self._hide_impl()
            self._visible = False
            self._signals.hidden.emit()
            self._logger.debug("Overlay hidden")
        except Exception as e:
            self._set_state(OverlayState.ERROR)
            self._signals.error_occurred.emit(e)
            self._logger.exception("Failed to hide overlay")
    
    def close(self) -> None:
        """Close and clean up the overlay."""
        if self._state == OverlayState.DESTROYING:
            return
            
        self._set_state(OverlayState.DESTROYING)
        self.hide()
        
        try:
            self._close_impl()
            self._initialized = False
            self._signals.destroyed.emit()
            self._logger.info("Overlay closed")
        except Exception as e:
            self._set_state(OverlayState.ERROR)
            self._signals.error_occurred.emit(e)
            self._logger.exception("Error while closing overlay")
    
    def render(self) -> None:
        """Render a frame of the overlay."""
        if not self._initialized or not self._visible:
            return
            
        try:
            start_time = time.time()
            self._render_impl()
            
            # Update render statistics
            frame_time = time.time() - start_time
            self._stats.update(frame_time)
            self._last_render_time = start_time
            
        except Exception as e:
            self._set_state(OverlayState.ERROR)
            self._signals.error_occurred.emit(e)
            self._logger.exception("Error while rendering overlay")

    # --- Source/geometry/opacity helpers ---------------------------------
    def get_config(self) -> OverlayConfig:
        """Return the current overlay configuration.

        Centralized accessor used by `OverlayManager` when evaluating reuse.
        """
        return self._config

    def update_source(self, *_args: Any, **_kwargs: Any) -> bool:  # pragma: no cover
        """Update the overlay's content/source handle.

        Base implementation is a safe no-op and returns False. Backends that
        support dynamic source switching (e.g., DWM window overlays) should
        override this to perform the swap and return True on acceptance.
        """
        self._logger.debug("update_source() not supported by this backend")
        return False

    def set_position(self, pos: QPoint) -> None:
        """Set the overlay's position via config and notify implementation."""
        try:
            self.update_config(position=pos)
        except Exception as e:  # pragma: no cover
            self._logger.exception("Failed to set position: %s", e)

    def set_size(self, size: QSize) -> None:
        """Set the overlay's size via config and notify implementation."""
        try:
            self.update_config(size=size)
        except Exception as e:  # pragma: no cover
            self._logger.exception("Failed to set size: %s", e)

    def set_geometry(self, rect: QRect) -> None:
        """Set the overlay's geometry via config and notify implementation."""
        try:
            self.update_config(position=rect.topLeft(), size=rect.size())
        except Exception as e:  # pragma: no cover
            self._logger.exception("Failed to set geometry: %s", e)

    def set_opacity(self, opacity: float) -> bool:
        """Set overlay opacity via config. Backends may override for custom behavior."""
        try:
            self.update_config(opacity=float(opacity))
            return True
        except Exception as e:  # pragma: no cover
            self._logger.exception("Failed to set opacity: %s", e)
            return False

    # --- Title management -------------------------------------------------
    def set_title(self, title: str) -> None:
        """Set the overlay's title and allow backends to reflect it.

        This updates the OverlayConfig.title and calls an implementation hook
        so backends with a native host window can apply the title.
        """
        try:
            old = getattr(self._config, "title", None)
            if old == title:
                return
            # Update config
            if hasattr(self._config, "title"):
                setattr(self._config, "title", title)
            # Notify backend (optional override)
            if hasattr(self, "_set_title_impl"):
                try:
                    # type: ignore[attr-defined]
                    self._set_title_impl(title)  # noqa: SLF001
                except Exception as e:  # pragma: no cover
                    # Backend implementations must fail fast as needed; base just logs
                    self._logger.error("Failed to apply title in backend: %s", e)
        except Exception as e:
            # Never crash caller on title updates
            self._logger.exception("Error setting overlay title: %s", e)

    # Optional implementation hook (backends may override)
    def _set_title_impl(self, title: str) -> None:  # pragma: no cover
        """Backend-specific hook to apply the title (override in subclass)."""
        pass
    
    def update_config(self, **kwargs) -> None:
        """Update the overlay configuration.
        
        Args:
            **kwargs: Configuration values to update
        """
        old_config = self._snapshot_config()
        
        # Update the config with new values
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        # Notify implementation of config changes
        self._config_updated(old_config, self._snapshot_config())

    def _snapshot_config(self) -> Dict[str, Any]:
        """Create a safe, JSON-like snapshot of the config avoiding Qt deep-copies.

        Converts common Qt types to primitives and handles non-serializable objects
        (e.g., QScreen) by storing a minimal identifier when available.
        """
        try:
            cfg = self._config
            result: Dict[str, Any] = {}

            def q_to_primitive(val: Any) -> Any:
                # Fast path for primitives
                if val is None or isinstance(val, (bool, int, float, str)):
                    return val
                # QPoint, QSize, QRect
                try:
                    from PySide6.QtCore import QPoint, QSize, QRect
                    if isinstance(val, QPoint):
                        return {"type": "QPoint", "x": val.x(), "y": val.y()}
                    if isinstance(val, QSize):
                        return {"type": "QSize", "w": val.width(), "h": val.height()}
                    if isinstance(val, QRect):
                        return {
                            "type": "QRect",
                            "x": val.x(),
                            "y": val.y(),
                            "w": val.width(),
                            "h": val.height(),
                        }
                except Exception:
                    pass
                # Mapping and sequences (shallow sanitize)
                if isinstance(val, dict):
                    return {str(k): q_to_primitive(v) for k, v in val.items()}
                if isinstance(val, (list, tuple)):
                    return [q_to_primitive(v) for v in val]
                # QScreen and similar: prefer name/geometry if present
                try:
                    name_attr = getattr(val, "name", None)
                    if callable(name_attr):
                        name = name_attr()
                    else:
                        name = None
                except Exception:
                    name = None
                try:
                    geom_attr = getattr(val, "geometry", None)
                    if callable(geom_attr):
                        g = geom_attr()
                        geom = {
                            "x": g.x(),
                            "y": g.y(),
                            "w": g.width(),
                            "h": g.height(),
                        }
                    else:
                        geom = None
                except Exception:
                    geom = None
                if name is not None or geom is not None:
                    return {"type": type(val).__name__, "name": name, "geometry": geom}
                # Fallback: type name only
                try:
                    return {"type": type(val).__name__}
                except Exception:
                    return "<unserializable>"

            # Manually enumerate dataclass fields to avoid asdict/deepcopy on Qt objects
            try:
                from dataclasses import fields
                for f in fields(cfg):
                    result[f.name] = q_to_primitive(getattr(cfg, f.name))
            except Exception:
                # As a last resort, iterate over __dict__
                for k, v in getattr(cfg, "__dict__", {}).items():
                    result[k] = q_to_primitive(v)

            return result
        except Exception as e:
            # Never let snapshotting break config updates
            self._logger.debug("Config snapshot failed: %s", e)
            try:
                return asdict(self._config)
            except Exception:
                return {}
    
    # Abstract methods that must be implemented by subclasses
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def _initialize_impl(self) -> None:
        """Implementation-specific initialization."""
        pass
    
    @abstractmethod
    def _show_impl(self) -> None:
        """Implementation-specific show logic."""
        pass
    
    @abstractmethod
    def _hide_impl(self) -> None:
        """Implementation-specific hide logic."""
        pass
    
    @abstractmethod
    def _close_impl(self) -> None:
        """Implementation-specific cleanup logic."""
        pass
    
    @abstractmethod
    def _render_impl(self) -> None:
        """Implementation-specific rendering logic."""
        pass
    
    @abstractmethod
    def _config_updated(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Handle configuration updates.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        pass
    
    # Helper methods
    # -------------------------------------------------------------------------
    
    def _set_state(self, new_state: OverlayState) -> None:
        """Update the overlay's state and emit signals."""
        if self._state == new_state:
            return
            
        old_state = self._state
        self._state = new_state
        self._signals.state_changed.emit(old_state, new_state)
        self._logger.debug("State changed: %s -> %s", old_state.name, new_state.name)
    
    def __del__(self) -> None:
        """Ensure resources are cleaned up when the overlay is garbage collected."""
        if self._initialized:
            self._logger.warning("Overlay was not properly closed before destruction")
            self.close()
