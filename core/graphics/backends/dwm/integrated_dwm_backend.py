from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRect, QSize
from core.logging import get_logger
from ...overlay import Overlay as OverlayBase
from ...types import OverlayConfig
from ...overlay_host import OverlayHost
from utils import window_validation as winval
from utils.window.thumbnail_manager import ThumbnailManager
from core.threading import ThreadManager
from ...dwm_composition_manager import get_dwm_composition_manager
from utils.resource_manager import get_resource_manager, ResourceType
from utils.window.overlay_constants import (
    OVERLAY_MIN_WIDTH,
    OVERLAY_MIN_HEIGHT,
    DEFAULT_ASPECT,
)


class IntegratedDWMOverlay(OverlayBase):
    """DWM backend overlay with integrated border rendering.
    
    This version eliminates the separate BorderOverlay window by using
    IntegratedBorderCanvas which renders borders directly on the canvas.
    
    Benefits:
    - No z-order management complexity
    - No mouse transparency issues
    - Simplified event handling
    - Better performance
    - Direct double-click support
    """
    
    def get_config(self):
        """Get the current configuration of this overlay."""
        return self._config
        
    def get_backend_type(self):
        """Get the backend type of this overlay."""
        from ...backend_manager import BackendType
        return BackendType.DWM

    def __init__(self, config: OverlayConfig):
        super().__init__(config)
        self._logger = get_logger("IntegratedDWMOverlay")
        self._config = config

        # Core components
        self._host: Optional[OverlayHost] = None
        self._canvas = None  # Will be IntegratedBorderCanvas
        self._thumbnail_manager: Optional[ThumbnailManager] = None
        self._thumbnail_id: Optional[int] = None

        # Window tracking
        self._source_hwnd: Optional[int] = None
        self._target_hwnd: Optional[int] = None

        # State tracking
        self._is_visible = False
        self._opacity = float(getattr(config, "opacity", 1.0) or 1.0)
        self._has_revealed = False
        
        # Cached aspect ratio for consistent scaling during manual resize
        self._source_aspect: Optional[float] = None

        # Swap management
        self._swap_in_flight = False
        self._pending_swap_hwnd: Optional[int] = None

        # Lock state for focus indicator
        self._is_window_locked: bool = False

        # App instance for window enumeration (injected by OverlayManager)
        self.app_instance = None

        # UI coalescer for thumbnail property updates (initialized when Qt/app ready)
        self._thumb_coalescer = None
        # Triple buffer for latest content-rect exchange between signal producer and UI drain
        self._rect_tb = None

        # ResourceManager registration IDs for deterministic cleanup
        self._resource_ids: dict[str, int] = {}
        
        # Failure/Recovery flags
        self._broken: bool = False
        self._needs_recreate: bool = False

    def _normalize_source_hwnd(self, hwnd: int) -> int:
        """Normalize to the top-level window (GA_ROOT) for reliable geometry.
        
        Some callers may pass child/owner HWNDs; querying client rect on those can
        yield tiny sizes (e.g., menu/status bars). We map to the root window to
        ensure GetClientRect reflects the actual content area.
        """
        try:
            import ctypes
            root = ctypes.windll.user32.GetAncestor(int(hwnd or 0), 2)  # GA_ROOT = 2
            return int(root) if root else int(hwnd or 0)
        except Exception:
            return int(hwnd or 0)

    def _initialize_impl(self) -> None:
        """Initialize the DWM overlay with integrated border rendering (impl)."""
        try:
            # Validate source window: use properties['hwnd'] injected by OverlayManager
            source_hwnd = None
            try:
                props = getattr(self._config, "properties", {}) or {}
                source_hwnd = int(props.get("hwnd") or 0)
            except Exception:
                source_hwnd = 0

            # Normalize to top-level to avoid child/owner client-rect quirks
            norm_hwnd = self._normalize_source_hwnd(source_hwnd)
            if norm_hwnd and norm_hwnd != source_hwnd:
                self._logger.debug(f"Normalized source HWND {source_hwnd} -> {norm_hwnd}")

            if not norm_hwnd or not winval.is_valid_window(norm_hwnd):
                raise RuntimeError(f"Invalid source window: {norm_hwnd}")

            self._source_hwnd = norm_hwnd
            
            # Cache the initial aspect ratio for consistent scaling during resize
            self._cache_source_aspect()

            # Create host window with integrated canvas
            self._host = OverlayHost(self._config)
            self._host.setWindowTitle(f"DWM Overlay - {source_hwnd}")

            # Use IntegratedBorderCanvas instead of regular OverlayCanvas
            from ui.overlays.integrated_border_canvas import IntegratedBorderCanvas
            self._canvas = IntegratedBorderCanvas(self._host)
            self._host.set_canvas(self._canvas)

            # Inject parent overlay so host can forward context menu, etc.
            try:
                self._host._parent_overlay = self
            except Exception:
                pass

            # Set content aspect ratio for proper DWM thumbnail scaling (16:9 default)
            try:
                source_rect = winval.get_window_rect(self._source_hwnd)
                if source_rect and source_rect[2] > source_rect[0] and source_rect[3] > source_rect[1]:
                    # rect format: (left, top, right, bottom) - calculate width/height properly
                    src_w = source_rect[2] - source_rect[0]
                    src_h = source_rect[3] - source_rect[1]
                    self._canvas.set_content_aspect(src_w, src_h)
                else:
                    # Fallback to default aspect ratio
                    self._canvas.set_content_aspect(*DEFAULT_ASPECT)
            except Exception:
                # Fallback to default aspect ratio
                self._canvas.set_content_aspect(*DEFAULT_ASPECT)

            # Connect canvas content rect changes to thumbnail updates
            self._canvas.contentRectChanged.connect(self._on_content_rect_changed)

            # Get ThreadManager directly from singleton to access factory methods
            if not self._thumb_coalescer:
                from core.threading import get_thread_manager
                try:
                    tm = get_thread_manager()
                    # Log ThreadManager instance state to diagnose missing attribute issues
                    try:
                        tm_type = type(tm).__name__
                        tm_id = id(tm)
                        has_coalescer = hasattr(tm, 'create_ui_coalescer')
                        shutdown_flag = getattr(tm, '_shutdown', None)
                        self._logger.debug(
                            f"ThreadManager instance for coalescer: type={tm_type}, id={tm_id}, "
                            f"has_create_ui_coalescer={has_coalescer}, shutdown={shutdown_flag}"
                        )
                    except Exception as _log_e:
                        self._logger.debug(f"Failed to introspect ThreadManager in coalescer init: {_log_e}")

                    if hasattr(tm, 'create_ui_coalescer'):
                        self._thumb_coalescer = tm.create_ui_coalescer(
                            name=f"dwm_thumb_{id(self)}",
                            capacity=128,
                            window_ms=7,
                        )
                        self._logger.info("DWM thumbnail UI coalescer initialized (window=7ms, cap=128)")
                    else:
                        # Explicit diagnostic when attribute is missing
                        try:
                            attrs = [a for a in dir(tm) if not a.startswith('_')]
                            self._logger.error(
                                f"ThreadManager missing create_ui_coalescer; available attrs (partial): {attrs[:50]}"
                            )
                        except Exception:
                            pass
                        self._thumb_coalescer = None
                        self._logger.warning("Proceeding without UI coalescer; using direct UI scheduling")

                    # Create a triple buffer for latest QRect exchange
                    try:
                        self._rect_tb = tm.create_triple_buffer()
                        # Register triple buffer for lifecycle tracking (best-effort)
                        try:
                            rm = get_resource_manager()
                            rid = rm.register(
                                self._rect_tb,
                                ResourceType.CUSTOM,
                                "TripleBuffer for DWM content rect",
                                cleanup_handler=lambda tb: setattr(tb, "_slots", [None, None, None]),
                                tags={"dwm", "triple_buffer", f"overlay:{id(self)}"},
                            )
                            self._resource_ids["triple_buffer"] = rid
                        except Exception:
                            pass
                    except Exception as e:
                        self._rect_tb = None
                        self._logger.debug(f"Failed to initialize TripleBuffer: {e}")
                except Exception as e:
                    # Non-fatal; fallback to direct UI scheduling
                    self._thumb_coalescer = None
                    self._rect_tb = None
                    self._logger.debug(f"Failed to initialize DWM thumbnail coalescer: {e}")

            # Window behavior is initialized inside OverlayHost; no extra setup needed

            # Initialize thumbnail manager and register for deterministic cleanup
            self._thumbnail_manager = ThumbnailManager()
            try:
                rm = get_resource_manager()
                rid = rm.register(
                    self._thumbnail_manager,
                    ResourceType.CUSTOM,
                    "ThumbnailManager for IntegratedDWMOverlay",
                    cleanup_handler=lambda m: m.cleanup(),
                    tags={"dwm", "thumbnail_manager", f"overlay:{id(self)}"},
                )
                self._resource_ids["thumb_mgr"] = rid
            except Exception:
                pass

            # Get host window handle for DWM operations
            self._target_hwnd = int(self._host.winId())

            # Apply initial geometry and positioning
            self._apply_initial_geometry()

            # Set up context menu (no border overlay needed)
            self._setup_context_menu()

            # Apply DWM attributes
            self._apply_dwm_attributes()
            
            # Apply window masking for rounded corners (preserves mouse events)
            self._apply_window_masking()
            
            self._logger.info(f"Integrated DWM overlay initialized for window {self._source_hwnd}")
        except Exception as e:
            # Re-raise to make base Overlay.set_state -> ERROR and emit signals
            self._logger.error(f"Initialization failed: {e}")
            raise

    def _register_thumbnail(self) -> bool:
        """Register DWM thumbnail for the source window."""
        try:
            if not self._thumbnail_manager or not self._source_hwnd or not self._target_hwnd:
                return False
                
            # Register thumbnail (dest first, then source)
            thumbnail_id = self._thumbnail_manager.register_thumbnail(
                self._target_hwnd, self._source_hwnd
            )
            
            if thumbnail_id is None:
                self._logger.error("DWM thumbnail registration failed")
                return False
                
            self._thumbnail_id = thumbnail_id
            self._logger.debug(f"Registered DWM thumbnail {thumbnail_id}")
            # Register the thumbnail binding with ResourceManager for cleanup ordering
            try:
                rm = get_resource_manager()
                class _ThumbBinding:
                    def __init__(self, mgr, hwnd):
                        self.mgr = mgr
                        self.hwnd = hwnd
                binding = _ThumbBinding(self._thumbnail_manager, self._target_hwnd)
                rid = rm.register(
                    binding,
                    ResourceType.CUSTOM,
                    "DWM Thumbnail binding (dest->src)",
                    cleanup_handler=lambda b: b.mgr.unregister_thumbnail(b.hwnd),
                    tags={"dwm", "thumbnail_binding", f"overlay:{id(self)}"},
                )
                self._resource_ids["thumb_binding"] = rid
            except Exception:
                pass
            
            # Set initial properties
            self._update_thumbnail_properties()
            return True
            
        except Exception as e:
            self._logger.error(f"Thumbnail registration failed: {e}")
            return False

    def _cache_source_aspect(self) -> None:
        """Cache the source window aspect ratio for consistent scaling during resize.
        
        Uses client area instead of window rect to avoid decoration and coordinate space issues
        that can cause distortion on secondary displays with different resolutions.
        """
        try:
            if not self._source_hwnd:
                self._source_aspect = None
                return
            
            # Try to get client area first (more accurate for content aspect ratio)
            try:
                import ctypes
                import ctypes.wintypes
                
                # Log basic source info for diagnostics
                try:
                    title = winval.get_window_title(self._source_hwnd)
                    wclass = winval.get_window_class(self._source_hwnd)
                    wrect = winval.get_window_rect(self._source_hwnd)
                    if wrect:
                        wr_w = max(0, wrect[2] - wrect[0])
                        wr_h = max(0, wrect[3] - wrect[1])
                        self._logger.debug(
                            f"ASPECT_CACHE: hwnd={self._source_hwnd}, title='{title}', class='{wclass}', win_rect={wr_w}x{wr_h}"
                        )
                except Exception:
                    pass

                # Get client rect which excludes window decorations
                client_rect = ctypes.wintypes.RECT()
                if ctypes.windll.user32.GetClientRect(self._source_hwnd, ctypes.byref(client_rect)):
                    src_w = max(0, client_rect.right - client_rect.left)
                    src_h = max(0, client_rect.bottom - client_rect.top)
                    if src_w > 0 and src_h > 0:
                        aspect = src_w / src_h
                        # Sanity thresholds to avoid tiny/invalid client areas
                        too_small = src_w < 64 or src_h < 64 or (src_w * src_h) < 5000
                        out_of_bounds = aspect < 0.2 or aspect > 5.0
                        if not too_small and not out_of_bounds:
                            self._source_aspect = aspect
                            self._logger.debug(
                                f"Cached source aspect ratio from client area: {self._source_aspect:.3f} ({src_w}x{src_h})"
                            )
                            return
                        else:
                            self._logger.debug(
                                f"Client rect suspicious ({src_w}x{src_h}, ar={aspect:.3f}); falling back to window rect"
                            )
            except Exception as e:
                self._logger.debug(f"Failed to get client rect, falling back to window rect: {e}")
            
            # Fallback to window rect if client rect fails
            source_rect = winval.get_window_rect(self._source_hwnd)
            if source_rect and len(source_rect) >= 4 and source_rect[2] > source_rect[0] and source_rect[3] > source_rect[1]:
                # rect format: (left, top, right, bottom) - calculate width/height properly
                src_w = source_rect[2] - source_rect[0]
                src_h = source_rect[3] - source_rect[1]
                if src_w > 0 and src_h > 0:
                    aspect = src_w / src_h
                    # Clamp to reasonable bounds to avoid wild distortion
                    clamped = max(0.2, min(5.0, aspect))
                    self._source_aspect = clamped
                    if clamped != aspect:
                        self._logger.debug(
                            f"Cached source aspect ratio from window rect (clamped): {self._source_aspect:.3f} (raw={aspect:.3f}, {src_w}x{src_h})"
                        )
                    else:
                        self._logger.debug(
                            f"Cached source aspect ratio from window rect: {self._source_aspect:.3f} ({src_w}x{src_h})"
                        )
                    return
            
            self._source_aspect = None
            self._logger.debug("Could not determine source aspect ratio")
        except Exception as e:
            self._source_aspect = None
            self._logger.debug(f"Failed to cache source aspect: {e}")

    def _update_thumbnail_properties(self, content_rect_override: Optional[QRect] = None) -> None:
        """Update DWM thumbnail properties with corner clipping and proper aspect ratio.

        If content_rect_override is provided, it will be used instead of reading
        from the canvas to ensure coherence with the latest published geometry.
        """
        try:
            if not self._thumbnail_manager or not self._target_hwnd or not self._canvas:
                return
            
            # Get content rect from integrated canvas
            content_rect = content_rect_override if content_rect_override is not None else self._canvas.content_rect()
            if content_rect.isEmpty():
                self._logger.debug("Content rect empty, skipping thumbnail update")
                return
            
            # Use cached aspect ratio for consistent scaling during manual resize
            source_aspect = self._source_aspect
            # Sanity clamp to avoid extreme distortions
            if source_aspect is not None and not (0.2 <= source_aspect <= 5.0):
                self._logger.debug(f"Clamping source aspect {source_aspect:.3f} to bounds [0.2, 5.0]")
                source_aspect = max(0.2, min(5.0, source_aspect))
            
            # Calculate aspect-ratio preserving destination rect (pillarbox OR letterbox, never both)
            dest_rect = content_rect
            if source_aspect is not None and source_aspect > 0:
                content_w = content_rect.width()
                content_h = content_rect.height()
                
                if content_w <= 0 or content_h <= 0:
                    # Invalid content rect, skip aspect ratio adjustment
                    pass
                else:
                    content_aspect = content_w / content_h
                    
                    if abs(source_aspect - content_aspect) > 0.01:  # Only adjust if significantly different
                        if source_aspect > content_aspect:
                            # Source is wider - letterbox (black bars top/bottom)
                            new_h = max(1, int(content_w / source_aspect))
                            y_offset = (content_h - new_h) // 2
                            dest_rect = QRect(content_rect.x(), content_rect.y() + y_offset, content_w, new_h)
                        else:
                            # Source is taller - pillarbox (black bars left/right)  
                            new_w = max(1, int(content_h * source_aspect))
                            x_offset = (content_w - new_w) // 2
                            dest_rect = QRect(content_rect.x() + x_offset, content_rect.y(), new_w, content_h)
            
            # Apply DWM thumbnail clipping with proper z-order handling
            if self._canvas and hasattr(self._canvas, '_border_metrics') and self._canvas._border_metrics:
                border_metrics = self._canvas._border_metrics
                corner_radius = border_metrics.corner_radius
                if corner_radius > 0:
                    # Ensure content rect is inset to prevent overlap
                    content_inset = max(1, int(corner_radius * 0.1))
                    dest_rect = dest_rect.adjusted(content_inset, content_inset, -content_inset, -content_inset)
                    
                    # Convert to physical pixels for DWM (DPI-aware scaling)
                    dpi_scale = self._host.devicePixelRatioF() if self._host else 1.0
                    
                    # Scale logical coordinates to physical pixels.
                    # IMPORTANT: DWM RECT expects right/bottom as exclusive edges (right = left + width).
                    left_px = int(round(dest_rect.x() * dpi_scale))
                    top_px = int(round(dest_rect.y() * dpi_scale))
                    right_px = left_px + int(round(dest_rect.width() * dpi_scale))
                    bottom_px = top_px + int(round(dest_rect.height() * dpi_scale))

                    # Update thumbnail properties on destination hwnd key
                    self._thumbnail_manager.update_thumbnail(
                        self._target_hwnd,
                        dest_rect=(left_px, top_px, right_px, bottom_px),
                        opacity=self._opacity,
                        visible=self._is_visible,
                        source_client_area_only=True,
                    )
                    
                    aspect_str = f"{source_aspect:.3f}" if source_aspect is not None else "None"
                    self._logger.debug(
                        f"Updated thumbnail properties with aspect ratio: "
                        f"source_aspect={aspect_str}, logical_rect={dest_rect.getRect()}, "
                        f"physical_rect=({left_px},{top_px},{right_px},{bottom_px}), "
                        f"dpi_scale={dpi_scale:.2f}, opacity={self._opacity}"
                    )
                    # Ensure focus indicator stays above after updates
                    self._ensure_focus_indicator_z_order()
                    return
            
            # Fallback to minimal inset for very small overlays
            if dest_rect.width() < 10 or dest_rect.height() < 10:
                dest_rect = dest_rect.adjusted(1, 1, -1, -1)
            
            # Convert to physical pixels for DWM (DPI-aware scaling)
            dpi_scale = self._host.devicePixelRatioF() if self._host else 1.0
            
            # Scale logical coordinates to physical pixels.
            # IMPORTANT: DWM RECT expects right/bottom as exclusive edges (right = left + width).
            left_px = int(round(dest_rect.x() * dpi_scale))
            top_px = int(round(dest_rect.y() * dpi_scale))
            right_px = left_px + int(round(dest_rect.width() * dpi_scale))
            bottom_px = top_px + int(round(dest_rect.height() * dpi_scale))

            # Update thumbnail properties on destination hwnd key
            self._thumbnail_manager.update_thumbnail(
                self._target_hwnd,
                dest_rect=(left_px, top_px, right_px, bottom_px),
                opacity=self._opacity,
                visible=self._is_visible,
                source_client_area_only=True,
            )
            
            # Only log scaling info when aspect ratio changes significantly
            if hasattr(self, '_last_logged_aspect') and self._last_logged_aspect:
                if source_aspect and abs(source_aspect - self._last_logged_aspect) < 0.05:
                    pass  # Skip logging for minor changes
                else:
                    aspect_str = f"{source_aspect:.3f}" if source_aspect is not None else "None"
                    self._logger.debug(
                        f"THUMBNAIL_SCALING: source_aspect={aspect_str}, "
                        f"dest_rect={dest_rect.getRect()}, physical_rect=({left_px},{top_px},{right_px},{bottom_px})"
                    )
                    self._last_logged_aspect = source_aspect
            else:
                self._last_logged_aspect = source_aspect
            
            # Ensure focus indicator stays above DWM thumbnail after updates
            self._ensure_focus_indicator_z_order()
        
        except Exception as e:
            self._logger.error(f"Thumbnail property update failed: {e}")

    def _ensure_focus_indicator_z_order(self) -> None:
        """Ensure focus indicator stays above DWM thumbnail after thumbnail updates."""
        try:
            if self._host and hasattr(self._host, '_focus_indicator'):
                focus_indicator = self._host._focus_indicator
                if focus_indicator and focus_indicator.isVisible():
                    # Defer z-order enforcement to avoid conflicts with DWM operations
                    ThreadManager.single_shot(5, lambda: self._raise_focus_indicator(focus_indicator))
        except Exception as e:
            self._logger.debug(f"Focus indicator z-order enforcement failed: {e}")

    def _raise_focus_indicator(self, focus_indicator) -> None:
        """Raise focus indicator above DWM thumbnail."""
        try:
            from utils.resource_manager import get_resource_manager
            rm = get_resource_manager()
            success = rm.bring_child_to_front(focus_indicator)
            self._logger.debug(f"DWM post-update focus indicator z-order: {success}")
        except Exception as e:
            self._logger.debug(f"Focus indicator raise failed: {e}")

    def _on_content_rect_changed(self, rect: QRect) -> None:
        """Handle content rectangle changes from the integrated canvas.

        Publish the latest rect via triple buffer and coalesce a single UI-thread
        drain to apply the most recent geometry.
        """
        try:
            # Publish the latest rect if triple buffer available
            if self._rect_tb is not None and rect is not None:
                try:
                    self._rect_tb.publish(rect)
                except Exception:
                    pass
            # Prefer coalesced updates during rapid resize/drag; fallback to direct UI scheduling
            if self._thumb_coalescer is not None:
                self._thumb_coalescer.submit(self._drain_and_update_thumbnail)
            else:
                ThreadManager.run_on_ui_thread(self._drain_and_update_thumbnail)
        except Exception:
            # Best-effort fallback
            ThreadManager.run_on_ui_thread(self._update_thumbnail_properties)

    def _drain_and_update_thumbnail(self) -> None:
        """Consume the latest content rect (if any) and update thumbnail once."""
        try:
            latest_rect = None
            if self._rect_tb is not None:
                try:
                    latest_rect = self._rect_tb.consume_latest()
                except Exception:
                    latest_rect = None
            self._update_thumbnail_properties(latest_rect)
        except Exception:
            # Fallback to current rect
            self._update_thumbnail_properties()

    def _apply_initial_geometry(self) -> None:
        """Apply initial overlay geometry and positioning."""
        try:
            if not self._host:
                return
            
            # Set minimum size aligned with integrated canvas constraints
            min_size = QSize(OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT)
            self._host.setMinimumSize(min_size)

            # Get accurate source window aspect ratio for proper scaling
            src_w, src_h = DEFAULT_ASPECT
            try:
                if self._source_hwnd:
                    rect = winval.get_window_rect(self._source_hwnd)
                    if rect and len(rect) >= 4 and rect[2] > rect[0] and rect[3] > rect[1]:
                        # rect format: (left, top, right, bottom) - calculate width/height properly
                        src_w, src_h = int(rect[2] - rect[0]), int(rect[3] - rect[1])
                        self._logger.debug(f"Source window aspect: {src_w}x{src_h} (AR: {src_w/src_h:.3f})")
            except Exception as e:
                self._logger.debug(f"Failed to get source window rect: {e}")
                pass

            # Calculate aspect ratio with bounds checking
            if src_h > 0:
                aspect = src_w / src_h
                # Clamp aspect ratio to reasonable bounds
                aspect = max(0.2, min(5.0, aspect))  # Between 1:5 and 5:1
            else:
                aspect = DEFAULT_ASPECT[0] / DEFAULT_ASPECT[1]
                
            # Store aspect ratio in canvas for scroll wheel resize
            if self._canvas and hasattr(self._canvas, 'set_content_aspect'):
                self._canvas.set_content_aspect(src_w, src_h)

            # Position at top-left of primary screen and size relative to screen
            screen = self._host.screen()
            if screen:
                screen_rect = screen.availableGeometry()
                scr_w = max(1, screen_rect.width())
                scr_h = max(1, screen_rect.height())

                # Insets for inner-content AR preservation (DPI-aware, from canvas)
                ix = iy = 0
                try:
                    if self._canvas and hasattr(self._canvas, 'get_content_insets'):
                        cx, cy = self._canvas.get_content_insets()
                        ix = max(0, int(cx))
                        iy = max(0, int(cy))
                except Exception:
                    ix = iy = 0

                # Choose default INNER width ~40% of screen width
                target_inner_w = int(scr_w * 0.4)
                target_inner_w = max(1, target_inner_w)
                target_inner_h = int(target_inner_w / aspect)

                # Constrain INNER height to ~45% of screen height
                max_inner_h = int(scr_h * 0.45)
                if target_inner_h > max_inner_h:
                    target_inner_h = max(1, max_inner_h)
                    target_inner_w = int(target_inner_h * aspect)

                # Convert INNER to OUTER by adding per-side insets
                target_w = target_inner_w + 2 * ix
                target_h = target_inner_h + 2 * iy

                # Final clamp not to exceed screen bounds (outer)
                target_w = min(target_w, scr_w)
                target_h = min(target_h, scr_h)

                # Ensure minimum constraints (outer)
                target_w = max(target_w, min_size.width())
                target_h = max(target_h, min_size.height())

                self._host.setGeometry(
                    screen_rect.x(), screen_rect.y(),
                    target_w, target_h
                )
            else:
                # Fallback when no screen is available
                # Keep aspect using a width of 400px baseline
                base_w = max(min_size.width(), 400)
                base_h = max(min_size.height(), int(base_w / aspect))
                self._host.resize(QSize(base_w, base_h))
            
        except Exception as e:
            self._logger.error(f"Initial geometry setup failed: {e}")

    def _setup_context_menu(self) -> None:
        """Set up context menu for the integrated overlay."""
        try:
            # Use the existing context menu system
            from utils.overlay_context_menu import OverlayContextMenu
            
            # Create context menu handler (integrated border approach)
            self._context_menu = OverlayContextMenu(
                self._host, 
                overlay_type='dwm'
                # No border_overlay parameter needed with integrated approach
            )
            
            # Attach to host (OverlayContextMenu.attach_to_overlay accepts a single overlay/widget)
            self._context_menu.attach_to_overlay(self._host)
            # Register context menu for deterministic cleanup
            try:
                rm = get_resource_manager()
                rid = rm.register(
                    self._context_menu,
                    ResourceType.CUSTOM,
                    "Overlay context menu (DWM)",
                    cleanup_handler=lambda cm: hasattr(cm, "detach_from_overlay") and cm.detach_from_overlay(),
                    tags={"context_menu", "dwm", f"overlay:{id(self)}"},
                )
                self._resource_ids["context_menu"] = rid
            except Exception:
                pass
            
        except Exception as e:
            self._logger.error(f"Context menu setup failed: {e}")

    def _apply_dwm_attributes(self) -> None:
        """Apply DWM composition attributes."""
        try:
            if not self._host:
                return
                
            dwm_manager = get_dwm_composition_manager()
            
            # Apply standard DWM attributes for overlay rendering
            # Pass the QWidget host; manager will obtain HWND via widget.winId()
            dwm_manager.apply_overlay_attributes(self._host)
            
        except Exception as e:
            self._logger.error(f"DWM attributes application failed: {e}")

    def _apply_window_masking(self) -> None:
        """Apply window-level masking for rounded corners while preserving mouse events."""
        try:
            if not self._host or not self._canvas:
                return
                
            # Only apply masking if rounded borders are enabled
            if (hasattr(self._canvas, '_border_metrics') and self._canvas._border_metrics and
                hasattr(self._canvas, '_rounded_enabled') and self._canvas._rounded_enabled):
                border_metrics = self._canvas._border_metrics
                if border_metrics.corner_radius > 0:
                    from PySide6.QtGui import QRegion, QPainterPath
                    
                    # Create rounded region for host window
                    path = QPainterPath()
                    host_rect = self._host.rect()
                    
                    # Use slightly smaller radius to ensure clean edges
                    mask_radius = max(1.0, border_metrics.corner_radius - 0.5)
                    path.addRoundedRect(host_rect, mask_radius, mask_radius)
                    
                    # Convert to region and apply as window mask
                    # This provides hardware-accelerated clipping while preserving all mouse events
                    region = QRegion(path.toFillPolygon().toPolygon())
                    self._host.setMask(region)
                    
                    self._logger.debug(f"Applied window mask with radius {mask_radius}")
                    return
            
            # Clear mask if rounded borders are disabled
            self._host.clearMask()
            
        except Exception as e:
            self._logger.error(f"Window masking failed: {e}")
            # Ensure mask is cleared on failure to prevent broken state
            try:
                if self._host:
                    self._host.clearMask()
            except Exception:
                pass

    def _show_impl(self) -> None:
        """Show the overlay (impl)."""
        try:
            if not self._host:
                raise RuntimeError("Overlay host not created")

            # Show host window first so native HWND is valid for DWM APIs
            self._host.show()
            self._host.raise_()
            self._host.activateWindow()

            # After showing, obtain destination HWND and register DWM thumbnail
            try:
                self._target_hwnd = int(self._host.winId())
            except Exception:
                self._target_hwnd = 0

            # Refresh source aspect after show (in case source window state changed)
            try:
                self._cache_source_aspect()
            except Exception:
                pass

            # Attempt immediate registration; if it fails, schedule background retries without failing show
            if not self._register_thumbnail():
                self._logger.warning(
                    "Initial DWM registration failed after show; scheduling deferred retries"
                )
                ThreadManager.single_shot(20, lambda: self._register_after_show_retry(1))

            # Update visibility state
            self._is_visible = True

            # Update thumbnail visibility
            self._update_thumbnail_properties()

            # Trigger fade-in if not revealed yet (1000ms for creation)
            if not self._has_revealed:
                self._fade_in_thumbnail(1000)
                self._has_revealed = True

            self._logger.debug("Integrated overlay shown")
        except Exception as e:
            self._logger.error(f"Show failed: {e}")
            raise

    def _register_after_show_retry(self, attempt: int) -> None:
        """Retry DWM thumbnail registration a few times after show until HWND is valid.

        This avoids E_INVALIDARG by ensuring the destination HWND exists before calling DWM.
        """
        try:
            max_attempts = 25  # ~500ms total at 20ms intervals
            if not self._host:
                return
            try:
                if not self._target_hwnd:
                    self._target_hwnd = int(self._host.winId())
            except Exception:
                self._target_hwnd = 0

            # Attempt registration regardless of pre-validation; rely on DWM + retry
            if not self._register_thumbnail():
                # If the failure is E_INVALIDARG, one more delayed attempt may succeed
                last_hr = None
                try:
                    if self._thumbnail_manager is not None:
                        last_hr = getattr(self._thumbnail_manager, 'last_hresult', None)
                except Exception:
                    last_hr = None
                if attempt < max_attempts and (last_hr in (None, -2147024809)):
                    ThreadManager.single_shot(20, lambda: self._register_after_show_retry(attempt + 1))
                    return
                # Give up quietly; overlay remains functional and may register on future swaps
                self._logger.debug("DWM thumbnail registration still failing after deferred retries; leaving overlay without thumbnail")
                return

            # Success: complete the normal show sequence
            self._is_visible = True
            self._update_thumbnail_properties()
            if not self._has_revealed:
                self._fade_in_thumbnail(1000)
                self._has_revealed = True
            self._logger.debug("Integrated overlay shown (deferred registration)")
        except Exception as e:
            self._logger.error(f"Deferred registration failed: {e}")

    def _hide_impl(self) -> None:
        """Hide the overlay (impl)."""
        try:
            if not self._host:
                return

            # Hide host window
            self._host.hide()

            # Update visibility state
            self._is_visible = False

            # Update thumbnail visibility
            self._update_thumbnail_properties()

            self._logger.debug("Integrated overlay hidden")
        except Exception as e:
            self._logger.error(f"Hide failed: {e}")
            raise

    def _close_impl(self) -> None:
        """Close and cleanup the overlay (impl)."""
        try:
            # Hide first
            try:
                self._hide_impl()
            except Exception:
                pass

            # Capture ThreadManager state during close path for diagnostics (singleton)
            try:
                from core.threading import get_thread_manager
                tm = get_thread_manager()
                self._logger.debug(
                    f"Close path ThreadManager: type={type(tm).__name__}, id={id(tm)}, "
                    f"has_create_ui_coalescer={hasattr(tm, 'create_ui_coalescer')}"
                )
            except Exception:
                pass

            # Prefer ResourceManager-driven cleanup where registered
            try:
                rm = get_resource_manager()
                for key in list(self._resource_ids.keys()):
                    try:
                        rid = self._resource_ids.pop(key, None)
                        if rid is not None:
                            rm.unregister(rid, force=True)  # type: ignore[arg-type]
                    except Exception:
                        pass
            except Exception:
                pass

            # Best-effort direct cleanup for thumbnail if still present
            if self._thumbnail_manager and self._target_hwnd:
                try:
                    self._thumbnail_manager.unregister_thumbnail(self._target_hwnd)
                except Exception:
                    pass
                self._thumbnail_id = None

            # Cleanup context menu
            if hasattr(self, '_context_menu') and self._context_menu:
                try:
                    self._context_menu.detach_from_overlay()
                except Exception:
                    pass

            # Close host window
            if self._host:
                try:
                    self._host.close()
                finally:
                    self._host = None

            self._canvas = None
            self._logger.debug("Integrated overlay closed")
        except Exception as e:
            self._logger.error(f"Close failed: {e}")
            raise

    def set_opacity(self, opacity: float) -> bool:
        """Set overlay opacity with unified canvas and thumbnail handling."""
        try:
            raw = float(opacity)
            clamped = max(0.1, min(1.0, raw))
            if clamped != raw:
                try:
                    self._logger.debug(f"[OPACITY] DWM clamp -> {clamped:.2f} (from {raw:.3f})")
                except Exception:
                    pass
            # Log at bounds for validation
            if clamped in (0.1, 1.0):
                try:
                    self._logger.debug(f"[OPACITY] DWM set -> {int(clamped*100)}%")
                except Exception:
                    pass
            self._opacity = clamped

            # Update canvas backdrop opacity (borders stay at full opacity)
            if self._canvas:
                self._canvas.set_opacity(self._opacity)

            # Update thumbnail opacity to match canvas for unified fade
            self._update_thumbnail_properties()

            return True
        except Exception as e:
            self._logger.error(f"Set opacity failed: {e}")
            return False

    def _fade_in_thumbnail(self, duration_ms: int = 1000) -> None:
        """Trigger fade-in animation for the thumbnail."""
        try:
            if not self._thumbnail_manager or not self._thumbnail_id:
                return
                
            # Calculate steps for smooth animation (60 FPS)
            frame_time = 16  # ~60 FPS
            max_steps = max(1, duration_ms // frame_time)
            
            def fade_step(step: int):
                if step >= max_steps:
                    # Ensure final opacity is set
                    if self._canvas:
                        self._canvas.set_opacity(self._opacity)
                    return
                    
                # Smooth easing curve (ease-out)
                progress = step / max_steps
                eased_progress = 1 - (1 - progress) ** 2
                current_opacity = self._opacity * eased_progress
                
                if self._canvas:
                    self._canvas.set_opacity(current_opacity)
                    
                # Schedule next step
                ThreadManager.single_shot(frame_time, lambda: fade_step(step + 1))
                
            # Start fade animation from 0 opacity
            if self._canvas:
                self._canvas.set_opacity(0.0)
            fade_step(0)
            
        except Exception as e:
            self._logger.error(f"Fade-in failed: {e}")

    # --- Overlay base hooks -------------------------------------------------
    def _render_impl(self) -> None:
        """DWM thumbnails are updated on events; no per-frame rendering required."""
        # No-op by design; keep available for future diagnostics
        return

    def _config_updated(self, old_config: dict, new_config: dict) -> None:
        """Apply configuration changes to the running overlay."""
        try:
            # Opacity changes
            old_op = float(old_config.get("opacity", 1.0) or 1.0)
            new_op = float(new_config.get("opacity", 1.0) or 1.0)
            if new_op != old_op:
                self.set_opacity(new_op)

            # Title change handled via _set_title_impl hook from base.set_title()

            # Geometry updates
            pos_changed = old_config.get("position") != new_config.get("position")
            size_changed = old_config.get("size") != new_config.get("size")
            if (pos_changed or size_changed) and self._host is not None:
                pos = new_config.get("position")
                size = new_config.get("size")
                try:
                    if pos and size:
                        self._host.setGeometry(QRect(pos, size))
                    elif pos:
                        g = self._host.geometry()
                        self._host.setGeometry(QRect(pos, g.size()))
                    elif size:
                        g = self._host.geometry()
                        self._host.setGeometry(QRect(g.topLeft(), size))
                finally:
                    self._update_thumbnail_properties()

            # Source hwnd swap via properties
            try:
                old_props = dict(old_config.get("properties") or {})
                new_props = dict(new_config.get("properties") or {})
                old_hwnd = int(old_props.get("hwnd") or 0)
                new_hwnd = int(new_props.get("hwnd") or 0)
            except Exception:
                old_hwnd = 0
                new_hwnd = 0
            if new_hwnd and new_hwnd != old_hwnd:
                # Perform swap on UI thread path
                self._handle_swap_window(new_hwnd)
        except Exception as e:
            self._logger.error(f"Config update failed: {e}")
            raise

    def _set_title_impl(self, title: str) -> None:  # pragma: no cover
        try:
            if self._host is not None:
                self._host.setWindowTitle(title)
        except Exception:
            # Non-fatal
            pass

    def _handle_swap_window(self, target_hwnd: int, record_mru: bool = False) -> None:
        """Handle window swap request."""
        try:
            if self._swap_in_flight:
                # Queue the request for later
                self._pending_swap_hwnd = target_hwnd
                self._pending_swap_record_mru = record_mru
                self._logger.debug(f"Swap in flight, queuing hwnd {target_hwnd}")
                return
            
            self._swap_in_flight = True
            self._pending_swap_hwnd = None
            self._pending_swap_record_mru = False
            
            # Perform the swap
            success = self._swap_source_hwnd(target_hwnd, record_mru=record_mru)
            if not success:
                self._logger.error(f"Failed to swap to hwnd {target_hwnd}")
        except Exception as e:
            self._logger.error(f"Window swap handler failed: {e}", exc_info=True)
            self._swap_in_flight = False

    def update_source(self, hwnd: int) -> bool:
        """Update the overlay's source window handle (OverlayManager reuse path).

        This is a thin wrapper that validates the argument and delegates to
        the existing swap handler, which queues the work on the UI thread.

        Returns True if the request was accepted for processing.
        """
        try:
            new_hwnd = int(hwnd) if hwnd is not None else 0
        except Exception:
            new_hwnd = 0

        if not new_hwnd or not winval.is_valid_window(new_hwnd):
            self._logger.error(f"update_source rejected invalid hwnd: {hwnd}")
            return False

        # Delegate to unified swap path
        self._handle_swap_window(new_hwnd)
        return True

    def _swap_source_hwnd(self, new_hwnd: int, record_mru: bool = False) -> bool:
        """Swap the source window handle."""
        self._logger.debug(
            f"_swap_source_hwnd start: new_hwnd={new_hwnd} target_hwnd={self._target_hwnd} visible={self._is_visible}"
        )
        success = False
        try:
            if not winval.is_valid_window(new_hwnd):
                self._logger.error(f"Invalid swap target: {new_hwnd}")
                return False

            # Reset reveal state for fade effect
            self._has_revealed = False

            # If overlay is not visible or host/target isn't ready, just update source and bail.
            # Registration will happen on next show/visibility change.
            if not self._host or not getattr(self._host, 'isVisible', lambda: False)() or not self._target_hwnd:
                self._source_hwnd = new_hwnd
                self._cache_source_aspect()
                self._logger.debug("Overlay hidden or target HWND not ready; deferred DWM registration until visible")
                return True

            # Unregister old thumbnail using destination HWND key
            if self._thumbnail_manager and self._target_hwnd:
                try:
                    self._logger.debug(
                        f"Unregistering previous thumbnail for target_hwnd={self._target_hwnd}"
                    )
                    self._thumbnail_manager.unregister_thumbnail(self._target_hwnd)
                except Exception as ue:
                    self._logger.debug(f"Unregister previous thumbnail exception: {ue}")
                finally:
                    self._thumbnail_id = None

            # Update source window
            self._source_hwnd = new_hwnd
            self._logger.debug(f"Updated source hwnd to {self._source_hwnd}")
            
            # Cache new aspect ratio for consistent scaling
            self._cache_source_aspect()

            # Register new thumbnail
            if not self._register_thumbnail():
                # Capture HRESULT and state for diagnostics
                last_hr = None
                try:
                    if self._thumbnail_manager is not None:
                        last_hr = getattr(self._thumbnail_manager, 'last_hresult', None)
                except Exception:
                    last_hr = None
                try:
                    src_ok = winval.is_valid_window(self._source_hwnd or 0)
                    tgt_ok = winval.is_valid_window(self._target_hwnd or 0)
                except Exception:
                    src_ok = tgt_ok = None
                self._logger.error(
                    f"Failed to register new thumbnail after swap. last_hresult={last_hr} src_ok={src_ok} tgt_ok={tgt_ok}"
                )
                # Mark as broken to allow manager-driven recovery
                self._broken = True
                self._needs_recreate = True
                success = False
            else:
                # Update properties and trigger fade-in (500ms for swap)
                self._update_thumbnail_properties()
                try:
                    # Force a repaint of the host to ensure immediate visual refresh
                    if self._host:
                        self._host.update()
                except Exception:
                    pass
                if self._is_visible:
                    self._fade_in_thumbnail(500)
                    self._has_revealed = True
                self._logger.info(f"Swapped to window {new_hwnd}")
                # Clear any previous failure flags now that we are healthy
                self._broken = False
                self._needs_recreate = False
                success = True
                
                # Add to MRU only when explicitly requested (e.g., context menu swaps)
                if record_mru:
                    try:
                        from core.switching.mru_manager import get_mru_manager
                        get_mru_manager().record(new_hwnd)
                        self._logger.debug(f"Added hwnd {new_hwnd} to MRU from context menu swap")
                    except Exception as e:
                        self._logger.debug(f"Failed to add context menu swap to MRU: {e}")
                
                success = True
            return success
        except Exception as e:
            self._logger.error(f"Source swap failed: {e}")
            return False
        finally:
            # Always release in-flight and dispatch any pending request
            try:
                pending = self._pending_swap_hwnd
                pending_record_mru = getattr(self, '_pending_swap_record_mru', False)
                self._pending_swap_hwnd = None
                self._pending_swap_record_mru = False
            except Exception:
                pending = None
                pending_record_mru = False
            if pending:
                # Allow next swap to start
                self._swap_in_flight = False
                self._logger.debug(f"Dispatching pending swap to hwnd {pending}")
                ThreadManager.single_shot(10, lambda: self._handle_swap_window(pending, record_mru=pending_record_mru))
            else:
                self._swap_in_flight = False
                self._logger.debug("_swap_source_hwnd complete: no pending swap")

    def _handle_reset_position(self) -> None:
        """Reset overlay position and size."""
        try:
            if not self._host:
                return
                
            # Clear border metrics cache on resize
            self._border_metrics = None
            
            # Update canvas layout if available
            if hasattr(self._canvas, '_update_layout'):
                self._canvas._update_layout()
            
            # Reapply window masking after resize
            if hasattr(self, '_host') and self._host:
                try:
                    # Get parent overlay to access masking method
                    parent = self.parent()
                    if parent and hasattr(parent, '_parent_overlay'):
                        overlay = parent._parent_overlay
                        if hasattr(overlay, '_apply_window_masking'):
                            overlay._apply_window_masking()
                except Exception as e:
                    logger = get_logger("IntegratedBorderCanvas")
                    logger.debug(f"Window mask update after resize failed: {e}")
            
            # Reset to initial geometry
            self._apply_initial_geometry()
            
            # Update thumbnail properties
            self._update_thumbnail_properties()
            
            self._logger.debug("Position reset completed")
            
        except Exception as e:
            self._logger.error(f"Position reset failed: {e}")

    def _handle_quit_application(self) -> None:
        """Handle quit application request."""
        try:
            from PySide6.QtCore import QCoreApplication
            self._logger.info("Quit application requested from overlay")
            QCoreApplication.quit()
        except Exception as e:
            self._logger.error(f"Quit application failed: {e}")

    # Properties for compatibility
    @property
    def is_visible(self) -> bool:
        return self._is_visible
        
    @property
    def source_hwnd(self) -> Optional[int]:
        return self._source_hwnd
    
    # Compatibility aliases used by switching controllers
    @property
    def _current_source_hwnd(self) -> Optional[int]:
        return self._source_hwnd
    
    @property
    def _src_hwnd(self) -> Optional[int]:
        return self._source_hwnd
        
    def toggle_window_lock(self) -> None:
        """Toggle the window lock state for the focus indicator."""
        self._is_window_locked = not self._is_window_locked
        self._logger.debug(f"Window lock toggled to: {self._is_window_locked}")
        
    @property
    def host_window(self):
        return self._host
