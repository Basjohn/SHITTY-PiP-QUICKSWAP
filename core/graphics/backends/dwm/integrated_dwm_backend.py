from __future__ import annotations

from typing import Optional
import time

from PySide6.QtCore import QRect, QSize
from core.logging import get_logger
from core.threading import ThreadManager
from core.settings import get_settings_manager
from ...overlay import Overlay as OverlayBase
from ...types import OverlayConfig
from ...overlay_host import OverlayHost
from utils import window_validation as winval
from utils.window.thumbnail_manager import ThumbnailManager
from ...dwm_composition_manager import get_dwm_composition_manager
from utils.resource_manager import get_resource_manager, ResourceType
from utils.window.overlay_constants import (
    OVERLAY_MIN_WIDTH,
    OVERLAY_MIN_HEIGHT,
    DEFAULT_ASPECT,
)
from utils.window.monitors import (
    get_available_geometry_for_monitor,
    ensure_within_available_desktop,
)
from utils.window.overlay_persistence import (
    nearest_corner_state_from_rect,
    geometry_from_state,
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
    
    # Aspect ratio bounds: chosen to prevent extreme distortions while allowing reasonable window shapes
    # MIN: 2.5:1 portrait (e.g., vertical phone screens)
    # MAX: 3:1 ultrawide (common ultrawide monitors)
    # Bounds chosen to avoid overlap with border/accent decorations on extreme ARs
    MIN_REASONABLE_AR = 0.4  # 2.5:1 portrait
    MAX_REASONABLE_AR = 3.0  # 3:1 ultrawide
    SAFE_FALLBACK_AR = 16.0 / 9.0  # 1.778 (standard widescreen)
    
    def get_config(self):
        """Get the current configuration of this overlay."""
        return self._config
        
    def get_backend_type(self):
        """Get the backend type of this overlay."""
        from ...backend_manager import BackendType
        return BackendType.DWM
    
    def get_source_hwnd(self) -> Optional[int]:
        """Get the current source window handle."""
        return self._source_hwnd

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
        # Transient override used during animations (e.g., crossfade on swap)
        self._visual_opacity: Optional[float] = None
        
        # Cached aspect ratio for consistent scaling during manual resize
        self._source_aspect: Optional[float] = None

        # Swap management
        self._swap_in_flight = False
        self._pending_swap_hwnd: Optional[int] = None
        self._pending_swap_record_mru: bool = False

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

            # Coalesce geometry changes and persist on idle (standalone DWM only)
            try:
                if not self._is_docking_overlay() and hasattr(self._host, 'geometryChanged'):
                    self._host.geometryChanged.connect(self._on_host_geometry_changed)
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
                        try:
                            self._thumb_coalescer = tm.create_ui_coalescer(
                                name=f"dwm_thumb_{id(self)}",
                                capacity=128,
                                window_ms=7,
                            )
                            if self._thumb_coalescer:
                                self._logger.info("DWM thumbnail UI coalescer initialized (window=7ms, cap=128)")
                            else:
                                self._logger.error("UI coalescer creation returned None")
                        except Exception as coalescer_e:
                            self._thumb_coalescer = None
                            self._logger.error(f"Failed to create UI coalescer: {coalescer_e}")
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

            # Apply initial geometry and positioning (try persisted first)
            if not self._apply_persisted_geometry():
                self._apply_initial_geometry()

            # Set up context menu (no border overlay needed)
            self._setup_context_menu()

            # Apply DWM attributes
            self._apply_dwm_attributes()
            
            # Apply window masking for rounded corners (preserves mouse events)
            self._apply_window_masking()
            
            # Deferred AR correction to eliminate minor initial gap without manual resize
            # Skip for docking overlays - DockingOverlayManager controls their geometry
            try:
                if not self._is_docking_overlay():
                    ThreadManager.single_shot(15, self._handle_correct_aspect)
            except Exception:
                pass

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
                    cw = max(0, client_rect.right - client_rect.left)
                    ch = max(0, client_rect.bottom - client_rect.top)
                    if cw > 0 and ch > 0:
                        # Clamp to reasonable bounds to avoid wild distortion
                        if cw > 0 and ch > 0:
                            self._source_aspect = cw / ch
                            if self._source_aspect < self.MIN_REASONABLE_AR or self._source_aspect > self.MAX_REASONABLE_AR:
                                self._logger.debug(
                                    f"Discarding out-of-bounds client aspect: {self._source_aspect:.3f} (bounds: {self.MIN_REASONABLE_AR:.1f}-{self.MAX_REASONABLE_AR:.1f})"
                                )
                            else:
                                self._logger.debug(
                                    f"Cached source aspect ratio from client rect: {self._source_aspect:.3f} ({cw}x{ch})"
                                )
                                # Propagate fresh content aspect to canvas for correct letterbox/pillarbox
                                try:
                                    if self._canvas and hasattr(self._canvas, 'set_content_aspect'):
                                        self._canvas.set_content_aspect(int(cw), int(ch))
                                except Exception:
                                    pass
                                return
            except Exception:
                # Ignore client-rect failures and fall back to window rect
                pass

            # Fallback: use full window rectangle when client rect is unavailable
            try:
                rect = winval.get_window_rect(self._source_hwnd)
                if rect and len(rect) >= 4:
                    src_w = max(0, int(rect[2] - rect[0]))
                    src_h = max(0, int(rect[3] - rect[1]))
                    if src_w > 0 and src_h > 0:
                        self._source_aspect = src_w / src_h
                        if self._source_aspect < self.MIN_REASONABLE_AR or self._source_aspect > self.MAX_REASONABLE_AR:
                            self._logger.debug(
                                f"Discarding out-of-bounds window aspect: {self._source_aspect:.3f} (bounds: {self.MIN_REASONABLE_AR:.1f}-{self.MAX_REASONABLE_AR:.1f})"
                            )
                        else:
                            self._logger.debug(
                                f"Cached source aspect ratio from window rect: {self._source_aspect:.3f} ({src_w}x{src_h})"
                            )
                            # Propagate fresh content aspect to canvas for correct letterbox/pillarbox
                            try:
                                if self._canvas and hasattr(self._canvas, 'set_content_aspect'):
                                    self._canvas.set_content_aspect(int(src_w), int(src_h))
                            except Exception:
                                pass
                        return
            except Exception:
                pass

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
            if source_aspect is not None and not (self.MIN_REASONABLE_AR <= source_aspect <= self.MAX_REASONABLE_AR):
                self._logger.debug(f"Clamping source aspect {source_aspect:.3f} to bounds [{self.MIN_REASONABLE_AR:.1f}, {self.MAX_REASONABLE_AR:.1f}]")
                source_aspect = max(self.MIN_REASONABLE_AR, min(self.MAX_REASONABLE_AR, source_aspect))
            
            # Calculate aspect-ratio preserving destination rect (pillarbox OR letterbox, never both)
            dest_rect = content_rect
            
            # Skip aspect ratio adjustment for docking overlays to prevent double boxing
            # Docking overlays handle aspect ratio at geometry level in sync_overlay_properties()
            is_docking_overlay = self._is_docking_overlay()
            
            if not is_docking_overlay and source_aspect is not None and source_aspect > 0:
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
            elif is_docking_overlay:
                # Reduce debug spam - only log once per overlay instance
                if not hasattr(self, '_logged_docking_detection'):
                    self._logger.debug("Docking overlay detected: skipping thumbnail-level aspect ratio adjustment")
                    self._logged_docking_detection = True
            
            # Determine current visual opacity (animation override takes precedence)
            current_opacity = self._visual_opacity if self._visual_opacity is not None else self._opacity

            # Apply DWM thumbnail clipping with proper z-order handling
            if self._canvas and hasattr(self._canvas, '_border_metrics') and self._canvas._border_metrics:
                border_metrics = self._canvas._border_metrics
                corner_radius = border_metrics.corner_radius
                if corner_radius > 0:
                    # Docking overlays already compute tight content_rect on the canvas (with border/accent margins).
                    # Avoid a second inset here to prevent excessive blank space around content.
                    if not is_docking_overlay:
                        # Non-docking overlays: apply a small safety inset to prevent overlap/escape
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
                        opacity=current_opacity,
                        visible=self._is_visible,
                        source_client_area_only=False,  # Capture full window to fix modern Windows apps
                    )
                    
                    # Suppress thumbnail property debug spam - only log during initialization or errors
                    if not hasattr(self, '_thumbnail_props_logged') or getattr(self, '_force_log_thumbnail_props', False):
                        aspect_str = f"{source_aspect:.3f}" if source_aspect is not None else "None"
                        self._logger.debug(
                            f"Updated thumbnail properties with aspect ratio: "
                            f"source_aspect={aspect_str}, logical_rect={dest_rect.getRect()}, "
                            f"physical_rect=({left_px},{top_px},{right_px},{bottom_px}), "
                            f"dpi_scale={dpi_scale:.2f}, opacity={self._opacity}"
                        )
                        self._thumbnail_props_logged = True
                        self._force_log_thumbnail_props = False
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
                opacity=current_opacity,
                visible=self._is_visible,
                source_client_area_only=False,  # Capture full window to fix modern Windows apps
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
            # Suppress debug spam - only log failures
            if not success:
                self._logger.debug("DWM post-update focus indicator z-order failed")
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
                aspect = max(self.MIN_REASONABLE_AR, min(self.MAX_REASONABLE_AR, aspect))  # Between 2.5:1 portrait and 3:1 ultrawide
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

    def _apply_persisted_geometry(self) -> bool:
        """Restore last saved nearest-corner geometry if available (standalone DWM only).
        Returns True if applied, else False.
        """
        try:
            self._logger.debug("[DWM_PERSIST] Checking for persisted geometry")
            
            if self._is_docking_overlay():
                self._logger.debug("[DWM_PERSIST] Skipping - detected as docking overlay")
                return False
                
            if not self._host:
                self._logger.debug("[DWM_PERSIST] Skipping - no host available")
                return False
                
            settings = get_settings_manager()
            # Standalone DWM overlays use overlays.dwm.last_state
            state = settings.get("overlays.dwm.last_state", None)
            self._logger.debug("[DWM_PERSIST] Checking overlays.dwm.last_state for standalone overlay")
            
            if not isinstance(state, dict):
                self._logger.debug(f"[DWM_PERSIST] No valid state data found - got {type(state)}")
                return False
                
            if not state:
                self._logger.debug("[DWM_PERSIST] State data is empty")
                return False
                
            self._logger.debug(f"[DWM_PERSIST] Found state data: {state}")
            mon_idx = int(state.get("monitor_index", 0))
            avail = get_available_geometry_for_monitor(mon_idx)
            # Gather aspect and insets similar to initial geometry
            min_size = QSize(OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT)
            # Aspect
            src_w, src_h = DEFAULT_ASPECT
            try:
                if self._source_hwnd:
                    rect = winval.get_window_rect(self._source_hwnd)
                    if rect and len(rect) >= 4 and rect[2] > rect[0] and rect[3] > rect[1]:
                        src_w, src_h = int(rect[2] - rect[0]), int(rect[3] - rect[1])
            except Exception:
                pass
            aspect = (src_w / src_h) if src_h else (DEFAULT_ASPECT[0] / DEFAULT_ASPECT[1])
            # Insets from canvas
            ix = iy = 0
            try:
                if self._canvas and hasattr(self._canvas, 'get_content_insets'):
                    cx, cy = self._canvas.get_content_insets()
                    ix = max(0, int(cx))
                    iy = max(0, int(cy))
            except Exception:
                ix = iy = 0

            rect_out = geometry_from_state(state, avail, min_size, aspect=aspect, insets=(ix, iy))
            if rect_out is None:
                return False
            # Clamp within available desktop to be safe
            pos = ensure_within_available_desktop(rect_out.topLeft(), rect_out.size())
            self._host.setGeometry(QRect(pos, rect_out.size()))
            # Success log for diagnostics
            try:
                self._logger.info("[DWM_PERSIST] Applied persisted DWM geometry")
            except Exception:
                pass
            return True
        except Exception as e:
            self._logger.debug(f"Apply persisted geometry failed: {e}")
            return False

    # Note: _persist_current_geometry is defined later in the file with logging and docking guard

    def _handle_quit_application(self) -> None:
        """Quit the application cleanly (wired from context menu)."""
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.quit()
            else:
                self._logger.error("Quit requested but no QApplication instance available")
        except Exception as e:
            self._logger.error(f"Error quitting application: {e}")

    def _handle_reset_position(self) -> None:
        """Reset overlay geometry to default top-left and default sizing.

        This ignores any persisted geometry and re-applies the initial geometry logic,
        including AR and canvas insets. This is the safety reset used by context menus.
        """
        try:
            if not self._host:
                return
            # Apply initial default geometry (top-left of primary, default sizing with AR)
            self._apply_initial_geometry()
            # Update thumbnail properties to reflect the change
            self._update_thumbnail_properties()
            self._logger.info("Overlay position reset to default (top-left, default size)")
        except Exception as e:
            self._logger.error(f"Reset position failed: {e}")

    def _handle_correct_aspect(self) -> None:
        """Correct aspect ratio/framing and adjust borders to tightly fit content.

        This refreshes the cached source aspect, resizes the overlay window to
        eliminate gaps between content and borders (tight framing), updates
        thumbnail properties, and reapplies window masking.
        """
        try:
            if not self._host:
                return
            
            # Refresh source aspect and propagate to canvas if available
            try:
                self._cache_source_aspect()
            except Exception:
                pass
            
            # For docking overlays, trigger a full sync to recompute tight sizing
            if self._is_docking_overlay():
                try:
                    # Get the docking manager and trigger a full sync
                    from core.graphics.docking.manager import DockingOverlayManager
                    import gc
                    for obj in gc.get_objects():
                        if isinstance(obj, DockingOverlayManager):
                            if self._is_managed_by_docking(obj):
                                # Trigger sync which will recompute all overlay sizes with tight framing
                                obj.sync_overlay_properties()
                                self._logger.info("[AR] Corrected aspect/framing for docking system (triggered full sync)")
                                return
                except Exception as e:
                    self._logger.debug(f"Docking sync failed, falling back to standalone logic: {e}")
            
            # For standalone overlays: resize to tightly fit content
            try:
                if self._canvas and self._source_aspect and self._source_aspect > 0:
                    # Get canvas insets (border + corner + accent margins)
                    ix, iy = 0, 0
                    if hasattr(self._canvas, 'get_content_insets'):
                        ix, iy = self._canvas.get_content_insets()
                    
                    # Get current screen for bounds checking
                    screen = self._host.screen()
                    if screen:
                        screen_rect = screen.availableGeometry()
                        max_w = screen_rect.width()
                        max_h = screen_rect.height()
                        
                        # Start with current inner content dimensions
                        current_outer = self._host.geometry()
                        current_inner_w = max(1, current_outer.width() - 2 * ix)
                        
                        # Calculate ideal inner dimensions that respect the aspect ratio
                        # We keep the width and adjust height, or vice versa
                        target_inner_w = current_inner_w
                        target_inner_h = max(1, int(target_inner_w / self._source_aspect))
                        
                        # If resulting height exceeds screen, constrain by height instead
                        potential_outer_h = target_inner_h + 2 * iy
                        if potential_outer_h > max_h:
                            target_inner_h = max(1, max_h - 2 * iy)
                            target_inner_w = max(1, int(target_inner_h * self._source_aspect))
                        
                        # Convert inner to outer by adding insets
                        target_w = target_inner_w + 2 * ix
                        target_h = target_inner_h + 2 * iy
                        
                        # Apply minimum size constraints
                        from utils.window.overlay_constants import OVERLAY_MIN_WIDTH, OVERLAY_MIN_HEIGHT
                        target_w = max(target_w, OVERLAY_MIN_WIDTH)
                        target_h = max(target_h, OVERLAY_MIN_HEIGHT)
                        
                        # Clamp to screen bounds
                        target_w = min(target_w, max_w)
                        target_h = min(target_h, max_h)
                        
                        # Resize the host (preserves position)
                        self._host.resize(target_w, target_h)
                        self._logger.info(f"[AR] Adjusted borders to fit content: {target_w}x{target_h} (inner: {target_inner_w}x{target_inner_h}, insets: {ix},{iy})")
            except Exception as e:
                self._logger.debug(f"Border adjustment failed, updating properties only: {e}")
            
            # Recompute thumbnail destination rects with current canvas/content aspect
            self._update_thumbnail_properties()
            
            # Ensure border mask aligns with current geometry/aspect
            try:
                self._apply_window_masking()
            except Exception:
                pass
            
        except Exception as e:
            self._logger.error(f"Correct AR failed: {e}")

    def _on_host_geometry_changed(self) -> None:
        """Coalesce geometry changes if needed. We avoid frequent writes; actual persist happens on hide/close."""
        # Intentionally no-op to minimize I/O; placeholder for future heuristics.
        return

    def _persist_current_geometry(self) -> None:
        """Persist current geometry to settings for standalone DWM overlays.

        Writes nearest-corner state to key 'overlays.dwm.last_state'. Docking overlays
        are excluded by design and should rely on DockingOverlayManager persistence.
        """
        try:
            # Do not persist for docking overlays; manager owns persistence there
            try:
                if self._is_docking_overlay():
                    return
            except Exception:
                pass
            if not self._host:
                return
            rect = self._host.geometry()
            if not rect or rect.isEmpty():
                return
            state = nearest_corner_state_from_rect(rect)
            if not state:
                return
            sm = get_settings_manager()
            sm.set("overlays.dwm.last_state", state)
            try:
                self._logger.info("[DWM_PERSIST] Saved DWM geometry")
            except Exception:
                pass
        except Exception as e:
            self._logger.debug(f"[DWM_PERSIST] Save failed: {e}")

    def _setup_context_menu(self) -> None:
        """Set up context menu for the integrated overlay."""
        try:
            # Check if this overlay is being used in docking mode
            # For docking mode, let docking overlays handle their own context menu setup
            # Don't block context menu creation entirely - just the standard OverlayContextMenu
            if self._is_docking_overlay():
                self._logger.debug("Docking mode detected - docking overlays will handle context menu setup")
                return
                
            # Use the existing context menu system for regular DWM overlays
            from utils.overlay_context_menu import OverlayContextMenu

            # Provide callbacks to enable docking mode switching from context menu
            actions = {
                'switch_to_docking': getattr(self, '_switch_to_docking', None),
                'switch_to_docking_normal': getattr(self, '_switch_to_docking_normal', None),
                'switch_to_docking_cycle': getattr(self, '_switch_to_docking_cycle', None),
            }
            config = {
                'actions': {k: v for k, v in actions.items() if callable(v)}
            }

            # Create context menu handler (integrated border approach)
            self._context_menu = OverlayContextMenu(
                self._host,
                overlay_type='dwm',
                config=config
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

    # --- Context menu callbacks: switch to docking from standalone DWM overlay ---
    def _switch_to_docking(self) -> None:
        """Switch to docking mode preserving current source window and current mode setting."""
        self._switch_to_docking_impl(None)

    def _switch_to_docking_normal(self) -> None:
        """Switch to docking mode and set docking.mode to 'normal' before creating."""
        self._switch_to_docking_impl('normal')

    def _switch_to_docking_cycle(self) -> None:
        """Switch to docking mode and set docking.mode to 'cycle' before creating."""
        self._switch_to_docking_impl('cycle')

    def _switch_to_docking_impl(self, mode: Optional[str]) -> None:
        try:
            hwnd = int(self.get_source_hwnd() or 0)
            if not hwnd:
                self._logger.warning("Switch to docking aborted: no source hwnd")
                return

            # Optionally set desired docking mode
            if mode in ('normal', 'cycle'):
                try:
                    get_settings_manager().set('docking.mode', mode)
                except Exception as e:
                    self._logger.debug(f"Failed to set docking.mode to {mode}: {e}")

            # Destroy existing single overlay(s) cleanly via OverlayManager
            try:
                from core.graphics.overlay_manager import OverlayManager
                OverlayManager().clear_all()
            except Exception as e:
                self._logger.debug(f"Failed to clear existing overlays before docking switch: {e}")

            # Create docking system targeting current window
            try:
                from core.graphics.docking.manager import DockingOverlayManager
                dom = DockingOverlayManager()
                ok = dom.create_docking_system([hwnd])
                if not ok:
                    self._logger.error("Failed to create docking overlay system from context menu")
            except Exception as e:
                self._logger.error(f"Docking creation error: {e}")
        except Exception as e:
            self._logger.error(f"Unhandled error switching to docking: {e}")

    def _is_docking_overlay(self) -> bool:
        """Check if this DWM overlay is being used in docking mode.
        
        Returns:
            bool: True if this is a docking mode overlay, False otherwise
        """
        try:
            # Fast-path: detect via config overlay_type to avoid expensive scans and false negatives
            try:
                overlay_type = getattr(self._config, 'overlay_type', None)
                if overlay_type is not None:
                    # Compare by name to avoid enum import cycles
                    name = str(getattr(overlay_type, 'name', overlay_type)).upper()
                    if name == 'DOCKING':
                        return True
            except Exception:
                pass
            
            # Check if THIS specific overlay is actually managed by a docking system
            from core.graphics.docking.manager import DockingOverlayManager
            
            # Try to access the global docking manager instance
            try:
                from core.application.core import get_app_core
                app_core = get_app_core()
                if app_core and hasattr(app_core, '_docking_manager'):
                    docking_manager = getattr(app_core, '_docking_manager', None)
                    if docking_manager and self._is_managed_by_docking(docking_manager):
                        return True
            except Exception:
                pass
            
            # Enhanced fallback: check if any DockingOverlayManager manages this overlay
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, DockingOverlayManager):
                    if self._is_managed_by_docking(obj):
                        return True
            
            return False
        except Exception as e:
            self._logger.debug(f"Docking detection failed: {e}")
            return False

    def _is_managed_by_docking(self, docking_manager) -> bool:
        """Check if this overlay is managed by the given docking manager."""
        try:
            if not docking_manager:
                return False
            
            # First check if this overlay is actually in the docking manager's collection
            overlays = getattr(docking_manager, '_overlays', {})
            if not overlays:
                return False
            
            # Look for this DWM overlay within the docking overlay wrappers
            found_in_collection = False
            for docking_overlay in overlays.values():
                if hasattr(docking_overlay, '_dwm_overlay') and docking_overlay._dwm_overlay is self:
                    found_in_collection = True
                    break
            
            if not found_in_collection:
                return False
            
            # Only return True if this overlay is in the collection AND the docking manager is active
            is_active = getattr(docking_manager, '_is_active', False)
            is_initializing = getattr(docking_manager, '_is_initializing', False)
            
            if is_active or is_initializing:
                # Only log once per overlay instance to prevent debug spam
                if not hasattr(self, '_logged_docking_detection'):
                    self._logger.debug("Detected as docking-managed overlay")
                    self._logged_docking_detection = True
                return True
            else:
                # Found in collection but docking is not active - this is a standalone overlay
                if not hasattr(self, '_logged_standalone_detection'):
                    self._logger.debug("Found in docking collection but docking not active - treating as standalone")
                    self._logged_standalone_detection = True
                return False
            
        except Exception:
            return False

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

    # --- Overlay base hooks (required by OverlayBase) ---
    def _render_impl(self) -> None:
        """DWM integrates rendering via thumbnails; nothing to render per-frame."""
        return

    def _config_updated(self, old_config: dict, new_config: dict) -> None:
        """Apply configuration changes to the running overlay (opacity, geometry, source)."""
        try:
            # Opacity changes
            try:
                old_op = float(old_config.get("opacity", 1.0) or 1.0)
                new_op = float(new_config.get("opacity", 1.0) or 1.0)
            except Exception:
                old_op = 1.0
                new_op = 1.0
            if new_op != old_op:
                self.set_opacity(new_op)

            # Geometry updates
            if self._host is not None:
                pos_changed = old_config.get("position") != new_config.get("position")
                size_changed = old_config.get("size") != new_config.get("size")
                if pos_changed or size_changed:
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
                self._handle_swap_window(new_hwnd)
        except Exception as e:
            self._logger.error(f"Config update failed: {e}")
            raise

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

        
            # Apply persisted geometry on show for standalone DWM overlays (hide/restore)
            try:
                if not self._is_docking_overlay():
                    self._apply_persisted_geometry()
            except Exception:
                pass

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

            # Prepare zero-opacity state and keep invisible until after first property push
            try:
                self._visual_opacity = 0.0
                if self._canvas:
                    self._canvas.set_opacity(0.0)
            except Exception:
                pass
            self._is_visible = False

            # Attempt immediate registration; if it fails, schedule background retries without failing show
            if not self._register_thumbnail():
                self._logger.warning(
                    "Initial DWM registration failed after show; scheduling deferred retries"
                )
                ThreadManager.single_shot(20, lambda: self._register_after_show_retry(1))
            else:
                # Push initial properties at 0 opacity while invisible
                self._update_thumbnail_properties()
                # Now mark visible and start fade-in (800ms)
                self._is_visible = True
                self._update_thumbnail_properties()
                if not self._has_revealed:
                    self._fade_in_thumbnail(800)
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
            # Keep invisible while pushing first properties at 0 opacity
            self._is_visible = False
            try:
                self._visual_opacity = 0.0
                if self._canvas:
                    self._canvas.set_opacity(0.0)
            except Exception:
                pass
            self._update_thumbnail_properties()
            # Now show and fade in at 800ms
            self._is_visible = True
            self._update_thumbnail_properties()
            if not self._has_revealed:
                self._fade_in_thumbnail(800)
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

            # Persist geometry sparingly on hide (standalone DWM only)
            try:
                self._persist_current_geometry()
            except Exception:
                pass
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
            clamped = max(0.01, min(1.0, raw))
            if clamped != raw:
                try:
                    self._logger.debug(f"[OPACITY] DWM clamp -> {clamped:.2f} (from {raw:.3f})")
                except Exception:
                    pass
            # Log at bounds for validation
            if clamped in (0.01, 1.0):
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
        """Trigger fade-in animation for the thumbnail using visual opacity override."""
        try:
            if not self._thumbnail_manager or not self._thumbnail_id:
                return

            # 60 FPS fade
            frame_time = 16
            max_steps = max(1, int(duration_ms // frame_time))

            def fade_step(step: int):
                if step >= max_steps:
                    # End animation: clear override and commit final opacity
                    self._visual_opacity = None
                    if self._canvas:
                        self._canvas.set_opacity(self._opacity)
                    self._update_thumbnail_properties()
                    return

                # Ease-out curve
                progress = step / max_steps
                eased = 1 - (1 - progress) ** 2
                current = self._opacity * eased

                # Apply to canvas and thumbnail
                self._visual_opacity = current
                if self._canvas:
                    self._canvas.set_opacity(current)
                self._update_thumbnail_properties()

                ThreadManager.single_shot(frame_time, lambda: fade_step(step + 1))

            # Start from 0
            self._visual_opacity = 0.0
            if self._canvas:
                self._canvas.set_opacity(0.0)
            self._update_thumbnail_properties()
            fade_step(0)
        except Exception as e:
            self._logger.error(f"Fade-in failed: {e}")

    def _handle_swap_window(self, new_hwnd: int, record_mru: bool = False) -> None:
        """Dispatch a source swap with simple in-flight queuing and UI-thread scheduling."""
        try:
            # Queue if a swap is already running
            if self._swap_in_flight:
                self._pending_swap_hwnd = new_hwnd
                self._pending_swap_record_mru = record_mru
                self._logger.debug(f"Queued pending swap to hwnd {new_hwnd}")
                return

            # Mark in-flight and schedule the actual swap on UI thread
            self._swap_in_flight = True
            ThreadManager.run_on_ui_thread(lambda: self._swap_source_hwnd(new_hwnd, record_mru))
        except Exception as e:
            # Ensure flag is cleared on dispatch failure
            self._swap_in_flight = False
            self._logger.error(f"Swap dispatch failed: {e}")

    def _swap_source_hwnd(self, new_hwnd: int, record_mru: bool = False) -> bool:
        """Swap the source window handle with a short crossfade to reduce visual artifacts."""
        self._logger.debug(
            f"_swap_source_hwnd start: new_hwnd={new_hwnd} target_hwnd={self._target_hwnd} visible={self._is_visible}"
        )
        success = False
        try:
            if not winval.is_valid_window(new_hwnd):
                self._logger.error(f"Invalid swap target: {new_hwnd}")
                return False

            # Skip redundant swap to same source to prevent unnecessary flicker
            try:
                if int(self._source_hwnd or 0) == int(new_hwnd or 0):
                    self._logger.debug("Swap requested to same hwnd; skipping re-register")
                    return True
            except Exception:
                pass

            # Reset reveal state for potential fade
            self._has_revealed = False

            # If overlay not ready/visible, just set source and bail; registration happens on next show
            if not self._host or not getattr(self._host, 'isVisible', lambda: False)() or not self._target_hwnd:
                self._source_hwnd = new_hwnd
                self._cache_source_aspect()
                self._logger.debug("Overlay hidden or target HWND not ready; deferred DWM registration until visible")
                return True

            # Quick fade-out to mask swap (no wait; single frame)
            try:
                self._visual_opacity = 0.0
                if self._canvas:
                    self._canvas.set_opacity(0.0)
                self._update_thumbnail_properties()
            except Exception:
                pass

            start_ts = time.perf_counter()
            # Unregister previous thumbnail
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

            # Update source
            self._source_hwnd = new_hwnd
            self._logger.debug(f"Updated source hwnd to {self._source_hwnd}")
            self._cache_source_aspect()

            # Register new thumbnail
            if not self._register_thumbnail():
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
                self._broken = True
                self._needs_recreate = True
                success = False
            else:
                # Push initial props for new source and fade in
                self._update_thumbnail_properties()
                try:
                    if self._host:
                        self._host.update()
                except Exception:
                    pass
                if self._is_visible:
                    dur_ms = int((time.perf_counter() - start_ts) * 1000.0)
                    # Pad fade slightly for longer backend operations
                    fade_ms = 250 if dur_ms < 120 else (350 if dur_ms < 300 else 500)
                    self._fade_in_thumbnail(fade_ms)
                    self._has_revealed = True
                self._logger.info(f"Swapped to window {new_hwnd}")
                self._broken = False
                self._needs_recreate = False
                success = True

                if record_mru:
                    try:
                        from core.switching.mru_manager import get_mru_manager
                        get_mru_manager().record(new_hwnd)
                        self._logger.debug(f"Added hwnd {new_hwnd} to MRU from context menu swap")
                    except Exception as e:
                        self._logger.debug(f"Failed to add context menu swap to MRU: {e}")

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
    def get_geometry(self) -> QRect:
        """Get the current geometry of the overlay."""
        try:
            if self._host:
                return self._host.geometry()
            return QRect()
        except Exception as e:
            self._logger.debug(f"Error getting geometry: {e}")
            return QRect()
    
    def get_opacity(self) -> float:
        """Get the current opacity of the overlay."""
        return self._opacity
    
    def update_source(self, hwnd: int) -> bool:
        """Public API used by OverlayManager/Docking to update the source window.

        Validates and delegates to the unified swap handler which schedules on the UI thread.
        Returns True if the request was accepted.
        """
        try:
            new_hwnd = int(hwnd) if hwnd is not None else 0
        except Exception:
            new_hwnd = 0

        if not new_hwnd or not winval.is_valid_window(new_hwnd):
            self._logger.error(f"update_source rejected invalid hwnd: {hwnd}")
            return False

        # Delegate to unified swap path (includes redundant-swap guard and crossfade)
        self._handle_swap_window(new_hwnd)
        return True
    
    def set_geometry(self, x: int, y: int, width: int, height: int) -> None:
        """Set the geometry of the overlay with x, y, width, height parameters."""
        try:
            if self._host:
                rect = QRect(x, y, width, height)
                self._host.setGeometry(rect)
        except Exception as e:
            self._logger.error(f"Error setting geometry: {e}")
    
    def set_geometry_with_letterbox(self, x: int, y: int, width: int, height: int, letterbox_type: str) -> None:
        """Set geometry with letterboxing/pillarboxing for aspect ratio compliance."""
        try:
            if self._host:
                self._host.setGeometry(x, y, width, height)
                # Update thumbnail properties after geometry change
                self._update_thumbnail_properties()
                self._logger.debug(f"Applied {letterbox_type}boxing geometry: {x},{y} {width}x{height}")
        except Exception as e:
            self._logger.error(f"Error setting geometry with letterbox: {e}")

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
