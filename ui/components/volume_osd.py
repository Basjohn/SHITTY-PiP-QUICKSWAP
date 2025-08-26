from __future__ import annotations

from typing import Optional, Any, Dict

from PySide6.QtCore import Qt, QRect, QSize, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QWidget

from core.application.core import get_app_core
from core.logging import get_logger
from core.threading import ThreadManager
from utils.resource_manager import get_resource_manager, ResourceType


class VolumeOSDWidget(QWidget):
    """
    Centralized Volume OSD widget.

    - Subscribes to 'media.volume.changed' events.
    - Coalesces and rate-limits UI updates via ThreadManager.create_ui_coalescer.
    - Registers itself with the ResourceManager for lifecycle tracking.
    - Thread-safe. UI updates always on the UI thread.

    Visuals: a compact rounded bar with app name and percentage. Styling is
    intentionally minimal; further theming can be done via QSS using objectName 'volumeOSD'.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("volumeOSD")
        # Frameless child widget by default; host can choose to reparent into an overlay
        self.setWindowFlags(Qt.Widget)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

        # State
        self._logger = get_logger("VolumeOSD")
        self._app_name: str = ""
        self._hwnd: Optional[int] = None
        self._volume: float = 0.0  # 0.0 - 1.0
        self._source: str = ""
        self._reason: str = ""
        self._visible_ms: int = 900
        self._hide_token: int = 0
        # When volume level is unavailable, show a textual indicator
        self._text_override: str = ""

        # Sizing
        self._width_px = 260
        self._height_px = 36
        self.resize(self.sizeHint())
        self.hide()  # start hidden

        # Core services
        app = get_app_core()
        self._events = app.events
        self._threads = app.threads
        self._rm = get_resource_manager()
        self._settings = app.settings
        # Diagnostics
        try:
            self._logger.info("VolumeOSDWidget initialized (subscribing to media.volume.changed)")
        except Exception:
            pass

        # UI coalescer (registered with RM by factory)
        self._coalescer = self._threads.create_ui_coalescer(
            name="VolumeOSD",
            capacity=32,
            mode="latest",
            window_ms=25,
        )

        # Subscribe to volume events (use string event type to ensure matching)
        self._subscription_id: Optional[str] = None
        self._first_event_logged: bool = False
        try:
            self._subscription_id = self._events.subscribe(
                "media.volume.changed",
                self._on_volume_event,
                priority=0,
                filter_fn=None,
                dispatch_on_ui=True,  # ensure handler runs on UI thread
            )
            self._vlog("Subscribed to 'media.volume.changed'")
        except Exception as e:
            self._logger.error(f"Failed to subscribe to media.volume.changed: {e}")

        # Register this widget in ResourceManager for lifecycle and cleanup
        try:
            self._resource_id = self._rm.register(
                self,
                ResourceType.GUI_COMPONENT,
                "Volume OSD widget",
                cleanup_handler=lambda w: w._cleanup(),
                tags={"ui", "volume", "osd"},
                created_by="VolumeOSDWidget",
            )
        except Exception as e:
            self._logger.debug(f"ResourceManager.register failed for VolumeOSD: {e}")
            self._resource_id = None

    # --- QWidget overrides ----------------------------------------------------
    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._width_px, self._height_px)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect()
        # Background
        bg = QColor("#000000")
        bg.setAlpha(160)
        painter.setBrush(QBrush(bg))
        painter.setPen(Qt.NoPen)
        radius = 8
        painter.drawRoundedRect(rect, radius, radius)

        # Volume fill bar
        inner = rect.adjusted(10, 10, -10, -10)
        track_color = QColor("#666666")
        track_color.setAlpha(180)
        painter.setBrush(QBrush(track_color))
        painter.drawRoundedRect(inner, 5, 5)

        vol = max(0.0, min(1.0, float(self._volume)))
        fill_w = int(inner.width() * vol)
        if fill_w > 0:
            fill_rect = QRect(inner.x(), inner.y(), fill_w, inner.height())
            fill_color = QColor("#000000")  # black fill
            fill_color.setAlpha(230)  # ~90% opacity
            painter.setBrush(QBrush(fill_color))
            painter.drawRoundedRect(fill_rect, 5, 5)

        # Text: app name and percent
        painter.setPen(QPen(QColor("#FFFFFF")))
        font = painter.font()
        font.setPointSizeF(max(8.0, font.pointSizeF()))
        font.setBold(True)
        painter.setFont(font)
        percent = int(round(vol * 100))
        label = ((self._app_name or "")[:24]).upper()
        if self._text_override:
            text = f"{label}  {self._text_override}".strip().upper()
        else:
            text = f"{label}  {percent}%".strip().upper()
        # Place text centered vertically, with left padding
        painter.drawText(inner.adjusted(8, 0, -8, 0), Qt.TextSingleLine, text)

    # --- Public API -----------------------------------------------------------
    def update_position(self, target_rect: Optional[QRect] = None) -> None:
        """Position within parent (if any): bottom-center with margin.

        If no parent, does nothing; hosting overlay/window should manage position.
        """
        p = self.parentWidget()
        if not p:
            return
        rect = target_rect if target_rect is not None else p.rect()
        w, h = self.width(), self.height()
        x = rect.x() + (rect.width() - w) // 2
        y = rect.y() + rect.height() - h - 12
        self.move(QPoint(max(0, x), max(0, y)))
        self.raise_()

    # --- Internal -------------------------------------------------------------
    def _on_volume_event(self, evt) -> None:
        """Event handler for 'media.volume.changed'. Always runs on UI thread."""
        try:
            data: Dict[str, Any] = evt.data or {}
            self._hwnd = data.get("hwnd")
            self._app_name = data.get("app_name") or data.get("session") or ""
            # Volume can be provided as 'volume' or 'level' in 0.0-1.0 or 0-100 scales
            raw = data.get("volume")
            if raw is None:
                raw = data.get("level")
            has_level = isinstance(raw, (int, float))
            if has_level:
                v = float(raw)
                self._volume = v / 100.0 if v > 1.0 else v
                self._text_override = ""
            else:
                # No level available, fall back to textual indicator based on reason
                self._volume = 0.0
                reason = str(data.get("reason") or "").lower()
                if reason == "up":
                    self._text_override = "VOLUME UP"
                elif reason == "down":
                    self._text_override = "VOLUME DOWN"
                else:
                    self._text_override = "VOLUME"
            self._source = str(data.get("source") or "")
            self._reason = str(data.get("reason") or "")
            self._vlog(
                f"evt volume_changed: hwnd={self._hwnd} app='{self._app_name}' vol={self._volume:.3f} src={self._source} reason={self._reason} text='{self._text_override}'"
            )
            # One-time info log to confirm we're receiving events without needing verbose setting
            if not self._first_event_logged:
                try:
                    self._logger.info(
                        f"VolumeOSD received first event: app='{self._app_name}' vol={self._volume:.2f} src={self._source} reason={self._reason} text='{self._text_override}'"
                    )
                except Exception:
                    pass
                self._first_event_logged = True
        except Exception as e:
            self._logger.error(f"Invalid media.volume.changed payload: {e}")
            return

        # Coalesce UI updates
        self._coalescer.submit(self._apply_state_and_repaint)

        # Show and schedule hide
        self._ensure_visible_then_hide()

    def _apply_state_and_repaint(self) -> None:
        try:
            if not self.isVisible():
                self._vlog("apply_state: showing OSD and updating position")
                self.show()
                self.raise_()
                # If hosted as child, keep to designated position
                self.update_position()
            else:
                self._vlog("apply_state: updating OSD")
            self.update()
        except Exception as e:
            self._logger.debug(f"apply_state repaint failed: {e}")

    def _ensure_visible_then_hide(self) -> None:
        try:
            # Show immediately on UI thread
            ThreadManager.run_on_ui_thread(self._show_now)
            # Schedule hide with token to avoid races
            self._hide_token += 1
            my_tok = int(self._hide_token)
            self._vlog(f"schedule_hide: token={my_tok} delay_ms={self._visible_ms}")
            def _hide_if_token_matches(tok: int) -> None:
                if self._hide_token == tok:
                    self._vlog(f"hide_now: token={tok} matched")
                    self.hide()
            ThreadManager.single_shot(max(0, int(self._visible_ms)), lambda: _hide_if_token_matches(my_tok))
        except Exception as e:
            self._logger.debug(f"ensure_visible_then_hide failed: {e}")

    def _show_now(self) -> None:
        try:
            if not self.isVisible():
                self._vlog("show_now: showing and positioning OSD")
                self.show()
                self.raise_()
                self.update_position()
            else:
                # Keep on top of siblings
                self._vlog("show_now: already visible; raise_ only")
                self.raise_()
        except Exception:
            pass

    def _vlog(self, msg: str) -> None:
        """Verbose logging gated by 'debug.volume_osd_verbose' setting."""
        try:
            if bool(self._settings.get("debug.volume_osd_verbose", False)):
                self._logger.debug(msg)
        except Exception:
            # Avoid any settings errors impacting UI flow
            pass

    def _cleanup(self) -> None:
        """Cleanup resources and unsubscribe. Safe to call multiple times."""
        # Unsubscribe from events
        try:
            if getattr(self, "_subscription_id", None):
                self._events.unsubscribe(self._subscription_id)  # type: ignore[arg-type]
                self._subscription_id = None
        except Exception as e:
            self._logger.debug(f"unsubscribe failed: {e}")
        # Stop coalescer
        try:
            if getattr(self, "_coalescer", None):
                self._coalescer.shutdown()
        except Exception:
            pass
        # Hide and schedule deletion
        try:
            self.hide()
            self.deleteLater()
        except Exception:
            pass


class VolumeOSDWindow(VolumeOSDWidget):
    """Top-level Volume OSD window layered above the overlay host.

    - Owns a native HWND with per-pixel alpha and no activation.
    - Positioned in SCREEN coordinates relative to the overlay host.
    - Z-order is enforced via ResourceManager to sit above the host without global topmost.
    """

    def __init__(self, host_widget: QWidget) -> None:  # type: ignore[name-defined]
        super().__init__(parent=None)
        self._host_ref: Optional[QWidget] = host_widget

        # Convert to top-level tool window with no activation and frameless
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowDoesNotAcceptFocus
            | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NativeWindow, True)

        self.hide()

    def update_position(self, target_rect: Optional[QRect] = None) -> None:  # noqa: N802
        """Position bottom-center relative to host in SCREEN coordinates and enforce z-order."""
        host = getattr(self, "_host_ref", None)
        if host is None:
            return

        # Determine host content rect
        rect = target_rect if target_rect is not None else host.rect()

        # Compute local position: bottom-center with 12px margin
        w, h = self.width(), self.height()
        x_local = (rect.width() - w) // 2
        y_local = rect.height() - h - 12
        if rect.width() - w < 0:
            x_local = 0
        if rect.height() - h < 0:
            y_local = 0

        # Map to global screen coordinates
        try:
            top_left = host.mapToGlobal(QPoint(0, 0))
            self.move(top_left + QPoint(max(0, x_local), max(0, y_local)))
        except Exception:
            # Best-effort: keep previous position on failure
            pass

        # Keep window raised and directly above the host
        try:
            self.raise_()
            rm = get_resource_manager()
            rm.place_window_above(self, host)
        except Exception:
            pass

    def _show_now(self) -> None:
        """Override to ensure z-order is enforced when showing the window."""
        try:
            if not self.isVisible():
                self._vlog("show_now(window): showing and positioning OSD")
                self.show()
                self.raise_()
                self.update_position()
            else:
                self._vlog("show_now(window): already visible; raise_ and place_above")
                self.raise_()
                self.update_position()
        except Exception:
            pass


__all__ = ["VolumeOSDWidget", "VolumeOSDWindow"]
