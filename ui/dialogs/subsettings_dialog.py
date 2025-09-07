from ui.components.circle_checkbox import CircleCheckBox
from ui.dialogs.keypassthrough_warning_dialog import KeyPassthroughWarningDialog
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QKeySequenceEdit, QFrame, QCheckBox, QPushButton, QScrollArea, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QMouseEvent
from core.settings import SettingsManager
from core.opacity.manager import OpacityManager
from utils.window.behavior import WindowBehaviorManager
from core.logging import get_logger
logger = get_logger(__name__)

class DoubleClickCheckBox(QCheckBox):
    """
    Minimal placeholder for DoubleClickCheckBox to satisfy package exports.
    Extend with double-click logic and QSS styling as needed.
    """
    def __init__(self, label: str = "", parent=None):
        super().__init__(label, parent)
        self.setObjectName("DoubleClickCheckBox")  # For QSS styling


class SubSettingsDialog(QDialog):
    """
    Canonical subsettings dialog with custom title bar, QSS-only visuals, and full hotkey/theming controls.
    """
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.settings_manager = SettingsManager()
        self.opacity_manager = OpacityManager()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        # Critical for rounded-corner transparency: allow per-pixel alpha
        # Without this, the outer rounded border will visually "bleed" as the
        # OS fills the window corners opaque. This must be set before showing
        # the dialog so QSS translucent backgrounds take effect.
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # Use a minimum size rather than fixed to avoid layout overlap at smaller DPIs
        self.setMinimumSize(380, 625)
        self.setObjectName("subsettingsDialog")
        self.setMouseTracking(True)
        
        # Initialize window behavior manager for centralized window behavior
        self.window_behavior = WindowBehaviorManager(self, 380, 520)
        
        self._setup_ui()
        # Only call _load_settings, _connect_signals, _apply_theme after widgets exist
        self._load_settings()
        self._connect_signals()
        # Theme is applied centrally by ThemeManager listening to SettingsManager

    def _setup_ui_titlebar(self):
        """
        Deprecated: legacy title bar-only setup. Kept for historical context; not used.
        """
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Custom Title Bar ---
        self.title_bar = QFrame()
        self.title_bar.setObjectName("titleFrame")
        self.title_bar.setFixedHeight(36)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(12, 0, 8, 0)
        title_layout.setSpacing(8)
        self.title_label = QLabel("SUBSETTINGS")
        self.title_label.setObjectName("titleLabel")
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        self.close_button = QPushButton("X")
        self.close_button.setObjectName("closeButton")
        # Match AboutDialog: no fixed size or custom cursor; styling via QSS
        self.close_button.clicked.connect(self.close)
        title_layout.addWidget(self.close_button)
        main_layout.addWidget(self.title_bar)

        # --- Mouse events: handled at dialog level by WindowBehaviorManager ---

    # --- Centralized window behavior for dragging ---
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if hasattr(self, 'window_behavior'):
            self.window_behavior.handle_mouse_press(event, self.is_draggable_region)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if hasattr(self, 'window_behavior'):
            self.window_behavior.handle_mouse_move(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if hasattr(self, 'window_behavior'):
            self.window_behavior.handle_mouse_release(event)
        else:
            super().mouseReleaseEvent(event)
            
    def leaveEvent(self, event):
        if hasattr(self, 'window_behavior'):
            self.window_behavior.handle_leave()
        super().leaveEvent(event)
        
    def is_draggable_region(self, pos):
        """Return True if position is in a non-interactive area.

        This makes the dialog draggable from any empty space, not only the title bar,
        while preserving normal interaction with inputs and controls.
        """
        w = self.childAt(pos)
        if w is None:
            return True
        # Protect explicit interactive elements from initiating drag
        from PySide6.QtWidgets import (
            QComboBox, QKeySequenceEdit, QCheckBox, QPushButton,
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QLabel,
        )
        # Treat our close label as non-draggable
        if isinstance(w, QLabel) and w.objectName() == "closeButton":
            return False
        if isinstance(w, (QComboBox, QKeySequenceEdit, QCheckBox, QPushButton, QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox)):
            return False
        # Otherwise allow dragging anywhere blank or on non-interactive widgets
        return True

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Custom Title Bar ---
        self.title_bar = QFrame()
        self.title_bar.setObjectName("titleFrame")
        self.title_bar.setFixedHeight(36)
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(12, 0, 8, 0)
        title_layout.setSpacing(8)
        self.title_label = QLabel("SUBSETTINGS")
        self.title_label.setObjectName("titleLabel")
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        self.close_button = QPushButton("X")
        self.close_button.setObjectName("closeButton")
        # Match AboutDialog: no fixed size or custom cursor; styling via QSS
        self.close_button.clicked.connect(self.close)
        title_layout.addWidget(self.close_button)
        main_layout.addWidget(self.title_bar)

        # --- Main Content wrapped in Scroll Area ---
        # Inner content widget holds the actual settings controls
        content_inner = QFrame()
        content_inner.setObjectName("settingsContentFrame")
        content_layout = QVBoxLayout(content_inner)
        content_layout.setSpacing(14)
        content_layout.setContentsMargins(18, 18, 18, 18)

        # Theme
        theme_label = QLabel("Theme:")
        theme_label.setObjectName("SubSettingsSectionLabel")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setObjectName("SettingsComboBox")
        content_layout.addWidget(theme_label)
        content_layout.addWidget(self.theme_combo)

        # Opacity hotkey section
        opacity_label = QLabel("Opacity Hotkeys:")
        opacity_label.setObjectName("SubSettingsSectionLabel")
        content_layout.addWidget(opacity_label)
        opacity_hotkey_layout = QHBoxLayout()
        self.opacity_hotkeys_checkbox = CircleCheckBox("Enable Opacity Hotkeys")
        self.opacity_hotkeys_checkbox.setObjectName("SettingsCheckBox")
        opacity_hotkey_layout.addWidget(self.opacity_hotkeys_checkbox)
        content_layout.addLayout(opacity_hotkey_layout)
        decrease_label = QLabel("Decrease Opacity:")
        decrease_label.setObjectName("SubSettingsSectionLabel")
        self.decrease_hotkey_edit = QKeySequenceEdit()
        self.decrease_hotkey_edit.setObjectName("SettingsKeySequenceEdit")
        content_layout.addWidget(decrease_label)
        content_layout.addWidget(self.decrease_hotkey_edit)
        increase_label = QLabel("Increase Opacity:")
        increase_label.setObjectName("SubSettingsSectionLabel")
        self.increase_hotkey_edit = QKeySequenceEdit()
        # Use unified object name for consistent QSS styling
        self.increase_hotkey_edit.setObjectName("SettingsKeySequenceEdit")
        content_layout.addWidget(increase_label)
        content_layout.addWidget(self.increase_hotkey_edit)

        # Quickswitch hotkey
        # Enable/Disable toggle for Quickswitch
        self.quickswitch_checkbox = CircleCheckBox("Enable Quickswitch Hotkey")
        self.quickswitch_checkbox.setObjectName("QuickswitchCheckBox")
        content_layout.addWidget(self.quickswitch_checkbox)

        # Quickswitch key selection
        quickswitch_label = QLabel("Quickswitch Hotkey:")
        quickswitch_label.setObjectName("SubSettingsSectionLabel")
        self.quickswitch_hotkey_edit = QKeySequenceEdit()
        # Use the same selector for all key sequence edits in subsettings
        self.quickswitch_hotkey_edit.setObjectName("SettingsKeySequenceEdit")
        content_layout.addWidget(quickswitch_label)
        content_layout.addWidget(self.quickswitch_hotkey_edit)

        # Autoswitch and Keypassthrough checkboxes
        self.autoswitch_checkbox = CircleCheckBox("Enable Autoswitch")
        self.autoswitch_checkbox.setObjectName("AutoswitchCheckBox")
        content_layout.addWidget(self.autoswitch_checkbox)
        self.keypassthrough_checkbox = CircleCheckBox("Enable Keypassthrough")
        self.keypassthrough_checkbox.setObjectName("KeypassthroughCheckBox")
        # Themed tooltip text; styling handled by QSS theme stylesheets
        self.keypassthrough_checkbox.setToolTip("Just use Media Control instead, it's completely safe.")
        content_layout.addWidget(self.keypassthrough_checkbox)

        # Display Locked Switching
        self.display_locked_checkbox = CircleCheckBox("Display Locked Switching")
        self.display_locked_checkbox.setObjectName("DisplayLockedSwitchingCheckBox")
        self.display_locked_checkbox.setToolTip("When enabled, QuickSwitch and Autoswitch only consider windows on the same monitor as the overlay's current content.")
        content_layout.addWidget(self.display_locked_checkbox)

        # Media Control toggle
        self.media_control_checkbox = CircleCheckBox("Media Control")
        self.media_control_checkbox.setObjectName("MediaControlCheckBox")
        self.media_control_checkbox.setToolTip("Route media keys (play/pause, next, previous, stop) to active media applications.")
        content_layout.addWidget(self.media_control_checkbox)

        # Removed click-through toggle - incompatible with overlay architecture

        # Capture FPS selection
        fps_label = QLabel("Monitor Capture FPS:")
        fps_label.setObjectName("SubSettingsSectionLabel")
        self.capture_fps_combo = QComboBox()
        self.capture_fps_combo.setObjectName("SettingsComboBox")
        # Common FPS presets; dialog will insert persisted custom value if needed
        self.capture_fps_combo.addItems(["15", "30", "60", "120", "144", "165"])
        content_layout.addWidget(fps_label)
        content_layout.addWidget(self.capture_fps_combo)

        # Rounded Borders toggle
        self.rounded_borders_checkbox = CircleCheckBox("Rounded Borders")
        self.rounded_borders_checkbox.setObjectName("RoundedBordersCheckBox")
        # Themed tooltip text; styling handled by QSS
        self.rounded_borders_checkbox.setToolTip("Pretty, but oh does it bleed.")
        content_layout.addWidget(self.rounded_borders_checkbox)

        # Create scroll area and embed the content
        scroll_area = QScrollArea()
        scroll_area.setObjectName("settingsScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Use a plain container as scroll widget to avoid stylesheet bleed
        scroll_container = QWidget()
        scroll_container.setLayout(QVBoxLayout())
        scroll_container.layout().setContentsMargins(0, 0, 0, 0)
        scroll_container.layout().setSpacing(0)
        scroll_container.layout().addWidget(content_inner)
        scroll_area.setWidget(scroll_container)

        main_layout.addWidget(scroll_area)

        # --- Custom Border ---
        self.border = QFrame(self)
        self.border.setObjectName("settingsDialogBorder")
        self.border.setGeometry(0, 0, self.width(), self.height())
        self.border.lower()  # Ensure border is always behind all content
        # The QSS selector QFrame#settingsDialogBorder must be defined in dark.qss/light.qss for visible border

    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        self.border.setGeometry(0, 0, self.width(), self.height())

    def _load_settings(self):
        """Load settings from the settings manager."""
        # Theme: use canonical 'theme' only (no alias fallback)
        theme = self.settings_manager.get("theme", "dark")
        self.theme_combo.setCurrentText(theme.capitalize())

        # Opacity hotkey settings
        opacity_enabled = self.settings_manager.get("hotkeys.opacity_enabled", True)
        self.opacity_hotkeys_checkbox.setChecked(opacity_enabled)

        # Canonical defaults: '-' for decrease, '=' for increase, '`' for quickswitch
        decrease_key = self.settings_manager.get("hotkeys.opacity_decrease", "-")
        if not decrease_key or decrease_key.lower() in ("", "ctrl+alt+down", "ctrl+shift+down"):
            decrease_key = "-"
            self.settings_manager.set("hotkeys.opacity_decrease", decrease_key)
        self.decrease_hotkey_edit.setKeySequence(QKeySequence(decrease_key))

        increase_key = self.settings_manager.get("hotkeys.opacity_increase", "=")
        if not increase_key or increase_key.lower() in ("", "ctrl+alt+up", "ctrl+shift+up"):
            increase_key = "="
            self.settings_manager.set("hotkeys.opacity_increase", increase_key)
        self.increase_hotkey_edit.setKeySequence(QKeySequence(increase_key))

        # Quickswitch hotkey
        quickswitch_key = self.settings_manager.get("hotkeys.opacity_quickswitch", "shift+x")
        if not quickswitch_key or quickswitch_key.lower() in ("",):
            quickswitch_key = "shift+x"
            self.settings_manager.set("hotkeys.opacity_quickswitch", quickswitch_key)
        self.quickswitch_hotkey_edit.setKeySequence(QKeySequence(quickswitch_key))

        # Quickswitch enable (default off)
        quickswitch_enabled = bool(self.settings_manager.get("hotkeys.quickswitch_enabled", False))
        self.quickswitch_checkbox.setChecked(quickswitch_enabled)

        # Feature toggles (stubbed)
        autoswitch_enabled = self.settings_manager.get("features.autoswitch_enabled", False)
        self.autoswitch_checkbox.setChecked(bool(autoswitch_enabled))
        keypassthrough_enabled = self.settings_manager.get("features.keypassthrough_enabled", False)
        self.keypassthrough_checkbox.setChecked(bool(keypassthrough_enabled))

        # Display Locked Switching (default off)
        display_locked = bool(self.settings_manager.get("features.display_locked_switching", False))
        self.display_locked_checkbox.setChecked(display_locked)

        # Media control (default off)
        media_control_enabled = bool(self.settings_manager.get("features.media_control_enabled", False))
        self.media_control_checkbox.setChecked(media_control_enabled)

        # Removed click-through setting

        # Capture FPS
        try:
            fps_value = int(float(self.settings_manager.get("capture.fps", 30)))
        except Exception:
            fps_value = 30
        # Insert custom value if not present
        if str(fps_value) not in [self.capture_fps_combo.itemText(i) for i in range(self.capture_fps_combo.count())]:
            self.capture_fps_combo.addItem(str(fps_value))
        self.capture_fps_combo.setCurrentText(str(fps_value))

        # Rounded borders (default off)
        rounded_enabled = bool(self.settings_manager.get("overlay.rounded_borders", False))
        self.rounded_borders_checkbox.setChecked(rounded_enabled)
        # Integrated system listens to SettingsManager; no direct overlay calls


    def _save_settings(self):
        """Save the current settings."""
        # Theme (persist canonical key only; no aliases/fallbacks)
        theme = self.theme_combo.currentText().lower()
        self.settings_manager.set("theme", theme, save_immediately=False)
        logger.debug(f"Saved theme setting: theme='{theme}'")

        # Batch all settings changes without immediate saves, then save once at the end
        
        # Opacity hotkey settings
        opacity_enabled = self.opacity_hotkeys_checkbox.isChecked()
        self.settings_manager.set("hotkeys.opacity_enabled", opacity_enabled, save_immediately=False)
        logger.debug(f"Saved hotkeys.opacity_enabled={opacity_enabled}")

        decrease_key = self.decrease_hotkey_edit.keySequence().toString()
        self.settings_manager.set("hotkeys.opacity_decrease", decrease_key, save_immediately=False)
        logger.debug(f"Saved hotkeys.opacity_decrease='{decrease_key}'")

        increase_key = self.increase_hotkey_edit.keySequence().toString()
        self.settings_manager.set("hotkeys.opacity_increase", increase_key, save_immediately=False)
        logger.debug(f"Saved hotkeys.opacity_increase='{increase_key}'")

        quickswitch_key = self.quickswitch_hotkey_edit.keySequence().toString()
        self.settings_manager.set("hotkeys.opacity_quickswitch", quickswitch_key, save_immediately=False)
        logger.debug(f"Saved hotkeys.opacity_quickswitch='{quickswitch_key}'")

        # Quickswitch enable toggle
        quickswitch_enabled = self.quickswitch_checkbox.isChecked()
        self.settings_manager.set("hotkeys.quickswitch_enabled", quickswitch_enabled, save_immediately=False)
        logger.debug(f"Saved hotkeys.quickswitch_enabled={quickswitch_enabled}")

        # Feature toggles
        autoswitch_enabled = self.autoswitch_checkbox.isChecked()
        self.settings_manager.set("features.autoswitch_enabled", autoswitch_enabled, save_immediately=False)
        logger.debug(f"Saved features.autoswitch_enabled={autoswitch_enabled}")
        
        keypassthrough_enabled = self.keypassthrough_checkbox.isChecked()
        self.settings_manager.set("features.keypassthrough_enabled", keypassthrough_enabled, save_immediately=False)
        logger.debug(f"Saved features.keypassthrough_enabled={keypassthrough_enabled}")

        # Display Locked Switching
        display_locked = self.display_locked_checkbox.isChecked()
        self.settings_manager.set("features.display_locked_switching", display_locked, save_immediately=False)
        logger.debug(f"Saved features.display_locked_switching={display_locked}")

        # Media control
        media_control_enabled = self.media_control_checkbox.isChecked()
        self.settings_manager.set("features.media_control_enabled", media_control_enabled, save_immediately=False)
        logger.debug(f"Saved features.media_control_enabled={media_control_enabled}")

        # Removed click-through setting

        # Rounded borders
        rounded_enabled = self.rounded_borders_checkbox.isChecked()
        self.settings_manager.set("overlay.rounded_borders", rounded_enabled, save_immediately=False)
        logger.debug(f"Saved overlay.rounded_borders={rounded_enabled}")
        
        # Single batched save at the end
        self.settings_manager.save()
        
        # Apply live changes after settings are persisted
        # Removed click-through overlay application

    
    def _on_theme_changed(self, theme_text):
        """Handle theme selection change."""
        theme = self.theme_combo.currentText().lower()
        # Persist canonical key only; no alias mirroring
        self.settings_manager.set("theme", theme)
        # Theme changes are applied immediately for visual feedback

    def _on_rounded_borders_changed(self, state):
        """Handle rounded borders toggle: apply live and persist immediately."""
        try:
            enabled = self.rounded_borders_checkbox.isChecked()
            # Persist immediately (no fallback)
            self.settings_manager.set("overlay.rounded_borders", enabled)
            # Integrated overlays update via settings change handlers
            logger.debug(f"Rounded borders toggled to {enabled} (integrated update via SettingsManager)")
        except Exception as e:
            logger.error(f"Error toggling rounded borders: {e}", exc_info=True)

    def _on_opacity_hotkeys_changed(self, state):
        """Handle opacity hotkeys enable/disable."""
        enabled = self.opacity_hotkeys_checkbox.isChecked()
        self.settings_manager.set("hotkeys.opacity_enabled", enabled)
        self.opacity_manager.update_hotkeys()

    def _on_decrease_hotkey_changed(self, key_sequence):
        """Handle decrease opacity hotkey change."""
        key = self.decrease_hotkey_edit.keySequence().toString()
        self.settings_manager.set("hotkeys.opacity_decrease", key)
        self.opacity_manager.update_hotkeys()
        logger.debug(f"Changed hotkeys.opacity_decrease to '{key}' and updated opacity hotkeys")

    def _on_increase_hotkey_changed(self, key_sequence):
        """Handle increase opacity hotkey change."""
        key = self.increase_hotkey_edit.keySequence().toString()
        self.settings_manager.set("hotkeys.opacity_increase", key)
        self.opacity_manager.update_hotkeys()
        logger.debug(f"Changed hotkeys.opacity_increase to '{key}' and updated opacity hotkeys")

    def _on_quickswitch_hotkey_changed(self, key_sequence):
        """Handle quickswitch hotkey change."""
        key = self.quickswitch_hotkey_edit.keySequence().toString()
        self.settings_manager.set("hotkeys.opacity_quickswitch", key)
        try:
            # Update QuickSwitchController live
            from core.switching.quickswitch_controller import get_quickswitch_controller
            get_quickswitch_controller().update_hotkeys()
            logger.debug(f"Changed hotkeys.opacity_quickswitch to '{key}' and updated QuickSwitch hotkey")
        except Exception as e:
            logger.error(f"Failed to update QuickSwitch hotkey live: {e}", exc_info=True)

    def _on_quickswitch_enabled_changed(self, state):
        """Handle quickswitch enable/disable toggle."""
        enabled = self.quickswitch_checkbox.isChecked()
        self.settings_manager.set("hotkeys.quickswitch_enabled", enabled)
        try:
            from core.switching.quickswitch_controller import get_quickswitch_controller
            get_quickswitch_controller().update_hotkeys()
            logger.debug(f"Quickswitch enabled toggled to {enabled} and applied")
        except Exception as e:
            logger.error(f"Failed to apply quickswitch enable setting: {e}")

    def _on_autoswitch_changed(self, state):
        """Handle autoswitch toggle (feature stub)."""
        enabled = self.autoswitch_checkbox.isChecked()
        self.settings_manager.set("features.autoswitch_enabled", enabled)
        try:
            from core.switching.autoswitch_controller import get_autoswitch_controller
            get_autoswitch_controller().apply_settings()
            logger.debug(f"Autoswitch toggled to {enabled} and applied")
        except Exception as e:
            logger.error(f"Failed to apply autoswitch setting: {e}", exc_info=True)

    def _on_display_locked_switching_changed(self, state):
        """Handle Display Locked Switching toggle: persist and notify controllers."""
        enabled = self.display_locked_checkbox.isChecked()
        self.settings_manager.set("features.display_locked_switching", enabled)
        try:
            # No heavy re-init required; controllers read the setting on use. Call applies for audit/logging.
            from core.switching.autoswitch_controller import get_autoswitch_controller
            get_autoswitch_controller().apply_settings()
        except Exception as e:
            logger.error(f"Failed to notify autoswitch after display_locked change: {e}")
        try:
            from core.switching.quickswitch_controller import get_quickswitch_controller
            # Quickswitch reads at invocation; just log via a no-op hotkey refresh to keep symmetry
            get_quickswitch_controller().update_hotkeys()
        except Exception as e:
            logger.error(f"Failed to notify quickswitch after display_locked change: {e}")

    def _on_keypassthrough_changed(self, state):
        """Handle keypassthrough toggle."""
        enabled = self.keypassthrough_checkbox.isChecked()
        # If enabling and not yet acknowledged, show the one-time warning
        if enabled:
            try:
                acked = bool(self.settings_manager.get("features.keypassthrough_warning_ack", False))
            except Exception:
                acked = False
            if not acked:
                dlg = KeyPassthroughWarningDialog(parent=self)
                result = dlg.exec()
                if result == QDialog.Accepted:
                    # Persist acknowledgment so we don't show again
                    self.settings_manager.set("features.keypassthrough_warning_ack", True)
                else:
                    # Revert checkbox without emitting signal loop
                    try:
                        self.keypassthrough_checkbox.blockSignals(True)
                        self.keypassthrough_checkbox.setChecked(False)
                    finally:
                        self.keypassthrough_checkbox.blockSignals(False)
                    self.settings_manager.set("features.keypassthrough_enabled", False)
                    logger.debug("Keypassthrough enable canceled by user (warning not acknowledged)")
                    return
        # Persist the final state
        self.settings_manager.set("features.keypassthrough_enabled", enabled)
        logger.debug(f"Keypassthrough toggled to {enabled}")
        
    def _on_media_control_changed(self, state):
        """Handle media control toggle."""
        enabled = self.media_control_checkbox.isChecked()
        self.settings_manager.set("features.media_control_enabled", enabled)
        logger.debug(f"Media control toggled to {enabled}")

    # Removed click-through change handler

    def _on_capture_fps_changed(self, text):
        """Handle capture FPS change: persist and apply live via PipelineManager."""
        try:
            fps = float(text)
            # Persist canonical key
            self.settings_manager.set("capture.fps", int(fps))
            # Apply live via PipelineManager (drop-in compatible)
            try:
                from core.graphics.pipeline_manager import get_pipeline_manager
                get_pipeline_manager().set_capture_rate(fps)
                logger.debug(f"Capture FPS changed to {fps} and applied to PipelineManager")
            except Exception as e:
                logger.error(f"Failed to apply capture FPS live: {e}")
        except Exception as e:
            logger.error(f"Invalid FPS value '{text}': {e}")

    def _connect_signals(self):
        """Connect UI signals to handlers."""
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.opacity_hotkeys_checkbox.stateChanged.connect(self._on_opacity_hotkeys_changed)
        self.decrease_hotkey_edit.keySequenceChanged.connect(self._on_decrease_hotkey_changed)
        self.increase_hotkey_edit.keySequenceChanged.connect(self._on_increase_hotkey_changed)
        self.quickswitch_hotkey_edit.keySequenceChanged.connect(self._on_quickswitch_hotkey_changed)
        self.quickswitch_checkbox.stateChanged.connect(self._on_quickswitch_enabled_changed)
        self.autoswitch_checkbox.stateChanged.connect(self._on_autoswitch_changed)
        self.keypassthrough_checkbox.stateChanged.connect(self._on_keypassthrough_changed)
        self.media_control_checkbox.stateChanged.connect(self._on_media_control_changed)
        # Removed click-through signal connection
        self.rounded_borders_checkbox.stateChanged.connect(self._on_rounded_borders_changed)
        self.display_locked_checkbox.stateChanged.connect(self._on_display_locked_switching_changed)
        self.capture_fps_combo.currentTextChanged.connect(self._on_capture_fps_changed)

    # Theme application is centralized in ThemeManager; dialog does not apply theme directly

    # Mouse events are handled by the centralized WindowBehaviorManager
    # See the mousePressEvent, mouseMoveEvent, mouseReleaseEvent, and leaveEvent methods
    # defined earlier in this class

    def keyPressEvent(self, event):
        """Close the subsettings dialog on ESC."""
        try:
            if event and event.key() == Qt.Key_Escape:
                event.accept()
                self.close()
                return
        except Exception:
            pass
        super().keyPressEvent(event)
