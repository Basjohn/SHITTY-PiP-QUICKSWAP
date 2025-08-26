"""
Simplified SubSettings Dialog for theme selection only.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
)

from core.logging import get_logger
from core.settings import SettingsManager

logger = get_logger(__name__)

class SubSettingsDialog(QDialog):
    """Simplified settings dialog for theme selection only."""
    
    def __init__(self, parent=None):
        """Initialize the simplified settings dialog."""
        super().__init__(parent)
        
        # Initialize core components
        self.settings_manager = SettingsManager()
        
        # Set window properties
        self.setWindowTitle("Settings")
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        
        # Setup UI
        self._setup_ui()
        
        # Load current settings
        self._load_settings()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("Settings")
        title_label.setObjectName("SettingsTitle")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Theme selection
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        theme_label.setObjectName("SettingsLabel")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setObjectName("SettingsComboBox")
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo, 1)
        layout.addLayout(theme_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("SettingsButton")
        self.apply_button.clicked.connect(self._apply_settings)
        
        self.close_button = QPushButton("Close")
        self.close_button.setObjectName("SettingsButton")
        self.close_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
        
        # Connect signals
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
    
    def _load_settings(self):
        """Load settings from the settings manager."""
        try:
            theme = self.settings_manager.get("theme", "dark")
            self.theme_combo.setCurrentText(theme.capitalize())
        except Exception as e:
            logger.error(f"Error loading settings: {e}", exc_info=True)
    
    def _apply_settings(self):
        """Apply the current settings."""
        try:
            theme = self.theme_combo.currentText().lower()
            self.settings_manager.set("theme", theme)
            
        except Exception as e:
            logger.error(f"Error applying settings: {e}", exc_info=True)
    
    def _on_theme_changed(self, theme_text):
        """Handle theme selection change."""
        # No direct theme application. Centralized via ThemeManager listening to SettingsManager.
        # Optionally, we could persist immediately here, but keep behavior to apply on button click.
        
    # Theme application is centralized; dialog does not apply theme directly
