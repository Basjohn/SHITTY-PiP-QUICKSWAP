"""
UI Package Initialization

This package contains the main user interface components for the SPQ application.
"""

from .main_dialog import MainDialog
from .dialogs.subsettings_dialog import SubSettingsDialog, CircleCheckBox, DoubleClickCheckBox

__all__ = [
    'MainDialog',
    'SubSettingsDialog',
    'CircleCheckBox',
    'DoubleClickCheckBox'
]