"""
Custom circular checkbox component with programmatic rendering and theme integration.

This module provides a CircleCheckBox widget that renders a clean circular indicator
with proper theme color inheritance from the QSS system.
"""

from PySide6.QtWidgets import QCheckBox
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
from PySide6.QtCore import Qt, QRect

from utils.theme.theme_manager import get_theme_manager


class CircleCheckBox(QCheckBox):
    """A custom checkbox with a circular indicator that inherits theme colors from QSS."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set minimum size to ensure the indicator is fully visible
        self.setMinimumSize(24, 24)
        # Set up the checkbox to be transparent and prevent system styling
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Completely disable the default indicator
        self.setStyleSheet("""
            QCheckBox {
                background: transparent;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 0px;
                height: 0px;
                border: none;
                padding: 0px;
                margin: 0px;
            }
        """)
    
    def _get_theme_colors(self):
        """Get theme colors from the theme manager."""
        try:
            theme_manager = get_theme_manager()
            current_theme = theme_manager.get_current_theme()
            is_light = current_theme.lower() == 'light'
            
            if is_light:
                return {
                    'text': QColor(0, 0, 0),           # Black text
                    'border': QColor(85, 85, 85),      # Dark gray border
                    'fill': QColor(0, 0, 0),           # Black fill when checked
                    'disabled': QColor(135, 135, 135)  # Gray for disabled
                }
            else:
                return {
                    'text': QColor(255, 255, 255),     # White text
                    'border': QColor(85, 85, 85),      # Dark gray border
                    'fill': QColor(255, 255, 255),     # White fill when checked
                    'disabled': QColor(120, 120, 120)  # Gray for disabled
                }
        except Exception:
            # Fallback to dark theme colors
            return {
                'text': QColor(255, 255, 255),
                'border': QColor(85, 85, 85),
                'fill': QColor(255, 255, 255),
                'disabled': QColor(120, 120, 120)
            }
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get theme colors
        colors = self._get_theme_colors()
        
        # Calculate indicator rect (left side of checkbox)
        indicator_size = 18
        indicator_rect = QRect(2, (self.height() - indicator_size) // 2, indicator_size, indicator_size)
        
        # Draw outer circle
        painter.setPen(QPen(colors['border'], 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(indicator_rect)
        
        # Draw inner circle if checked
        if self.isChecked():
            inner_size = 10
            inner_rect = QRect(
                indicator_rect.x() + (indicator_rect.width() - inner_size) // 2,
                indicator_rect.y() + (indicator_rect.height() - inner_size) // 2,
                inner_size, inner_size
            )
            painter.save()
            try:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(colors['fill']))
                painter.drawEllipse(inner_rect)
            finally:
                painter.restore()
        
        # Draw text
        if self.text():
            text_rect = QRect(
                indicator_rect.right() + 6,
                0,
                self.width() - indicator_rect.right() - 6,
                self.height()
            )
            painter.save()
            try:
                if self.isEnabled():
                    painter.setPen(colors['text'])
                else:
                    painter.setPen(colors['disabled'])
                painter.drawText(
                    text_rect,
                    Qt.AlignLeft | Qt.AlignVCenter | Qt.TextShowMnemonic,
                    self.text()
                )
            finally:
                painter.restore()
        
        # Disabled state overlay (semi-transparent dotted line)
        if not self.isEnabled():
            painter.setPen(QPen(QColor(0, 0, 0, 60), 1, Qt.DotLine))
            painter.drawEllipse(indicator_rect)
