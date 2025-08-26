from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtCore import Qt, Property

class ResizeIndicator(QWidget):
    """
    A subtle diagonal line indicator for the bottom right corner to hint at resizability.
    Only visible in the bottom right, styled via QSS if needed.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resizeIndicator")
        self.setFixedSize(22, 22)
        self._indicatorColor = QColor(255,255,255,255)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def getIndicatorColor(self):
        return self._indicatorColor

    def setIndicatorColor(self, color):
        self._indicatorColor = color
        self.update()

    indicatorColor = Property(QColor, getIndicatorColor, setIndicatorColor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        # Inset by the border radius (8px) so the line never sticks out
        radius = 8
        pen = QPen(self.indicatorColor, 2)
        painter.setPen(pen)
        painter.drawLine(radius, self.height()-radius, self.width()-radius, radius)
        painter.end()
