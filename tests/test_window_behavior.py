#!/usr/bin/env python3
"""
Test script to debug window behavior issues.
"""
import sys

def main():
    # Import GUI and behavior modules only when running as a script to avoid pytest warnings
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
    from PySide6.QtCore import Qt
    from utils.window.behavior import WindowBehaviorManager

    class DemoWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Test Window Behavior")
            self.setGeometry(100, 100, 400, 300)

            # Set window flags for frameless window
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setMouseTracking(True)

            # Create central widget
            central_widget = QWidget()
            layout = QVBoxLayout()
            label = QLabel("Drag me by the title bar\nResize from any edge or corner")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            central_widget.setLayout(layout)
            self.setCentralWidget(central_widget)

            # Initialize window behavior manager
            self.window_behavior = WindowBehaviorManager(self, 200, 150)

        def mousePressEvent(self, event):
            # Allow dragging from top 30 pixels
            def is_draggable(pos):
                return pos.y() <= 30

            self.window_behavior.handle_mouse_press(event, is_draggable)

        def mouseMoveEvent(self, event):
            self.window_behavior.handle_mouse_move(event)

        def mouseReleaseEvent(self, event):
            self.window_behavior.handle_mouse_release(event)

        def leaveEvent(self, event):
            self.window_behavior.handle_leave()

    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
