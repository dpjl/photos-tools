"""Photo Comparator — entry point.

Usage:
    python main.py

Dependencies (install once):
    pip install PySide6 Pillow
"""
import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from app.widgets.main_window import MainWindow


def _dark_palette(app: QApplication) -> QPalette:
    p = QPalette()
    c = p.setColor   # shorthand
    W = QPalette.ColorRole.Window
    c(QPalette.ColorGroup.All,     W,                          QColor(30, 30, 30))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.WindowText,    QColor(200, 200, 200))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.Base,          QColor(20, 20, 20))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.AlternateBase, QColor(38, 38, 38))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.ToolTipBase,   QColor(30, 30, 30))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.ToolTipText,   QColor(200, 200, 200))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.Text,          QColor(200, 200, 200))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.Button,        QColor(50, 50, 50))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.ButtonText,    QColor(200, 200, 200))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.BrightText,    QColor(255, 255, 255))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.Link,          QColor(66, 148, 220))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.Highlight,     QColor(42, 110, 200))
    c(QPalette.ColorGroup.All,     QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    c(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,   QColor(100, 100, 100))
    c(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,   QColor(100, 100, 100))
    c(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,         QColor(100, 100, 100))
    return p


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Photo Comparator")
    app.setOrganizationName("PhotoTools")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette(app))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
