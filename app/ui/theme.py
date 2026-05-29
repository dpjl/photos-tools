"""ui/theme.py — Thème sombre de l'application."""

from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

DARK_STYLESHEET = """
    QMainWindow, QWidget { background: #141420; color: #ccc; font-family: 'Segoe UI', sans-serif; }
    QMenuBar { background: #16162a; color: #ccc; }
    QMenuBar::item:selected { background: #2a2a4a; }
    QMenu { background: #1e1e2e; color: #ccc; border: 1px solid #333; }
    QMenu::item:selected { background: #2a2a4a; }
    QScrollBar:vertical { background: #1a1a2e; width: 8px; border: none; }
    QScrollBar::handle:vertical { background: #3a3a5a; border-radius: 4px; min-height: 20px; }
    QScrollBar:horizontal { background: #1a1a2e; height: 8px; border: none; }
    QScrollBar::handle:horizontal { background: #3a3a5a; border-radius: 4px; min-width: 20px; }
    QSplitter::handle { background: #2a2a4a; }
"""


def apply_dark_theme(widget: QMainWindow) -> None:
    widget.setStyleSheet(DARK_STYLESHEET)
