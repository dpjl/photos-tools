"""PhotoTile — single-image tile used in the comparison grid.

A tile wraps a ZoomGraphicsView with a header label (directory name + filename)
and an overlay placeholder for "loading" and "absent" states.

Signals
-------
solo_requested(int)  — tile index; emitted on double-click → expand to solo
best_selected(int)   — tile index; emitted on Ctrl+Click → mark as best
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .zoom_graphics_view import ZoomGraphicsView


class PhotoTile(QFrame):
    solo_requested = Signal(int)   # tile index
    best_selected = Signal(int)    # tile index

    _STYLE_NORMAL = "QFrame#PhotoTile { border: 1px solid #3a3a3a; background: #181818; }"
    _STYLE_BEST = "QFrame#PhotoTile { border: 2px solid #4CAF50; background: #181818; }"

    def __init__(self, tile_index: int, dir_name: str, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("PhotoTile")
        self.tile_index = tile_index
        self._is_best = False
        self._has_image = False

        self.setStyleSheet(self._STYLE_NORMAL)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # --- Header: directory name ---
        self._dir_label = QLabel(dir_name)
        self._dir_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dir_label.setMaximumHeight(18)
        self._dir_label.setStyleSheet(
            "color: #888; font-size: 10px; background: transparent; padding: 0px;"
        )
        layout.addWidget(self._dir_label)

        # --- Filename label ---
        self._file_label = QLabel("")
        self._file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._file_label.setMaximumHeight(15)
        self._file_label.setStyleSheet(
            "color: #5a9fd4; font-size: 9px; background: transparent; padding: 0px;"
        )
        layout.addWidget(self._file_label)

        # --- Image viewer ---
        self.viewer = ZoomGraphicsView(self)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.viewer)

        # --- Overlay (loading / absent) ---
        self._overlay = QLabel("Absent")
        self._overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._overlay.setStyleSheet(
            "color: #444; font-size: 18px; background: #141414; border: none;"
        )
        layout.addWidget(self._overlay)
        self._show_overlay("Absent")   # start in absent state

        # Wire viewer signals up to tile-level signals
        self.viewer.double_clicked.connect(self._on_double_click)
        self.viewer.ctrl_clicked.connect(self._on_ctrl_click)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_loading(self):
        self._has_image = False
        self.viewer.clear_pixmap()
        self._file_label.setText("")
        self._show_overlay("Chargement…")

    def set_absent(self):
        self._has_image = False
        self.viewer.clear_pixmap()
        self._file_label.setText("")
        self._show_overlay("Absent")

    def set_pixmap(self, pixmap, filename: str = ""):
        self._overlay.hide()
        self.viewer.show()
        self.viewer.set_pixmap(pixmap)
        self._file_label.setText(filename)
        self._has_image = True

    def set_as_best(self, is_best: bool):
        self._is_best = is_best
        self.setStyleSheet(self._STYLE_BEST if is_best else self._STYLE_NORMAL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show_overlay(self, text: str):
        self.viewer.hide()
        self._overlay.setText(text)
        self._overlay.show()

    def _on_double_click(self):
        self.solo_requested.emit(self.tile_index)

    def _on_ctrl_click(self):
        self.best_selected.emit(self.tile_index)

    # Ctrl+click on non-viewer area (title label, etc.)
    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.best_selected.emit(self.tile_index)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._has_image and not (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.solo_requested.emit(self.tile_index)
        super().mouseDoubleClickEvent(event)
