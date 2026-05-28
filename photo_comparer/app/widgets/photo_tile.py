"""PhotoTile — single-image tile used in the comparison grid.

A tile wraps a ZoomGraphicsView with a header label (directory name + filename)
and an overlay placeholder for "loading" and "absent" states.

Signals
-------
solo_requested(int)  — tile index; emitted on double-click → expand to solo
best_selected(int)   — tile index; emitted on Ctrl+Click → mark as best
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QTextBlockFormat, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPlainTextEdit,
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
        self._file_size = None   # cached file size in bytes

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

        # --- File size label ---
        self._size_label = QLabel("")
        self._size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._size_label.setMaximumHeight(13)
        self._size_label.setStyleSheet(
            "color: #fff; font-size: 8px; background: transparent; padding: 0px;"
        )
        self._size_label.hide()
        layout.addWidget(self._size_label)

        # --- Image viewer ---
        self.viewer = ZoomGraphicsView(self)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.viewer)

        # --- JSON text viewer ---
        self._json_view = QPlainTextEdit()
        self._json_view.setReadOnly(True)
        self._json_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        font = QFont("Consolas, Courier New, monospace")
        font.setPointSize(8)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._json_view.setFont(font)
        self._json_view.setStyleSheet(
            "QPlainTextEdit { background: #0d0d0d; color: #c8c8c8;"
            " border: none; selection-background-color: #264f78; }"
        )
        self._json_view.hide()
        layout.addWidget(self._json_view)

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

    def set_loading(self, file_size: int = None):
        self._has_image = False
        self.viewer.clear_pixmap()
        self._json_view.hide()
        self._file_label.setText("")
        self._update_size_label(file_size)
        self._show_overlay("Chargement…")

    def set_absent(self):
        self._has_image = False
        self.viewer.clear_pixmap()
        self._json_view.hide()
        self._file_label.setText("")
        self._size_label.hide()
        self._file_size = None
        self._show_overlay("Absent")

    def set_pixmap(self, pixmap, filename: str = "", file_size: int = None):
        self._overlay.hide()
        self._json_view.hide()
        self.viewer.show()
        self.viewer.set_pixmap(pixmap)
        self._file_label.setText(filename)
        self._update_size_label(file_size)
        self._has_image = True

    def set_json(self, padded_lines: list, file_size, diff_line_indices: set, gap_line_indices: set):
        """Display LCS-aligned JSON content with diff and gap highlighting.

        padded_lines     : list[str | None] — None entries are alignment gaps
        diff_line_indices: row indices where this tile’s content differs
        gap_line_indices : row indices where this tile has no line (None gap)
        """
        content = "\n".join(line if line is not None else "" for line in padded_lines)
        self._overlay.hide()
        self.viewer.hide()
        self._json_view.show()
        self._json_view.setPlainText(content)
        self._apply_diff_highlighting(diff_line_indices, gap_line_indices)
        self._update_size_label(file_size)
        self._has_image = False

    def set_json_absent(self):
        """Show absent overlay in JSON mode."""
        self.viewer.hide()
        self._json_view.hide()
        self._file_label.setText("")
        self._size_label.hide()
        self._file_size = None
        self._show_overlay("Absent")

    def set_as_best(self, is_best: bool):
        self._is_best = is_best
        self.setStyleSheet(self._STYLE_BEST if is_best else self._STYLE_NORMAL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_size_label(self, file_size: int = None):
        if file_size is not None:
            self._file_size = file_size
        if self._file_size is not None:
            self._size_label.setText(self._format_size(self._file_size))
            self._size_label.show()
        else:
            self._size_label.hide()

    @staticmethod
    def _format_size(n: int) -> str:
        """Return human-readable size with byte count, using French thin-space separator."""
        def _fmt_bytes(v: int) -> str:
            s = f"{v:,}".replace(",", "\u202f")  # narrow no-break space
            return f"{s}\u202fo"

        if n < 1024:
            return _fmt_bytes(n)
        elif n < 1024 * 1024:
            return f"{n / 1024:.1f}\u202fKo ({_fmt_bytes(n)})"
        else:
            return f"{n / (1024 * 1024):.1f}\u202fMo ({_fmt_bytes(n)})"

    def _apply_diff_highlighting(self, diff_line_indices: set, gap_line_indices: set = None):
        """Highlight diff lines in yellow and gap lines in dark grey."""
        if gap_line_indices is None:
            gap_line_indices = set()

        doc = self._json_view.document()
        cursor = QTextCursor(doc)

        diff_fmt = QTextBlockFormat()
        diff_fmt.setBackground(QColor(255, 200, 0, 70))

        gap_fmt = QTextBlockFormat()
        gap_fmt.setBackground(QColor(40, 40, 40, 255))   # solid dark grey stripe

        normal_fmt = QTextBlockFormat()
        normal_fmt.setBackground(QColor(0, 0, 0, 0))

        block = doc.begin()
        i = 0
        while block.isValid():
            cursor.setPosition(block.position())
            if i in gap_line_indices:
                cursor.setBlockFormat(gap_fmt)
            elif i in diff_line_indices:
                cursor.setBlockFormat(diff_fmt)
            else:
                cursor.setBlockFormat(normal_fmt)
            block = block.next()
            i += 1

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
