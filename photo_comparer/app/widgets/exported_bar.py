"""ExportedBar — thin status strip at the bottom of ComparisonView.

Shows a thumbnail + metadata for the already-exported photo of the current
group.  Hidden automatically when no export exists for the current group.

Signals
-------
restore_requested(source_path: str)
    Emitted when the user clicks "↩ Restaurer sélection".
    ComparisonView uses it to re-light the green border on the right tile.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from ..utils.export_store import ExportRecord
from ..utils.thumbnail_cache import generate_thumbnail

_HEIGHT = 56

_FRAME_STYLE = (
    "ExportedBar {"
    "  background: #17241a;"
    "  border-top: 1px solid #2a5c35;"
    "}"
)
_INFO_STYLE = "color: #7ec896; font-size: 11px; background: transparent;"
_BTN_STYLE = (
    "QPushButton {"
    "  background: #1c3d24; color: #7ec896;"
    "  border: 1px solid #2a5c35; border-radius: 3px;"
    "  padding: 2px 10px; font-size: 11px;"
    "}"
    "QPushButton:hover { background: #265c32; color: #aaeaaa; }"
)


class ExportedBar(QFrame):
    restore_requested = Signal(str)   # source_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ExportedBar")
        self.setFixedHeight(_HEIGHT)
        self.setStyleSheet(_FRAME_STYLE)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(12)

        # Small thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(64, 46)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            "border: 1px solid #2a5c35; background: #0e1a12;"
        )
        layout.addWidget(self._thumb)

        # Info text
        self._info = QLabel()
        self._info.setStyleSheet(_INFO_STYLE)
        self._info.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(self._info, 1)   # stretch

        # Restore button
        self._btn = QPushButton("↩  Restaurer sélection")
        self._btn.setFixedHeight(30)
        self._btn.setStyleSheet(_BTN_STYLE)
        self._btn.setToolTip(
            "Réactiver le cadre vert sur la version exportée"
        )
        self._btn.clicked.connect(self._on_restore_clicked)
        layout.addWidget(self._btn)

        self._source_path: str = ""
        self.hide()   # invisible until show_record() is called

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_record(self, record: ExportRecord) -> None:
        """Display export info and make the bar visible."""
        self._source_path = record.source_path

        # Thumbnail — generate_thumbnail is instant for cached files;
        # the source was already viewed so the cache should be warm.
        px = self._load_thumb(Path(record.source_path))
        if px:
            self._thumb.setPixmap(px)
        else:
            self._thumb.setText("?")
            self._thumb.setStyleSheet(
                self._thumb.styleSheet() + " color:#555; font-size:18px;"
            )

        self._info.setText(
            f"<b style='color:#aaeaaa;'>✓ Exportée</b>"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;{record.output_filename}"
            f"&nbsp;&nbsp;<span style='color:#3a7a50;'>depuis</span>&nbsp;"
            f"<i>{record.dir_name}</i>"
        )
        self.show()

    def clear_record(self) -> None:
        """Hide the bar and reset its state."""
        self._source_path = ""
        self._thumb.clear()
        self._info.clear()
        self.hide()

    # ------------------------------------------------------------------

    def _load_thumb(self, path: Path) -> Optional[QPixmap]:
        thumb = generate_thumbnail(path)
        if not thumb:
            return None
        px = QPixmap(str(thumb)).scaled(
            64, 46,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return px if not px.isNull() else None

    def _on_restore_clicked(self):
        if self._source_path:
            self.restore_requested.emit(self._source_path)
