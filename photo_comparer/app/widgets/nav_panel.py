"""NavPanel — left-side navigation panel.

Shows all photo groups as a scrollable list with:
  • Lazy-loaded thumbnails (generated/cached via thumbnail_cache)
  • A filter bar to jump quickly to a group by key fragment
  • Visual badges: ✓ = copied to output, ● = absent in some directories

Thumbnails are generated in a QThreadPool worker; only the file path is
emitted across the thread boundary (QPixmap creation happens on the GUI
thread).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models.directory_manager import DirectoryManager
from ..utils.thumbnail_cache import generate_thumbnail

THUMB_W = 90
THUMB_H = 68


# ---------------------------------------------------------------------------
# Background thumbnail worker
# ---------------------------------------------------------------------------

class _ThumbSignals(QObject):
    done = Signal(str, str)   # group_key, thumb_file_path


class _ThumbTask(QRunnable):
    def __init__(self, key: str, path: Path):
        super().__init__()
        self.key = key
        self.path = path
        self.signals = _ThumbSignals()
        self.setAutoDelete(True)

    def run(self):
        thumb = generate_thumbnail(self.path)
        if thumb is not None:
            self.signals.done.emit(self.key, str(thumb))


# ---------------------------------------------------------------------------
# NavPanel
# ---------------------------------------------------------------------------

class NavPanel(QWidget):
    group_selected = Signal(int)   # group index in DirectoryManager.groups

    def __init__(self, dir_manager: DirectoryManager, parent: QWidget = None):
        super().__init__(parent)
        self._dir_manager = dir_manager
        self._key_to_row: Dict[str, int] = {}
        self._copied_keys: set = set()
        self._pool = QThreadPool.globalInstance()

        self.setMinimumWidth(130)
        self.setMaximumWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(4)

        # Filter bar
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filtrer… (ex: 0077)")
        self._filter.setClearButtonEnabled(True)
        self._filter.setStyleSheet(
            "QLineEdit { background: #252525; color: #ccc;"
            " border: 1px solid #555; border-radius: 3px; padding: 3px; }"
        )
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        # Photo count
        self._count_label = QLabel("0 groupes")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._count_label)

        # List
        self._list = QListWidget()
        self._list.setIconSize(QSize(THUMB_W, THUMB_H))
        self._list.setSpacing(1)
        self._list.setUniformItemSizes(True)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setStyleSheet(
            "QListWidget { background: #1a1a1a; border: none; outline: none; }"
            "QListWidget::item { color: #bbb; padding: 3px 4px;"
            "  border-bottom: 1px solid #2a2a2a; }"
            "QListWidget::item:selected { background: #1e4a7a; color: #fff; }"
            "QListWidget::item:hover { background: #252525; }"
        )
        self._list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self):
        """Rebuild the list from DirectoryManager.groups."""
        self._list.blockSignals(True)
        self._list.clear()
        self._key_to_row = {}

        total_dirs = self._dir_manager.dir_count()
        placeholder = self._make_placeholder()

        for i, group in enumerate(self._dir_manager.groups):
            text = self._item_text(group.key, group.photo_count(), total_dirs)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setSizeHint(QSize(THUMB_W + 20, THUMB_H + 22))
            item.setIcon(QIcon(placeholder))
            self._list.addItem(item)
            self._key_to_row[group.key] = i

            # Launch async thumbnail load
            photo = group.any_photo()
            if photo:
                task = _ThumbTask(group.key, photo)
                task.signals.done.connect(self._on_thumb_loaded)
                self._pool.start(task)

        self._list.blockSignals(False)
        self._count_label.setText(
            f"{self._dir_manager.group_count()} groupes"
        )

    def select_group(self, index: int):
        """Programmatically select a group by its index (signals will fire)."""
        self._list.setCurrentRow(index)
        self._list.scrollToItem(
            self._list.item(index),
            QAbstractItemView.ScrollHint.EnsureVisible,
        )

    def mark_copied(self, key: str):
        """Mark a single group as having been copied to output."""
        self._copied_keys.add(key)
        self._refresh_item_text(key)

    def set_copied_keys(self, keys: set) -> None:
        """Bulk-set the copied keys (e.g. loaded from export sidecars).

        Updates badge text for every key present in the list; safe to call
        before or after refresh().
        """
        self._copied_keys = set(keys)
        for key in keys:
            self._refresh_item_text(key)

    def _refresh_item_text(self, key: str) -> None:
        row = self._key_to_row.get(key)
        if row is None:
            return
        item = self._list.item(row)
        if item:
            group = self._dir_manager.get_group_at(
                item.data(Qt.ItemDataRole.UserRole)
            )
            if group:
                total = self._dir_manager.dir_count()
                item.setText(self._item_text(key, group.photo_count(), total))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _item_text(self, key: str, count: int, total: int) -> str:
        if key in self._copied_keys:
            badge = " ✓"
        elif count < total:
            badge = f" ({count}/{total})"
        else:
            badge = ""
        return key + badge

    @staticmethod
    def _make_placeholder() -> QPixmap:
        px = QPixmap(THUMB_W, THUMB_H)
        px.fill(QColor(35, 35, 35))
        return px

    def _on_thumb_loaded(self, key: str, thumb_path: str):
        row = self._key_to_row.get(key)
        if row is None:
            return
        item = self._list.item(row)
        if item is None:
            return
        px = QPixmap(thumb_path).scaled(
            THUMB_W, THUMB_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if not px.isNull():
            item.setIcon(QIcon(px))

    def _apply_filter(self, text: str):
        text = text.strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            group_idx = item.data(Qt.ItemDataRole.UserRole)
            group = self._dir_manager.get_group_at(group_idx)
            match = (not text) or (text in group.key.lower())
            item.setHidden(not match)

    def _on_row_changed(self, row: int):
        if row < 0:
            return
        item = self._list.item(row)
        if item:
            self.group_selected.emit(item.data(Qt.ItemDataRole.UserRole))
