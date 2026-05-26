"""DirSelectorDialog — choose up to MAX_INPUT_DIRS source directories
and one output directory.

Each row shows:
  • An editable path field + Browse button
  • A live file-count label (green = valid dir, red = invalid)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..models.directory_manager import IMAGE_EXTENSIONS

MAX_INPUT_DIRS = 10
_BTN_STYLE = (
    "QPushButton { background:#3a3a3a; color:#ccc; border:1px solid #555;"
    " border-radius:3px; padding:3px 6px; }"
    "QPushButton:hover { background:#4a4a4a; }"
)
_EDIT_STYLE = (
    "QLineEdit { background:#1e1e1e; color:#ccc; border:1px solid #444;"
    " border-radius:3px; padding:3px; }"
)
_GROUP_STYLE = (
    "QGroupBox { color:#999; border:1px solid #444; border-radius:4px;"
    " margin-top:8px; padding-top:6px; }"
    "QGroupBox::title { subcontrol-origin:margin; left:8px; }"
)


class DirSelectorDialog(QDialog):
    def __init__(
        self,
        current_dirs: List[str],
        output_dir: str,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Sélection des répertoires")
        self.setMinimumSize(660, 560)
        self.setStyleSheet("background:#262626; color:#ccc;")

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # --- Source directories ---
        src_group = QGroupBox(f"Répertoires source (jusqu'à {MAX_INPUT_DIRS})")
        src_group.setStyleSheet(_GROUP_STYLE)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:#1e1e1e; border:none;")
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setSpacing(4)
        grid.setContentsMargins(6, 4, 6, 4)

        self._dir_edits: List[QLineEdit] = []
        self._count_labels: List[QLabel] = []

        for i in range(MAX_INPUT_DIRS):
            lbl = QLabel(f"Dir {i + 1}:")
            lbl.setStyleSheet("color:#777; font-size:10px;")
            lbl.setFixedWidth(36)

            edit = QLineEdit(current_dirs[i] if i < len(current_dirs) else "")
            edit.setStyleSheet(_EDIT_STYLE)
            edit.setPlaceholderText("(vide = ignoré)")
            edit.textChanged.connect(lambda text, idx=i: self._update_count(idx))

            browse = QPushButton("…")
            browse.setFixedWidth(28)
            browse.setStyleSheet(_BTN_STYLE)
            browse.clicked.connect(lambda _, idx=i: self._browse_src(idx))

            count = QLabel("")
            count.setFixedWidth(88)
            count.setStyleSheet("color:#666; font-size:10px;")

            grid.addWidget(lbl, i, 0)
            grid.addWidget(edit, i, 1)
            grid.addWidget(browse, i, 2)
            grid.addWidget(count, i, 3)

            self._dir_edits.append(edit)
            self._count_labels.append(count)

        scroll.setWidget(inner)
        src_layout = QVBoxLayout(src_group)
        src_layout.addWidget(scroll)
        root.addWidget(src_group)

        # --- Output directory ---
        out_group = QGroupBox("Répertoire de sortie (meilleures photos)")
        out_group.setStyleSheet(_GROUP_STYLE)
        out_row = QHBoxLayout(out_group)
        out_row.setContentsMargins(6, 4, 6, 4)

        self._out_edit = QLineEdit(output_dir)
        self._out_edit.setStyleSheet(_EDIT_STYLE)
        self._out_edit.setPlaceholderText("Dossier de destination…")
        out_browse = QPushButton("…")
        out_browse.setFixedWidth(28)
        out_browse.setStyleSheet(_BTN_STYLE)
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(self._out_edit)
        out_row.addWidget(out_browse)
        root.addWidget(out_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet(
            "QPushButton { background:#3a3a3a; color:#ccc; border:1px solid #555;"
            " border-radius:3px; padding:4px 14px; }"
            "QPushButton:hover { background:#4a4a4a; }"
            "QPushButton:default { border-color:#5a9fd4; }"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        # Initial count refresh for pre-filled paths
        for i in range(MAX_INPUT_DIRS):
            if self._dir_edits[i].text():
                self._update_count(i)

    # ------------------------------------------------------------------

    def get_input_dirs(self) -> List[str]:
        return [e.text().strip() for e in self._dir_edits if e.text().strip()]

    def get_output_dir(self) -> str:
        return self._out_edit.text().strip()

    # ------------------------------------------------------------------

    def _browse_src(self, idx: int):
        start = self._dir_edits[idx].text() or ""
        path = QFileDialog.getExistingDirectory(
            self, f"Répertoire source {idx + 1}", start
        )
        if path:
            self._dir_edits[idx].setText(path)

    def _browse_out(self):
        start = self._out_edit.text() or ""
        path = QFileDialog.getExistingDirectory(
            self, "Répertoire de sortie", start
        )
        if path:
            self._out_edit.setText(path)

    def _update_count(self, idx: int):
        text = self._dir_edits[idx].text().strip()
        lbl = self._count_labels[idx]
        if not text:
            lbl.setText("")
            return
        p = Path(text)
        if p.is_dir():
            try:
                count = sum(
                    1 for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
                )
                lbl.setText(f"{count} images")
                lbl.setStyleSheet("color:#4CAF50; font-size:10px;")
            except PermissionError:
                lbl.setText("accès refusé")
                lbl.setStyleSheet("color:#f44336; font-size:10px;")
        else:
            lbl.setText("❌ invalide")
            lbl.setStyleSheet("color:#f44336; font-size:10px;")
