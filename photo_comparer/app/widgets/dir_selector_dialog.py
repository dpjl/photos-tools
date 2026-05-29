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
    QCheckBox,
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
        current_enabled: List[bool] = None,
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
        self._dir_checks: List[QCheckBox] = []
        self._dir_browses: List[QPushButton] = []
        self._count_labels: List[QLabel] = []

        _ARROW_STYLE = (
            "QPushButton { background:#2e2e2e; color:#aaa; border:1px solid #444;"
            " border-radius:3px; padding:1px 4px; font-size:11px; }"
            "QPushButton:hover { background:#3d3d3d; color:#fff; }"
            "QPushButton:disabled { color:#444; border-color:#333; }"
        )
        _CHECK_STYLE = (
            "QCheckBox { spacing:0px; }"
            "QCheckBox::indicator { width:14px; height:14px; border:1px solid #555;"
            " border-radius:2px; background:#2a2a2a; }"
            "QCheckBox::indicator:checked { background:#2e7a2e; border-color:#4CAF50; }"
        )

        for i in range(MAX_INPUT_DIRS):
            enabled = (current_enabled[i] if current_enabled and i < len(current_enabled) else True)

            chk = QCheckBox()
            chk.setChecked(enabled)
            chk.setToolTip("Activer / désactiver ce répertoire")
            chk.setStyleSheet(_CHECK_STYLE)
            chk.setFixedWidth(20)
            chk.stateChanged.connect(lambda state, idx=i: self._toggle_row_enabled(idx))

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

            btn_up = QPushButton("▲")
            btn_up.setFixedWidth(24)
            btn_up.setStyleSheet(_ARROW_STYLE)
            btn_up.setToolTip("Monter")
            btn_up.setEnabled(i > 0)
            btn_up.clicked.connect(lambda _, idx=i: self._swap_rows(idx, idx - 1))

            btn_down = QPushButton("▼")
            btn_down.setFixedWidth(24)
            btn_down.setStyleSheet(_ARROW_STYLE)
            btn_down.setToolTip("Descendre")
            btn_down.setEnabled(i < MAX_INPUT_DIRS - 1)
            btn_down.clicked.connect(lambda _, idx=i: self._swap_rows(idx, idx + 1))

            grid.addWidget(chk,      i, 0)
            grid.addWidget(lbl,      i, 1)
            grid.addWidget(edit,     i, 2)
            grid.addWidget(browse,   i, 3)
            grid.addWidget(count,    i, 4)
            grid.addWidget(btn_up,   i, 5)
            grid.addWidget(btn_down, i, 6)

            self._dir_checks.append(chk)
            self._dir_edits.append(edit)
            self._dir_browses.append(browse)
            self._count_labels.append(count)

            if not enabled:
                self._toggle_row_enabled(i)

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
        """Return only enabled non-empty dirs (used by DirectoryManager)."""
        return [
            e.text().strip()
            for e, chk in zip(self._dir_edits, self._dir_checks)
            if e.text().strip() and chk.isChecked()
        ]

    def get_all_dir_data(self) -> List[dict]:
        """Return all non-empty rows as [{path, enabled}] — used for persistence."""
        result = []
        for e, chk in zip(self._dir_edits, self._dir_checks):
            path = e.text().strip()
            if path:
                result.append({"path": path, "enabled": chk.isChecked()})
        return result

    def get_output_dir(self) -> str:
        return self._out_edit.text().strip()

    # ------------------------------------------------------------------

    def _toggle_row_enabled(self, idx: int):
        enabled = self._dir_checks[idx].isChecked()
        alpha = "ff" if enabled else "55"
        self._dir_edits[idx].setEnabled(enabled)
        self._dir_edits[idx].setStyleSheet(
            _EDIT_STYLE if enabled
            else "QLineEdit { background:#181818; color:#555; border:1px solid #333;"
                 " border-radius:3px; padding:3px; }"
        )
        self._dir_browses[idx].setEnabled(enabled)
        self._count_labels[idx].setStyleSheet(
            "color:#666; font-size:10px;" if enabled
            else "color:#444; font-size:10px;"
        )

    def _swap_rows(self, a: int, b: int):
        if b < 0 or b >= MAX_INPUT_DIRS:
            return
        text_a = self._dir_edits[a].text()
        text_b = self._dir_edits[b].text()
        self._dir_edits[a].setText(text_b)
        self._dir_edits[b].setText(text_a)
        # Also swap checkbox states (triggers _toggle_row_enabled via signal)
        checked_a = self._dir_checks[a].isChecked()
        checked_b = self._dir_checks[b].isChecked()
        self._dir_checks[a].setChecked(checked_b)
        self._dir_checks[b].setChecked(checked_a)

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
                    1 for f in p.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                    and not f.name.lower().endswith(".mask.png")
                )
                lbl.setText(f"{count} images")
                lbl.setStyleSheet("color:#4CAF50; font-size:10px;")
            except PermissionError:
                lbl.setText("accès refusé")
                lbl.setStyleSheet("color:#f44336; font-size:10px;")
        else:
            lbl.setText("❌ invalide")
            lbl.setStyleSheet("color:#f44336; font-size:10px;")
