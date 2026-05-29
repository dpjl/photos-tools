"""MainWindow — application shell.

Layout
------
  [Toolbar]
  ┌─────────────┬─────────────────────────────────────────┐
  │  NavPanel   │          ComparisonView                 │
  └─────────────┴─────────────────────────────────────────┘
  [StatusBar]

Keyboard shortcuts
------------------
  ← / →        previous / next photo group
  ↑ / ↓        same (for single-hand operation)
  Escape        exit solo mode
  Ctrl+F        focus filter bar in nav panel
  Ctrl+Enter    copy best to output
  Ctrl+= / -   zoom in / out
  Ctrl+0        fit image
  1             zoom to 100 %
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..models.directory_manager import DirectoryManager
from ..utils import export_store
from .comparison_view import ComparisonView
from .dir_selector_dialog import DirSelectorDialog
from .nav_panel import NavPanel
from .note_panel import NotePanel

CONFIG_PATH = Path.home() / ".photo_comparer_config.json"
NOTES_PATH  = Path.home() / ".photo_comparer_notes.json"

_TOOLBAR_STYLE = """
QToolBar {
    background: #242424;
    border-bottom: 1px solid #3a3a3a;
    spacing: 4px;
    padding: 3px 6px;
}
QPushButton {
    background: #363636;
    color: #ccc;
    border: 1px solid #505050;
    border-radius: 3px;
    padding: 3px 10px;
    min-width: 28px;
}
QPushButton:hover  { background: #464646; }
QPushButton:pressed { background: #2a2a2a; }
QPushButton:disabled { color: #555; background: #2a2a2a; border-color: #3a3a3a; }
QLabel { color: #aaa; padding: 0 4px; }
"""

_COPY_BTN_ON = (
    "QPushButton { background:#1a5c1a; color:#ddd; border:1px solid #2e7a2e;"
    " border-radius:3px; padding:3px 12px; font-weight:bold; }"
    "QPushButton:hover { background:#2a7a2a; }"
)
_COPY_BTN_OFF = (
    "QPushButton { background:#2a2a2a; color:#555; border:1px solid #3a3a3a;"
    " border-radius:3px; padding:3px 12px; }"
)

_MODE_BTN_ACTIVE = (
    "QPushButton { background:#1a3a6a; color:#8ac4ff; border:1px solid #2a5a9a;"
    " border-radius:3px; padding:3px 10px; font-weight:bold; }"
    "QPushButton:hover { background:#234e8a; }"
)
_MODE_BTN_INACTIVE = (
    "QPushButton { background:#363636; color:#888; border:1px solid #505050;"
    " border-radius:3px; padding:3px 10px; }"
    "QPushButton:hover { background:#464646; color:#bbb; }"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Comparator")
        self.resize(1440, 900)

        self._dir_manager = DirectoryManager()
        self._current_group_index: int = -1
        self._best_path: Optional[Path] = None
        self._best_dir_index: int = -1   # dir index of the current best selection
        self._all_dir_data: list = []    # [{"path": str, "enabled": bool}, ...]
        self._notes: dict = {}           # group_key → note text

        # --- Widgets ---
        self._nav = NavPanel(self._dir_manager, self)
        self._cmp = ComparisonView(self._dir_manager, self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._nav)
        splitter.addWidget(self._cmp)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([180, 1260])

        # Note panel (hidden by default, toggled via toolbar)
        self._note_panel = NotePanel()
        self._note_panel.hide()
        self._note_panel.note_changed.connect(self._on_note_changed)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(splitter)
        central_layout.addWidget(self._note_panel)
        self.setCentralWidget(central)

        # --- Toolbar ---
        self._setup_toolbar()

        # --- Status bar ---
        self._status = QStatusBar()
        self._status.setStyleSheet("color:#888; font-size:11px;")
        self.setStatusBar(self._status)

        # --- Signals ---
        self._nav.group_selected.connect(self._on_group_selected)
        self._cmp.best_changed.connect(self._on_best_changed)

        # --- Keyboard shortcuts ---
        self._setup_shortcuts()

        # --- Restore config ---
        self._load_notes()
        self._load_config()

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _setup_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setStyleSheet(_TOOLBAR_STYLE)
        self.addToolBar(tb)

        # Directories button
        dirs_btn = QPushButton("Répertoires…")
        dirs_btn.setToolTip("Choisir les répertoires à comparer (Ctrl+D)")
        dirs_btn.clicked.connect(self._open_dirs_dialog)
        tb.addWidget(dirs_btn)
        tb.addSeparator()

        # Mode toggle: Images / JSON
        self._mode_img_btn = QPushButton("Images")
        self._mode_img_btn.setToolTip("Afficher les images (Ctrl+I)")
        self._mode_img_btn.setStyleSheet(_MODE_BTN_ACTIVE)

        self._mode_json_btn = QPushButton("{ } JSON")
        self._mode_json_btn.setToolTip("Afficher les JSON associés (Ctrl+J)")
        self._mode_json_btn.setStyleSheet(_MODE_BTN_INACTIVE)

        self._mode_img_btn.clicked.connect(lambda: self._on_mode_changed("images"))
        self._mode_json_btn.clicked.connect(lambda: self._on_mode_changed("json"))

        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(2)
        mode_layout.addWidget(self._mode_img_btn)
        mode_layout.addWidget(self._mode_json_btn)
        tb.addWidget(mode_widget)
        tb.addSeparator()

        # Zoom controls
        def _add_btn(text, tip, slot, width=None):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            if width:
                b.setFixedWidth(width)
            tb.addWidget(b)
            return b

        _add_btn("−", "Zoom arrière (Ctrl+−)", self._cmp.zoom_out, 28)
        _add_btn("+", "Zoom avant (Ctrl++)", self._cmp.zoom_in, 28)
        _add_btn("Fit", "Ajuster à la vue (Ctrl+0)", self._cmp.zoom_fit)
        _add_btn("1:1", "Zoom 100 % (touche 1)", self._cmp.zoom_100)
        tb.addSeparator()

        # Navigation info label (stretchy spacer before copy button)
        self._info_lbl = QLabel("Aucune photo — ouvrir des répertoires")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        tb.addWidget(self._info_lbl)
        tb.addSeparator()

        # Notes toggle button
        self._notes_btn = QPushButton("\U0001f4dd Notes")
        self._notes_btn.setToolTip("Afficher / masquer les notes (N)")
        self._notes_btn.setStyleSheet(_MODE_BTN_INACTIVE)
        self._notes_btn.clicked.connect(self._toggle_notes_panel)
        tb.addWidget(self._notes_btn)
        tb.addSeparator()

        # Copy best button
        self._copy_btn = QPushButton("Copier la meilleure  →")
        self._copy_btn.setToolTip(
            "Copier la photo sélectionnée (Ctrl+Entrée) vers le répertoire de sortie"
        )
        self._copy_btn.setEnabled(False)
        self._copy_btn.setStyleSheet(_COPY_BTN_OFF)
        self._copy_btn.clicked.connect(self._copy_best)
        tb.addWidget(self._copy_btn)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _setup_shortcuts(self):
        def sc(seq, slot):
            s = QShortcut(QKeySequence(seq), self)
            s.activated.connect(slot)

        sc("Ctrl+D", self._open_dirs_dialog)
        sc("Ctrl+I", lambda: self._on_mode_changed("images"))
        sc("Ctrl+J", lambda: self._on_mode_changed("json"))
        sc("Ctrl+=", self._cmp.zoom_in)
        sc("Ctrl++", self._cmp.zoom_in)
        sc("Ctrl+-", self._cmp.zoom_out)
        sc("Ctrl+0", self._cmp.zoom_fit)
        sc("1",      self._cmp.zoom_100)
        sc("Ctrl+Return", self._copy_best)
        sc("Ctrl+F", lambda: self._nav._filter.setFocus())
        sc("N", self._toggle_notes_panel)

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self._navigate(+1)
        elif k in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self._navigate(-1)
        elif k == Qt.Key.Key_Escape:
            self._cmp.exit_solo()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def _open_dirs_dialog(self):
        # Build flat lists for the dialog from _all_dir_data
        all_paths   = [d["path"]    for d in self._all_dir_data]
        all_enabled = [d["enabled"] for d in self._all_dir_data]
        dlg = DirSelectorDialog(
            current_dirs=all_paths,
            current_enabled=all_enabled,
            output_dir=str(self._dir_manager.output_dir)
            if self._dir_manager.output_dir
            else "",
            parent=self,
        )
        if dlg.exec() != DirSelectorDialog.DialogCode.Accepted:
            return

        dirs = dlg.get_input_dirs()
        out = dlg.get_output_dir()
        self._all_dir_data = dlg.get_all_dir_data()
        self._dir_manager.set_directories(dirs, out or None)
        self._cmp.setup_directories()
        self._load_export_records()          # populate records before first nav
        self._nav.refresh()
        self._current_group_index = -1
        self._best_path = None
        self._best_dir_index = -1
        self._copy_btn.setEnabled(False)
        self._copy_btn.setStyleSheet(_COPY_BTN_OFF)
        self._update_info_label()
        self._save_config()

        # Auto-select first group
        if self._dir_manager.group_count() > 0:
            self._nav.select_group(0)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate(self, delta: int):
        total = self._dir_manager.group_count()
        if total == 0:
            return
        new_idx = max(0, min(total - 1, self._current_group_index + delta))
        if new_idx != self._current_group_index:
            self._nav.select_group(new_idx)   # fires group_selected → _on_group_selected

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_group_selected(self, group_index: int):
        self._current_group_index = group_index
        group = self._dir_manager.get_group_at(group_index)
        if group is None:
            return
        # Reset first — load_group may re-enable via best_changed if a best exists
        self._best_path = None
        self._best_dir_index = -1
        self._copy_btn.setEnabled(False)
        self._copy_btn.setStyleSheet(_COPY_BTN_OFF)
        self._cmp.load_group(group)
        total = self._dir_manager.group_count()
        self._info_lbl.setText(f"  {group.key}   ({group_index + 1} / {total})  ")
        # Update note panel for this group
        self._note_panel.set_group(group.key, self._notes.get(group.key, ""))

    def _on_best_changed(self, dir_index: int, path):
        self._best_path = path
        self._best_dir_index = dir_index
        can_copy = (
            path is not None and self._dir_manager.output_dir is not None
        )
        self._copy_btn.setEnabled(can_copy)
        self._copy_btn.setStyleSheet(_COPY_BTN_ON if can_copy else _COPY_BTN_OFF)
        if path:
            dir_name = (
                self._dir_manager.dir_names[dir_index]
                if 0 <= dir_index < len(self._dir_manager.dir_names)
                else "?"
            )
            self._status.showMessage(
                f"Sélectionnée : {path.name}  [{dir_name}]  — Ctrl+Entrée ou bouton pour copier"
            )
        else:
            self._status.clearMessage()

    def _toggle_notes_panel(self):
        visible = not self._note_panel.isVisible()
        self._note_panel.setVisible(visible)
        self._notes_btn.setStyleSheet(
            _MODE_BTN_ACTIVE if visible else _MODE_BTN_INACTIVE
        )

    def _on_note_changed(self, key: str, text: str):
        if text:
            self._notes[key] = text
        else:
            self._notes.pop(key, None)
        self._save_notes()

    def _on_mode_changed(self, mode: str):
        is_json = (mode == "json")
        self._mode_img_btn.setStyleSheet(
            _MODE_BTN_INACTIVE if is_json else _MODE_BTN_ACTIVE
        )
        self._mode_json_btn.setStyleSheet(
            _MODE_BTN_ACTIVE if is_json else _MODE_BTN_INACTIVE
        )
        self._cmp.set_mode(mode)

    def _copy_best(self):
        if not self._best_path:
            return
        if not self._dir_manager.output_dir:
            QMessageBox.warning(
                self,
                "Pas de répertoire de sortie",
                "Veuillez définir un répertoire de sortie dans 'Répertoires…'.",
            )
            return

        out_dir = self._dir_manager.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / self._best_path.name

        if dest.exists():
            reply = QMessageBox.question(
                self,
                "Fichier existant",
                f"« {dest.name} » existe déjà.\nRemplacer ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            shutil.copy2(str(self._best_path), str(dest))
        except OSError as exc:
            QMessageBox.critical(self, "Erreur de copie", str(exc))
            return

        group = self._dir_manager.get_group_at(self._current_group_index)
        if group:
            self._nav.mark_copied(group.key)
            # Save sidecar so the selection survives future sessions
            dir_name = (
                self._dir_manager.dir_names[self._best_dir_index]
                if 0 <= self._best_dir_index < len(self._dir_manager.dir_names)
                else "?"
            )
            record = export_store.ExportRecord(
                group_key=group.key,
                source_path=str(self._best_path),
                dir_name=dir_name,
                output_filename=dest.name,
            )
            export_store.save(out_dir, record)
            self._cmp.add_export_record(record)

        self._status.showMessage(f"✓ Copié : {dest}", 5000)
        self._copy_btn.setEnabled(False)
        self._copy_btn.setStyleSheet(_COPY_BTN_OFF)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_info_label(self):
        count = self._dir_manager.group_count()
        if count == 0:
            self._info_lbl.setText("Aucune photo — ouvrir des répertoires")
        else:
            self._info_lbl.setText(f"{count} groupes de photos")

    def _load_export_records(self) -> None:
        """Scan the output dir for sidecars and sync nav badges."""
        copied_keys = self._cmp.load_export_records(
            self._dir_manager.output_dir
        )
        self._nav.set_copied_keys(copied_keys)

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _load_config(self):
        try:
            if not CONFIG_PATH.exists():
                return
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # Support new format (dir_data) and legacy format (input_dirs)
            if "dir_data" in cfg:
                self._all_dir_data = cfg["dir_data"]
                dirs = [d["path"] for d in self._all_dir_data if d.get("enabled", True) and Path(d["path"]).is_dir()]
            else:
                dirs = [d for d in cfg.get("input_dirs", []) if Path(d).is_dir()]
                self._all_dir_data = [{"path": d, "enabled": True} for d in dirs]
            out = cfg.get("output_dir", "")
            if not dirs:
                return
            self._dir_manager.set_directories(dirs, out or None)
            self._cmp.setup_directories()
            self._load_export_records()      # restore selections from sidecars
            self._nav.refresh()
            self._update_info_label()
            if self._dir_manager.group_count() > 0:
                self._nav.select_group(0)
        except Exception:
            pass   # Config errors are non-fatal

    def _save_config(self):
        try:
            cfg = {
                "dir_data": self._all_dir_data,
                "output_dir": str(self._dir_manager.output_dir)
                if self._dir_manager.output_dir
                else "",
            }
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_notes(self):
        try:
            if NOTES_PATH.exists():
                with open(NOTES_PATH, "r", encoding="utf-8") as f:
                    self._notes = json.load(f)
        except Exception:
            self._notes = {}

    def _save_notes(self):
        try:
            with open(NOTES_PATH, "w", encoding="utf-8") as f:
                json.dump(self._notes, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def closeEvent(self, event):
        self._note_panel.flush()
        self._save_config()
        super().closeEvent(event)
