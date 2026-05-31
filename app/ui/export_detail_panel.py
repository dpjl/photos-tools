"""ui/export_detail_panel.py — Panneau latéral de détail d'un export.

Affiche les informations, la comparaison de paramètres, et les actions
pour un export sélectionné dans la mosaïque.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea,
    QMessageBox, QSizePolicy, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from core.export_manager import ExportManager, ExportEntry
from ui.param_compare import ParamCompareWidget


class ExportDetailPanel(QWidget):
    """Panneau latéral pour un export sélectionné.

    Signals:
        restore_requested(ExportEntry) — demande de restauration des paramètres
        delete_requested(ExportEntry)  — demande de suppression (après confirmation)
        export_deleted(ExportEntry)    — export effectivement supprimé
    """

    restore_requested = pyqtSignal(object)  # ExportEntry
    delete_requested  = pyqtSignal(object)  # ExportEntry
    export_deleted    = pyqtSignal(object)  # ExportEntry
    best_changed      = pyqtSignal(object, int)  # ExportEntry, index (or 0 to clear)

    _WIDTH = 240

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self._WIDTH)
        self.setStyleSheet("background: #1a1a2e;")

        self._entry: Optional[ExportEntry] = None
        self._export_manager: Optional[ExportManager] = None
        self._all_entries: list[ExportEntry] = []
        self._best_index: Optional[int] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        # ── En-tête ──
        self._title = QLabel()
        self._title.setWordWrap(True)
        self._title.setStyleSheet(
            "color: #dde; font-size: 14px; font-weight: bold;"
        )
        lay.addWidget(self._title)

        self._date_label = QLabel()
        self._date_label.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(self._date_label)

        # ── Séparateur ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2a2a4a;")
        lay.addWidget(sep)

        # ── Zone de comparaison de paramètres ──
        self._param_compare = ParamCompareWidget()
        self._param_scroll = QScrollArea()
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._param_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        self._param_scroll.setWidget(self._param_compare)
        lay.addWidget(self._param_scroll, stretch=1)

        # ── Séparateur ──
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #2a2a4a;")
        lay.addWidget(sep2)

        # ── Actions ──
        self._best_btn = QPushButton("★ Retenir cet export")
        self._best_btn.setStyleSheet(
            "QPushButton { background:#2a2a1e; color:#da3; border:1px solid #4a4a2a;"
            "  border-radius:4px; padding:8px; font-size:11px; }"
            "QPushButton:hover { background:#3a3a2a; color:#fd6; }"
        )
        self._best_btn.clicked.connect(self._on_best)
        lay.addWidget(self._best_btn)

        self._restore_btn = QPushButton("↩ Restaurer les paramètres")
        self._restore_btn.setStyleSheet(
            "QPushButton { background:#1e2a3e; color:#8ac; border:1px solid #2a4a6a;"
            "  border-radius:4px; padding:8px; font-size:11px; }"
            "QPushButton:hover { background:#2a3a5e; color:#bdf; }"
        )
        self._restore_btn.clicked.connect(self._on_restore)
        lay.addWidget(self._restore_btn)

        self._delete_btn = QPushButton("🗑 Supprimer l'export")
        self._delete_btn.setStyleSheet(
            "QPushButton { background:#2e1a1a; color:#c88; border:1px solid #4a2a2a;"
            "  border-radius:4px; padding:8px; font-size:11px; }"
            "QPushButton:hover { background:#3e2a2a; color:#faa; }"
        )
        self._delete_btn.clicked.connect(self._on_delete)
        lay.addWidget(self._delete_btn)

    # ── API publique ──────────────────────────────────────────────────────────

    def set_export_manager(self, mgr: ExportManager):
        self._export_manager = mgr

    def set_entry(
        self,
        entry: Optional[ExportEntry],
        all_entries: Optional[list[ExportEntry]] = None,
    ):
        """Affiche les détails d'un export. None pour vider."""
        self._entry = entry
        self._all_entries = all_entries or []

        if entry is None:
            self._title.setText("")
            self._date_label.setText("")
            self._param_compare.set_data(None, [])
            return

        self._title.setText(f"Export {entry.index:03d}")

        date_str = ""
        if entry.exported_at:
            try:
                dt = datetime.fromisoformat(entry.exported_at)
                date_str = dt.strftime("%d/%m/%Y à %H:%M:%S")
            except Exception:
                date_str = entry.exported_at
        self._date_label.setText(date_str)

        self._update_best_btn()
        self._build_param_comparison(entry, all_entries or [])

    # ── Comparaison de paramètres ─────────────────────────────────────────────

    def _build_param_comparison(
        self,
        entry: ExportEntry,
        all_entries: list[ExportEntry],
    ):
        """Délègue à ParamCompareWidget."""
        self._param_compare.set_data(entry, all_entries)

    def set_best_index(self, best_index: Optional[int]):
        """Met à jour l'état du bouton 'Retenir'."""
        self._best_index = best_index
        self._update_best_btn()

    def _update_best_btn(self):
        if self._entry and self._best_index == self._entry.index:
            self._best_btn.setText("★ Retenu")
            self._best_btn.setStyleSheet(
                "QPushButton { background:#3a3a1e; color:#fd6; border:1px solid #6a6a3a;"
                "  border-radius:4px; padding:8px; font-size:11px; font-weight:bold; }"
                "QPushButton:hover { background:#4a4a2a; }"
            )
        else:
            self._best_btn.setText("☆ Retenir cet export")
            self._best_btn.setStyleSheet(
                "QPushButton { background:#2a2a1e; color:#da3; border:1px solid #4a4a2a;"
                "  border-radius:4px; padding:8px; font-size:11px; }"
                "QPushButton:hover { background:#3a3a2a; color:#fd6; }"
            )

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_best(self):
        if self._entry:
            self.best_changed.emit(self._entry, self._entry.index)

    def _on_restore(self):
        if self._entry:
            self.restore_requested.emit(self._entry)

    def _on_delete(self):
        if not self._entry or not self._export_manager:
            return
        reply = QMessageBox.question(
            self,
            "Supprimer l'export",
            f"Supprimer l'export {self._entry.index:03d} et tous ses fichiers associés ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._export_manager.delete_export(self._entry)
            self.export_deleted.emit(self._entry)
