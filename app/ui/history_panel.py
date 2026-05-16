"""ui/history_panel.py — Bande de chips pour naviguer dans l'historique des runs.

Clic gauche  → signal version_activated(run_id) — charger les vignettes de ce run

Le clic droit sur un chip est intentionnellement ignoré : la source B se définit
via clic droit sur une vignette dans le ThumbnailStrip, ce qui est plus direct.
"""

from __future__ import annotations
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent

from core.history import HistoryEntry


class HistoryChip(QWidget):
    """Un chip représentant un run dans l'historique.

    Un chip est soit « actif » (bleu = version dont les vignettes sont affichées)
    soit neutre. Les informations A/B sont portées par les labels des vues.
    """

    clicked = pyqtSignal(int)  # run_id — clic gauche uniquement

    def __init__(self, entry: HistoryEntry, parent=None):
        super().__init__(parent)
        self._run_id = entry.run_id
        self._active = False

        self.setFixedHeight(32)
        self.setMinimumWidth(90)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(4)

        run_label = QLabel(f"v{entry.run_id}")
        run_label.setStyleSheet("color: #ddd; font-weight: 600; font-size: 11px;")
        layout.addWidget(run_label)

        time_label = QLabel(entry.time_str)
        time_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(time_label)

        dots = "●" * len(entry.completed_steps)
        dot_lbl = QLabel(dots)
        dot_lbl.setStyleSheet("color: #2ecc71; font-size: 8px; letter-spacing: 2px;")
        layout.addWidget(dot_lbl)

        self._update_style()

    def set_active(self, val: bool):
        self._active = val
        self._update_style()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._run_id)

    def _update_style(self):
        if self._active:
            border = "2px solid #3a7bd5;"
            bg     = "#1a2a4a"
        else:
            border = "1px solid #333;"
            bg     = "#1e1e2e"
        self.setStyleSheet(
            f"HistoryChip {{ background: {bg}; border: {border} border-radius: 5px; }}"
            "HistoryChip:hover { background: #252535; }"
        )


class HistoryPanel(QWidget):
    """Bande horizontale de chips représentant l'historique des runs."""

    version_activated = pyqtSignal(int)  # run_id du chip cliqué

    def __init__(self, parent=None):
        super().__init__(parent)
        self._chips:     dict[int, HistoryChip] = {}
        self._active_id: Optional[int]          = None

        self.setFixedHeight(38)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("Historique :")
        lbl.setStyleSheet("color: #666; font-size: 10px; padding: 0 8px;")
        outer.addWidget(lbl)

        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("border: none; background: transparent;")
        outer.addWidget(self._scroll, stretch=1)

        self._inner = QWidget()
        self._inner_layout = QHBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(6)
        self._inner_layout.addStretch()
        self._scroll.setWidget(self._inner)

    # ── API ──────────────────────────────────────────────────────────────────

    def add_entry(self, entry: HistoryEntry):
        chip = HistoryChip(entry)
        chip.clicked.connect(self._on_activate)
        self._chips[entry.run_id] = chip
        stretch_idx = self._inner_layout.count() - 1
        self._inner_layout.insertWidget(stretch_idx, chip)
        self._scroll.horizontalScrollBar().setValue(
            self._scroll.horizontalScrollBar().maximum()
        )

    def set_active(self, run_id: Optional[int]):
        """Marque le chip `run_id` comme actif (bleu), désactive les autres."""
        if self._active_id is not None and self._active_id in self._chips:
            self._chips[self._active_id].set_active(False)
        self._active_id = run_id
        if run_id is not None and run_id in self._chips:
            self._chips[run_id].set_active(True)

    def clear(self):
        """Supprime tous les chips (nouvelle image chargée)."""
        for chip in self._chips.values():
            self._inner_layout.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()
        self._active_id = None

    # ── Slot ─────────────────────────────────────────────────────────────────

    def _on_activate(self, run_id: int):
        self.set_active(run_id)
        self.version_activated.emit(run_id)
