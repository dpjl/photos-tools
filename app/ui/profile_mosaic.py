"""ui/profile_mosaic.py — Grille 2×2 comparant les 4 profils AutoColor.

Chaque cellule affiche l'aperçu rapide d'un profil (naturel, neutre, classique,
actuel). Un clic sur une cellule émet ``profile_selected(str)`` et met en
évidence la cellule sélectionnée avec un encadré bleu.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from ui.image_view import ndarray_to_qpixmap


# ──────────────────────────────────────────────────────────────────────────────
# Cellule individuelle
# ──────────────────────────────────────────────────────────────────────────────

class ProfileCell(QFrame):
    """Un aperçu + étiquette pour un profil AutoColor."""

    clicked = pyqtSignal()

    def __init__(self, profile_id: str, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._profile   = profile_id
        self._orig_pix: Optional[QPixmap] = None

        self.setObjectName("ProfileCell")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumSize(180, 160)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(3, 3, 3, 3)
        vbox.setSpacing(0)

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._img_lbl.setMinimumSize(100, 80)
        vbox.addWidget(self._img_lbl, stretch=1)

        self._name_lbl = QLabel(label)
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_lbl.setFixedHeight(22)
        vbox.addWidget(self._name_lbl)

        self.set_selected(False)

    # ── Image ─────────────────────────────────────────────────────────────────

    def set_image(self, img: np.ndarray) -> None:
        """Affiche l'image (BGR ndarray)."""
        self._orig_pix = ndarray_to_qpixmap(img)
        self._img_lbl.setText("")
        self._rescale_pix()

    def set_computing(self) -> None:
        """Passe en mode 'calcul en cours'."""
        self._orig_pix = None
        self._img_lbl.setPixmap(QPixmap())
        self._img_lbl.setText("Calcul…")
        self._img_lbl.setStyleSheet("color:#555; font-size:11px;")

    def _rescale_pix(self) -> None:
        if self._orig_pix is None or self._orig_pix.isNull():
            return
        avail = self._img_lbl.size()
        if avail.width() <= 0 or avail.height() <= 0:
            return
        scaled = self._orig_pix.scaled(
            avail,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._img_lbl.setPixmap(scaled)
        self._img_lbl.setStyleSheet("")

    # ── Sélection visuelle ────────────────────────────────────────────────────

    def set_selected(self, selected: bool) -> None:
        border_color = "#3a7bd5" if selected else "#252540"
        border_width = 3 if selected else 2
        bg_color     = "#1e2540" if selected else "#16162a"
        lbl_color    = "#5aabf0" if selected else "#888"
        lbl_weight   = "bold"    if selected else "normal"

        self.setStyleSheet(f"""
            QFrame#ProfileCell {{
                border: {border_width}px solid {border_color};
                border-radius: 5px;
                background: {bg_color};
            }}
        """)
        self._name_lbl.setStyleSheet(
            f"color:{lbl_color}; font-size:11px; font-weight:{lbl_weight};"
            " background:transparent;"
        )

    # ── Événements Qt ─────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._rescale_pix()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# Widget mosaïque 2×2
# ──────────────────────────────────────────────────────────────────────────────

# Ordre et libellés affichés dans les cellules
_PROFILES: list[tuple[str, str]] = [
    ("naturel",   "Naturel"),
    ("neutre",    "Neutre"),
    ("classique", "Classique"),
    ("actuel",    "Actuel"),
]


class ProfileMosaicWidget(QWidget):
    """Grille 2×2 montrant les 4 profils AutoColor côte à côte.

    Signaux
    -------
    profile_selected(str) : nom du profil cliqué (« naturel », …).
    """

    profile_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cells: dict[str, ProfileCell] = {}
        self._selected: str = "actuel"
        self._build()

    def _build(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        for i, (pid, label) in enumerate(_PROFILES):
            row, col = divmod(i, 2)
            cell = ProfileCell(pid, label)
            cell.clicked.connect(lambda p=pid: self.profile_selected.emit(p))
            grid.addWidget(cell, row, col)
            self._cells[pid] = cell

    # ── API publique ──────────────────────────────────────────────────────────

    def set_images(self, images: dict[str, np.ndarray]) -> None:
        """Met à jour les 4 cellules avec les images calculées."""
        for pid, img in images.items():
            if pid in self._cells:
                self._cells[pid].set_image(img)

    def set_selected(self, profile: str) -> None:
        """Met en évidence la cellule du profil sélectionné."""
        self._selected = profile
        for pid, cell in self._cells.items():
            cell.set_selected(pid == profile)

    def set_computing(self, computing: bool) -> None:
        """Affiche/masque l'indicateur de calcul sur toutes les cellules."""
        if computing:
            for cell in self._cells.values():
                cell.set_computing()
