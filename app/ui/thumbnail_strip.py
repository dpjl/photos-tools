"""ui/thumbnail_strip.py — Bande de vignettes des étapes.

Chaque vignette représente le résultat d'une étape (ou l'image originale).
Clic gauche          → signal selected_a(step_id)
Clic droit           → menu contextuel :
                          • Afficher en B   → signal selected_b(step_id)
                          • Enregistrer…    → signal save_requested(step_id)
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QSizePolicy, QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QMouseEvent

THUMB_W = 96
THUMB_H = 72


def _make_thumb(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap(THUMB_W, THUMB_H)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    qimg = QImage(rgb.data.tobytes(), new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ThumbnailCard(QWidget):
    """Une vignette dans la bande de vignettes."""

    clicked_left  = pyqtSignal(str)  # step_id
    clicked_right = pyqtSignal(str)  # step_id — "Afficher en B"
    save_requested = pyqtSignal(str) # step_id — "Enregistrer…"

    def __init__(self, step_id: str, label: str, parent=None):
        super().__init__(parent)
        self._step_id    = step_id
        self._selected_a = False
        self._selected_b = False
        self._has_image  = False  # True une fois qu'une image réelle est chargée

        self.setFixedSize(THUMB_W + 8, THUMB_H + 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 2)
        layout.setSpacing(2)

        self._img_lbl = QLabel()
        self._img_lbl.setFixedSize(THUMB_W, THUMB_H)
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._img_lbl)

        self._text_lbl = QLabel(label)
        self._text_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._text_lbl)

        self._update_style()

    def set_image(self, img: Optional[np.ndarray]):
        pix = _make_thumb(img) if img is not None else QPixmap(THUMB_W, THUMB_H)
        self._img_lbl.setPixmap(pix)
        if img is not None:
            self._has_image = True

    def set_selected_a(self, val: bool):
        self._selected_a = val
        self._update_style()

    def set_selected_b(self, val: bool):
        self._selected_b = val
        self._update_style()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked_left.emit(self._step_id)
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())

    def _show_context_menu(self, global_pos):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #1e1e2e; color: #ccc; border: 1px solid #333; }"
            "QMenu::item:selected { background: #2a3a6a; }"
            "QMenu::item:disabled { color: #444; }"
            "QMenu::separator { background: #333; height: 1px; margin: 3px 6px; }"
        )

        set_b = menu.addAction("Afficher en B")
        menu.addSeparator()
        save_act = menu.addAction("Enregistrer…")
        if not self._has_image:
            save_act.setEnabled(False)

        chosen = menu.exec(global_pos)
        if chosen == set_b:
            self.clicked_right.emit(self._step_id)
        elif chosen == save_act:
            self.save_requested.emit(self._step_id)

    def _update_style(self):
        if self._selected_a and self._selected_b:
            bc = "#9b59b6"; bg = "#211630"; tc = "#c39bd3"
        elif self._selected_a:
            bc = "#4a9eff"; bg = "#162040"; tc = "#7ab8ff"
        elif self._selected_b:
            bc = "#ff5555"; bg = "#2a1010"; tc = "#ff8888"
        else:
            bc = "transparent"; bg = "#1a1a2e"; tc = "#aaa"
        # Border sur le QLabel : rendu garanti (contrairement au QWidget parent)
        # border: 2px transparent réservé même à l'état neutre → pas de saut visuel
        self._img_lbl.setStyleSheet(
            f"background: #111; border: 2px solid {bc}; border-radius: 2px;"
        )
        self._text_lbl.setStyleSheet(f"color: {tc}; font-size: 9px;")
        self.setStyleSheet(
            f"ThumbnailCard {{ background: {bg}; border-radius: 4px; }}"
            "ThumbnailCard:hover { background: #252535; }"
        )


class ThumbnailStrip(QWidget):
    """Bande horizontale de vignettes (une par étape + originale).

    La barre de titre affiche la version en cours de navigation.
    Clic gauche vignette → signal selected_a(step_id)
    Clic droit  vignette → signal selected_b(step_id)
    """

    selected_a = pyqtSignal(str)   # step_id (ou "original")
    selected_b = pyqtSignal(str)
    save_requested = pyqtSignal(str)  # step_id — l'utilisateur veut sauvegarder cette vignette

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: dict[str, ThumbnailCard] = {}
        self._order: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # En-tête : label de version + aide contextuelle
        header = QWidget()
        header.setFixedHeight(18)
        header.setStyleSheet("background: #111124;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)

        self._version_lbl = QLabel()
        self._version_lbl.setStyleSheet("color: #555; font-size: 10px;")
        header_layout.addWidget(self._version_lbl)
        header_layout.addStretch()

        hint = QLabel("← clic gauche = A  •  clic droit = B")
        hint.setStyleSheet("color: #333; font-size: 9px;")
        header_layout.addWidget(hint)
        root.addWidget(header)

        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFixedHeight(THUMB_H + 38)
        self._scroll.setStyleSheet("border: none; background: #141420;")
        root.addWidget(self._scroll)

        self._inner = QWidget()
        self._inner_layout = QHBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(8, 0, 8, 0)
        self._inner_layout.setSpacing(6)
        self._inner_layout.addStretch()
        self._scroll.setWidget(self._inner)

    # ── API ──────────────────────────────────────────────────────────────────

    def rebuild(self, step_ids: list[str], step_short_names: dict[str, str]):
        """Reconstruit la bande avec les étapes données (+ original)."""
        # Vider
        for card in self._cards.values():
            self._inner_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._order.clear()

        # Original
        all_ids    = ["original"] + step_ids
        all_labels = {"original": "Original"} | step_short_names

        for sid in all_ids:
            card = ThumbnailCard(sid, all_labels.get(sid, sid))
            card.clicked_left.connect(self._on_click_left)
            card.clicked_right.connect(self._on_click_right)
            card.save_requested.connect(self.save_requested)
            self._cards[sid] = card
            self._order.append(sid)
            # Insérer avant le stretch
            self._inner_layout.insertWidget(len(self._order) - 1, card)

    def update_image(self, step_id: str, img: Optional[np.ndarray]):
        if step_id in self._cards:
            self._cards[step_id].set_image(img)

    def clear_images(self):
        for card in self._cards.values():
            card.set_image(None)

    def set_version_label(self, text: str):
        """Affiche la version en cours dans l'en-tête (ex. 'v3 — 14:32')."""
        self._version_lbl.setText(text)

    def set_active_a(self, step_id: Optional[str]):
        for sid, card in self._cards.items():
            card.set_selected_a(sid == step_id)

    def set_active_b(self, step_id: Optional[str]):
        for sid, card in self._cards.items():
            card.set_selected_b(sid == step_id)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_click_left(self, step_id: str):
        self.set_active_a(step_id)
        self.selected_a.emit(step_id)

    def _on_click_right(self, step_id: str):
        self.set_active_b(step_id)
        self.selected_b.emit(step_id)
