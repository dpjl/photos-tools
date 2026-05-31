"""ui/export_mosaic.py — Vue mosaïque des exports versionnés.

Affiche les exports dans une grille de SyncedImageView synchronisées.
La grille s'adapte automatiquement au nombre d'exports pour remplir
tout l'espace disponible.  Double-clic → mode plein format.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QSizePolicy, QStackedLayout, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThreadPool, QRunnable, QObject, QTimer
from PyQt6.QtGui import QMouseEvent

from core.export_manager import ExportEntry
from ui.image_view import SyncedImageView, ndarray_to_qpixmap


# ══════════════════════════════════════════════════════════════════════════════
# Chargement asynchrone des images
# ══════════════════════════════════════════════════════════════════════════════

class _LoadSignals(QObject):
    loaded = pyqtSignal(int, object)  # index, ndarray (BGR) or None


class _LoadTask(QRunnable):
    """Charge une image d'export en arrière-plan."""

    def __init__(self, index: int, image_path: str):
        super().__init__()
        self.index = index
        self.image_path = image_path
        self.signals = _LoadSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            self.signals.loaded.emit(self.index, img)
        except Exception:
            self.signals.loaded.emit(self.index, None)


# ══════════════════════════════════════════════════════════════════════════════
# Cellule de la mosaïque — SyncedImageView + label discret
# ══════════════════════════════════════════════════════════════════════════════

class ExportTile(QFrame):
    """Une cellule contenant un SyncedImageView plein format et un label."""

    clicked = pyqtSignal(object)           # ExportEntry
    double_clicked = pyqtSignal(object)    # ExportEntry

    _BORDER_NORMAL = "2px solid #2a2a4a"
    _BORDER_SELECTED = "2px solid #3a8fd4"

    def __init__(self, entry: ExportEntry, parent=None):
        super().__init__(parent)
        self.entry = entry
        self._selected = False

        self.setFrameShape(QFrame.Shape.NoFrame)
        self._update_style()
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Vue image (SyncedImageView)
        self.viewer = SyncedImageView()
        self.viewer.setStyleSheet("border: none; background: #0e0e1a;")
        lay.addWidget(self.viewer, stretch=1)

        # Label discret en overlay bas
        date_str = ""
        if entry.exported_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(entry.exported_at)
                date_str = dt.strftime("%d/%m/%Y %H:%M")
            except Exception:
                date_str = entry.exported_at
        label_text = f"Export {entry.index:03d}"
        if date_str:
            label_text += f" — {date_str}"
        self._label = QLabel(label_text)
        self._label.setFixedHeight(18)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "color: #888; font-size: 10px; background: #12122a;"
            " border: none; padding: 0 4px;"
        )
        lay.addWidget(self._label)

        # Badge "retenu" (étoile en haut à droite, masqué par défaut)
        self._best_badge = QLabel("★")
        self._best_badge.setFixedSize(22, 22)
        self._best_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._best_badge.setStyleSheet(
            "background: #d4a03a; color: #fff; font-size: 14px;"
            " border-radius: 11px; border: none;"
        )
        self._best_badge.setVisible(False)
        # Positionner en overlay via le parent
        self._best_badge.setParent(self)

        # Intercepter les clics sur le viewer
        self.viewer.viewport().installEventFilter(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Repositionner le badge en haut à droite
        self._best_badge.move(self.width() - 28, 6)

    def set_best(self, is_best: bool):
        """Affiche ou masque le badge 'retenu'."""
        self._best_badge.setVisible(is_best)
        self._best_badge.raise_()

    def set_image(self, img: np.ndarray):
        """Affiche l'image complète dans le viewer."""
        self.viewer.set_image(img)

    def set_selected(self, selected: bool):
        if self._selected != selected:
            self._selected = selected
            self._update_style()

    def _update_style(self):
        border = self._BORDER_SELECTED if self._selected else self._BORDER_NORMAL
        self.setStyleSheet(
            f"ExportTile {{ background: #12122a; border: {border}; border-radius: 3px; }}"
        )

    def eventFilter(self, obj, event):
        """Intercepte les événements souris du viewport du viewer."""
        from PyQt6.QtCore import QEvent
        if obj is self.viewer.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.clicked.emit(self.entry)
            elif event.type() == QEvent.Type.MouseButtonDblClick:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.double_clicked.emit(self.entry)
                    return True  # consommer le double-clic
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Vue mosaïque principale
# ══════════════════════════════════════════════════════════════════════════════

class ExportMosaicView(QWidget):
    """Affiche les exports en grille de SyncedImageView synchronisées.

    La grille s'adapte :
      1 export  → 1×1 (plein écran)
      2 exports → 1×2
      3-4       → 2×2
      5-6       → 2×3
      etc.      → ceil(sqrt(N)) colonnes
    """

    export_selected = pyqtSignal(object)    # ExportEntry
    export_activated = pyqtSignal(object)   # ExportEntry (double-clic)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._entries: list[ExportEntry] = []
        self._tiles: list[ExportTile] = []
        self._selected_entry: Optional[ExportEntry] = None
        self._pool = QThreadPool.globalInstance()
        self._fullscreen_entry: Optional[ExportEntry] = None
        self._best_index: Optional[int] = None

        # Layout empilé : 0=mosaïque, 1=plein format
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._pending_zoom: Optional[tuple] = None  # zoom à appliquer après chargement

        # ── Page mosaïque ──
        self._mosaic_page = QWidget()
        self._mosaic_page.setStyleSheet("background: #12122a;")
        self._grid_layout = QGridLayout(self._mosaic_page)
        self._grid_layout.setContentsMargins(2, 2, 2, 2)
        self._grid_layout.setSpacing(3)
        self._stack.addWidget(self._mosaic_page)

        # ── Label vide ──
        self._empty_label = QLabel("Aucun export disponible")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #555; font-size: 13px;")
        self._empty_label.setVisible(False)

        # ── Page plein format ──
        self._full_view = SyncedImageView()
        self._stack.addWidget(self._full_view)

        self._stack.setCurrentIndex(0)

    # ── API publique ──────────────────────────────────────────────────────────

    def set_exports(self, entries: list[ExportEntry]):
        """Remplace la liste d'exports affichés (reset complet)."""
        self._entries = list(entries)
        self._selected_entry = None
        self._fullscreen_entry = None
        self._pending_zoom = None
        self._stack.setCurrentIndex(0)
        self._rebuild_grid()

    def update_exports_if_changed(self, entries: list[ExportEntry]):
        """Met à jour seulement si la liste a changé (préserve l'état)."""
        new_indices = [e.index for e in entries]
        cur_indices = [e.index for e in self._entries]
        if new_indices == cur_indices:
            return  # même liste → ne rien toucher
        self.set_exports(entries)

    def refresh(self):
        """Force le rechargement depuis la liste courante."""
        self._rebuild_grid()

    def get_full_view(self) -> SyncedImageView:
        """Retourne la vue plein format pour la synchronisation de zoom."""
        return self._full_view

    def is_fullscreen(self) -> bool:
        return self._stack.currentIndex() == 1

    def selected_entry(self) -> Optional[ExportEntry]:
        return self._selected_entry

    def set_best_index(self, best_index: Optional[int]):
        """Met à jour le badge 'retenu' sur les tiles."""
        self._best_index = best_index
        for tile in self._tiles:
            tile.set_best(tile.entry.index == best_index)

    def get_zoom_state(self) -> tuple:
        """Retourne l'état zoom/position (normalisé) depuis la vue active."""
        if self.is_fullscreen():
            return self._full_view.get_zoom_state()
        # En mode mosaïque, lire depuis le premier viewer qui a une image
        for tile in self._tiles:
            if tile.viewer.has_image():
                return tile.viewer.get_zoom_state()
        return (1.0, 0.5, 0.5)

    def apply_zoom_state(self, zoom_ratio: float, cx_rel: float, cy_rel: float) -> None:
        """Applique l'état zoom/position à toutes les vues (mosaïque + plein format)."""
        self._pending_zoom = (zoom_ratio, cx_rel, cy_rel)
        if self.is_fullscreen():
            self._full_view.apply_zoom_state(zoom_ratio, cx_rel, cy_rel)
        else:
            for tile in self._tiles:
                if tile.viewer.has_image():
                    tile.viewer.apply_zoom_state(zoom_ratio, cx_rel, cy_rel)

    # ── Construction de la grille ─────────────────────────────────────────────

    def _rebuild_grid(self):
        """Reconstruit la grille de tiles."""
        # Déconnecter les peers entre eux
        for tile in self._tiles:
            for other in self._tiles:
                if other is not tile:
                    tile.viewer.remove_peer(other.viewer)

        # Vider la grille
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w and w is not self._empty_label:
                w.setParent(None)
                w.deleteLater()
        self._tiles.clear()

        n = len(self._entries)
        if n == 0:
            self._empty_label.setVisible(True)
            self._grid_layout.addWidget(self._empty_label, 0, 0)
            return
        self._empty_label.setVisible(False)
        if self._empty_label.parent() is self._mosaic_page:
            self._empty_label.setParent(None)

        # Calculer la grille optimale
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        # Créer les tiles
        for i, entry in enumerate(self._entries):
            tile = ExportTile(entry)
            tile.clicked.connect(self._on_tile_clicked)
            tile.double_clicked.connect(self._on_tile_double_clicked)
            self._tiles.append(tile)

            r = i // cols
            c = i % cols
            self._grid_layout.addWidget(tile, r, c)

        # Stretch égal pour toutes les lignes et colonnes
        for r in range(rows):
            self._grid_layout.setRowStretch(r, 1)
        for c in range(cols):
            self._grid_layout.setColumnStretch(c, 1)

        # Connecter les peers pour synchronisation zoom/pan
        viewers = [tile.viewer for tile in self._tiles]
        for i, v1 in enumerate(viewers):
            for v2 in viewers[i + 1:]:
                v1.add_peer(v2)

        # Charger les images en arrière-plan
        for entry in self._entries:
            self._load_image_async(entry)

        # Auto-sélectionner le premier export
        if self._tiles:
            first = self._tiles[0].entry
            self._selected_entry = first
            self._tiles[0].set_selected(True)
            self.export_selected.emit(first)

    def _load_image_async(self, entry: ExportEntry):
        if not os.path.exists(entry.image_path):
            return
        task = _LoadTask(entry.index, entry.image_path)
        task.signals.loaded.connect(self._on_image_loaded)
        self._pool.start(task)

    def _on_image_loaded(self, index: int, img):
        if img is None:
            return
        for tile in self._tiles:
            if tile.entry.index == index:
                tile.set_image(img)
                # Appliquer le zoom en attente après chargement
                if self._pending_zoom is not None:
                    QTimer.singleShot(
                        0, lambda t=tile, z=self._pending_zoom: t.viewer.apply_zoom_state(*z)
                    )
                break

    # ── Sélection ─────────────────────────────────────────────────────────────

    def _on_tile_clicked(self, entry: ExportEntry):
        self._selected_entry = entry
        for tile in self._tiles:
            tile.set_selected(tile.entry.index == entry.index)
        self.export_selected.emit(entry)

    # ── Plein format ──────────────────────────────────────────────────────────

    def _on_tile_double_clicked(self, entry: ExportEntry):
        self._enter_fullscreen(entry)
        self.export_activated.emit(entry)

    def _enter_fullscreen(self, entry: ExportEntry):
        """Bascule en mode plein format pour un export."""
        self._fullscreen_entry = entry
        img = cv2.imread(entry.image_path, cv2.IMREAD_COLOR)
        if img is None:
            return
        # Récupérer le zoom actuel des viewers mosaïque
        zoom_state = None
        for tile in self._tiles:
            if tile.viewer.has_image():
                zoom_state = tile.viewer.get_zoom_state()
                break
        self._full_view.set_image(img)
        if zoom_state and zoom_state[0] > 0:
            QTimer.singleShot(
                0, lambda z=zoom_state: self._full_view.apply_zoom_state(*z)
            )
        self._stack.setCurrentIndex(1)

    def exit_fullscreen(self):
        """Retour au mode mosaïque."""
        if self._stack.currentIndex() != 1:
            return
        # Propager le zoom du plein format vers les viewers mosaïque
        state = self._full_view.get_zoom_state()
        if state[0] > 0:
            for tile in self._tiles:
                if tile.viewer.has_image():
                    tile.viewer.apply_zoom_state(*state)
            # Sauvegarder aussi dans _shared_zoom du parent
            parent = self.parent()
            while parent is not None:
                if hasattr(parent, "_shared_zoom"):
                    parent._shared_zoom = state
                    break
                parent = parent.parent()
        self._fullscreen_entry = None
        self._stack.setCurrentIndex(0)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.is_fullscreen():
            self.exit_fullscreen()
            return
        super().keyPressEvent(event)

