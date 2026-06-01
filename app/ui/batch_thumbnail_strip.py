"""ui/batch_thumbnail_strip.py — Bande de miniatures pour le mode batch.

Indicateurs visuels :
  - Bordure bleue        : image sélectionnée (navigation)
  - Case bleue bas-gauche: incluse dans la sélection d'exécution
  - Point orange         : paramètres personnalisés
  - Check vert           : image traitée et sauvegardée
  - Spinner              : image en cours de traitement
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject, QSize
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QBrush


_THUMB_W = 128
_THUMB_H = 96
_CARD_W  = _THUMB_W + 8   # 8 px pour la bordure
_CARD_H  = _THUMB_H + 8 + 20   # + label


# ══════════════════════════════════════════════════════════════════════════════
# Worker de chargement des miniatures (QThread)
# ══════════════════════════════════════════════════════════════════════════════

class _ThumbLoader(QObject):
    """Charge les miniatures en arrière-plan, une par une."""

    thumb_ready = pyqtSignal(str, object)   # (file_path, QPixmap)

    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self._paths   = paths
        self._cancel  = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        for path in self._paths:
            if self._cancel:
                break
            pix = _load_thumbnail(path)
            self.thumb_ready.emit(path, pix)

    def start_in_thread(self) -> QThread:
        thread = QThread()
        self.moveToThread(thread)
        thread.started.connect(self.run)
        thread.start()
        self._thread = thread
        return thread


def _load_thumbnail(path: str) -> QPixmap:
    """Charge et redimensionne une image en miniature (128×96 max)."""
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return _placeholder_pixmap()
        ih, iw = img.shape[:2]
        scale = min(_THUMB_W / iw, _THUMB_H / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = np.ascontiguousarray(rgb)
        qimg = QImage(data.tobytes(), nw, nh, nw * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
    except Exception:
        return _placeholder_pixmap()


def _placeholder_pixmap() -> QPixmap:
    px = QPixmap(_THUMB_W, _THUMB_H)
    px.fill(QColor(40, 40, 60))
    return px


# ══════════════════════════════════════════════════════════════════════════════
# Carte miniature individuelle
# ══════════════════════════════════════════════════════════════════════════════

class BatchThumbCard(QWidget):
    """Carte miniature d'une image du batch."""

    # file_path + modificateurs clavier (Qt.KeyboardModifiers)
    clicked       = pyqtSignal(str, object)
    reset_request = pyqtSignal(str)   # file_path — menu contextuel

    def __init__(self, file_path: str, parent=None) -> None:
        super().__init__(parent)
        self._file_path  = file_path
        self._pixmap:    Optional[QPixmap] = None
        self._selected    = False
        self._run_selected = False    # dans la sélection d'exécution
        self._customized  = False
        self._done        = False
        self._running     = False

        self.setFixedSize(_CARD_W, _CARD_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Zone image
        self._img_lbl = QLabel()
        self._img_lbl.setFixedSize(_THUMB_W, _THUMB_H)
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet("background: #1a1a2e;")
        layout.addWidget(self._img_lbl)

        # Nom de fichier (tronqué)
        name = os.path.basename(file_path)
        if len(name) > 18:
            name = name[:7] + "…" + name[-8:]
        self._name_lbl = QLabel(name)
        self._name_lbl.setFixedHeight(16)
        self._name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_lbl.setStyleSheet("color:#888; font-size:9px;")
        layout.addWidget(self._name_lbl)

        self._refresh_style()

    # ── API ────────────────────────────────────────────────────────────────────

    def set_pixmap(self, pix: QPixmap) -> None:
        self._pixmap = pix
        self._img_lbl.setPixmap(
            pix.scaled(_THUMB_W, _THUMB_H, Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )

    def set_selected(self, v: bool) -> None:
        self._selected = v
        self._refresh_style()

    def set_run_selected(self, v: bool) -> None:
        self._run_selected = v
        self.update()

    def set_customized(self, v: bool) -> None:
        self._customized = v
        self._refresh_style()
        self.update()

    def set_done(self, v: bool) -> None:
        self._done    = v
        self._running = False
        self._refresh_style()
        self.update()

    def set_running(self, v: bool) -> None:
        self._running = v
        self._done    = False
        self._refresh_style()
        self.update()

    # ── Rendu ────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        super().paintEvent(_event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Indicateur en cours de sélection d'exécution (case bleue bas-gauche)
        if self._run_selected and not self._running:
            p.setBrush(QBrush(QColor(58, 123, 213)))    # #3a7bd5
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(2, _CARD_H - 14, 12, 12, 2, 2)
            p.setPen(QPen(QColor(255, 255, 255), 1.8))
            # Petit check blanc
            p.drawLine(4, _CARD_H - 8, 7, _CARD_H - 5)
            p.drawLine(7, _CARD_H - 5, 12, _CARD_H - 11)

        # Indicateur personnalisé (point orange en haut à droite)
        if self._customized and not self._done:
            p.setBrush(QBrush(QColor(255, 140, 0)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(_CARD_W - 12, 2, 10, 10)

        # Indicateur traité (check vert)
        if self._done:
            p.setBrush(QBrush(QColor(46, 204, 113)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(_CARD_W - 14, 2, 12, 12)
            p.setPen(QPen(QColor(255, 255, 255), 2.0))
            p.drawLine(_CARD_W - 11, 8, _CARD_W - 8, 11)
            p.drawLine(_CARD_W - 8, 11, _CARD_W - 4, 5)

        # Indicateur en cours (cercle animé — juste un contour orangé)
        if self._running:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 180, 0), 2.5))
            p.drawEllipse(4, 4, _CARD_W - 8, _THUMB_H)

        p.end()

    def _refresh_style(self) -> None:
        if self._running:
            border_color = "#ffb400"
            border_w     = 2
        elif self._selected:
            border_color = "#3a7bd5"
            border_w     = 2
        elif self._done:
            border_color = "#27ae60"
            border_w     = 2
        else:
            border_color = "#2a2a3a"
            border_w     = 1
        self.setStyleSheet(
            f"QWidget {{ background: #1a1a2e; border-radius: 4px;"
            f" border: {border_w}px solid {border_color}; }}"
        )
        self._name_lbl.setStyleSheet(
            f"color: {'#9de' if self._selected else '#888'}; font-size:9px; border:none;"
        )

    # ── Événements ────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._file_path, event.modifiers())

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background:#1e1e38; color:#ccc; border:1px solid #3a3a5a; }"
            "QMenu::item:selected { background:#2a2a58; }"
        )
        act_reset = menu.addAction("↺  Réinitialiser aux valeurs par défaut")
        act = menu.exec(self.mapToGlobal(pos))
        if act == act_reset:
            self.reset_request.emit(self._file_path)


# ══════════════════════════════════════════════════════════════════════════════
# Bande de miniatures
# ══════════════════════════════════════════════════════════════════════════════

class BatchThumbnailStrip(QWidget):
    """Bande horizontale scrollable de miniatures d'images du batch."""

    image_selected    = pyqtSignal(str)   # file_path
    image_reset       = pyqtSignal(str)   # file_path

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_CARD_H + 16)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("background: #12122a;")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("border:none; background:#12122a;")
        self._scroll.setFixedHeight(_CARD_H + 12)
        outer.addWidget(self._scroll)

        self._container = QWidget()
        self._container.setStyleSheet("background:#12122a;")
        self._container.setMinimumHeight(_CARD_H)
        self._row = QHBoxLayout(self._container)
        self._row.setContentsMargins(4, 0, 4, 0)
        self._row.setSpacing(6)
        self._row.addStretch()
        self._scroll.setWidget(self._container)

        self._cards: dict[str, BatchThumbCard] = {}   # file_path → card
        self._selected_path: Optional[str] = None
        self._run_paths: set[str] = set()              # sélection pour exécution
        self._override_pixmaps: dict[str, QPixmap] = {}

        self._loader: Optional[_ThumbLoader] = None
        self._loader_thread: Optional[QThread] = None

    # ── API ────────────────────────────────────────────────────────────────────

    def load_images(self, file_paths: list[str]) -> None:
        """Peuple la bande avec les fichiers donnés et lance le chargement."""
        self._stop_loader()
        self._cards.clear()
        self._selected_path = None
        self._run_paths.clear()
        self._override_pixmaps.clear()

        # Nettoyer le layout
        while self._row.count() > 1:
            item = self._row.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        # Créer les cartes
        for path in file_paths:
            card = BatchThumbCard(path)
            card.clicked.connect(self._on_card_clicked)
            card.reset_request.connect(self.image_reset)
            self._cards[path] = card
            self._row.insertWidget(self._row.count() - 1, card)

        # Lancer le chargement des miniatures en arrière-plan
        if file_paths:
            self._loader = _ThumbLoader(file_paths)
            self._loader.thumb_ready.connect(self._on_thumb_ready)
            self._loader_thread = self._loader.start_in_thread()

    def select(self, file_path: str) -> None:
        """Sélectionne visuellement la carte correspondante (navigation)."""
        if self._selected_path and self._selected_path in self._cards:
            self._cards[self._selected_path].set_selected(False)
        self._selected_path = file_path
        if file_path in self._cards:
            self._cards[file_path].set_selected(True)
            self._scroll_to(file_path)

    def get_run_selection(self) -> list[str]:
        """Retourne les chemins sélectionnés pour exécution (ordre des cartes)."""
        return [p for p in self._cards if p in self._run_paths]

    def set_run_selected(self, file_path: str, v: bool) -> None:
        """Active/désactive le badge de sélection d'exécution d'une carte."""
        if v:
            self._run_paths.add(file_path)
        else:
            self._run_paths.discard(file_path)
        if file_path in self._cards:
            self._cards[file_path].set_run_selected(v)

    def clear_run_selection(self) -> None:
        """Vide la sélection d'exécution."""
        for path in list(self._run_paths):
            if path in self._cards:
                self._cards[path].set_run_selected(False)
        self._run_paths.clear()

    def set_customized(self, file_path: str, v: bool) -> None:
        if file_path in self._cards:
            self._cards[file_path].set_customized(v)

    def set_done(self, file_path: str, v: bool) -> None:
        if file_path in self._cards:
            self._cards[file_path].set_done(v)

    def set_running(self, file_path: str, v: bool) -> None:
        if file_path in self._cards:
            self._cards[file_path].set_running(v)

    def update_result_thumb(self, file_path: str, img: np.ndarray) -> None:
        """Met à jour la miniature avec l'image résultat."""
        if file_path not in self._cards:
            return
        ih, iw = img.shape[:2]
        scale = min(_THUMB_W / iw, _THUMB_H / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        data  = np.ascontiguousarray(rgb)
        qimg  = QImage(data.tobytes(), nw, nh, nw * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._override_pixmaps[file_path] = pix
        self._cards[file_path].set_pixmap(pix)

    def use_original_thumb(self, file_path: str) -> None:
        """Réaffiche l'image originale pour une carte."""
        self._override_pixmaps.pop(file_path, None)
        if file_path in self._cards:
            self._cards[file_path].set_pixmap(_load_thumbnail(file_path))

    def count(self) -> int:
        return len(self._cards)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_card_clicked(self, file_path: str, modifiers: object) -> None:
        # Qt.KeyboardModifiers est un QFlags — utiliser & directement, pas int()
        ctrl  = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        if ctrl:
            # Ctrl+clic : basculer l'appartenance à la sélection d'exécution
            if file_path in self._run_paths:
                self._run_paths.discard(file_path)
                if file_path in self._cards:
                    self._cards[file_path].set_run_selected(False)
            else:
                self._run_paths.add(file_path)
                if file_path in self._cards:
                    self._cards[file_path].set_run_selected(True)

        elif shift and self._selected_path:
            # Shift+clic : sélectionner la plage entre image courante et celle-ci
            paths = list(self._cards.keys())
            try:
                i1 = paths.index(self._selected_path)
                i2 = paths.index(file_path)
            except ValueError:
                i1, i2 = 0, 0
            start, end = min(i1, i2), max(i1, i2)
            # Vider d'abord la sélection précédente
            for p in list(self._run_paths):
                if p in self._cards:
                    self._cards[p].set_run_selected(False)
            self._run_paths.clear()
            for p in paths[start : end + 1]:
                self._run_paths.add(p)
                if p in self._cards:
                    self._cards[p].set_run_selected(True)

        else:
            # Clic simple : naviguer + sélectionner seule pour exécution
            for p in list(self._run_paths):
                if p in self._cards:
                    self._cards[p].set_run_selected(False)
            self._run_paths = {file_path}
            if file_path in self._cards:
                self._cards[file_path].set_run_selected(True)
            self.select(file_path)
            self.image_selected.emit(file_path)

    def _on_thumb_ready(self, file_path: str, pix: QPixmap) -> None:
        if file_path in self._cards:
            self._cards[file_path].set_pixmap(
                self._override_pixmaps.get(file_path, pix)
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _scroll_to(self, file_path: str) -> None:
        if file_path not in self._cards:
            return
        card = self._cards[file_path]
        self._scroll.ensureWidgetVisible(card)

    def _stop_loader(self) -> None:
        if self._loader:
            self._loader.cancel()
        if self._loader_thread:
            self._loader_thread.quit()
            self._loader_thread.wait(500)
        self._loader        = None
        self._loader_thread = None

    def closeEvent(self, event) -> None:
        self._stop_loader()
        super().closeEvent(event)
