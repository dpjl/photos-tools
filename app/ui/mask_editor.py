"""ui/mask_editor.py — Panneau et dialogue d'édition du masque de retouche.

  MaskCanvasPanel : canvas (cf. ui/mask_canvas.py) + barre d'outils, intégrable
                    en ligne (mode batch) ou dans un dialogue.
  MaskEditorDialog : dialogue autonome enveloppant MaskCanvasPanel.

Usage typique :
    dlg = MaskEditorDialog(parent=self, image_bgr=img, initial_mask=step.get_mask())
    if dlg.exec() == QDialog.DialogCode.Accepted:
        step.set_mask(dlg.get_mask())

Controles :
  Ctrl+clic gauche  peindre        Clic gauche (zone)  selectionner
  Ctrl+clic droit   effacer        V                   voir dessous (peek)
  Molette           taille pinceau Suppr               effacer la zone
  Ctrl+Molette      zoom           Ctrl+Z / Ctrl+Y     annuler / retablir
"""

from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QWidget, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut

from ui.mask_canvas import MaskCanvas


# ══════════════════════════════════════════════════════════════════════════════
# Panel embarquable (canvas + contrôles)
# ══════════════════════════════════════════════════════════════════════════════

class MaskCanvasPanel(QWidget):
    """Canvas + panneau de contrôles, utilisable en ligne ou dans un dialogue.

    En mode inline (batch) ::

        panel = MaskCanvasPanel(image_bgr)
        # intégrer dans un layout...
        panel.set_image(new_bgr, saved_mask)   # changement d'image
        mask = panel.get_mask()

    En mode dialogue : utiliser MaskEditorDialog.
    """

    accepted = pyqtSignal()   # émis par le bouton Valider (si show_ok_cancel=True)
    rejected = pyqtSignal()   # émis par le bouton Annuler  (si show_ok_cancel=True)

    def __init__(
        self,
        image_bgr:      np.ndarray,
        initial_mask:   np.ndarray | None = None,
        parent=None,
        show_ok_cancel: bool = False,
        sidebar_width:  int  = 215,
        title:          str  = "Masque de retouche",
    ) -> None:
        super().__init__(parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Canvas (gauche, expansif) ─────────────────────────────────────────
        self._canvas = MaskCanvas(image_bgr, initial_mask)
        self._canvas.brush_size_changed.connect(self._on_canvas_brush_changed)
        root.addWidget(self._canvas, stretch=1)

        # ── Barre d'outils (droite, largeur fixe, défilable) ─────────────────
        sidebar = QWidget()
        sidebar.setStyleSheet("background: #1a1a2e;")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(8, 10, 8, 10)
        sl.setSpacing(5)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("color:#ddd; font-size:12px; font-weight:700;")
        sl.addWidget(title_lbl)
        sl.addWidget(self._hline())

        tips = QLabel(
            "Ctrl+clic G/D : peindre / effacer\n"
            "Clic zone : sélectionner\n"
            "   V : voir dessous · Suppr : efface\n"
            "Clic vide : déplacer\n"
            "Molette : taille · Ctrl+molette : zoom\n"
            "Ctrl+Z / Y : annuler / rétablir"
        )
        tips.setStyleSheet("color:#7a9ab0; font-size:9px; font-family:Consolas,monospace;")
        sl.addWidget(tips)
        sl.addWidget(self._hline())

        size_row = QHBoxLayout()
        size_row.setSpacing(4)
        lbl_brush = QLabel("Pinceau")
        lbl_brush.setStyleSheet("color:#bbb; font-size:10px;")
        size_row.addWidget(lbl_brush)
        size_row.addStretch()
        self._brush_val_lbl = QLabel("30 px")
        self._brush_val_lbl.setStyleSheet("color:#9de; font-size:11px; font-weight:700;")
        size_row.addWidget(self._brush_val_lbl)
        sl.addLayout(size_row)

        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(1, 300)
        self._brush_slider.setValue(30)
        self._brush_slider.valueChanged.connect(self._on_slider_brush)
        sl.addWidget(self._brush_slider)

        sl.addWidget(self._hline())

        row_edit = QHBoxLayout()
        row_edit.setSpacing(4)
        btn_clear = QPushButton("\U0001f5d1 Effacer")
        btn_clear.setToolTip("Effacer tout le masque")
        btn_clear.clicked.connect(self._canvas.clear_mask)
        self._style(btn_clear)
        row_edit.addWidget(btn_clear)

        btn_undo = QPushButton("\u21a9 Annuler")
        btn_undo.setToolTip("Annuler (Ctrl+Z)")
        btn_undo.clicked.connect(self._canvas.undo)
        self._style(btn_undo)
        row_edit.addWidget(btn_undo)
        sl.addLayout(row_edit)

        self._sidebar_layout     = sl
        self._sidebar_insert_idx = sl.count()   # position avant le stretch
        sl.addStretch()

        if show_ok_cancel:
            sl.addWidget(self._hline())
            btn_ok = QPushButton("\u2713  Valider le masque")
            btn_ok.clicked.connect(self.accepted.emit)
            self._style(btn_ok, accent=True)
            sl.addWidget(btn_ok)

            btn_cancel = QPushButton("\u2717  Annuler")
            btn_cancel.clicked.connect(self.rejected.emit)
            self._style(btn_cancel)
            sl.addWidget(btn_cancel)

        # Sidebar d\u00e9filable (le panneau de contr\u00f4les peut \u00eatre haut)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(sidebar_width)
        scroll.setWidget(sidebar)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background:#1a1a2e; border:none; }"
            "QScrollBar:vertical { background:#16162a; width:9px; margin:0; }"
            "QScrollBar::handle:vertical { background:#33335a; border-radius:4px; min-height:30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }"
        )
        root.addWidget(scroll)

    # ── API publique ──────────────────────────────────────────────────────────

    def get_mask(self) -> np.ndarray:
        return self._canvas.get_mask()

    def get_base_image(self) -> np.ndarray:
        """Image de référence en BGR (espace de coordonnées du masque)."""
        return cv2.cvtColor(self._canvas._orig_img_rgb, cv2.COLOR_RGB2BGR)

    def get_display_image(self) -> np.ndarray:
        """Image actuellement affichée en BGR (preview courant)."""
        return self._canvas.get_display_image()

    def mask_shape(self) -> tuple:
        """Dimensions (h, w) de l'espace du masque."""
        return (self._canvas._img_h, self._canvas._img_w)

    def set_mask(self, mask: np.ndarray, push_undo: bool = True) -> None:
        """Remplace le masque courant (``push_undo=False`` pour le live)."""
        self._canvas.set_mask(mask, push_undo=push_undo)

    def push_undo(self) -> None:
        """Empile l'état courant du masque dans l'historique."""
        self._canvas.push_undo()

    # ── Revue VLM ──────────────────────────────────────────────────────────────

    def enter_review(self, labels: np.ndarray, categories: dict,
                     reasons: dict | None = None) -> None:
        self._canvas.enter_review(labels, categories, reasons)

    def apply_review(self) -> int:
        return self._canvas.apply_review()

    def in_review(self) -> bool:
        return self._canvas.in_review()

    def review_counts(self) -> tuple:
        return self._canvas.review_counts()

    def add_to_sidebar(self, widget: "QWidget") -> None:
        """Insère un widget dans la sidebar juste avant le stretch final."""
        self._sidebar_layout.insertWidget(self._sidebar_insert_idx, widget)
        self._sidebar_insert_idx += 1

    def set_image(
        self,
        image_bgr:    np.ndarray,
        initial_mask: np.ndarray | None = None,
    ) -> None:
        """Change l'image affichée (mode batch — changement de photo)."""
        self._canvas.reset_image(image_bgr, initial_mask)

    # ── Helpers UI ────────────────────────────────────────────────────────────

    @staticmethod
    def _hline() -> QFrame:
        f = QFrame()
        f.setFixedHeight(1)
        f.setStyleSheet("background:#252545; margin:2px 0;")
        return f

    @staticmethod
    def _style(btn: QPushButton, accent: bool = False) -> None:
        if accent:
            btn.setStyleSheet(
                "QPushButton { background:#1e3a52; color:#b8e0f7; border-radius:4px;"
                "  padding:5px 6px; font-size:10px; }"
                "QPushButton:hover { background:#2a5577; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
                "  padding:5px 6px; font-size:10px; }"
                "QPushButton:hover { background:#2a2a50; color:#ccc; }"
            )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_slider_brush(self, val: int) -> None:
        self._canvas.set_brush_px(val)
        self._brush_val_lbl.setText(f"{val} px")

    def _on_canvas_brush_changed(self, val: int) -> None:
        """Synchronise le slider quand la taille change via la molette."""
        self._brush_slider.blockSignals(True)
        self._brush_slider.setValue(val)
        self._brush_slider.blockSignals(False)
        self._brush_val_lbl.setText(f"{val} px")

# ══════════════════════════════════════════════════════════════════════════════
# Dialogue éditeur
# ══════════════════════════════════════════════════════════════════════════════

class MaskEditorDialog(QDialog):
    """Dialogue d'édition du masque — wrapper fin autour de MaskCanvasPanel."""

    def __init__(
        self,
        parent,
        image_bgr:    np.ndarray,
        initial_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Masque de retouche \u2014 peinture par pinceau")
        self.setModal(True)
        self.setMinimumSize(900, 600)
        self.resize(1100, 750)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._panel = MaskCanvasPanel(image_bgr, initial_mask, show_ok_cancel=True)
        self._panel.accepted.connect(self.accept)
        self._panel.rejected.connect(self.reject)
        root.addWidget(self._panel)

        # Ctrl+Z / Ctrl+Y pour ce dialogue (le keyPressEvent est dans MaskEditorDialog)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._panel._canvas.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._panel._canvas.redo)

    def get_mask(self) -> np.ndarray:
        return self._panel.get_mask()
