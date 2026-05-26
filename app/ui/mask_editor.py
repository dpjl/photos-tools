"""ui/mask_editor.py — Editeur de masque de retouche par pinceau.

Usage typique :
    dlg = MaskEditorDialog(parent=self, image_bgr=img, initial_mask=step.get_mask())
    if dlg.exec() == QDialog.DialogCode.Accepted:
        step.set_mask(dlg.get_mask())

Controles :
  Clic gauche   peindre (zone a retoucher)
  Clic droit    effacer
  Molette       taille du pinceau
  Ctrl+Z        annuler le dernier coup de pinceau
"""

from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QSizePolicy, QWidget, QFrame, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

_MAX_UNDO = 20   # nombre maximum d'etats undo conserves en memoire


# ══════════════════════════════════════════════════════════════════════════════
# Canvas de dessin
# ══════════════════════════════════════════════════════════════════════════════

class MaskCanvas(QWidget):
    """Zone de dessin : image de fond + overlay rouge semi-transparent pour le masque."""

    brush_size_changed = pyqtSignal(int)   # emis quand la taille change via molette

    def __init__(
        self,
        image_bgr:    np.ndarray,
        initial_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Image en RGB pour QImage
        # _orig_img_rgb : image de reference (pour la detection automatique) — ne change pas
        # _base_pixmap  : ce qui est affiche (peut etre mis a jour par auto-niveaux)
        self._orig_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = self._orig_img_rgb.shape[:2]
        self._img_w, self._img_h = iw, ih

        # Pixmap base (image seule, creee une fois puis eventuellement remplacee)
        self._base_pixmap: QPixmap = self._make_rgb_pixmap(self._orig_img_rgb)

        # Masque courant (uint8 H x W, 255 = a retoucher)
        if initial_mask is not None and initial_mask.shape[:2] == (ih, iw):
            self._mask = initial_mask.copy()
        else:
            self._mask = np.zeros((ih, iw), dtype=np.uint8)

        # Cache du pixmap composite (image + overlay)
        self._dirty:           bool             = True
        self._composite_cache: QPixmap | None   = None

        # Etat de dessin
        self._brush_px:       int              = 30
        self._painting:       bool             = False
        self._erasing:        bool             = False
        self._last_canvas_pt: QPoint | None    = None
        self._cursor_canvas:  QPoint | None    = None

        # Historique undo
        self._undo_stack: list[np.ndarray] = []

        # Zoom / pan
        self._zoom:     float          = 1.0
        self._pan_x:    float          = 0.0
        self._pan_y:    float          = 0.0
        self._panning:  bool           = False
        self._pan_last: QPoint | None  = None

    # ── API publique ──────────────────────────────────────────────────────────

    def get_mask(self) -> np.ndarray:
        return self._mask.copy()

    def reset_image(
        self,
        image_bgr:    np.ndarray,
        initial_mask: np.ndarray | None = None,
    ) -> None:
        """Charge une nouvelle image (mode batch — changement de photo)."""
        self._orig_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = self._orig_img_rgb.shape[:2]
        self._img_w, self._img_h = iw, ih
        self._base_pixmap = self._make_rgb_pixmap(self._orig_img_rgb)
        if initial_mask is not None and initial_mask.shape[:2] == (ih, iw):
            self._mask = initial_mask.copy()
        else:
            self._mask = np.zeros((ih, iw), dtype=np.uint8)
        self._dirty           = True
        self._composite_cache = None
        self._undo_stack.clear()
        self._painting        = False
        self._erasing         = False
        self._last_canvas_pt  = None
        self._zoom            = 1.0
        self._pan_x           = 0.0
        self._pan_y           = 0.0
        self._panning         = False
        self._pan_last        = None
        self.update()

    def set_display_image(self, bgr: np.ndarray) -> None:
        """Change l'image affichee (ex. version auto-niveaux). Le masque n'est pas affecte."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._base_pixmap = self._make_rgb_pixmap(rgb)
        self._dirty = True
        self.update()

    def set_mask(self, mask: np.ndarray) -> None:
        """Remplace le masque (avec historique undo)."""
        self._push_undo()
        self._mask = mask.copy()
        self._invalidate()

    def clear_mask(self) -> None:
        self._push_undo()
        self._mask[:] = 0
        self._invalidate()

    def undo(self) -> None:
        if self._undo_stack:
            self._mask = self._undo_stack.pop()
            self._invalidate()

    def set_brush_px(self, px: int) -> None:
        self._brush_px = max(1, min(300, px))

    def get_brush_px(self) -> int:
        return self._brush_px

    # ── Conversion de coordonnees ─────────────────────────────────────────────

    def _display_rect(self) -> QRect:
        """Rect d'affichage de l'image (centrée, ratio préservé, zoom/pan appliqués)."""
        cw, ch = self.width(), self.height()
        if cw == 0 or ch == 0:
            return QRect(0, 0, 0, 0)
        fit_scale = min(cw / self._img_w, ch / self._img_h)
        scale     = fit_scale * self._zoom
        dw        = int(self._img_w * scale)
        dh        = int(self._img_h * scale)
        base_x    = (cw - dw) // 2
        base_y    = (ch - dh) // 2
        return QRect(base_x + int(self._pan_x), base_y + int(self._pan_y), dw, dh)

    def _zoom_at(self, canvas_pt: QPoint, factor: float) -> None:
        """Zoom centré sur canvas_pt (Ctrl+molette)."""
        r = self._display_rect()
        if r.width() == 0:
            return
        fx = (canvas_pt.x() - r.x()) / r.width()
        fy = (canvas_pt.y() - r.y()) / r.height()
        self._zoom = max(0.5, min(16.0, self._zoom * factor))
        cw, ch     = self.width(), self.height()
        fit_scale  = min(cw / self._img_w, ch / self._img_h) if cw and ch else 1.0
        new_scale  = fit_scale * self._zoom
        dw, dh     = int(self._img_w * new_scale), int(self._img_h * new_scale)
        self._pan_x = canvas_pt.x() - fx * dw - (cw - dw) // 2
        self._pan_y = canvas_pt.y() - fy * dh - (ch - dh) // 2
        self.update()

    def _canvas_to_img(self, pt: QPoint) -> tuple[int, int] | None:
        """Convertit les coordonnees canvas en coordonnees image."""
        r = self._display_rect()
        if r.width() == 0 or r.height() == 0:
            return None
        ix = int((pt.x() - r.x()) * self._img_w / r.width())
        iy = int((pt.y() - r.y()) * self._img_h / r.height())
        ix = int(np.clip(ix, 0, self._img_w - 1))
        iy = int(np.clip(iy, 0, self._img_h - 1))
        return ix, iy

    def _brush_canvas_radius(self) -> float:
        """Rayon du pinceau en pixels canvas."""
        r = self._display_rect()
        if self._img_w == 0:
            return 0.0
        return self._brush_px * r.width() / self._img_w

    # ── Dessin ───────────────────────────────────────────────────────────────

    def _paint_segment(self, p1: QPoint, p2: QPoint, erase: bool) -> None:
        """Dessine une suite de cercles interpoles entre deux points canvas."""
        c1 = self._canvas_to_img(p1)
        c2 = self._canvas_to_img(p2)
        if c1 is None or c2 is None:
            return
        x1, y1 = c1
        x2, y2 = c2
        dist  = max(1, int(np.hypot(x2 - x1, y2 - y1)))
        steps = max(1, dist // max(1, self._brush_px // 2))
        val   = 0 if erase else 255
        for i in range(steps + 1):
            t  = i / steps
            ix = int(round(x1 + (x2 - x1) * t))
            iy = int(round(y1 + (y2 - y1) * t))
            cv2.circle(self._mask, (ix, iy), self._brush_px, val, -1)
        self._invalidate()

    def _push_undo(self) -> None:
        self._undo_stack.append(self._mask.copy())
        if len(self._undo_stack) > _MAX_UNDO:
            self._undo_stack.pop(0)

    def _invalidate(self) -> None:
        self._dirty = True
        self.update()

    # ── Evenements souris ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning  = True
            self._pan_last = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._push_undo()
            self._painting = True
            self._erasing  = False
        elif event.button() == Qt.MouseButton.RightButton:
            self._push_undo()
            self._erasing  = True
            self._painting = False
        else:
            return
        self._last_canvas_pt = pos
        coords = self._canvas_to_img(pos)
        if coords:
            val = 0 if self._erasing else 255
            cv2.circle(self._mask, coords, self._brush_px, val, -1)
            self._invalidate()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._panning and self._pan_last is not None:
            self._pan_x += pos.x() - self._pan_last.x()
            self._pan_y += pos.y() - self._pan_last.y()
            self._pan_last = pos
            self.update()
            event.accept()
            return
        self._cursor_canvas = pos
        if (self._painting or self._erasing) and self._last_canvas_pt is not None:
            self._paint_segment(self._last_canvas_pt, pos, self._erasing)
            self._last_canvas_pt = pos
        else:
            self.update()  # reafficher juste le curseur

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning  = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._painting       = False
            self._erasing        = False
            self._last_canvas_pt = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl + molette = zoom
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            self._zoom_at(event.position().toPoint(), factor)
        else:
            # Molette seule = taille du pinceau
            step = 10 if abs(delta) >= 120 else 3
            if delta > 0:
                self._brush_px = min(300, self._brush_px + step)
            else:
                self._brush_px = max(2, self._brush_px - step)
            self.update()
            self.brush_size_changed.emit(self._brush_px)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Z and (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.undo()

    # ── Rendu ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_rgb_pixmap(rgb: np.ndarray) -> QPixmap:
        h, w = rgb.shape[:2]
        # Forcer une copie contigue pour QImage
        data = np.ascontiguousarray(rgb)
        qimg = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _make_overlay_pixmap(self) -> QPixmap:
        """Pixmap RGBA : rouge semi-transparent sur les zones masquees."""
        ih, iw = self._mask.shape
        rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
        rgba[self._mask > 127] = (255, 60, 60, 160)
        data = np.ascontiguousarray(rgba)
        qimg = QImage(data.tobytes(), iw, ih, iw * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    def _get_composite(self) -> QPixmap:
        """Retourne le pixmap composite (image + overlay), recalcule si necessaire."""
        if self._dirty or self._composite_cache is None:
            iw, ih = self._img_w, self._img_h
            px = QPixmap(iw, ih)
            p  = QPainter(px)
            p.drawPixmap(0, 0, self._base_pixmap)
            p.drawPixmap(0, 0, self._make_overlay_pixmap())
            p.end()
            self._composite_cache = px
            self._dirty           = False
        return self._composite_cache

    def paintEvent(self, _event):
        rect = self._display_rect()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Fond sombre
        p.fillRect(self.rect(), QColor(22, 22, 32))

        # Image + overlay masque
        p.drawPixmap(rect, self._get_composite())

        # Curseur pinceau (cercle tirete)
        if self._cursor_canvas is not None:
            r = self._brush_canvas_radius()
            if r >= 1:
                pen = QPen(QColor(255, 230, 80), 1.5, Qt.PenStyle.DashLine)
                p.setPen(pen)
                cx, cy = self._cursor_canvas.x(), self._cursor_canvas.y()
                p.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))
        p.end()


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
    ) -> None:
        super().__init__(parent)
        self._orig_img_bgr       = image_bgr.copy()
        self._auto_levels_active = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Canvas (gauche, expansif) ─────────────────────────────────────────
        self._canvas = MaskCanvas(image_bgr, initial_mask)
        self._canvas.brush_size_changed.connect(self._on_canvas_brush_changed)
        root.addWidget(self._canvas, stretch=1)

        # ── Barre d'outils (droite, largeur fixe) ────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(sidebar_width)
        sidebar.setStyleSheet("background: #1a1a2e;")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(12, 14, 12, 14)
        sl.setSpacing(8)

        title = QLabel("Masque de retouche")
        title.setStyleSheet("color:#ddd; font-size:13px; font-weight:700;")
        sl.addWidget(title)
        sl.addWidget(self._hline())

        tips = QLabel(
            "Clic gauche  \u2192 peindre\n"
            "Clic droit   \u2192 effacer\n"
            "Molette      \u2192 taille pinceau\n"
            "Ctrl+Z       \u2192 annuler"
        )
        tips.setStyleSheet("color:#7a9ab0; font-size:10px; font-family:Consolas,monospace;")
        sl.addWidget(tips)
        sl.addWidget(self._hline())

        lbl_brush = QLabel("Taille du pinceau :")
        lbl_brush.setStyleSheet("color:#bbb; font-size:11px;")
        sl.addWidget(lbl_brush)

        size_row = QHBoxLayout()
        self._brush_val_lbl = QLabel("30 px")
        self._brush_val_lbl.setStyleSheet(
            "color:#9de; font-size:13px; font-weight:700; min-width:55px;"
        )
        size_row.addWidget(self._brush_val_lbl)
        size_row.addStretch()
        sl.addLayout(size_row)

        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(2, 300)
        self._brush_slider.setValue(30)
        self._brush_slider.valueChanged.connect(self._on_slider_brush)
        sl.addWidget(self._brush_slider)

        sl.addWidget(self._hline())

        btn_clear = QPushButton("\U0001f5d1  Effacer tout le masque")
        btn_clear.clicked.connect(self._canvas.clear_mask)
        self._style(btn_clear)
        sl.addWidget(btn_clear)

        btn_undo = QPushButton("\u21a9  Annuler (Ctrl+Z)")
        btn_undo.clicked.connect(self._canvas.undo)
        self._style(btn_undo)
        sl.addWidget(btn_undo)

        sl.addWidget(self._hline())

        lbl_auto = QLabel("D\u00e9tection automatique :")
        lbl_auto.setStyleSheet("color:#bbb; font-size:11px;")
        sl.addWidget(lbl_auto)

        lbl_sens = QLabel("Sensibilit\u00e9 :")
        lbl_sens.setStyleSheet("color:#789; font-size:10px;")
        sl.addWidget(lbl_sens)

        self._sens_slider = QSlider(Qt.Orientation.Horizontal)
        self._sens_slider.setRange(0, 100)
        self._sens_slider.setValue(50)
        sl.addWidget(self._sens_slider)

        self._add_cb = QCheckBox("Ajouter au masque existant")
        self._add_cb.setStyleSheet(
            "QCheckBox { color:#999; font-size:10px; }"
            "QCheckBox::indicator { width:13px; height:13px; }"
        )
        self._add_cb.setChecked(False)
        sl.addWidget(self._add_cb)

        btn_detect = QPushButton("\u26a1  D\u00e9tecter artefacts")
        btn_detect.clicked.connect(self._run_auto_detect)
        self._style(btn_detect, accent=True)
        sl.addWidget(btn_detect)
        sl.addWidget(self._hline())

        self._auto_btn = QPushButton("\U0001f4d0  Auto niveaux (aper\u00e7u)")
        self._auto_btn.setCheckable(True)
        self._auto_btn.clicked.connect(self._toggle_auto_levels)
        self._style(self._auto_btn)
        sl.addWidget(self._auto_btn)

        lbl_auto_note = QLabel(
            "Aper\u00e7u seulement \u2014 le masque\nreste dans les coord. d'origine."
        )
        lbl_auto_note.setStyleSheet("color:#556; font-size:9px;")
        sl.addWidget(lbl_auto_note)
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

        root.addWidget(sidebar)

    # ── API publique ──────────────────────────────────────────────────────────

    def get_mask(self) -> np.ndarray:
        return self._canvas.get_mask()

    def set_image(
        self,
        image_bgr:    np.ndarray,
        initial_mask: np.ndarray | None = None,
    ) -> None:
        """Change l'image affichée (mode batch — changement de photo)."""
        self._orig_img_bgr = image_bgr.copy()
        if self._auto_levels_active:
            self._auto_btn.setChecked(False)
            self._auto_btn.setText("\U0001f4d0  Auto niveaux (aper\u00e7u)")
            self._auto_levels_active = False
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
                "  padding:6px 8px; font-size:11px; }"
                "QPushButton:hover { background:#2a5577; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
                "  padding:6px 8px; font-size:11px; }"
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

    def _run_auto_detect(self) -> None:
        from steps.step_inpaint import _detect_artifacts
        sensitivity = self._sens_slider.value()
        img_bgr     = cv2.cvtColor(self._canvas._orig_img_rgb, cv2.COLOR_RGB2BGR)
        auto_mask   = _detect_artifacts(img_bgr, sensitivity)
        if self._add_cb.isChecked():
            auto_mask = np.maximum(self._canvas.get_mask(), auto_mask)
        self._canvas.set_mask(auto_mask)

    def _toggle_auto_levels(self) -> None:
        if self._auto_btn.isChecked():
            from steps.step_autocolor import _auto_color
            corrected = _auto_color(self._orig_img_bgr)
            self._canvas.set_display_image(corrected)
            self._auto_btn.setText("\u2713 Auto niveaux actif (aper\u00e7u)")
            self._auto_levels_active = True
        else:
            self._canvas.set_display_image(self._orig_img_bgr)
            self._auto_btn.setText("\U0001f4d0  Auto niveaux (aper\u00e7u)")
            self._auto_levels_active = False


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

    def get_mask(self) -> np.ndarray:
        return self._panel.get_mask()
