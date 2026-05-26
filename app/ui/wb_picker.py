"""ui/wb_picker.py — Dialogue de selection du point de reference pour la balance des blancs.

Usage typique :
    dlg = WBPickerDialog(parent, image_bgr, initial_pick=step.get_pick_point())
    if dlg.exec() == QDialog.DialogCode.Accepted:
        pt = dlg.get_pick_point()
        if pt:
            step.set_pick_point(*pt)
        else:
            step.clear_pick_point()

La position cliquee est sauvegardee en coordonnees image ; les valeurs RGB
sont toujours lues depuis l'image ORIGINALE passee au dialogue (pas l'image
auto-niveaux si l'apercu est actif).

Le bouton « Auto niveaux (apercu) » modifie uniquement l'affichage du canvas
pour faciliter la selection visuelle. Cette correction est jetee a la fermeture.
"""

from __future__ import annotations

import math
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QSizePolicy, QWidget, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush


# ══════════════════════════════════════════════════════════════════════════════
# Canvas
# ══════════════════════════════════════════════════════════════════════════════

class WBPickerCanvas(QWidget):
    """Zone d'affichage : image + indicateur du point selectionne.

    Clic gauche  : selectionner le point de reference.
    Survol       : afficher le curseur + les valeurs RGB.
    """

    pick_changed  = pyqtSignal(int, int)   # (x, y) en coordonnees image
    cursor_moved  = pyqtSignal(int, int)   # (x, y) en coordonnees image (survol)

    def __init__(
        self,
        image_bgr:    np.ndarray,
        initial_pick: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Image originale (pour echantillonnage RGB) — ne change jamais
        self._orig_bgr = image_bgr.copy()
        ih, iw = image_bgr.shape[:2]
        self._img_w, self._img_h = iw, ih

        # Image d'affichage (peut etre remplacee par la version auto-niveaux)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._disp_pixmap: QPixmap = self._make_rgb_pixmap(rgb)

        # Point selectionne
        self._pick_pt: tuple[int, int] | None = initial_pick

        # Patch radius (pour le cercle de visualisation)
        self._patch_radius: int = 5

        # Curseur
        self._cursor_canvas: QPoint | None = None

        # Zoom / pan
        self._zoom:     float          = 1.0
        self._pan_x:    float          = 0.0
        self._pan_y:    float          = 0.0
        self._panning:  bool           = False
        self._pan_last: QPoint | None  = None

    # ── API publique ──────────────────────────────────────────────────────────

    def get_pick_point(self) -> tuple[int, int] | None:
        return self._pick_pt

    def set_pick_point(self, pt: tuple[int, int] | None) -> None:
        self._pick_pt = pt
        self.update()

    def clear_pick_point(self) -> None:
        self._pick_pt = None
        self.update()

    def set_display_image(self, bgr: np.ndarray) -> None:
        """Change l'image affichee (auto-niveaux, etc.). Pas d'effet sur les coords."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._disp_pixmap = self._make_rgb_pixmap(rgb)
        self.update()

    def set_patch_radius(self, r: int) -> None:
        self._patch_radius = max(1, r)
        self.update()

    def get_orig_bgr(self) -> np.ndarray:
        return self._orig_bgr

    def reset_image(
        self,
        image_bgr:    np.ndarray,
        initial_pick: tuple[int, int] | None = None,
    ) -> None:
        """Réinitialise le canvas pour une nouvelle image (mode batch)."""
        self._orig_bgr = image_bgr.copy()
        ih, iw = image_bgr.shape[:2]
        self._img_w, self._img_h = iw, ih
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._disp_pixmap = self._make_rgb_pixmap(rgb)
        self._pick_pt = initial_pick
        self._cursor_canvas = None
        self._zoom    = 1.0
        self._pan_x   = 0.0
        self._pan_y   = 0.0
        self._panning = False
        self._pan_last = None
        self.update()

    # ── Conversions ──────────────────────────────────────────────────────────

    def _display_rect(self) -> QRect:
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
        """Zoom centré sur canvas_pt (molette)."""
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
        r = self._display_rect()
        if r.width() == 0 or r.height() == 0:
            return None
        ix = int((pt.x() - r.x()) * self._img_w / r.width())
        iy = int((pt.y() - r.y()) * self._img_h / r.height())
        ix = int(np.clip(ix, 0, self._img_w - 1))
        iy = int(np.clip(iy, 0, self._img_h - 1))
        return ix, iy

    def _img_to_canvas(self, ix: int, iy: int) -> QPoint:
        r = self._display_rect()
        cx = int(r.x() + ix * r.width()  / self._img_w)
        cy = int(r.y() + iy * r.height() / self._img_h)
        return QPoint(cx, cy)

    def _patch_canvas_radius(self) -> float:
        r = self._display_rect()
        if self._img_w == 0:
            return 0.0
        return self._patch_radius * r.width() / self._img_w

    # ── Evenements ───────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning  = True
            self._pan_last = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            coords = self._canvas_to_img(event.position().toPoint())
            if coords:
                self._pick_pt = coords
                self.pick_changed.emit(*coords)
                self.update()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._panning and self._pan_last is not None:
            self._pan_x   += pos.x() - self._pan_last.x()
            self._pan_y   += pos.y() - self._pan_last.y()
            self._pan_last = pos
            self.update()
            event.accept()
            return
        self._cursor_canvas = pos
        coords = self._canvas_to_img(pos)
        if coords:
            self.cursor_moved.emit(*coords)
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning  = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()

    def wheelEvent(self, event):
        """Molette = zoom (centré sous le curseur)."""
        delta  = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom_at(event.position().toPoint(), factor)

    def leaveEvent(self, event):
        self._cursor_canvas = None
        self.update()

    # ── Rendu ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_rgb_pixmap(rgb: np.ndarray) -> QPixmap:
        h, w = rgb.shape[:2]
        data = np.ascontiguousarray(rgb)
        qimg = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def paintEvent(self, _event):
        rect = self._display_rect()
        p    = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fond
        p.fillRect(self.rect(), QColor(22, 22, 32))

        # Image
        p.drawPixmap(rect, self._disp_pixmap)

        # Point selectionne : cercle dore + reticule
        if self._pick_pt is not None:
            cx, cy = self._img_to_canvas(*self._pick_pt).x(), self._img_to_canvas(*self._pick_pt).y()
            cr      = max(6, self._patch_canvas_radius())

            # Cercle exterieur
            p.setPen(QPen(QColor(30, 30, 30), 3.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(int(cx - cr), int(cy - cr), int(cr * 2), int(cr * 2))
            p.setPen(QPen(QColor(255, 200, 40), 2.0))
            p.drawEllipse(int(cx - cr), int(cy - cr), int(cr * 2), int(cr * 2))

            # Reticule interne
            arm = int(cr * 0.7)
            p.setPen(QPen(QColor(30, 30, 30), 2.5))
            p.drawLine(cx - arm, cy, cx + arm, cy)
            p.drawLine(cx, cy - arm, cx, cy + arm)
            p.setPen(QPen(QColor(255, 200, 40), 1.5))
            p.drawLine(cx - arm, cy, cx + arm, cy)
            p.drawLine(cx, cy - arm, cx, cy + arm)

        # Curseur souris (fin reticule blanc)
        if self._cursor_canvas is not None:
            mx, my = self._cursor_canvas.x(), self._cursor_canvas.y()
            half   = 12
            p.setPen(QPen(QColor(0, 0, 0), 2.0))
            p.drawLine(mx - half, my, mx + half, my)
            p.drawLine(mx, my - half, mx, my + half)
            p.setPen(QPen(QColor(255, 255, 255), 1.0))
            p.drawLine(mx - half, my, mx + half, my)
            p.drawLine(mx, my - half, mx, my + half)

        p.end()


# ══════════════════════════════════════════════════════════════════════════════
# Panel embarquable (canvas + contrôles)
# ══════════════════════════════════════════════════════════════════════════════

class WBPickerPanel(QWidget):
    """Canvas + panneau de contrôles de balance des blancs, embarquable.

    En mode inline (batch) ::

        panel = WBPickerPanel(image_bgr)
        # intégrer dans un layout...
        panel.set_image(new_bgr, saved_pick, saved_radius)
        pt     = panel.get_pick_point()
        radius = panel.get_patch_radius()

    En mode dialogue : utiliser WBPickerDialog.
    """

    accepted = pyqtSignal()   # émis par le bouton Valider (si show_ok_cancel=True)
    rejected = pyqtSignal()   # émis par le bouton Annuler  (si show_ok_cancel=True)

    def __init__(
        self,
        image_bgr:      np.ndarray,
        initial_pick:   tuple[int, int] | None = None,
        parent=None,
        show_ok_cancel: bool = False,
        sidebar_width:  int  = 225,
    ) -> None:
        super().__init__(parent)
        self._orig_bgr           = image_bgr.copy()
        self._auto_levels_active = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Canvas ────────────────────────────────────────────────────────────
        self._canvas = WBPickerCanvas(image_bgr, initial_pick)
        self._canvas.pick_changed.connect(self._on_pick_changed)
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        root.addWidget(self._canvas, stretch=1)

        # ── Barre latérale ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(sidebar_width)
        sidebar.setStyleSheet("background: #1a1a2e;")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(12, 14, 12, 14)
        sl.setSpacing(8)

        title = QLabel("Balance des blancs")
        title.setStyleSheet("color:#ddd; font-size:13px; font-weight:700;")
        sl.addWidget(title)
        sl.addWidget(self._hline())

        tips = QLabel(
            "Cliquez sur une zone\n"
            "neutre (blanc ou gris)\n"
            "de l'image.\n\n"
            "La correction sera calculée\n"
            "depuis les couleurs actuelles\n"
            "au moment du traitement."
        )
        tips.setStyleSheet("color:#7a9ab0; font-size:10px;")
        tips.setWordWrap(True)
        sl.addWidget(tips)
        sl.addWidget(self._hline())

        lbl_cur = QLabel("Sous le curseur :")
        lbl_cur.setStyleSheet("color:#bbb; font-size:10px;")
        sl.addWidget(lbl_cur)

        cur_row = QHBoxLayout()
        self._cursor_color_box = self._color_box()
        cur_row.addWidget(self._cursor_color_box)
        self._cursor_rgb_lbl = QLabel("—")
        self._cursor_rgb_lbl.setStyleSheet(
            "color:#9de; font-size:10px; font-family:Consolas,monospace;"
        )
        cur_row.addWidget(self._cursor_rgb_lbl, stretch=1)
        sl.addLayout(cur_row)
        sl.addWidget(self._hline())

        lbl_pick = QLabel("Point sélectionné :")
        lbl_pick.setStyleSheet("color:#bbb; font-size:11px; font-weight:600;")
        sl.addWidget(lbl_pick)

        pick_row = QHBoxLayout()
        self._pick_color_box = self._color_box()
        pick_row.addWidget(self._pick_color_box)
        self._pick_info_lbl = QLabel("Aucun")
        self._pick_info_lbl.setStyleSheet(
            "color:#ccc; font-size:10px; font-family:Consolas,monospace;"
        )
        self._pick_info_lbl.setWordWrap(True)
        pick_row.addWidget(self._pick_info_lbl, stretch=1)
        sl.addLayout(pick_row)

        self._muls_lbl = QLabel("")
        self._muls_lbl.setStyleSheet(
            "color:#9de; font-size:10px; font-family:Consolas,monospace;"
        )
        self._muls_lbl.setWordWrap(True)
        sl.addWidget(self._muls_lbl)

        self._warn_lbl = QLabel("")
        self._warn_lbl.setStyleSheet("color:#f99; font-size:10px;")
        self._warn_lbl.setWordWrap(True)
        sl.addWidget(self._warn_lbl)
        sl.addWidget(self._hline())

        lbl_rad = QLabel("Rayon d'échantillonnage :")
        lbl_rad.setStyleSheet("color:#bbb; font-size:11px;")
        sl.addWidget(lbl_rad)

        rad_row = QHBoxLayout()
        self._rad_val_lbl = QLabel("5 px")
        self._rad_val_lbl.setStyleSheet(
            "color:#9de; font-size:12px; font-weight:700; min-width:42px;"
        )
        rad_row.addWidget(self._rad_val_lbl)
        rad_row.addStretch()
        sl.addLayout(rad_row)

        self._rad_slider = QSlider(Qt.Orientation.Horizontal)
        self._rad_slider.setRange(1, 30)
        self._rad_slider.setValue(5)
        self._rad_slider.valueChanged.connect(self._on_radius_changed)
        sl.addWidget(self._rad_slider)

        btn_clear_wb = QPushButton("✖  Effacer la sélection")
        btn_clear_wb.clicked.connect(self._clear_pick)
        self._style(btn_clear_wb)
        sl.addWidget(btn_clear_wb)
        sl.addWidget(self._hline())

        self._auto_btn = QPushButton("📐  Auto niveaux (aperçu)")
        self._auto_btn.setCheckable(True)
        self._auto_btn.clicked.connect(self._toggle_auto_levels)
        self._style(self._auto_btn)
        sl.addWidget(self._auto_btn)

        lbl_auto_info = QLabel(
            "Aperçu seulement — les\ncoord. conservent leurs\nvaleurs d'origine."
        )
        lbl_auto_info.setStyleSheet("color:#556; font-size:9px;")
        sl.addWidget(lbl_auto_info)
        sl.addStretch()

        if show_ok_cancel:
            sl.addWidget(self._hline())
            btn_ok = QPushButton("✓  Valider")
            btn_ok.clicked.connect(self.accepted.emit)
            self._style(btn_ok, accent=True)
            sl.addWidget(btn_ok)

            btn_cancel = QPushButton("✗  Annuler")
            btn_cancel.clicked.connect(self.rejected.emit)
            self._style(btn_cancel)
            sl.addWidget(btn_cancel)

        root.addWidget(sidebar)

        if initial_pick is not None:
            self._refresh_pick_info(initial_pick[0], initial_pick[1])

    # ── API publique ──────────────────────────────────────────────────────────

    def get_pick_point(self) -> tuple[int, int] | None:
        return self._canvas.get_pick_point()

    def get_patch_radius(self) -> int:
        return self._rad_slider.value()

    def set_image(
        self,
        image_bgr:    np.ndarray,
        initial_pick: tuple[int, int] | None = None,
        patch_radius: int = 5,
    ) -> None:
        """Change l'image affichée (mode batch — changement de photo)."""
        self._orig_bgr = image_bgr.copy()
        if self._auto_levels_active:
            self._auto_btn.setChecked(False)
            self._auto_btn.setText("📐  Auto niveaux (aperçu)")
            self._auto_levels_active = False
        self._canvas.reset_image(image_bgr, initial_pick)
        self._rad_slider.setValue(patch_radius)
        self._rad_val_lbl.setText(f"{patch_radius} px")
        self._pick_color_box.setStyleSheet(
            "background:#333; border-radius:3px; border:1px solid #444;"
        )
        if initial_pick is not None:
            self._refresh_pick_info(initial_pick[0], initial_pick[1])
        else:
            self._pick_info_lbl.setText("Aucun")
            self._muls_lbl.setText("")
            self._warn_lbl.setText("")

    # ── Helpers UI ────────────────────────────────────────────────────────────

    @staticmethod
    def _hline() -> QFrame:
        f = QFrame()
        f.setFixedHeight(1)
        f.setStyleSheet("background:#252545; margin:2px 0;")
        return f

    @staticmethod
    def _color_box() -> QLabel:
        lbl = QLabel()
        lbl.setFixedSize(20, 20)
        lbl.setStyleSheet("background:#333; border-radius:3px; border:1px solid #444;")
        return lbl

    @staticmethod
    def _style(btn: QPushButton, accent: bool = False) -> None:
        if accent:
            btn.setStyleSheet(
                "QPushButton { background:#1e3a52; color:#b8e0f7; border-radius:4px;"
                "  padding:6px 8px; font-size:11px; }"
                "QPushButton:hover { background:#2a5577; }"
                "QPushButton:checked { background:#2a5577; color:#fff; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
                "  padding:6px 8px; font-size:11px; }"
                "QPushButton:hover { background:#2a2a50; color:#ccc; }"
                "QPushButton:checked { background:#2a4a6a; color:#adf; }"
            )

    def _set_color_box(self, box: QLabel, r: float, g: float, b: float) -> None:
        ri, gi, bi = int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))
        box.setStyleSheet(
            f"background: rgb({ri},{gi},{bi}); border-radius:3px; border:1px solid #555;"
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_cursor_moved(self, ix: int, iy: int) -> None:
        r_val = int(self._orig_bgr[iy, ix, 2])
        g_val = int(self._orig_bgr[iy, ix, 1])
        b_val = int(self._orig_bgr[iy, ix, 0])
        self._cursor_rgb_lbl.setText(f"R:{r_val:3d}  G:{g_val:3d}  B:{b_val:3d}")
        self._set_color_box(self._cursor_color_box, r_val, g_val, b_val)

    def _on_pick_changed(self, ix: int, iy: int) -> None:
        self._refresh_pick_info(ix, iy)

    def _refresh_pick_info(self, ix: int, iy: int) -> None:
        from steps.step_wb import sample_patch_info
        radius = self._rad_slider.value()
        info   = sample_patch_info(self._orig_bgr, ix, iy, radius)
        r, g, b = info["rgb_means"]
        mr, mg, mb = info["muls"]
        self._set_color_box(self._pick_color_box, r, g, b)
        self._pick_info_lbl.setText(
            f"({ix}, {iy})\nR:{r:.0f}  G:{g:.0f}  B:{b:.0f}"
        )
        self._muls_lbl.setText(f"×{mr:.3f}  ×{mg:.3f}  ×{mb:.3f}")
        if info["too_dark"]:
            self._warn_lbl.setText("⚠ Zone trop sombre.\nChoisissez une zone plus claire.")
        else:
            self._warn_lbl.setText("")

    def _on_radius_changed(self, val: int) -> None:
        self._rad_val_lbl.setText(f"{val} px")
        self._canvas.set_patch_radius(val)
        pt = self._canvas.get_pick_point()
        if pt is not None:
            self._refresh_pick_info(*pt)

    def _clear_pick(self) -> None:
        self._canvas.clear_pick_point()
        self._pick_color_box.setStyleSheet(
            "background:#333; border-radius:3px; border:1px solid #444;"
        )
        self._pick_info_lbl.setText("Aucun")
        self._muls_lbl.setText("")
        self._warn_lbl.setText("")

    def _toggle_auto_levels(self) -> None:
        if self._auto_btn.isChecked():
            from steps.step_autocolor import _auto_color
            corrected = _auto_color(self._orig_bgr)
            self._canvas.set_display_image(corrected)
            self._auto_btn.setText("✓ Auto niveaux actif (aperçu)")
            self._auto_levels_active = True
        else:
            self._canvas.set_display_image(self._orig_bgr)
            self._auto_btn.setText("📐  Auto niveaux (aperçu)")
            self._auto_levels_active = False


# ══════════════════════════════════════════════════════════════════════════════
# Dialogue
# ══════════════════════════════════════════════════════════════════════════════

class WBPickerDialog(QDialog):
    """Dialogue de balance des blancs — wrapper fin autour de WBPickerPanel."""

    def __init__(
        self,
        parent,
        image_bgr:    np.ndarray,
        initial_pick: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Balance des blancs — sélection du point de référence")
        self.setModal(True)
        self.setMinimumSize(900, 600)
        self.resize(1100, 750)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._panel = WBPickerPanel(image_bgr, initial_pick, show_ok_cancel=True)
        self._panel.accepted.connect(self.accept)
        self._panel.rejected.connect(self.reject)
        root.addWidget(self._panel)

    def get_pick_point(self) -> tuple[int, int] | None:
        return self._panel.get_pick_point()

    def get_patch_radius(self) -> int:
        return self._panel.get_patch_radius()


    def __init__(
        self,
        parent,
        image_bgr:    np.ndarray,
        initial_pick: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Balance des blancs — sélection du point de référence")
        self.setModal(True)
        self.setMinimumSize(900, 600)
        self.resize(1100, 750)

        self._orig_bgr             = image_bgr.copy()
        self._auto_levels_active   = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Canvas ────────────────────────────────────────────────────────────
        self._canvas = WBPickerCanvas(image_bgr, initial_pick)
        self._canvas.pick_changed.connect(self._on_pick_changed)
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        root.addWidget(self._canvas, stretch=1)

        # ── Barre laterale ────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(225)
        sidebar.setStyleSheet("background: #1a1a2e;")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(12, 14, 12, 14)
        sl.setSpacing(8)

        # Titre
        title = QLabel("Balance des blancs")
        title.setStyleSheet("color:#ddd; font-size:13px; font-weight:700;")
        sl.addWidget(title)
        sl.addWidget(self._hline())

        # Instructions
        tips = QLabel(
            "Cliquez sur une zone\n"
            "neutre (blanc ou gris)\n"
            "de l'image.\n\n"
            "La correction sera calculée\n"
            "depuis les couleurs actuelles\n"
            "au moment du traitement."
        )
        tips.setStyleSheet("color:#7a9ab0; font-size:10px;")
        tips.setWordWrap(True)
        sl.addWidget(tips)
        sl.addWidget(self._hline())

        # Couleur sous le curseur
        lbl_cur = QLabel("Sous le curseur :")
        lbl_cur.setStyleSheet("color:#bbb; font-size:10px;")
        sl.addWidget(lbl_cur)

        cur_row = QHBoxLayout()
        self._cursor_color_box = self._color_box()
        cur_row.addWidget(self._cursor_color_box)
        self._cursor_rgb_lbl = QLabel("—")
        self._cursor_rgb_lbl.setStyleSheet(
            "color:#9de; font-size:10px; font-family:Consolas,monospace;"
        )
        cur_row.addWidget(self._cursor_rgb_lbl, stretch=1)
        sl.addLayout(cur_row)
        sl.addWidget(self._hline())

        # Point selectionne
        lbl_pick = QLabel("Point sélectionné :")
        lbl_pick.setStyleSheet("color:#bbb; font-size:11px; font-weight:600;")
        sl.addWidget(lbl_pick)

        pick_row = QHBoxLayout()
        self._pick_color_box = self._color_box()
        pick_row.addWidget(self._pick_color_box)
        self._pick_info_lbl = QLabel("Aucun")
        self._pick_info_lbl.setStyleSheet(
            "color:#ccc; font-size:10px; font-family:Consolas,monospace;"
        )
        self._pick_info_lbl.setWordWrap(True)
        pick_row.addWidget(self._pick_info_lbl, stretch=1)
        sl.addLayout(pick_row)

        self._muls_lbl = QLabel("")
        self._muls_lbl.setStyleSheet("color:#9de; font-size:10px; font-family:Consolas,monospace;")
        self._muls_lbl.setWordWrap(True)
        sl.addWidget(self._muls_lbl)

        self._warn_lbl = QLabel("")
        self._warn_lbl.setStyleSheet("color:#f99; font-size:10px;")
        self._warn_lbl.setWordWrap(True)
        sl.addWidget(self._warn_lbl)
        sl.addWidget(self._hline())

        # Rayon de l'echantillon
        lbl_rad = QLabel("Rayon d'échantillonnage :")
        lbl_rad.setStyleSheet("color:#bbb; font-size:11px;")
        sl.addWidget(lbl_rad)

        rad_row = QHBoxLayout()
        self._rad_val_lbl = QLabel("5 px")
        self._rad_val_lbl.setStyleSheet("color:#9de; font-size:12px; font-weight:700; min-width:42px;")
        rad_row.addWidget(self._rad_val_lbl)
        rad_row.addStretch()
        sl.addLayout(rad_row)

        self._rad_slider = QSlider(Qt.Orientation.Horizontal)
        self._rad_slider.setRange(1, 30)
        self._rad_slider.setValue(5)
        self._rad_slider.valueChanged.connect(self._on_radius_changed)
        sl.addWidget(self._rad_slider)

        btn_clear = QPushButton("✖  Effacer la sélection")
        btn_clear.clicked.connect(self._clear_pick)
        self._style(btn_clear)
        sl.addWidget(btn_clear)
        sl.addWidget(self._hline())

        # Bouton auto-niveaux
        self._auto_btn = QPushButton("📐  Auto niveaux (aperçu)")
        self._auto_btn.setCheckable(True)
        self._auto_btn.clicked.connect(self._toggle_auto_levels)
        self._style(self._auto_btn)
        sl.addWidget(self._auto_btn)

        lbl_auto_info = QLabel("Aperçu seulement — les\ncoordonnées conservent\nleurs valeurs d'origine.")
        lbl_auto_info.setStyleSheet("color:#556; font-size:9px;")
        sl.addWidget(lbl_auto_info)

        sl.addStretch()
        sl.addWidget(self._hline())

        # Valider / Annuler
        btn_ok = QPushButton("✓  Valider")
        btn_ok.clicked.connect(self.accept)
        self._style(btn_ok, accent=True)
        sl.addWidget(btn_ok)

        btn_cancel = QPushButton("✗  Annuler")
        btn_cancel.clicked.connect(self.reject)
        self._style(btn_cancel)
        sl.addWidget(btn_cancel)

        root.addWidget(sidebar)

        # Initialiser l'affichage si un point est deja selectionne
        if initial_pick is not None:
            self._refresh_pick_info(initial_pick[0], initial_pick[1])

    # ── Helpers UI ────────────────────────────────────────────────────────────

    @staticmethod
    def _hline() -> QFrame:
        f = QFrame()
        f.setFixedHeight(1)
        f.setStyleSheet("background:#252545; margin:2px 0;")
        return f

    @staticmethod
    def _color_box() -> QLabel:
        lbl = QLabel()
        lbl.setFixedSize(20, 20)
        lbl.setStyleSheet("background:#333; border-radius:3px; border:1px solid #444;")
        return lbl

    @staticmethod
    def _style(btn: QPushButton, accent: bool = False) -> None:
        if accent:
            btn.setStyleSheet(
                "QPushButton { background:#1e3a52; color:#b8e0f7; border-radius:4px;"
                "  padding:6px 8px; font-size:11px; }"
                "QPushButton:hover { background:#2a5577; }"
                "QPushButton:checked { background:#2a5577; color:#fff; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
                "  padding:6px 8px; font-size:11px; }"
                "QPushButton:hover { background:#2a2a50; color:#ccc; }"
                "QPushButton:checked { background:#2a4a6a; color:#adf; }"
            )

    def _set_color_box(self, box: QLabel, r: float, g: float, b: float) -> None:
        ri, gi, bi = int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))
        box.setStyleSheet(
            f"background: rgb({ri},{gi},{bi}); border-radius:3px; border:1px solid #555;"
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_cursor_moved(self, ix: int, iy: int) -> None:
        """Met a jour l'affichage des valeurs RGB sous le curseur."""
        r_val = int(self._orig_bgr[iy, ix, 2])
        g_val = int(self._orig_bgr[iy, ix, 1])
        b_val = int(self._orig_bgr[iy, ix, 0])
        self._cursor_rgb_lbl.setText(f"R:{r_val:3d}  G:{g_val:3d}  B:{b_val:3d}")
        self._set_color_box(self._cursor_color_box, r_val, g_val, b_val)

    def _on_pick_changed(self, ix: int, iy: int) -> None:
        self._refresh_pick_info(ix, iy)

    def _refresh_pick_info(self, ix: int, iy: int) -> None:
        """Met a jour la section « Point selectionne » en lisant l'image d'ORIGINE."""
        from steps.step_wb import sample_patch_info
        radius = self._rad_slider.value()
        info   = sample_patch_info(self._orig_bgr, ix, iy, radius)
        r, g, b = info["rgb_means"]
        mr, mg, mb = info["muls"]
        self._set_color_box(self._pick_color_box, r, g, b)
        self._pick_info_lbl.setText(
            f"({ix}, {iy})\n"
            f"R:{r:.0f}  G:{g:.0f}  B:{b:.0f}"
        )
        self._muls_lbl.setText(f"×{mr:.3f}  ×{mg:.3f}  ×{mb:.3f}")
        if info["too_dark"]:
            self._warn_lbl.setText(
                "⚠ Zone trop sombre.\nChoisissez une zone plus claire."
            )
        else:
            self._warn_lbl.setText("")

    def _on_radius_changed(self, val: int) -> None:
        self._rad_val_lbl.setText(f"{val} px")
        self._canvas.set_patch_radius(val)
        pt = self._canvas.get_pick_point()
        if pt is not None:
            self._refresh_pick_info(*pt)

    def _clear_pick(self) -> None:
        self._canvas.clear_pick_point()
        self._pick_color_box.setStyleSheet(
            "background:#333; border-radius:3px; border:1px solid #444;"
        )
        self._pick_info_lbl.setText("Aucun")
        self._muls_lbl.setText("")
        self._warn_lbl.setText("")

    def _toggle_auto_levels(self) -> None:
        if self._auto_btn.isChecked():
            # Activer : calculer et afficher la version auto-niveaux
            from steps.step_autocolor import _auto_color
            corrected = _auto_color(self._orig_bgr)
            self._canvas.set_display_image(corrected)
            self._auto_btn.setText("✓ Auto niveaux actif (aperçu)")
            self._auto_levels_active = True
        else:
            # Desactiver : revenir a l'image originale
            self._canvas.set_display_image(self._orig_bgr)
            self._auto_btn.setText("📐  Auto niveaux (aperçu)")
            self._auto_levels_active = False

    # ── Resultat ──────────────────────────────────────────────────────────────

    def get_pick_point(self) -> tuple[int, int] | None:
        return self._canvas.get_pick_point()

    def get_patch_radius(self) -> int:
        return self._rad_slider.value()
