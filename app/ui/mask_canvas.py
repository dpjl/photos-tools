"""ui/mask_canvas.py — Canvas de dessin du masque de retouche.

Widget Qt affichant l'image + un overlay (masque rouge, sélection, mode revue VLM
violet/vert) et gérant le dessin au pinceau, le zoom/pan, l'historique undo/redo,
la sélection de zones, le mode « peek » et la revue des candidats VLM.

Utilisé par MaskCanvasPanel (cf. ui/mask_editor.py).
"""

from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtWidgets import QWidget, QSizePolicy, QToolTip
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

_MAX_UNDO = 20   # nombre maximum d'etats undo conserves en memoire


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
        # Infobulle lisible (raison du VLM en mode revue) : texte clair sur fond sombre
        self.setStyleSheet(
            "QToolTip { color:#eaeaf2; background-color:#23232e;"
            " border:1px solid #4a4a6a; padding:4px; font-size:11px; }"
        )

        # Image en RGB pour QImage
        # _orig_img_rgb : image de reference
        # _base_pixmap  : ce qui est affiche
        self._orig_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ih, iw = self._orig_img_rgb.shape[:2]
        self._img_w, self._img_h = iw, ih
        self._display_w, self._display_h = iw, ih

        # Image actuellement affichée en BGR (= base au départ, puis preview courant
        # via set_display_image). Sert de source à la détection auto d'artefacts.
        self._display_bgr = image_bgr.copy()

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

        # Historique undo/redo
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []  # vidé à chaque nouveau trait

        # Sélection d'une zone (composante connexe) + mode « peek »
        self._selected_mask: np.ndarray | None = None   # uint8, zone sélectionnée
        self._peek:          bool               = False  # masque l'overlay pour voir dessous

        # Mode « revue VLM » : candidats colorés (violet=décor à retirer, vert=défaut gardé)
        self._review_labels: np.ndarray | None = None   # int32, id de composante candidate
        self._review_cats:   dict               = {}     # {id: "scene"|"defect"}
        self._review_reasons: dict              = {}     # {id: texte (verdict + raison VLM)}
        self._review_scene:  np.ndarray | None = None   # bool, pixels à retirer (violet)
        self._review_defect: np.ndarray | None = None   # bool, pixels gardés (vert)
        self._review_hover_lbl: int             = 0      # composante survolée (anti-clignotement)

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
        self._display_w, self._display_h = iw, ih
        self._display_bgr = image_bgr.copy()
        self._base_pixmap = self._make_rgb_pixmap(self._orig_img_rgb)
        if initial_mask is not None and initial_mask.shape[:2] == (ih, iw):
            self._mask = initial_mask.copy()
        else:
            self._mask = np.zeros((ih, iw), dtype=np.uint8)
        self._dirty           = True
        self._composite_cache = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._selected_mask   = None
        self._peek            = False
        self._clear_review_state()
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
        """Change l'image affichee (preview courant). Le masque n'est pas affecte."""
        self._display_bgr = bgr.copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dh, dw = rgb.shape[:2]
        self._display_w, self._display_h = dw, dh
        self._base_pixmap = self._make_rgb_pixmap(rgb)
        self._dirty = True
        self.update()

    def get_display_image(self) -> np.ndarray:
        """Image actuellement affichée en BGR (preview courant ou base)."""
        return self._display_bgr.copy()

    def set_mask(self, mask: np.ndarray, push_undo: bool = True) -> None:
        """Remplace le masque. ``push_undo=False`` pour les mises à jour live."""
        if push_undo:
            self._push_undo()
        self._mask = mask.copy()
        self._selected_mask = None
        self._clear_review_state()
        self._invalidate()

    def push_undo(self) -> None:
        """Empile l'état courant (point de restauration avant une action externe)."""
        self._push_undo()

    def clear_mask(self) -> None:
        self._push_undo()
        self._mask[:] = 0
        self._selected_mask = None
        self._clear_review_state()
        self._invalidate()

    def undo(self) -> None:
        if self._undo_stack:
            self._redo_stack.append(self._mask.copy())
            self._mask = self._undo_stack.pop()
            self._selected_mask = None
            self._clear_review_state()
            self._invalidate()

    # ── Sélection / suppression d'une zone, peek ──────────────────────────────

    def select_component_at(self, ix: int, iy: int) -> bool:
        """Sélectionne la composante connexe du masque sous (ix, iy). Retourne True si OK."""
        if not (0 <= iy < self._mask.shape[0] and 0 <= ix < self._mask.shape[1]):
            return False
        if self._mask[iy, ix] == 0:
            self.clear_selection()
            return False
        n, labels = cv2.connectedComponents((self._mask > 0).astype(np.uint8), 8)
        lbl = labels[iy, ix]
        if lbl == 0:
            return False
        self._selected_mask = (labels == lbl).astype(np.uint8) * 255
        self._invalidate()
        return True

    def clear_selection(self) -> None:
        if self._selected_mask is not None:
            self._selected_mask = None
            self._invalidate()

    def has_selection(self) -> bool:
        return self._selected_mask is not None

    def delete_selected_component(self) -> bool:
        """Efface la zone sélectionnée du masque (undo possible)."""
        if self._selected_mask is None:
            return False
        self._push_undo()
        self._mask[self._selected_mask > 0] = 0
        self._selected_mask = None
        self._invalidate()
        return True

    def toggle_peek(self) -> None:
        """Affiche/masque temporairement l'overlay rouge (voir la photo dessous)."""
        self._peek = not self._peek
        self._invalidate()

    # ── Mode revue VLM ────────────────────────────────────────────────────────

    def in_review(self) -> bool:
        return self._review_labels is not None

    def _clear_review_state(self) -> None:
        self._review_labels = None
        self._review_cats = {}
        self._review_reasons = {}
        self._review_scene = None
        self._review_defect = None
        self._review_hover_lbl = 0

    def _rebuild_review_masks(self) -> None:
        """Recalcule les masques booléens violet/vert depuis les catégories."""
        if self._review_labels is None:
            return
        scene = np.zeros(self._mask.shape, dtype=bool)
        defect = np.zeros(self._mask.shape, dtype=bool)
        for lbl, cat in self._review_cats.items():
            comp = self._review_labels == lbl
            if cat == "scene":
                scene |= comp
            else:
                defect |= comp
        self._review_scene, self._review_defect = scene, defect

    def enter_review(self, labels: np.ndarray, categories: dict,
                     reasons: dict | None = None) -> None:
        """Entre en mode revue : colore les candidats (violet=décor, vert=défaut)."""
        if labels.shape[:2] != self._mask.shape[:2]:
            labels = cv2.resize(labels.astype(np.int32),
                                (self._mask.shape[1], self._mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        self._review_labels = labels.astype(np.int32)
        self._review_cats = dict(categories)
        self._review_reasons = dict(reasons or {})
        self._review_hover_lbl = 0
        self._selected_mask = None
        self._rebuild_review_masks()
        self._invalidate()

    def review_counts(self) -> tuple[int, int]:
        """(nb décor, nb défaut) dans la revue courante."""
        scene = sum(1 for v in self._review_cats.values() if v == "scene")
        return scene, len(self._review_cats) - scene

    def _toggle_review_at(self, ix: int, iy: int) -> bool:
        """Bascule la catégorie de la composante candidate sous (ix, iy)."""
        if self._review_labels is None:
            return False
        if not (0 <= iy < self._review_labels.shape[0] and 0 <= ix < self._review_labels.shape[1]):
            return False
        lbl = int(self._review_labels[iy, ix])
        if lbl == 0:
            return False
        self._review_cats[lbl] = "defect" if self._review_cats.get(lbl) == "scene" else "scene"
        self._rebuild_review_masks()
        self._invalidate()
        return True

    def apply_review(self) -> int:
        """Supprime du masque toutes les zones « décor » (violet) et quitte la revue.

        Retourne le nombre de composantes retirées.
        """
        if self._review_labels is None:
            return 0
        n_scene = sum(1 for v in self._review_cats.values() if v == "scene")
        self._push_undo()
        if self._review_scene is not None:
            self._mask[self._review_scene] = 0
        self._clear_review_state()
        self._invalidate()
        return n_scene

    def redo(self) -> None:
        if self._redo_stack:
            self._undo_stack.append(self._mask.copy())
            self._mask = self._redo_stack.pop()
            self._selected_mask = None
            self._clear_review_state()
            self._invalidate()

    def get_zoom_state(self) -> tuple:
        """Retourne (zoom_ratio, cx_rel, cy_rel) normalisés — interface commune."""
        cw, ch = self.width(), self.height()
        if cw == 0 or ch == 0 or self._display_w == 0 or self._display_h == 0:
            return (1.0, 0.5, 0.5)
        fit  = min(cw / self._display_w, ch / self._display_h)
        dw   = self._display_w * fit * self._zoom
        dh   = self._display_h * fit * self._zoom
        cx_rel = 0.5 - self._pan_x / dw if dw else 0.5
        cy_rel = 0.5 - self._pan_y / dh if dh else 0.5
        return (self._zoom, cx_rel, cy_rel)

    def apply_zoom_state(self, zoom_ratio: float, cx_rel: float, cy_rel: float) -> None:
        """Applique un état de vue normalisé (reçu de l'onglet précédent)."""
        self._zoom = max(0.5, min(16.0, zoom_ratio))
        cw, ch = self.width(), self.height()
        if cw == 0 or ch == 0 or self._display_w == 0 or self._display_h == 0:
            return
        fit  = min(cw / self._display_w, ch / self._display_h)
        dw   = self._display_w * fit * self._zoom
        dh   = self._display_h * fit * self._zoom
        self._pan_x = (0.5 - cx_rel) * dw
        self._pan_y = (0.5 - cy_rel) * dh
        self.update()

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
        fit_scale = min(cw / self._display_w, ch / self._display_h)
        scale     = fit_scale * self._zoom
        dw        = int(self._display_w * scale)
        dh        = int(self._display_h * scale)
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
        fit_scale  = min(cw / self._display_w, ch / self._display_h) if cw and ch else 1.0
        new_scale  = fit_scale * self._zoom
        dw, dh     = int(self._display_w * new_scale), int(self._display_h * new_scale)
        self._pan_x = canvas_pt.x() - fx * dw - (cw - dw) // 2
        self._pan_y = canvas_pt.y() - fy * dh - (ch - dh) // 2
        self.update()

    def _canvas_to_img(self, pt: QPoint) -> tuple[int, int] | None:
        """Convertit les coordonnees canvas en coordonnees image."""
        r = self._display_rect()
        if r.width() == 0 or r.height() == 0:
            return None
        dx = (pt.x() - r.x()) * self._display_w / r.width()
        dy = (pt.y() - r.y()) * self._display_h / r.height()
        ix = int(dx * self._img_w / max(1, self._display_w))
        iy = int(dy * self._img_h / max(1, self._display_h))
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
        self._redo_stack.clear()  # nouveau trait → plus de redo disponible

    def _invalidate(self) -> None:
        self._dirty = True
        self.update()

    # ── Evenements souris ─────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        pos  = event.position().toPoint()
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning  = True
            self._pan_last = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if ctrl:
                # Ctrl+clic gauche → peindre
                self._push_undo()
                self._selected_mask = None
                self._clear_review_state()
                self._painting = True
                self._erasing  = False
                self._last_canvas_pt = pos
                coords = self._canvas_to_img(pos)
                if coords:
                    cv2.circle(self._mask, coords, self._brush_px, 255, -1)
                    self._invalidate()
            elif self.in_review():
                # Mode revue : clic sur un candidat = on attend le double-clic
                # (bascule de catégorie) ; ailleurs = déplacement.
                coords = self._canvas_to_img(pos)
                if not (coords and self._review_labels[coords[1], coords[0]] > 0):
                    self._panning  = True
                    self._pan_last = pos
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                # Clic gauche : sélectionner la zone rouge sous le curseur,
                # sinon déplacer (pan).
                coords = self._canvas_to_img(pos)
                if coords and self._mask[coords[1], coords[0]] > 0:
                    self.select_component_at(coords[0], coords[1])
                else:
                    self.clear_selection()
                    self._panning  = True
                    self._pan_last = pos
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.RightButton:
            if ctrl:
                # Ctrl+clic droit → effacer
                self._push_undo()
                self._selected_mask = None
                self._clear_review_state()
                self._erasing  = True
                self._painting = False
                self._last_canvas_pt = pos
                coords = self._canvas_to_img(pos)
                if coords:
                    cv2.circle(self._mask, coords, self._brush_px, 0, -1)
                    self._invalidate()
            # Clic droit sans Ctrl : ignoré

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
            if self.in_review():
                self._update_review_tooltip(pos, event.globalPosition().toPoint())
            self.update()  # reafficher juste le curseur

    def _update_review_tooltip(self, canvas_pt: QPoint, global_pt: QPoint) -> None:
        """Affiche la raison du VLM en infobulle quand on survole un candidat."""
        coords = self._canvas_to_img(canvas_pt)
        lbl = int(self._review_labels[coords[1], coords[0]]) if coords else 0
        if lbl == self._review_hover_lbl:
            return                          # même composante : éviter le clignotement
        self._review_hover_lbl = lbl
        text = self._review_reasons.get(lbl) if lbl else None
        if text:
            QToolTip.showText(global_pt, text, self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            if self._panning:
                self._panning  = False
                self._pan_last = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._painting       = False
            self._erasing        = False
            self._last_canvas_pt = None

    def mouseDoubleClickEvent(self, event):
        """En mode revue : double-clic sur un candidat = bascule décor ↔ défaut."""
        if self.in_review() and event.button() == Qt.MouseButton.LeftButton \
                and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            coords = self._canvas_to_img(event.position().toPoint())
            if coords and self._toggle_review_at(coords[0], coords[1]):
                self._panning = False
                self._pan_last = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Ctrl + molette = zoom
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            self._zoom_at(event.position().toPoint(), factor)
        else:
            # Molette seule = taille du pinceau
            if delta == 0:
                event.accept()
                return
            steps = 1 if delta > 0 else -1
            self._brush_px = max(1, min(300, self._brush_px + steps))
            self.update()
            self.brush_size_changed.emit(self._brush_px)
        event.accept()

    def keyPressEvent(self, event):
        """V = peek (voir dessous) ; Suppr/Retour arrière = effacer la zone sélectionnée.

        Les autres touches (dont Ctrl+Z/Y) sont relayées au parent.
        """
        if event.modifiers() == Qt.KeyboardModifier.NoModifier:
            if event.key() == Qt.Key.Key_V:
                self.toggle_peek()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                if self.delete_selected_component():
                    event.accept()
                    return
        super().keyPressEvent(event)

    # ── Rendu ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_rgb_pixmap(rgb: np.ndarray) -> QPixmap:
        h, w = rgb.shape[:2]
        # Forcer une copie contigue pour QImage
        data = np.ascontiguousarray(rgb)
        qimg = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _make_overlay_pixmap(self) -> QPixmap:
        """Pixmap RGBA : rouge (zones masquées), jaune (zone sélectionnée).

        En mode « peek », l'overlay est entièrement transparent → on voit la photo.
        """
        ih, iw = self._mask.shape
        rgba = np.zeros((ih, iw, 4), dtype=np.uint8)
        if not self._peek:
            rgba[self._mask > 127] = (255, 60, 60, 160)          # masque (gardé) en rouge
            if self._review_labels is not None:
                # Revue VLM : violet = décor (à retirer), vert = défaut (gardé)
                if self._review_defect is not None:
                    rgba[self._review_defect] = (60, 210, 90, 180)
                if self._review_scene is not None:
                    rgba[self._review_scene] = (190, 60, 235, 190)
            elif self._selected_mask is not None:
                rgba[self._selected_mask > 0] = (255, 220, 40, 210)  # sélection en jaune
        data = np.ascontiguousarray(rgba)
        qimg = QImage(data.tobytes(), iw, ih, iw * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg)

    def _get_composite(self) -> QPixmap:
        """Retourne le pixmap composite (image + overlay), recalcule si necessaire."""
        if self._dirty or self._composite_cache is None:
            iw, ih = self._display_w, self._display_h
            px = QPixmap(iw, ih)
            p  = QPainter(px)
            p.drawPixmap(0, 0, self._base_pixmap)
            overlay = self._make_overlay_pixmap()
            if overlay.width() != iw or overlay.height() != ih:
                overlay = overlay.scaled(
                    iw, ih,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            p.drawPixmap(0, 0, overlay)
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
