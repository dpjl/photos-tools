"""ui/image_view.py — SyncedImageView avec synchronisation tenant compte du ratio d'aspect.

Principe de synchronisation :
    zoom_ratio = zoom_absolu / fit_zoom  (dimensionless)
    cx_rel     = scene_cx / img_width    (normalisé 0-1)
    cy_rel     = scene_cy / img_height   (normalisé 0-1)

Chaque vue recompute ses propres valeurs absolues depuis ces grandeurs normalisées,
ce qui permet de comparer des images de tailles différentes (ex. avant/après upscale).
"""

from __future__ import annotations
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QTransform, QMouseEvent, QPainter, QColor, QFont


def ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convertit un ndarray BGR uint8 en QPixmap."""
    if img is None:
        return QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class SyncedImageView(QGraphicsView):
    """Vue d'image avec zoom/déplacement synchronisés (ratio d'aspect aware)."""

    # Emis quand l'utilisateur modifie la vue (pour notifier les peers)
    view_changed = pyqtSignal(float, float, float)  # zoom_ratio, cx_rel, cy_rel

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._peers: list["SyncedImageView"] = []
        self._syncing = False

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: #1a1a2e; border: none;")

        self._drag_start: Optional[QPointF] = None
        self._preview_mode: bool = False

    # ── Preview banner ───────────────────────────────────────────────────────

    def set_preview_mode(self, enabled: bool) -> None:
        """Active/désactive le bandeau 'APERÇU' superposé à l'image."""
        if enabled != self._preview_mode:
            self._preview_mode = enabled
            self.viewport().update()

    def drawForeground(self, painter: QPainter, rect):
        super().drawForeground(painter, rect)
        if not self._preview_mode:
            return
        vp = self.viewport().rect()
        text = "APERÇU"
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.save()
        painter.resetTransform()
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        margin = 6
        pad    = 6
        bw, bh = tw + pad * 2, th + pad
        bx = vp.right()  - bw - margin
        by = vp.bottom() - bh - margin
        painter.setOpacity(0.82)
        painter.setBrush(QColor("#f39c12"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(bx, by, bw, bh, 4, 4)
        painter.setOpacity(1.0)
        painter.setPen(QColor("#141414"))
        painter.drawText(bx + pad, by + th - 2, text)
        painter.restore()

    # ── Image ────────────────────────────────────────────────────────────────

    def set_image(self, img: Optional[np.ndarray], preserve_zoom: bool = False):
        """Affiche img.

        Si preserve_zoom=True et que la vue affichait déjà une image avec un
        zoom non-trivial, l'état zoom/pan est conservé pour la nouvelle image.
        """
        restore: Optional[tuple[float, float, float]] = None
        if preserve_zoom and self._pix_item is not None:
            fz = self._fit_zoom()
            cz = self._current_zoom()
            if fz > 0 and abs(cz / fz - 1.0) > 0.05:   # vraiment zoomé
                sc = self.mapToScene(self.viewport().rect().center())
                iw = self._pix_item.pixmap().width()
                ih = self._pix_item.pixmap().height()
                restore = (
                    cz / fz,
                    sc.x() / iw if iw > 0 else 0.5,
                    sc.y() / ih if ih > 0 else 0.5,
                )

        self._scene.clear()
        self._pix_item = None
        if img is None:
            return
        pix = ndarray_to_qpixmap(img)
        self._pix_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))

        if restore is not None:
            self._apply_sync(*restore)
        else:
            self.fit_in_view()

    def set_pixmap(self, pix: QPixmap):
        self._scene.clear()
        self._pix_item = None
        if pix.isNull():
            return
        self._pix_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.fit_in_view()

    def has_image(self) -> bool:
        return self._pix_item is not None

    # ── Peers (synchronisation) ──────────────────────────────────────────────

    def add_peer(self, other: "SyncedImageView"):
        if other not in self._peers:
            self._peers.append(other)
        if self not in other._peers:
            other._peers.append(self)

    def remove_peer(self, other: "SyncedImageView"):
        if other in self._peers:
            self._peers.remove(other)
        if self in other._peers:
            other._peers.remove(self)

    # ── Fit / zoom ───────────────────────────────────────────────────────────

    def fit_in_view(self):
        if self._pix_item is None:
            return
        rect = self._pix_item.boundingRect()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _fit_zoom(self) -> float:
        """Zoom absolu pour que l'image tienne dans la vue."""
        if self._pix_item is None:
            return 1.0
        pw = self.viewport().width()
        ph = self.viewport().height()
        iw = self._pix_item.pixmap().width()
        ih = self._pix_item.pixmap().height()
        if iw <= 0 or ih <= 0:
            return 1.0
        return min(pw / iw, ph / ih)

    def _current_zoom(self) -> float:
        return self.transform().m11()

    # ── Synchronisation ──────────────────────────────────────────────────────

    def _broadcast(self):
        """Calcule (zoom_ratio, cx_rel, cy_rel) et notifie les peers."""
        if self._syncing or self._pix_item is None:
            return
        fz = self._fit_zoom()
        if fz <= 0:
            return
        zoom_ratio = self._current_zoom() / fz
        sc = self.mapToScene(self.viewport().rect().center())
        iw = self._pix_item.pixmap().width()
        ih = self._pix_item.pixmap().height()
        cx_rel = sc.x() / iw if iw > 0 else 0.5
        cy_rel = sc.y() / ih if ih > 0 else 0.5
        for peer in self._peers:
            peer._apply_sync(zoom_ratio, cx_rel, cy_rel)

    def _apply_sync(self, zoom_ratio: float, cx_rel: float, cy_rel: float):
        """Applique un état de vue normalisé (reçu d'un peer)."""
        if self._syncing or self._pix_item is None:
            return
        self._syncing = True
        fz = self._fit_zoom()
        new_zoom = max(zoom_ratio * fz, 0.01)
        iw = self._pix_item.pixmap().width()
        ih = self._pix_item.pixmap().height()
        scene_cx = cx_rel * iw
        scene_cy = cy_rel * ih
        self.setTransform(QTransform().scale(new_zoom, new_zoom))
        self.centerOn(QPointF(scene_cx, scene_cy))
        self._syncing = False

    # ── Événements ──────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        if self._pix_item is None:
            return
        # Zoom centré sur le curseur
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        cur = self._current_zoom()
        fz  = self._fit_zoom()
        new = max(cur * factor, fz * 0.5)  # min = 50% du fit
        new = min(new, fz * 40)            # max = 40×

        # Ancre sur le curseur
        old_pos = self.mapToScene(event.position().toPoint())
        self.setTransform(QTransform().scale(new, new))
        new_pos = self.mapToScene(event.position().toPoint())
        delta   = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        self._broadcast()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._broadcast()

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)
        self._drag_start = None
        self._broadcast()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-fit si l'image est présente et qu'on était en mode fit
        if self._pix_item is not None:
            cur = self._current_zoom()
            fz  = self._fit_zoom()
            # Si on était proche du fit-zoom (±10 %), rester en fit
            if fz > 0 and abs(cur - fz) / fz < 0.1:
                self.fit_in_view()
