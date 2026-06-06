"""ui/crop_editor.py — Editeur de recadrage manuel."""

from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
    QWidget, QFrame, QSizePolicy,
)


class CropCanvas(QWidget):
    """Canvas image + rectangle de recadrage normalise."""

    crop_changed = pyqtSignal(object)  # tuple[float, float, float, float] | None

    def __init__(
        self,
        image_bgr: np.ndarray,
        initial_rect: tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._img_w = 1
        self._img_h = 1
        self._pixmap = QPixmap()
        self._rect = _normalize_rect(initial_rect)
        self._drag_start_img: tuple[int, int] | None = None
        self._drawing = False
        self._panning = False
        self._pan_last: QPoint | None = None

        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.reset_image(image_bgr, initial_rect)

    def get_crop_rect(self):
        return self._rect

    def set_crop_rect(self, rect) -> None:
        self._rect = _normalize_rect(rect)
        self.update()

    def clear_crop(self) -> None:
        if self._rect is None:
            return
        self._rect = None
        self.crop_changed.emit(None)
        self.update()

    def reset_image(self, image_bgr: np.ndarray, initial_rect=None) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._img_w = max(1, w)
        self._img_h = max(1, h)
        self._pixmap = self._make_rgb_pixmap(rgb)
        self._rect = _normalize_rect(initial_rect)
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drawing = False
        self._panning = False
        self._pan_last = None
        self.update()

    def set_display_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._pixmap = self._make_rgb_pixmap(rgb)
        self.update()

    def get_zoom_state(self) -> tuple:
        cw, ch = self.width(), self.height()
        if cw == 0 or ch == 0 or self._img_w == 0 or self._img_h == 0:
            return (1.0, 0.5, 0.5)
        fit = min(cw / self._img_w, ch / self._img_h)
        dw = self._img_w * fit * self._zoom
        dh = self._img_h * fit * self._zoom
        cx_rel = 0.5 - self._pan_x / dw if dw else 0.5
        cy_rel = 0.5 - self._pan_y / dh if dh else 0.5
        return (self._zoom, cx_rel, cy_rel)

    def apply_zoom_state(self, zoom_ratio: float, cx_rel: float, cy_rel: float) -> None:
        self._zoom = max(0.5, min(16.0, zoom_ratio))
        cw, ch = self.width(), self.height()
        if cw == 0 or ch == 0 or self._img_w == 0 or self._img_h == 0:
            return
        fit = min(cw / self._img_w, ch / self._img_h)
        dw = self._img_w * fit * self._zoom
        dh = self._img_h * fit * self._zoom
        self._pan_x = (0.5 - cx_rel) * dw
        self._pan_y = (0.5 - cy_rel) * dh
        self.update()

    def _display_rect(self) -> QRect:
        cw, ch = self.width(), self.height()
        if cw == 0 or ch == 0:
            return QRect(0, 0, 0, 0)
        fit_scale = min(cw / self._img_w, ch / self._img_h)
        scale = fit_scale * self._zoom
        dw = int(self._img_w * scale)
        dh = int(self._img_h * scale)
        base_x = (cw - dw) // 2
        base_y = (ch - dh) // 2
        return QRect(base_x + int(self._pan_x), base_y + int(self._pan_y), dw, dh)

    def _canvas_to_img(self, pt: QPoint) -> tuple[int, int] | None:
        r = self._display_rect()
        if r.width() == 0 or r.height() == 0:
            return None
        ix = int((pt.x() - r.x()) * self._img_w / r.width())
        iy = int((pt.y() - r.y()) * self._img_h / r.height())
        ix = int(np.clip(ix, 0, self._img_w - 1))
        iy = int(np.clip(iy, 0, self._img_h - 1))
        return ix, iy

    def _img_rect_to_canvas(self, rect) -> QRect:
        r = self._display_rect()
        x0 = int(r.x() + rect[0] * r.width())
        y0 = int(r.y() + rect[1] * r.height())
        x1 = int(r.x() + rect[2] * r.width())
        y1 = int(r.y() + rect[3] * r.height())
        return QRect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _zoom_at(self, canvas_pt: QPoint, factor: float) -> None:
        r = self._display_rect()
        if r.width() == 0:
            return
        fx = (canvas_pt.x() - r.x()) / r.width()
        fy = (canvas_pt.y() - r.y()) / r.height()
        self._zoom = max(0.5, min(16.0, self._zoom * factor))
        cw, ch = self.width(), self.height()
        fit_scale = min(cw / self._img_w, ch / self._img_h) if cw and ch else 1.0
        new_scale = fit_scale * self._zoom
        dw = int(self._img_w * new_scale)
        dh = int(self._img_h * new_scale)
        self._pan_x = canvas_pt.x() - fx * dw - (cw - dw) // 2
        self._pan_y = canvas_pt.y() - fy * dh - (ch - dh) // 2
        self.update()

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            coords = self._canvas_to_img(pos)
            if coords is None:
                return
            self._drawing = True
            self._drag_start_img = coords
            self._rect = _rect_from_pixels(coords, coords, self._img_w, self._img_h)
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._panning and self._pan_last is not None:
            self._pan_x += pos.x() - self._pan_last.x()
            self._pan_y += pos.y() - self._pan_last.y()
            self._pan_last = pos
            self.update()
            event.accept()
            return
        if self._drawing and self._drag_start_img is not None:
            coords = self._canvas_to_img(pos)
            if coords is not None:
                self._rect = _rect_from_pixels(
                    self._drag_start_img, coords, self._img_w, self._img_h
                )
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            self._drag_start_img = None
            self._rect = _normalize_rect(self._rect)
            self.crop_changed.emit(self._rect)
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom_at(event.position().toPoint(), factor)
        event.accept()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(22, 22, 32))
        display = self._display_rect()
        p.drawPixmap(display, self._pixmap)

        if self._rect is not None:
            crop = self._img_rect_to_canvas(self._rect)
            outside = QBrush(QColor(0, 0, 0, 110))
            p.fillRect(display.x(), display.y(), display.width(), crop.y() - display.y(), outside)
            p.fillRect(display.x(), crop.bottom(), display.width(), display.bottom() - crop.bottom(), outside)
            p.fillRect(display.x(), crop.y(), crop.x() - display.x(), crop.height(), outside)
            p.fillRect(crop.right(), crop.y(), display.right() - crop.right(), crop.height(), outside)
            p.setPen(QPen(QColor(255, 210, 80), 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(crop)
        p.end()

    @staticmethod
    def _make_rgb_pixmap(rgb: np.ndarray) -> QPixmap:
        h, w = rgb.shape[:2]
        data = np.ascontiguousarray(rgb)
        qimg = QImage(data.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)


class CropCanvasPanel(QWidget):
    """Canvas + barre laterale de recadrage, utilisable en batch ou dialogue."""

    accepted = pyqtSignal()
    rejected = pyqtSignal()
    crop_changed = pyqtSignal(object)

    def __init__(
        self,
        image_bgr: np.ndarray,
        initial_rect=None,
        parent=None,
        show_ok_cancel: bool = False,
        sidebar_width: int = 215,
    ) -> None:
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._canvas = CropCanvas(image_bgr, initial_rect)
        self._canvas.crop_changed.connect(self.crop_changed)
        root.addWidget(self._canvas, stretch=1)

        sidebar = QWidget()
        sidebar.setFixedWidth(sidebar_width)
        sidebar.setStyleSheet("background: #1a1a2e;")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(12, 14, 12, 14)
        sl.setSpacing(8)

        title = QLabel("Recadrage")
        title.setStyleSheet("color:#ddd; font-size:13px; font-weight:700;")
        sl.addWidget(title)
        sl.addWidget(self._hline())

        tips = QLabel(
            "Clic gauche + glisser\n"
            "dessine le recadrage\n\n"
            "Molette        -> zoom\n"
            "Clic milieu    -> déplacer"
        )
        tips.setStyleSheet("color:#7a9ab0; font-size:10px;")
        tips.setWordWrap(True)
        sl.addWidget(tips)
        sl.addWidget(self._hline())

        clear_btn = QPushButton("Effacer le recadrage")
        clear_btn.clicked.connect(self._canvas.clear_crop)
        self._style(clear_btn)
        sl.addWidget(clear_btn)
        sl.addStretch()

        if show_ok_cancel:
            sl.addWidget(self._hline())
            ok_btn = QPushButton("Valider le recadrage")
            ok_btn.clicked.connect(self.accepted.emit)
            self._style(ok_btn, accent=True)
            sl.addWidget(ok_btn)
            cancel_btn = QPushButton("Annuler")
            cancel_btn.clicked.connect(self.rejected.emit)
            self._style(cancel_btn)
            sl.addWidget(cancel_btn)

        root.addWidget(sidebar)

    def get_crop_rect(self):
        return self._canvas.get_crop_rect()

    def set_image(self, image_bgr: np.ndarray, initial_rect=None) -> None:
        self._canvas.reset_image(image_bgr, initial_rect)

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background:#252545; margin:2px 0;")
        return line

    @staticmethod
    def _style(btn: QPushButton, accent: bool = False) -> None:
        if accent:
            btn.setStyleSheet(
                "QPushButton { background:#1e3a52; color:#b8e0f7; border-radius:4px;"
                " padding:6px 8px; font-size:11px; }"
                "QPushButton:hover { background:#2a5577; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
                " padding:6px 8px; font-size:11px; }"
                "QPushButton:hover { background:#2a2a50; color:#ccc; }"
            )


class CropEditorDialog(QDialog):
    """Dialogue de recadrage manuel."""

    def __init__(self, parent, image_bgr: np.ndarray, initial_rect=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recadrage manuel")
        self.setModal(True)
        self.setMinimumSize(900, 600)
        self.resize(1100, 750)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._panel = CropCanvasPanel(image_bgr, initial_rect, show_ok_cancel=True)
        self._panel.accepted.connect(self.accept)
        self._panel.rejected.connect(self.reject)
        root.addWidget(self._panel)

        QShortcut(QKeySequence("Esc"), self, self.reject)

    def get_crop_rect(self):
        return self._panel.get_crop_rect()


def _normalize_rect(rect):
    if rect is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in rect]
    left, right = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    top, bottom = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if right - left < 0.002 or bottom - top < 0.002:
        return None
    return (left, top, right, bottom)


def _rect_from_pixels(p0, p1, width: int, height: int):
    x0, y0 = p0
    x1, y1 = p1
    return _normalize_rect((
        x0 / max(1, width),
        y0 / max(1, height),
        (x1 + 1) / max(1, width),
        (y1 + 1) / max(1, height),
    ))
