"""ZoomGraphicsView — smooth, synchronized image viewer.

Viewport state is stored in normalized image coordinates:
  cx, cy  ∈ [0, 1]  center of the visible area (0.5, 0.5 = image center)
  zoom    ∈ [MIN, MAX]  1.0 = fit-to-view

This means that when two viewers share the same (cx, cy, zoom), they
display the same *relative* region of their image regardless of resolution.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QTransform
from PySide6.QtWidgets import QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ZoomGraphicsView(QGraphicsView):
    viewport_changed = Signal(float, float, float)  # cx, cy, zoom
    double_clicked = Signal()
    ctrl_clicked = Signal()

    ZOOM_STEP = 1.15
    MIN_ZOOM = 0.05
    MAX_ZOOM = 64.0

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.setScene(self._scene)

        # Normalized viewport state
        self._cx: float = 0.5
        self._cy: float = 0.5
        self._zoom: float = 1.0

        self._img_w: int = 0
        self._img_h: int = 0

        self._sync_guard: bool = False
        self._pan_origin = None

        # Appearance
        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.Antialiasing
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QBrush(QColor(22, 22, 22)))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pixmap(self, pixmap):
        self._item.setPixmap(pixmap)
        self._img_w = pixmap.width()
        self._img_h = pixmap.height()
        self._scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self._cx = 0.5
        self._cy = 0.5
        self._zoom = 1.0
        self._apply_transform()

    def clear_pixmap(self):
        from PySide6.QtGui import QPixmap
        self._item.setPixmap(QPixmap())
        self._img_w = 0
        self._img_h = 0

    def fit_image(self):
        """Reset to fit-to-view."""
        self._cx = 0.5
        self._cy = 0.5
        self._zoom = 1.0
        self._apply_transform()
        self._emit_viewport()

    def zoom_to_actual(self):
        """1 image pixel = 1 screen pixel, centered."""
        if self._img_w <= 0:
            return
        fit = self._fit_scale()
        if fit > 0:
            self._zoom = 1.0 / fit
        self._apply_transform()
        self._emit_viewport()

    def adjust_zoom(self, direction: int):
        """Zoom in (direction > 0) or out (< 0), centered on image center."""
        factor = self.ZOOM_STEP if direction > 0 else 1.0 / self.ZOOM_STEP
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if new_zoom != self._zoom:
            self._zoom = new_zoom
            self._apply_transform()
            self._emit_viewport()

    def apply_viewport(self, cx: float, cy: float, zoom: float):
        """Receive synchronized viewport from another viewer."""
        if self._sync_guard or self._img_w <= 0:
            return
        self._sync_guard = True
        self._cx = cx
        self._cy = cy
        self._zoom = zoom
        self._apply_transform()
        self._sync_guard = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_scale(self) -> float:
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw <= 0 or vh <= 0 or self._img_w <= 0 or self._img_h <= 0:
            return 1.0
        return min(vw / self._img_w, vh / self._img_h)

    def _apply_transform(self):
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw <= 0 or vh <= 0 or self._img_w <= 0 or self._img_h <= 0:
            return
        scale = self._fit_scale() * self._zoom
        self.setTransform(QTransform().scale(scale, scale))
        self.centerOn(self._cx * self._img_w, self._cy * self._img_h)

    def _emit_viewport(self):
        if not self._sync_guard:
            self.viewport_changed.emit(self._cx, self._cy, self._zoom)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def wheelEvent(self, event):
        if self._img_w <= 0:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return

        pos = event.position()
        vx, vy = pos.x(), pos.y()
        vw = self.viewport().width()
        vh = self.viewport().height()

        fit = self._fit_scale()
        actual = fit * self._zoom

        # Cursor position in normalized image space
        cursor_img_x = self._cx + (vx - vw / 2) / (actual * self._img_w)
        cursor_img_y = self._cy + (vy - vh / 2) / (actual * self._img_h)

        factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        new_actual = fit * new_zoom
        # Adjust center to keep cursor_img_x,y fixed under the cursor
        new_cx = cursor_img_x - (vx - vw / 2) / (new_actual * self._img_w)
        new_cy = cursor_img_y - (vy - vh / 2) / (new_actual * self._img_h)

        self._zoom = new_zoom
        self._cx = max(-0.5, min(1.5, new_cx))
        self._cy = max(-0.5, min(1.5, new_cy))

        self._apply_transform()
        self._emit_viewport()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.ctrl_clicked.emit()
            else:
                self._pan_origin = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._pan_origin is not None and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            if self._img_w <= 0:
                return
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()

            fit = self._fit_scale()
            actual = fit * self._zoom
            dx = -delta.x() / (actual * self._img_w)
            dy = -delta.y() / (actual * self._img_h)

            self._cx = max(-0.5, min(1.5, self._cx + dx))
            self._cy = max(-0.5, min(1.5, self._cy + dy))

            self._apply_transform()
            self._emit_viewport()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._pan_origin is not None:
            self._pan_origin = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.double_clicked.emit()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_transform()
