#!/usr/bin/env python3
"""
restore_gui.py — Interface graphique PyQt6 pour la restauration photo ancienne.

Usage :
    python restore_gui.py [image_path]

Layout :
    Gauche   : panneau de contrôle (étapes, paramètres, boutons)
    Centre   : grande vue principale (simple ou split A/B synchronisé)
    Bas      : strip de vignettes — clic gauche = vue A, clic droit = vue B
               (split s'active automatiquement quand B est sélectionné)

Navigation :
    Molette          : zoom centré sur la souris
    Glisser (gauche) : panoramique
    Double-clic      : ajuster à l'écran
    Touche F         : ajuster à l'écran
    Touche 1         : zoom 100 %
"""

import os
import sys
import traceback

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QScrollArea,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QFileDialog,
    QFrame, QMessageBox, QToolButton, QSizePolicy, QProgressBar,
    QGraphicsView, QGraphicsScene,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer, QSize
from PyQt6.QtGui import (
    QImage, QPixmap, QTransform, QColor, QPainter, QAction,
    QKeySequence, QCursor, QIcon,
)

# ---------------------------------------------------------------------------
# Import du core de traitement
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from restore_core import (
        stretch_histogram, apply_clahe, boost_saturation,
        apply_scunet, apply_gfpgan, blend_faces_from_gfpgan,
        correct_color_rembg, make_grid,
        GFPGAN_MODEL_PATH, SCUNET_MODELS_DIR,
    )
    _CORE_AVAILABLE = True
    _CORE_ERROR = ""
except ImportError as _err:
    _CORE_AVAILABLE = False
    _CORE_ERROR = str(_err)

# ---------------------------------------------------------------------------
# Définition data-driven des étapes
# ---------------------------------------------------------------------------
STEP_DEFS = [
    {
        "id":         "color",
        "name":       "1 · Correction couleur",
        "short_name": "Couleur",
        "desc":       "Étirement histogramme · CLAHE · Saturation",
        "slow":       False,
        "params": [
            {"key": "low_pct",    "label": "Pct. bas histogramme",
             "type": "float", "default": 1.0,  "min": 0.0,  "max": 5.0,   "step": 0.1},
            {"key": "high_pct",   "label": "Pct. haut histogramme",
             "type": "float", "default": 99.0, "min": 94.0, "max": 100.0, "step": 0.1},
            {"key": "roi_inset",  "label": "Marge ROI",
             "type": "float", "default": 0.06, "min": 0.0,  "max": 0.2,   "step": 0.01},
            {"key": "clip_limit", "label": "CLAHE — force contraste",
             "type": "float", "default": 2.5,  "min": 0.5,  "max": 10.0,  "step": 0.1},
            {"key": "sat_factor", "label": "Saturation (×)",
             "type": "float", "default": 1.35, "min": 0.5,  "max": 2.5,   "step": 0.05},
        ],
    },
    {
        "id":         "gfpgan",
        "name":       "2 · GFPGAN — visages",
        "short_name": "GFPGAN",
        "desc":       "Restauration IA des visages  ·  ~30 s CPU",
        "slow":       True,
        "params": [
            {"key": "upscale", "label": "Facteur d'upscale",
             "type": "int", "default": 1, "min": 1, "max": 2, "step": 1,
             "tooltip": "upscale > 1 multiplie considérablement la durée de traitement"},
            {"key": "weight",  "label": "Force de restauration",
             "type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
             "tooltip": "0.0 = visage original intact\n"
                        "0.5 = restauration modérée (défaut GFPGAN)\n"
                        "1.0 = restauration maximale (peut donner un effet plastique)"},
        ],
    },
    {
        "id":         "scunet",
        "name":       "3 · SCUNet — débruitage",
        "short_name": "SCUNet",
        "desc":       "Débruitage IA aveugle (grain argentique)  ·  ~50 s CPU",
        "slow":       True,
        "params": [
            {"key": "mode",         "label": "Modèle",
             "type": "choice", "default": "gan", "choices": ["gan", "psnr"],
             "tooltip": "gan = perceptuel, texture naturelle (recommandé)\n"
                        "psnr = conservateur, zéro hallucination"},
            {"key": "blend_expand", "label": "Expansion zones visages",
             "type": "float", "default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05,
             "tooltip": "Taille relative de la zone de recomposition GFPGAN\n"
                        "autour de chaque visage détecté"},
        ],
    },
    {
        "id":         "rembg",
        "name":       "4 · Cast argentique",
        "short_name": "Cast",
        "desc":       "Correction cast rouge / jaune  ·  ~15 s CPU",
        "slow":       True,
        "params": [
            {"key": "strength_a", "label": "Force rouge (a*)",
             "type": "float", "default": 1.3,  "min": 0.0,  "max": 3.0,  "step": 0.05},
            {"key": "strength_b", "label": "Force jaune (b*)",
             "type": "float", "default": 0.4,  "min": 0.0,  "max": 2.0,  "step": 0.05},
            {"key": "sigma_fill", "label": "Diffusion (σ px)",
             "type": "int",   "default": 120,  "min": 20,   "max": 300,  "step": 10},
            {"key": "ref_x1",     "label": "Zone réf. début",
             "type": "float", "default": 0.03, "min": 0.0,  "max": 0.2,  "step": 0.01},
            {"key": "ref_x2",     "label": "Zone réf. fin",
             "type": "float", "default": 0.20, "min": 0.05, "max": 0.5,  "step": 0.01},
        ],
    },
]

STEP_IDS   = [s["id"]         for s in STEP_DEFS]
STEP_NAMES = {s["id"]: s["name"]       for s in STEP_DEFS}
STEP_SHORT = {s["id"]: s["short_name"] for s in STEP_DEFS}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    """Convertit une image OpenCV BGR en QPixmap."""
    h, w = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())  # .copy() détache du buffer numpy


# ===========================================================================
# SyncedImageView — QGraphicsView avec zoom/pan synchronisable
# ===========================================================================

class SyncedImageView(QGraphicsView):
    """Visionneuse d'image avec zoom molette, pan par glisser, et sync entre vues."""

    # Émis à chaque changement de viewport (zoom ou pan)
    viewport_changed = pyqtSignal(object, object)   # (QTransform, QPointF center scene)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene    = QGraphicsScene(self)
        self._pix_item = None      # QGraphicsPixmapItem courant
        self._peers: list["SyncedImageView"] = []
        self._syncing  = False
        self._drag_start  = None  # QPoint au début du drag
        self._drag_origin = None  # Centre scène au début du drag

        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#1c1c1c"))
        self.setStyleSheet("border: none;")
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Label flottant avec le nom de l'étape
        self._step_label = QLabel(self)
        self._step_label.setStyleSheet(
            "background: rgba(0,0,0,170); color: #eee; "
            "padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;"
        )
        self._step_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._step_label.hide()

        # Placeholder "Aucune image" centré
        self._placeholder = QLabel("Aucune image\n\nOuvrir une image pour commencer", self)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #444; font-size: 13px;")
        self._placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._placeholder.show()

    # ── Redimensionnement ──────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Recentrer le placeholder
        self._placeholder.setGeometry(self.viewport().rect())

    # ── API publique ───────────────────────────────────────────────────────

    def set_image(self, bgr: np.ndarray | None):
        """Charge une nouvelle image (BGR numpy) dans la scène."""
        self._scene.clear()
        self._pix_item = None
        if bgr is None:
            self._placeholder.show()
            return
        self._placeholder.hide()
        pixmap = bgr_to_pixmap(bgr)
        self._pix_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

    def fit_image(self):
        """Ajuste le zoom pour afficher l'image en entier."""
        if self._pix_item is None:
            return
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._broadcast()

    def set_step_label(self, text: str):
        if text:
            self._step_label.setText(text)
            self._step_label.adjustSize()
            self._step_label.move(8, 8)
            self._step_label.show()
        else:
            self._step_label.hide()

    def link_peer(self, other: "SyncedImageView"):
        """Lie cette vue à une autre pour synchroniser zoom et pan."""
        if other not in self._peers:
            self._peers.append(other)
        if self not in other._peers:
            other._peers.append(self)

    def unlink_all(self):
        """Retire tous les liens de synchronisation."""
        for peer in list(self._peers):
            if self in peer._peers:
                peer._peers.remove(self)
        self._peers.clear()

    # ── Événements souris ─────────────────────────────────────────────────

    def wheelEvent(self, event):
        if self._pix_item is None:
            return
        delta  = event.angleDelta().y()
        factor = 1.18 ** (delta / 120.0)
        # Limites de zoom : 2 % – 3200 %
        current = self.transform().m11()
        if (factor < 1.0 and current < 0.03) or (factor > 1.0 and current > 32.0):
            return
        self.scale(factor, factor)
        self._broadcast()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_start  = event.position().toPoint()
            self._drag_origin = self.mapToScene(self.viewport().rect().center())
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and self._pix_item is not None:
            delta  = event.position().toPoint() - self._drag_start
            scale  = self.transform().m11()
            center = QPointF(
                self._drag_origin.x() - delta.x() / scale,
                self._drag_origin.y() - delta.y() / scale,
            )
            self.centerOn(center)
            self._broadcast()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.fit_image()
        super().mouseDoubleClickEvent(event)

    # ── Sync ─────────────────────────────────────────────────────────────

    def _broadcast(self):
        """Envoie le viewport courant à tous les pairs."""
        if self._syncing:
            return
        t = self.transform()
        c = self.mapToScene(self.viewport().rect().center())
        for peer in self._peers:
            peer._apply_sync(t, c)

    def _apply_sync(self, transform: QTransform, center: QPointF):
        """Reçoit un viewport d'un pair et l'applique."""
        if self._syncing or self._pix_item is None:
            return
        self._syncing = True
        self.setTransform(transform)
        self.centerOn(center)
        self._syncing = False


# ===========================================================================
# ThumbnailCard — vignette cliquable dans la strip
# ===========================================================================

THUMB_W, THUMB_H = 160, 106
_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class ThumbnailCard(QFrame):
    """
    Vignette d'une étape dans la strip du bas.
    Clic gauche  → sélectionne comme vue A (panneau gauche / vue unique).
    Clic droit   → sélectionne comme vue B (panneau droit, active le split).
    """
    selected_a = pyqtSignal(str)   # step_id
    selected_b = pyqtSignal(str)   # step_id

    def __init__(self, step_id: str, label: str, parent=None):
        super().__init__(parent)
        self.step_id  = step_id
        self._active_a = False
        self._active_b = False
        self._spinner_idx = 0
        self._state = "empty"

        self.setFixedSize(THUMB_W + 8, THUMB_H + 46)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Clic gauche : vue principale\nClic droit : vue comparaison (split)")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 4)
        lay.setSpacing(2)

        # Zone image
        self._img_lbl = QLabel()
        self._img_lbl.setFixedSize(THUMB_W, THUMB_H)
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet("background: #2a2a2a; border-radius: 3px;")
        lay.addWidget(self._img_lbl)

        # Nom de l'étape
        name_lbl = QLabel(label)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("font-size: 10px; color: #bbb;")
        lay.addWidget(name_lbl)

        # Badges A / B
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(2, 0, 2, 0)
        self._badge_a = self._make_badge("A", "#3a7bd5")
        self._badge_b = self._make_badge("B", "#c0392b")
        badge_row.addWidget(self._badge_a)
        badge_row.addStretch()
        badge_row.addWidget(self._badge_b)
        lay.addLayout(badge_row)

        self._set_style_border()

        # Timer spinner
        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._spin_tick)

    def _make_badge(self, text: str, color: str) -> QLabel:
        b = QLabel(text)
        b.setFixedSize(20, 16)
        b.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b.setStyleSheet(
            f"background:{color}; color:white; border-radius:3px; "
            f"font-size:9px; font-weight:bold;"
        )
        b.hide()
        return b

    # ── États visuels ──────────────────────────────────────────────────────

    def set_image(self, bgr: np.ndarray):
        """Affiche une miniature calculée à partir d'une image BGR."""
        self._timer.stop()
        self._state = "done"
        thumb = cv2.resize(bgr, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        self._img_lbl.setPixmap(bgr_to_pixmap(thumb))
        self._img_lbl.setStyleSheet("background: #2a2a2a; border-radius: 3px;")
        self._set_style_border()

    def set_running(self):
        self._state = "running"
        self._spinner_idx = 0
        self._img_lbl.setStyleSheet("background: #2a2a2a; border-radius: 3px; "
                                    "color: #aaa; font-size: 20px;")
        self._img_lbl.setPixmap(QPixmap())
        self._img_lbl.setText(_SPINNER[0])
        self._timer.start()
        self._set_style_border()

    def set_empty(self):
        self._timer.stop()
        self._state = "empty"
        self._img_lbl.setPixmap(QPixmap())
        self._img_lbl.setText("—")
        self._img_lbl.setStyleSheet("background: #2a2a2a; border-radius: 3px; "
                                    "color: #444; font-size: 22px;")
        self._set_style_border()

    def set_error(self):
        self._timer.stop()
        self._state = "error"
        self._img_lbl.setPixmap(QPixmap())
        self._img_lbl.setText("✕")
        self._img_lbl.setStyleSheet("background: #2a2a2a; border-radius: 3px; "
                                    "color: #c0392b; font-size: 22px;")
        self._set_style_border()

    def _spin_tick(self):
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
        self._img_lbl.setText(_SPINNER[self._spinner_idx])

    # ── Sélection A / B ────────────────────────────────────────────────────

    def set_selected_a(self, v: bool):
        self._active_a = v
        self._badge_a.setVisible(v)
        self._set_style_border()

    def set_selected_b(self, v: bool):
        self._active_b = v
        self._badge_b.setVisible(v)
        self._set_style_border()

    def _set_style_border(self):
        if self._active_a and self._active_b:
            border = "2px solid #9b59b6"
        elif self._active_a:
            border = "2px solid #3a7bd5"
        elif self._active_b:
            border = "2px solid #c0392b"
        elif self._state == "running":
            border = "1px solid #666"
        else:
            border = "1px solid #444"
        self.setStyleSheet(f"QFrame {{ border: {border}; border-radius: 5px; background: #1e1e1e; }}")

    # ── Souris ─────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_a.emit(self.step_id)
        elif event.button() == Qt.MouseButton.RightButton:
            self.selected_b.emit(self.step_id)
        super().mousePressEvent(event)


# ===========================================================================
# ParamRow — slider + spinbox pour un paramètre
# ===========================================================================

class ParamRow(QWidget):
    """Ligne de paramètre : label + slider + spinbox (ou combobox)."""

    value_changed = pyqtSignal(str, object)   # (key, value)

    def __init__(self, pdef: dict, parent=None):
        super().__init__(parent)
        self._key     = pdef["key"]
        self._ptype   = pdef["type"]
        self._default = pdef["default"]
        self._blocked = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 1, 0, 3)
        outer.setSpacing(2)

        # Ligne du label
        lbl = QLabel(pdef["label"])
        lbl.setStyleSheet("font-size: 10px; color: #aaa;")
        if pdef.get("tooltip"):
            lbl.setToolTip(pdef["tooltip"])
        outer.addWidget(lbl)

        if self._ptype == "choice":
            self._combo = QComboBox()
            self._combo.addItems(pdef["choices"])
            idx = pdef["choices"].index(pdef["default"]) if pdef["default"] in pdef["choices"] else 0
            self._combo.setCurrentIndex(idx)
            self._combo.currentTextChanged.connect(self._on_combo)
            outer.addWidget(self._combo)
        else:
            row = QHBoxLayout()
            row.setSpacing(4)
            row.setContentsMargins(0, 0, 0, 0)

            step  = pdef.get("step", 1)
            decs  = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
            scale = 10 ** decs
            self._scale = scale
            self._decs  = decs

            self._slider = QSlider(Qt.Orientation.Horizontal)
            self._slider.setMinimum(int(pdef["min"] * scale))
            self._slider.setMaximum(int(pdef["max"] * scale))
            self._slider.setValue(int(pdef["default"] * scale))
            self._slider.setSingleStep(int(step * scale))

            if self._ptype == "float":
                self._spin = QDoubleSpinBox()
                self._spin.setDecimals(decs)
                self._spin.setMinimum(pdef["min"])
                self._spin.setMaximum(pdef["max"])
                self._spin.setSingleStep(step)
                self._spin.setValue(pdef["default"])
                self._spin.setFixedWidth(68)
            else:  # int
                self._spin = QSpinBox()
                self._spin.setMinimum(int(pdef["min"]))
                self._spin.setMaximum(int(pdef["max"]))
                self._spin.setSingleStep(int(step))
                self._spin.setValue(int(pdef["default"]))
                self._spin.setFixedWidth(58)

            self._slider.valueChanged.connect(self._on_slider)
            self._spin.valueChanged.connect(self._on_spin)

            row.addWidget(self._slider, 1)
            row.addWidget(self._spin)
            outer.addLayout(row)

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _on_slider(self, v: int):
        if self._blocked:
            return
        self._blocked = True
        val = v / self._scale if self._ptype == "float" else v
        self._spin.setValue(val)
        self.value_changed.emit(self._key, val)
        self._blocked = False

    def _on_spin(self, v):
        if self._blocked:
            return
        self._blocked = True
        self._slider.setValue(int(v * self._scale) if self._ptype == "float" else int(v))
        self.value_changed.emit(self._key, v)
        self._blocked = False

    def _on_combo(self, text: str):
        self.value_changed.emit(self._key, text)

    # ── API ─────────────────────────────────────────────────────────────────

    def get_value(self):
        if self._ptype == "choice":
            return self._combo.currentText()
        return self._spin.value()

    def reset(self):
        if self._ptype == "choice":
            self._combo.setCurrentIndex(0)
        elif self._ptype == "float":
            self._spin.setValue(float(self._default))
        else:
            self._spin.setValue(int(self._default))


# ===========================================================================
# StepPanel — section accordéon pour une étape
# ===========================================================================

_STATUS_CFG = {
    "idle":     ("—",            "#666"),
    "ok":       ("✓ OK",         "#27ae60"),
    "stale":    ("⚠ Obsolète",   "#e67e22"),
    "running":  ("⏳ En cours…", "#3498db"),
    "error":    ("✕ Erreur",     "#e74c3c"),
    "disabled": ("—",            "#444"),
}


class StepPanel(QWidget):
    """Panneau accordéon pour une étape : header cliquable + body paramètres."""

    rerun_requested = pyqtSignal(str)   # step_id
    param_changed   = pyqtSignal(str)   # step_id

    def __init__(self, sdef: dict, parent=None):
        super().__init__(parent)
        self.step_id  = sdef["id"]
        self._expanded = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 6)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet(
            "QWidget { background: #2c2c2c; border-radius: 4px; }"
            "QWidget:hover { background: #333; }"
        )
        hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        hlay = QHBoxLayout(hdr)
        hlay.setContentsMargins(6, 5, 6, 5)

        self._arrow = QToolButton()
        self._arrow.setArrowType(Qt.ArrowType.DownArrow)
        self._arrow.setFixedSize(14, 14)
        self._arrow.setStyleSheet("QToolButton { border: none; color: #888; background: transparent; }")
        self._arrow.clicked.connect(self._toggle)

        self._enable_cb = QCheckBox()
        self._enable_cb.setChecked(True)
        self._enable_cb.stateChanged.connect(self._on_enable_changed)
        self._enable_cb.setToolTip("Activer / désactiver cette étape")

        name_lbl = QLabel(sdef["name"])
        name_lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #ddd;")
        if sdef.get("slow"):
            name_lbl.setToolTip(sdef["desc"])

        self._status_lbl = QLabel("—")
        self._status_lbl.setStyleSheet("font-size: 10px; color: #666;")
        self._status_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        hlay.addWidget(self._arrow)
        hlay.addWidget(self._enable_cb)
        hlay.addWidget(name_lbl, 1)
        hlay.addWidget(self._status_lbl)

        root.addWidget(hdr)
        hdr.mousePressEvent = lambda _: self._toggle()

        # ── Body ────────────────────────────────────────────────────────────
        self._body = QWidget()
        blay = QVBoxLayout(self._body)
        blay.setContentsMargins(8, 4, 4, 4)
        blay.setSpacing(0)

        self._param_rows: dict[str, ParamRow] = {}
        for pdef in sdef.get("params", []):
            row = ParamRow(pdef)
            row.value_changed.connect(
                lambda _k, _v, sid=self.step_id: self.param_changed.emit(sid)
            )
            self._param_rows[pdef["key"]] = row
            blay.addWidget(row)

        # Boutons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 6, 0, 0)
        btn_row.setSpacing(4)

        self._rerun_btn = QPushButton("▶ Recalculer depuis ici")
        self._rerun_btn.setStyleSheet("font-size: 10px; padding: 3px 8px;")
        self._rerun_btn.clicked.connect(lambda: self.rerun_requested.emit(self.step_id))

        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet("font-size: 10px; padding: 3px 8px;")
        reset_btn.clicked.connect(self._reset_params)

        btn_row.addWidget(self._rerun_btn, 1)
        btn_row.addWidget(reset_btn)
        blay.addLayout(btn_row)

        root.addWidget(self._body)

    # ── API ─────────────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {k: row.get_value() for k, row in self._param_rows.items()}

    def is_enabled(self) -> bool:
        return self._enable_cb.isChecked()

    def set_status(self, status: str):
        txt, color = _STATUS_CFG.get(status, _STATUS_CFG["idle"])
        self._status_lbl.setText(txt)
        self._status_lbl.setStyleSheet(f"font-size: 10px; color: {color};")
        self._rerun_btn.setEnabled(status != "running")

    # ── Private ─────────────────────────────────────────────────────────────

    def _toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._arrow.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )

    def _on_enable_changed(self, _state):
        self._body.setEnabled(self._enable_cb.isChecked())

    def _reset_params(self):
        for row in self._param_rows.values():
            row.reset()


# ===========================================================================
# PipelineWorker — traitement en QThread
# ===========================================================================

class PipelineWorker(QThread):
    """Exécute les étapes du pipeline en arrière-plan.

    Paramètres init :
        steps_to_run : list[str]   — IDs des étapes à exécuter, dans l'ordre
        initial_img  : np.ndarray  — image en entrée de la première étape
        all_params   : dict        — {step_id: {key: value}}
        context      : dict        — contexte partagé (ex. face_bboxes de GFPGAN)
    """

    step_started = pyqtSignal(str)                      # step_id
    step_done    = pyqtSignal(str, object, dict)         # step_id, ndarray, extras
    step_failed  = pyqtSignal(str, str)                  # step_id, message
    all_done     = pyqtSignal()

    def __init__(self, steps_to_run, initial_img, all_params, context, parent=None):
        super().__init__(parent)
        self._steps  = steps_to_run
        self._img    = initial_img
        self._params = all_params
        self._ctx    = dict(context)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        img = self._img
        for step_id in self._steps:
            if self._cancel:
                break
            if img is None:
                self.step_failed.emit(step_id, "Pas d'image en entrée (étape précédente échouée)")
                continue

            self.step_started.emit(step_id)
            p = self._params.get(step_id, {})

            try:
                result, extras = self._run_step(step_id, img, p)
            except Exception as exc:
                traceback.print_exc()
                self.step_failed.emit(step_id, str(exc))
                # On ne met PAS img = None : les étapes suivantes utilisent
                # le dernier résultat valide comme entrée
                continue

            self._ctx.update(extras)
            self.step_done.emit(step_id, result, extras)
            img = result

        self.all_done.emit()

    def _run_step(self, step_id: str, img: np.ndarray, p: dict):
        """Exécute une étape et retourne (result, extras)."""
        if step_id == "color":
            result = stretch_histogram(img,
                low_pct=p.get("low_pct", 1.0),
                high_pct=p.get("high_pct", 99.0),
                roi_inset=p.get("roi_inset", 0.06),
            )
            result = apply_clahe(result, clip_limit=p.get("clip_limit", 2.5))
            result = boost_saturation(result, factor=p.get("sat_factor", 1.35))
            return result, {}

        elif step_id == "gfpgan":
            out = apply_gfpgan(img, upscale=p.get("upscale", 1), weight=p.get("weight", 0.5))
            if out is None:
                raise RuntimeError("GFPGAN indisponible (vérifier le modèle / les dépendances)")
            result, face_bboxes = out
            return result, {"face_bboxes": face_bboxes}

        elif step_id == "scunet":
            scunet_out = apply_scunet(img, mode=p.get("mode", "gan"))
            if scunet_out is None:
                raise RuntimeError("SCUNet indisponible (vérifier le modèle / les dépendances)")
            face_bboxes = self._ctx.get("face_bboxes", [])
            if face_bboxes:
                result = blend_faces_from_gfpgan(
                    img, scunet_out, face_bboxes,
                    expand=p.get("blend_expand", 0.4),
                )
            else:
                result = scunet_out
            return result, {}

        elif step_id == "rembg":
            result = correct_color_rembg(
                img,
                strength_a=p.get("strength_a", 1.3),
                strength_b=p.get("strength_b", 0.4),
                sigma_fill=int(p.get("sigma_fill", 120)),
                ref_x1=p.get("ref_x1", 0.03),
                ref_x2=p.get("ref_x2", 0.20),
            )
            return result, {}

        else:
            return img, {}


# ===========================================================================
# MainApp — fenêtre principale
# ===========================================================================

class MainApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Restauration Photo — Pipeline IA")
        self.resize(1460, 900)

        # ── État interne ──────────────────────────────────────────────────
        self._original:      np.ndarray | None = None
        self._step_results:  dict[str, np.ndarray] = {}
        self._step_context:  dict = {}
        self._step_stale:    dict[str, bool] = {s: True for s in STEP_IDS}
        self._worker:        PipelineWorker | None = None
        self._current_a:     str = "original"
        self._current_b:     str | None = None
        self._progress_done: int = 0

        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._apply_theme()
        self._setup_shortcuts()

        if not _CORE_AVAILABLE:
            QMessageBox.critical(
                self, "Erreur import restore_core",
                f"restore_core.py est introuvable ou contient une erreur :\n\n{_CORE_ERROR}\n\n"
                "Assurez-vous que restore_core.py est dans le même dossier que restore_gui.py."
            )

    # ── Construction de l'UI ───────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Séparateur horizontal principal : panneau contrôle | zone images
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(4)
        root.addWidget(h_split, 1)

        # ── Panneau gauche : contrôles ────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFixedWidth(300)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        ctrl_widget = QWidget()
        ctrl_scroll.setWidget(ctrl_widget)
        ctrl_lay = QVBoxLayout(ctrl_widget)
        ctrl_lay.setContentsMargins(8, 10, 8, 10)
        ctrl_lay.setSpacing(6)

        # Bouton Ouvrir
        open_btn = QPushButton("📂  Ouvrir une image…")
        open_btn.setMinimumHeight(36)
        open_btn.clicked.connect(self.open_image)
        ctrl_lay.addWidget(open_btn)

        # Bouton Lancer
        self._run_btn = QPushButton("▶  Lancer le pipeline")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(lambda: self._run_from("color"))
        ctrl_lay.addWidget(self._run_btn)

        # Bouton Annuler
        self._cancel_btn = QPushButton("⏹  Annuler")
        self._cancel_btn.setMinimumHeight(28)
        self._cancel_btn.hide()
        self._cancel_btn.clicked.connect(self._cancel_pipeline)
        ctrl_lay.addWidget(self._cancel_btn)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #3a3a3a;")
        ctrl_lay.addWidget(sep)

        # Panneaux d'étapes
        self._step_panels: dict[str, StepPanel] = {}
        for sdef in STEP_DEFS:
            panel = StepPanel(sdef)
            panel.param_changed.connect(self._on_param_changed)
            panel.rerun_requested.connect(self._run_from)
            self._step_panels[sdef["id"]] = panel
            ctrl_lay.addWidget(panel)

        ctrl_lay.addStretch()

        # Boutons export
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #3a3a3a;")
        ctrl_lay.addWidget(sep2)

        export_btn = QPushButton("💾  Enregistrer le résultat…")
        export_btn.clicked.connect(self._export_result)
        ctrl_lay.addWidget(export_btn)

        export_grid_btn = QPushButton("🔲  Exporter la grille de comparaison…")
        export_grid_btn.clicked.connect(self._export_grid)
        ctrl_lay.addWidget(export_grid_btn)

        h_split.addWidget(ctrl_scroll)

        # ── Zone droite : vues + strip vignettes ──────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        h_split.addWidget(right)
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)

        # Séparateur vertical des deux vues A | B
        self._view_split = QSplitter(Qt.Orientation.Horizontal)
        self._view_split.setHandleWidth(3)
        right_lay.addWidget(self._view_split, 1)

        self._view_a = SyncedImageView()
        self._view_b = SyncedImageView()
        self._view_a.set_step_label("Originale")
        self._view_split.addWidget(self._view_a)
        self._view_split.addWidget(self._view_b)
        self._view_b.hide()

        # Strip de vignettes (scroll horizontal)
        thumb_scroll = QScrollArea()
        thumb_scroll.setWidgetResizable(True)
        thumb_scroll.setFixedHeight(THUMB_H + 62)
        thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        thumb_container = QWidget()
        thumb_lay = QHBoxLayout(thumb_container)
        thumb_lay.setContentsMargins(8, 6, 8, 6)
        thumb_lay.setSpacing(8)

        self._thumbs: dict[str, ThumbnailCard] = {}

        # Vignette "Originale"
        orig_card = ThumbnailCard("original", "Originale")
        orig_card.selected_a.connect(self._on_select_a)
        orig_card.selected_b.connect(self._on_select_b)
        orig_card.set_selected_a(True)
        self._thumbs["original"] = orig_card
        thumb_lay.addWidget(orig_card)

        # Vignettes des étapes
        for sdef in STEP_DEFS:
            card = ThumbnailCard(sdef["id"], sdef["short_name"])
            card.selected_a.connect(self._on_select_a)
            card.selected_b.connect(self._on_select_b)
            self._thumbs[sdef["id"]] = card
            thumb_lay.addWidget(card)

        thumb_lay.addStretch()
        thumb_scroll.setWidget(thumb_container)
        right_lay.addWidget(thumb_scroll)

    def _setup_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&Fichier")
        a_open = QAction("&Ouvrir…", self)
        a_open.setShortcut(QKeySequence.StandardKey.Open)
        a_open.triggered.connect(self.open_image)

        a_save = QAction("&Enregistrer le résultat…", self)
        a_save.setShortcut(QKeySequence.StandardKey.Save)
        a_save.triggered.connect(self._export_result)

        a_grid = QAction("Exporter la &grille…", self)
        a_grid.triggered.connect(self._export_grid)

        a_quit = QAction("&Quitter", self)
        a_quit.setShortcut(QKeySequence.StandardKey.Quit)
        a_quit.triggered.connect(self.close)

        file_menu.addAction(a_open)
        file_menu.addAction(a_save)
        file_menu.addAction(a_grid)
        file_menu.addSeparator()
        file_menu.addAction(a_quit)

        view_menu = mb.addMenu("&Vue")
        a_fit = QAction("&Ajuster à l'écran  [F]", self)
        a_fit.triggered.connect(self._fit_all)
        a_100 = QAction("Zoom &100 %  [1]", self)
        a_100.triggered.connect(self._zoom_100)
        a_close_b = QAction("Fermer la vue &comparaison  [Éch]", self)
        a_close_b.triggered.connect(lambda: self._select_b(None))
        view_menu.addAction(a_fit)
        view_menu.addAction(a_100)
        view_menu.addSeparator()
        view_menu.addAction(a_close_b)

    def _setup_status_bar(self):
        sb = self.statusBar()
        self._status_lbl = QLabel("Prêt — Ouvrez une image pour commencer")
        self._prog_bar   = QProgressBar()
        self._prog_bar.setMaximumWidth(200)
        self._prog_bar.setRange(0, 4)
        self._prog_bar.setValue(0)
        self._prog_bar.hide()
        sb.addWidget(self._status_lbl, 1)
        sb.addPermanentWidget(self._prog_bar)

    def _setup_shortcuts(self):
        from PyQt6.QtGui import QShortcut
        QShortcut(QKeySequence("F"),      self, self._fit_all)
        QShortcut(QKeySequence("1"),      self, self._zoom_100)
        QShortcut(QKeySequence("Escape"), self, lambda: self._select_b(None))

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget      { background: #1a1a1a; color: #ddd; }
            QScrollArea               { background: #1a1a1a; border: none; }
            QPushButton {
                background: #2d2d2d; color: #ddd;
                border: 1px solid #484848; border-radius: 4px; padding: 4px 10px;
            }
            QPushButton:hover         { background: #383838; }
            QPushButton:pressed       { background: #222; }
            QPushButton:disabled      { color: #555; border-color: #333; }
            QSlider::groove:horizontal {
                background: #3a3a3a; height: 4px; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #5a8fd8; width: 13px; height: 13px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::handle:horizontal:hover { background: #6fa3f0; }
            QDoubleSpinBox, QSpinBox, QComboBox {
                background: #252525; border: 1px solid #484848;
                border-radius: 3px; padding: 2px 4px; color: #ddd;
            }
            QDoubleSpinBox::up-button, QSpinBox::up-button,
            QDoubleSpinBox::down-button, QSpinBox::down-button {
                background: #333; border: none; width: 14px;
            }
            QCheckBox                 { color: #bbb; spacing: 5px; }
            QCheckBox::indicator      { width: 14px; height: 14px; }
            QGroupBox                 { border: 1px solid #3a3a3a; border-radius: 4px; }
            QMenuBar                  { background: #222; border-bottom: 1px solid #333; }
            QMenuBar::item:selected   { background: #3a3a3a; }
            QMenu                     { background: #252525; border: 1px solid #444; }
            QMenu::item:selected      { background: #3a3a3a; }
            QSplitter::handle         { background: #2a2a2a; }
            QScrollBar:horizontal     { background: #1e1e1e; height: 9px; }
            QScrollBar::handle:horizontal { background: #444; border-radius: 4px; min-width: 20px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            QScrollBar:vertical       { background: #1e1e1e; width: 9px; }
            QScrollBar::handle:vertical { background: #444; border-radius: 4px; min-height: 20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QProgressBar {
                background: #252525; border: 1px solid #444;
                border-radius: 3px; text-align: center; font-size: 10px;
            }
            QProgressBar::chunk      { background: #3a7bd5; border-radius: 2px; }
            QStatusBar               { background: #1e1e1e; border-top: 1px solid #2e2e2e; }
            QToolButton              { border: none; background: transparent; }
        """)

    # ── Chargement d'image ─────────────────────────────────────────────────

    def open_image(self, path: str = ""):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Ouvrir une image",
                _SCRIPT_DIR,
                "Images (*.jpg *.jpeg *.png *.tiff *.tif *.bmp);;Tous les fichiers (*)"
            )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        # Fallback pour TIFFs 16-bit ou formats moins courants
        if img is None:
            try:
                from PIL import Image as _PIL
                pil = _PIL.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                pass

        if img is None:
            QMessageBox.critical(self, "Erreur de lecture",
                                 f"Impossible de lire le fichier :\n{path}")
            return

        # Convertir 16-bit → 8-bit si nécessaire
        if img.dtype != np.uint8:
            img = (img / (img.max() / 255.0)).clip(0, 255).astype(np.uint8)

        self._original     = img
        self._step_results = {}
        self._step_context = {}
        self._step_stale   = {s: True for s in STEP_IDS}

        # Réinitialiser l'UI
        for sid, thumb in self._thumbs.items():
            if sid != "original":
                thumb.set_empty()
        for panel in self._step_panels.values():
            panel.set_status("stale")

        self._thumbs["original"].set_image(img)

        # Fermer le panneau B si ouvert
        self._select_b(None)
        self._select_a("original")
        self._view_a.fit_image()

        self._run_btn.setEnabled(True)
        h, w = img.shape[:2]
        self.setWindowTitle(f"Restauration Photo — {os.path.basename(path)}")
        self._status_lbl.setText(
            f"Image chargée : {w}×{h} px  ·  {os.path.basename(path)}"
        )

    # ── Sélection des vues A / B ───────────────────────────────────────────

    def _on_select_a(self, step_id: str):
        self._select_a(step_id)

    def _on_select_b(self, step_id: str):
        # Re-clic droit sur B courant = fermer le split
        if self._current_b == step_id:
            self._select_b(None)
        else:
            self._select_b(step_id)

    def _select_a(self, step_id: str):
        if self._current_a:
            self._thumbs[self._current_a].set_selected_a(False)
        self._current_a = step_id
        self._thumbs[step_id].set_selected_a(True)

        img = self._get_img(step_id)
        self._view_a.set_image(img)
        self._view_a.set_step_label(self._step_label(step_id))
        if img is not None:
            QTimer.singleShot(50, self._view_a.fit_image)

    def _select_b(self, step_id: str | None):
        if self._current_b:
            self._thumbs[self._current_b].set_selected_b(False)
        self._current_b = step_id

        if step_id is None:
            self._view_b.hide()
            self._view_b.unlink_all()
            return

        self._thumbs[step_id].set_selected_b(True)
        img = self._get_img(step_id)
        self._view_b.set_image(img)
        self._view_b.set_step_label(self._step_label(step_id))

        # Lier la synchronisation
        self._view_b.unlink_all()
        self._view_a.link_peer(self._view_b)

        self._view_b.show()
        # Synchroniser B sur le viewport actuel de A
        t = self._view_a.transform()
        c = self._view_a.mapToScene(self._view_a.viewport().rect().center())
        self._view_b._apply_sync(t, c)

    def _get_img(self, step_id: str) -> np.ndarray | None:
        if step_id == "original":
            return self._original
        return self._step_results.get(step_id)

    def _step_label(self, step_id: str) -> str:
        if step_id == "original":
            return "Originale"
        return STEP_NAMES.get(step_id, step_id)

    # ── Exécution du pipeline ──────────────────────────────────────────────

    def _on_param_changed(self, step_id: str):
        """Invalide cette étape et toutes les suivantes."""
        invalidate = False
        for sid in STEP_IDS:
            if sid == step_id:
                invalidate = True
            if invalidate:
                self._step_stale[sid] = True
                if self._step_panels[sid].is_enabled():
                    self._step_panels[sid].set_status("stale")

    def _run_from(self, start_id: str):
        if self._original is None:
            return
        if self._worker and self._worker.isRunning():
            return

        # Construire la liste des étapes à exécuter (depuis start_id, enabled seulement)
        steps = []
        started = False
        for sid in STEP_IDS:
            if sid == start_id:
                started = True
            if started and self._step_panels[sid].is_enabled():
                steps.append(sid)

        if not steps:
            return

        # Image en entrée de la première étape
        first_idx = STEP_IDS.index(steps[0])
        if first_idx == 0:
            input_img = self._original
        else:
            prev_id   = STEP_IDS[first_idx - 1]
            input_img = self._step_results.get(prev_id, self._original)

        all_params = {sid: self._step_panels[sid].get_params() for sid in STEP_IDS}

        # Réinitialiser les statuts des étapes à exécuter
        for sid in steps:
            self._step_panels[sid].set_status("idle")
            self._thumbs[sid].set_empty()

        # UI pendant l'exécution
        self._run_btn.setEnabled(False)
        self._cancel_btn.show()
        self._prog_bar.setRange(0, len(steps))
        self._prog_bar.setValue(0)
        self._prog_bar.show()
        self._progress_done = 0
        self._status_lbl.setText("Pipeline en cours…")

        self._worker = PipelineWorker(steps, input_img, all_params,
                                      self._step_context, self)
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.step_failed.connect(self._on_step_failed)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    def _cancel_pipeline(self):
        if self._worker:
            self._worker.cancel()
        self._status_lbl.setText("Annulation en cours…")
        self._cancel_btn.setEnabled(False)

    # ── Signaux du worker ──────────────────────────────────────────────────

    def _on_step_started(self, step_id: str):
        self._step_panels[step_id].set_status("running")
        self._thumbs[step_id].set_running()
        self._status_lbl.setText(f"Étape en cours : {STEP_NAMES.get(step_id, step_id)}")

    def _on_step_done(self, step_id: str, result: object, extras: dict):
        result = np.asarray(result)
        self._step_results[step_id] = result
        self._step_context.update(extras)
        self._step_stale[step_id] = False
        self._step_panels[step_id].set_status("ok")
        self._thumbs[step_id].set_image(result)
        self._progress_done += 1
        self._prog_bar.setValue(self._progress_done)

        # Rafraîchir la vue principale si cette étape y est affichée
        if self._current_a == step_id:
            self._view_a.set_image(result)
        if self._current_b == step_id:
            self._view_b.set_image(result)

    def _on_step_failed(self, step_id: str, error: str):
        self._step_panels[step_id].set_status("error")
        self._thumbs[step_id].set_error()
        self._status_lbl.setText(f"Erreur — {STEP_SHORT.get(step_id, step_id)} : {error}")
        self._progress_done += 1
        self._prog_bar.setValue(self._progress_done)

    def _on_all_done(self):
        self._run_btn.setEnabled(self._original is not None)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.hide()
        self._prog_bar.hide()
        cancelled = self._worker and self._worker._cancel
        if cancelled:
            self._status_lbl.setText("Pipeline annulé")
        else:
            n_ok = sum(1 for sid in STEP_IDS if sid in self._step_results)
            self._status_lbl.setText(f"Pipeline terminé ✓  — {n_ok} étape(s) calculées")

    # ── Actions de vue ─────────────────────────────────────────────────────

    def _fit_all(self):
        self._view_a.fit_image()
        if self._current_b:
            self._view_b.fit_image()

    def _zoom_100(self):
        t = QTransform()
        self._view_a.setTransform(t)
        if self._current_b:
            self._view_b.setTransform(t)
        self._view_a._broadcast()

    # ── Export ─────────────────────────────────────────────────────────────

    def _export_result(self):
        # Exporter le dernier résultat calculé
        result = None
        for sid in reversed(STEP_IDS):
            if sid in self._step_results:
                result = self._step_results[sid]
                break
        if result is None:
            result = self._original
        if result is None:
            QMessageBox.warning(self, "Rien à exporter",
                                "Aucun résultat disponible. Ouvrez et traitez une image d'abord.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le résultat",
            _SCRIPT_DIR,
            "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tif)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
        cv2.imwrite(path, result, params)
        self._status_lbl.setText(f"Exporté : {os.path.basename(path)}")

    def _export_grid(self):
        images = {}
        if self._original is not None:
            images["Original"] = self._original
        for sdef in STEP_DEFS:
            sid = sdef["id"]
            if sid in self._step_results:
                images[sdef["short_name"]] = self._step_results[sid]

        if len(images) < 2:
            QMessageBox.information(self, "Grille",
                "Calculez au moins une étape avant d'exporter la grille.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter la grille",
            _SCRIPT_DIR,
            "JPEG (*.jpg);;PNG (*.png)"
        )
        if not path:
            return

        grid = make_grid(images)
        ext  = os.path.splitext(path)[1].lower()
        cv2.imwrite(path, grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 90] if ext in (".jpg", ".jpeg") else [])
        self._status_lbl.setText(f"Grille exportée : {os.path.basename(path)}")

    # ── Fermeture propre ───────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        event.accept()


# ===========================================================================
# Point d'entrée
# ===========================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Restauration Photo")

    window = MainApp()
    window.show()

    # Si un chemin d'image est passé en argument CLI, l'ouvrir directement
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.open_image(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
