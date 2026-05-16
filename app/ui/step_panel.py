"""ui/step_panel.py — Panneau d'une étape avec drag-and-drop et états visuels.

Hiérarchie :
    StepListWidget
        └── StepPanel (×N)
                ├── _header  (cliquable → expand/collapse)
                │     ├── DragHandle
                │     ├── _enable_cb
                │     ├── _accent_bar (état visuel)
                │     ├── _name_lbl
                │     └── _status_lbl
                └── _body (masqué par défaut)
                      ├── ParamRow × N
                      └── QHBoxLayout (boutons)
"""

from __future__ import annotations
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QFrame, QSizePolicy, QScrollArea,
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QCursor

from ui.param_widgets import ParamRow


# ══════════════════════════════════════════════════════════════════════════════
# DragHandle
# ══════════════════════════════════════════════════════════════════════════════

class DragHandle(QLabel):
    """Poignée de drag (⋮⋮) dans l'en-tête d'un StepPanel."""

    drag_started = pyqtSignal()  # émis quand le mouvement dépasse le seuil

    def __init__(self, parent=None):
        super().__init__("⋮⋮", parent)
        self.setCursor(QCursor(Qt.CursorShape.SizeVerCursor))
        self.setStyleSheet("color: #555; font-size: 13px; padding: 0 6px;")
        self.setFixedWidth(24)
        self._pressing = False
        self._press_pos: Optional[QPoint] = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressing = True
            self._press_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pressing and self._press_pos is not None:
            delta = event.globalPosition().toPoint() - self._press_pos
            if delta.manhattanLength() > 6:
                self._pressing = False
                self._press_pos = None
                self.drag_started.emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._pressing = False
        super().mouseReleaseEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
# StepPanel
# ══════════════════════════════════════════════════════════════════════════════

# États visuels  →  couleur de la barre d'accent
_STATE_COLORS = {
    "idle":     "#444",
    "ok":       "#2ecc71",
    "stale":    "#f39c12",
    "running":  "#3a7bd5",
    "error":    "#e74c3c",
    "disabled": "#2a2a2a",
}

_STATE_LABELS = {
    "idle":     "",
    "ok":       "✓",
    "stale":    "⚠ obsolète",
    "running":  "⏳ en cours…",
    "error":    "✗ erreur",
    "disabled": "désactivé",
}


class StepPanel(QWidget):
    """Panneau d'une étape : en-tête + corps pliable."""

    enabled_changed  = pyqtSignal(str, bool)         # (step_id, enabled)
    param_changed    = pyqtSignal(str, str, object)  # (step_id, key, value)
    rerun_requested  = pyqtSignal(str)               # step_id
    drag_requested   = pyqtSignal(str)               # step_id — relayé depuis DragHandle
    overlay_toggled  = pyqtSignal(str, bool)          # (step_id, enabled) — sans recalcul

    def __init__(self, step, parent=None):
        super().__init__(parent)
        self._step      = step
        self._state     = "idle"
        self._param_rows: dict[str, ParamRow] = {}
        self._expanded  = False

        self.setObjectName("StepPanel")
        self._build_ui()
        self._apply_state("idle")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── En-tête ─────────────────────────────────────────────────────────
        self._header = QWidget()
        self._header.setObjectName("StepHeader")
        self._header.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._header.setFixedHeight(38)
        hdr_layout = QHBoxLayout(self._header)
        hdr_layout.setContentsMargins(0, 0, 8, 0)
        hdr_layout.setSpacing(0)

        # Barre d'accent latérale
        self._accent_bar = QFrame()
        self._accent_bar.setFixedWidth(4)
        self._accent_bar.setObjectName("AccentBar")
        hdr_layout.addWidget(self._accent_bar)

        # Drag handle
        self._drag_handle = DragHandle()
        self._drag_handle.drag_started.connect(
            lambda: self.drag_requested.emit(self._step.id)
        )
        hdr_layout.addWidget(self._drag_handle)

        # Checkbox activer/désactiver
        self._enable_cb = QCheckBox()
        self._enable_cb.setChecked(getattr(self._step, "enabled_by_default", True))
        self._enable_cb.setStyleSheet("margin-right: 4px;")
        self._enable_cb.toggled.connect(
            lambda v: self.enabled_changed.emit(self._step.id, v)
        )
        hdr_layout.addWidget(self._enable_cb)

        # Flèche expand/collapse
        self._arrow = QLabel("▶")
        self._arrow.setStyleSheet("color: #666; font-size: 9px; padding: 0 4px;")
        self._arrow.setFixedWidth(16)
        hdr_layout.addWidget(self._arrow)

        # Nom de l'étape
        self._name_lbl = QLabel(self._step.name)
        self._name_lbl.setStyleSheet("color: #ddd; font-size: 12px; font-weight: 600;")
        hdr_layout.addWidget(self._name_lbl, stretch=1)

        # Statut court
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #888; font-size: 10px;")
        hdr_layout.addWidget(self._status_lbl)

        self._header.mousePressEvent = lambda e: self._toggle()
        root.addWidget(self._header)

        # ── Corps (paramètres) ───────────────────────────────────────────────
        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(28, 4, 8, 8)
        body_layout.setSpacing(4)

        for pdef in self._step.param_defs:
            row = ParamRow(pdef)
            row.value_changed.connect(
                lambda key, val, sid=self._step.id:
                    self.param_changed.emit(sid, key, val)
            )
            self._param_rows[pdef["key"]] = row
            body_layout.addWidget(row)

        # Checkbox overlay (optionnelle, pour les étapes avec has_overlay=True)
        self._overlay_cb: Optional[QCheckBox] = None
        if getattr(self._step, "has_overlay", False):
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet("background: #2a2a3a; margin: 2px 0;")
            sep.setFixedHeight(1)
            body_layout.addWidget(sep)
            self._overlay_cb = QCheckBox("Afficher détections (overlay)")
            self._overlay_cb.setStyleSheet(
                "QCheckBox { color: #7ec8c8; font-size: 11px; margin-top: 2px; }"
            )
            self._overlay_cb.toggled.connect(
                lambda v, sid=self._step.id: self.overlay_toggled.emit(sid, v)
            )
            body_layout.addWidget(self._overlay_cb)

        # Bouton Recalculer
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        self._rerun_btn = QPushButton("▶  Recalculer depuis ici")
        self._rerun_btn.setStyleSheet(
            "QPushButton { background: #2a3f5f; color: #aad; border-radius: 4px; padding: 4px 12px; font-size: 11px; }"
            "QPushButton:hover { background: #3a5f8f; }"
        )
        self._rerun_btn.clicked.connect(lambda: self.rerun_requested.emit(self._step.id))
        btn_row.addWidget(self._rerun_btn)
        btn_row.addStretch()
        body_layout.addLayout(btn_row)

        self._body.setVisible(False)
        root.addWidget(self._body)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #2a2a3a;")
        sep.setFixedHeight(1)
        root.addWidget(sep)

        # Style global du panneau
        self.setStyleSheet("""
            QWidget#StepPanel { background: #1e1e2e; }
            QWidget#StepHeader:hover { background: #252535; }
        """)

    # ── API publique ─────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {k: row.value() for k, row in self._param_rows.items()}

    def set_params(self, params: dict):
        for k, v in params.items():
            if k in self._param_rows:
                self._param_rows[k].set_value(v)

    def set_state(self, state: str, message: str = ""):
        self._state = state
        self._apply_state(state, message)

    def is_enabled(self) -> bool:
        return self._enable_cb.isChecked()

    def set_enabled(self, val: bool):
        self._enable_cb.setChecked(val)

    def reset_overlay(self):
        """Désactive le checkbox overlay sans émettre de signal (ex. nouvelle image)."""
        if self._overlay_cb is not None:
            self._overlay_cb.blockSignals(True)
            self._overlay_cb.setChecked(False)
            self._overlay_cb.blockSignals(False)

    # ── Interne ──────────────────────────────────────────────────────────────

    def _toggle(self):
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._arrow.setText("▼" if self._expanded else "▶")

    def _apply_state(self, state: str, message: str = ""):
        color = _STATE_COLORS.get(state, "#444")
        self._accent_bar.setStyleSheet(f"background: {color}; border-radius: 2px;")
        label = message if message else _STATE_LABELS.get(state, "")
        self._status_lbl.setText(label)

        is_disabled = state == "disabled" or not self._enable_cb.isChecked()
        alpha = "0.4" if is_disabled else "1.0"
        self._name_lbl.setStyleSheet(
            f"color: #ddd; font-size: 12px; font-weight: 600; opacity: {alpha};"
        )


# ══════════════════════════════════════════════════════════════════════════════
# StepListWidget — liste des panneaux avec drag-and-drop
# ══════════════════════════════════════════════════════════════════════════════

class StepListWidget(QWidget):
    """Conteneur des StepPanels avec réordonnancement par drag-and-drop."""

    order_changed    = pyqtSignal(list)           # list[str] nouvel ordre des step_ids
    param_changed    = pyqtSignal(str, str, object)
    enabled_changed  = pyqtSignal(str, bool)
    rerun_requested  = pyqtSignal(str)
    overlay_toggled  = pyqtSignal(str, bool)      # relayé depuis les panneaux

    def __init__(self, parent=None):
        super().__init__(parent)
        self._panels: dict[str, StepPanel] = {}
        self._order:  list[str]            = []

        self._vbox = QVBoxLayout(self)
        self._vbox.setContentsMargins(0, 0, 0, 0)
        self._vbox.setSpacing(0)
        self._vbox.addStretch()

        # Drag state
        self._dragging:    Optional[str]   = None
        self._drag_insert: int             = 0

        # Indicateur d'insertion
        self._insert_line = QFrame(self)
        self._insert_line.setFixedHeight(3)
        self._insert_line.setStyleSheet("background: #3a7bd5; border-radius: 1px;")
        self._insert_line.hide()
        self._insert_line.raise_()

        self.setMouseTracking(True)

    def add_steps(self, steps: list):
        for step in steps:
            panel = StepPanel(step)
            panel.drag_requested.connect(self._start_drag)
            panel.param_changed.connect(self.param_changed)
            panel.enabled_changed.connect(self.enabled_changed)
            panel.rerun_requested.connect(self.rerun_requested)
            panel.overlay_toggled.connect(self.overlay_toggled)
            self._panels[step.id] = panel
            self._order.append(step.id)
            self._vbox.insertWidget(len(self._order) - 1, panel)

    def get_panel(self, step_id: str) -> Optional[StepPanel]:
        return self._panels.get(step_id)

    def reset_overlays(self):
        """Réinitialise tous les checkboxes overlay (sans émettre de signal)."""
        for panel in self._panels.values():
            panel.reset_overlay()

    def get_order(self) -> list[str]:
        return list(self._order)

    def get_all_params(self) -> dict[str, dict]:
        return {sid: self._panels[sid].get_params() for sid in self._order}

    def get_enabled(self) -> dict[str, bool]:
        return {sid: self._panels[sid].is_enabled() for sid in self._order}

    def set_all_states(self, states: dict[str, str]):
        for sid, state in states.items():
            if sid in self._panels:
                self._panels[sid].set_state(state)

    # ── Drag ────────────────────────────────────────────────────────────────

    def _start_drag(self, step_id: str):
        self._dragging    = step_id
        self._drag_insert = self._order.index(step_id)
        self.grabMouse()
        self._panels[step_id].setStyleSheet(
            "QWidget#StepPanel { background: #14141e; opacity: 0.6; }"
        )

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        y = event.position().y()
        insert_idx = len(self._order)
        for i, sid in enumerate(self._order):
            panel = self._panels[sid]
            mid   = panel.pos().y() + panel.height() / 2
            if y < mid:
                insert_idx = i
                break
        self._drag_insert = insert_idx

        # Afficher l'indicateur
        if insert_idx < len(self._order):
            target_panel = self._panels[self._order[insert_idx]]
            line_y = target_panel.pos().y() - 2
        else:
            last = self._panels[self._order[-1]]
            line_y = last.pos().y() + last.height() + 2

        self._insert_line.setGeometry(0, line_y, self.width(), 3)
        self._insert_line.show()
        self._insert_line.raise_()

    def mouseReleaseEvent(self, event):
        if self._dragging is None:
            return
        self.releaseMouse()
        self._insert_line.hide()

        step_id  = self._dragging
        self._dragging = None
        old_idx  = self._order.index(step_id)
        new_idx  = self._drag_insert

        # Remettre le style normal
        self._panels[step_id].setStyleSheet("")

        if new_idx > old_idx:
            new_idx -= 1

        if old_idx != new_idx:
            self._order.pop(old_idx)
            self._order.insert(new_idx, step_id)
            self._rebuild_layout()
            self.order_changed.emit(list(self._order))

    def _rebuild_layout(self):
        for panel in self._panels.values():
            self._vbox.removeWidget(panel)
        for sid in self._order:
            stretch_idx = self._vbox.count() - 1
            self._vbox.insertWidget(stretch_idx, self._panels[sid])
