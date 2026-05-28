"""ui/control_panel.py — Panneau de contrôle gauche (liste des étapes + bouton run)."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ui.step_panel import StepListWidget


class ControlPanel(QWidget):
    """Panneau gauche : bouton Run + liste des étapes paramétrables."""

    run_requested               = pyqtSignal()
    stop_requested              = pyqtSignal()
    order_changed               = pyqtSignal(list)
    order_reordered             = pyqtSignal(list, list)      # (old_order, new_order)
    param_changed               = pyqtSignal(str, str, object)
    param_propagate_requested   = pyqtSignal(str, str, object)
    enabled_changed             = pyqtSignal(str, bool)
    enabled_propagate_requested = pyqtSignal(str, bool)
    rerun_requested             = pyqtSignal(str)
    overlay_toggled             = pyqtSignal(str, bool)
    mask_edit_requested         = pyqtSignal(str)
    color_picker_requested      = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self.setStyleSheet("background: #16162a;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barre supérieure ─────────────────────────────────────────────────
        top_bar = QWidget()
        top_bar.setFixedHeight(48)
        top_bar.setStyleSheet("background: #1e1e38; border-bottom: 1px solid #2a2a4a;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 0, 12, 0)

        title = QLabel("Pipeline")
        title.setStyleSheet("color: #ddd; font-size: 13px; font-weight: 700;")
        top_layout.addWidget(title)
        top_layout.addStretch()

        self._run_btn = QPushButton("▶  Lancer")
        self._run_btn.setStyleSheet(
            "QPushButton { background: #2a6496; color: #fff; border-radius: 5px;"
            "  padding: 6px 16px; font-size: 12px; font-weight: 600; }"
            "QPushButton:hover { background: #3a74a6; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self._run_btn.clicked.connect(self.run_requested)
        top_layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton("⏹")
        self._stop_btn.setFixedWidth(36)
        self._stop_btn.setStyleSheet(
            "QPushButton { background: #5a2020; color: #f66; border-radius: 5px;"
            "  padding: 6px; font-size: 12px; }"
            "QPushButton:hover { background: #7a3030; }"
        )
        self._stop_btn.hide()
        self._stop_btn.clicked.connect(self.stop_requested)
        top_layout.addWidget(self._stop_btn)

        root.addWidget(top_bar)

        # ── Liste des étapes (scrollable) ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none;")
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.step_list = StepListWidget()
        self.step_list.order_changed.connect(self.order_changed)
        self.step_list.order_reordered.connect(self.order_reordered)
        self.step_list.param_changed.connect(self.param_changed)
        self.step_list.param_propagate_requested.connect(self.param_propagate_requested)
        self.step_list.enabled_changed.connect(self.enabled_changed)
        self.step_list.enabled_propagate_requested.connect(self.enabled_propagate_requested)
        self.step_list.rerun_requested.connect(self.rerun_requested)
        self.step_list.overlay_toggled.connect(self.overlay_toggled)
        self.step_list.mask_edit_requested.connect(self.mask_edit_requested)
        self.step_list.color_picker_requested.connect(self.color_picker_requested)

        scroll.setWidget(self.step_list)
        root.addWidget(scroll, stretch=1)

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._stop_btn.setVisible(running)
