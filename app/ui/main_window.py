"""ui/main_window.py — Fenêtre principale de l'application."""

from __future__ import annotations
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QFileDialog, QStatusBar,
)
from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QKeySequence, QShortcut, QAction, QUndoStack

from core.history import HistoryManager
from core.pipeline import PipelineWorker
from steps import ALL_STEPS
from ui.image_view import SyncedImageView
from ui.thumbnail_strip import ThumbnailStrip
from ui.history_panel import HistoryPanel
from ui.control_panel import ControlPanel
from ui.theme import apply_dark_theme
from ui.mixins.file_io import FileIOMixin
from ui.mixins.batch import BatchMixin
from ui.mixins.editors import EditorsMixin
from ui.mixins.views import ViewsMixin
from ui.mixins.params import ParamsMixin
from ui.mixins.pipeline import PipelineMixin
from ui.mixins.preview import PreviewMixin


class MainApp(
    PipelineMixin,
    PreviewMixin,
    ParamsMixin,
    ViewsMixin,
    FileIOMixin,
    BatchMixin,
    EditorsMixin,
    QMainWindow,
):
    """Fenêtre principale de l'outil de restauration photo."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Restauration Photo — v2")
        self.resize(1400, 900)

        # ── État ─────────────────────────────────────────────────────────────
        self._original:       Optional[np.ndarray]      = None
        self._original_path:  str                       = ""
        self._history:        HistoryManager             = HistoryManager()
        self._worker:         Optional[PipelineWorker]  = None
        self._active_run_id:  Optional[int]             = None  # run affiché dans les vignettes

        # Sources A / B : (run_id ou None, step_id ou "original")
        self._view_a: tuple[Optional[int], str] = (None, "original")
        self._view_b: Optional[tuple[Optional[int], str]] = None

        # Overlay détections (step_redeye) — actif sans recalcul
        self._overlay_step: Optional[str] = None   # step_id dont l'overlay est actif

        # Fenêtre batch (une seule instance à la fois)
        self._batch_window = None

        # Flag pour éviter de push une 2ème commande lors du redo/undo d'un réordre
        self._applying_order = False
        # Flag pour éviter de push lors du redo/undo d'activation
        self._applying_enabled = False

        # Undo/Redo — pile de commandes paramétriques
        self._undo_stack = QUndoStack(self)
        # Snapshot des valeurs courantes des paramètres (pour capturer old_val)
        self._params_snapshot: dict[str, dict] = {
            s.id: dict(s.default_params()) for s in ALL_STEPS
        }
        # Buffer pour regrouper les changements de preset en une seule commande macro
        self._preset_buf: list = []
        self._preset_flush_pending = False

        # ── Preview instantané (étapes rapides) ──────────────────────────────
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)          # 150 ms de debounce
        self._preview_timer.timeout.connect(self._run_preview)
        self._preview_step_id: Optional[str]       = None
        self._preview_active:  bool                = False

        # Ordre et paramètres courants
        self._step_order:   list[str]       = [s.id for s in ALL_STEPS]
        self._step_enabled: dict[str, bool] = {
            s.id: getattr(s, "enabled_by_default", True) for s in ALL_STEPS
        }

        self._steps_by_id = {s.id: s for s in ALL_STEPS}

        # Images des etapes du dernier run (pour l'editeur de masque)
        self._last_step_images: dict[str, np.ndarray] = {}

        self._build_ui()
        apply_dark_theme(self)
        self._setup_shortcuts()

    # ══════════════════════════════════════════════════════════════════════════
    # Construction de l'UI
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Barre de menu ────────────────────────────────────────────────────
        self._build_menu()

        # ── Zone principale : panneau gauche + vues ──────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Panneau de contrôle (gauche)
        self._ctrl = ControlPanel()
        self._ctrl.step_list.add_steps(ALL_STEPS)
        self._ctrl.run_requested.connect(self._run_pipeline)
        self._ctrl.stop_requested.connect(self._stop_pipeline)
        self._ctrl.order_changed.connect(self._on_order_changed)
        self._ctrl.param_changed.connect(self._on_param_changed)
        self._ctrl.enabled_changed.connect(self._on_enabled_changed)
        self._ctrl.rerun_requested.connect(self._run_from)
        self._ctrl.overlay_toggled.connect(self._on_overlay_toggled)
        self._ctrl.mask_edit_requested.connect(self._on_mask_edit_requested)
        self._ctrl.color_picker_requested.connect(self._on_color_picker_requested)
        splitter.addWidget(self._ctrl)

        # Zone d'affichage (droite)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Vues A et B côte à côte
        views_widget = QWidget()
        views_layout = QHBoxLayout(views_widget)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(2)

        self._view_a_widget = SyncedImageView()
        self._view_b_widget = SyncedImageView()
        self._view_a_widget.add_peer(self._view_b_widget)

        # Labels A / B enrichis ("A \u2014 v2 \u00b7 GFPGAN")
        def _wrap_view(view: SyncedImageView, prefix: str) -> tuple[QWidget, QLabel]:
            w = QWidget()
            l = QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(0)
            lbl = QLabel(prefix)
            lbl.setFixedHeight(18)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #1a1a2e; color: #777; font-size: 10px;")
            l.addWidget(lbl)
            l.addWidget(view, stretch=1)
            return w, lbl

        self._label_a, self._view_a_lbl = _wrap_view(self._view_a_widget, "A")
        self._label_b, self._view_b_lbl = _wrap_view(self._view_b_widget, "B")
        views_layout.addWidget(self._label_a, stretch=1)
        views_layout.addWidget(self._label_b, stretch=1)
        self._label_b.setVisible(False)   # masqué jusqu'à sélection B

        right_layout.addWidget(views_widget, stretch=1)

        # Historique des runs
        self._history_panel = HistoryPanel()
        self._history_panel.version_activated.connect(self._on_history_activated)
        right_layout.addWidget(self._history_panel)

        # Bande de vignettes
        self._thumb_strip = ThumbnailStrip()
        self._thumb_strip.selected_a.connect(self._on_thumb_a)
        self._thumb_strip.selected_b.connect(self._on_thumb_b)
        self._thumb_strip.save_requested.connect(self._on_save_thumb)
        right_layout.addWidget(self._thumb_strip)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1060])

        main_layout.addWidget(splitter, stretch=1)

        # ── Barre de statut ──────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet("background: #111124; color: #888; font-size: 10px;")
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Prêt — ouvrir une image avec Ctrl+O")

    def _build_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("background: #16162a; color: #ccc;")

        file_menu = menubar.addMenu("Fichier")

        open_act = QAction("Ouvrir…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self._open_image)
        file_menu.addAction(open_act)

        save_act = QAction("Enregistrer le résultat…", self)
        save_act.setShortcut(QKeySequence("Ctrl+S"))
        save_act.triggered.connect(self._save_result)
        file_menu.addAction(save_act)

        file_menu.addSeparator()
        self._batch_act = QAction("📂  Batch…", self)
        self._batch_act.setShortcut(QKeySequence("Ctrl+B"))
        self._batch_act.triggered.connect(self._on_batch_requested)
        file_menu.addAction(self._batch_act)

        self._recent_menu = file_menu.addMenu("📋  Récents")
        self._rebuild_recent_menu()

        file_menu.addSeparator()
        quit_act = QAction("Quitter", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+R"), self, self._run_pipeline)
        QShortcut(QKeySequence("F"), self, self._fit_views)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo_stack.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._undo_stack.redo)

