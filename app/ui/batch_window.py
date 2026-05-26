"""ui/batch_window.py — Fenêtre de traitement par lots (mode batch).

Layout ::

    ┌── Toolbar: [Sortie: /...][Parcourir][\u25b6 Lancer la sélection][↓ Appliquer à toutes][⚡ Lancer le batch] ───────┐
    ├────────── BatchThumbnailStrip ──────────────────────────────────────────────┤
    │ StepListWidget  │       ImageView (SyncedImageView)     │  QTabWidget      │
    │ (≈260 px)       │                                       │  [Masque][Blanc] │
    │                 │            (expanding)                │  MaskCanvasPanel │
    │                 │                                       │  WBPickerPanel   │
    └─────────────────┴───────────────────────────────────────┴──────────────────┘

Chaque image possède sa propre configuration (step_order, params, masque, WB).
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSplitter, QScrollArea, QTabWidget,
    QFileDialog, QStatusBar, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal, QObject
from PyQt6.QtGui import QCloseEvent

from core.pipeline import PipelineWorker
from core.batch import BatchSession, BatchImageConfig, build_step_log, save_recipe
from steps import ALL_STEPS
from ui.step_panel import StepListWidget
from ui.batch_thumbnail_strip import BatchThumbnailStrip
from ui.mask_editor import MaskCanvasPanel
from ui.wb_picker import WBPickerPanel
from ui.image_view import SyncedImageView

# Étapes dont les paramètres déclenchent un aperçu rapide
_FAST_PREVIEW_IDS = frozenset({"color", "facehighlight", "ddcolor_lut", "autocolor", "wb",
                               "lightleak", "rembg"})


class BatchWindow(QMainWindow):
    """Fenêtre de traitement par lots."""

    # Émis lors de la fermeture — la fenêtre principale peut se réabonner
    closed = pyqtSignal()

    def __init__(self, parent, session: BatchSession) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch — Traitement par lots")
        self.resize(1600, 950)

        self._session     = session
        self._steps_by_id = {s.id: s for s in ALL_STEPS}
        self._current_cfg: Optional[BatchImageConfig] = None
        self._worker:      Optional[PipelineWorker]   = None

        # La fenêtre doit se souvenir de l'image originale (pour masque/WB)
        self._current_orig: Optional[np.ndarray] = None

        # Aperçu rapide : timer debounce
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._do_preview_update)

        self._build_ui()
        self._apply_theme()

        # Sélectionner la première image si présente
        if session.images:
            self._navigate_to(session.images[0])
            self._strip.select(session.images[0].file_path)

    # ══════════════════════════════════════════════════════════════════════════
    # Construction de l'UI
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barre d'outils ────────────────────────────────────────────────────
        root.addWidget(self._build_toolbar())

        # ── Bande de miniatures ───────────────────────────────────────────────
        self._strip = BatchThumbnailStrip()
        self._strip.image_selected.connect(self._on_strip_image_selected)
        self._strip.image_reset.connect(self._on_strip_image_reset)
        root.addWidget(self._strip)

        # ── Zone principale : étapes | image | panneaux ────────────────────────
        root.addWidget(self._build_main_area(), stretch=1)

        # ── Barre de statut ───────────────────────────────────────────────────
        self._statusbar = QStatusBar()
        self._statusbar.setStyleSheet("background:#111124; color:#888; font-size:10px;")
        self.setStatusBar(self._statusbar)

        # Peupler la bande
        if self._session.images:
            self._strip.load_images([c.file_path for c in self._session.images])

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(48)
        bar.setStyleSheet("background:#1e1e38; border-bottom:1px solid #2a2a4a;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(8)

        lbl_out = QLabel("Sortie :")
        lbl_out.setStyleSheet("color:#aaa; font-size:11px;")
        lay.addWidget(lbl_out)

        self._output_lbl = QLabel(self._session.output_dir or "(non défini)")
        self._output_lbl.setStyleSheet(
            "color:#9de; font-size:11px; max-width:400px;"
        )
        self._output_lbl.setToolTip(self._session.output_dir)
        lay.addWidget(self._output_lbl, stretch=1)

        btn_browse = QPushButton("📁  Parcourir…")
        self._style_btn(btn_browse)
        btn_browse.clicked.connect(self._choose_output_dir)
        lay.addWidget(btn_browse)

        lay.addStretch()

        self._selection_btn = QPushButton("▶  Lancer la sélection")
        self._style_btn(self._selection_btn, accent=True)
        self._selection_btn.setToolTip(
            "Traiter et sauvegarder les images sélectionnées (Ctrl+clic / Shift+clic)"
        )
        self._selection_btn.clicked.connect(self._run_on_selection)
        lay.addWidget(self._selection_btn)

        self._apply_btn = QPushButton("↓  Appliquer à toutes")
        self._style_btn(self._apply_btn)
        self._apply_btn.setToolTip(
            "Propager les paramètres de l’image courante vers les images non personnalisées"
        )
        self._apply_btn.clicked.connect(self._apply_to_all_uncustomized)
        lay.addWidget(self._apply_btn)

        self._batch_btn = QPushButton("⚡  Lancer le batch")
        self._style_btn(self._batch_btn, accent=True)
        self._batch_btn.setToolTip("Traiter et sauvegarder toutes les images")
        self._batch_btn.clicked.connect(self._run_batch)
        lay.addWidget(self._batch_btn)

        self._stop_btn = QPushButton("⏹")
        self._stop_btn.setFixedWidth(36)
        self._stop_btn.setEnabled(False)
        self._style_btn(self._stop_btn)
        self._stop_btn.clicked.connect(self._stop)
        lay.addWidget(self._stop_btn)

        return bar

    def _build_main_area(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background:#1a1a2e; }")

        # ── Panneau gauche : liste des étapes ─────────────────────────────────
        step_container = QWidget()
        step_container.setMinimumWidth(250)
        step_container.setMaximumWidth(380)
        step_container.setStyleSheet("background:#16162a;")
        scl = QVBoxLayout(step_container)
        scl.setContentsMargins(0, 0, 0, 0)
        scl.setSpacing(0)

        hdr = QLabel("Étapes — par image")
        hdr.setFixedHeight(32)
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(
            "background:#1e1e38; color:#ddd; font-size:12px; font-weight:700;"
            " border-bottom:1px solid #2a2a4a;"
        )
        scl.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none; background:#16162a;")

        self._step_list = StepListWidget()
        self._step_list.add_steps(ALL_STEPS)
        self._step_list.order_changed.connect(self._on_order_changed)
        self._step_list.param_changed.connect(self._on_param_changed)
        self._step_list.enabled_changed.connect(self._on_enabled_changed)
        self._step_list.mask_edit_requested.connect(self._on_mask_edit_requested)
        self._step_list.color_picker_requested.connect(self._on_color_picker_requested)
        scroll.setWidget(self._step_list)
        scl.addWidget(scroll, stretch=1)
        splitter.addWidget(step_container)

        # ── Panneau droit : onglets ────────────────────────────────────────────
        right = QWidget()
        right.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right.setStyleSheet("background:#1a1a2e;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._tabs.setStyleSheet(
            "QTabWidget::pane { border:none; background:#1a1a2e; }"
            "QTabBar::tab { background:#16162a; color:#888; padding:6px 14px;"
            " border-bottom:2px solid transparent; }"
            "QTabBar::tab:selected { color:#9de; border-bottom:2px solid #3a7bd5; }"
            "QTabBar::tab:hover { color:#bbb; }"
        )

        # Créer les panneaux avec une image 1×1 — mis à jour dès la première navigation
        _dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        self._mask_panel = MaskCanvasPanel(_dummy, None, show_ok_cancel=False)
        self._wb_panel   = WBPickerPanel(
            _dummy, None, show_ok_cancel=False, sidebar_width=210
        )
        # Onglets d'aperçu image (zoom/pan via SyncedImageView)
        self._origin_view = SyncedImageView()
        self._dest_view   = SyncedImageView()

        self._tabs.addTab(self._mask_panel,   "Masque")     # index 0
        self._tabs.addTab(self._wb_panel,     "Blanc")      # index 1
        self._tabs.addTab(self._origin_view,  "Originale")  # index 2
        self._tabs.addTab(self._dest_view,    "Résultat")   # index 3
        self._tabs.currentChanged.connect(self._on_tab_changed)
        rl.addWidget(self._tabs)

        # Connecter les changements du point WB et du rayon pour déclencher l'aperçu
        self._wb_panel._canvas.pick_changed.connect(self._on_wb_pick_changed)
        self._wb_panel._rad_slider.valueChanged.connect(self._on_wb_radius_changed)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1200])
        return splitter

    # ══════════════════════════════════════════════════════════════════════════
    # Navigation entre images
    # ══════════════════════════════════════════════════════════════════════════

    def _on_strip_image_selected(self, file_path: str) -> None:
        cfg = self._config_for(file_path)
        if cfg is None:
            return
        if self._current_cfg is not None and self._current_cfg is not cfg:
            self._save_current_state()
        self._navigate_to(cfg)

    def _navigate_to(self, cfg: BatchImageConfig) -> None:
        """Charge la configuration d'une image dans l'UI."""
        self._current_cfg = cfg

        # ── Image originale ───────────────────────────────────────────────────
        img = cv2.imread(cfg.file_path, cv2.IMREAD_COLOR)
        self._current_orig = img

        # ── Étapes ────────────────────────────────────────────────────────────
        for sid, enabled in cfg.step_enabled.items():
            panel = self._step_list.get_panel(sid)
            if panel:
                panel.set_enabled(enabled)
        for sid, params in cfg.step_params.items():
            panel = self._step_list.get_panel(sid)
            if panel:
                panel.set_params(params)
        self._step_list.set_order(cfg.step_order)

        # ── Panneaux édition ───────────────────────────────────────────────
        if img is not None:
            # Charger l'image résultat depuis le disque si dispo (sans la stocker en mém.)
            result = self._get_result_from_disk(cfg)
            display = result if result is not None else img
            self._mask_panel.set_image(display, cfg.inpaint_mask)
            self._wb_panel.set_image(display, cfg.wb_pick, cfg.wb_patch_radius)

        # ── Onglet Originale ──────────────────────────────────────────────────
        self._origin_view.set_image(img)

        # ── Onglet Résultat (chargement lazeux si onglet actif) ───────────────
        if self._tabs.currentWidget() is self._dest_view:
            self._update_dest_view(cfg)
        else:
            self._dest_view.set_image(None)

        fname = os.path.basename(cfg.file_path)
        self._statusbar.showMessage(f"{fname}  —  prêt")

        # Mettre à jour les panneaux (mosaïque ou aperçu rapide)
        self._schedule_preview_update()

    def _save_current_state(self) -> None:
        """Sauvegarde l'état courant de l'UI dans _current_cfg."""
        cfg = self._current_cfg
        if cfg is None:
            return
        cfg.step_order      = self._step_list.get_order()
        cfg.step_enabled    = self._step_list.get_enabled()
        cfg.step_params     = self._step_list.get_all_params()
        cfg.inpaint_mask    = self._mask_panel.get_mask()
        cfg.wb_pick         = self._wb_panel.get_pick_point()
        cfg.wb_patch_radius = self._wb_panel.get_patch_radius()
        save_recipe(cfg)

    def _config_for(self, file_path: str) -> Optional[BatchImageConfig]:
        for cfg in self._session.images:
            if cfg.file_path == file_path:
                return cfg
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # Slots — changements dans la liste d'étapes
    # ══════════════════════════════════════════════════════════════════════════

    @pyqtSlot(list)
    def _on_order_changed(self, order: list[str]) -> None:
        if self._current_cfg:
            self._current_cfg.step_order = order
            self._mark_customized()

    @pyqtSlot(str, str, object)
    def _on_param_changed(self, step_id: str, key: str, value) -> None:
        if self._current_cfg:
            self._current_cfg.step_params.setdefault(step_id, {})[key] = value
            # Ne pas marquer customized lors du chargement automatique d'un preset de profil
            panel = self._step_list.get_panel(step_id)
            if panel is None or not panel.is_loading_preset():
                self._mark_customized()
        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview_update()

    @pyqtSlot(str, bool)
    def _on_enabled_changed(self, step_id: str, enabled: bool) -> None:
        if self._current_cfg:
            self._current_cfg.step_enabled[step_id] = enabled
            self._mark_customized()
        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview_update()

    @pyqtSlot(str)
    def _on_mask_edit_requested(self, step_id: str) -> None:
        """Bascule sur l'onglet Masque."""
        self._tabs.setCurrentIndex(0)

    @pyqtSlot(str)
    def _on_color_picker_requested(self, step_id: str) -> None:
        """Bascule sur l'onglet Blanc."""
        self._tabs.setCurrentIndex(1)

    # ══════════════════════════════════════════════════════════════════════════
    # Aperçu rapide — mosaïque ou panneau unique (Masque / Blanc)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Signaux WB ────────────────────────────────────────────────────────────

    @pyqtSlot(int, int)
    def _on_wb_pick_changed(self, x: int, y: int) -> None:
        """Le point de balance des blancs vient d'être déplacé."""
        if self._current_cfg is not None:
            self._current_cfg.wb_pick = (x, y)
            self._mark_customized()
        self._schedule_preview_update()

    @pyqtSlot(int)
    def _on_wb_radius_changed(self, radius: int) -> None:
        """Le rayon du patch WB vient de changer."""
        if self._current_cfg is not None:
            self._current_cfg.wb_patch_radius = radius
            self._mark_customized()
        self._schedule_preview_update()

    # ── Timer debounce ────────────────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_tab_changed(self, index: int) -> None:
        """Au changement d'onglet : charge le résultat si nécessaire, lance l'aperçu."""
        if self._tabs.currentWidget() is self._dest_view and self._current_cfg is not None:
            self._update_dest_view(self._current_cfg)
        self._schedule_preview_update()

    def _schedule_preview_update(self) -> None:
        """Lance le timer debounce (300 ms) pour mettre à jour l'aperçu."""
        if self._current_orig is None:
            return
        self._preview_timer.start()

    def _do_preview_update(self) -> None:
        """Calcule le fast-pipeline si l'onglet actif est Masque ou Blanc."""
        tab = self._tabs.currentWidget()
        if tab in (self._mask_panel, self._wb_panel):
            self._compute_and_show_fast_preview()

    # ── Aperçu rapide (tabs Masque / Blanc) ───────────────────────────────────

    def _compute_and_show_fast_preview(self) -> None:
        """Calcule le fast-pipeline pour le profil courant et met à jour les panneaux."""
        if self._current_orig is None or self._current_cfg is None:
            return
        self._inject_instance_state(self._current_cfg)
        img = self._compute_single_profile_preview()
        if img is not None:
            self._refresh_panels(img)

    def _compute_single_profile_preview(self) -> Optional[np.ndarray]:
        """Exécute les fast steps pour le profil courant (une seule passe)."""
        cfg = self._current_cfg
        if cfg is None or self._current_orig is None:
            return None
        out = self._current_orig.copy()
        ctx: dict = {}
        for sid in cfg.step_order:
            if sid not in _FAST_PREVIEW_IDS:
                continue
            if not cfg.step_enabled.get(sid, True):
                continue
            step  = self._steps_by_id.get(sid)
            panel = self._step_list.get_panel(sid)
            if step is None or panel is None:
                continue
            try:
                result, extras = step.process(out, panel.get_params(), ctx)
                out = result
                ctx.update(extras)
            except Exception:
                pass
        return out

    def _refresh_panels(self, img: np.ndarray) -> None:
        """Met à jour les panneaux Masque et Blanc avec l'image fournie.

        Le point WB courant est récupéré depuis le canvas (l'utilisateur
        peut l'avoir bougé sans qu'on ait encore sauvegardé).
        """
        if self._current_cfg is None:
            return
        cfg = self._current_cfg
        current_pick   = self._wb_panel.get_pick_point()
        current_radius = self._wb_panel.get_patch_radius()
        self._mask_panel.set_image(img, cfg.inpaint_mask)
        self._wb_panel.set_image(img, current_pick, current_radius)

    # ── Helpers image résultat ────────────────────────────────────────────────

    def _get_result_from_disk(self, cfg: BatchImageConfig) -> Optional[np.ndarray]:
        """Charge l'image résultat depuis le dossier de sortie si elle existe.

        Ne stocke rien dans cfg : l'appelant gère la durée de vie du tableau.
        """
        if not self._session.output_dir:
            return None
        path = os.path.join(self._session.output_dir, os.path.basename(cfg.file_path))
        if os.path.exists(path):
            return cv2.imread(path, cv2.IMREAD_COLOR)
        return None

    def _update_dest_view(self, cfg: BatchImageConfig) -> None:
        """Met à jour l'onglet Résultat depuis le disque (chargement à la demande)."""
        result = self._get_result_from_disk(cfg)
        self._dest_view.set_image(result)

    # ══════════════════════════════════════════════════════════════════════════
    # Lancer la sélection
    # ══════════════════════════════════════════════════════════════════════════════

    def _run_on_selection(self) -> None:
        """Lance le batch uniquement sur la sélection d’exécution.

        Si la sélection est vide, utilise l’image courante seule.
        """
        if self._worker and self._worker.isRunning():
            return
        if not self._session.output_dir:
            self._choose_output_dir()
            if not self._session.output_dir:
                return

        self._save_current_state()

        run_paths = self._strip.get_run_selection()
        if not run_paths:
            # Fallback : image courante
            if self._current_cfg is None:
                return
            run_paths = [self._current_cfg.file_path]

        queue: list[BatchImageConfig] = [
            cfg for cfg in self._session.images
            if cfg.file_path in set(run_paths)
        ]
        # Respecter l’ordre de la bande
        order_index = {p: i for i, p in enumerate(run_paths)}
        queue.sort(key=lambda c: order_index.get(c.file_path, 0))

        self._run_batch_from_queue(queue)

    # ══════════════════════════════════════════════════════════════════════════════
    # Appliquer à toutes les images non personnalisées
    # ══════════════════════════════════════════════════════════════════════════════

    def _apply_to_all_uncustomized(self) -> None:
        """Propage step_enabled, step_params et step_order de l’image courante
        vers toutes les images qui n’ont pas été personnalisées."""
        self._save_current_state()
        cfg = self._current_cfg
        if cfg is None:
            return

        targets = [
            c for c in self._session.images
            if c is not cfg and not c.customized
        ]
        if not targets:
            self._statusbar.showMessage("Aucune image non personnalisée à mettre à jour.")
            return

        for target in targets:
            target.step_order   = list(cfg.step_order)
            target.step_enabled = dict(cfg.step_enabled)
            target.step_params  = {k: dict(v) for k, v in cfg.step_params.items()}

        self._statusbar.showMessage(
            f"Paramètres appliqués à {len(targets)} image(s) non personnalisée(s)."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Batch (toutes les images)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_batch(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._session.images:
            return
        if not self._session.output_dir:
            self._choose_output_dir()
            if not self._session.output_dir:
                return

        self._save_current_state()
        self._run_batch_from_queue(list(self._session.images))

    def _run_batch_from_queue(self, queue: list["BatchImageConfig"]) -> None:
        """Démarre un traitement batch sur la liste fournie."""
        self._batch_queue    = queue
        self._batch_step_log: dict[str, list[dict]] = {}
        self._set_buttons_running(True)
        self._process_next_batch()

    def _process_next_batch(self) -> None:
        if not self._batch_queue:
            self._batch_done()
            return
        cfg = self._batch_queue.pop(0)
        cfg.batch_status = "running"
        self._strip.set_running(cfg.file_path, True)
        self._strip.select(cfg.file_path)
        self._navigate_to(cfg)

        img = cv2.imread(cfg.file_path, cv2.IMREAD_COLOR)
        if img is None:
            cfg.batch_status = "error"
            self._strip.set_running(cfg.file_path, False)
            self._statusbar.showMessage(f"✗ Impossible de lire : {cfg.filename}")
            self._process_next_batch()
            return

        self._inject_instance_state(cfg)
        self._start_worker(cfg, img)

    def _batch_done(self) -> None:
        self._set_buttons_running(False)
        n = len(self._session.images)
        self._statusbar.showMessage(f"Batch terminé — {n} image(s) traitée(s)")

    # ══════════════════════════════════════════════════════════════════════════
    # Worker
    # ══════════════════════════════════════════════════════════════════════════

    def _start_worker(
        self,
        cfg:  BatchImageConfig,
        img:  np.ndarray,
    ) -> None:
        enabled_order = [
            sid for sid in cfg.step_order
            if cfg.step_enabled.get(sid, True)
        ]
        steps = [
            self._steps_by_id[sid]
            for sid in enabled_order
            if sid in self._steps_by_id
        ]

        self._worker = PipelineWorker(
            run_id      = 0,
            steps       = steps,
            initial_img = img,
            all_params  = cfg.step_params,
            context     = {},
            step_order  = cfg.step_order,
            step_enabled= cfg.step_enabled,
        )
        self._worker_cfg  = cfg
        self._worker_step_results: dict[str, str] = {}

        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.step_failed.connect(self._on_step_failed)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    @pyqtSlot(str)
    def _on_step_started(self, step_id: str) -> None:
        panel = self._step_list.get_panel(step_id)
        if panel:
            panel.set_state("running")
        step = self._steps_by_id.get(step_id)
        self._statusbar.showMessage(
            f"⏳ {step.name if step else step_id}  —  {self._worker_cfg.filename}"
        )

    @pyqtSlot(str, object, dict)
    def _on_step_done(self, step_id: str, img: np.ndarray, extras: dict) -> None:
        panel = self._step_list.get_panel(step_id)
        if panel:
            panel.set_state("ok")
        self._worker_step_results[step_id] = "ok"

    @pyqtSlot(str, str)
    def _on_step_failed(self, step_id: str, msg: str) -> None:
        panel = self._step_list.get_panel(step_id)
        if panel:
            panel.set_state("error", msg[:60])
        self._worker_step_results[step_id] = "error"
        self._statusbar.showMessage(f"✗ {step_id} : {msg[:80]}")

    @pyqtSlot(object)
    def _on_all_done(self, entry) -> None:
        cfg        = self._worker_cfg
        result_img = (entry.step_results.get(entry.completed_steps[-1])
                      if entry.completed_steps else None)
        cfg.context = entry.context

        if result_img is not None:
            # Miniature (petite copie, pas de problème mémoire)
            self._strip.update_result_thumb(cfg.file_path, result_img)
            # Mettre à jour les panneaux si c'est l'image courante
            if cfg is self._current_cfg:
                self._mask_panel.set_image(result_img, cfg.inpaint_mask)
                self._wb_panel.set_image(result_img, cfg.wb_pick, cfg.wb_patch_radius)
                if self._tabs.currentWidget() is self._dest_view:
                    self._dest_view.set_image(result_img)
            # Ne pas stocker result_img dans cfg — libère la mémoire immédiatement
            # Le résultat sera relu depuis le disque si nécessaire.

        cfg.batch_status = "done"
        self._strip.set_running(cfg.file_path, False)
        self._strip.set_done(cfg.file_path, True)

        # Sidecar résultat JSON
        step_log = build_step_log(
            step_order   = cfg.step_order,
            step_enabled = cfg.step_enabled,
            step_params  = cfg.step_params,
            step_results = self._worker_step_results,
            context      = cfg.context,
            steps_by_id  = self._steps_by_id,
        )
        try:
            self._session.save_result(cfg, step_log)
        except Exception as exc:
            self._statusbar.showMessage(f"Erreur sauvegarde : {exc}")

        self._process_next_batch()

    # ══════════════════════════════════════════════════════════════════════════
    # Injection de l'état des singletons d'étapes
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_panels_with_result(self, cfg: BatchImageConfig) -> None:
        """[inutilisé — conservé pour compatibilité] Appelle set_image depuis le disque."""
        result = self._get_result_from_disk(cfg)
        if result is not None:
            self._mask_panel.set_image(result, cfg.inpaint_mask)
            self._wb_panel.set_image(result, cfg.wb_pick, cfg.wb_patch_radius)

    def _inject_instance_state(self, cfg: BatchImageConfig) -> None:
        """Injecte le masque et le point WB dans les singletons d'étapes."""
        inpaint_step = self._steps_by_id.get("inpaint")
        if inpaint_step and hasattr(inpaint_step, "set_mask"):
            if cfg.inpaint_mask is not None:
                inpaint_step.set_mask(cfg.inpaint_mask)
            else:
                inpaint_step.clear_mask()

        wb_step = self._steps_by_id.get("wb")
        if wb_step and hasattr(wb_step, "set_pick_point"):
            if cfg.wb_pick is not None:
                wb_step.set_pick_point(*cfg.wb_pick)
            else:
                wb_step.clear_pick_point()

    def _clear_instance_state(self) -> None:
        """Réinitialise les singletons à la fermeture."""
        inpaint_step = self._steps_by_id.get("inpaint")
        if inpaint_step and hasattr(inpaint_step, "clear_mask"):
            inpaint_step.clear_mask()
        wb_step = self._steps_by_id.get("wb")
        if wb_step and hasattr(wb_step, "clear_pick_point"):
            wb_step.clear_pick_point()

    # ══════════════════════════════════════════════════════════════════════════
    # Utilitaires UI
    # ══════════════════════════════════════════════════════════════════════════

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Dossier de sortie", self._session.output_dir or ""
        )
        if path:
            self._session.output_dir = path
            self._output_lbl.setText(path)
            self._output_lbl.setToolTip(path)

    def _stop(self) -> None:
        if self._worker:
            self._worker.cancel()
        # Vider la file d'attente batch
        if hasattr(self, "_batch_queue"):
            self._batch_queue.clear()
        self._set_buttons_running(False)
        self._statusbar.showMessage("Arrêté.")

    def _mark_customized(self) -> None:
        cfg = self._current_cfg
        if cfg and not cfg.customized:
            cfg.customized = True
            self._strip.set_customized(cfg.file_path, True)

    def _set_buttons_running(self, running: bool) -> None:
        self._selection_btn.setEnabled(not running)
        self._batch_btn.setEnabled(not running)
        self._apply_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)

    def _on_strip_image_reset(self, file_path: str) -> None:
        cfg = self._config_for(file_path)
        if cfg is None:
            return
        self._session.reset_to_defaults(cfg)
        self._strip.set_customized(file_path, False)
        self._strip.set_done(file_path, False)
        if self._current_cfg is cfg:
            self._navigate_to(cfg)
        self._statusbar.showMessage(f"{os.path.basename(file_path)} — réinitialisé.")

    @staticmethod
    def _style_btn(btn: QPushButton, accent: bool = False) -> None:
        if accent:
            btn.setStyleSheet(
                "QPushButton { background:#2a6496; color:#fff; border-radius:4px;"
                "  padding:6px 14px; font-size:11px; font-weight:600; }"
                "QPushButton:hover { background:#3a74a6; }"
                "QPushButton:disabled { background:#333; color:#555; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
                "  padding:6px 12px; font-size:11px; }"
                "QPushButton:hover { background:#2a2a50; color:#ccc; }"
                "QPushButton:disabled { background:#1a1a2a; color:#444; }"
            )

    def _apply_theme(self) -> None:
        self.setStyleSheet("QMainWindow { background:#12122a; }")

    # ══════════════════════════════════════════════════════════════════════════
    # Fermeture
    # ══════════════════════════════════════════════════════════════════════════

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Batch en cours",
                "Un traitement est en cours. Voulez-vous l'arrêter et fermer ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop()

        self._save_current_state()
        self._clear_instance_state()
        self.closed.emit()
        super().closeEvent(event)
