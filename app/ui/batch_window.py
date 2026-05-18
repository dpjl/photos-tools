"""ui/batch_window.py — Fenêtre de traitement par lots (mode batch).

Layout ::

    ┌── Toolbar: [Sortie: /...][Parcourir][▶ Tester][⚡ Lancer le batch] ──────┐
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
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QObject
from PyQt6.QtGui import QCloseEvent

from core.pipeline import PipelineWorker
from core.batch import BatchSession, BatchImageConfig, build_step_log
from steps import ALL_STEPS
from ui.step_panel import StepListWidget
from ui.batch_thumbnail_strip import BatchThumbnailStrip
from ui.mask_editor import MaskCanvasPanel
from ui.wb_picker import WBPickerPanel


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

        # Étapes auxquelles les résultats intermédiaires appartiennent (Tester)
        self._test_step_results: dict[str, np.ndarray] = {}

        # La fenêtre doit se souvenir de l'image originale (pour masque/WB)
        self._current_orig: Optional[np.ndarray] = None

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

        self._test_btn = QPushButton("▶  Tester")
        self._style_btn(self._test_btn, accent=True)
        self._test_btn.setToolTip("Traiter l'image courante sans sauvegarder")
        self._test_btn.clicked.connect(self._run_test)
        lay.addWidget(self._test_btn)

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

        # ── Panneau droit : onglets Masque / Blanc (prend tout l'espace restant) ─
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
        self._tabs.addTab(self._mask_panel, "Masque")
        self._tabs.addTab(self._wb_panel,   "Blanc")
        rl.addWidget(self._tabs)
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
            display = cfg.result_img if cfg.result_img is not None else img
            self._mask_panel.set_image(display, cfg.inpaint_mask)
            self._wb_panel.set_image(display, cfg.wb_pick, cfg.wb_patch_radius)

        fname = os.path.basename(cfg.file_path)
        self._statusbar.showMessage(f"{fname}  —  prêt")

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
            self._mark_customized()

    @pyqtSlot(str, bool)
    def _on_enabled_changed(self, step_id: str, enabled: bool) -> None:
        if self._current_cfg:
            self._current_cfg.step_enabled[step_id] = enabled
            self._mark_customized()

    @pyqtSlot(str)
    def _on_mask_edit_requested(self, step_id: str) -> None:
        """Bascule sur l'onglet Masque."""
        self._tabs.setCurrentIndex(0)

    @pyqtSlot(str)
    def _on_color_picker_requested(self, step_id: str) -> None:
        """Bascule sur l'onglet Blanc."""
        self._tabs.setCurrentIndex(1)

    # ══════════════════════════════════════════════════════════════════════════
    # Test (image courante seulement)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_test(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        if self._current_cfg is None:
            return
        self._save_current_state()
        cfg = self._current_cfg
        img = cv2.imread(cfg.file_path, cv2.IMREAD_COLOR)
        if img is None:
            self._statusbar.showMessage("Impossible de lire l'image.")
            return
        self._inject_instance_state(cfg)
        self._start_worker(cfg, img, mode="test")

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

        # Sauvegarder l'état de l'image courante avant de lancer
        self._save_current_state()

        self._batch_queue = list(self._session.images)
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
        self._start_worker(cfg, img, mode="batch")

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
        mode: str,   # "test" | "batch"
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
        self._worker_mode = mode
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
        cfg  = self._worker_cfg
        mode = self._worker_mode

        cfg.result_img = entry.step_results.get(
            entry.completed_steps[-1]
        ) if entry.completed_steps else None
        cfg.context = entry.context

        if cfg.result_img is not None:
            self._strip.update_result_thumb(cfg.file_path, cfg.result_img)
            # Afficher le résultat dans les panneaux d'édition
            self._refresh_panels_with_result(cfg)

        if mode == "batch":
            cfg.batch_status = "done"
            self._strip.set_running(cfg.file_path, False)
            self._strip.set_done(cfg.file_path, True)

            # Sidecar JSON
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
        else:
            # mode "test"
            self._set_buttons_running(False)
            fname = os.path.basename(cfg.file_path)
            self._statusbar.showMessage(
                f"Test terminé — {fname}  ({len(entry.completed_steps)} étape(s))"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Injection de l'état des singletons d'étapes
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_panels_with_result(self, cfg: BatchImageConfig) -> None:
        """Met à jour les panneaux Masque et Blanc avec l'image résultat."""
        result = cfg.result_img
        if result is None:
            return
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
        self._test_btn.setEnabled(not running)
        self._batch_btn.setEnabled(not running)
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
