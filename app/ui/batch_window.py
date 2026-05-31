"""ui/batch_window.py — Fenêtre de traitement par lots (mode batch).

Layout ::

    ┌── Toolbar: [Sortie: /...][Parcourir][▶ Lancer la sélection][↓ Appliquer à toutes][⚡ Lancer le batch] ───────┐
    ├────────── BatchThumbnailStrip ──────────────────────────────────────────────┤
    │ StepListWidget  │       ImageView (SyncedImageView)     │  QTabWidget      │
    │ (≈260 px)       │                                       │  [Masque][Blanc] │
    │                 │            (expanding)                │  MaskCanvasPanel │
    │                 │                                       │  WBPickerPanel   │
    └─────────────────┴───────────────────────────────────────┴──────────────────┘

Chaque image possède sa propre configuration (step_order, params, masque, WB).

Logique métier découpée en mixins (ui/batch_mixins/) :
    NavMixin     — navigation entre images, sauvegarde, injection état
    ParamsMixin  — paramètres, undo/redo, propagation
    PreviewMixin — aperçu rapide, preview complet, onglets, overlays
    RunMixin     — exécution pipeline (sélection, batch, worker)
    ExportsMixin — exports versionnés, mode lecture seule, diff JSON
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSplitter, QScrollArea, QTabWidget,
    QStatusBar, QSizePolicy, QMessageBox, QFileDialog, QProgressDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QUndoStack, QKeySequence, QShortcut

from core.pipeline import PipelineWorker
from core.batch import BatchSession, BatchImageConfig
from steps import ALL_STEPS
from ui.step_panel import StepListWidget
from ui.batch_thumbnail_strip import BatchThumbnailStrip
from ui.mask_editor import MaskCanvasPanel
from ui.wb_picker import WBPickerPanel
from ui.image_view import SyncedImageView
from ui.export_mosaic import ExportMosaicView
from ui.export_detail_panel import ExportDetailPanel
from ui.notifications import NotificationManager, Level
from ui.batch_window_constants import (
    _FAST_PREVIEW_IDS, _PREVIEW_TABS,
    _TAB_PREVIEW, _TAB_MASK, _TAB_WB, _TAB_REDEYE,
    _TAB_ORIGIN, _TAB_RESULT,
)
from core.export_manager import ExportManager, ExportEntry
from ui.batch_mixins.nav import NavMixin
from ui.batch_mixins.params import ParamsMixin
from ui.batch_mixins.preview import PreviewMixin
from ui.batch_mixins.run import RunMixin
from ui.batch_mixins.exports import ExportsMixin


class BatchWindow(
    RunMixin,
    ExportsMixin,
    PreviewMixin,
    ParamsMixin,
    NavMixin,
    QMainWindow,
):
    """Fenêtre de traitement par lots."""

    closed = pyqtSignal()

    def __init__(self, parent, session: BatchSession) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch — Traitement par lots")
        self.resize(1600, 950)

        self._session     = session
        self._steps_by_id = {s.id: s for s in ALL_STEPS}
        self._current_cfg: Optional[BatchImageConfig] = None
        self._worker:      Optional[PipelineWorker]   = None

        self._current_orig: Optional[np.ndarray] = None

        # Aperçu rapide : timer debounce
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._do_preview_update)

        # Undo/Redo
        self._undo_stack = QUndoStack(self)
        self._preset_buf: list = []
        self._preset_flush_pending = False
        self._applying_order = False
        self._applying_enabled = False
        self._applying_export_view = False

        # Zoom partagé entre onglets
        self._prev_tab_index: int = 0
        self._shared_zoom: Optional[tuple] = None

        # Compteurs batch
        self._batch_run_total: int  = 0
        self._batch_run_done:  int  = 0
        self._batch_run_start: float = 0.0
        self._image_start_time: float = 0.0

        # Export visionné
        self._is_viewing_export:  bool          = False
        self._viewed_export_entry = None  # ExportEntry | None
        self._viewed_export_list: list = []  # list[ExportEntry]

        # Preview complet
        self._full_preview_worker: Optional[PipelineWorker] = None
        self._preview_full_img:    Optional[np.ndarray]     = None
        self._preview_overlay_step: Optional[str]           = None
        self._viewed_export_data:  Optional[dict]           = None
        self._preview_stale:       bool                     = True
        self._last_fast_preview:   Optional[np.ndarray]     = None

        self._build_ui()
        self._apply_theme()

        self._notif = NotificationManager(self)

        # Raccourcis clavier
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo)

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

        root.addWidget(self._build_toolbar())

        self._strip = BatchThumbnailStrip()
        self._strip.image_selected.connect(self._on_strip_image_selected)
        self._strip.image_reset.connect(self._on_strip_image_reset)
        root.addWidget(self._strip)

        root.addWidget(self._build_main_area(), stretch=1)

        self._statusbar = QStatusBar()
        self._statusbar.setStyleSheet("background:#111124; color:#888; font-size:10px;")
        self.setStatusBar(self._statusbar)

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
        self._output_lbl.setStyleSheet("color:#9de; font-size:11px; max-width:400px;")
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
            "Propager les paramètres de l'image courante vers les images non personnalisées"
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

        lay.addSpacing(12)

        self._export_best_btn = QPushButton("📤  Exporter les retenus")
        self._style_btn(self._export_best_btn)
        self._export_best_btn.setToolTip(
            "Exporter les meilleurs exports (ou derniers) en JPEG vers un répertoire"
        )
        self._export_best_btn.clicked.connect(self._export_all_best)
        lay.addWidget(self._export_best_btn)

        return bar

    def _build_main_area(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background:#1a1a2e; }")

        # ── Panneau gauche : liste des étapes ─────────────────────────────────
        step_container = QWidget()
        step_container.setMinimumWidth(240)
        step_container.setMaximumWidth(340)
        step_container.setStyleSheet("background:#16162a;")
        scl = QVBoxLayout(step_container)
        scl.setContentsMargins(0, 0, 0, 0)
        scl.setSpacing(0)

        hdr_w = QWidget()
        hdr_w.setFixedHeight(32)
        hdr_w.setStyleSheet("background:#1e1e38; border-bottom:1px solid #2a2a4a;")
        hdr_lay = QHBoxLayout(hdr_w)
        hdr_lay.setContentsMargins(12, 0, 4, 0)
        hdr_lay.setSpacing(4)
        hdr_lbl = QLabel("Étapes — par image")
        hdr_lbl.setStyleSheet("color:#ddd; font-size:12px; font-weight:700;")
        hdr_lay.addWidget(hdr_lbl, stretch=1)

        self._reload_recipe_btn = QPushButton("↺")
        self._reload_recipe_btn.setFixedSize(24, 24)
        self._reload_recipe_btn.setToolTip(
            "Recharger depuis le fichier .recipe.json sur disque\n"
            "(annule les modifications non sauvegardées)"
        )
        self._reload_recipe_btn.setStyleSheet(
            "QPushButton { background:#2a2a4a; color:#9de; border-radius:4px;"
            "  font-size:14px; }"
            "QPushButton:hover { background:#3a3a6a; color:#cef; }"
        )
        self._reload_recipe_btn.clicked.connect(self._reload_from_disk)
        hdr_lay.addWidget(self._reload_recipe_btn)
        scl.addWidget(hdr_w)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none; background:#16162a;")

        self._step_list = StepListWidget()
        self._step_list.add_steps(ALL_STEPS)
        self._step_list.set_batch_mode(True)
        self._step_list.order_changed.connect(self._on_order_changed)
        self._step_list.param_changed.connect(self._on_param_changed)
        self._step_list.param_propagate_requested.connect(self._on_param_propagate)
        self._step_list.enabled_changed.connect(self._on_enabled_changed)
        self._step_list.enabled_propagate_requested.connect(self._on_enabled_propagate)
        self._step_list.order_reordered.connect(self._on_order_reordered)
        self._step_list.mask_edit_requested.connect(self._on_mask_edit_requested)
        self._step_list.color_picker_requested.connect(self._on_color_picker_requested)
        self._step_list.overlay_toggled.connect(self._on_overlay_toggled)
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
        rl.setSpacing(0)

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

        _SIDEBAR_W = 240
        _dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        self._mask_panel = MaskCanvasPanel(_dummy, None, show_ok_cancel=False,
                                           sidebar_width=_SIDEBAR_W)
        self._wb_panel   = WBPickerPanel(
            _dummy, None, show_ok_cancel=False, sidebar_width=_SIDEBAR_W
        )
        self._redeye_panel = MaskCanvasPanel(_dummy, None, show_ok_cancel=False,
                                              sidebar_width=_SIDEBAR_W)

        self._origin_view = SyncedImageView()

        # Onglet Preview
        self._preview_view = SyncedImageView()
        preview_sidebar = QWidget()
        preview_sidebar.setFixedWidth(_SIDEBAR_W)
        preview_sidebar.setStyleSheet("background:#16162a;")
        ps_lay = QVBoxLayout(preview_sidebar)
        ps_lay.setContentsMargins(8, 12, 8, 12)
        ps_lay.setSpacing(10)

        self._preview_status_lbl = QLabel("Preview rapide")
        self._preview_status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_status_lbl.setWordWrap(True)
        self._preview_status_lbl.setStyleSheet(
            "color:#9ab; font-size:10px; border:1px solid #2a2a4a;"
            " border-radius:3px; padding:4px; background:#111124;"
        )
        ps_lay.addWidget(self._preview_status_lbl)

        self._full_preview_btn = QPushButton("⚡ Preview complet")
        self._full_preview_btn.setToolTip(
            "Exécute l'intégralité du pipeline (toutes les étapes activées)\n"
            "sans sauvegarder le résultat."
        )
        self._full_preview_btn.setStyleSheet(
            "QPushButton { background:#1e2a1e; color:#6e8; border:1px solid #3a5a3a;"
            "  border-radius:4px; padding:5px 8px; font-size:11px; }"
            "QPushButton:hover { background:#2a3a2a; color:#aea; }"
            "QPushButton:disabled { color:#445; border-color:#223; }"
        )
        self._full_preview_btn.clicked.connect(self._run_full_preview)
        ps_lay.addWidget(self._full_preview_btn)
        ps_lay.addStretch()

        preview_wrapper = QWidget()
        pw_lay = QHBoxLayout(preview_wrapper)
        pw_lay.setContentsMargins(0, 0, 0, 0)
        pw_lay.setSpacing(0)
        pw_lay.addWidget(self._preview_view, stretch=1)
        pw_lay.addWidget(preview_sidebar)

        self._tabs.addTab(preview_wrapper,                                "Preview")       # _TAB_PREVIEW
        self._tabs.addTab(self._mask_panel,                               "Masque")        # _TAB_MASK
        self._tabs.addTab(self._wb_panel,                                 "Blanc")         # _TAB_WB
        self._tabs.addTab(self._redeye_panel,                             "Yeux rouges")   # _TAB_REDEYE
        self._tabs.addTab(self._wrap_with_sidebar(self._origin_view, _SIDEBAR_W), "Originale")  # _TAB_ORIGIN

        # ── Onglet Résultat : mosaïque d'exports ──
        self._export_mosaic = ExportMosaicView()
        self._export_mosaic.export_selected.connect(self._on_mosaic_export_selected)
        self._export_mosaic.export_activated.connect(self._on_mosaic_export_activated)
        result_wrapper = QWidget()
        result_lay = QHBoxLayout(result_wrapper)
        result_lay.setContentsMargins(0, 0, 0, 0)
        result_lay.setSpacing(0)
        result_lay.addWidget(self._export_mosaic, stretch=1)
        # Panneau détail export (sidebar, toujours visible)
        self._export_detail = ExportDetailPanel()
        self._export_detail.restore_requested.connect(self._on_detail_restore)
        self._export_detail.export_deleted.connect(self._on_detail_deleted)
        self._export_detail.best_changed.connect(self._on_detail_best_changed)
        self._export_detail.nav_switch_requested.connect(self._on_detail_nav_switch)
        result_lay.addWidget(self._export_detail)
        self._tabs.addTab(result_wrapper, "Résultat")   # _TAB_RESULT

        self._tabs.currentChanged.connect(self._on_tab_changed)
        rl.addWidget(self._tabs)

        self._wb_panel._canvas.pick_changed.connect(self._on_wb_pick_changed)
        self._wb_panel._rad_slider.valueChanged.connect(self._on_wb_radius_changed)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1200])
        return splitter

    @staticmethod
    def _wrap_with_sidebar(view: QWidget, sidebar_width: int) -> QWidget:
        wrapper = QWidget()
        lay = QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(view, stretch=1)
        spacer = QWidget()
        spacer.setFixedWidth(sidebar_width)
        spacer.setStyleSheet("background: #1a1a2e;")
        lay.addWidget(spacer)
        return wrapper

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers image résultat
    # ══════════════════════════════════════════════════════════════════════════

    def _get_result_from_disk(self, cfg: BatchImageConfig) -> Optional[np.ndarray]:
        """Charge l'image du dernier export versionné depuis output_dir."""
        if not self._session.output_dir:
            return None
        mgr = self._session.get_export_manager()
        stem, ext = os.path.splitext(cfg.filename)
        exports = mgr.list_exports(stem, ext)
        if not exports:
            return None
        return mgr.load_image(exports[-1])

    def _update_dest_view(self, cfg: BatchImageConfig, force: bool = False) -> None:
        """Met à jour la mosaïque d'exports pour l'image courante.

        Si force=False (tab switch), préserve l'état (fullscreen, sélection).
        Si force=True (nouvelle image, nouvel export), reset complet.
        """
        if not self._session.output_dir:
            self._export_mosaic.set_exports([])
            return
        mgr = self._session.get_export_manager()
        stem, ext = os.path.splitext(cfg.filename)
        exports = mgr.list_exports(stem, ext)
        best_idx = mgr.get_best_index(stem)
        if force:
            self._export_mosaic.set_exports(exports)
        else:
            self._export_mosaic.update_exports_if_changed(exports)
        self._export_mosaic.set_best_index(best_idx)
        self._export_detail.set_best_index(best_idx)

    # ── Signaux mosaïque ─────────────────────────────────────────────────────

    def _on_mosaic_export_selected(self, entry) -> None:
        """Un export a été sélectionné dans la mosaïque."""
        mgr = self._session.get_export_manager()
        self._export_detail.set_export_manager(mgr)
        # Propager le best index courant
        if self._current_cfg:
            stem = os.path.splitext(self._current_cfg.filename)[0]
            self._export_detail.set_best_index(mgr.get_best_index(stem))
        self._export_detail.set_entry(
            entry,
            all_entries=self._export_mosaic._entries,
        )

    def _on_mosaic_export_activated(self, entry) -> None:
        """Double-clic sur un export → mode plein format."""
        pass  # Le basculement est géré par ExportMosaicView

    def _on_detail_restore(self, entry) -> None:
        """Restaure les paramètres depuis un export sélectionné."""
        from core.batch import _apply_recipe
        cfg = self._current_cfg
        if cfg is None or entry.recipe_data is None:
            return
        mgr = self._session.get_export_manager()
        data = dict(entry.recipe_data)
        mask = mgr.load_mask(entry)
        if mask is not None:
            data["_mask"] = mask
        redeye = mgr.load_redeye_mask(entry)
        if redeye is not None:
            data["_redeye_mask"] = redeye
        _apply_recipe(cfg, data)
        # Rafraîchir l'UI
        self._applying_order = True
        for sid, enabled in cfg.step_enabled.items():
            panel = self._step_list.get_panel(sid)
            if panel:
                panel.set_enabled(enabled)
        for sid, params in cfg.step_params.items():
            panel = self._step_list.get_panel(sid)
            if panel:
                panel.set_params(params)
        self._step_list.set_order(cfg.step_order)
        self._applying_order = False
        # Restaurer la position WB dans l'éditeur
        if cfg.wb_pick:
            self._wb_panel._canvas.set_pick_point(cfg.wb_pick)
            self._wb_panel._refresh_pick_info(*cfg.wb_pick)
        else:
            self._wb_panel._clear_pick()
        if cfg.wb_patch_radius:
            self._wb_panel._rad_slider.setValue(cfg.wb_patch_radius)
            self._wb_panel._rad_val_lbl.setText(f"{cfg.wb_patch_radius} px")
        self._statusbar.showMessage("Configuration restaurée depuis l'export.")
        self._schedule_preview_update()

    def _on_detail_deleted(self, entry) -> None:
        """Un export a été supprimé — rafraîchir la mosaïque."""
        if self._current_cfg:
            self._update_dest_view(self._current_cfg, force=True)
            self._refresh_export_dropdown(self._current_cfg)

    def _on_detail_best_changed(self, entry, index) -> None:
        """Un export a été marqué comme retenu."""
        cfg = self._current_cfg
        if not cfg:
            return
        mgr = self._session.get_export_manager()
        stem = os.path.splitext(cfg.filename)[0]
        mgr.set_best(stem, index)
        self._export_mosaic.set_best_index(index)
        self._export_detail.set_best_index(index)

    def _on_detail_nav_switch(self, entry) -> None:
        """Navigation vers un export via les pills de l'onglet Résultat."""
        mgr = self._session.get_export_manager()
        self._export_detail.set_export_manager(mgr)
        if self._current_cfg:
            stem = os.path.splitext(self._current_cfg.filename)[0]
            self._export_detail.set_best_index(mgr.get_best_index(stem))
        self._export_detail.set_entry(
            entry,
            all_entries=self._export_mosaic._entries,
        )
        self._export_mosaic.switch_to_entry(entry)

    def _export_all_best(self) -> None:
        """Exporte les meilleurs exports (ou derniers) en JPEG vers un répertoire."""
        if not self._session.output_dir:
            QMessageBox.warning(
                self, "Export impossible",
                "Aucun répertoire de sortie configuré."
            )
            return

        dest = QFileDialog.getExistingDirectory(
            self, "Répertoire de destination JPEG"
        )
        if not dest:
            return

        mgr = self._session.get_export_manager()
        configs = self._session.images
        if not configs:
            return

        # Collecter les paires (stem, ext) à exporter
        items: list[tuple[str, str, str]] = []  # (stem, ext, original_filename)
        for cfg in configs:
            stem, ext = os.path.splitext(cfg.filename)
            best = mgr.get_best_entry(stem, ext)
            if best and os.path.exists(best.image_path):
                items.append((stem, best.image_path, cfg.filename))

        if not items:
            self._statusbar.showMessage("Aucun export à exporter.")
            return

        progress = QProgressDialog(
            "Export JPEG en cours…", "Annuler", 0, len(items), self
        )
        progress.setWindowTitle("Export JPEG")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setValue(0)
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        exported = 0
        for i, (stem, src_path, original_name) in enumerate(items):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            progress.setLabelText(f"{stem}.jpg  ({i + 1}/{len(items)})")
            QApplication.processEvents()

            img = cv2.imread(src_path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            jpg_name = stem + ".jpg"
            jpg_path = os.path.join(dest, jpg_name)
            cv2.imwrite(
                jpg_path, img,
                [cv2.IMWRITE_JPEG_QUALITY, 95,
                 cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                 cv2.IMWRITE_JPEG_PROGRESSIVE, 1],
            )
            exported += 1

        progress.setValue(len(items))
        self._statusbar.showMessage(
            f"Export terminé : {exported}/{len(items)} images JPEG → {dest}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Utilitaires UI
    # ══════════════════════════════════════════════════════════════════════════

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
        self._session.save_session_meta()
        self._clear_instance_state()
        self.closed.emit()
        super().closeEvent(event)
