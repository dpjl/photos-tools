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
import json
import os
import time
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSplitter, QScrollArea, QTabWidget,
    QFileDialog, QStatusBar, QSizePolicy, QMessageBox, QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal, QObject
from PyQt6.QtGui import QCloseEvent, QUndoStack, QKeySequence, QShortcut

from core.pipeline import PipelineWorker
from core.batch import (
    BatchSession, BatchImageConfig, build_step_log,
    save_recipe, load_recipe, _apply_recipe,
    list_export_recipes, load_export_recipe,
)
from steps import ALL_STEPS
from ui.step_panel import StepListWidget
from ui.batch_thumbnail_strip import BatchThumbnailStrip
from ui.mask_editor import MaskCanvasPanel
from ui.wb_picker import WBPickerPanel
from ui.image_view import SyncedImageView
from ui.json_diff_panel import JsonDiffPanel
from ui.notifications import NotificationManager, Level
from ui.param_history import SetParamCommand, PropagateParamCommand, PropagateEnabledCommand, MoveStepCommand, ToggleStepCommand

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

        # Undo/Redo — pile locale à l'image courante
        self._undo_stack = QUndoStack(self)
        # Buffer pour regrouper les changements de preset en une macro
        self._preset_buf: list = []
        self._preset_flush_pending = False
        # Flag pour éviter de push une 2ème commande lors du redo/undo d'un réordre
        self._applying_order = False
        # Flag pour éviter de push lors du redo/undo d'activation
        self._applying_enabled = False
        # État de zoom partagé entre les onglets
        self._prev_tab_index: int = 0
        self._shared_zoom: Optional[tuple] = None

        # Compteurs de temps pour les notifications batch
        self._batch_run_total: int  = 0    # total images dans le run courant
        self._batch_run_done:  int  = 0    # images terminées dans le run courant
        self._batch_run_start: float = 0.0 # timestamp début du run
        self._image_start_time: float = 0.0  # timestamp début de l'image courante

        self._build_ui()
        self._apply_theme()

        # Système de notifications flottantes
        # Ancre = QMainWindow lui-même : les toasts sont indépendants de l'UI interne
        self._notif = NotificationManager(self)

        # Raccourcis undo/redo : route vers le bon canvas selon l'onglet actif
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)

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

        # Combo exports
        self._export_combo = QComboBox()
        self._export_combo.setMaximumWidth(140)
        self._export_combo.setToolTip(
            "Visualiser une configuration exportée précédente (lecture seule)"
        )
        self._export_combo.setStyleSheet(
            "QComboBox { background:#1a1a30; color:#9de; border:1px solid #2a2a4a;"
            "  border-radius:3px; padding:2px 4px; font-size:10px; }"
            "QComboBox::drop-down { border:none; }"
            "QComboBox QAbstractItemView { background:#1a1a30; color:#9de;"
            "  selection-background-color:#2a3a5a; border:1px solid #3a3a6a; }"
        )
        self._export_combo.currentIndexChanged.connect(self._on_export_selected)
        hdr_lay.addWidget(self._export_combo)

        # Bouton restaurer
        self._restore_export_btn = QPushButton("↩")
        self._restore_export_btn.setFixedSize(24, 24)
        self._restore_export_btn.setVisible(False)
        self._restore_export_btn.setToolTip(
            "Restaurer cette configuration en mode édition\n"
            "(chargement mémoire uniquement — sauvegarder ensuite si souhaité)"
        )
        self._restore_export_btn.setStyleSheet(
            "QPushButton { background:#2a3a2a; color:#6e8; border-radius:4px;"
            "  font-size:14px; }"
            "QPushButton:hover { background:#3a4a3a; color:#aea; }"
        )
        self._restore_export_btn.clicked.connect(self._restore_export)
        hdr_lay.addWidget(self._restore_export_btn)

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
        self._step_list.set_batch_mode(True)  # active les boutons ⬇ de propagation
        self._step_list.order_changed.connect(self._on_order_changed)
        self._step_list.param_changed.connect(self._on_param_changed)
        self._step_list.param_propagate_requested.connect(self._on_param_propagate)
        self._step_list.enabled_changed.connect(self._on_enabled_changed)
        self._step_list.enabled_propagate_requested.connect(self._on_enabled_propagate)
        self._step_list.order_reordered.connect(self._on_order_reordered)
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
        _SIDEBAR_W = 215  # largeur unifiée pour tous les onglets
        _dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        self._mask_panel = MaskCanvasPanel(_dummy, None, show_ok_cancel=False,
                                           sidebar_width=_SIDEBAR_W)
        self._wb_panel   = WBPickerPanel(
            _dummy, None, show_ok_cancel=False, sidebar_width=_SIDEBAR_W
        )
        # Onglets d'aperçu image (zoom/pan via SyncedImageView)
        self._origin_view = SyncedImageView()
        self._dest_view   = SyncedImageView()
        # Sync bidirectionnelle désactivée : gérée manuellement par _shared_zoom
        # self._origin_view.add_peer(self._dest_view)

        self._tabs.addTab(self._mask_panel,                               "Masque")    # 0
        self._tabs.addTab(self._wb_panel,                                 "Blanc")     # 1
        self._tabs.addTab(self._wrap_with_sidebar(self._origin_view, _SIDEBAR_W), "Originale")  # 2
        self._tabs.addTab(self._wrap_with_sidebar(self._dest_view,   _SIDEBAR_W), "Résultat")   # 3

        # Onglets diff JSON
        self._diff_source_panel = JsonDiffPanel()
        self._diff_source_panel.set_refresh_fn(self._refresh_diff_source)
        self._tabs.addTab(self._diff_source_panel, "Δ Exports")   # 4
        self._tabs.currentChanged.connect(self._on_tab_changed)
        rl.addWidget(self._tabs)

        # Connecter les changements du point WB et du rayon pour déclencher l'aperçu
        self._wb_panel._canvas.pick_changed.connect(self._on_wb_pick_changed)
        self._wb_panel._rad_slider.valueChanged.connect(self._on_wb_radius_changed)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1200])
        return splitter

    @staticmethod
    def _wrap_with_sidebar(view: QWidget, sidebar_width: int) -> QWidget:
        """Enveloppe view dans un QWidget avec un spacer fixe à droite de même largeur
        que les sidebars Masque/Blanc, pour que l'espace image soit identique partout."""
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
        # Réinitialiser le zoom partagé → chaque nouvelle image commence au fit
        self._shared_zoom = None
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
        if self._tabs.currentIndex() == 3:
            self._update_dest_view(cfg)
        else:
            self._dest_view.set_image(None)

        fname = os.path.basename(cfg.file_path)
        self._statusbar.showMessage(f"{fname}  —  prêt")

        # Réinitialiser le mode export et recharger le dropdown
        self._set_viewing_export(None, _navigate_call=True)
        self._refresh_export_dropdown(cfg)

        # Mettre à jour les panneaux (mosaïque ou aperçu rapide)
        self._schedule_preview_update()
        # Actualiser les onglets diff si actifs
        self._refresh_if_diff_tab()

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
        # Vider la pile undo : chaque image a sa propre histoire de changements
        self._undo_stack.clear()
        self._preset_buf.clear()
        self._preset_flush_pending = False

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
        if not self._applying_order and self._current_cfg:
            self._current_cfg.step_order = order
            self._mark_customized()

    @pyqtSlot(list, list)
    def _on_order_reordered(self, old_order: list, new_order: list) -> None:
        """Push une commande undo pour le réordonnancement."""
        if self._applying_order:
            return
        cfg = self._current_cfg
        def apply(order: list) -> None:
            self._applying_order = True
            self._step_list.set_order(order)
            if cfg:
                cfg.step_order = order
            self._applying_order = False
        cmd = MoveStepCommand(
            step_id   = new_order[0] if new_order else "",
            old_order = old_order,
            new_order = new_order,
            apply_fn  = apply,
        )
        self._undo_stack.push(cmd)

    @pyqtSlot(str, str, object)
    def _on_param_changed(self, step_id: str, key: str, value) -> None:
        if self._current_cfg:
            # Capturer old_val AVANT mise à jour (step_params sert de snapshot)
            old_val = self._current_cfg.step_params.get(step_id, {}).get(key)
            self._current_cfg.step_params.setdefault(step_id, {})[key] = value
            # Ne pas marquer customized lors du chargement automatique d'un preset de profil
            panel = self._step_list.get_panel(step_id)
            is_preset = panel is not None and panel.is_loading_preset()
            if not is_preset:
                self._mark_customized()

            # Enregistrer dans la pile undo
            if is_preset:
                self._preset_buf.append((step_id, key, old_val, value))
                if not self._preset_flush_pending:
                    self._preset_flush_pending = True
                    from PyQt6.QtCore import QTimer as _QTimer
                    _QTimer.singleShot(0, self._flush_preset_undo)
            else:
                if self._preset_buf:
                    self._flush_preset_undo()
                cmd = SetParamCommand(
                    self._apply_param_silent, step_id, key, old_val, value
                )
                self._undo_stack.push(cmd)

        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview_update()

    def _apply_param_silent(self, step_id: str, key: str, val) -> None:
        """Applique val silencieusement (undo/redo) : widget + cfg.step_params + preview."""
        if self._current_cfg:
            self._current_cfg.step_params.setdefault(step_id, {})[key] = val
        panel = self._step_list.get_panel(step_id)
        if panel:
            row = panel._param_rows.get(key)
            if row is not None:
                row.set_value(val)  # _block=True, pas de signal
        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview_update()

    def _flush_preset_undo(self) -> None:
        """Vide le buffer preset et pousse une macro undo regroupant tous les changements."""
        self._preset_flush_pending = False
        buf = self._preset_buf
        self._preset_buf = []
        if not buf:
            return
        self._undo_stack.beginMacro("Changer profil")
        for sid, key, old, new in buf:
            cmd = SetParamCommand(self._apply_param_silent, sid, key, old, new)
            self._undo_stack.push(cmd)  # redo() immédiat → no-op (_first=True)
        self._undo_stack.endMacro()

    @pyqtSlot(str, str, object)
    def _on_param_propagate(self, step_id: str, key: str, value) -> None:
        """Propage un seul paramètre aux images sélectionnées (ou toutes si aucune)."""
        self._save_current_state()

        # Déterminer les images cibles
        run_paths = self._strip.get_run_selection()
        if run_paths:
            targets = [c for c in self._session.images if c.file_path in set(run_paths)]
        else:
            targets = list(self._session.images)

        if not targets:
            return

        # Snapshot des anciennes valeurs avant propagation
        old_vals = {
            cfg.file_path: cfg.step_params.get(step_id, {}).get(key)
            for cfg in targets
        }

        # Appliquer immédiatement
        for cfg in targets:
            cfg.step_params.setdefault(step_id, {})[key] = value
            save_recipe(cfg)

        # Mettre à jour l'UI si l'image courante est dans les cibles
        if self._current_cfg in targets:
            panel = self._step_list.get_panel(step_id)
            if panel:
                row = panel._param_rows.get(key)
                if row is not None:
                    row.set_value(value)
            if step_id in _FAST_PREVIEW_IDS:
                self._schedule_preview_update()

        # Pousser la commande undo (le _first=True fait que redo() est no-op au push)
        cmd = PropagateParamCommand(
            step_id, key, value,
            targets, old_vals,
            lambda: self._current_cfg,
            self._apply_param_silent,
            save_recipe,
        )
        self._undo_stack.push(cmd)

        n = len(targets)
        self._statusbar.showMessage(f"Paramètre « {key} » propagé à {n} image(s).")
        if hasattr(self, "_notif"):
            self._notif.notify(
                f"Propagé : «{key}»",
                f"Étape {step_id}  •  {n} image(s)",
                level=Level.INFO,
                duration=3500,
            )

    @pyqtSlot(str, bool)
    def _on_enabled_propagate(self, step_id: str, enabled: bool) -> None:
        """Propage l'état activé/désactivé d'une étape aux images sélectionnées."""
        self._save_current_state()

        run_paths = self._strip.get_run_selection()
        if run_paths:
            targets = [c for c in self._session.images if c.file_path in set(run_paths)]
        else:
            targets = list(self._session.images)

        if not targets:
            return

        # Snapshot des anciennes valeurs avant propagation
        old_vals = {cfg.file_path: cfg.step_enabled.get(step_id) for cfg in targets}

        # Appliquer immédiatement
        for cfg in targets:
            cfg.step_enabled[step_id] = enabled
            save_recipe(cfg)

        # Mettre à jour l'UI si l'image courante est dans les cibles
        if self._current_cfg in targets:
            panel = self._step_list.get_panel(step_id)
            if panel:
                panel.set_enabled(enabled)

        # Undo : commande dédiée
        def _apply_enabled_ui(sid: str, val: bool) -> None:
            p = self._step_list.get_panel(sid)
            if p:
                p.set_enabled(val)

        cmd = PropagateEnabledCommand(
            step_id, enabled, targets, old_vals,
            lambda: self._current_cfg,
            _apply_enabled_ui,
            save_recipe,
        )
        self._undo_stack.push(cmd)

        n = len(targets)
        state_str = "activé" if enabled else "désactivé"
        self._statusbar.showMessage(f"Étape « {step_id} » {state_str} sur {n} image(s).")
        if hasattr(self, "_notif"):
            self._notif.notify(
                f"Étape {state_str} : {step_id}",
                f"Propagé à {n} image(s)",
                level=Level.INFO,
                duration=3500,
            )

    @pyqtSlot(str, bool)
    def _on_enabled_changed(self, step_id: str, enabled: bool) -> None:
        if self._applying_enabled:
            return
        # Capturer l'ancienne valeur avant mise à jour
        old_val = None
        if self._current_cfg:
            old_val = self._current_cfg.step_enabled.get(step_id)
            self._current_cfg.step_enabled[step_id] = enabled
            self._mark_customized()
        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview_update()
        # Pousser la commande undo
        if old_val is not None and old_val != enabled:
            cmd = ToggleStepCommand(
                self._apply_enabled_silent, step_id, old_val, enabled
            )
            self._undo_stack.push(cmd)

    def _apply_enabled_silent(self, step_id: str, val: bool) -> None:
        """Applique l'état activé sans émettre de commande undo (utilisé par undo/redo)."""
        self._applying_enabled = True
        panel = self._step_list.get_panel(step_id)
        if panel:
            panel._enable_cb.blockSignals(True)
            panel._enable_cb.setChecked(val)
            panel._enable_cb.blockSignals(False)
        if self._current_cfg:
            self._current_cfg.step_enabled[step_id] = val
        self._applying_enabled = False

    @pyqtSlot(str)
    def _on_mask_edit_requested(self, step_id: str) -> None:
        """Bascule sur l'onglet Masque."""
        self._tabs.setCurrentIndex(0)

    @pyqtSlot(str)
    def _on_color_picker_requested(self, step_id: str) -> None:
        """Bascule sur l'onglet Blanc."""
        self._tabs.setCurrentIndex(1)

    def _undo(self) -> None:
        """Ctrl+Z : route vers le bon gestionnaire selon l'onglet actif."""
        idx = self._tabs.currentIndex()
        if idx == 0:    # Masque
            self._mask_panel._canvas.undo()
        elif idx == 1:  # Blanc
            self._wb_panel._canvas.undo()
            # Sync cfg.wb_pick : undo() ne peut pas émettre pick_changed(None)
            if self._current_cfg:
                self._current_cfg.wb_pick = self._wb_panel._canvas._pick_pt
            self._schedule_preview_update()
        else:
            self._undo_stack.undo()

    def _redo(self) -> None:
        """Ctrl+Y : route vers le bon gestionnaire selon l'onglet actif."""
        idx = self._tabs.currentIndex()
        if idx == 0:    # Masque
            self._mask_panel._canvas.redo()
        elif idx == 1:  # Blanc
            self._wb_panel._canvas.redo()
            # Sync cfg.wb_pick après redo
            if self._current_cfg:
                self._current_cfg.wb_pick = self._wb_panel._canvas._pick_pt
            self._schedule_preview_update()
        else:
            self._undo_stack.redo()

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
        """Au changement d'onglet : synchronise le zoom et charge le résultat si nécessaire."""
        from PyQt6.QtCore import QTimer

        # 1. Sauvegarder le zoom de l'onglet qu'on quitte (seulement si actif et visible)
        prev_canvas = self._get_tab_canvas(self._prev_tab_index)
        if prev_canvas is not None and hasattr(prev_canvas, "get_zoom_state"):
            state = prev_canvas.get_zoom_state()
            if state[0] > 0:  # état valide
                self._shared_zoom = state
        self._prev_tab_index = index

        # 2. Charger le résultat si besoin (AVANT d'appliquer le zoom)
        if index == 3 and self._current_cfg is not None:
            self._update_dest_view(self._current_cfg)

        # 3. Actualiser les onglets diff
        if index == 4:
            self._refresh_diff_source()

        # 4. Appliquer le zoom au nouveau canvas, différé après le layout Qt
        #    (set_image appelle fit_in_view → sans différé, le zoom serait écrasé)
        zoom = self._shared_zoom
        if zoom is not None:
            new_canvas = self._get_tab_canvas(index)
            if new_canvas is not None and hasattr(new_canvas, "apply_zoom_state"):
                QTimer.singleShot(
                    0, lambda c=new_canvas, z=zoom: c.apply_zoom_state(*z)
                )

        self._schedule_preview_update()

    def _get_tab_canvas(self, index: int):
        """Retourne le canvas zôomable de l'onglet (index 0–3) ou None."""
        if index == 0:
            return self._mask_panel._canvas
        if index == 1:
            return self._wb_panel._canvas
        if index == 2:
            return self._origin_view
        if index == 3:
            return self._dest_view
        return None

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

        Utilise uniquement set_display_image sur les deux canvases pour
        préserver le masque, le point WB et le zoom en cours.
        """
        if self._current_cfg is None:
            return
        # set_display_image : image affichée seulement, sans toucher au masque, WB ni zoom
        self._mask_panel._canvas.set_display_image(img)
        self._wb_panel._canvas.set_display_image(img)

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

    def _reload_from_disk(self) -> None:
        """Recharge le recipe .json depuis le disque et réinitialise l'UI."""
        cfg = self._current_cfg
        if cfg is None:
            self._statusbar.showMessage("Aucune image sélectionnée.")
            return
        recipe = load_recipe(cfg.file_path)
        if recipe is None:
            self._statusbar.showMessage(
                f"Aucun recipe .json trouvé pour : {os.path.basename(cfg.file_path)}"
            )
            return
        _apply_recipe(cfg, recipe)
        self._navigate_to(cfg)
        self._undo_stack.clear()
        self._statusbar.showMessage(
            f"Recipe rechargé depuis le disque : {os.path.basename(cfg.file_path)}"
        )
        if hasattr(self, "_notif"):
            self._notif.notify(
                "Recipe rechargé",
                os.path.basename(cfg.file_path),
                level=Level.WARNING,
                duration=3000,
            )

    def _current_recipe_dict(self) -> Optional[dict]:
        """Génère le dict recipe depuis l'état actuel de l'UI (non encore sauvegardé)."""
        cfg = self._current_cfg
        if cfg is None:
            return None
        pick    = self._wb_panel.get_pick_point()
        canvas  = self._mask_panel._canvas
        has_mask = canvas._mask is not None and bool(canvas._mask.any())
        return {
            "version":        1,
            "customized":     getattr(cfg, "customized", False),
            "step_order":     self._step_list.get_order(),
            "step_enabled":   self._step_list.get_enabled(),
            "step_params":    self._step_list.get_all_params(),
            "wb_pick":        list(pick) if pick else None,
            "wb_patch_radius": self._wb_panel.get_patch_radius(),
            "has_mask":       has_mask,
        }

    def _refresh_diff_source(self) -> None:
        """Actualise l'onglet Δ Exports : version sélectionnée vs précédente (n-1).

        Mode édition : actuel (UI) vs dernier export sur disque.
        Mode lecture seule (export visionné) : export N vs export N-1.
        """
        if self._current_cfg is None:
            self._diff_source_panel.update_diff(None, None)
            return

        stem = os.path.splitext(os.path.basename(self._current_cfg.file_path))[0]
        exports = self._viewed_export_list  # [(N, path), ...] déjà trié

        if self._is_viewing_export and self._viewed_export_path:
            # Mode lecture seule : diff export N vs export N-1
            current_data = load_export_recipe(self._viewed_export_path)
            # Trouver le prédécesseur dans la liste
            idx_in_list = next(
                (i for i, (_, p) in enumerate(exports) if p == self._viewed_export_path),
                None,
            )
            if idx_in_list is not None and idx_in_list > 0:
                prev_n, prev_path = exports[idx_in_list - 1]
                ref_data = load_export_recipe(prev_path)
                ref_label = f"{stem} — Export {prev_n:03d}"
            else:
                ref_data  = None
                ref_label = "(pas de version précédente)"
            cur_n = exports[idx_in_list][0] if idx_in_list is not None else "?"
            cur_label = f"{stem} — Export {cur_n:03d}"
        else:
            # Mode édition : actuel vs dernier export
            current_data = self._current_recipe_dict()
            if exports:
                last_n, last_path = exports[-1]
                ref_data  = load_export_recipe(last_path)
                ref_label = f"{stem} — Export {last_n:03d}"
            else:
                ref_data  = None
                ref_label = "(pas encore exporté)"
            cur_label = f"{stem} (actuel)"

        self._diff_source_panel.update_diff(
            current_data, ref_data,
            left_label  = cur_label,
            right_label = ref_label,
        )

    def _refresh_if_diff_tab(self) -> None:
        """Actualise le panneau diff si l'onglet actif est l'onglet Δ Exports."""
        if self._tabs.currentIndex() == 4:
            self._refresh_diff_source()

    # ══════════════════════════════════════════════════════════════════════════
    # Export versionné — dropdown + lecture seule
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_export_dropdown(self, cfg: BatchImageConfig) -> None:
        """Repeuple le combo exports pour l'image donnée et revient à 'Courante'."""
        if not self._session.output_dir:
            exports = []
        else:
            exports = list_export_recipes(cfg.file_path, self._session.output_dir)
        self._viewed_export_list = exports

        self._export_combo.blockSignals(True)
        self._export_combo.clear()
        self._export_combo.addItem("Courante")
        for n, path in exports:
            label = f"Export {n:03d}"
            self._export_combo.addItem(label, userData=path)
        self._export_combo.setCurrentIndex(0)
        self._export_combo.blockSignals(False)
        # Mettre à jour la visibilité selon le nombre d'exports
        self._export_combo.setVisible(True)

    @pyqtSlot(int)
    def _on_export_selected(self, index: int) -> None:
        """Slot déclenché quand l'utilisateur sélectionne un élément du combo."""
        if index <= 0:
            self._set_viewing_export(None)
        else:
            path = self._export_combo.itemData(index)
            self._set_viewing_export(path)

    def _set_viewing_export(
        self,
        path: Optional[str],
        _navigate_call: bool = False,
    ) -> None:
        """Active ou désactive le mode lecture seule sur un export versionné.

        path=None  → mode édition normal.
        path=str   → mode lecture seule : charge l'export dans l'UI sans
                     modifier _current_cfg ; active le verrou.
        _navigate_call : si True, n'émet pas de signal vers le combo (évite
                         les boucles lors de _navigate_to).
        """
        if path is None:
            # ── Retour en mode édition ────────────────────────────────────
            self._is_viewing_export  = False
            self._viewed_export_path = None
            self._step_list.setEnabled(True)
            self._restore_export_btn.setVisible(False)
            # Réafficher la config courante dans l'UI
            cfg = self._current_cfg
            if cfg is not None:
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
            if not _navigate_call:
                self._export_combo.blockSignals(True)
                self._export_combo.setCurrentIndex(0)
                self._export_combo.blockSignals(False)
            self._refresh_if_diff_tab()
        else:
            # ── Mode lecture seule ────────────────────────────────────────
            data = load_export_recipe(path)
            if data is None:
                self._statusbar.showMessage("Impossible de charger l'export.")
                self._export_combo.blockSignals(True)
                self._export_combo.setCurrentIndex(0)
                self._export_combo.blockSignals(False)
                return
            self._is_viewing_export  = True
            self._viewed_export_path = path
            # Appliquer dans l'UI sans toucher _current_cfg
            self._applying_order = True
            if "step_order" in data:
                self._step_list.set_order(data["step_order"])
            if "step_enabled" in data:
                for sid, val in data["step_enabled"].items():
                    panel = self._step_list.get_panel(sid)
                    if panel:
                        panel.set_enabled(val)
            if "step_params" in data:
                for sid, params in data["step_params"].items():
                    panel = self._step_list.get_panel(sid)
                    if panel:
                        panel.set_params(params)
            self._applying_order = False
            # Verrouiller
            self._step_list.setEnabled(False)
            self._restore_export_btn.setVisible(True)
            n_str = os.path.basename(path)
            self._statusbar.showMessage(f"Lecture seule : {n_str}")
            self._refresh_if_diff_tab()

    def _restore_export(self) -> None:
        """Restaure la configuration de l'export visionné dans _current_cfg (en mémoire)."""
        if not self._is_viewing_export or not self._viewed_export_path:
            return
        cfg = self._current_cfg
        if cfg is None:
            return
        data = load_export_recipe(self._viewed_export_path)
        if data is None:
            return
        _apply_recipe(cfg, data)
        self._set_viewing_export(None)
        # Recharger l'UI depuis cfg mis à jour
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
        self._statusbar.showMessage("Configuration restaurée depuis l'export (non sauvegardée).")
        if hasattr(self, "_notif"):
            stem = os.path.splitext(os.path.basename(cfg.file_path))[0]
            self._notif.notify(
                "Configuration restaurée",
                f"{stem} — export chargé en mémoire",
                level=Level.WARNING,
                duration=4000,
            )

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
        self._batch_run_total = len(queue)
        self._batch_run_done  = 0
        self._batch_run_start = time.monotonic()
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
        self._image_start_time = time.monotonic()

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
        if hasattr(self, "_notif") and self._batch_run_total > 0:
            elapsed = time.monotonic() - self._batch_run_start
            # Fermer le toast de progression en cours avant d'afficher le résumé
            self._notif.dismiss_key("batch_progress")
            body = (
                f"Durée totale : {elapsed:.1f}s"
                + (f"  •  {elapsed/self._batch_run_done:.1f}s/image"
                   if self._batch_run_done > 0 else "")
            )
            self._notif.notify(
                f"⚡ Batch terminé — {self._batch_run_done}/{self._batch_run_total} image(s)",
                body,
                level=Level.SUCCESS,
                duration=12000,
            )

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
                # set_display_image préserve masque, WB et zoom en cours
                self._mask_panel._canvas.set_display_image(result_img)
                self._wb_panel._canvas.set_display_image(result_img)
                if self._tabs.currentIndex() == 3:
                    self._dest_view.set_image(result_img)
            # result_img n'est pas stocké dans cfg — libéré après écriture disque.

        cfg.batch_status = "done"
        self._strip.set_running(cfg.file_path, False)
        self._strip.set_done(cfg.file_path, True)

        # Notification image traitée (mise à jour in-place via key)
        if hasattr(self, "_notif"):
            elapsed = time.monotonic() - self._image_start_time
            self._batch_run_done += 1
            total_str = f"/{self._batch_run_total}" if self._batch_run_total > 1 else ""
            self._notif.notify(
                f"✓ {cfg.filename}",
                f"Traité en {elapsed:.1f}s  •  {self._batch_run_done}{total_str}",
                level=Level.SUCCESS,
                duration=5000,
                key="batch_progress",
            )

        # Rafraîchir le dropdown si c'est l'image courante (nouvel export vient d'être créé)
        if cfg is self._current_cfg:
            self._refresh_export_dropdown(cfg)

        # Sidecar résultat JSON + image
        step_log = build_step_log(
            step_order   = cfg.step_order,
            step_enabled = cfg.step_enabled,
            step_params  = cfg.step_params,
            step_results = self._worker_step_results,
            context      = cfg.context,
            steps_by_id  = self._steps_by_id,
        )
        try:
            self._session.save_result(cfg, result_img, step_log)
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
            self._session.save_session_meta()

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
        self._session.save_session_meta()
        self._clear_instance_state()
        self.closed.emit()
        super().closeEvent(event)
