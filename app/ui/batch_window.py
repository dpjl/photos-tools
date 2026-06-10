"""ui/batch_window.py — Fenêtre de traitement par lots (mode batch).

Layout ::

    ┌── Toolbar: [Sortie: /...][Parcourir][▶ Lancer la sélection][⚡ Lancer le batch] ─────────────────────────────┐
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
    QSlider, QCheckBox, QFrame, QComboBox, QProgressBar, QInputDialog, QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QUndoStack, QKeySequence, QShortcut

from core.config_diff import changed_step_details
from core.pipeline import PipelineWorker
from core.batch import BatchSession, BatchImageConfig
from steps import ALL_STEPS
from ui.step_panel import StepListWidget
from ui.batch_thumbnail_strip import BatchThumbnailStrip
from ui.mask_editor import MaskCanvasPanel
from ui.crop_editor import CropCanvasPanel
from ui.wb_picker import WBPickerPanel
from ui.image_view import SyncedImageView
from ui.export_mosaic import ExportMosaicView
from ui.export_detail_panel import ExportDetailPanel
from ui.notifications import NotificationManager, Level
from ui.batch_window_constants import (
    _FAST_PREVIEW_IDS, _PREVIEW_TABS,
    _TAB_PREVIEW, _TAB_MASK, _TAB_WB, _TAB_REDEYE,
    _TAB_CROP, _TAB_ORIGIN, _TAB_RESULT,
)
from core.export_manager import ExportManager, ExportEntry
from core.image_metadata import write_jpeg_with_source_exif
from ui.batch_mixins.nav import NavMixin
from ui.batch_mixins.params import ParamsMixin
from ui.batch_mixins.preview import PreviewMixin
from ui.batch_mixins.run import RunMixin
from ui.batch_mixins.exports import ExportsMixin


# Niveaux de « Couverture » → échelles d'inférence fusionnées (max multi-échelle).
# L'échelle haute (1600) donne la précision/les détails fins ; l'échelle basse
# assure la continuité des longues rayures faibles. Plus la basse est petite,
# plus on capte de rayures/poussières (mais plus grossièrement).
_ARTIFACT_COVER_LEVELS = [
    ("Fine",      (1600,)),       # haute résolution seule — précis, peut fragmenter
    ("Standard",  (1600, 640)),   # fusion équilibrée — défaut
    ("Élevée",    (1600, 500)),
    ("Maximale",  (1600, 360)),
]
_ARTIFACT_COVER_DEFAULT = 1   # « Standard » — rayures complètes + propre

# Valeurs par défaut des réglages de détection d'artefacts (pour le bouton « Défaut »)
_ARTIFACT_THRESH_DEFAULT = 50   # seuil IA ×100
_ARTIFACT_COLOR_DEFAULT  = 40   # sensibilité couleur (écart chromatique)
_ARTIFACT_DILATE_DEFAULT = 1    # dilatation (px)


class _ArtifactDetectWorker(QThread):
    """Calcule la carte de probabilité de rayures hors du thread UI.

    Seule l'inférence neuronale (lente, + téléchargement au 1er appel) tourne
    ici ; le seuillage, la dilatation et les points colorés (rapides) sont
    appliqués dans le thread UI pour permettre un réglage live.
    """

    done   = pyqtSignal(object)   # liste de cartes de proba par échelle (ou None)
    failed = pyqtSignal(str)

    def __init__(self, image_bgr: np.ndarray, scales: tuple) -> None:
        super().__init__()
        self._image = image_bgr
        self._scales = scales

    def run(self) -> None:
        try:
            from core.artifact_detect import ScratchDetector
            # On renvoie les cartes PAR échelle : la fusion (max ↔ consensus) se
            # fait ensuite dans l'UI, en live, sans relancer l'inférence.
            self.done.emit(
                ScratchDetector.get().detect_prob_scales(self._image, self._scales)
            )
        except Exception as exc:  # noqa: BLE001 — remonté à l'UI via signal
            self.failed.emit(str(exc))


class _VLMRefineWorker(QThread):
    """Affine le masque avec un VLM local hors du thread UI."""

    progress = pyqtSignal(int, int)   # (fait, total)
    done     = pyqtSignal(object)     # RefineResult
    failed   = pyqtSignal(str)

    def __init__(self, image_bgr: np.ndarray, mask: np.ndarray, model_id: str,
                 min_len: int, min_thick: int) -> None:
        super().__init__()
        self._image = image_bgr
        self._mask = mask
        self._model_id = model_id
        self._min_len = min_len
        self._min_thick = min_thick

    def run(self) -> None:
        try:
            from core.vlm_refine import VLMRefiner
            res = VLMRefiner.get().refine(
                self._image, self._mask, self._model_id,
                min_len=self._min_len, min_thick=self._min_thick,
                on_progress=lambda k, t: self.progress.emit(k, t),
            )
            self.done.emit(res)
        except Exception as exc:  # noqa: BLE001 — remonté à l'UI via signal
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


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
    _SETTINGS_ORG = "PlusTekPhoto"
    _SETTINGS_APP = "RestaurationPhoto"
    _SETTING_STRIP_HIDDEN = "batch/thumbnail_strip_hidden"

    def __init__(self, parent, session: BatchSession) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch — Traitement par lots")
        self.resize(1600, 950)

        self._session     = session
        self._steps_by_id = {s.id: s for s in ALL_STEPS}
        self._current_cfg: Optional[BatchImageConfig] = None
        self._worker:      Optional[PipelineWorker]   = None
        self._settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        self._strip_hidden = self._read_bool_setting(
            self._SETTING_STRIP_HIDDEN, False
        )

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
        self._has_unretained:      bool                     = False

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
        self._apply_strip_visibility(persist=False)

        root.addWidget(self._build_main_area(), stretch=1)

        self._statusbar = QStatusBar()
        self._statusbar.setStyleSheet("background:#111124; color:#888; font-size:10px;")
        self.setStatusBar(self._statusbar)

        if self._session.images:
            self._strip.load_images([c.file_path for c in self._session.images])
            self._refresh_all_strip_thumbs()
        self._refresh_retained_count()
        self._update_result_diff_indicator()

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

        self._toggle_strip_btn = QPushButton()
        self._toggle_strip_btn.setCheckable(True)
        self._style_btn(self._toggle_strip_btn)
        self._toggle_strip_btn.toggled.connect(self._on_strip_visibility_toggled)
        lay.addWidget(self._toggle_strip_btn)
        self._sync_strip_toggle_button()

        self._retained_count_lbl = QLabel()
        self._retained_count_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._retained_count_lbl.setMinimumWidth(138)
        lay.addWidget(self._retained_count_lbl)
        self._refresh_retained_count()

        self._prev_image_btn = QPushButton("‹")
        self._prev_image_btn.setFixedWidth(30)
        self._style_btn(self._prev_image_btn)
        self._prev_image_btn.setToolTip("Photo précédente du batch")
        self._prev_image_btn.clicked.connect(self._go_to_previous_image)
        lay.addWidget(self._prev_image_btn)

        self._next_image_btn = QPushButton("›")
        self._next_image_btn.setFixedWidth(30)
        self._style_btn(self._next_image_btn)
        self._next_image_btn.setToolTip("Photo suivante du batch")
        self._next_image_btn.clicked.connect(self._go_to_next_image)
        lay.addWidget(self._next_image_btn)

        self._first_unretained_btn = QPushButton("☆")
        self._first_unretained_btn.setFixedWidth(30)
        self._style_btn(self._first_unretained_btn)
        self._first_unretained_btn.setToolTip(
            "Aller à la première photo sans résultat retenu"
        )
        self._first_unretained_btn.clicked.connect(self._go_to_first_unretained)
        lay.addWidget(self._first_unretained_btn)
        self._update_batch_nav_buttons()

        lay.addStretch()

        # ── Jauge VRAM + déchargement global des modèles ─────────────────────
        self._vram_bar = QProgressBar()
        self._vram_bar.setFixedSize(132, 22)
        self._vram_bar.setTextVisible(True)
        self._vram_bar.setToolTip("Mémoire GPU utilisée / totale (toute la carte).")
        lay.addWidget(self._vram_bar)

        self._unload_all_btn = QPushButton("🧹")
        self._unload_all_btn.setFixedWidth(34)
        self._unload_all_btn.setToolTip(
            "Décharger TOUS les modèles chargés (SCUNet, GFPGAN, LaMa, VLM…)\n"
            "pour libérer la VRAM. Ils seront rechargés à la demande."
        )
        self._style_btn(self._unload_all_btn)
        self._unload_all_btn.clicked.connect(self._unload_all_models)
        lay.addWidget(self._unload_all_btn)

        self._vram_timer = QTimer(self)
        self._vram_timer.setInterval(2500)
        self._vram_timer.timeout.connect(self._update_vram_gauge)
        self._vram_timer.start()
        self._update_vram_gauge()

        self._selection_btn = QPushButton("▶  Lancer la sélection")
        self._style_btn(self._selection_btn, accent=True)
        self._selection_btn.setToolTip(
            "Traiter et sauvegarder les images sélectionnées (Ctrl+clic / Shift+clic)"
        )
        self._selection_btn.clicked.connect(self._run_on_selection)
        lay.addWidget(self._selection_btn)

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

    def _read_bool_setting(self, key: str, default: bool) -> bool:
        """Lit un booléen QSettings en restant robuste selon le backend."""
        value = self._settings.value(key, default)
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _on_strip_visibility_toggled(self, hidden: bool) -> None:
        self._strip_hidden = hidden
        self._apply_strip_visibility(persist=True)

    def _apply_strip_visibility(self, persist: bool) -> None:
        if hasattr(self, "_strip"):
            self._strip.setVisible(not self._strip_hidden)
        self._sync_strip_toggle_button()
        if persist:
            self._settings.setValue(self._SETTING_STRIP_HIDDEN, self._strip_hidden)

    def _sync_strip_toggle_button(self) -> None:
        if not hasattr(self, "_toggle_strip_btn"):
            return
        btn = self._toggle_strip_btn
        previous = btn.blockSignals(True)
        btn.setChecked(self._strip_hidden)
        if self._strip_hidden:
            btn.setText("▾  Afficher les vignettes")
            btn.setToolTip("Afficher le bandeau des photos du batch")
        else:
            btn.setText("▴  Masquer les vignettes")
            btn.setToolTip("Masquer le bandeau des photos du batch pour agrandir les vues")
        btn.blockSignals(previous)

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

        self._recipe_diff_lbl = QLabel("Aucun résultat")
        self._recipe_diff_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr_lay.addWidget(self._recipe_diff_lbl)
        self._set_recipe_diff_state(False, 0)

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
        self._step_list.crop_edit_requested.connect(self._on_crop_edit_requested)
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

        # Largeur de la barre d'outils, IDENTIQUE pour TOUS les onglets-image
        # (Preview, Masque, Blanc, Yeux rouges, Recadrage, Originale) afin que
        # la zone d'image — et donc la position/échelle de la photo — se
        # superpose parfaitement d'un onglet à l'autre (comparaison facile).
        # Compacte : les contrôles du panneau sont dimensionnés pour cette largeur.
        _SIDEBAR_W = 250
        _dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        self._mask_panel = MaskCanvasPanel(_dummy, None, show_ok_cancel=False,
                                           sidebar_width=_SIDEBAR_W)
        self._mask_auto_btn = QPushButton("🔍  Détecter les artefacts")
        self._mask_auto_btn.setToolTip(
            "Détecte automatiquement rayures, traits, plis, poussières\n"
            "et points colorés étrangers (IA + analyse), puis ajoute les\n"
            "zones correspondantes au masque de retouche."
        )
        self._mask_auto_btn.setStyleSheet(
            "QPushButton { background:#1e2a1e; color:#6e8; border:1px solid #3a5a3a;"
            "  border-radius:4px; padding:5px 6px; font-size:10px; }"
            "QPushButton:hover { background:#2a3a2a; color:#aea; }"
            "QPushButton:disabled { color:#445; border-color:#223; }"
        )
        self._mask_auto_btn.clicked.connect(self._mask_auto_detect)
        self._mask_panel.add_to_sidebar(self._mask_auto_btn)
        self._mask_panel.add_to_sidebar(self._build_artifact_controls())
        self._mask_panel.add_to_sidebar(self._build_vlm_controls())
        self._vlm_worker: Optional[QThread] = None
        self._vlm_last_result = None

        # État de la détection d'artefacts (cache pour le réglage live)
        self._artifact_worker: Optional[QThread] = None
        self._artifact_probs: Optional[list] = None             # cartes IA par échelle
        self._artifact_base_mask: Optional[np.ndarray] = None   # masque avant auto
        self._artifact_base_image: Optional[np.ndarray] = None  # image analysée (BGR)
        self._artifact_spots: Optional[np.ndarray] = None       # cache points colorés
        self._artifact_spots_dev: float = -1.0                  # sensibilité du cache
        self._artifact_pending_report: bool = False             # afficher le bilan après inférence
        self._wb_panel   = WBPickerPanel(
            _dummy, None, show_ok_cancel=False, sidebar_width=_SIDEBAR_W
        )
        self._redeye_panel = MaskCanvasPanel(_dummy, None, show_ok_cancel=False,
                                              sidebar_width=_SIDEBAR_W)
        _btn_redeye_auto = QPushButton("🔍  Détecter pupilles auto")
        _btn_redeye_auto.setToolTip(
            "Détecte automatiquement les iris et ajoute\n"
            "les cercles correspondants dans le masque."
        )
        _btn_redeye_auto.setStyleSheet(
            "QPushButton { background:#1e2a1e; color:#6e8; border:1px solid #3a5a3a;"
            "  border-radius:4px; padding:5px 6px; font-size:10px; }"
            "QPushButton:hover { background:#2a3a2a; color:#aea; }"
            "QPushButton:disabled { color:#445; border-color:#223; }"
        )
        _btn_redeye_auto.clicked.connect(self._redeye_auto_detect)
        self._redeye_panel.add_to_sidebar(_btn_redeye_auto)

        self._crop_panel = CropCanvasPanel(_dummy, None, show_ok_cancel=False,
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

        self._preview_info_lbl = QLabel("")
        self._preview_info_lbl.setWordWrap(True)
        self._preview_info_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._preview_info_lbl.setStyleSheet(
            "color:#92a9b8; font-size:10px; font-family:Consolas,monospace;"
            " border:1px solid #242446; border-radius:4px; padding:6px;"
            " background:#111124;"
        )
        ps_lay.addWidget(self._preview_info_lbl)

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
        self._tabs.addTab(self._crop_panel,                               "Recadrage")     # _TAB_CROP
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
        self._crop_panel.crop_changed.connect(self._on_crop_changed)
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
    # Masque — détection automatique d'artefacts (rayures, plis, points colorés)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_artifact_controls(self) -> QWidget:
        """Panneau de réglages de la détection d'artefacts (live)."""
        box = QFrame()
        box.setStyleSheet("QFrame { background:#16201a; border-radius:4px; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 5, 6, 6)
        lay.setSpacing(3)

        def _lbl(text: str) -> QLabel:
            q = QLabel(text)
            q.setStyleSheet("color:#9ab; font-size:9px; background:transparent;")
            return q

        cb_css = "QCheckBox { color:#bcd; font-size:10px; background:transparent; }"

        # Cases : rayures (IA) / points colorés
        self._artifact_scratch_cb = QCheckBox("Rayures/plis (IA)")
        self._artifact_spots_cb   = QCheckBox("Points colorés")
        for cb in (self._artifact_scratch_cb, self._artifact_spots_cb):
            cb.setChecked(True)
            cb.setStyleSheet(cb_css)
            cb.toggled.connect(self._on_artifact_param_changed)
            lay.addWidget(cb)

        # Consensus multi-échelle : exige un accord entre résolutions → retire
        # les faux positifs linéaires (arêtes d'objets, motifs). Live (max↔moyenne).
        self._artifact_consensus_cb = QCheckBox("Consensus (anti-FP)")
        self._artifact_consensus_cb.setChecked(False)
        self._artifact_consensus_cb.setToolTip(
            "Exige qu'une rayure soit détectée à plusieurs résolutions.\n"
            "Retire les faux positifs type arêtes/motifs, mais capte un peu\n"
            "moins les rayures et poussières les plus faibles."
        )
        self._artifact_consensus_cb.setStyleSheet(cb_css)
        self._artifact_consensus_cb.toggled.connect(self._on_artifact_param_changed)
        lay.addWidget(self._artifact_consensus_cb)

        # Couverture = résolution d'inférence (bas = plus couvrant, relance l'IA)
        self._artifact_cover_lbl = _lbl("Couv. : Standard")
        lay.addWidget(self._artifact_cover_lbl)
        self._artifact_cover_slider = QSlider(Qt.Orientation.Horizontal)
        self._artifact_cover_slider.setRange(0, 3)   # 0=Fine … 3=Maximale
        self._artifact_cover_slider.setValue(_ARTIFACT_COVER_DEFAULT)
        self._artifact_cover_slider.valueChanged.connect(self._on_artifact_cover_label)
        self._artifact_cover_slider.sliderReleased.connect(self._on_artifact_cover_changed)
        lay.addWidget(self._artifact_cover_slider)

        # Seuil IA (0.05–0.95)
        self._artifact_thresh_lbl = _lbl("Seuil : 0.50")
        lay.addWidget(self._artifact_thresh_lbl)
        self._artifact_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._artifact_thresh_slider.setRange(5, 95)
        self._artifact_thresh_slider.setValue(_ARTIFACT_THRESH_DEFAULT)
        self._artifact_thresh_slider.valueChanged.connect(self._on_artifact_param_changed)
        lay.addWidget(self._artifact_thresh_slider)

        # Sensibilité couleur (écart chromatique, plus bas = plus sensible)
        self._artifact_color_lbl = _lbl("Couleur : 40")
        lay.addWidget(self._artifact_color_lbl)
        self._artifact_color_slider = QSlider(Qt.Orientation.Horizontal)
        self._artifact_color_slider.setRange(15, 90)
        self._artifact_color_slider.setValue(_ARTIFACT_COLOR_DEFAULT)
        self._artifact_color_slider.valueChanged.connect(self._on_artifact_param_changed)
        lay.addWidget(self._artifact_color_slider)

        # Dilatation (px)
        self._artifact_dilate_lbl = _lbl("Dilat. : 1 px")
        lay.addWidget(self._artifact_dilate_lbl)
        self._artifact_dilate_slider = QSlider(Qt.Orientation.Horizontal)
        self._artifact_dilate_slider.setRange(0, 6)
        self._artifact_dilate_slider.setValue(_ARTIFACT_DILATE_DEFAULT)
        self._artifact_dilate_slider.valueChanged.connect(self._on_artifact_param_changed)
        lay.addWidget(self._artifact_dilate_slider)

        # Bouton : restaurer les valeurs par défaut
        btn_default = QPushButton("↺  Défaut")
        btn_default.setToolTip("Réinitialise tous les réglages de détection ci-dessus.")
        btn_default.setStyleSheet(
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 6px; font-size:10px; }"
            "QPushButton:hover { background:#2a2a50; color:#ccc; }"
        )
        btn_default.clicked.connect(self._reset_artifact_params)
        lay.addWidget(btn_default)

        return box

    def _reset_artifact_params(self) -> None:
        """Restaure les réglages de détection d'artefacts à leurs valeurs par défaut."""
        widgets = (
            self._artifact_scratch_cb, self._artifact_spots_cb,
            self._artifact_consensus_cb,
            self._artifact_cover_slider, self._artifact_thresh_slider,
            self._artifact_color_slider, self._artifact_dilate_slider,
        )
        for w in widgets:
            w.blockSignals(True)
        self._artifact_scratch_cb.setChecked(True)
        self._artifact_spots_cb.setChecked(True)
        self._artifact_consensus_cb.setChecked(False)
        self._artifact_cover_slider.setValue(_ARTIFACT_COVER_DEFAULT)
        self._artifact_thresh_slider.setValue(_ARTIFACT_THRESH_DEFAULT)
        self._artifact_color_slider.setValue(_ARTIFACT_COLOR_DEFAULT)
        self._artifact_dilate_slider.setValue(_ARTIFACT_DILATE_DEFAULT)
        for w in widgets:
            w.blockSignals(False)

        # Resynchroniser les libellés
        self._on_artifact_cover_label()
        self._update_artifact_labels()

        # Si une détection a déjà tourné, relancer avec les réglages par défaut
        # (couverture incluse → réinférence) ; sinon il n'y a rien à recomposer.
        if self._artifact_base_image is not None:
            self._run_artifact_inference(report=False)

    def _artifact_scales(self) -> tuple:
        """Échelles d'inférence correspondant au niveau de couverture choisi."""
        return _ARTIFACT_COVER_LEVELS[self._artifact_cover_slider.value()][1]

    def _mask_auto_detect(self) -> None:
        """Lance la détection et applique le résultat (réseau en tâche de fond)."""
        if self._artifact_worker is not None:
            return  # détection déjà en cours
        # On détecte sur l'image ACTUELLEMENT affichée (= preview courant), pas sur
        # l'originale figée : ainsi un preview SCUNet/couleur change la détection.
        img = self._mask_panel.get_display_image()
        if img is None or img.size <= 3:
            self._statusbar.showMessage("Aucune image chargée.")
            return

        # Une nouvelle détection invalide une revue VLM en cours.
        self._vlm_apply_btn.setEnabled(False)
        # Point de référence : masque manuel actuel + image analysée.
        self._mask_panel.push_undo()           # une seule entrée undo pour toute l'auto-détection
        self._artifact_base_mask = self._mask_panel.get_mask()
        self._artifact_base_image = img
        self._artifact_probs = None
        self._artifact_spots = None
        self._artifact_spots_dev = -1.0
        self._run_artifact_inference(report=True)

    def _run_artifact_inference(self, report: bool) -> None:
        """Calcule (ou recalcule) la carte IA pour l'image de référence courante."""
        if self._artifact_worker is not None or self._artifact_base_image is None:
            return

        if not self._artifact_scratch_cb.isChecked():
            # Pas d'IA demandée → uniquement points colorés (immédiat).
            self._artifact_probs = None
            self._recompute_artifact_mask(report=report)
            return

        self._artifact_pending_report = report
        self._mask_auto_btn.setEnabled(False)
        self._statusbar.showMessage(
            "Détection des artefacts en cours… "
            "(1er appel : téléchargement du modèle, peut être long)"
        )
        worker = _ArtifactDetectWorker(self._artifact_base_image, self._artifact_scales())
        worker.done.connect(self._on_artifacts_detected)
        worker.failed.connect(self._on_artifacts_failed)
        worker.finished.connect(self._on_artifact_worker_finished)
        self._artifact_worker = worker
        worker.start()

    def _on_artifact_cover_label(self) -> None:
        """Met à jour le libellé de couverture pendant le glissement (sans relancer)."""
        name = _ARTIFACT_COVER_LEVELS[self._artifact_cover_slider.value()][0]
        self._artifact_cover_lbl.setText(f"Couv. : {name}")

    def _on_artifact_cover_changed(self) -> None:
        """Relâchement du curseur de couverture → relance l'inférence (≈0,4 s)."""
        self._on_artifact_cover_label()
        if self._artifact_base_image is not None:
            self._run_artifact_inference(report=False)

    def _on_artifacts_detected(self, probs) -> None:
        self._artifact_probs = probs
        self._recompute_artifact_mask(report=self._artifact_pending_report)

    def _on_artifacts_failed(self, message: str) -> None:
        self._statusbar.showMessage(f"Erreur détection artefacts : {message}")

    def _on_artifact_worker_finished(self) -> None:
        self._mask_auto_btn.setEnabled(True)
        self._artifact_worker = None

    def _update_artifact_labels(self) -> None:
        """Resynchronise les libellés des réglages avec les valeurs des curseurs."""
        thr = self._artifact_thresh_slider.value() / 100.0
        self._artifact_thresh_lbl.setText(f"Seuil : {thr:.2f}")
        self._artifact_color_lbl.setText(
            f"Couleur : {self._artifact_color_slider.value()}"
        )
        self._artifact_dilate_lbl.setText(
            f"Dilat. : {self._artifact_dilate_slider.value()} px"
        )

    def _on_artifact_param_changed(self) -> None:
        """Réglage live : recombine le masque depuis le cache, sans relancer l'IA."""
        self._update_artifact_labels()
        # Recombiner seulement si une détection a déjà été lancée sur cette image
        if self._artifact_base_image is not None:
            self._recompute_artifact_mask(report=False)

    def _recompute_artifact_mask(self, report: bool) -> None:
        """Recombine masque manuel + détections selon les réglages courants."""
        from core.artifact_detect import (
            mask_from_prob, dilate_mask, detect_color_spots, fuse_probs,
        )
        base_mask = self._artifact_base_mask
        base_img  = self._artifact_base_image
        if base_mask is None or base_img is None:
            return

        # Les détections sont calculées dans l'espace de l'image analysée (preview).
        detected = np.zeros(base_img.shape[:2], dtype=np.uint8)
        if self._artifact_scratch_cb.isChecked() and self._artifact_probs:
            thr  = self._artifact_thresh_slider.value() / 100.0
            # Fusion max ↔ consensus (moyenne) — live, sans relancer l'IA.
            prob = fuse_probs(self._artifact_probs, self._artifact_consensus_cb.isChecked())
            detected = cv2.bitwise_or(detected, mask_from_prob(prob, thr))
        if self._artifact_spots_cb.isChecked():
            dev = float(self._artifact_color_slider.value())
            if self._artifact_spots is None or self._artifact_spots_dev != dev:
                self._artifact_spots = detect_color_spots(base_img, chroma_dev=dev)
                self._artifact_spots_dev = dev
            detected = cv2.bitwise_or(detected, self._artifact_spots)

        detected = dilate_mask(detected, self._artifact_dilate_slider.value())

        # Réaligner sur l'espace du masque si le preview a d'autres dimensions
        # (ex. recadrage / upscale dans le pipeline).
        if detected.shape[:2] != base_mask.shape[:2]:
            detected = cv2.resize(
                detected, (base_mask.shape[1], base_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = cv2.bitwise_or(base_mask, detected)
        self._mask_panel.set_mask(mask, push_undo=False)

        if report:
            n = int(cv2.connectedComponents(detected)[0]) - 1
            if n <= 0:
                self._statusbar.showMessage("Aucun artefact détecté automatiquement.")
            else:
                self._statusbar.showMessage(
                    f"{n} zone(s) d'artefact détectée(s) — réglez seuil/sensibilité à droite."
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Masque — affinage par VLM local (défaut vs décor)
    # ══════════════════════════════════════════════════════════════════════════

    def _build_vlm_controls(self) -> QWidget:
        """Panneau : modèle + token + seuils candidats + affinage + inspection."""
        from core.vlm_refine import VLM_MODELS, DEFAULT_MODEL

        box = QFrame()
        box.setStyleSheet("QFrame { background:#201a26; border-radius:4px; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 5, 6, 6)
        lay.setSpacing(3)

        def _lbl(text: str) -> QLabel:
            q = QLabel(text)
            q.setStyleSheet("color:#a89ab0; font-size:9px; background:transparent;")
            return q

        # Style commun des boutons secondaires (compacts)
        sec_css = (
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 4px; font-size:10px; }"
            "QPushButton:hover { background:#2a2a50; color:#ccc; }"
            "QPushButton:disabled { color:#445; }"
        )

        title = QLabel("Affinage IA (VLM)")
        title.setStyleSheet("color:#c9b; font-size:10px; font-weight:700; background:transparent;")
        lay.addWidget(title)

        # Modèle + bouton token (gated repos comme Gemma)
        row = QHBoxLayout(); row.setSpacing(4)
        self._vlm_model_combo = QComboBox()
        for label, repo, fits in VLM_MODELS:
            self._vlm_model_combo.addItem(label, repo)
        idx = self._vlm_model_combo.findData(DEFAULT_MODEL)
        if idx >= 0:
            self._vlm_model_combo.setCurrentIndex(idx)
        self._vlm_model_combo.setStyleSheet(
            "QComboBox { background:#1a1426; color:#cbd; border:1px solid #3a2a4a;"
            "  border-radius:4px; padding:3px 6px; font-size:10px; }"
            "QComboBox QAbstractItemView { background:#1a1426; color:#cbd;"
            "  selection-background-color:#3a2a5a; }"
        )
        row.addWidget(self._vlm_model_combo, stretch=1)
        btn_token = QPushButton("🔑")
        btn_token.setFixedWidth(28)
        btn_token.setToolTip("Renseigner un token HuggingFace (pour les modèles\n"
                             "restreints comme Gemma). Enregistré de façon permanente.")
        btn_token.setStyleSheet(
            "QPushButton { background:#1a1426; color:#cb9; border:1px solid #3a2a4a;"
            "  border-radius:4px; padding:3px; font-size:11px; }"
            "QPushButton:hover { background:#2a2040; }"
        )
        btn_token.clicked.connect(self._set_hf_token)
        row.addWidget(btn_token)
        lay.addLayout(row)

        # Seuils de sélection des candidats
        self._vlm_line_lbl = _lbl("Lignes ≥ 45 px")
        lay.addWidget(self._vlm_line_lbl)
        self._vlm_line_slider = QSlider(Qt.Orientation.Horizontal)
        self._vlm_line_slider.setRange(20, 200); self._vlm_line_slider.setValue(45)
        self._vlm_line_slider.valueChanged.connect(
            lambda v: self._vlm_line_lbl.setText(f"Lignes ≥ {v} px"))
        lay.addWidget(self._vlm_line_slider)

        self._vlm_blob_lbl = _lbl("Taches ≥ 12 px")
        lay.addWidget(self._vlm_blob_lbl)
        self._vlm_blob_slider = QSlider(Qt.Orientation.Horizontal)
        self._vlm_blob_slider.setRange(6, 60); self._vlm_blob_slider.setValue(12)
        self._vlm_blob_slider.valueChanged.connect(
            lambda v: self._vlm_blob_lbl.setText(f"Taches ≥ {v} px"))
        lay.addWidget(self._vlm_blob_slider)

        btn_cand = QPushButton("👁  Voir les candidats")
        btn_cand.setToolTip("Affiche, sans lancer le VLM, les zones qui seront\n"
                            "analysées (lignes + grosses taches).")
        btn_cand.setStyleSheet(sec_css)
        btn_cand.clicked.connect(self._show_vlm_candidates)
        lay.addWidget(btn_cand)

        self._vlm_refine_btn = QPushButton("✨  Affiner (VLM)")
        self._vlm_refine_btn.setToolTip(
            "Envoie les zones suspectes (lignes + grosses taches) à un VLM\n"
            "local qui retire celles qui sont en réalité des éléments du décor.\n"
            "1er usage : téléchargement du modèle (~8 Go)."
        )
        self._vlm_refine_btn.setStyleSheet(
            "QPushButton { background:#2a1e38; color:#c9a6e8; border:1px solid #5a3a7a;"
            "  border-radius:4px; padding:5px 6px; font-size:10px; }"
            "QPushButton:hover { background:#3a2a50; color:#dcb6ff; }"
            "QPushButton:disabled { color:#544; border-color:#332; }"
        )
        self._vlm_refine_btn.clicked.connect(self._vlm_refine)
        lay.addWidget(self._vlm_refine_btn)

        self._vlm_progress = QProgressBar()
        self._vlm_progress.setVisible(False)
        self._vlm_progress.setTextVisible(True)
        self._vlm_progress.setFixedHeight(14)
        self._vlm_progress.setStyleSheet(
            "QProgressBar { background:#16121e; border:1px solid #3a2a4a;"
            "  border-radius:4px; color:#dcb; font-size:9px; text-align:center; }"
            "QProgressBar::chunk { background:#7a4caa; border-radius:3px; }"
        )
        lay.addWidget(self._vlm_progress)

        # Bouton d'application de la revue (supprime les zones violettes = décor)
        self._vlm_apply_btn = QPushButton("✅  Appliquer")
        self._vlm_apply_btn.setEnabled(False)
        self._vlm_apply_btn.setToolTip(
            "Supprime du masque toutes les zones marquées DÉCOR (violet).\n"
            "Double-cliquez une zone pour changer violet ↔ vert avant d'appliquer."
        )
        self._vlm_apply_btn.setStyleSheet(
            "QPushButton { background:#1e3320; color:#9de0a8; border:1px solid #3a6a45;"
            "  border-radius:4px; padding:5px 6px; font-size:10px; }"
            "QPushButton:hover { background:#274530; color:#bff0c8; }"
            "QPushButton:disabled { color:#455; border-color:#233; }"
        )
        self._vlm_apply_btn.clicked.connect(self._vlm_apply_review)
        lay.addWidget(self._vlm_apply_btn)

        # Boutons secondaires côte à côte (conversation / décharger)
        row2 = QHBoxLayout(); row2.setSpacing(4)
        self._vlm_convo_btn = QPushButton("\U0001f4ac  Conversation")
        self._vlm_convo_btn.setEnabled(False)
        self._vlm_convo_btn.setToolTip("Voir l'échange complet avec le VLM.")
        self._vlm_convo_btn.setStyleSheet(sec_css)
        self._vlm_convo_btn.clicked.connect(self._show_vlm_conversation)
        row2.addWidget(self._vlm_convo_btn)

        self._vlm_unload_btn = QPushButton("🗑  Décharger")
        self._vlm_unload_btn.setEnabled(False)
        self._vlm_unload_btn.setToolTip("Libère la mémoire GPU occupée par le modèle VLM.")
        self._vlm_unload_btn.setStyleSheet(
            "QPushButton { background:#1e1e38; color:#9ab; border-radius:4px;"
            "  padding:4px 4px; font-size:10px; }"
            "QPushButton:hover { background:#3a2030; color:#e99; }"
            "QPushButton:disabled { color:#445; }"
        )
        self._vlm_unload_btn.clicked.connect(self._unload_vlm)
        row2.addWidget(self._vlm_unload_btn)
        lay.addLayout(row2)

        return box

    def _set_hf_token(self) -> None:
        """Saisie + enregistrement permanent d'un token HuggingFace."""
        from core.vlm_refine import set_hf_token, has_hf_token
        existing = "déjà enregistré" if has_hf_token() else "aucun"
        token, ok = QInputDialog.getText(
            self, "Token HuggingFace",
            f"Token HF (pour les modèles restreints, ex. Gemma).\n"
            f"État actuel : {existing}.\nColler le token (hf_…) :",
            QLineEdit.EchoMode.Password,
        )
        if ok and token.strip():
            try:
                set_hf_token(token)
                self._statusbar.showMessage("Token HuggingFace enregistré.")
            except Exception as exc:
                self._statusbar.showMessage(f"Échec enregistrement token : {exc}")

    def _vlm_candidate_params(self) -> tuple[int, int]:
        return self._vlm_line_slider.value(), self._vlm_blob_slider.value()

    def _show_vlm_candidates(self) -> None:
        """Aperçu (sans VLM) des zones qui seront analysées."""
        from core.vlm_refine import candidates_overlay
        from ui.vlm_candidates_dialog import VLMCandidatesDialog
        img  = self._mask_panel.get_display_image()
        mask = self._mask_panel.get_mask()
        if mask is None or not mask.any():
            self._statusbar.showMessage("Le masque est vide — lancez d'abord la détection.")
            return
        min_len, min_thick = self._vlm_candidate_params()
        rgb, nl, nb, nk = candidates_overlay(img, mask, min_len, min_thick)
        VLMCandidatesDialog(self, rgb, nl, nb, nk).exec()

    def _unload_vlm(self) -> None:
        from core.vlm_refine import VLMRefiner
        VLMRefiner.get().unload()
        self._vlm_unload_btn.setEnabled(False)
        self._update_vram_gauge()
        self._statusbar.showMessage("VLM déchargé — VRAM libérée.")

    # ── Jauge VRAM / déchargement global ──────────────────────────────────────

    def _update_vram_gauge(self) -> None:
        from core.model_memory import gpu_memory
        info = gpu_memory()
        if info is None:
            self._vram_bar.setMaximum(100)
            self._vram_bar.setValue(0)
            self._vram_bar.setFormat("VRAM n/a")
            return
        used, total = info
        pct = used / total if total else 0.0
        self._vram_bar.setMaximum(max(1, total))
        self._vram_bar.setValue(used)
        self._vram_bar.setFormat(f"VRAM {used/1024:.1f}/{total/1024:.1f} Go")
        chunk = "#5a8f5a" if pct < 0.6 else ("#b89a3a" if pct < 0.85 else "#c05a5a")
        self._vram_bar.setStyleSheet(
            "QProgressBar { background:#14142a; border:1px solid #2a2a4a;"
            " border-radius:4px; color:#dde; font-size:9px; text-align:center; }"
            f"QProgressBar::chunk {{ background:{chunk}; border-radius:3px; }}"
        )

    def _unload_all_models(self) -> None:
        if (self._worker is not None and self._worker.isRunning()) or \
           (self._full_preview_worker is not None and self._full_preview_worker.isRunning()) or \
           (self._vlm_worker is not None):
            self._statusbar.showMessage("Impossible de décharger pendant un traitement.")
            return
        from core.model_memory import unload_all_models
        freed = unload_all_models(list(self._steps_by_id.values()))
        self._vlm_unload_btn.setEnabled(False)
        self._update_vram_gauge()
        self._statusbar.showMessage(f"{len(freed)} modèle(s) déchargé(s) — VRAM libérée.")

    def _vlm_refine(self) -> None:
        """Lance l'affinage VLM du masque courant (en tâche de fond)."""
        if self._vlm_worker is not None:
            return
        img  = self._mask_panel.get_display_image()
        mask = self._mask_panel.get_mask()
        if img is None or img.size <= 3:
            self._statusbar.showMessage("Aucune image chargée.")
            return
        if mask is None or not mask.any():
            self._statusbar.showMessage("Le masque est vide — lancez d'abord la détection.")
            return

        model_id = self._vlm_model_combo.currentData()
        min_len, min_thick = self._vlm_candidate_params()
        self._vlm_refine_btn.setEnabled(False)
        self._vlm_progress.setValue(0)
        self._vlm_progress.setFormat("préparation…")
        self._vlm_progress.setVisible(True)
        self._statusbar.showMessage(
            "Affinage VLM… (1er usage : téléchargement du modèle, peut être long)"
        )
        worker = _VLMRefineWorker(img, mask, model_id, min_len, min_thick)
        worker.progress.connect(self._on_vlm_progress)
        worker.done.connect(self._on_vlm_done)
        worker.failed.connect(self._on_vlm_failed)
        worker.finished.connect(self._on_vlm_finished)
        self._vlm_worker = worker
        worker.start()

    def _on_vlm_progress(self, k: int, total: int) -> None:
        if total > 0:
            self._vlm_progress.setMaximum(total)
            self._vlm_progress.setValue(k)
            self._vlm_progress.setFormat(f"%v / %m  candidats")
            self._statusbar.showMessage(f"Affinage VLM : analyse {k}/{total}…")
        else:
            self._vlm_progress.setMaximum(0)  # indéterminé (chargement modèle)
            self._vlm_progress.setFormat("chargement du modèle…")

    def _on_vlm_done(self, result) -> None:
        self._vlm_last_result = result
        self._vlm_convo_btn.setEnabled(True)
        self._vlm_unload_btn.setEnabled(True)
        # Entrer en mode REVUE (rien n'est supprimé tout de suite) : les candidats
        # s'affichent en violet (décor) / vert (défaut) sur le masque.
        self._mask_panel.enter_review(
            result.review_labels, result.review_categories, result.review_reasons)
        self._vlm_apply_btn.setEnabled(True)
        n_scene, n_def = self._mask_panel.review_counts()
        self._statusbar.showMessage(
            f"VLM ({result.model_id.split('/')[-1]}) en {result.elapsed:.0f}s : "
            f"{n_scene} décor (violet), {n_def} défaut (vert). "
            f"Survolez pour la raison, double-clic pour corriger, puis « Appliquer »."
        )

    def _vlm_apply_review(self) -> None:
        """Applique la revue : supprime les zones décor (violet) du masque."""
        if not self._mask_panel.in_review():
            self._vlm_apply_btn.setEnabled(False)
            return
        n = self._mask_panel.apply_review()
        self._vlm_apply_btn.setEnabled(False)
        self._statusbar.showMessage(f"{n} zone(s) décor supprimée(s) du masque.")

    def _on_vlm_failed(self, message: str) -> None:
        first = message.splitlines()[0] if message else "erreur inconnue"
        self._statusbar.showMessage(f"Erreur VLM : {first}")
        # Trace complète dans une boîte de dialogue (utile pour déboguer un modèle).
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Affinage VLM — erreur")
        box.setText("L'affinage par le VLM a échoué.")
        box.setInformativeText(first)
        box.setDetailedText(message)
        box.exec()

    def _on_vlm_finished(self) -> None:
        self._vlm_refine_btn.setEnabled(True)
        self._vlm_progress.setVisible(False)
        self._vlm_worker = None

    def _show_vlm_conversation(self) -> None:
        if self._vlm_last_result is None:
            return
        from ui.vlm_conversation_dialog import VLMConversationDialog
        VLMConversationDialog(self, self._vlm_last_result).exec()

    # ══════════════════════════════════════════════════════════════════════════
    # Yeux rouges — détection auto
    # ══════════════════════════════════════════════════════════════════════════

    def _redeye_auto_detect(self) -> None:
        """Détecte les iris automatiquement et ajoute les cercles dans le masque."""
        from PyQt6.QtWidgets import QApplication
        img = getattr(self, "_current_orig", None)
        if img is None:
            self._statusbar.showMessage("Aucune image chargée.")
            return
        redeye_step = self._steps_by_id.get("redeye")
        if redeye_step is None or not hasattr(redeye_step, "detect_iris_mask"):
            return

        self._statusbar.showMessage("Détection des iris en cours…")
        QApplication.processEvents()

        try:
            new_mask = redeye_step.detect_iris_mask(img)
        except Exception as exc:
            self._statusbar.showMessage(f"Erreur détection : {exc}")
            return

        if new_mask is None:
            self._statusbar.showMessage("Aucun iris détecté automatiquement.")
            return

        existing = self._redeye_panel.get_mask()
        merged   = cv2.bitwise_or(existing, new_mask)
        self._redeye_panel._canvas.set_mask(merged)
        n_eyes = int(cv2.connectedComponents(new_mask)[0]) - 1
        self._statusbar.showMessage(f"{n_eyes} iris détecté(s) — masque mis à jour.")

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

    def _refresh_all_strip_thumbs(self) -> None:
        """Met à jour les vignettes batch depuis les exports existants."""
        if not self._session.output_dir:
            return
        mgr = self._session.get_export_manager()
        sources = [os.path.splitext(cfg.filename) for cfg in self._session.images]
        selected = mgr.select_thumbnail_exports(sources)
        items = []
        for cfg in self._session.images:
            key = os.path.splitext(cfg.filename)
            entry, is_best = selected.get(key, (None, False))
            if entry is not None:
                items.append((cfg.file_path, entry.image_path, is_best))
        self._strip.update_result_thumbs_from_paths(items)

    def _refresh_strip_thumb(
        self,
        cfg: BatchImageConfig,
        restore_original: bool = True,
    ) -> None:
        """Affiche l'originale, le meilleur export, ou le dernier export."""
        if not self._session.output_dir:
            if restore_original:
                self._strip.use_original_thumb(cfg.file_path)
            return

        mgr = self._session.get_export_manager()
        entry, is_best = self._strip_export_for_cfg(mgr, cfg)
        if entry is None:
            if restore_original:
                self._strip.use_original_thumb(cfg.file_path)
            return

        img = mgr.load_image(entry)
        if img is None:
            if restore_original:
                self._strip.use_original_thumb(cfg.file_path)
            return
        self._strip.update_result_thumb(cfg.file_path, img, is_best=is_best)

    @staticmethod
    def _strip_export_for_cfg(mgr: ExportManager, cfg: BatchImageConfig):
        """Retourne l'export à utiliser pour la miniature et s'il est retenu."""
        stem, ext = os.path.splitext(cfg.filename)
        exports = mgr.list_exports(stem, ext)
        if not exports:
            return None, False
        best_idx = mgr.get_best_index(stem)
        if best_idx is not None:
            for entry in exports:
                if entry.index == best_idx:
                    return entry, True
        return exports[-1], False

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
        if self._current_orig is not None:
            # Recharger les panneaux d'édition avec les masques restaurés
            # (sinon le panneau reste vide et le prochain _sync_editor_state
            #  réécraserait cfg.inpaint_mask avec ce masque vide).
            result  = self._get_result_from_disk(cfg)
            display = result if result is not None else self._current_orig
            self._mask_panel.set_image(display, cfg.inpaint_mask)
            self._redeye_panel.set_image(self._current_orig, cfg.redeye_mask)
            self._crop_panel.set_image(self._current_orig, cfg.crop_rect)
            # Le cache de détection d'artefacts est lié à l'image précédente
            self._artifact_probs = None
            self._artifact_base_mask = None
            self._artifact_base_image = None
            self._artifact_spots = None
            self._artifact_spots_dev = -1.0
        self._update_result_diff_indicator()
        self._statusbar.showMessage("Configuration restaurée depuis l'export.")
        self._schedule_preview_update()

    def _on_detail_deleted(self, entry) -> None:
        """Un export a été supprimé — rafraîchir la mosaïque."""
        if self._current_cfg:
            self._update_dest_view(self._current_cfg, force=True)
            self._refresh_export_dropdown(self._current_cfg)
            self._refresh_strip_thumb(self._current_cfg)
            self._refresh_retained_count()
            self._update_result_diff_indicator()

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
        self._refresh_strip_thumb(cfg)
        self._refresh_retained_count()

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

        # Collecter les exports retenus avec leur photo source pour recopier l'EXIF
        items: list[tuple[str, str, str]] = []  # (stem, export_path, source_path)
        for cfg in configs:
            stem, ext = os.path.splitext(cfg.filename)
            best = mgr.get_best_entry(stem, ext)
            if best and os.path.exists(best.image_path):
                items.append((stem, best.image_path, cfg.file_path))

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
        for i, (stem, src_path, source_path) in enumerate(items):
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
            ok = write_jpeg_with_source_exif(
                img,
                jpg_path,
                source_path,
                quality=95,
                optimize=True,
                progressive=True,
            )
            if ok:
                exported += 1

        progress.setValue(len(items))
        self._statusbar.showMessage(
            f"Export terminé : {exported}/{len(items)} images JPEG → {dest}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Indicateurs de configuration
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_retained_count(self) -> None:
        """Met à jour le compteur visuel des photos retenues."""
        if not hasattr(self, "_retained_count_lbl"):
            return
        total = len(self._session.images)
        retained = 0
        if self._session.output_dir:
            mgr = self._session.get_export_manager()
            sources = [os.path.splitext(cfg.filename) for cfg in self._session.images]
            selected = mgr.select_thumbnail_exports(sources)
            retained = sum(1 for _, is_best in selected.values() if is_best)

        self._retained_count_lbl.setText(f"★  {retained} / {total} retenue(s)")
        self._retained_count_lbl.setToolTip(
            "Nombre de photos marquées comme retenues dans le batch"
        )
        complete = total > 0 and retained == total
        self._has_unretained = total > 0 and not complete
        bg = "#173322" if complete else "#3a2a12"
        fg = "#a8efb3" if complete else "#ffd18a"
        border = "#347a4a" if complete else "#8f6a24"
        self._retained_count_lbl.setStyleSheet(
            f"color:{fg}; background:{bg}; border:1px solid {border};"
            " border-radius:5px; padding:5px 10px; font-size:12px;"
            " font-weight:800;"
        )
        if hasattr(self, "_first_unretained_btn"):
            self._first_unretained_btn.setEnabled(self._has_unretained)

    def _latest_export_recipe_for_cfg(self, cfg: BatchImageConfig) -> Optional[dict]:
        """Retourne la recette du dernier export de l'image courante."""
        if not self._session.output_dir:
            return None
        mgr = self._session.get_export_manager()
        stem, ext = os.path.splitext(cfg.filename)
        exports = mgr.list_exports(stem, ext)
        if not exports:
            return None
        return mgr.load_recipe(exports[-1])

    def _update_result_diff_indicator(self) -> None:
        """Compare les étapes courantes au dernier résultat de l'image."""
        if not hasattr(self, "_step_list"):
            return
        cfg = self._current_cfg
        recipe = self._latest_export_recipe_for_cfg(cfg) if cfg is not None else None
        if recipe is None:
            self._step_list.set_result_diff(set())
            self._set_recipe_diff_state(False, 0)
            return

        changed = changed_step_details(
            self._step_list.get_order(),
            self._step_list.get_enabled(),
            self._step_list.get_all_params(),
            list(recipe.get("step_order", [])),
            dict(recipe.get("step_enabled", {})),
            dict(recipe.get("step_params", {})),
            self._param_labels_by_step(),
        )
        self._step_list.set_result_diff(changed)
        self._set_recipe_diff_state(True, len(changed))

    def _param_labels_by_step(self) -> dict[str, dict[str, str]]:
        """Retourne les libellés UI des paramètres, groupés par étape."""
        labels: dict[str, dict[str, str]] = {}
        for step in ALL_STEPS:
            labels[step.id] = {
                pdef["key"]: pdef.get("label", pdef["key"])
                for pdef in getattr(step, "param_defs", [])
            }
        return labels

    def _set_recipe_diff_state(self, has_result: bool, changed_count: int) -> None:
        if not hasattr(self, "_recipe_diff_lbl"):
            return
        if not has_result:
            text = "Aucun résultat"
            style = (
                "color:#778; background:#202038; border:1px solid #303050;"
                " border-radius:4px; padding:2px 6px; font-size:9px;"
            )
        elif changed_count:
            text = f"{changed_count} modif."
            style = (
                "color:#ffd18a; background:#3a2a12; border:1px solid #8f6a24;"
                " border-radius:4px; padding:2px 6px; font-size:9px; font-weight:700;"
            )
        else:
            text = "À jour"
            style = (
                "color:#9ee6a8; background:#163020; border:1px solid #2f6a42;"
                " border-radius:4px; padding:2px 6px; font-size:9px; font-weight:700;"
            )
        self._recipe_diff_lbl.setText(text)
        self._recipe_diff_lbl.setStyleSheet(style)

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
