"""batch_mixins/genref.py — Onglet « Réf. IA » : couleur par référence FLUX.

Workflow batch de l'étape genref (voir core/genref.py) :

  1. L'« entrée » est le preview complet jusqu'à l'étape genref (exclue),
     calculé automatiquement (asynchrone) à l'entrée dans l'onglet.
  2. Le VLM rédige un prompt (éditable dans le bandeau au-dessus du
     triptyque), FLUX génère la référence → enregistrée comme VERSION
     sidecar à côté de la source ({stem}.genref.NNN.json/.png).
  3. La liste de droite montre l'historique des versions (jamais supprimé
     automatiquement) ; sélectionner une version l'active pour l'image
     (cfg.genref_version → recette/exports) ; la LUT est ré-ajustée à la
     volée sur l'entrée courante par l'étape (fit ~0,3 s, cache).
  4. Triptyque entrée / référence / résultat ; double-clic sur une image →
     agrandissement plein onglet avec zoom partagé entre onglets.

Si l'étape est cochée mais qu'aucune version n'est active, elle est ignorée
au traitement — le statut de la sidebar le rend explicite.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QPlainTextEdit, QProgressBar, QListWidget, QListWidgetItem,
    QStackedWidget, QMessageBox, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, QTimer, QEvent, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from core import genref, genref_store
from core.pipeline import PipelineWorker
from ui.image_view import SyncedImageView
from ui.batch_window_constants import _TAB_GENREF


# ══════════════════════════════════════════════════════════════════════════
# Widgets et workers internes
# ══════════════════════════════════════════════════════════════════════════

class _TriptychPane(QLabel):
    """Vignette cliquable du triptyque (double-clic → agrandissement)."""

    double_clicked = pyqtSignal(str)    # rôle : "input" | "ref" | "result"

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self._role = role
        self._img: Optional[np.ndarray] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background:#101018; color:#445; border:1px solid #26263a;"
            " border-radius:4px; font-size:11px;"
        )
        self.setText("—")
        self.setToolTip("Double-clic : agrandir (zoom partagé entre onglets)")

    def set_bgr(self, img: Optional[np.ndarray]) -> None:
        self._img = img
        self._rescale()

    def image(self) -> Optional[np.ndarray]:
        return self._img

    def _rescale(self) -> None:
        if self._img is None:
            self.setPixmap(QPixmap())
            self.setText("—")
            return
        rgb = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.strides[0], QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def mouseDoubleClickEvent(self, event):
        if self._img is not None:
            self.double_clicked.emit(self._role)


class _GenRefGenWorker(QThread):
    """Génération FLUX + enregistrement de la version sidecar."""

    phase    = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    done     = pyqtSignal(object)          # GenRefVersion
    failed   = pyqtSignal(str)

    def __init__(self, source_path: str, input_bgr: np.ndarray,
                 prompt: str, style: str, seed: int,
                 cast: str, items: list, parent=None):
        super().__init__(parent)
        self._source_path = source_path
        self._input = input_bgr
        self._prompt = prompt
        self._style = style
        self._seed = seed
        self._cast = cast
        self._items = items

    def run(self):
        try:
            try:
                from core.model_memory import unload_all_models
                from steps import ALL_STEPS
                unload_all_models(ALL_STEPS)
            except Exception:
                pass
            self.phase.emit("Génération de la référence (FLUX)…")
            ref = genref.GenRefEngine.get().generate(
                self._input, self._prompt, seed=self._seed,
                on_progress=self.progress.emit)
            version = genref_store.save_version(
                self._source_path, ref, self._prompt, self._style,
                self._seed, cast=self._cast, items=self._items)
            self.done.emit(version)
        except Exception as exc:
            self.failed.emit(str(exc))


class _GenRefPromptWorker(QThread):
    """Rédaction du prompt par le VLM."""

    done   = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, img_bgr: np.ndarray, style: str, parent=None):
        super().__init__(parent)
        self._img = img_bgr
        self._style = style

    def run(self):
        try:
            self.done.emit(genref.write_prompt(self._img, self._style))
        except Exception as exc:
            self.failed.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════
# Mixin
# ══════════════════════════════════════════════════════════════════════════

class GenRefMixin:
    """Onglet « Réf. IA » : génération, historique de versions, triptyque."""

    # ── Construction de l'onglet ─────────────────────────────────────────

    def _build_genref_tab(self, sidebar_width: int) -> QWidget:
        # État
        self._genref_input_img: Optional[np.ndarray] = None
        self._genref_input_worker: Optional[PipelineWorker] = None
        self._genref_gen_worker: Optional[_GenRefGenWorker] = None
        self._genref_prompt_worker: Optional[_GenRefPromptWorker] = None
        self._genref_versions: list[genref_store.GenRefVersion] = []
        self._genref_prompt_info: dict = {}
        self._genref_enlarged_role: Optional[str] = None

        wrapper = QWidget()
        lay = QHBoxLayout(wrapper)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Zone gauche : bandeau prompt + triptyque/agrandi ──────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(2)

        self._genref_prompt_edit = QPlainTextEdit()
        self._genref_prompt_edit.setFixedHeight(76)
        self._genref_prompt_edit.setPlaceholderText(
            "Prompt FLUX — « 🤖 Rédiger » pour une proposition du VLM, ou "
            "écrire en anglais (style « manuel »). Nommer la couleur du "
            "voile (« magenta ») renforce beaucoup la correction."
        )
        self._genref_prompt_edit.setStyleSheet(
            "QPlainTextEdit { background:#14141f; color:#cdd;"
            " border:1px solid #2a2a4a; border-radius:4px; font-size:11px; }"
        )
        left_lay.addWidget(self._genref_prompt_edit)

        self._genref_stack = QStackedWidget()

        # Page 0 : triptyque
        trip = QWidget()
        trip_lay = QHBoxLayout(trip)
        trip_lay.setContentsMargins(0, 0, 0, 0)
        trip_lay.setSpacing(4)
        self._genref_panes: dict[str, _TriptychPane] = {}
        for role, title in (("input", "Entrée (pipeline jusqu'à l'étape)"),
                            ("ref", "Référence FLUX (palette seulement)"),
                            ("result", "Résultat LUT")):
            col = QWidget()
            col_lay = QVBoxLayout(col)
            col_lay.setContentsMargins(0, 0, 0, 0)
            col_lay.setSpacing(2)
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#88a; font-size:10px;")
            col_lay.addWidget(lbl)
            pane = _TriptychPane(role)
            pane.double_clicked.connect(self._genref_enlarge)
            self._genref_panes[role] = pane
            col_lay.addWidget(pane, stretch=1)
            trip_lay.addWidget(col, stretch=1)
        self._genref_stack.addWidget(trip)

        # Page 1 : vue agrandie (zoom partagé avec les autres onglets)
        self._genref_big_view = SyncedImageView()
        self._genref_big_view.setToolTip("Double-clic : revenir au triptyque")
        self._genref_big_view.viewport().installEventFilter(self)
        self._genref_stack.addWidget(self._genref_big_view)

        left_lay.addWidget(self._genref_stack, stretch=1)
        lay.addWidget(left, stretch=1)

        # ── Sidebar droite ────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(sidebar_width)
        sidebar.setStyleSheet("background:#16162a;")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(8, 10, 8, 10)
        sb.setSpacing(8)

        self._genref_status_lbl = QLabel("")
        self._genref_status_lbl.setWordWrap(True)
        self._genref_status_lbl.setStyleSheet(
            "color:#9ab; font-size:10px; border:1px solid #2a2a4a;"
            " border-radius:3px; padding:4px; background:#111124;"
        )
        sb.addWidget(self._genref_status_lbl)

        refresh_row = QHBoxLayout()
        refresh_row.setSpacing(4)
        lbl_hist = QLabel("Versions générées")
        lbl_hist.setStyleSheet("color:#aab; font-size:11px; font-weight:600;")
        refresh_row.addWidget(lbl_hist, stretch=1)
        self._genref_refresh_btn = QPushButton("↻")
        self._genref_refresh_btn.setFixedSize(22, 22)
        self._genref_refresh_btn.setToolTip(
            "Recalculer l'entrée (pipeline jusqu'à l'étape) et rafraîchir"
        )
        self._genref_refresh_btn.setStyleSheet(self._genref_btn_css())
        self._genref_refresh_btn.clicked.connect(
            lambda: self._genref_update_input(force=True))
        refresh_row.addWidget(self._genref_refresh_btn)
        sb.addLayout(refresh_row)

        self._genref_versions_list = QListWidget()
        self._genref_versions_list.setStyleSheet(
            "QListWidget { background:#111124; color:#bcd; font-size:10px;"
            "  border:1px solid #2a2a4a; border-radius:3px; }"
            "QListWidget::item { padding:3px 4px; }"
            "QListWidget::item:selected { background:#2a6496; color:#fff; }"
        )
        self._genref_versions_list.setMinimumHeight(110)
        self._genref_versions_list.itemSelectionChanged.connect(
            self._genref_on_version_selected)
        sb.addWidget(self._genref_versions_list, stretch=1)

        self._genref_delete_btn = QPushButton("🗑  Supprimer la version")
        self._genref_delete_btn.setToolTip(
            "Supprime définitivement les fichiers de la version sélectionnée\n"
            "(l'historique n'est jamais supprimé automatiquement)."
        )
        self._genref_delete_btn.setStyleSheet(self._genref_btn_css())
        self._genref_delete_btn.clicked.connect(self._genref_delete_selected)
        sb.addWidget(self._genref_delete_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#2a2a3a;")
        sep.setFixedHeight(1)
        sb.addWidget(sep)

        style_row = QHBoxLayout()
        style_row.setSpacing(4)
        lbl_style = QLabel("Style")
        lbl_style.setStyleSheet("color:#aab; font-size:10px;")
        style_row.addWidget(lbl_style)
        self._genref_style_combo = QComboBox()
        self._genref_style_combo.addItems(genref.PROMPT_STYLES)
        self._genref_style_combo.setStyleSheet("font-size:10px;")
        style_row.addWidget(self._genref_style_combo, stretch=1)
        lbl_seed = QLabel("Graine")
        lbl_seed.setStyleSheet("color:#aab; font-size:10px;")
        style_row.addWidget(lbl_seed)
        self._genref_seed_spin = QSpinBox()
        self._genref_seed_spin.setRange(0, 9999)
        self._genref_seed_spin.setValue(42)
        self._genref_seed_spin.setStyleSheet("font-size:10px;")
        style_row.addWidget(self._genref_seed_spin)
        sb.addLayout(style_row)

        self._genref_write_btn = QPushButton("🤖  Rédiger le prompt (VLM)")
        self._genref_write_btn.setStyleSheet(self._genref_btn_css())
        self._genref_write_btn.clicked.connect(self._genref_write_prompt)
        sb.addWidget(self._genref_write_btn)

        self._genref_gen_btn = QPushButton("✨  Générer (FLUX, ~2-3 min)")
        self._genref_gen_btn.setStyleSheet(self._genref_btn_css(accent=True))
        self._genref_gen_btn.clicked.connect(self._genref_generate)
        sb.addWidget(self._genref_gen_btn)

        self._genref_progress = QProgressBar()
        self._genref_progress.setFixedHeight(12)
        self._genref_progress.setVisible(False)
        sb.addWidget(self._genref_progress)

        lay.addWidget(sidebar)
        return wrapper

    @staticmethod
    def _genref_btn_css(accent: bool = False) -> str:
        bg, hover = ("#2a6496", "#3a84b6") if accent else ("#1e3a52", "#2a5577")
        return (f"QPushButton {{ background:{bg}; color:#d8ecf7;"
                f" border-radius:4px; padding:5px 6px; font-size:10px; }}"
                f"QPushButton:hover {{ background:{hover}; }}"
                f"QPushButton:disabled {{ background:#222230; color:#556; }}")

    # ── Agrandissement / retour (zoom partagé) ───────────────────────────

    def _genref_enlarge(self, role: str) -> None:
        pane = self._genref_panes.get(role)
        if pane is None or pane.image() is None:
            return
        self._genref_enlarged_role = role
        self._genref_prompt_edit.setVisible(False)   # superposition exacte
        self._genref_big_view.set_image(pane.image())
        self._genref_stack.setCurrentIndex(1)
        if self._shared_zoom is not None:
            zoom = self._shared_zoom
            QTimer.singleShot(
                0, lambda: self._genref_big_view.apply_zoom_state(*zoom))

    def _genref_back_to_triptych(self) -> None:
        state = self._genref_big_view.get_zoom_state()
        if state[0] > 0:
            self._shared_zoom = state
        self._genref_enlarged_role = None
        self._genref_prompt_edit.setVisible(True)
        self._genref_stack.setCurrentIndex(0)

    def eventFilter(self, obj, event):
        if (hasattr(self, "_genref_big_view")
                and obj is self._genref_big_view.viewport()
                and event.type() == QEvent.Type.MouseButtonDblClick):
            self._genref_back_to_triptych()
            return True
        return super().eventFilter(obj, event)

    def _genref_tab_canvas(self):
        """Canvas zoomable de l'onglet (None si triptyque affiché)."""
        if getattr(self, "_genref_enlarged_role", None) is not None:
            return self._genref_big_view
        return None

    # ── Entrée dans l'onglet / changement d'image ────────────────────────

    def _genref_on_tab_entered(self) -> None:
        self._genref_refresh_versions()
        self._genref_update_input()

    def _genref_on_image_changed(self) -> None:
        """Appelé à chaque navigation : réinitialise l'état de l'onglet."""
        self._genref_input_img = None
        self._genref_prompt_info = {}
        if hasattr(self, "_genref_panes"):
            for pane in self._genref_panes.values():
                pane.set_bgr(None)
            self._genref_prompt_edit.setPlainText("")
            if self._genref_enlarged_role is not None:
                self._genref_back_to_triptych()
            if self._tabs.currentIndex() == _TAB_GENREF:
                self._genref_on_tab_entered()

    # ── Calcul de l'entrée (pipeline jusqu'à l'étape genref) ─────────────

    def _genref_update_input(self, force: bool = False) -> None:
        cfg = self._current_cfg
        if cfg is None or self._current_orig is None:
            return
        if self._genref_input_img is not None and not force:
            self._genref_refresh_triptych()
            return
        if (self._genref_input_worker is not None
                and self._genref_input_worker.isRunning()):
            return

        self._sync_editor_state()
        self._inject_instance_state(cfg)

        before_ids = []
        for sid in cfg.step_order:
            if sid == "genref":
                break
            if cfg.step_enabled.get(sid, True) and sid in self._steps_by_id:
                before_ids.append(sid)

        if not before_ids:
            # genref en tête de pipeline : l'entrée est l'original
            self._genref_input_img = self._current_orig.copy()
            self._genref_refresh_triptych()
            return

        self._genref_status_lbl.setText(
            "⏳ Calcul de l'entrée (pipeline jusqu'à l'étape)…")
        steps = [self._steps_by_id[sid] for sid in before_ids]
        self._genref_input_worker = PipelineWorker(
            run_id      = -2,
            steps       = steps,
            initial_img = self._current_orig.copy(),
            all_params  = cfg.step_params,
            context     = {},
            step_order  = before_ids,
            step_enabled= cfg.step_enabled,
        )
        self._genref_input_worker.all_done.connect(self._genref_on_input_done)
        self._genref_input_worker.start()

    def _genref_on_input_done(self, entry) -> None:
        result = (entry.step_results.get(entry.completed_steps[-1])
                  if entry.completed_steps else None)
        self._genref_input_img = (result if result is not None
                                  else self._current_orig)
        self._genref_refresh_triptych()

    # ── Versions ──────────────────────────────────────────────────────────

    def _genref_refresh_versions(self) -> None:
        cfg = self._current_cfg
        lst = self._genref_versions_list
        lst.blockSignals(True)
        lst.clear()
        self._genref_versions = []
        item_none = QListWidgetItem("— aucune (étape ignorée) —")
        item_none.setData(Qt.ItemDataRole.UserRole, None)
        lst.addItem(item_none)
        selected_row = 0
        if cfg is not None:
            self._genref_versions = genref_store.list_versions(cfg.file_path)
            for i, v in enumerate(self._genref_versions):
                item = QListWidgetItem(v.label)
                item.setData(Qt.ItemDataRole.UserRole, v.index)
                item.setToolTip(v.prompt)
                lst.addItem(item)
                if cfg.genref_version == v.index:
                    selected_row = i + 1
        lst.setCurrentRow(selected_row)
        lst.blockSignals(False)
        self._genref_update_status()

    def _genref_selected_version(self) -> Optional[genref_store.GenRefVersion]:
        item = self._genref_versions_list.currentItem()
        if item is None:
            return None
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return None
        for v in self._genref_versions:
            if v.index == index:
                return v
        return None

    def _genref_on_version_selected(self) -> None:
        cfg = self._current_cfg
        if cfg is None:
            return
        version = self._genref_selected_version()
        new_index = version.index if version is not None else None
        if cfg.genref_version != new_index:
            cfg.genref_version = new_index
            self._mark_customized()
        if version is not None and version.prompt:
            self._genref_prompt_edit.setPlainText(version.prompt)
        self._inject_instance_state(cfg)
        self._genref_refresh_triptych()
        self._schedule_preview_update()
        self._update_result_diff_indicator()

    def _genref_delete_selected(self) -> None:
        version = self._genref_selected_version()
        cfg = self._current_cfg
        if version is None or cfg is None:
            self._statusbar.showMessage("Sélectionner une version à supprimer.")
            return
        resp = QMessageBox.question(
            self, "Supprimer la version",
            f"Supprimer définitivement la version v{version.index:03d} "
            f"(prompt + image FLUX) ?\nCette action est irréversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        genref_store.delete_version(version)
        if cfg.genref_version == version.index:
            cfg.genref_version = None
            self._inject_instance_state(cfg)
            self._schedule_preview_update()
        self._genref_refresh_versions()
        self._genref_refresh_triptych()
        self._statusbar.showMessage(f"Version v{version.index:03d} supprimée.")

    # ── Triptyque ─────────────────────────────────────────────────────────

    def _genref_refresh_triptych(self) -> None:
        cfg = self._current_cfg
        panes = self._genref_panes
        panes["input"].set_bgr(self._genref_input_img)

        version = None
        if cfg is not None and cfg.genref_version is not None:
            version = genref_store.load_version(cfg.file_path,
                                                cfg.genref_version)
        ref = version.load_ref() if version is not None else None
        panes["ref"].set_bgr(ref)

        result = None
        if (ref is not None and self._genref_input_img is not None):
            step = self._steps_by_id.get("genref")
            panel = self._step_list.get_panel("genref")
            if step is not None and panel is not None:
                try:
                    result, _ = step.process(
                        self._genref_input_img, panel.get_params(), {})
                except Exception:
                    result = None
        panes["result"].set_bgr(result)

        # Si une vue est agrandie, la rafraîchir aussi
        role = self._genref_enlarged_role
        if role is not None:
            img = panes[role].image()
            if img is not None:
                self._genref_big_view.set_image(img, preserve_zoom=True)
        self._genref_update_status()

    def _genref_update_status(self) -> None:
        cfg = self._current_cfg
        if cfg is None:
            self._genref_status_lbl.setText("Aucune image sélectionnée.")
            return
        enabled = cfg.step_enabled.get("genref", False)
        parts = []
        if self._genref_input_img is None:
            parts.append("Entrée : non calculée (↻).")
        if cfg.genref_version is None:
            if enabled:
                parts.append("⚠ Étape cochée mais AUCUNE version active : "
                             "elle sera ignorée au traitement.")
            else:
                parts.append("Aucune version active.")
        else:
            parts.append(f"Version active : v{cfg.genref_version:03d} "
                         "(LUT ré-ajustée automatiquement sur l'entrée).")
            if not enabled:
                parts.append("ℹ Étape décochée dans le pipeline.")
        self._genref_status_lbl.setText("\n".join(parts))

    # ── Rédaction du prompt ───────────────────────────────────────────────

    def _genref_busy(self) -> bool:
        for w in (self._genref_prompt_worker, self._genref_gen_worker):
            if w is not None and w.isRunning():
                return True
        return False

    def _genref_set_busy(self, busy: bool, label: str = "") -> None:
        for w in (self._genref_write_btn, self._genref_gen_btn,
                  self._genref_style_combo, self._genref_seed_spin,
                  self._genref_delete_btn, self._genref_versions_list):
            w.setEnabled(not busy)
        if label:
            self._genref_status_lbl.setText(label)

    def _genref_write_prompt(self) -> None:
        if self._genref_busy():
            return
        if self._genref_input_img is None:
            self._statusbar.showMessage(
                "Entrée non calculée — utiliser ↻ d'abord.")
            return
        style = self._genref_style_combo.currentText()
        if style == "manuel":
            self._genref_status_lbl.setText(
                "Style « manuel » : écrire le prompt dans le bandeau.")
            return
        self._genref_set_busy(True, "⏳ Rédaction du prompt (VLM)…")
        self._genref_prompt_worker = _GenRefPromptWorker(
            self._genref_input_img, style, self)
        self._genref_prompt_worker.done.connect(self._genref_on_prompt_done)
        self._genref_prompt_worker.failed.connect(self._genref_on_failed)
        self._genref_prompt_worker.start()

    def _genref_on_prompt_done(self, info: dict) -> None:
        self._genref_set_busy(False)
        self._genref_prompt_info = info
        self._genref_prompt_edit.setPlainText(info.get("prompt", ""))
        cast = info.get("cast", "")
        self._genref_status_lbl.setText(
            f"Prompt rédigé (voile : {cast}) — éditable avant génération.")

    # ── Génération ────────────────────────────────────────────────────────

    def _genref_generate(self) -> None:
        cfg = self._current_cfg
        if cfg is None or self._genref_busy():
            return
        if self._worker is not None and self._worker.isRunning():
            self._statusbar.showMessage(
                "Génération indisponible pendant un traitement batch.")
            return
        if self._genref_input_img is None:
            self._statusbar.showMessage(
                "Entrée non calculée — utiliser ↻ d'abord.")
            return
        prompt = self._genref_prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.information(
                self, "Prompt manquant",
                "Rédiger le prompt (🤖) ou l'écrire dans le bandeau avant "
                "de générer.")
            return
        style = self._genref_style_combo.currentText()
        info = self._genref_prompt_info
        cast = info.get("cast") or genref.cast_name(self._genref_input_img)

        self._genref_set_busy(True, "⏳ Préparation (déchargement modèles)…")
        self._genref_progress.setVisible(True)
        self._genref_progress.setRange(0, 0)
        self._genref_gen_worker = _GenRefGenWorker(
            cfg.file_path, self._genref_input_img, prompt, style,
            int(self._genref_seed_spin.value()), cast,
            list(info.get("items", [])), self)
        self._genref_gen_worker.phase.connect(self._genref_status_lbl.setText)
        self._genref_gen_worker.progress.connect(self._genref_on_progress)
        self._genref_gen_worker.done.connect(self._genref_on_generated)
        self._genref_gen_worker.failed.connect(self._genref_on_failed)
        self._genref_gen_worker.start()

    def _genref_on_progress(self, i: int, n: int) -> None:
        self._genref_progress.setRange(0, n)
        self._genref_progress.setValue(i)

    def _genref_on_generated(self, version) -> None:
        self._genref_set_busy(False)
        self._genref_progress.setVisible(False)
        cfg = self._current_cfg
        # La nouvelle version devient la version active de l'image
        if cfg is not None and cfg.file_path == version.source_path:
            cfg.genref_version = version.index
            self._mark_customized()
            self._inject_instance_state(cfg)
        self._genref_refresh_versions()
        self._genref_refresh_triptych()
        self._schedule_preview_update()
        self._statusbar.showMessage(
            f"Référence v{version.index:03d} générée et activée.")
        if hasattr(self, "_notif"):
            from ui.notifications import Level
            self._notif.notify("Référence IA générée",
                               f"v{version.index:03d} — {version.style} / "
                               f"graine {version.seed}",
                               level=Level.SUCCESS, duration=5000)

    def _genref_on_failed(self, msg: str) -> None:
        self._genref_set_busy(False)
        self._genref_progress.setVisible(False)
        self._genref_status_lbl.setText("")
        QMessageBox.warning(self, "Échec", f"L'opération a échoué :\n{msg}")
