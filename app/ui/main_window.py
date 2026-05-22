"""ui/main_window.py — Fenêtre principale de l'application."""

from __future__ import annotations
from typing import Optional
import os
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QFileDialog, QStatusBar, QPushButton,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QKeySequence, QShortcut, QAction

from core.history import HistoryManager, HistoryEntry
from core.pipeline import PipelineWorker
from steps import ALL_STEPS
from ui.image_view import SyncedImageView, ndarray_to_qpixmap
from ui.step_panel import StepListWidget
from ui.thumbnail_strip import ThumbnailStrip
from ui.history_panel import HistoryPanel
from ui.control_panel import ControlPanel


# Étapes dont un changement de paramètre déclenche le mini-pipeline de preview rapide
_FAST_PREVIEW_IDS = frozenset({"color", "facehighlight", "autocolor", "wb"})


class MainApp(QMainWindow):
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
        self._apply_theme()
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
        splitter.setSizes([320, 1080])

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

        file_menu.addSeparator()
        quit_act = QAction("Quitter", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+R"), self, self._run_pipeline)
        QShortcut(QKeySequence("F"), self, self._fit_views)

    # ══════════════════════════════════════════════════════════════════════════
    # Image
    # ══════════════════════════════════════════════════════════════════════════

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une image",
            os.path.dirname(self._original_path) if self._original_path else "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;Tous (*)",
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            try:
                import tifffile
                import numpy as np
                raw = tifffile.imread(path)
                if raw.dtype != np.uint8:
                    raw = (raw / raw.max() * 255).astype(np.uint8)
                if raw.ndim == 2:
                    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                elif raw.shape[2] == 4:
                    raw = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
                img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            except Exception as e:
                self._status_bar.showMessage(f"Erreur lecture : {e}")
                return
        self._original      = img
        self._original_path = path
        self._clear_preview()   # annuler tout aperçu lié à l'image précédente
        self._history.clear()
        self._last_step_images.clear()
        self._active_run_id = None
        self._history_panel.clear()  # supprime tous les chips de l'image précédente

        # Réinitialiser l'overlay si actif
        self._overlay_step = None
        self._ctrl.step_list.reset_overlays()

        # Bande vide : uniquement "Original"
        self._thumb_strip.rebuild([], {})
        self._thumb_strip.update_image("original", img)
        self._thumb_strip.set_version_label("")

        # Vue A = original
        self._view_a = (None, "original")
        self._view_b = None
        self._label_b.setVisible(False)
        self._refresh_views()
        self._update_view_labels()
        self._update_thumb_highlights()

        self._status_bar.showMessage(
            f"Image ouverte : {os.path.basename(path)}  "
            f"({img.shape[1]}×{img.shape[0]} px)"
        )

    def _on_batch_requested(self) -> None:
        """Ouvre la fenêtre batch avec un snapshot de la configuration courante."""
        from core.batch import BatchSession, BatchImageConfig
        from ui.batch_window import BatchWindow

        # Si déjà ouverte, la ramener au premier plan
        if self._batch_window is not None:
            self._batch_window.raise_()
            self._batch_window.activateWindow()
            return

        # Choisir le dossier source
        start_dir = (
            os.path.dirname(self._original_path)
            if self._original_path
            else ""
        )
        folder = QFileDialog.getExistingDirectory(
            self, "Dossier d'images à traiter en batch", start_dir
        )
        if not folder:
            return

        # Snapshot de la config courante
        defaults = BatchImageConfig(
            file_path       = "",
            step_order      = list(self._ctrl.step_list.get_order()),
            step_enabled    = dict(self._ctrl.step_list.get_enabled()),
            step_params     = {
                k: dict(v)
                for k, v in self._ctrl.step_list.get_all_params().items()
            },
        )

        session = BatchSession()
        session.load_folder(folder, defaults)

        if not session.images:
            self._status_bar.showMessage("Aucune image trouvée dans ce dossier.")
            return

        self._batch_window = BatchWindow(self, session)
        self._batch_window.closed.connect(self._on_batch_closed)
        self._ctrl.set_running(True)   # désactive le bouton Lancer principal
        self._batch_window.show()

    def _on_batch_closed(self) -> None:
        self._batch_window = None
        self._ctrl.set_running(False)

    def _save_result(self):
        """Enregistre l'image affichée dans la vue A (Ctrl+S)."""
        img = self._get_image(*self._view_a)
        if img is None:
            self._status_bar.showMessage("Rien à enregistrer.")
            return
        stem = os.path.splitext(os.path.basename(self._original_path))[0]
        default = os.path.join(
            os.path.dirname(self._original_path),
            f"{stem}_restauree.jpg",
        )
        self._save_dialog(img, default)

    def _on_save_thumb(self, step_id: str):
        """Enregistre l'image correspondant à la vignette choisie."""
        run_id = None if step_id == "original" else self._active_run_id
        img = self._get_image(run_id, step_id)
        if img is None:
            self._status_bar.showMessage("Image non disponible.")
            return
        stem    = os.path.splitext(os.path.basename(self._original_path))[0]
        ext     = os.path.splitext(self._original_path)[1].lower() or ".jpg"
        step    = self._steps_by_id.get(step_id)
        sname   = step.short_name.lower().replace(" ", "_") if step else step_id
        ver     = f"_v{run_id}" if run_id is not None else ""
        default = os.path.join(
            os.path.dirname(self._original_path),
            f"{stem}_{sname}{ver}{ext}",
        )
        self._save_dialog(img, default)

    def _save_dialog(self, img, default_path: str):
        """Ouvre la boîte de dialogue de sauvegarde et écrit le fichier."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer l'image", default_path,
            "JPEG (*.jpg *.jpeg);;PNG sans perte (*.png);;TIFF sans perte (*.tif *.tiff)",
        )
        if not path:
            return
        self._save_image_to_path(img, path)

    def _save_image_to_path(self, img, path: str):
        """Sauvegarde img vers path en déduisant le format depuis l'extension."""
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".jpg", ".jpeg"):
                ok = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == ".png":
                ok = cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            else:  # .tif / .tiff ou autre
                ok = cv2.imwrite(path, img)
            if ok:
                self._status_bar.showMessage(f"Enregistrée : {os.path.basename(path)}")
            else:
                self._status_bar.showMessage(f"Erreur écriture : {path}")
        except Exception as e:
            self._status_bar.showMessage(f"Erreur : {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _run_pipeline(self):
        if self._original is None:
            self._status_bar.showMessage("Ouvrir d'abord une image (Ctrl+O)")
            return
        self._run_from_step(None)

    def _run_from(self, step_id: str):
        """Relancer à partir d'une étape donnée (à partir d'un résultat existant)."""
        self._run_from_step(step_id)

    def _run_from_step(self, from_step_id: Optional[str]):
        if self._worker and self._worker.isRunning():
            return

        # Déterminer les étapes à exécuter
        enabled_order = [
            s for s in self._step_order
            if self._step_enabled.get(s, True)
        ]
        steps = [self._steps_by_id[sid] for sid in enabled_order]

        if from_step_id is not None and from_step_id in enabled_order:
            start_idx = enabled_order.index(from_step_id)
            steps = steps[start_idx:]

        # Image d'entrée
        if from_step_id is not None:
            prev_idx = enabled_order.index(from_step_id) - 1 if from_step_id in enabled_order else -1
            if prev_idx >= 0:
                prev_id = enabled_order[prev_idx]
                entry   = self._history.latest()
                initial = entry.step_results.get(prev_id) if entry else None
                initial = initial if initial is not None else self._original
            else:
                initial = self._original
        else:
            initial = self._original

        # Contexte courant
        latest = self._history.latest()
        ctx = dict(latest.context) if latest else {}

        # Paramètres
        params  = self._ctrl.step_list.get_all_params()
        enabled = self._ctrl.step_list.get_enabled()

        run_id = self._history.next_id()

        # Reconstruire la bande avec les étapes planifiées (placeholders vides)
        step_names = {s.id: s.short_name for s in ALL_STEPS}
        self._thumb_strip.rebuild([s.id for s in steps], step_names)
        self._thumb_strip.update_image("original", self._original)
        self._thumb_strip.set_version_label(f"Run #{run_id} \u2014 en cours\u2026")
        self._update_thumb_highlights()

        self._worker = PipelineWorker(
            run_id=run_id,
            steps=steps,
            initial_img=initial,
            all_params=params,
            context=ctx,
            step_order=self._step_order,
            step_enabled=enabled,
        )
        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.step_failed.connect(self._on_step_failed)
        self._worker.all_done.connect(self._on_all_done)
        self._clear_preview()   # annuler tout aperçu en cours avant de lancer
        self._worker.start()

        self._ctrl.set_running(True)
        self._status_bar.showMessage(f"Run #{run_id} — démarré…")

    def _stop_pipeline(self):
        if self._worker:
            self._worker.cancel()

    # ── Slots du worker ─────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_step_started(self, step_id: str):
        panel = self._ctrl.step_list.get_panel(step_id)
        if panel:
            panel.set_state("running")
        step = self._steps_by_id.get(step_id)
        self._status_bar.showMessage(f"⏳ {step.name if step else step_id} en cours…")

    @pyqtSlot(str, object, dict)
    def _on_step_done(self, step_id: str, img: np.ndarray, extras: dict):
        self._last_step_images[step_id] = img
        panel = self._ctrl.step_list.get_panel(step_id)
        if panel:
            panel.set_state("ok")
            if "effective_params" in extras:
                panel.set_params(extras["effective_params"])
        self._thumb_strip.update_image(step_id, img)
        # Mise à jour en direct de la vue A si elle cible cette étape
        if self._view_a[1] == step_id:
            self._view_a_widget.set_image(img)

    @pyqtSlot(str, str)
    def _on_step_failed(self, step_id: str, msg: str):
        panel = self._ctrl.step_list.get_panel(step_id)
        if panel:
            panel.set_state("error", msg[:60])
        self._status_bar.showMessage(f"✗ {step_id} : {msg}")

    @pyqtSlot(object)
    def _on_all_done(self, entry: HistoryEntry):
        self._history.add(entry)
        self._active_run_id = entry.run_id
        self._history_panel.add_entry(entry)
        self._history_panel.set_active(entry.run_id)
        self._thumb_strip.set_version_label(f"v{entry.run_id} — {entry.time_str}")

        # Vue A → dernière étape terminée
        if entry.completed_steps:
            last_step = entry.completed_steps[-1]
            self._view_a = (entry.run_id, last_step)
            self._refresh_view_a()
            self._update_view_labels()
            self._update_thumb_highlights()

        self._ctrl.set_running(False)
        self._status_bar.showMessage(
            f"Run #{entry.run_id} terminé — {len(entry.completed_steps)} étape(s)"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Overlay détections (actif sans recalcul du pipeline)
    # ══════════════════════════════════════════════════════════════════════════

    @pyqtSlot(str, bool)
    def _on_overlay_toggled(self, step_id: str, enabled: bool):
        """Active/désactive l'overlay de détection sans relancer le pipeline."""
        self._overlay_step = step_id if enabled else None
        self._refresh_view_a()
        self._update_view_labels()

    @pyqtSlot(str)
    def _on_mask_edit_requested(self, step_id: str):
        """Ouvre l'éditeur de masque pour l'étape step_id."""
        from PyQt6.QtWidgets import QDialog
        from ui.mask_editor import MaskEditorDialog

        step = self._steps_by_id.get(step_id)
        if step is None or not hasattr(step, "set_mask"):
            return
        img = self._get_step_input_image(step_id)
        if img is None:
            self._status_bar.showMessage("Ouvrez une image avant de peindre le masque.")
            return
        dlg = MaskEditorDialog(self, img, step.get_mask())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            step.set_mask(dlg.get_mask())
            self._status_bar.showMessage(f"Masque mis à jour pour « {step.name} ».")

    @pyqtSlot(str)
    def _on_color_picker_requested(self, step_id: str):
        """Ouvre la pipette de balance des blancs pour l'étape step_id."""
        from PyQt6.QtWidgets import QDialog
        from ui.wb_picker import WBPickerDialog

        step = self._steps_by_id.get(step_id)
        if step is None or not hasattr(step, "set_pick_point"):
            return
        img = self._get_step_input_image(step_id)
        if img is None:
            self._status_bar.showMessage(
                "Ouvrez une image avant de sélectionner le point blanc."
            )
            return
        dlg = WBPickerDialog(self, img, step.get_pick_point())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            pt = dlg.get_pick_point()
            if pt is not None:
                radius = dlg.get_patch_radius()
                step.set_pick_point(*pt)
                panel = self._ctrl.step_list.get_panel(step_id)
                if panel is not None:
                    panel.set_params({"patch_radius": radius})
                self._status_bar.showMessage(
                    f"Point blanc sélectionné en ({pt[0]}, {pt[1]}) — « {step.name} »."
                )
            else:
                step.clear_pick_point()
                self._status_bar.showMessage(f"Sélection effacée pour « {step.name} ».")

    def _get_step_input_image(self, step_id: str) -> np.ndarray | None:
        """Retourne l'image en entrée de l'étape step_id (sortie de la précédente)."""
        order = self._ctrl.step_list.get_order()
        idx   = order.index(step_id) if step_id in order else -1
        if idx > 0:
            img = self._last_step_images.get(order[idx - 1])
        else:
            img = None
        return img if img is not None else self._original

    def _build_overlay_img(self) -> Optional[np.ndarray]:
        """Dessine les annotations de détection sur l'image originale.

        Dispatche selon le step_id dont l'overlay est actif.
        """
        if self._original is None:
            return None
        context: dict = {}
        latest = self._history.latest()
        if latest:
            context = latest.context

        overlay = self._original.copy()

        if self._overlay_step == "redeye":
            for det in context.get("redeye_detections", []):
                iris = det.get("iris")
                if iris is None:
                    continue
                ix, iy, ir = iris
                corrected = det.get("corrected", False)
                color = (0, 220, 80) if corrected else (0, 180, 200)
                cv2.circle(overlay, (int(ix), int(iy)), max(int(ir), 2),
                           color, 2, cv2.LINE_AA)
                cv2.circle(overlay, (int(ix), int(iy)), 1,
                           (0, 255, 150) if corrected else (0, 220, 240), -1, cv2.LINE_AA)

        elif self._overlay_step == "facehighlight":
            for det in context.get("highlight_detections", []):
                bbox = det.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                overexp   = det.get("overexp", 0.0)
                color = (0, 170, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.putText(overlay, f"{overexp:.0%}",
                            (x1 + 3, y1 + 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

        return overlay

    # ══════════════════════════════════════════════════════════════════════════
    # Navigation historique
    # ══════════════════════════════════════════════════════════════════════════

    def _on_history_activated(self, run_id: int):
        entry = self._history.get(run_id)
        if entry is None:
            return
        self._active_run_id = run_id

        # Reconstruire la bande avec les étapes calculées de cette version
        step_names = {s.id: s.short_name for s in ALL_STEPS}
        self._thumb_strip.rebuild(entry.completed_steps, step_names)
        self._thumb_strip.update_image("original", self._original)
        for step_id, img in entry.step_results.items():
            self._thumb_strip.update_image(step_id, img)
        self._thumb_strip.set_version_label(f"v{run_id} — {entry.time_str}")
        self._update_thumb_highlights()

    # ══════════════════════════════════════════════════════════════════════════
    # Sélection des vues via vignettes
    # ══════════════════════════════════════════════════════════════════════════

    def _on_thumb_a(self, step_id: str):
        run_id = None if step_id == "original" else self._active_run_id
        self._view_a = (run_id, step_id)
        self._refresh_view_a(preserve_zoom=True)
        self._update_view_labels()
        self._update_thumb_highlights()

    def _on_thumb_b(self, step_id: str):
        run_id = None if step_id == "original" else self._active_run_id
        self._view_b = (run_id, step_id)
        self._label_b.setVisible(True)
        self._refresh_view_b(preserve_zoom=True)
        self._update_view_labels()
        self._update_thumb_highlights()

    # ══════════════════════════════════════════════════════════════════════════
    # Rafraîchissement des vues
    # ══════════════════════════════════════════════════════════════════════════

    def _get_image(self, run_id: Optional[int], step_id: str) -> Optional[np.ndarray]:
        if step_id == "original" or run_id is None:
            return self._original
        entry = self._history.get(run_id)
        return entry.step_results.get(step_id) if entry else None

    def _refresh_views(self):
        self._refresh_view_a()
        self._refresh_view_b()

    def _refresh_view_a(self, preserve_zoom: bool = False):
        if self._overlay_step is not None:
            img = self._build_overlay_img()
        else:
            img = self._get_image(*self._view_a)
        self._view_a_widget.set_image(img, preserve_zoom)

    def _refresh_view_b(self, preserve_zoom: bool = False):
        if self._view_b is None:
            return
        img = self._get_image(*self._view_b)
        self._view_b_widget.set_image(img, preserve_zoom)

    def _fit_views(self):
        self._view_a_widget.fit_in_view()
        self._view_b_widget.fit_in_view()

    def _make_label_text(self, prefix: str, run_id: Optional[int], step_id: str) -> str:
        """Compose le label d'une vue : 'A — Original' ou 'A — v2 · GFPGAN'."""
        if step_id == "original":
            return f"{prefix} — Original"
        step  = self._steps_by_id.get(step_id)
        sname = step.short_name if step else step_id
        ver   = f"v{run_id}" if run_id is not None else "?"
        return f"{prefix} — {ver} · {sname}"

    def _update_view_labels(self):
        """Met à jour les labels des deux vues selon les sources courantes."""
        if self._overlay_step is not None:
            step  = self._steps_by_id.get(self._overlay_step)
            sname = step.short_name if step else self._overlay_step
            self._view_a_lbl.setText(f"A — Overlay {sname}")
        else:
            self._view_a_lbl.setText(self._make_label_text("A", *self._view_a))
        if self._view_b is not None:
            self._view_b_lbl.setText(self._make_label_text("B", *self._view_b))
        else:
            self._view_b_lbl.setText("B")

    def _update_thumb_highlights(self):
        """Met à jour les bordures A/B des vignettes selon la version active.

        Règle : une vignette est encadrée en bleu (A) ou rouge (B) seulement si
        sa source provient du même run que la version actuellement affichée
        (ou si c'est l'image originale, toujours disponible).
        """
        a_run, a_step = self._view_a
        b_run, b_step = (self._view_b if self._view_b else (None, None))

        a_highlight = (
            a_step
            if (a_step == "original" or a_run == self._active_run_id)
            else None
        )
        b_highlight = (
            b_step
            if (b_step is not None and (b_step == "original" or b_run == self._active_run_id))
            else None
        )
        self._thumb_strip.set_active_a(a_highlight)
        self._thumb_strip.set_active_b(b_highlight)

    # ══════════════════════════════════════════════════════════════════════════
    # Changements de paramètres / ordre
    # ══════════════════════════════════════════════════════════════════════════

    def _on_order_changed(self, new_order: list[str]):
        self._step_order = new_order
        self._mark_stale_from(new_order[0] if new_order else None)

    def _on_param_changed(self, step_id: str, key: str, value):
        self._mark_stale_from(step_id)
        self._schedule_preview(step_id)

    # ══════════════════════════════════════════════════════════════════════════
    # Preview instantané (étapes rapides)
    # ══════════════════════════════════════════════════════════════════════════

    def _schedule_preview(self, step_id: str) -> None:
        """Déclenche un preview différé (debounce 150 ms) pour l'étape donnée."""
        step = self._steps_by_id.get(step_id)
        if step is None:
            return
        if not (step_id in _FAST_PREVIEW_IDS or getattr(step, "previewable", False)):
            return
        if self._worker is not None and self._worker.isRunning():
            return   # pipeline en cours, pas de preview concurrent
        self._preview_step_id = step_id
        self._preview_timer.start()

    def _run_preview(self) -> None:
        """Calcule l'aperçu : mini-pipeline rapide ou aperçu isolé selon l'étape."""
        step_id = self._preview_step_id
        if step_id is None:
            return
        if step_id in _FAST_PREVIEW_IDS:
            self._run_fast_preview()
        else:
            self._run_single_preview(step_id)

    def _run_fast_preview(self) -> None:
        """Mini-pipeline color→facehighlight→autocolor→wb (étapes activées seulement)."""
        # Entrée = sortie de la dernière étape précédant le groupe rapide (ou original)
        first_fast = next((sid for sid in self._step_order if sid in _FAST_PREVIEW_IDS), None)
        if first_fast is None:
            return
        img = self._get_preview_input(first_fast)
        if img is None:
            return

        out = img
        ran_count = 0
        for sid in self._step_order:
            if sid not in _FAST_PREVIEW_IDS:
                continue
            if not self._step_enabled.get(sid, True):
                continue
            step = self._steps_by_id.get(sid)
            panel = self._ctrl.step_list.get_panel(sid)
            if step is None or panel is None:
                continue
            params = panel.get_params()
            try:
                result, extras = step.process(out, params, {})
            except Exception:
                continue
            out = result
            ran_count += 1
            if "effective_params" in extras:
                panel.set_params(extras["effective_params"])

        if ran_count == 0:
            return
        self._preview_active = True
        self._view_a_widget.set_image(out, preserve_zoom=True)
        self._view_a_widget.set_preview_mode(True)
        self._view_a_lbl.setText("A — APERÇU · Corrections rapides")
        self._view_a_lbl.setStyleSheet(
            "background: #1a1a2e; color: #f39c12; font-size: 10px; font-weight: bold;"
        )

    def _run_single_preview(self, step_id: str) -> None:
        """Aperçu isolé pour une étape previewable hors groupe rapide."""
        step = self._steps_by_id.get(step_id)
        if step is None:
            return
        input_img = self._get_preview_input(step_id)
        if input_img is None:
            return
        panel = self._ctrl.step_list.get_panel(step_id)
        params = panel.get_params() if panel else step.default_params()
        try:
            result, extras = step.process(input_img, params, {})
        except Exception:
            return
        if "effective_params" in extras and panel is not None:
            panel.set_params(extras["effective_params"])
        self._preview_active = True
        self._view_a_widget.set_image(result, preserve_zoom=True)
        self._view_a_widget.set_preview_mode(True)
        self._view_a_lbl.setText(f"A — APERÇU · {step.short_name}")
        self._view_a_lbl.setStyleSheet(
            "background: #1a1a2e; color: #f39c12; font-size: 10px; font-weight: bold;"
        )

    def _clear_preview(self) -> None:
        """Annule l'aperçu et restaure l'affichage normal de la vue A."""
        if not self._preview_active:
            return
        self._preview_active = False
        self._preview_timer.stop()
        self._view_a_widget.set_preview_mode(False)
        self._view_a_lbl.setStyleSheet(
            "background: #1a1a2e; color: #777; font-size: 10px;"
        )
        # Restaurer l'image normale de la vue A
        img = self._get_image(*self._view_a)
        if img is not None:
            self._view_a_widget.set_image(img, preserve_zoom=True)
        self._view_a_lbl.setText(self._make_label_text("A", *self._view_a))

    def _get_preview_input(self, step_id: str) -> Optional[np.ndarray]:
        """Retourne l'image en entrée de step_id (sortie de l'étape précédente)."""
        if step_id not in self._step_order:
            return self._original
        idx = self._step_order.index(step_id)
        if idx == 0:
            return self._original
        # Chercher en arrière la dernière étape activée ayant un résultat
        for i in range(idx - 1, -1, -1):
            sid = self._step_order[i]
            if sid in self._last_step_images and self._step_enabled.get(sid, True):
                return self._last_step_images[sid]
        return self._original

    def _on_enabled_changed(self, step_id: str, enabled: bool):
        self._step_enabled[step_id] = enabled
        panel = self._ctrl.step_list.get_panel(step_id)
        if panel:
            panel.set_state("disabled" if not enabled else "stale")

    def _mark_stale_from(self, step_id: Optional[str]):
        """Marque comme obsolètes toutes les étapes à partir de step_id."""
        if step_id is None:
            return
        if step_id not in self._step_order:
            return
        start = self._step_order.index(step_id)
        for sid in self._step_order[start:]:
            panel = self._ctrl.step_list.get_panel(sid)
            if panel and panel.is_enabled():
                panel.set_state("stale")

    # ══════════════════════════════════════════════════════════════════════════
    # Thème
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #141420; color: #ccc; font-family: 'Segoe UI', sans-serif; }
            QMenuBar { background: #16162a; color: #ccc; }
            QMenuBar::item:selected { background: #2a2a4a; }
            QMenu { background: #1e1e2e; color: #ccc; border: 1px solid #333; }
            QMenu::item:selected { background: #2a2a4a; }
            QScrollBar:vertical { background: #1a1a2e; width: 8px; border: none; }
            QScrollBar::handle:vertical { background: #3a3a5a; border-radius: 4px; min-height: 20px; }
            QScrollBar:horizontal { background: #1a1a2e; height: 8px; border: none; }
            QScrollBar::handle:horizontal { background: #3a3a5a; border-radius: 4px; min-width: 20px; }
            QSplitter::handle { background: #2a2a4a; }
        """)
