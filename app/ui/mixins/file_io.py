"""ui/mixins/file_io.py — Ouverture et sauvegarde d'images."""

from __future__ import annotations

import os

import cv2
import numpy as np
from PyQt6.QtWidgets import QFileDialog

from core.export_manager import ExportManager


class FileIOMixin:

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

        # Vider la pile undo et resynchroniser le snapshot
        self._undo_stack.clear()
        self._params_snapshot = self._ctrl.step_list.get_all_params()
        self._update_result_diff_indicator()

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

    def _save_result(self):
        """Enregistre l'image affichée dans la vue A (Ctrl+S).

        Premier appel : demande le répertoire de sortie.
        Appels suivants : export automatique avec numéro incrémenté.
        """
        img = self._get_image(*self._view_a)
        if img is None:
            self._status_bar.showMessage("Rien à enregistrer.")
            return
        if not self._original_path:
            self._status_bar.showMessage("Aucune image source.")
            return

        # Choisir le répertoire de sortie si pas encore défini
        if not getattr(self, "_output_dir", None):
            path = QFileDialog.getExistingDirectory(
                self, "Dossier de sortie",
                os.path.dirname(self._original_path),
            )
            if not path:
                return
            self._output_dir = path

        stem, ext = os.path.splitext(os.path.basename(self._original_path))
        if not ext:
            ext = ".jpg"

        # Construire le recipe depuis l'état courant
        recipe = {
            "version":         2,
            "source_filename": os.path.basename(self._original_path),
            "step_order":      self._ctrl.step_list.get_order(),
            "step_enabled":    self._ctrl.step_list.get_enabled(),
            "step_params":     self._ctrl.step_list.get_all_params(),
            "crop_rect":       self._export_crop_rect(),
        }

        mgr = ExportManager(self._output_dir)
        entry = mgr.save_export(stem, ext, img, recipe)

        self._status_bar.showMessage(
            f"Exporté : {os.path.basename(entry.image_path)}"
        )

        # Rafraîchir la mosaïque d'exports si disponible
        if hasattr(self, "_refresh_single_exports"):
            self._refresh_single_exports()
            # Basculer vers l'onglet Exports
            if hasattr(self, "_bottom_tabs"):
                self._bottom_tabs.setCurrentIndex(1)

    def _export_crop_rect(self):
        crop_step = self._steps_by_id.get("crop")
        if crop_step is None or not hasattr(crop_step, "get_crop_rect"):
            return None
        rect = crop_step.get_crop_rect()
        return list(rect) if rect else None

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
