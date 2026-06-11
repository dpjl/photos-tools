"""ui/mixins/editors.py — Overlay de détection, éditeur de masque et pipette WB."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import pyqtSlot


class EditorsMixin:

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
    def _on_crop_edit_requested(self, step_id: str):
        """Ouvre l'éditeur de recadrage pour l'étape step_id."""
        from PyQt6.QtWidgets import QDialog
        from ui.crop_editor import CropEditorDialog

        step = self._steps_by_id.get(step_id)
        if step is None or not hasattr(step, "set_crop_rect"):
            return
        img = self._get_step_input_image(step_id)
        if img is None:
            self._status_bar.showMessage("Ouvrez une image avant de recadrer.")
            return
        dlg = CropEditorDialog(self, img, step.get_crop_rect())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            step.set_crop_rect(dlg.get_crop_rect())
            self._status_bar.showMessage(f"Recadrage mis à jour pour « {step.name} ».")
            self._mark_stale_from(step_id)
            self._update_result_diff_indicator()
            if hasattr(self, "_schedule_preview"):
                self._schedule_preview(step_id)

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

    @pyqtSlot(str)
    def _on_genref_requested(self, step_id: str):
        """Ouvre le générateur de référence IA pour l'étape step_id."""
        from PyQt6.QtWidgets import QDialog
        from ui.genref_dialog import GenRefDialog

        step = self._steps_by_id.get(step_id)
        if step is None:
            return
        img = self._get_step_input_image(step_id)
        if img is None:
            self._status_bar.showMessage(
                "Ouvrez une image avant de générer une référence IA."
            )
            return
        panel = self._ctrl.step_list.get_panel(step_id)
        cur = panel.get_params() if panel is not None else step.default_params()
        dlg = GenRefDialog(self, img,
                           style=str(cur.get("style", "court")),
                           seed=int(cur.get("graine", 42)))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Aligner les params du panneau sur l'entrée choisie
            if panel is not None:
                panel.set_params({"style": dlg.current_style(),
                                  "graine": dlg.current_seed()})
            self._status_bar.showMessage(
                f"Référence IA prête ({dlg.current_style()} / graine "
                f"{dlg.current_seed()}) — « {step.name} »."
            )
            self._mark_stale_from(step_id)
            self._update_result_diff_indicator()
            if hasattr(self, "_schedule_preview"):
                self._schedule_preview(step_id)

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
