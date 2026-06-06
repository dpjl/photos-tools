"""ui/mixins/preview.py — Aperçu instantané pour les étapes rapides (debounce 150 ms)."""

from __future__ import annotations

from typing import Optional

import numpy as np


# Étapes dont un changement de paramètre déclenche le mini-pipeline de preview rapide
_FAST_PREVIEW_IDS = frozenset({"color", "facehighlight", "ddcolor_lut", "autocolor", "wb",
                               "lightleak", "rembg", "crop"})


class PreviewMixin:

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
