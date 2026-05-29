"""ui/mixins/views.py — Navigation vues, historique, vignettes et labels."""

from __future__ import annotations

from typing import Optional

import numpy as np

from steps import ALL_STEPS


class ViewsMixin:

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
