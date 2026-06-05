"""ui/mixins/pipeline.py — Exécution du pipeline de traitement et slots du worker."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import pyqtSlot

from core.pipeline import PipelineWorker
from core.history import HistoryEntry
from steps import ALL_STEPS


class PipelineMixin:

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
                # Synchroniser le snapshot avec les valeurs effectives du run
                for k, v in extras["effective_params"].items():
                    self._params_snapshot.setdefault(step_id, {})[k] = v
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
        # Resynchroniser le snapshot complet après le run (effective_params éventuels)
        self._params_snapshot = self._ctrl.step_list.get_all_params()
        self._update_result_diff_indicator()
