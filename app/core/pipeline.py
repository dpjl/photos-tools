"""core/pipeline.py — Worker QThread pour l'exécution du pipeline."""

from __future__ import annotations
import traceback
from datetime import datetime

from PyQt6.QtCore import QThread, pyqtSignal

from core.history import HistoryEntry


class PipelineWorker(QThread):
    """Exécute une séquence d'étapes en arrière-plan.

    Paramètres :
        run_id       : identifiant unique du run (depuis HistoryManager.next_id)
        steps        : list[StepBase] — étapes à exécuter, dans l'ordre voulu
        initial_img  : np.ndarray — image en entrée de la première étape
        all_params   : dict[step_id, dict] — paramètres de toutes les étapes
        context      : dict — contexte partagé initial (ex. face_bboxes du run précédent)
        step_order   : list[str] — ordre complet des étapes dans le pipeline (pour l'historique)
        step_enabled : dict[str, bool] — état activé/désactivé pour chaque étape
    """

    step_started = pyqtSignal(str)               # step_id
    step_done    = pyqtSignal(str, object, dict)  # step_id, ndarray, extras
    step_failed  = pyqtSignal(str, str)           # step_id, message
    all_done     = pyqtSignal(object)             # HistoryEntry

    def __init__(self, run_id, steps, initial_img, all_params,
                 context, step_order, step_enabled, parent=None):
        super().__init__(parent)
        self._run_id       = run_id
        self._steps        = steps          # list[StepBase]
        self._img          = initial_img
        self._params       = all_params
        self._ctx          = dict(context)
        self._step_order   = list(step_order)
        self._step_enabled = dict(step_enabled)
        self._cancel       = False
        self._results: dict = {}

    def cancel(self):
        self._cancel = True

    def run(self):
        img = self._img
        for step in self._steps:
            if self._cancel:
                break
            if img is None:
                self.step_failed.emit(step.id, "Aucune image en entrée")
                continue

            self.step_started.emit(step.id)
            p = self._params.get(step.id, step.default_params())

            try:
                result, extras = step.process(img, p, self._ctx)
                self._ctx.update(extras)
                self._results[step.id] = result
                self.step_done.emit(step.id, result, dict(extras))
                img = result
            except Exception as exc:
                traceback.print_exc()
                self.step_failed.emit(step.id, str(exc))
                # On ne stoppe pas : les étapes suivantes reçoivent la dernière image valide

        entry = HistoryEntry(
            run_id       = self._run_id,
            timestamp    = datetime.now(),
            step_order   = self._step_order,
            step_enabled = self._step_enabled,
            step_params  = {k: dict(v) for k, v in self._params.items()},
            step_results = dict(self._results),
            context      = dict(self._ctx),
        )
        self.all_done.emit(entry)
