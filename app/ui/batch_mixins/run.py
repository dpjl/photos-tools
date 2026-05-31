"""batch_mixins/run.py — Exécution du pipeline (sélection, batch, worker)."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QFileDialog

from core.batch import BatchImageConfig, save_recipe
from core.pipeline import PipelineWorker
from ui.batch_window_constants import _TAB_RESULT

if TYPE_CHECKING:
    pass


class RunMixin:
    """Lancement du pipeline sur la sélection ou le batch complet."""

    # ── Lancer la sélection ───────────────────────────────────────────────────

    def _run_on_selection(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        if not self._session.output_dir:
            self._choose_output_dir()
            if not self._session.output_dir:
                return
        self._save_current_state()
        run_paths = self._strip.get_run_selection()
        if not run_paths:
            if self._current_cfg is None:
                return
            run_paths = [self._current_cfg.file_path]
        queue: list[BatchImageConfig] = [
            cfg for cfg in self._session.images
            if cfg.file_path in set(run_paths)
        ]
        order_index = {p: i for i, p in enumerate(run_paths)}
        queue.sort(key=lambda c: order_index.get(c.file_path, 0))
        self._run_batch_from_queue(queue)

    # ── Appliquer à toutes ────────────────────────────────────────────────────

    def _apply_to_all_uncustomized(self) -> None:
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

    # ── Batch complet ─────────────────────────────────────────────────────────

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
        self._batch_queue    = queue
        self._batch_step_log: dict[str, list[dict]] = {}
        self._batch_run_total = len(queue)
        self._batch_run_done  = 0
        self._batch_run_start = time.monotonic()
        self._set_buttons_running(True)
        self._process_next_batch()

    def _process_next_batch(self) -> None:
        from ui.notifications import Level
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
        from ui.notifications import Level
        self._set_buttons_running(False)
        n = len(self._session.images)
        self._statusbar.showMessage(f"Batch terminé — {n} image(s) traitée(s)")
        if hasattr(self, "_notif") and self._batch_run_total > 0:
            elapsed = time.monotonic() - self._batch_run_start
            self._notif.dismiss_key("batch_progress")
            body = (
                f"{self._batch_run_done} / {self._batch_run_total} image(s)"
                f"  •  {elapsed:.0f}s"
            )
            self._notif.notify("Batch terminé", body, level=Level.SUCCESS, duration=8000)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _start_worker(self, cfg: BatchImageConfig, img: np.ndarray) -> None:
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
        from ui.notifications import Level
        cfg        = self._worker_cfg
        result_img = (entry.step_results.get(entry.completed_steps[-1])
                      if entry.completed_steps else None)
        cfg.context = entry.context

        if result_img is not None:
            self._strip.update_result_thumb(cfg.file_path, result_img)
            if cfg is self._current_cfg:
                self._mask_panel._canvas.set_display_image(result_img)
                self._wb_panel._canvas.set_display_image(result_img)
                self._redeye_panel._canvas.set_display_image(result_img)
                if self._tabs.currentIndex() == _TAB_RESULT:
                    self._update_dest_view(cfg, force=True)

        cfg.batch_status = "done"
        self._strip.set_running(cfg.file_path, False)
        self._strip.set_done(cfg.file_path, True)

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

        step_log = None  # Plus de step_log — le recipe contient tout
        try:
            self._session.save_result(cfg, result_img)
        except Exception as exc:
            self._statusbar.showMessage(f"Erreur sauvegarde : {exc}")

        if cfg is self._current_cfg:
            self._refresh_export_dropdown(cfg)
            self._refresh_if_diff_tab()

        # Libérer le cache GPU entre chaque image du batch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self._process_next_batch()

    # ── Contrôles ─────────────────────────────────────────────────────────────

    def _stop(self) -> None:
        if self._worker:
            self._worker.cancel()
        if hasattr(self, "_batch_queue"):
            self._batch_queue.clear()
        self._set_buttons_running(False)
        self._statusbar.showMessage("Arrêté.")

    def _set_buttons_running(self, running: bool) -> None:
        self._selection_btn.setEnabled(not running)
        self._batch_btn.setEnabled(not running)
        self._apply_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._full_preview_btn.setEnabled(not running)

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Dossier de sortie", self._session.output_dir or ""
        )
        if path:
            self._session.output_dir = path
            self._output_lbl.setText(path)
            self._output_lbl.setToolTip(path)
            self._session.save_session_meta()
