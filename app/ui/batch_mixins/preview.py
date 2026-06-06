"""batch_mixins/preview.py — Aperçu rapide, preview complet, onglets et overlays."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import pyqtSlot, QTimer

from ui.batch_window_constants import (
    _FAST_PREVIEW_IDS, _PREVIEW_TABS,
    _TAB_MASK, _TAB_WB, _TAB_REDEYE,
    _TAB_CROP,
    _TAB_PREVIEW, _TAB_RESULT,
    _TAB_ORIGIN,
)
from core.pipeline import PipelineWorker
from core.image_info import format_preview_info


class PreviewMixin:
    """Aperçu rapide (fast-pipeline), preview complet, gestion des onglets, overlays."""

    # ── Signaux WB ────────────────────────────────────────────────────────────

    @pyqtSlot(int, int)
    def _on_wb_pick_changed(self, x: int, y: int) -> None:
        if self._current_cfg is not None:
            self._current_cfg.wb_pick = (x, y)
            self._mark_customized()
        self._schedule_preview_update()

    @pyqtSlot(int)
    def _on_wb_radius_changed(self, radius: int) -> None:
        if self._current_cfg is not None:
            self._current_cfg.wb_patch_radius = radius
            self._mark_customized()
        self._schedule_preview_update()

    @pyqtSlot(object)
    def _on_crop_changed(self, rect) -> None:
        if self._current_cfg is not None:
            self._current_cfg.crop_rect = rect
            self._mark_customized()
            self._inject_instance_state(self._current_cfg)
        self._schedule_preview_update()

    # ── Onglets ───────────────────────────────────────────────────────────────

    @pyqtSlot(int)
    def _on_tab_changed(self, index: int) -> None:
        """Au changement d'onglet : synchronise le zoom et charge le résultat si nécessaire."""
        # 1. Sauvegarder le zoom de l'onglet qu'on quitte
        prev = self._prev_tab_index
        prev_canvas = self._get_tab_canvas(prev)
        if prev_canvas is not None and hasattr(prev_canvas, "get_zoom_state"):
            state = prev_canvas.get_zoom_state()
            if state[0] > 0:
                self._shared_zoom = state
        self._prev_tab_index = index

        # 2. Sync éditeurs → cfg en mémoire quand on quitte un onglet éditeur
        if prev in (_TAB_MASK, _TAB_WB, _TAB_REDEYE, _TAB_CROP):
            self._sync_editor_state()

        # 3. Charger le résultat si besoin
        if index == _TAB_RESULT and self._current_cfg is not None:
            self._update_dest_view(self._current_cfg)

        # 4. (Ancien onglet diff supprimé)

        # 5. Appliquer le zoom au nouveau canvas, différé après le layout Qt
        zoom = self._shared_zoom
        if zoom is not None:
            new_canvas = self._get_tab_canvas(index)
            if new_canvas is not None and hasattr(new_canvas, "apply_zoom_state"):
                QTimer.singleShot(
                    0, lambda c=new_canvas, z=zoom: c.apply_zoom_state(*z)
                )

        # 6. Recalculer l'aperçu seulement si nécessaire
        if self._preview_stale:
            self._schedule_preview_update()

    def _get_tab_canvas(self, index: int):
        """Retourne le canvas zôomable de l'onglet ou None."""
        if index == _TAB_PREVIEW:
            return self._preview_view
        if index == _TAB_MASK:
            return self._mask_panel._canvas
        if index == _TAB_WB:
            return self._wb_panel._canvas
        if index == _TAB_REDEYE:
            return self._redeye_panel._canvas
        if index == _TAB_CROP:
            return self._crop_panel._canvas
        if index == _TAB_ORIGIN:
            return self._origin_view
        if index == _TAB_RESULT:
            return self._export_mosaic
        return None

    # ── Timer debounce ────────────────────────────────────────────────────────

    def _schedule_preview_update(self) -> None:
        """Lance le timer debounce (300 ms) pour mettre à jour l'aperçu."""
        if self._current_orig is None:
            return
        self._preview_stale = True
        self._preview_timer.start()

    def _update_preview_info(self, preview_img: Optional[np.ndarray] = None) -> None:
        if not hasattr(self, "_preview_info_lbl"):
            return
        cfg = self._current_cfg
        if cfg is None:
            self._preview_info_lbl.setText("")
            return
        original_dims = None
        if self._current_orig is not None:
            h, w = self._current_orig.shape[:2]
            original_dims = (w, h)
        if preview_img is None:
            if self._preview_full_img is not None:
                preview_img = self._preview_full_img
            elif self._last_fast_preview is not None:
                preview_img = self._last_fast_preview
            else:
                preview_img = self._current_orig
        preview_dims = None
        if preview_img is not None:
            ph, pw = preview_img.shape[:2]
            preview_dims = (pw, ph)
        self._preview_info_lbl.setText(
            format_preview_info(cfg.file_path, original_dims, preview_dims)
        )

    def _do_preview_update(self) -> None:
        """Calcule le fast-pipeline si l'onglet actif est éligible et que l'aperçu est périmé."""
        if not self._preview_stale:
            return
        idx = self._tabs.currentIndex()
        if idx in _PREVIEW_TABS:
            self._compute_and_show_fast_preview()

    # ── Aperçu rapide ─────────────────────────────────────────────────────────

    def _compute_and_show_fast_preview(self) -> None:
        """Calcule le fast-pipeline pour le profil courant et met à jour les panneaux."""
        if self._current_orig is None or self._current_cfg is None:
            return
        self._sync_editor_state()
        self._inject_instance_state(self._current_cfg)
        if self._is_viewing_export and self._viewed_export_data is not None:
            data = self._viewed_export_data
            img = self._compute_single_profile_preview(
                step_order   = data.get("step_order"),
                step_enabled = data.get("step_enabled"),
                step_params  = data.get("step_params"),
            )
        else:
            img = self._compute_single_profile_preview()
        if img is not None:
            self._refresh_panels(img)

    def _compute_single_profile_preview(
        self,
        step_order:   Optional[list] = None,
        step_enabled: Optional[dict] = None,
        step_params:  Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """Exécute les fast steps pour le profil courant (une seule passe)."""
        cfg = self._current_cfg
        if cfg is None or self._current_orig is None:
            return None
        _order   = step_order   if step_order   is not None else cfg.step_order
        _enabled = step_enabled if step_enabled is not None else cfg.step_enabled
        out = self._current_orig.copy()
        ctx: dict = {}
        for sid in _order:
            if sid not in _FAST_PREVIEW_IDS:
                continue
            if not _enabled.get(sid, True):
                continue
            step  = self._steps_by_id.get(sid)
            panel = self._step_list.get_panel(sid)
            if step is None or panel is None:
                continue
            params = (step_params.get(sid) if step_params is not None and sid in step_params
                      else panel.get_params())
            try:
                result, extras = step.process(out, params, ctx)
                out = result
                ctx.update(extras)
            except Exception:
                pass
        return out

    def _refresh_panels(self, img: np.ndarray) -> None:
        """Met à jour les panneaux Masque, Blanc, Yeux rouges et Preview."""
        if self._current_cfg is None:
            return
        self._mask_panel._canvas.set_display_image(img)
        self._wb_panel._canvas.set_display_image(img)
        self._redeye_panel._canvas.set_display_image(img)
        self._preview_full_img = None
        self._last_fast_preview = img
        self._preview_stale = False
        self._preview_view.set_image(img, preserve_zoom=True)
        self._preview_status_lbl.setText("Preview rapide")
        self._update_preview_info(img)

    # ── Preview complet ───────────────────────────────────────────────────────

    def _run_full_preview(self) -> None:
        """Démarre un PipelineWorker ALL steps sur l'image courante, sans sauvegarder."""
        cfg = self._current_cfg
        img = self._current_orig
        if cfg is None or img is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._statusbar.showMessage("Preview complet indisponible pendant le batch.")
            return
        if self._full_preview_worker is not None and self._full_preview_worker.isRunning():
            self._full_preview_worker.cancel()
            self._full_preview_worker.wait(500)

        self._preview_full_img = None
        self._full_preview_btn.setEnabled(False)
        self._preview_status_lbl.setText("⏳ Calcul...")
        self._sync_editor_state()
        self._inject_instance_state(cfg)

        enabled_order = [sid for sid in cfg.step_order if cfg.step_enabled.get(sid, True)]
        steps = [self._steps_by_id[sid] for sid in enabled_order if sid in self._steps_by_id]

        self._full_preview_worker = PipelineWorker(
            run_id      = -1,
            steps       = steps,
            initial_img = img.copy(),
            all_params  = cfg.step_params,
            context     = {},
            step_order  = cfg.step_order,
            step_enabled= cfg.step_enabled,
        )
        self._full_preview_worker.all_done.connect(self._on_full_preview_done)
        self._full_preview_worker.step_started.connect(
            lambda sid: self._preview_status_lbl.setText(f"⏳ {sid}…")
        )
        self._full_preview_worker.start()

    @pyqtSlot(object)
    def _on_full_preview_done(self, entry) -> None:
        """Reçoit le résultat du preview complet et met à jour l'onglet Preview."""
        result_img = (
            entry.step_results.get(entry.completed_steps[-1])
            if entry.completed_steps else None
        )
        if self._current_cfg is not None:
            self._current_cfg.context.update(entry.context)

        self._preview_full_img = result_img
        self._preview_stale = False
        self._last_fast_preview = None
        self._full_preview_btn.setEnabled(True)

        if result_img is not None:
            self._preview_status_lbl.setText("Preview complet")
            self._mask_panel._canvas.set_display_image(result_img)
            self._wb_panel._canvas.set_display_image(result_img)
            self._redeye_panel._canvas.set_display_image(result_img)
            if self._preview_overlay_step:
                overlay = self._build_overlay_img(result_img)
                self._preview_view.set_image(
                    overlay if overlay is not None else result_img,
                    preserve_zoom=True,
                )
            else:
                self._preview_view.set_image(result_img, preserve_zoom=True)
            self._update_preview_info(result_img)
        else:
            self._preview_status_lbl.setText("Preview complet (sans résultat)")
            self._update_preview_info()

    # ── Overlay de détection ──────────────────────────────────────────────────

    @pyqtSlot(str, bool)
    def _on_overlay_toggled(self, step_id: str, enabled: bool) -> None:
        self._preview_overlay_step = step_id if enabled else None
        self._refresh_preview_with_overlay()

    def _refresh_preview_with_overlay(self) -> None:
        if self._preview_full_img is None:
            return
        if self._preview_overlay_step:
            overlay = self._build_overlay_img(self._preview_full_img)
            self._preview_view.set_image(
                overlay if overlay is not None else self._preview_full_img,
                preserve_zoom=True,
            )
        else:
            self._preview_view.set_image(self._preview_full_img, preserve_zoom=True)

    def _build_overlay_img(self, base_img: np.ndarray) -> Optional[np.ndarray]:
        """Dessine les annotations de détection sur base_img."""
        if base_img is None or self._current_cfg is None:
            return None
        context = self._current_cfg.context
        overlay = base_img.copy()

        if self._preview_overlay_step == "redeye":
            for det in context.get("redeye_detections", []):
                iris = det.get("iris")
                if iris is None:
                    continue
                ix, iy, ir = iris
                corrected = det.get("corrected", False)
                color = (0, 220, 80) if corrected else (0, 180, 200)
                cv2.circle(overlay, (int(ix), int(iy)), max(int(ir), 2), color, 2, cv2.LINE_AA)
                cv2.circle(overlay, (int(ix), int(iy)), 1,
                           (0, 255, 150) if corrected else (0, 220, 240), -1, cv2.LINE_AA)

        elif self._preview_overlay_step == "facehighlight":
            for det in context.get("highlight_detections", []):
                bbox = det.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                color = (0, 170, 255)
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 2, cv2.LINE_AA)
                overexp = det.get("overexp", 0.0)
                label = f"{overexp:.0%}"
                cv2.putText(overlay, label, (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return overlay
