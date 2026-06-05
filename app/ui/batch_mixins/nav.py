"""batch_mixins/nav.py — Navigation entre images, sauvegarde/chargement de l'état."""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

from core.batch import BatchImageConfig, save_recipe, load_recipe, _apply_recipe
from ui.batch_window_constants import _TAB_RESULT

if TYPE_CHECKING:
    pass


class NavMixin:
    """Navigation entre images, synchronisation éditeurs ↔ cfg, injection dans les steps."""

    # ── Navigation ────────────────────────────────────────────────────────────

    def _on_strip_image_selected(self, file_path: str) -> None:
        cfg = self._config_for(file_path)
        if cfg is None:
            return
        if self._current_cfg is not None:
            self._save_current_state()
        self._navigate_to(cfg)

    def _navigate_to(self, cfg: BatchImageConfig) -> None:
        """Charge la configuration d'une image dans l'UI."""
        # Réinitialiser le zoom partagé → chaque nouvelle image commence au fit
        self._shared_zoom = None
        self._current_cfg = cfg
        self._preview_stale = True
        self._preview_full_img = None
        self._last_fast_preview = None

        # ── Image originale ───────────────────────────────────────────────────
        img = cv2.imread(cfg.file_path, cv2.IMREAD_COLOR)
        self._current_orig = img

        # ── Étapes ────────────────────────────────────────────────────────────
        for sid, enabled in cfg.step_enabled.items():
            panel = self._step_list.get_panel(sid)
            if panel:
                panel.set_enabled(enabled)
        for sid, params in cfg.step_params.items():
            panel = self._step_list.get_panel(sid)
            if panel:
                panel.set_params(params)
        self._step_list.set_order(cfg.step_order)

        # ── Panneaux édition ───────────────────────────────────────────────
        if img is not None:
            result = self._get_result_from_disk(cfg)
            display = result if result is not None else img
            self._mask_panel.set_image(display, cfg.inpaint_mask)
            self._wb_panel.set_image(display, cfg.wb_pick, cfg.wb_patch_radius)
            self._redeye_panel.set_image(display, cfg.redeye_mask)

        # ── Onglet Originale ──────────────────────────────────────────────────
        self._origin_view.set_image(img)

        # ── Preview : afficher l'originale immédiatement, le preview calculé ensuite ──
        if img is not None:
            self._preview_view.set_image(img)
            self._preview_status_lbl.setText("…")

        # ── Onglet Résultat (chargement lazeux si onglet actif) ───────────────
        if self._tabs.currentIndex() == _TAB_RESULT:
            self._update_dest_view(cfg, force=True)

        fname = os.path.basename(cfg.file_path)
        self._statusbar.showMessage(f"{fname}  —  prêt")

        # Réinitialiser le mode export et recharger la liste
        self._is_viewing_export = False
        self._viewed_export_entry = None
        self._viewed_export_data = None
        self._refresh_export_dropdown(cfg)

        # Mettre à jour les panneaux (mosaïque ou aperçu rapide)
        self._schedule_preview_update()
        # Actualiser les onglets diff si actifs
        self._refresh_if_diff_tab()
        self._update_result_diff_indicator()

    # ── Sauvegarde de l'état ──────────────────────────────────────────────────

    def _save_current_state(self) -> None:
        """Sauvegarde l'état courant de l'UI dans _current_cfg."""
        cfg = self._current_cfg
        if cfg is None:
            return
        cfg.step_order      = self._step_list.get_order()
        cfg.step_enabled    = self._step_list.get_enabled()
        cfg.step_params     = self._step_list.get_all_params()
        self._sync_editor_state()
        save_recipe(cfg)
        # Vider la pile undo : chaque image a sa propre histoire de changements
        self._undo_stack.clear()
        self._preset_buf.clear()
        self._preset_flush_pending = False
        self._update_result_diff_indicator()

    def _sync_editor_state(self) -> None:
        """Copie l'état des panneaux éditeurs dans _current_cfg (mémoire, pas disque)."""
        cfg = self._current_cfg
        if cfg is None:
            return
        raw_mask = self._mask_panel.get_mask()
        cfg.inpaint_mask = raw_mask if (raw_mask is not None and raw_mask.any()) else None
        raw_redeye = self._redeye_panel.get_mask()
        cfg.redeye_mask = raw_redeye if (raw_redeye is not None and raw_redeye.any()) else None
        cfg.wb_pick = self._wb_panel.get_pick_point()
        cfg.wb_patch_radius = self._wb_panel.get_patch_radius()

    def _config_for(self, file_path: str) -> Optional[BatchImageConfig]:
        for cfg in self._session.images:
            if cfg.file_path == file_path:
                return cfg
        return None

    # ── Injection état dans les singletons ────────────────────────────────────

    def _inject_instance_state(self, cfg: BatchImageConfig) -> None:
        """Injecte le masque et le point WB dans les singletons d'étapes."""
        inpaint_step = self._steps_by_id.get("inpaint")
        if inpaint_step and hasattr(inpaint_step, "set_mask"):
            if cfg.inpaint_mask is not None:
                inpaint_step.set_mask(cfg.inpaint_mask)
            else:
                inpaint_step.clear_mask()

        redeye_step = self._steps_by_id.get("redeye")
        if redeye_step and hasattr(redeye_step, "set_redeye_mask"):
            if cfg.redeye_mask is not None:
                redeye_step.set_redeye_mask(cfg.redeye_mask)
            else:
                redeye_step.clear_redeye_mask()

        wb_step = self._steps_by_id.get("wb")
        if wb_step and hasattr(wb_step, "set_pick_point"):
            if cfg.wb_pick is not None:
                wb_step.set_pick_point(*cfg.wb_pick)
            else:
                wb_step.clear_pick_point()

    def _clear_instance_state(self) -> None:
        """Réinitialise les singletons à la fermeture."""
        inpaint_step = self._steps_by_id.get("inpaint")
        if inpaint_step and hasattr(inpaint_step, "clear_mask"):
            inpaint_step.clear_mask()
        redeye_step = self._steps_by_id.get("redeye")
        if redeye_step and hasattr(redeye_step, "clear_redeye_mask"):
            redeye_step.clear_redeye_mask()
        wb_step = self._steps_by_id.get("wb")
        if wb_step and hasattr(wb_step, "clear_pick_point"):
            wb_step.clear_pick_point()

    # ── Rechargement depuis disque ────────────────────────────────────────────

    def _reload_from_disk(self) -> None:
        """Recharge le recipe .json depuis le disque et réinitialise l'UI."""
        from ui.notifications import Level
        cfg = self._current_cfg
        if cfg is None:
            self._statusbar.showMessage("Aucune image sélectionnée.")
            return
        recipe = load_recipe(cfg.file_path)
        if recipe is None:
            self._statusbar.showMessage(
                f"Aucun recipe .json trouvé pour : {os.path.basename(cfg.file_path)}"
            )
            return
        _apply_recipe(cfg, recipe)
        self._navigate_to(cfg)
        self._undo_stack.clear()
        self._statusbar.showMessage(
            f"Recipe rechargé depuis le disque : {os.path.basename(cfg.file_path)}"
        )
        if hasattr(self, "_notif"):
            self._notif.notify(
                "Recipe rechargé",
                os.path.basename(cfg.file_path),
                level=Level.WARNING,
                duration=3000,
            )

    def _on_strip_image_reset(self, file_path: str) -> None:
        cfg = self._config_for(file_path)
        if cfg is None:
            return
        self._session.reset_to_defaults(cfg)
        self._strip.set_customized(file_path, False)
        self._strip.set_done(file_path, False)
        if self._current_cfg is cfg:
            self._navigate_to(cfg)
        self._statusbar.showMessage(f"{os.path.basename(file_path)} — réinitialisé.")
