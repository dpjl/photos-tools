"""batch_mixins/exports.py — Visualisation d'exports versionnés, diff JSON."""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import pyqtSlot

from core.batch import (
    BatchImageConfig, load_export_recipe,
    list_export_recipes, _apply_recipe,
)
from ui.batch_window_constants import _TAB_DIFF


class ExportsMixin:
    """Dropdown exports, mode lecture seule, diff JSON, restauration."""

    def _refresh_export_dropdown(self, cfg: BatchImageConfig) -> None:
        """Repeuple le combo exports pour l'image donnée et revient à 'Courante'."""
        exports = list_export_recipes(cfg.file_path)
        self._viewed_export_list = exports
        self._export_combo.blockSignals(True)
        self._export_combo.clear()
        self._export_combo.addItem("Courante")
        for n, path in exports:
            label = f"Export {n:03d}"
            self._export_combo.addItem(label, userData=path)
        self._export_combo.setCurrentIndex(0)
        self._export_combo.blockSignals(False)
        self._export_combo.setVisible(True)

    @pyqtSlot(int)
    def _on_export_selected(self, index: int) -> None:
        if index <= 0:
            self._set_viewing_export(None)
        else:
            path = self._export_combo.itemData(index)
            self._set_viewing_export(path)

    def _set_viewing_export(
        self,
        path: Optional[str],
        _navigate_call: bool = False,
    ) -> None:
        """Active ou désactive le mode lecture seule sur un export versionné."""
        if path is None:
            # Retour en mode édition
            self._is_viewing_export  = False
            self._viewed_export_path = None
            self._viewed_export_data = None
            self._step_list.setEnabled(True)
            self._restore_export_btn.setVisible(False)
            cfg = self._current_cfg
            if cfg is not None:
                self._applying_order = True
                for sid, enabled in cfg.step_enabled.items():
                    panel = self._step_list.get_panel(sid)
                    if panel:
                        panel.set_enabled(enabled)
                for sid, params in cfg.step_params.items():
                    panel = self._step_list.get_panel(sid)
                    if panel:
                        panel.set_params(params)
                self._step_list.set_order(cfg.step_order)
                self._applying_order = False
            if not _navigate_call:
                self._export_combo.blockSignals(True)
                self._export_combo.setCurrentIndex(0)
                self._export_combo.blockSignals(False)
            self._refresh_if_diff_tab()
            if not _navigate_call:
                self._schedule_preview_update()
        else:
            # Mode lecture seule
            data = load_export_recipe(path)
            if data is None:
                self._statusbar.showMessage("Impossible de charger l'export.")
                self._export_combo.blockSignals(True)
                self._export_combo.setCurrentIndex(0)
                self._export_combo.blockSignals(False)
                return
            self._is_viewing_export  = True
            self._viewed_export_path = path
            self._viewed_export_data = data
            self._applying_order       = True
            self._applying_export_view = True
            if "step_order" in data:
                self._step_list.set_order(data["step_order"])
            if "step_enabled" in data:
                for sid, val in data["step_enabled"].items():
                    panel = self._step_list.get_panel(sid)
                    if panel:
                        panel.set_enabled(val)
            if "step_params" in data:
                for sid, params in data["step_params"].items():
                    panel = self._step_list.get_panel(sid)
                    if panel:
                        panel.set_params(params)
            self._applying_order       = False
            self._applying_export_view = False
            self._step_list.setEnabled(False)
            self._restore_export_btn.setVisible(True)
            n_str = os.path.basename(path)
            self._statusbar.showMessage(f"Lecture seule : {n_str}")
            self._refresh_if_diff_tab()
            self._schedule_preview_update()

    def _restore_export(self) -> None:
        """Restaure la configuration de l'export visionné dans _current_cfg."""
        from ui.notifications import Level
        if not self._is_viewing_export or not self._viewed_export_path:
            return
        cfg = self._current_cfg
        if cfg is None:
            return
        data = load_export_recipe(self._viewed_export_path)
        if data is None:
            return
        _apply_recipe(cfg, data)
        self._set_viewing_export(None)
        self._applying_order = True
        for sid, enabled in cfg.step_enabled.items():
            panel = self._step_list.get_panel(sid)
            if panel:
                panel.set_enabled(enabled)
        for sid, params in cfg.step_params.items():
            panel = self._step_list.get_panel(sid)
            if panel:
                panel.set_params(params)
        self._step_list.set_order(cfg.step_order)
        self._applying_order = False
        self._statusbar.showMessage("Configuration restaurée depuis l'export (non sauvegardée).")
        if hasattr(self, "_notif"):
            stem = os.path.splitext(os.path.basename(cfg.file_path))[0]
            self._notif.notify(
                "Configuration restaurée",
                f"{stem} — export chargé en mémoire",
                level=Level.WARNING,
                duration=4000,
            )

    # ── Diff JSON ─────────────────────────────────────────────────────────────

    def _current_recipe_dict(self) -> Optional[dict]:
        """Génère le dict recipe depuis l'état actuel de l'UI."""
        cfg = self._current_cfg
        if cfg is None:
            return None
        pick    = self._wb_panel.get_pick_point()
        canvas  = self._mask_panel._canvas
        has_mask = canvas._mask is not None and bool(canvas._mask.any())
        return {
            "version":        1,
            "customized":     getattr(cfg, "customized", False),
            "step_order":     self._step_list.get_order(),
            "step_enabled":   self._step_list.get_enabled(),
            "step_params":    self._step_list.get_all_params(),
            "wb_pick":        list(pick) if pick else None,
            "wb_patch_radius": self._wb_panel.get_patch_radius(),
            "has_mask":       has_mask,
        }

    def _refresh_diff_source(self) -> None:
        """Actualise l'onglet Δ Exports."""
        if self._current_cfg is None:
            self._diff_source_panel.update_diff(None, None)
            return

        stem = os.path.splitext(os.path.basename(self._current_cfg.file_path))[0]
        exports = self._viewed_export_list

        if self._is_viewing_export and self._viewed_export_path:
            current_data = load_export_recipe(self._viewed_export_path)
            idx_in_list = next(
                (i for i, (_, p) in enumerate(exports) if p == self._viewed_export_path),
                None,
            )
            if idx_in_list is not None and idx_in_list > 0:
                prev_n, prev_path = exports[idx_in_list - 1]
                ref_data = load_export_recipe(prev_path)
                ref_label = f"{stem} — Export {prev_n:03d}"
            else:
                ref_data  = None
                ref_label = "(pas de version précédente)"
            cur_n = exports[idx_in_list][0] if idx_in_list is not None else "?"
            cur_label = f"{stem} — Export {cur_n:03d}"
        else:
            current_data = self._current_recipe_dict()
            if exports:
                last_n, last_path = exports[-1]
                ref_data  = load_export_recipe(last_path)
                ref_label = f"{stem} — Export {last_n:03d}"
            else:
                ref_data  = None
                ref_label = "(pas encore exporté)"
            cur_label = f"{stem} (actuel)"

        self._diff_source_panel.update_diff(
            current_data, ref_data,
            left_label  = cur_label,
            right_label = ref_label,
        )

    def _refresh_if_diff_tab(self) -> None:
        """Actualise le panneau diff si l'onglet actif est l'onglet Δ Exports."""
        if self._tabs.currentIndex() == _TAB_DIFF:
            self._refresh_diff_source()
