"""batch_mixins/exports.py — Gestion des exports versionnés (sans dropdown)."""

from __future__ import annotations

import os
from typing import Optional

from core.batch import BatchImageConfig
from core.export_manager import ExportManager, ExportEntry


class ExportsMixin:
    """Rafraîchissement de la liste d'exports et helpers recipe."""

    def _refresh_export_dropdown(self, cfg: BatchImageConfig) -> None:
        """Met à jour la liste d'exports pour l'image (utilisé par run/nav)."""
        mgr = self._session.get_export_manager()
        stem, ext = os.path.splitext(cfg.filename)
        self._viewed_export_list = mgr.list_exports(stem, ext)
        # Rafraîchir la mosaïque si elle est affichée
        if hasattr(self, "_export_mosaic"):
            self._update_dest_view(cfg, force=True)

    def _current_recipe_dict(self) -> Optional[dict]:
        """Génère le dict recipe depuis l'état actuel de l'UI."""
        cfg = self._current_cfg
        if cfg is None:
            return None
        pick    = self._wb_panel.get_pick_point()
        canvas  = self._mask_panel._canvas
        has_mask = canvas._mask is not None and bool(canvas._mask.any())
        return {
            "version":        2,
            "customized":     getattr(cfg, "customized", False),
            "step_order":     self._step_list.get_order(),
            "step_enabled":   self._step_list.get_enabled(),
            "step_params":    self._step_list.get_all_params(),
            "wb_pick":        list(pick) if pick else None,
            "wb_patch_radius": self._wb_panel.get_patch_radius(),
            "has_mask":       has_mask,
        }

    def _refresh_diff_source(self) -> None:
        """No-op — l'ancien onglet Δ Exports a été supprimé."""
        pass

    def _refresh_if_diff_tab(self) -> None:
        """No-op — l'ancien onglet Δ Exports a été supprimé."""
        pass
