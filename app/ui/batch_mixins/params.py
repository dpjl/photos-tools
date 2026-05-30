"""batch_mixins/params.py — Gestion des paramètres, undo/redo, propagation."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSlot

from core.batch import save_recipe
from ui.param_history import (
    SetParamCommand, PropagateParamCommand,
    PropagateEnabledCommand, MoveStepCommand, ToggleStepCommand,
)
from ui.batch_window_constants import (
    _FAST_PREVIEW_IDS, _TAB_MASK, _TAB_WB, _TAB_REDEYE,
)


class ParamsMixin:
    """Changement de paramètres, undo/redo, propagation à la sélection."""

    # ── Changements d'ordre ───────────────────────────────────────────────────

    @pyqtSlot(list)
    def _on_order_changed(self, order: list[str]) -> None:
        if not self._applying_order and self._current_cfg:
            self._current_cfg.step_order = order
            self._mark_customized()

    @pyqtSlot(list, list)
    def _on_order_reordered(self, old_order: list, new_order: list) -> None:
        if self._applying_order:
            return
        cfg = self._current_cfg
        def apply(order: list) -> None:
            self._applying_order = True
            self._step_list.set_order(order)
            if cfg:
                cfg.step_order = order
            self._applying_order = False
        cmd = MoveStepCommand(
            step_id   = new_order[0] if new_order else "",
            old_order = old_order,
            new_order = new_order,
            apply_fn  = apply,
        )
        self._undo_stack.push(cmd)

    # ── Changements de paramètres ─────────────────────────────────────────────

    @pyqtSlot(str, str, object)
    def _on_param_changed(self, step_id: str, key: str, value) -> None:
        if self._applying_export_view:
            return
        if self._current_cfg:
            old_val = self._current_cfg.step_params.get(step_id, {}).get(key)
            self._current_cfg.step_params.setdefault(step_id, {})[key] = value
            panel = self._step_list.get_panel(step_id)
            is_preset = panel is not None and panel.is_loading_preset()
            if not is_preset:
                self._mark_customized()
            if is_preset:
                self._preset_buf.append((step_id, key, old_val, value))
                if not self._preset_flush_pending:
                    self._preset_flush_pending = True
                    from PyQt6.QtCore import QTimer as _QTimer
                    _QTimer.singleShot(0, self._flush_preset_undo)
            else:
                if self._preset_buf:
                    self._flush_preset_undo()
                cmd = SetParamCommand(
                    self._apply_param_silent, step_id, key, old_val, value
                )
                self._undo_stack.push(cmd)

        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview_update()
            if self._full_preview_worker is not None and self._full_preview_worker.isRunning():
                self._full_preview_worker.cancel()
                self._preview_status_lbl.setText("Preview rapide")
                self._full_preview_btn.setEnabled(True)

    def _apply_param_silent(self, step_id: str, key: str, val) -> None:
        """Applique val silencieusement (undo/redo) : widget + cfg.step_params + preview."""
        if self._current_cfg:
            self._current_cfg.step_params.setdefault(step_id, {})[key] = val
        panel = self._step_list.get_panel(step_id)
        if panel:
            row = panel._param_rows.get(key)
            if row is not None:
                row.set_value(val)
        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview_update()

    def _flush_preset_undo(self) -> None:
        self._preset_flush_pending = False
        buf = self._preset_buf
        self._preset_buf = []
        if not buf:
            return
        self._undo_stack.beginMacro("Changer profil")
        for sid, key, old, new in buf:
            cmd = SetParamCommand(self._apply_param_silent, sid, key, old, new)
            self._undo_stack.push(cmd)
        self._undo_stack.endMacro()

    # ── Propagation de paramètres ─────────────────────────────────────────────

    @pyqtSlot(str, str, object)
    def _on_param_propagate(self, step_id: str, key: str, value) -> None:
        """Propage un seul paramètre aux images sélectionnées (ou toutes si aucune)."""
        from ui.notifications import Level
        self._save_current_state()
        run_paths = self._strip.get_run_selection()
        if run_paths:
            targets = [c for c in self._session.images if c.file_path in set(run_paths)]
        else:
            targets = list(self._session.images)
        if not targets:
            return
        old_vals = {
            cfg.file_path: cfg.step_params.get(step_id, {}).get(key)
            for cfg in targets
        }
        for cfg in targets:
            cfg.step_params.setdefault(step_id, {})[key] = value
            save_recipe(cfg)
        if self._current_cfg in targets:
            panel = self._step_list.get_panel(step_id)
            if panel:
                row = panel._param_rows.get(key)
                if row is not None:
                    row.set_value(value)
            if step_id in _FAST_PREVIEW_IDS:
                self._schedule_preview_update()
        cmd = PropagateParamCommand(
            step_id, key, value,
            targets, old_vals,
            lambda: self._current_cfg,
            self._apply_param_silent,
            save_recipe,
        )
        self._undo_stack.push(cmd)
        n = len(targets)
        self._statusbar.showMessage(f"Paramètre « {key} » propagé à {n} image(s).")
        if hasattr(self, "_notif"):
            self._notif.notify(
                f"Propagé : «{key}»",
                f"Étape {step_id}  •  {n} image(s)",
                level=Level.INFO,
                duration=3500,
            )

    @pyqtSlot(str, bool)
    def _on_enabled_propagate(self, step_id: str, enabled: bool) -> None:
        """Propage l'état activé/désactivé d'une étape aux images sélectionnées."""
        from ui.notifications import Level
        self._save_current_state()
        run_paths = self._strip.get_run_selection()
        if run_paths:
            targets = [c for c in self._session.images if c.file_path in set(run_paths)]
        else:
            targets = list(self._session.images)
        if not targets:
            return
        old_vals = {cfg.file_path: cfg.step_enabled.get(step_id) for cfg in targets}
        for cfg in targets:
            cfg.step_enabled[step_id] = enabled
            save_recipe(cfg)
        if self._current_cfg in targets:
            panel = self._step_list.get_panel(step_id)
            if panel:
                panel.set_enabled(enabled)

        def _apply_enabled_ui(sid: str, val: bool) -> None:
            p = self._step_list.get_panel(sid)
            if p:
                p.set_enabled(val)

        cmd = PropagateEnabledCommand(
            step_id, enabled, targets, old_vals,
            lambda: self._current_cfg,
            _apply_enabled_ui,
            save_recipe,
        )
        self._undo_stack.push(cmd)
        n = len(targets)
        state_str = "activé" if enabled else "désactivé"
        self._statusbar.showMessage(f"Étape « {step_id} » {state_str} sur {n} image(s).")
        if hasattr(self, "_notif"):
            self._notif.notify(
                f"Étape {state_str} : {step_id}",
                f"Propagé à {n} image(s)",
                level=Level.INFO,
                duration=3500,
            )

    # ── Activation/désactivation d'étape ──────────────────────────────────────

    @pyqtSlot(str, bool)
    def _on_enabled_changed(self, step_id: str, enabled: bool) -> None:
        if self._applying_enabled:
            return
        if self._applying_export_view:
            return
        old_val = None
        if self._current_cfg:
            old_val = self._current_cfg.step_enabled.get(step_id)
            self._current_cfg.step_enabled[step_id] = enabled
            self._mark_customized()
        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview_update()
        if old_val is not None and old_val != enabled:
            cmd = ToggleStepCommand(
                self._apply_enabled_silent, step_id, old_val, enabled
            )
            self._undo_stack.push(cmd)

    def _apply_enabled_silent(self, step_id: str, val: bool) -> None:
        self._applying_enabled = True
        panel = self._step_list.get_panel(step_id)
        if panel:
            panel._enable_cb.blockSignals(True)
            panel._enable_cb.setChecked(val)
            panel._enable_cb.blockSignals(False)
        if self._current_cfg:
            self._current_cfg.step_enabled[step_id] = val
        self._applying_enabled = False

    # ── Undo / Redo ───────────────────────────────────────────────────────────

    def _undo(self) -> None:
        """Ctrl+Z : route vers le bon gestionnaire selon l'onglet actif."""
        idx = self._tabs.currentIndex()
        if idx == _TAB_MASK:
            self._mask_panel._canvas.undo()
        elif idx == _TAB_WB:
            self._wb_panel._canvas.undo()
            if self._current_cfg:
                self._current_cfg.wb_pick = self._wb_panel._canvas._pick_pt
            self._schedule_preview_update()
        elif idx == _TAB_REDEYE:
            self._redeye_panel._canvas.undo()
        else:
            self._undo_stack.undo()

    def _redo(self) -> None:
        """Ctrl+Y : route vers le bon gestionnaire selon l'onglet actif."""
        idx = self._tabs.currentIndex()
        if idx == _TAB_MASK:
            self._mask_panel._canvas.redo()
        elif idx == _TAB_WB:
            self._wb_panel._canvas.redo()
            if self._current_cfg:
                self._current_cfg.wb_pick = self._wb_panel._canvas._pick_pt
            self._schedule_preview_update()
        elif idx == _TAB_REDEYE:
            self._redeye_panel._canvas.redo()
        else:
            self._undo_stack.redo()

    # ── Navigation éditeurs ───────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_mask_edit_requested(self, step_id: str) -> None:
        self._tabs.setCurrentIndex(_TAB_MASK)

    @pyqtSlot(str)
    def _on_color_picker_requested(self, step_id: str) -> None:
        self._tabs.setCurrentIndex(_TAB_WB)

    def _mark_customized(self) -> None:
        cfg = self._current_cfg
        if cfg and not cfg.customized:
            cfg.customized = True
            self._strip.set_customized(cfg.file_path, True)
