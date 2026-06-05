"""ui/mixins/params.py — Paramètres des étapes, ordre du pipeline, undo/redo."""

from __future__ import annotations

from typing import Optional

from ui.param_history import SetParamCommand, MoveStepCommand, ToggleStepCommand
from ui.mixins.preview import _FAST_PREVIEW_IDS


class ParamsMixin:

    def _on_order_changed(self, new_order: list[str]):
        if not self._applying_order:
            self._step_order = new_order
        self._mark_stale_from(new_order[0] if new_order else None)
        self._update_result_diff_indicator()

    def _on_order_reordered(self, old_order: list[str], new_order: list[str]) -> None:
        """Push une commande undo pour le réordonnancement par drag-and-drop."""
        if self._applying_order:
            return

        def apply(order: list) -> None:
            self._applying_order = True
            self._ctrl.step_list.set_order(order)
            self._step_order = order
            self._mark_stale_from(order[0] if order else None)
            self._update_result_diff_indicator()
            self._applying_order = False

        cmd = MoveStepCommand(
            step_id   = new_order[0] if new_order else "",
            old_order = old_order,
            new_order = new_order,
            apply_fn  = apply,
        )
        self._undo_stack.push(cmd)

    def _on_param_changed(self, step_id: str, key: str, value):
        """Capture old_val, met à jour le snapshot, enregistre la commande undo."""
        # Capturer l'ancienne valeur AVANT de mettre à jour le snapshot
        old_val = self._params_snapshot.get(step_id, {}).get(key)
        # Mettre à jour le snapshot
        self._params_snapshot.setdefault(step_id, {})[key] = value

        # Détecter si on est dans un chargement de preset (macro undo)
        panel = self._ctrl.step_list.get_panel(step_id)
        is_preset = panel is not None and panel.is_loading_preset()

        if is_preset:
            # Bufferer pour créer une macro via QTimer.singleShot(0)
            self._preset_buf.append((step_id, key, old_val, value))
            if not self._preset_flush_pending:
                self._preset_flush_pending = True
                from PyQt6.QtCore import QTimer as _QTimer
                _QTimer.singleShot(0, self._flush_preset_undo)
        else:
            # Vider un éventuel buffer résiduel (ne devrait pas arriver)
            if self._preset_buf:
                self._flush_preset_undo()
            cmd = SetParamCommand(
                self._apply_param_silent, step_id, key, old_val, value
            )
            self._undo_stack.push(cmd)

        self._mark_stale_from(step_id)
        self._update_result_diff_indicator()
        self._schedule_preview(step_id)

    def _apply_param_silent(self, step_id: str, key: str, val) -> None:
        """Applique val dans le widget (sans émettre de signal) et met à jour le snapshot."""
        panel = self._ctrl.step_list.get_panel(step_id)
        if panel is None:
            return
        row = panel._param_rows.get(key)
        if row is not None:
            row.set_value(val)  # _block=True, pas de signal
        self._params_snapshot.setdefault(step_id, {})[key] = val
        self._update_result_diff_indicator()
        self._schedule_preview(step_id)

    def _flush_preset_undo(self) -> None:
        """Vide le buffer preset et pousse une macro undo regroupant tous les changements."""
        self._preset_flush_pending = False
        buf = self._preset_buf
        self._preset_buf = []
        if not buf:
            return
        self._undo_stack.beginMacro("Changer profil")
        for sid, key, old, new in buf:
            cmd = SetParamCommand(self._apply_param_silent, sid, key, old, new)
            self._undo_stack.push(cmd)  # redo() immédiat → no-op (_first=True)
        self._undo_stack.endMacro()

    def _on_enabled_changed(self, step_id: str, enabled: bool):
        if self._applying_enabled:
            return
        old_val = self._step_enabled.get(step_id)
        self._step_enabled[step_id] = enabled
        panel = self._ctrl.step_list.get_panel(step_id)
        if panel:
            panel.set_state("disabled" if not enabled else "stale")
        if step_id in _FAST_PREVIEW_IDS:
            self._schedule_preview(step_id)
        self._update_result_diff_indicator()
        # Pousser la commande undo
        if old_val is not None and old_val != enabled:
            cmd = ToggleStepCommand(
                self._apply_enabled_silent, step_id, old_val, enabled
            )
            self._undo_stack.push(cmd)

    def _apply_enabled_silent(self, step_id: str, val: bool) -> None:
        """Applique l'état activé sans émettre de commande undo."""
        self._applying_enabled = True
        self._step_enabled[step_id] = val
        panel = self._ctrl.step_list.get_panel(step_id)
        if panel:
            panel._enable_cb.blockSignals(True)
            panel._enable_cb.setChecked(val)
            panel._enable_cb.blockSignals(False)
            panel.set_state("disabled" if not val else "stale")
        self._update_result_diff_indicator()
        self._applying_enabled = False

    def _mark_stale_from(self, step_id: Optional[str]):
        """Marque comme obsolètes toutes les étapes à partir de step_id."""
        if step_id is None:
            return
        if step_id not in self._step_order:
            return
        start = self._step_order.index(step_id)
        for sid in self._step_order[start:]:
            panel = self._ctrl.step_list.get_panel(sid)
            if panel and panel.is_enabled():
                panel.set_state("stale")
