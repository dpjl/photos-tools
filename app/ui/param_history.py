"""ui/param_history.py — Commandes annulables pour les paramètres du pipeline.

Utilise le système QUndoCommand / QUndoStack de Qt (module QtGui depuis Qt6).

Commandes définies :
  - SetParamCommand        : changement d'un seul paramètre (merge sur glissement rapide)
  - MoveStepCommand        : réordonnancement d'une étape (drag-and-drop)
  - PropagateParamCommand  : propagation d'un paramètre à N images (mode batch)
  - PropagateEnabledCommand: propagation de l'état activé/désactivé (mode batch)
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from PyQt6.QtGui import QUndoCommand


def _cmd_id(step_id: str, key: str) -> int:
    """Identifiant stable et non nul pour le merge de commandes (même step/clé)."""
    h = hash(f"{step_id}\x00{key}") & 0x7FFF_FFFF
    return h if h != 0 else 1


class SetParamCommand(QUndoCommand):
    """Commande annulable pour le changement d'un seul paramètre.

    Comportement :
    - Le premier ``redo()`` est un no-op car la valeur est déjà dans le widget
      au moment du ``QUndoStack.push()`` (Qt appelle redo() immédiatement).
    - Des changements rapides sur le même (step_id, key) dans la fenêtre
      MERGE_MS sont fusionnés en un seul step d'undo.
    """

    MERGE_MS: float = 1_000.0  # fenêtre de fusion en millisecondes

    def __init__(
        self,
        apply_fn: Callable[[str, str, object], None],
        step_id:  str,
        key:      str,
        old_val,
        new_val,
    ) -> None:
        super().__init__(f"Param {step_id}/{key}")
        self._apply = apply_fn
        self._sid   = step_id
        self._key   = key
        self._old   = old_val
        self._new   = new_val
        self._time  = time.monotonic()
        self._first = True  # redo() immédiat au push → no-op

    # ── Interface QUndoCommand ────────────────────────────────────────────────

    def id(self) -> int:  # noqa: A003
        return _cmd_id(self._sid, self._key)

    def mergeWith(self, other: "QUndoCommand") -> bool:
        if not isinstance(other, SetParamCommand):
            return False
        if other._sid != self._sid or other._key != self._key:
            return False
        elapsed_ms = (other._time - self._time) * 1_000
        if elapsed_ms > self.MERGE_MS:
            return False
        # Absorber : conserver notre old_val, prendre le new_val du dernier
        self._new  = other._new
        self._time = other._time
        return True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return  # valeur déjà dans le widget via l'action utilisateur
        self._apply(self._sid, self._key, self._new)

    def undo(self) -> None:
        self._apply(self._sid, self._key, self._old)


class ToggleStepCommand(QUndoCommand):
    """Commande annulable pour l'activation/désactivation d'une étape."""

    def __init__(
        self,
        apply_fn: Callable[[str, bool], None],
        step_id:  str,
        old_val:  bool,
        new_val:  bool,
    ) -> None:
        verb = "activé" if new_val else "désactivé"
        super().__init__(f"Étape {step_id} {verb}")
        self._apply = apply_fn
        self._sid   = step_id
        self._old   = old_val
        self._new   = new_val
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self._apply(self._sid, self._new)

    def undo(self) -> None:
        self._apply(self._sid, self._old)


class MoveStepCommand(QUndoCommand):
    """Commande annulable pour le réordonnancement d'une étape par drag-and-drop.

    apply_fn reçoit le nouvel ordre complet (list[str]) et doit :
      - appeler step_list.set_order(order) (UI)
      - mettre à jour le modèle de données (état fenêtre / BatchImageConfig)
    """

    def __init__(
        self,
        step_id:   str,
        old_order: list,
        new_order: list,
        apply_fn:  Callable[[list], None],
    ) -> None:
        super().__init__(f"Déplacer étape {step_id}")
        self._old   = list(old_order)
        self._new   = list(new_order)
        self._apply = apply_fn
        self._first = True  # redo() no-op au push : l'ordre est déjà appliqué

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)


class PropagateParamCommand(QUndoCommand):
    """Commande annulable pour la propagation d'un paramètre à N images (mode batch).

    Les anciennes valeurs sont capturées au moment de la création (snapshot),
    ce qui garantit un undo correct même après navigation entre images.
    """

    def __init__(
        self,
        step_id:         str,
        key:             str,
        new_val,
        targets,          # list[BatchImageConfig]
        old_vals: dict,   # {file_path → old_value}   — snapshot au moment du push
        get_current_fn:  Callable,   # () → Optional[BatchImageConfig]
        apply_silent_fn: Callable,   # (step_id, key, val) → None  (cfg courante)
        save_recipe_fn:  Callable,   # (cfg) → None
    ) -> None:
        n = len(targets)
        super().__init__(f"Propager {key} → {n} image(s)")
        self._sid     = step_id
        self._key     = key
        self._new     = new_val
        self._targets = list(targets)
        self._old     = dict(old_vals)
        self._get_cur = get_current_fn
        self._apply   = apply_silent_fn
        self._save    = save_recipe_fn
        self._first   = True  # redo() immédiat au push → no-op

    def redo(self) -> None:
        if self._first:
            self._first = False
            return  # déjà effectué par l'action de propagation directe
        current = self._get_cur()
        for cfg in self._targets:
            cfg.step_params.setdefault(self._sid, {})[self._key] = self._new
            self._save(cfg)
        if current is not None and current in self._targets:
            self._apply(self._sid, self._key, self._new)

    def undo(self) -> None:
        current = self._get_cur()
        for cfg in self._targets:
            old = self._old.get(cfg.file_path)
            if old is not None:
                cfg.step_params.setdefault(self._sid, {})[self._key] = old
            self._save(cfg)
        if current is not None and current in self._targets:
            old = self._old.get(current.file_path)
            if old is not None:
                self._apply(self._sid, self._key, old)


class PropagateEnabledCommand(QUndoCommand):
    """Commande annulable pour la propagation de l'état activé/désactivé d'une étape."""

    def __init__(
        self,
        step_id:        str,
        new_val:        bool,
        targets,         # list[BatchImageConfig]
        old_vals: dict,  # {file_path → old_bool}
        get_current_fn:  Callable,
        apply_ui_fn:     Callable,   # (step_id, enabled) → None  (panel UI de l'image courante)
        save_recipe_fn:  Callable,
    ) -> None:
        state_str = "activé" if new_val else "désactivé"
        super().__init__(f"Étape {step_id} {state_str} → {len(targets)} image(s)")
        self._sid     = step_id
        self._new     = new_val
        self._targets = list(targets)
        self._old     = dict(old_vals)
        self._get_cur = get_current_fn
        self._apply   = apply_ui_fn
        self._save    = save_recipe_fn
        self._first   = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        current = self._get_cur()
        for cfg in self._targets:
            cfg.step_enabled[self._sid] = self._new
            self._save(cfg)
        if current is not None and current in self._targets:
            self._apply(self._sid, self._new)

    def undo(self) -> None:
        current = self._get_cur()
        for cfg in self._targets:
            old = self._old.get(cfg.file_path)
            if old is not None:
                cfg.step_enabled[self._sid] = old
            self._save(cfg)
        if current is not None and current in self._targets:
            old = self._old.get(current.file_path)
            if old is not None:
                self._apply(self._sid, old)
