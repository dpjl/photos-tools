"""core/history.py — Gestion de l'historique des exécutions du pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class HistoryEntry:
    """Une entrée dans l'historique : tous les paramètres + images d'un run."""
    run_id:       int
    timestamp:    datetime
    step_order:   list[str]             # IDs des étapes dans l'ordre d'exécution
    step_enabled: dict[str, bool]       # {step_id: enabled}
    step_params:  dict[str, dict]       # {step_id: {key: value}}
    step_results: dict[str, np.ndarray] # {step_id: image BGR}  ← peut être vidé (LRU)
    context:      dict                  # contexte partagé (face_bboxes, etc.)
    label:        str = ""

    def __post_init__(self):
        if not self.label:
            n = len(self.step_results)
            steps_short = ", ".join(
                s.split("·")[-1].strip() if "·" in s else s
                for s in self.step_order[:4]
            )
            self.label = f"Run #{self.run_id}"
            self.n_steps = n

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    @property
    def completed_steps(self) -> list[str]:
        return [s for s in self.step_order if s in self.step_results]

    def has_images(self) -> bool:
        return bool(self.step_results)


class HistoryManager:
    """Gestionnaire de l'historique — max 30 entrées, LRU sur les images."""

    MAX_ENTRIES = 30

    def __init__(self):
        self._entries: list[HistoryEntry] = []
        self._counter: int = 0

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def add(self, entry: HistoryEntry) -> None:
        self._entries.append(entry)
        # LRU : libérer les images de la plus ancienne entrée si on dépasse la limite
        if len(self._entries) > self.MAX_ENTRIES:
            oldest = self._entries.pop(0)
            oldest.step_results.clear()

    def get(self, run_id: int) -> Optional[HistoryEntry]:
        for e in self._entries:
            if e.run_id == run_id:
                return e
        return None

    def latest(self) -> Optional[HistoryEntry]:
        return self._entries[-1] if self._entries else None

    def all(self) -> list[HistoryEntry]:
        return list(self._entries)

    def clear(self):
        self._entries.clear()
        self._counter = 0
