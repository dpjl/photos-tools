"""core/batch.py — Modèle de données pour le mode batch.

BatchImageConfig : configuration et résultat pour une image du batch.
BatchSession     : session complète (dossier source, dossier de sortie, liste d'images).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# Extensions d'image acceptées
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ══════════════════════════════════════════════════════════════════════════════
# Configuration par image
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchImageConfig:
    """Configuration complète pour une image du batch.

    Chaque image peut avoir ses propres paramètres d'étapes, son propre ordre,
    son masque d'inpainting et son point de balance des blancs.
    """

    file_path:       str

    # Paramètres du pipeline (initialisés depuis les defaults du batch)
    step_order:      list[str]           = field(default_factory=list)
    step_enabled:    dict[str, bool]     = field(default_factory=dict)
    step_params:     dict[str, dict]     = field(default_factory=dict)

    # État des étapes avec état instance (inpaint / WB)
    inpaint_mask:    Optional[np.ndarray] = field(default=None, repr=False)
    wb_pick:         Optional[tuple[int, int]] = None
    wb_patch_radius: int = 5

    # Résultat du dernier run (Tester ou batch) — non sérialisé
    result_img:      Optional[np.ndarray] = field(default=None, repr=False)
    context:         dict                 = field(default_factory=dict)

    # Indicateur visuel : différent des defaults du batch ?
    customized:      bool = False

    # Statut batch : None / "pending" / "running" / "done" / "error"
    batch_status:    Optional[str] = None

    @property
    def filename(self) -> str:
        return os.path.basename(self.file_path)

    def deep_copy_params(self) -> "BatchImageConfig":
        """Retourne une copie profonde des paramètres (sans images volumineuses)."""
        cfg = BatchImageConfig(
            file_path       = self.file_path,
            step_order      = list(self.step_order),
            step_enabled    = dict(self.step_enabled),
            step_params     = {k: dict(v) for k, v in self.step_params.items()},
            inpaint_mask    = self.inpaint_mask.copy() if self.inpaint_mask is not None else None,
            wb_pick         = self.wb_pick,
            wb_patch_radius = self.wb_patch_radius,
            customized      = self.customized,
        )
        return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Session batch
# ══════════════════════════════════════════════════════════════════════════════

class BatchSession:
    """Session batch : dossier source, dossier de sortie, configs par image.

    Usage ::

        session = BatchSession()
        session.load_folder("/path/to/photos", defaults_config)
        # session.images contient une BatchImageConfig par image
        session.output_dir = "/path/to/output"
        session.save_result(config, result_img, step_log)
    """

    def __init__(self) -> None:
        self.source_dir:  str = ""
        self.output_dir:  str = ""
        self.images:      list[BatchImageConfig] = []
        self._defaults:   Optional[BatchImageConfig] = None   # prototype (snapshot)

    # ── Chargement ────────────────────────────────────────────────────────────

    def load_folder(
        self,
        path:     str,
        defaults: BatchImageConfig,
    ) -> None:
        """Scanne le dossier et crée une BatchImageConfig par image."""
        self.source_dir = path
        self._defaults  = defaults

        # Dossier de sortie par défaut : sous-dossier "batch_output"
        self.output_dir = os.path.join(path, "batch_output")

        entries = sorted(
            e for e in os.listdir(path)
            if os.path.splitext(e)[1].lower() in IMAGE_EXTENSIONS
        )
        self.images = [self.make_config(os.path.join(path, e)) for e in entries]

    def make_config(self, file_path: str) -> BatchImageConfig:
        """Crée une BatchImageConfig initialisée depuis les defaults."""
        d = self._defaults
        if d is None:
            return BatchImageConfig(file_path=file_path)
        return BatchImageConfig(
            file_path       = file_path,
            step_order      = list(d.step_order),
            step_enabled    = dict(d.step_enabled),
            step_params     = {k: dict(v) for k, v in d.step_params.items()},
            wb_patch_radius = d.wb_patch_radius,
        )

    def reset_to_defaults(self, config: BatchImageConfig) -> None:
        """Réinitialise une config aux valeurs du batch defaults."""
        d = self._defaults
        if d is None:
            return
        config.step_order      = list(d.step_order)
        config.step_enabled    = dict(d.step_enabled)
        config.step_params     = {k: dict(v) for k, v in d.step_params.items()}
        config.inpaint_mask    = None
        config.wb_pick         = None
        config.wb_patch_radius = d.wb_patch_radius
        config.customized      = False
        config.result_img      = None
        config.context         = {}

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    def save_result(
        self,
        config:   BatchImageConfig,
        step_log: list[dict],
    ) -> tuple[str, str]:
        """Sauvegarde l'image résultat et le fichier sidecar JSON.

        Retourne (chemin_image, chemin_sidecar).
        """
        os.makedirs(self.output_dir, exist_ok=True)

        fname     = config.filename
        stem, ext = os.path.splitext(fname)
        out_img   = os.path.join(self.output_dir, fname)
        out_json  = os.path.join(self.output_dir, stem + ".json")

        # ── Image ─────────────────────────────────────────────────────────────
        if config.result_img is not None:
            _write_image(config.result_img, out_img)

        # ── Sidecar JSON ──────────────────────────────────────────────────────
        sidecar = {
            "source":       fname,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "output":       fname,
            "steps":        step_log,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)

        return out_img, out_json


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def _write_image(img: np.ndarray, path: str) -> bool:
    """Écrit img vers path en déduisant le format depuis l'extension."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".jpg", ".jpeg"):
            return cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif ext == ".png":
            return cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            return cv2.imwrite(path, img)
    except Exception:
        return False


def build_step_log(
    step_order:   list[str],
    step_enabled: dict[str, bool],
    step_params:  dict[str, dict],
    step_results: dict[str, str],   # step_id → "ok" | "error" | "skipped"
    context:      dict,
    steps_by_id:  dict,             # step_id → StepBase instance
) -> list[dict]:
    """Construit la liste des entrées du sidecar JSON pour les étapes."""
    log = []
    for step_id in step_order:
        step    = steps_by_id.get(step_id)
        name    = step.name if step else step_id
        enabled = step_enabled.get(step_id, True)
        params  = step_params.get(step_id, {})
        status  = step_results.get(step_id, "skipped")

        entry: dict = {
            "id":      step_id,
            "name":    name,
            "enabled": enabled,
            "status":  status,
            "params":  params,
        }

        # Extras spécifiques par étape
        if step_id == "redeye" and "redeye_detections" in context:
            corrected_eyes = [
                {
                    "center": [int(d["iris"][0]), int(d["iris"][1])],
                    "radius": int(d["iris"][2]),
                }
                for d in context["redeye_detections"]
                if d.get("corrected", False)
            ]
            entry["corrected_eyes"] = corrected_eyes

        if step_id == "inpaint" and "inpaint_mask_pixels" in context:
            entry["mask_pixels"] = int(context["inpaint_mask_pixels"])

        if step_id == "wb":
            if context.get("wb_applied"):
                entry["wb_pick"]      = list(context.get("wb_pick", []))
                entry["wb_patch_rgb"] = [round(v, 1) for v in context.get("wb_patch_rgb", [])]
                entry["wb_muls"]      = [round(v, 4) for v in context.get("wb_muls", [])]

        log.append(entry)
    return log
