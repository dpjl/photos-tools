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
    # Note : save_result attend result_img en paramètre explicite.

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
            and not e.endswith(".mask.png")
        )
        self.images = []
        for e in entries:
            full_path = os.path.join(path, e)
            cfg = self.make_config(full_path)
            recipe = load_recipe(full_path)
            if recipe:
                _apply_recipe(cfg, recipe)
            self.images.append(cfg)

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
        config:    BatchImageConfig,
        result_img: Optional[np.ndarray],
        step_log:  list[dict],
    ) -> tuple[str, str]:
        """Sauvegarde l'image résultat et le fichier sidecar JSON.

        result_img est passé explicitement (non stocké dans config) afin de
        libérer la mémoire dès que possible après l'écriture.
        Retourne (chemin_image, chemin_sidecar).
        """
        os.makedirs(self.output_dir, exist_ok=True)

        fname     = config.filename
        stem, ext = os.path.splitext(fname)
        out_img   = os.path.join(self.output_dir, fname)
        out_json  = os.path.join(self.output_dir, stem + ".result.json")

        # ── Image ─────────────────────────────────────────────────────────────
        if result_img is not None:
            _write_image(result_img, out_img)

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

    def save_recipe(self, config: "BatchImageConfig") -> Optional[str]:
        """Délègue à la fonction module-level save_recipe."""
        return save_recipe(config)


# ══════════════════════════════════════════════════════════════════════════════
# Sidecar recette (fichier source)
# ══════════════════════════════════════════════════════════════════════════════

def save_recipe(config: "BatchImageConfig") -> Optional[str]:
    """Sauvegarde la configuration de restauration («recette») à côté du fichier source.

    Crée ``{stem}.recipe.json`` et, si un masque inpaint est présent,
    ``{stem}.mask.png`` dans le même dossier que l'image source.

    Retourne le chemin du recipe, ou ``None`` si échec.
    """
    source_dir = os.path.dirname(config.file_path)
    stem       = os.path.splitext(os.path.basename(config.file_path))[0]
    recipe_path = os.path.join(source_dir, stem + ".recipe.json")

    recipe: dict = {
        "version":        1,
        "customized":     config.customized,
        "step_order":     config.step_order,
        "step_enabled":   config.step_enabled,
        "step_params":    config.step_params,
        "wb_pick":        list(config.wb_pick) if config.wb_pick else None,
        "wb_patch_radius": config.wb_patch_radius,
        "has_mask":       config.inpaint_mask is not None,
    }
    try:
        with open(recipe_path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)

        if config.inpaint_mask is not None:
            mask_path = os.path.join(source_dir, stem + ".mask.png")
            cv2.imwrite(mask_path, config.inpaint_mask)

        return recipe_path
    except Exception:
        return None


def load_recipe(file_path: str) -> Optional[dict]:
    """Charge le fichier recipe à côté du fichier source.

    Retourne le dictionnaire (avec éventuellement ``_mask`` injecté)
    ou ``None`` si absent ou invalide.
    """
    source_dir  = os.path.dirname(file_path)
    stem        = os.path.splitext(os.path.basename(file_path))[0]
    recipe_path = os.path.join(source_dir, stem + ".recipe.json")

    if not os.path.exists(recipe_path):
        return None
    try:
        with open(recipe_path, "r", encoding="utf-8") as f:
            data: dict = json.load(f)

        if data.get("has_mask"):
            mask_path = os.path.join(source_dir, stem + ".mask.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    data["_mask"] = mask

        return data
    except Exception:
        return None


def _apply_recipe(config: "BatchImageConfig", recipe: dict) -> None:
    """Applique un recipe chargé à une BatchImageConfig."""
    if "step_order" in recipe:
        config.step_order = list(recipe["step_order"])
    if "step_enabled" in recipe:
        # Fusionner plutôt que remplacer : les steps absents du recipe (ajoutés
        # après la sauvegarde) conservent leur valeur enabled_by_default issue
        # de make_config.
        config.step_enabled.update(recipe["step_enabled"])
    if "step_params" in recipe:
        # Même logique : fusionner pour conserver les params par défaut des
        # nouveaux steps.
        for k, v in recipe["step_params"].items():
            config.step_params[k] = dict(v)
    if recipe.get("wb_pick") is not None:
        config.wb_pick = tuple(recipe["wb_pick"])
    if "wb_patch_radius" in recipe:
        config.wb_patch_radius = int(recipe["wb_patch_radius"])
    config.customized = bool(recipe.get("customized", False))
    if "_mask" in recipe:
        config.inpaint_mask = recipe["_mask"]


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
