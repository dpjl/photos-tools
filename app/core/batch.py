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

# Fichier de session batch (mémorise output_dir et autres préférences de session)
_SESSION_FILENAME = ".batch_session.json"


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

    # État des étapes avec état instance (inpaint / WB / redeye / zones rouges)
    inpaint_mask:    Optional[np.ndarray] = field(default=None, repr=False)
    redeye_mask:     Optional[np.ndarray] = field(default=None, repr=False)
    redzone_mask:    Optional[np.ndarray] = field(default=None, repr=False)
    crop_rect:       Optional[tuple[float, float, float, float]] = None
    wb_pick:         Optional[tuple[int, int]] = None
    wb_patch_radius: int = 5
    # Version de référence IA active ({stem}.genref.NNN.*) — None = aucune
    genref_version:  Optional[int] = None

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
            redeye_mask     = self.redeye_mask.copy() if self.redeye_mask is not None else None,
            redzone_mask    = self.redzone_mask.copy() if self.redzone_mask is not None else None,
            crop_rect       = self.crop_rect,
            wb_pick         = self.wb_pick,
            wb_patch_radius = self.wb_patch_radius,
            genref_version  = self.genref_version,
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

        # Restaurer le dossier de sortie de la session précédente si disponible
        session_path = os.path.join(path, _SESSION_FILENAME)
        if os.path.exists(session_path):
            try:
                with open(session_path, encoding="utf-8") as _f:
                    _meta = json.load(_f)
                if _meta.get("output_dir"):
                    self.output_dir = _meta["output_dir"]
            except Exception:
                pass

        from core.genref_store import GENREF_SIDECAR_TAG
        entries = sorted(
            e for e in os.listdir(path)
            if os.path.splitext(e)[1].lower() in IMAGE_EXTENSIONS
            and not e.endswith(".mask.png")
            and not e.endswith(".redeye_mask.png")
            and not e.endswith(".redzone_mask.png")
            and GENREF_SIDECAR_TAG not in e   # références IA sidecar ≠ images du batch
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

    def save_session_meta(self) -> None:
        """Persiste les préférences de session (output_dir) dans .batch_session.json."""
        if not self.source_dir:
            return
        path = os.path.join(self.source_dir, _SESSION_FILENAME)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"output_dir": self.output_dir}, f, ensure_ascii=False)
        except Exception:
            pass

    def reset_to_defaults(self, config: BatchImageConfig) -> None:
        """Réinitialise une config aux valeurs du batch defaults."""
        d = self._defaults
        if d is None:
            return
        config.step_order      = list(d.step_order)
        config.step_enabled    = dict(d.step_enabled)
        config.step_params     = {k: dict(v) for k, v in d.step_params.items()}
        config.inpaint_mask    = None
        config.redeye_mask     = None
        config.redzone_mask    = None
        config.crop_rect       = None
        config.wb_pick         = None
        config.wb_patch_radius = d.wb_patch_radius
        config.genref_version  = None
        config.customized      = False
        config.result_img      = None
        config.context         = {}

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    def get_export_manager(self):
        """Retourne un ExportManager pour le répertoire de sortie courant."""
        from core.export_manager import ExportManager
        return ExportManager(self.output_dir)

    def save_result(
        self,
        config:    BatchImageConfig,
        result_img: Optional[np.ndarray],
    ) -> Optional["ExportEntry"]:
        """Sauvegarde un export versionné (image + recipe + masques).

        Retourne l'ExportEntry créé, ou None si l'image n'a pas été fournie.
        """
        if result_img is None:
            return None

        mgr = self.get_export_manager()
        stem, ext = os.path.splitext(config.filename)

        recipe = build_recipe_dict(config)
        masks = {
            "mask":         config.inpaint_mask,
            "redeye_mask":  config.redeye_mask,
            "redzone_mask": config.redzone_mask,
        }

        entry = mgr.save_export(stem, ext, result_img, recipe, masks)
        return entry

    def save_recipe(self, config: "BatchImageConfig") -> Optional[str]:
        """Délègue à la fonction module-level save_recipe."""
        return save_recipe(config)


# ══════════════════════════════════════════════════════════════════════════════
# Construction du dict recipe à partir d'une config
# ══════════════════════════════════════════════════════════════════════════════

def build_recipe_dict(config: "BatchImageConfig") -> dict:
    """Construit un dict recipe depuis une BatchImageConfig.

    Utilisé par ExportManager et par save_recipe().
    """
    return {
        "version":         2,
        "source_filename": config.filename,
        "customized":      config.customized,
        "step_order":      config.step_order,
        "step_enabled":    config.step_enabled,
        "step_params":     config.step_params,
        "crop_rect":       list(config.crop_rect) if config.crop_rect else None,
        "wb_pick":         list(config.wb_pick) if config.wb_pick else None,
        "wb_patch_radius":  config.wb_patch_radius,
        "genref_version":   config.genref_version,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Sidecar recette (fichier source)
# ══════════════════════════════════════════════════════════════════════════════
def _mask_has_pixels(mask: Optional[np.ndarray]) -> bool:
    """Retourne True uniquement si mask est un ndarray contenant au moins un pixel non nul.

    Un tableau all-zeros (masque vide mais alloué) est traité comme 'pas de masque'.
    """
    return mask is not None and bool(mask.any())

def save_recipe(config: "BatchImageConfig") -> Optional[str]:
    """Sauvegarde la configuration de restauration («recette») à côté du fichier source.

    Crée ``{stem}.recipe.json`` et, si un masque inpaint est présent,
    ``{stem}.mask.png`` dans le même dossier que l'image source.

    Retourne le chemin du recipe, ou ``None`` si échec.
    """
    source_dir = os.path.dirname(config.file_path)
    stem       = os.path.splitext(os.path.basename(config.file_path))[0]
    recipe_path = os.path.join(source_dir, stem + ".recipe.json")

    recipe = build_recipe_dict(config)
    # Le recipe de travail utilise version 1 (format local)
    recipe["version"] = 1
    recipe["has_mask"] = _mask_has_pixels(config.inpaint_mask)
    recipe["has_redeye_mask"] = _mask_has_pixels(config.redeye_mask)
    recipe["has_redzone_mask"] = _mask_has_pixels(config.redzone_mask)
    # Pas besoin de source_filename dans le recipe de travail
    recipe.pop("source_filename", None)
    try:
        with open(recipe_path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)

        if _mask_has_pixels(config.inpaint_mask):
            mask_path = os.path.join(source_dir, stem + ".mask.png")
            cv2.imwrite(mask_path, config.inpaint_mask)

        if _mask_has_pixels(config.redeye_mask):
            redeye_path = os.path.join(source_dir, stem + ".redeye_mask.png")
            cv2.imwrite(redeye_path, config.redeye_mask)

        if _mask_has_pixels(config.redzone_mask):
            redzone_path = os.path.join(source_dir, stem + ".redzone_mask.png")
            cv2.imwrite(redzone_path, config.redzone_mask)

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

        if data.get("has_redeye_mask"):
            redeye_path = os.path.join(source_dir, stem + ".redeye_mask.png")
            if os.path.exists(redeye_path):
                rmask = cv2.imread(redeye_path, cv2.IMREAD_GRAYSCALE)
                if rmask is not None:
                    data["_redeye_mask"] = rmask

        if data.get("has_redzone_mask"):
            redzone_path = os.path.join(source_dir, stem + ".redzone_mask.png")
            if os.path.exists(redzone_path):
                zmask = cv2.imread(redzone_path, cv2.IMREAD_GRAYSCALE)
                if zmask is not None:
                    data["_redzone_mask"] = zmask

        return data
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# Exports versionnés — stockés à côté de la SOURCE
# (même répertoire que {stem}.recipe.json et {stem}.mask.png)
# ════════════════════════════════════════════════════════════════════════════

def list_export_recipes(file_path: str) -> list[tuple[int, str]]:
    """Liste les exports versionnés ``{stem}.export.NNN.json`` à côté de la source.

    Retourne ``[(N, chemin_absolu), ...]`` trié par N croissant.
    Utilise os.scandir (pas glob) pour être robuste face aux caractères spéciaux
    dans le chemin du répertoire (crochets, points d'interrogation, etc.).
    """
    source_dir = os.path.dirname(os.path.abspath(file_path))
    stem   = os.path.splitext(os.path.basename(file_path))[0]
    prefix = f"{stem}.export."
    suffix = ".json"
    result: list[tuple[int, str]] = []
    try:
        for entry in os.scandir(source_dir):
            if not entry.is_file():
                continue
            name = entry.name
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            try:
                num_str = name[len(prefix) : -len(suffix)]
                n = int(num_str)
                if n > 0:
                    result.append((n, entry.path))
            except (ValueError, IndexError):
                continue
    except OSError:
        pass
    result.sort(key=lambda x: x[0])
    return result


def next_export_index(file_path: str) -> int:
    """Retourne le prochain numéro d'export disponible (max existant + 1, ou 1)."""
    exports = list_export_recipes(file_path)
    return exports[-1][0] + 1 if exports else 1


def save_export_recipe(config: "BatchImageConfig") -> Optional[str]:
    """Sauvegarde un export versionné ``{stem}.export.NNN.json`` à côté de la source.

    Appelé après chaque traitement réussi. Même structure que save_recipe()
    avec en plus les clés ``exported_at`` et ``export_index``.
    Retourne le chemin du fichier créé, ou ``None`` si échec.
    """
    try:
        source_dir = os.path.dirname(config.file_path)
        stem  = os.path.splitext(os.path.basename(config.file_path))[0]
        idx   = next_export_index(config.file_path)
        fname = f"{stem}.export.{idx:03d}.json"
        path  = os.path.join(source_dir, fname)
        recipe: dict = {
            "version":        1,
            "exported_at":    datetime.now().isoformat(timespec="seconds"),
            "export_index":   idx,
            "customized":     config.customized,
            "step_order":     config.step_order,
            "step_enabled":   config.step_enabled,
            "step_params":    config.step_params,
            "crop_rect":      list(config.crop_rect) if config.crop_rect else None,
            "wb_pick":        list(config.wb_pick) if config.wb_pick else None,
            "wb_patch_radius": config.wb_patch_radius,
            "genref_version":  config.genref_version,
            "has_mask":       _mask_has_pixels(config.inpaint_mask),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return None


def load_export_recipe(path: str) -> Optional[dict]:
    """Charge un export versionné depuis son chemin absolu.

    Retourne le dictionnaire ou ``None`` si absent ou invalide.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _merge_step_order(saved: list[str], canonical: list[str]) -> list[str]:
    """Complète un ordre d'étapes sauvegardé avec les étapes ajoutées depuis.

    Chaque étape de ``canonical`` absente de ``saved`` est insérée juste après
    la dernière étape canonique qui la précède et qui est présente — elle
    retrouve ainsi sa position par défaut dans le pipeline (et pas en fin de
    liste, où elle s'exécuterait après recadrage/upscale).
    """
    order = list(saved)
    for idx, sid in enumerate(canonical):
        if sid in order:
            continue
        preceding = [p for p in canonical[:idx] if p in order]
        pos = order.index(preceding[-1]) + 1 if preceding else 0
        order.insert(pos, sid)
    return order


def _apply_recipe(config: "BatchImageConfig", recipe: dict) -> None:
    """Applique un recipe chargé à une BatchImageConfig.

    Les étapes absentes du recipe (ajoutées à l'application après la
    sauvegarde) sont remises à leur état par défaut : insérées à leur
    position canonique dans l'ordre, désactivées si tel est leur défaut,
    paramètres par défaut. Sans cela, un recipe antérieur à une étape
    l'exclurait du pipeline (preview/run) ou, à l'inverse, une restauration
    conserverait l'état coché courant d'une étape inconnue du résultat.
    """
    from steps import ALL_STEPS
    canonical = [s.id for s in ALL_STEPS]
    defaults_enabled = {s.id: s.enabled_by_default for s in ALL_STEPS}
    defaults_params = {s.id: s.default_params() for s in ALL_STEPS}

    if "step_order" in recipe:
        config.step_order = _merge_step_order(
            list(recipe["step_order"]), canonical
        )
    if "step_enabled" in recipe:
        saved_enabled = recipe["step_enabled"]
        for sid in canonical:
            if sid not in saved_enabled:
                config.step_enabled[sid] = defaults_enabled[sid]
        config.step_enabled.update(saved_enabled)
    if "step_params" in recipe:
        saved_params = recipe["step_params"]
        for sid in canonical:
            if sid not in saved_params:
                config.step_params[sid] = dict(defaults_params[sid])
        for k, v in saved_params.items():
            config.step_params[k] = dict(v)
    if "crop_rect" in recipe:
        rect = recipe.get("crop_rect")
        config.crop_rect = (
            tuple(float(v) for v in rect)
            if rect is not None else None
        )
    if recipe.get("wb_pick") is not None:
        config.wb_pick = tuple(recipe["wb_pick"])
    if "wb_patch_radius" in recipe:
        config.wb_patch_radius = int(recipe["wb_patch_radius"])
    if "genref_version" in recipe:
        raw = recipe.get("genref_version")
        config.genref_version = int(raw) if raw is not None else None
    config.customized = bool(recipe.get("customized", False))
    if "_mask" in recipe:
        config.inpaint_mask = recipe["_mask"]
    if "_redeye_mask" in recipe:
        config.redeye_mask = recipe["_redeye_mask"]
    if "_redzone_mask" in recipe:
        config.redzone_mask = recipe["_redzone_mask"]


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

        if step_id == "redzone" and "redzone_mask_pixels" in context:
            entry["mask_pixels"] = int(context["redzone_mask_pixels"])

        if step_id == "wb":
            if context.get("wb_applied"):
                entry["wb_pick"]      = list(context.get("wb_pick", []))
                entry["wb_patch_rgb"] = [round(v, 1) for v in context.get("wb_patch_rgb", [])]
                entry["wb_muls"]      = [round(v, 4) for v in context.get("wb_muls", [])]

        log.append(entry)
    return log
