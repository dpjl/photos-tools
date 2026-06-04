"""core/export_manager.py — Service unifié de gestion des exports versionnés.

ExportEntry   : représente un export versionné sur disque.
ExportManager : CRUD sur les exports dans un répertoire de sortie.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Modèle
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExportEntry:
    """Représente un export versionné sur disque."""

    index:            int
    image_path:       str
    recipe_path:      str
    mask_path:        Optional[str]       = None
    redeye_mask_path: Optional[str]       = None
    exported_at:      Optional[str]       = None  # ISO 8601
    recipe_data:      Optional[dict]      = field(default=None, repr=False)


# ══════════════════════════════════════════════════════════════════════════════
# Nommage
# ══════════════════════════════════════════════════════════════════════════════

def _export_stem(stem: str, index: int) -> str:
    """Retourne le préfixe ``{stem}.export.{NNN}``."""
    return f"{stem}.export.{index:03d}"


def _image_name(stem: str, index: int, ext: str) -> str:
    return f"{_export_stem(stem, index)}{ext}"


def _recipe_name(stem: str, index: int) -> str:
    return f"{_export_stem(stem, index)}.recipe.json"


def _mask_name(stem: str, index: int) -> str:
    return f"{_export_stem(stem, index)}.mask.png"


def _redeye_mask_name(stem: str, index: int) -> str:
    return f"{_export_stem(stem, index)}.redeye_mask.png"


# ══════════════════════════════════════════════════════════════════════════════
# Service
# ══════════════════════════════════════════════════════════════════════════════

class ExportManager:
    """Gère les exports versionnés dans un répertoire de sortie.

    Nommage dans output_dir :
        photo.export.001.tiff
        photo.export.001.recipe.json
        photo.export.001.mask.png
        photo.export.001.redeye_mask.png
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    # ── Listing ───────────────────────────────────────────────────────────────

    def list_exports(self, stem: str, ext: str) -> list[ExportEntry]:
        """Liste les exports versionnés pour un stem donné.

        Scanne le répertoire de sortie à la recherche de fichiers
        ``{stem}.export.{NNN}{ext}`` et de leurs fichiers associés.
        Retourne une liste triée par index croissant.
        """
        if not os.path.isdir(self.output_dir):
            return []

        prefix = f"{stem}.export."
        entries: dict[int, ExportEntry] = {}

        try:
            for item in os.scandir(self.output_dir):
                if not item.is_file():
                    continue
                name = item.name
                if not name.startswith(prefix):
                    continue

                # Essayer de matcher image : {stem}.export.{NNN}{ext}
                img_suffix = ext
                if name.endswith(img_suffix):
                    num_str = name[len(prefix):-len(img_suffix)]
                    try:
                        n = int(num_str)
                    except ValueError:
                        continue
                    if n <= 0:
                        continue
                    if n not in entries:
                        entries[n] = ExportEntry(
                            index=n,
                            image_path=item.path,
                            recipe_path=os.path.join(
                                self.output_dir, _recipe_name(stem, n)
                            ),
                        )
                    else:
                        entries[n].image_path = item.path
        except OSError:
            return []

        # Compléter avec les masques et charger exported_at depuis le recipe
        for n, entry in entries.items():
            mask_p = os.path.join(self.output_dir, _mask_name(stem, n))
            if os.path.exists(mask_p):
                entry.mask_path = mask_p

            redeye_p = os.path.join(self.output_dir, _redeye_mask_name(stem, n))
            if os.path.exists(redeye_p):
                entry.redeye_mask_path = redeye_p

            if os.path.exists(entry.recipe_path):
                try:
                    with open(entry.recipe_path, encoding="utf-8") as f:
                        data = json.load(f)
                    entry.exported_at = data.get("exported_at")
                    entry.recipe_data = data
                except Exception:
                    pass

        return sorted(entries.values(), key=lambda e: e.index)

    def select_thumbnail_exports(
        self,
        sources: list[tuple[str, str]],
    ) -> dict[tuple[str, str], tuple[ExportEntry, bool]]:
        """Sélectionne les exports à afficher en vignettes en un seul scan.

        Pour chaque ``(stem, ext)``, retourne l'export retenu si disponible,
        sinon le dernier export. Le booléen indique si l'export est retenu.
        """
        if not os.path.isdir(self.output_dir):
            return {}

        keys_by_stem: dict[str, list[tuple[str, str]]] = {}
        for stem, ext in sources:
            keys_by_stem.setdefault(stem, []).append((stem, ext))
        for keys in keys_by_stem.values():
            keys.sort(key=lambda item: len(item[1]), reverse=True)

        candidates: dict[tuple[str, str], dict[int, str]] = {
            (stem, ext): {} for stem, ext in sources
        }
        marker = ".export."

        try:
            for item in os.scandir(self.output_dir):
                if not item.is_file() or marker not in item.name:
                    continue
                stem, suffix = item.name.rsplit(marker, 1)
                for key in keys_by_stem.get(stem, []):
                    ext = key[1]
                    if not suffix.endswith(ext):
                        continue
                    num_str = suffix[: -len(ext)] if ext else suffix
                    try:
                        index = int(num_str)
                    except ValueError:
                        continue
                    if index > 0:
                        candidates[key][index] = item.path
                    break
        except OSError:
            return {}

        best_map = self._load_best_map()
        selected: dict[tuple[str, str], tuple[ExportEntry, bool]] = {}
        for key, by_index in candidates.items():
            if not by_index:
                continue
            stem, ext = key
            best_idx = best_map.get(stem)
            if best_idx in by_index:
                index = best_idx
                is_best = True
            else:
                index = max(by_index)
                is_best = False
            selected[key] = (
                ExportEntry(
                    index=index,
                    image_path=by_index[index],
                    recipe_path=os.path.join(
                        self.output_dir, _recipe_name(stem, index)
                    ),
                ),
                is_best,
            )
        return selected

    # ── Index ─────────────────────────────────────────────────────────────────

    def next_index(self, stem: str, ext: str) -> int:
        """Retourne le prochain numéro d'export (max existant + 1, ou 1)."""
        exports = self.list_exports(stem, ext)
        return exports[-1].index + 1 if exports else 1

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    def save_export(
        self,
        stem:        str,
        ext:         str,
        image:       np.ndarray,
        recipe:      dict,
        masks:       Optional[dict[str, Optional[np.ndarray]]] = None,
    ) -> ExportEntry:
        """Sauvegarde un export versionné (image + recipe + masques).

        Parameters
        ----------
        stem : nom de base du fichier source (sans extension)
        ext  : extension du fichier source (ex: ``.tiff``)
        image : image résultat (BGR uint8)
        recipe : dictionnaire de paramètres de la recette
        masks : ``{"mask": ndarray|None, "redeye_mask": ndarray|None}``

        Returns
        -------
        ExportEntry créé.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        idx = self.next_index(stem, ext)
        masks = masks or {}

        # ── Image ─────────────────────────────────────────────────────────
        img_path = os.path.join(self.output_dir, _image_name(stem, idx, ext))
        _write_image(image, img_path)

        # ── Masques ───────────────────────────────────────────────────────
        has_mask = _mask_has_pixels(masks.get("mask"))
        has_redeye = _mask_has_pixels(masks.get("redeye_mask"))

        mask_path = None
        if has_mask:
            mask_path = os.path.join(self.output_dir, _mask_name(stem, idx))
            cv2.imwrite(mask_path, masks["mask"])

        redeye_path = None
        if has_redeye:
            redeye_path = os.path.join(
                self.output_dir, _redeye_mask_name(stem, idx)
            )
            cv2.imwrite(redeye_path, masks["redeye_mask"])

        # ── Recipe ────────────────────────────────────────────────────────
        recipe_out = dict(recipe)
        recipe_out["version"] = 2
        recipe_out["exported_at"] = datetime.now().isoformat(timespec="seconds")
        recipe_out["export_index"] = idx
        recipe_out["has_mask"] = has_mask
        recipe_out["has_redeye_mask"] = has_redeye

        recipe_path = os.path.join(self.output_dir, _recipe_name(stem, idx))
        with open(recipe_path, "w", encoding="utf-8") as f:
            json.dump(recipe_out, f, ensure_ascii=False, indent=2)

        return ExportEntry(
            index=idx,
            image_path=img_path,
            recipe_path=recipe_path,
            mask_path=mask_path,
            redeye_mask_path=redeye_path,
            exported_at=recipe_out["exported_at"],
            recipe_data=recipe_out,
        )

    # ── Suppression ───────────────────────────────────────────────────────────

    def delete_export(self, entry: ExportEntry) -> None:
        """Supprime un export et tous ses fichiers associés."""
        for path in (
            entry.image_path,
            entry.recipe_path,
            entry.mask_path,
            entry.redeye_mask_path,
        ):
            if path and os.path.exists(path):
                os.remove(path)

    # ── Chargement ────────────────────────────────────────────────────────────

    def load_recipe(self, entry: ExportEntry) -> Optional[dict]:
        """Charge le recipe JSON d'un export."""
        if entry.recipe_data is not None:
            return entry.recipe_data
        if not os.path.exists(entry.recipe_path):
            return None
        try:
            with open(entry.recipe_path, encoding="utf-8") as f:
                data = json.load(f)
            entry.recipe_data = data
            return data
        except Exception:
            return None

    def load_image(self, entry: ExportEntry) -> Optional[np.ndarray]:
        """Charge l'image exportée (BGR uint8)."""
        if not os.path.exists(entry.image_path):
            return None
        img = cv2.imread(entry.image_path, cv2.IMREAD_COLOR)
        return img

    def load_mask(self, entry: ExportEntry) -> Optional[np.ndarray]:
        """Charge le masque inpainting de l'export."""
        if not entry.mask_path or not os.path.exists(entry.mask_path):
            return None
        return cv2.imread(entry.mask_path, cv2.IMREAD_GRAYSCALE)

    def load_redeye_mask(self, entry: ExportEntry) -> Optional[np.ndarray]:
        """Charge le masque yeux rouges de l'export."""
        if not entry.redeye_mask_path or not os.path.exists(entry.redeye_mask_path):
            return None
        return cv2.imread(entry.redeye_mask_path, cv2.IMREAD_GRAYSCALE)

    # ── Sélection du meilleur export ──────────────────────────────────────────

    def _best_path(self) -> str:
        return os.path.join(self.output_dir, "_best.json")

    def _load_best_map(self) -> dict[str, int]:
        p = self._best_path()
        if not os.path.exists(p):
            return {}
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_best_map(self, data: dict[str, int]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self._best_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_best_index(self, stem: str) -> Optional[int]:
        """Retourne l'index de l'export marqué comme meilleur, ou None."""
        return self._load_best_map().get(stem)

    def set_best(self, stem: str, index: int) -> None:
        """Marque un export comme le meilleur pour un stem donné."""
        data = self._load_best_map()
        data[stem] = index
        self._save_best_map(data)

    def clear_best(self, stem: str) -> None:
        """Retire la sélection du meilleur export pour un stem."""
        data = self._load_best_map()
        if stem in data:
            del data[stem]
            self._save_best_map(data)

    def get_best_entry(self, stem: str, ext: str) -> Optional[ExportEntry]:
        """Retourne le meilleur export (ou le dernier si non défini)."""
        exports = self.list_exports(stem, ext)
        if not exports:
            return None
        best_idx = self.get_best_index(stem)
        if best_idx is not None:
            for e in exports:
                if e.index == best_idx:
                    return e
        return exports[-1]  # dernier par défaut


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def _mask_has_pixels(mask: Optional[np.ndarray]) -> bool:
    """True si mask est un ndarray contenant au moins un pixel non nul."""
    return mask is not None and bool(mask.any())


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
