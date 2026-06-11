"""core/genref_store.py — Versions de référence IA stockées à côté de la source.

En mode batch, chaque génération FLUX est conservée comme une « version »
sidecar dans le répertoire d'entrée, sur le modèle des masques :

    {stem}.genref.001.json   métadonnées (prompt, style, graine, date…)
    {stem}.genref.001.png    image de référence FLUX (≈1 Mpx)

Principes :
- l'historique n'est JAMAIS supprimé automatiquement (suppression manuelle
  uniquement) : une bonne référence reste applicable même si l'image de base
  change (la LUT est ré-ajustée à la volée par l'étape) ;
- les fichiers .genref.* ne doivent pas être chargés comme images du batch
  (exclusion dans BatchSession.load_folder).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# Marqueur commun à tous les fichiers sidecar de référence IA
GENREF_SIDECAR_TAG = ".genref."


@dataclass
class GenRefVersion:
    """Une version de référence générée pour une image source."""

    source_path: str                  # chemin de l'image source
    index:       int                  # numéro NNN
    prompt:      str = ""
    style:       str = ""
    seed:        int = 0
    cast:        str = ""
    items:       list = field(default_factory=list)
    created_at:  str = ""

    @property
    def json_path(self) -> str:
        return _sidecar_path(self.source_path, self.index, "json")

    @property
    def png_path(self) -> str:
        return _sidecar_path(self.source_path, self.index, "png")

    @property
    def label(self) -> str:
        """Libellé court pour l'UI (liste des versions)."""
        date = self.created_at[:16].replace("T", " ") if self.created_at else "?"
        return f"v{self.index:03d} · {self.style or '?'} · graine {self.seed} · {date}"

    def load_ref(self) -> Optional[np.ndarray]:
        """Charge l'image de référence (BGR) depuis le disque."""
        return cv2.imread(self.png_path, cv2.IMREAD_COLOR)


def _sidecar_path(source_path: str, index: int, ext: str) -> str:
    source_dir = os.path.dirname(source_path)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(source_dir, f"{stem}{GENREF_SIDECAR_TAG}{index:03d}.{ext}")


def list_versions(source_path: str) -> list[GenRefVersion]:
    """Liste les versions existantes pour cette image, triées par index."""
    source_dir = os.path.dirname(os.path.abspath(source_path))
    stem = os.path.splitext(os.path.basename(source_path))[0]
    prefix = f"{stem}{GENREF_SIDECAR_TAG}"
    out: list[GenRefVersion] = []
    try:
        entries = os.scandir(source_dir)
    except OSError:
        return out
    for entry in entries:
        name = entry.name
        if not (entry.is_file() and name.startswith(prefix)
                and name.endswith(".json")):
            continue
        try:
            index = int(name[len(prefix):-len(".json")])
        except ValueError:
            continue
        version = load_version(source_path, index)
        if version is not None:
            out.append(version)
    out.sort(key=lambda v: v.index)
    return out


def load_version(source_path: str, index: int) -> Optional[GenRefVersion]:
    """Charge les métadonnées d'une version (sans l'image de référence)."""
    path = _sidecar_path(source_path, index, "json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    return GenRefVersion(
        source_path=source_path,
        index=index,
        prompt=str(meta.get("prompt", "")),
        style=str(meta.get("style", "")),
        seed=int(meta.get("seed", 0)),
        cast=str(meta.get("cast", "")),
        items=list(meta.get("items", [])),
        created_at=str(meta.get("created_at", "")),
    )


def next_index(source_path: str) -> int:
    versions = list_versions(source_path)
    return versions[-1].index + 1 if versions else 1


def save_version(
    source_path: str,
    ref_bgr:     np.ndarray,
    prompt:      str,
    style:       str,
    seed:        int,
    cast:        str = "",
    items:       Optional[list] = None,
) -> GenRefVersion:
    """Crée une nouvelle version (index suivant) à côté de la source."""
    version = GenRefVersion(
        source_path=source_path,
        index=next_index(source_path),
        prompt=prompt,
        style=style,
        seed=int(seed),
        cast=cast,
        items=list(items or []),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    meta = {
        "version": 1,
        "index": version.index,
        "created_at": version.created_at,
        "prompt": version.prompt,
        "style": version.style,
        "seed": version.seed,
        "cast": version.cast,
        "items": version.items,
    }
    with open(version.json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    cv2.imwrite(version.png_path, ref_bgr)
    return version


def delete_version(version: GenRefVersion) -> bool:
    """Supprime les fichiers d'une version (action manuelle uniquement)."""
    ok = True
    for path in (version.json_path, version.png_path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            ok = False
    return ok
