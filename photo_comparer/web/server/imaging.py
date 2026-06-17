"""Service d'images : sert les fichiers web-natifs tels quels, transcode en JPEG
les formats que les navigateurs ne savent pas afficher (TIFF/TIF), avec un cache
disque par (chemin, mtime). Parité avec le fallback Pillow de ``image_loader.py``.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

# Formats que les navigateurs affichent nativement → servis sans transcodage.
WEB_NATIVE = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
# Formats à transcoder en JPEG pour l'affichage navigateur.
TRANSCODE = {".tiff", ".tif"}

CACHE_DIR = Path.home() / ".photo_comparer_cache" / "display"


def _cache_path(image_path: Path, suffix: str) -> Path:
    try:
        mtime = str(image_path.stat().st_mtime)
    except OSError:
        mtime = "0"
    raw = f"{image_path.resolve()}{mtime}{suffix}".encode()
    return CACHE_DIR / (hashlib.md5(raw).hexdigest() + ".jpg")


def resolve(image_path: Path) -> Tuple[Optional[bytes], Optional[Path], str]:
    """Retourne ``(bytes, file_path, media_type)``.

    - Pour un format web-natif : ``(None, chemin_original, media_type)`` — le serveur
      pourra renvoyer un ``FileResponse`` (streaming efficace).
    - Pour un format à transcoder : ``(jpeg_bytes, chemin_cache, "image/jpeg")``.
    - En cas d'erreur : ``(None, None, "")``.
    """
    ext = image_path.suffix.lower()

    if ext in WEB_NATIVE:
        return None, image_path, _media_type(ext)

    if ext in TRANSCODE:
        cache = _cache_path(image_path, "full")
        if cache.exists():
            return None, cache, "image/jpeg"
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=92)
                data = buf.getvalue()
            try:
                cache.write_bytes(data)
            except OSError:
                pass
            return data, None, "image/jpeg"
        except Exception:
            return None, None, ""

    # Format inconnu : tenter un transcodage Pillow par sécurité.
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=92)
            return buf.getvalue(), None, "image/jpeg"
    except Exception:
        return None, None, ""


def _media_type(ext: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")
