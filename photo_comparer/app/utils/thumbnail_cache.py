"""Thumbnail generation and disk-cache utilities.

Thumbnails are stored in ~/.photo_comparer_cache/ as small JPEG files.
The cache key is an MD5 of the absolute path + mtime so stale entries
are automatically regenerated.
"""
import hashlib
from pathlib import Path

from PIL import Image

THUMB_SIZE = (200, 160)
CACHE_DIR = Path.home() / ".photo_comparer_cache"


def _cache_key(image_path: Path) -> str:
    try:
        mtime = str(image_path.stat().st_mtime)
    except OSError:
        mtime = "0"
    raw = f"{image_path.resolve()}{mtime}".encode()
    return hashlib.md5(raw).hexdigest()


def get_thumb_path(image_path: Path) -> Path:
    return CACHE_DIR / (_cache_key(image_path) + ".jpg")


def generate_thumbnail(image_path: Path) -> Path | None:
    """Return path to (possibly newly generated) thumbnail, or None on error."""
    thumb = get_thumb_path(image_path)
    if thumb.exists():
        return thumb
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)
            img.save(thumb, "JPEG", quality=70, optimize=True)
        return thumb
    except Exception:
        return None
