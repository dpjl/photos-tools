"""core/image_info.py — Helpers compacts d'information image."""

from __future__ import annotations

import os
from fractions import Fraction
from typing import Optional

from PIL import ExifTags, Image


_EXIF_NAMES = {
    ExifTags.TAGS.get(271, "Make"): "Appareil",
    ExifTags.TAGS.get(272, "Model"): "Modèle",
    ExifTags.TAGS.get(33434, "ExposureTime"): "Temps",
    ExifTags.TAGS.get(33437, "FNumber"): "Ouverture",
    ExifTags.TAGS.get(34855, "ISOSpeedRatings"): "ISO",
    ExifTags.TAGS.get(37386, "FocalLength"): "Focale",
    ExifTags.TAGS.get(42036, "LensModel"): "Objectif",
}

_EXIF_ORDER = (
    "Prise",
    "Appareil",
    "Modèle",
    "Objectif",
    "Focale",
    "Temps",
    "Ouverture",
    "ISO",
)

_DATE_TAGS = (
    36867,  # DateTimeOriginal
    36868,  # DateTimeDigitized
    306,    # DateTime, frequent dans les TIFF/scans
)
_EXIF_IFD_TAG = int(ExifTags.IFD.Exif)


def image_dimensions(path: str) -> Optional[tuple[int, int]]:
    """Retourne (largeur, hauteur) sans charger toute l'image si possible."""
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def format_file_size(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:
        return "?"
    units = ("o", "Ko", "Mo", "Go")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "o":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} o"


def format_dimensions(dims: Optional[tuple[int, int]]) -> str:
    if not dims:
        return "?"
    return f"{int(dims[0])}×{int(dims[1])}"


def read_exif_summary(path: str, max_items: int = 8) -> list[tuple[str, str]]:
    """Retourne quelques EXIF utiles sous forme (label, valeur)."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return []
            raw = {}
            date_value = _first_exif_value(exif, _DATE_TAGS)
            if date_value not in (None, ""):
                raw["Prise"] = _format_exif_value("Prise", date_value)
            for key, value in exif.items():
                name = ExifTags.TAGS.get(key, str(key))
                label = _EXIF_NAMES.get(name)
                if label and value not in (None, ""):
                    raw[label] = _format_exif_value(label, value)
            out = [(label, raw[label]) for label in _EXIF_ORDER if label in raw]
            return out[:max_items]
    except Exception:
        return []
    return []


def _first_exif_value(exif: Image.Exif, tags: tuple[int, ...]):
    """Lit un tag dans l'IFD principal puis l'IFD EXIF, par ordre de priorité."""
    for tag in tags:
        value = exif.get(tag)
        if value not in (None, ""):
            return value
    try:
        exif_ifd = exif.get_ifd(_EXIF_IFD_TAG)
    except Exception:
        exif_ifd = {}
    for tag in tags:
        value = exif_ifd.get(tag)
        if value not in (None, ""):
            return value
    return None


def format_preview_info(
    source_path: str,
    original_dims: Optional[tuple[int, int]],
    preview_dims: Optional[tuple[int, int]],
) -> str:
    lines = [
        f"Fichier  {format_file_size(source_path)}",
        f"Original {format_dimensions(original_dims)}",
        f"Preview  {format_dimensions(preview_dims)}",
    ]
    exif = read_exif_summary(source_path)
    if exif:
        lines.append("")
        lines.extend(f"{label:<8} {value}" for label, value in exif)
    else:
        lines.append("")
        lines.append("EXIF     aucune donnée")
    return "\n".join(lines)


def _format_exif_value(label: str, value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if label == "Temps":
        text = _fraction_text(value)
        return text if "/" in text else f"{text}s"
    if label == "Ouverture":
        return "f/" + _number_text(value)
    if label == "Focale":
        return _number_text(value) + " mm"
    if isinstance(value, tuple):
        return ", ".join(_number_text(v) for v in value)
    return str(value).strip()


def _number_text(value) -> str:
    if isinstance(value, Fraction):
        value = float(value)
    try:
        f = float(value)
    except Exception:
        return str(value)
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    return f"{f:.2f}".rstrip("0").rstrip(".")


def _fraction_text(value) -> str:
    if isinstance(value, Fraction):
        if value.denominator > 1:
            return f"{value.numerator}/{value.denominator}s"
        return f"{value.numerator}s"
    try:
        f = float(value)
    except Exception:
        return str(value)
    if 0 < f < 1:
        den = round(1.0 / f)
        if den > 1:
            return f"1/{den}s"
    return _number_text(f)
