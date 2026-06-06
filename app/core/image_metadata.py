"""core/image_metadata.py — Écriture d'images en conservant les métadonnées."""

from __future__ import annotations

import ctypes
import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import ExifTags, Image


_DATETIME_TAG = int(ExifTags.Base.DateTime)
_DATETIME_ORIGINAL_TAG = int(ExifTags.Base.DateTimeOriginal)
_DATETIME_DIGITIZED_TAG = int(ExifTags.Base.DateTimeDigitized)
_OFFSET_TIME_TAG = int(ExifTags.Base.OffsetTime)
_OFFSET_TIME_ORIGINAL_TAG = int(ExifTags.Base.OffsetTimeOriginal)
_OFFSET_TIME_DIGITIZED_TAG = int(ExifTags.Base.OffsetTimeDigitized)
_EXIF_IFD_TAG = int(ExifTags.IFD.Exif)

_DATE_TAGS = (
    (_DATETIME_ORIGINAL_TAG, _OFFSET_TIME_ORIGINAL_TAG),
    (_DATETIME_DIGITIZED_TAG, _OFFSET_TIME_DIGITIZED_TAG),
    (_DATETIME_TAG, _OFFSET_TIME_TAG),
)


def write_jpeg_with_source_exif(
    img: np.ndarray,
    path: str,
    source_path: str,
    quality: int = 95,
    optimize: bool = True,
    progressive: bool = True,
) -> bool:
    """Écrit un JPEG en recopiant l'EXIF et la date de prise de vue source.

    ``img`` est attendu au format BGR uint8, comme le reste du pipeline.
    Si la source ne contient pas d'EXIF exploitable, le JPEG est écrit sans
    métadonnées et sans ajustement de date.
    """
    try:
        exif_bytes = _read_exif_bytes(source_path)
        capture_date = _read_capture_datetime(source_path)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pil_img = _bgr_to_pillow(img)
        save_kwargs = {
            "quality": quality,
            "optimize": optimize,
            "progressive": progressive,
        }
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        pil_img.save(path, "JPEG", **save_kwargs)
    except Exception:
        return False

    try:
        if capture_date is not None:
            _set_file_datetime(path, capture_date)
    except Exception:
        pass
    return True


def _bgr_to_pillow(img: np.ndarray) -> Image.Image:
    """Convertit une image OpenCV en image Pillow."""
    if img.ndim == 2:
        return Image.fromarray(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _read_exif_bytes(source_path: str) -> Optional[bytes]:
    """Retourne le bloc EXIF source, si Pillow peut le lire."""
    try:
        with Image.open(source_path) as source:
            raw = source.info.get("exif")
            if raw:
                return raw
            exif = source.getexif()
            if exif:
                return exif.tobytes()
    except Exception:
        return None
    return None


def _read_capture_datetime(source_path: str) -> Optional[datetime]:
    """Lit la meilleure date EXIF disponible pour dater le fichier exporté."""
    try:
        with Image.open(source_path) as source:
            exif = source.getexif()
            if not exif:
                return None
            for date_tag, offset_tag in _DATE_TAGS:
                value = _exif_get(exif, date_tag)
                if not value:
                    continue
                dt = _parse_exif_datetime(value, _exif_get(exif, offset_tag))
                if dt is not None:
                    return dt
    except Exception:
        return None
    return None


def _exif_get(exif: Image.Exif, tag: int):
    """Cherche un tag dans l'IFD principal puis dans l'IFD EXIF."""
    value = exif.get(tag)
    if value is not None:
        return value
    try:
        return exif.get_ifd(_EXIF_IFD_TAG).get(tag)
    except Exception:
        return None


def _parse_exif_datetime(value, offset=None) -> Optional[datetime]:
    """Parse les dates EXIF classiques ``YYYY:MM:DD HH:MM:SS``."""
    text = _clean_exif_text(value)
    if not text:
        return None
    offset_text = _clean_exif_text(offset)

    candidates = [text]
    if offset_text:
        candidates.insert(0, f"{text}{offset_text}")
    for fmt in ("%Y:%m:%d %H:%M:%S%z", "%Y:%m:%d %H:%M:%S"):
        for candidate in candidates:
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _clean_exif_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip().strip("\x00")


def _set_file_datetime(path: str, dt: datetime) -> None:
    """Applique la date EXIF au fichier, et aussi à la création sous Windows."""
    timestamp = dt.timestamp() if dt.tzinfo is not None else time.mktime(dt.timetuple())
    os.utime(path, (timestamp, timestamp))
    if os.name == "nt":
        _set_windows_creation_time(path, timestamp)


def _set_windows_creation_time(path: str, timestamp: float) -> None:
    """Définit la date de création Windows quand l'API système est disponible."""
    windows_tick = int((timestamp + 11644473600) * 10000000)
    low = windows_tick & 0xFFFFFFFF
    high = windows_tick >> 32

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_uint32),
                    ("dwHighDateTime", ctypes.c_uint32)]

    creation_time = FILETIME(low, high)
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.SetFileTime.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(FILETIME),
        ctypes.c_void_p,
        ctypes.POINTER(FILETIME),
    )
    kernel32.SetFileTime.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int

    handle = kernel32.CreateFileW(
        path,
        0x0100,  # FILE_WRITE_ATTRIBUTES
        0x00000007,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
        None,
        3,  # OPEN_EXISTING
        0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        return
    try:
        kernel32.SetFileTime(
            handle,
            ctypes.byref(creation_time),
            None,
            ctypes.byref(creation_time),
        )
    finally:
        kernel32.CloseHandle(handle)
