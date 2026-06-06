"""steps/step_crop.py — Recadrage manuel.

Le rectangle est stocke en coordonnees normalisees (x0, y0, x1, y1), ce qui le
rend reutilisable meme si une etape precedente change la taille de l'image.
"""

from __future__ import annotations

import numpy as np

from steps.base import StepBase


class CropStep(StepBase):
    id         = "crop"
    name       = "Recadrage manuel"
    short_name = "Crop"
    slow       = False
    enabled_by_default = False
    has_crop_editor = True
    previewable = True

    param_defs = []

    def __init__(self) -> None:
        super().__init__()
        self._rect: tuple[float, float, float, float] | None = None

    def set_crop_rect(
        self,
        rect: tuple[float, float, float, float] | None,
    ) -> None:
        self._rect = _normalize_rect(rect)

    def get_crop_rect(self) -> tuple[float, float, float, float] | None:
        return self._rect

    def clear_crop_rect(self) -> None:
        self._rect = None

    def process(self, img: np.ndarray, params: dict, context: dict):
        rect = _normalize_rect(self._rect)
        if rect is None:
            return img.copy(), {"crop_rect": None}
        h, w = img.shape[:2]
        x0 = int(round(rect[0] * w))
        y0 = int(round(rect[1] * h))
        x1 = int(round(rect[2] * w))
        y1 = int(round(rect[3] * h))
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        return img[y0:y1, x0:x1].copy(), {
            "crop_rect": rect,
            "crop_pixels": (x0, y0, x1, y1),
        }


def _normalize_rect(
    rect: tuple[float, float, float, float] | list | None,
) -> tuple[float, float, float, float] | None:
    if rect is None or len(rect) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in rect]
    left, right = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    top, bottom = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if right - left < 0.002 or bottom - top < 0.002:
        return None
    return (left, top, right, bottom)
