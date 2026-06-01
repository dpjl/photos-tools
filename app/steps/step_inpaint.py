"""steps/step_inpaint.py — Etape 4 : Retouche par inpainting (LaMa).

L'utilisateur peint un masque dans l'editeur integre.

Modele : LaMa (Large Mask inpainting), identique a lama-cleaner.
Strategie HD : crop avec marge contextuelle.
"""

from __future__ import annotations

import cv2
import numpy as np

from steps.base import StepBase


class InpaintStep(StepBase):
    id               = "inpaint"
    name             = "Retouche (inpainting)"
    short_name       = "Retouche"
    slow             = True
    enabled_by_default = False
    has_overlay      = False
    has_mask_editor  = True   # active le bouton « Peindre le masque » dans l'UI

    param_defs = []

    def __init__(self) -> None:
        super().__init__()
        self._mask: np.ndarray | None = None   # uint8 H x W, 255 = zone a retoucher

    # ── API appelee depuis l'UI ──────────────────────────────────────────────

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Stocke le masque peint manuellement."""
        self._mask = mask.copy() if mask is not None else None

    def get_mask(self) -> np.ndarray | None:
        return self._mask

    def clear_mask(self) -> None:
        self._mask = None

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def process(
        self,
        img:     np.ndarray,
        params:  dict,
        context: dict,
    ) -> tuple[np.ndarray, dict]:
        from core.lama_inpaint import LamaInpainter

        h, w = img.shape[:2]

        # Masque manuel (redimensionne si l'image a change depuis le dessin)
        if self._mask is not None:
            combined = self._mask
            if combined.shape[:2] != (h, w):
                combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            combined = np.zeros((h, w), dtype=np.uint8)

        if combined.max() == 0:
            return img.copy(), {"inpaint_mask_pixels": 0}

        result = LamaInpainter.get().inpaint(img, combined)
        return result, {"inpaint_mask_pixels": int((combined > 0).sum())}
