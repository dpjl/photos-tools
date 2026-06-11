"""steps/step_redzone.py — Correction des zones rouges résiduelles (cast localisé).

Supprime les voiles rosés/magenta de détérioration argentique dans les zones
du masque, sans toucher la luminance ni la texture (cf. core/redzone.py).

Le masque est construit dans l'onglet « Zones rouges » du mode batch
(détection auto + vérification VLM + édition manuelle) ou peint à la main.
"""

from __future__ import annotations

import cv2
import numpy as np

from steps.base import StepBase


class RedZoneStep(StepBase):
    id                 = "redzone"
    name               = "Zones rouges (cast localisé)"
    short_name         = "Zones rouges"
    slow               = True            # rembg requis si protection personnes
    enabled_by_default = False
    has_mask_editor    = True            # bouton « Peindre le masque » dans l'UI

    param_defs = [
        {"key": "strength",  "label": "Force",            "type": "float",
         "default": 1.0,  "min": 0.0, "max": 1.0,  "step": 0.05},
        {"key": "luma",      "label": "Corr. luminance",  "type": "float",
         "default": 0.0,  "min": 0.0, "max": 1.0,  "step": 0.05},
        {"key": "dilation",  "label": "Dilatation",       "type": "int",
         "default": 60,   "min": 0,   "max": 120,  "step": 5},
        {"key": "feather",   "label": "Fondu σ",          "type": "int",
         "default": 20,   "min": 5,   "max": 60,   "step": 5},
        {"key": "protect",   "label": "Protéger personnes", "type": "bool",
         "default": True},
    ]

    def __init__(self) -> None:
        super().__init__()
        self._mask: np.ndarray | None = None   # uint8 H x W, 255 = zone rouge

    # ── API appelée depuis l'UI ──────────────────────────────────────────────

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Stocke le masque des zones rouges."""
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
        from core.redzone import remove_red_cast

        h, w = img.shape[:2]

        mask = self._mask
        if mask is None or not mask.any():
            return img.copy(), {"redzone_mask_pixels": 0}
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        person_alpha = None
        if bool(params.get("protect", True)):
            person_alpha = self._person_alpha(img)

        result = remove_red_cast(
            img, mask,
            person_alpha  = person_alpha,
            strength      = float(params.get("strength", 1.0)),
            luma_strength = float(params.get("luma", 0.0)),
            dilate_px     = int(params.get("dilation", 60)),
            feather_sigma = float(params.get("feather", 20)),
        )
        return result, {"redzone_mask_pixels": int((mask > 127).sum())}

    @staticmethod
    def _person_alpha(img: np.ndarray) -> np.ndarray:
        """Alpha 0–1 des personnes (U2-Net rembg, session partagée)."""
        from PIL import Image
        from rembg import remove
        from core.rembg_session import get_shared_session

        pil_in = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        alpha  = np.array(remove(pil_in, session=get_shared_session()))[:, :, 3]
        return alpha.astype(np.float32) / 255.0
