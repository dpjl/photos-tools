"""steps/step_inpaint.py — Etape 4 : Retouche par inpainting (LaMa).

Deux modes :
  - Manuel  : l'utilisateur peint un masque dans l'editeur integre.
  - Auto    : detection automatique des artefacts de scan (poussiere, rayures).
  - Mixte   : superposition des deux masques.

Modele : LaMa (Large Mask inpainting), identique a lama-cleaner.
Strategie HD : crop avec marge contextuelle.
"""

from __future__ import annotations

import cv2
import numpy as np

from steps.base import StepBase


class InpaintStep(StepBase):
    id               = "inpaint"
    name             = "4 · Retouche (inpainting)"
    short_name       = "Retouche"
    slow             = True
    enabled_by_default = False
    has_overlay      = False
    has_mask_editor  = True   # active le bouton « Peindre le masque » dans l'UI

    param_defs = [
        {
            "key":     "auto_detect",
            "label":   "Détection auto des artefacts",
            "type":    "bool",
            "default": False,
        },
        {
            "key":     "sensitivity",
            "label":   "Sensibilité détection (0-100)",
            "type":    "int",
            "default": 50,
            "min":     0,
            "max":     100,
            "step":    5,
        },
    ]

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

        d           = self.default_params()
        auto_detect = bool(params.get("auto_detect", d["auto_detect"]))
        sensitivity = int(params.get("sensitivity",  d["sensitivity"]))

        h, w = img.shape[:2]

        # Masque automatique
        if auto_detect:
            auto_mask = _detect_artifacts(img, sensitivity)
        else:
            auto_mask = np.zeros((h, w), dtype=np.uint8)

        # Masque manuel (redimensionne si l'image a change depuis le dessin)
        if self._mask is not None:
            manual = self._mask
            if manual.shape[:2] != (h, w):
                manual = cv2.resize(manual, (w, h), interpolation=cv2.INTER_NEAREST)
            combined = np.maximum(manual, auto_mask)
        else:
            combined = auto_mask

        if combined.max() == 0:
            return img.copy(), {"inpaint_mask_pixels": 0}

        result = LamaInpainter.get().inpaint(img, combined)
        return result, {"inpaint_mask_pixels": int((combined > 0).sum())}


# ── Detection automatique d'artefacts ─────────────────────────────────────────

def _detect_artifacts(img: np.ndarray, sensitivity: int) -> np.ndarray:
    """Detecte les petits artefacts de scan (poussiere, marques locales).

    sensitivity 0-100 :
      0  = tres conservateur (seulement les artefacts tres marques)
      100 = agressif (capture plus d'elements, risque de faux positifs)

    Retourne un masque uint8 H x W (255 = artefact probable).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # Fond local estime par ouverture morphologique avec un grand element structurant.
    # L'ouverture supprime les petites excroissances lumineuses (poussiere claire)
    # tout en preservant les grandes structures de la scene.
    k_size = max(15, min(h, w) // 50) | 1   # impair, ~1-2% du plus petit cote
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel).astype(np.float32)

    # Difference absolue : regions tres differentes du fond local
    diff = np.abs(gray - background).astype(np.uint8)

    # Seuil inversement proportionnel a la sensibilite
    # sensitivity=0  -> thresh=40  (seulement les artefacts tres contrasted)
    # sensitivity=50 -> thresh=22
    # sensitivity=100 -> thresh=5
    thresh = max(5, int(40 - sensitivity * 0.35))
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # Filtrage par taille : supprimer les composantes trop grandes
    # (elles correspondent a du contenu reel, pas a de la poussiere)
    max_area = max(50, int(h * w * 0.001))   # au plus 0.1% de l'image
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered = np.zeros_like(mask)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] <= max_area:
            filtered[labels == lbl] = 255

    # Dilatation pour couvrir les bords des artefacts
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(filtered, k_dil, iterations=1)
