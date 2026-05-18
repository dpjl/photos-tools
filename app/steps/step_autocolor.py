"""steps/step_autocolor.py — Étape 5 : correction automatique de la couleur.

Combine plusieurs techniques complémentaires :

1. **Balance des blancs automatique** (Shades of Grey, Finlayson & Trezzi 2004)
   Estime l'illuminant par la p-norme (p=6) de chaque canal. Supérieur au
   « Grey World » classique ; robuste aux scènes peu uniformes.

2. **Niveaux automatiques par canal** (étirement percentile)
   Étire chaque canal de [p_lo, p_hi] → [0, 255] pour corriger l'exposition
   et supprimer les éventuelles dérives résiduelles canal par canal.

3. **Saturation** (espace HSV)

4. **Gamma**

5. **Chaleur** (décalage b* en espace LAB) — tons chauds/froids

Désactivée par défaut ; à activer si les étapes précédentes ne corrigent
pas suffisamment la dominante.
"""

from __future__ import annotations
import cv2
import numpy as np

from steps.base import StepBase


class AutoColorStep(StepBase):
    id                 = "autocolor"
    name               = "Auto niveaux & couleurs"
    short_name         = "AutoColor"
    slow               = False
    enabled_by_default = True    # activée par défaut

    param_defs = [
        {"key": "mode",       "label": "Mode",                     "type": "choice",
         "default": "naturel", "choices": ["naturel", "neutre", "actuel"]},
        {"key": "wb_strength", "label": "Force correction",        "type": "float",
         "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "clip_lo",    "label": "Seuil noir (%)",           "type": "float",
         "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
        {"key": "clip_hi",    "label": "Seuil blanc (%)",          "type": "float",
         "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
        {"key": "saturation", "label": "Saturation",               "type": "float",
         "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05},
        {"key": "gamma",      "label": "Gamma",                    "type": "float",
         "default": 1.0, "min": 0.5, "max": 2.5, "step": 0.05},
        {"key": "warmth",     "label": "Chaleur (b* ±30)",         "type": "int",
         "default": 0, "min": -30, "max": 30, "step": 1},
    ]

    def process(self, img: np.ndarray, params: dict, context: dict):
        return _auto_color(
            img,
            mode       =str(  params.get("mode",        "naturel")),
            wb_strength=float(params.get("wb_strength", 0.8)),
            clip_lo    =float(params.get("clip_lo",     0.5)),
            clip_hi    =float(params.get("clip_hi",     0.5)),
            saturation =float(params.get("saturation",  1.0)),
            gamma      =float(params.get("gamma",       1.0)),
            warmth     =int(  params.get("warmth",      0)),
        ), {}


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions de traitement internes
# ──────────────────────────────────────────────────────────────────────────────

_ROI_INSET = 0.06   # fraction de bord ignoree pour les calculs percentile ROI


def _roi_slice(h: int, w: int):
    """Renvoie (row_slice, col_slice) ignorant les bords de scan."""
    dy, dx = max(1, int(h * _ROI_INSET)), max(1, int(w * _ROI_INSET))
    return slice(dy, h - dy), slice(dx, w - dx)


def _shades_of_grey_gains(img, p: float = 6.0):
    """Gains WB par l'algorithme Shades of Grey (p=6)."""
    img_f = img.astype(np.float32)
    norm  = np.mean(img_f ** p, axis=(0, 1)) ** (1.0 / p)
    return norm.mean() / (norm + 1e-6)


def _auto_levels(img, clip_lo: float, clip_hi: float):
    """Etirement histogramme par canal (mode actuel, sans inset ROI)."""
    result = np.empty_like(img, dtype=np.float32)
    for i in range(3):
        p_lo = float(np.percentile(img[:, :, i], clip_lo))
        p_hi = float(np.percentile(img[:, :, i], 100.0 - clip_hi))
        if p_hi > p_lo:
            result[:, :, i] = (img[:, :, i].astype(np.float32) - p_lo) * 255.0 / (p_hi - p_lo)
        else:
            result[:, :, i] = img[:, :, i].astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def _correct_naturel(img, wb_strength: float, clip_lo: float, clip_hi: float,
                     max_stretch: float = 4.0):
    """Mode naturel : Approche D+ plafonnee.

    1. Retire le voile de plancher par canal (percentile clip_lo en ROI bordee).
    2. Blanc commun = percentile (100-clip_hi) de la luminance moyenne pixel.
       Si la plage utile d'un canal est trop etroite (facteur > max_stretch),
       bascule sur l'etirement per-canal standard pour ce canal uniquement.
    3. wb_strength pondule la correction (0 = image inchangee).
    """
    h, w = img.shape[:2]
    rs, cs = _roi_slice(h, w)
    roi = img[rs, cs]
    img_f = img.astype(np.float32)
    p_lo_pct, p_hi_pct = clip_lo, 100.0 - clip_hi
    black = np.array([float(np.percentile(roi[:, :, i], p_lo_pct)) for i in range(3)])
    white_lum = float(np.percentile(roi.astype(np.float32).mean(axis=2), p_hi_pct))
    result = np.empty_like(img_f)
    for i in range(3):
        range_D = white_lum - black[i]
        if range_D >= 255.0 / max_stretch:
            result[:, :, i] = (img_f[:, :, i] - black[i]) * 255.0 / range_D
        else:
            p_hi_c = float(np.percentile(roi[:, :, i], p_hi_pct))
            r_std  = p_hi_c - black[i]
            if r_std > 0:
                result[:, :, i] = (img_f[:, :, i] - black[i]) * 255.0 / r_std
            else:
                result[:, :, i] = img_f[:, :, i]
    corrected = np.clip(result, 0, 255).astype(np.uint8)
    if wb_strength < 1.0:
        blend = img_f * (1.0 - wb_strength) + corrected.astype(np.float32) * wb_strength
        return np.clip(blend, 0, 255).astype(np.uint8)
    return corrected


def _correct_neutre(img, wb_strength: float, clip_lo: float, clip_hi: float):
    """Mode neutre : etirement ROI par canal, puis Grey World post.

    1. Etire chaque canal independamment sur son percentile ROI bordee.
    2. Applique Grey World APRES l'etirement.
       wb_strength controle la force du Grey World (0 = etirement seul).
    """
    h, w = img.shape[:2]
    rs, cs = _roi_slice(h, w)
    roi = img[rs, cs]
    img_f = img.astype(np.float32)
    p_lo_pct, p_hi_pct = clip_lo, 100.0 - clip_hi
    stretched = np.empty_like(img_f)
    for i in range(3):
        p_lo = float(np.percentile(roi[:, :, i], p_lo_pct))
        p_hi = float(np.percentile(roi[:, :, i], p_hi_pct))
        if p_hi > p_lo:
            stretched[:, :, i] = (img_f[:, :, i] - p_lo) * 255.0 / (p_hi - p_lo)
        else:
            stretched[:, :, i] = img_f[:, :, i]
    stretched = np.clip(stretched, 0, 255)
    if wb_strength > 0.0:
        means   = stretched.mean(axis=(0, 1))
        target  = means.mean()
        gains   = target / (means + 1e-6)
        partial = 1.0 + (gains - 1.0) * wb_strength
        stretched = np.clip(stretched * partial[np.newaxis, np.newaxis, :], 0, 255)
    return stretched.astype(np.uint8)


def _auto_color(img, mode: str = "naturel", wb_strength: float = 0.8,
                clip_lo: float = 0.5, clip_hi: float = 0.5,
                saturation: float = 1.0, gamma: float = 1.0, warmth: int = 0):
    """Applique la correction automatique complete selon le mode choisi."""
    out = img.copy()

    # 1. Correction couleur selon mode
    if mode == "naturel":
        out = _correct_naturel(out, wb_strength, clip_lo, clip_hi)
    elif mode == "neutre":
        out = _correct_neutre(out, wb_strength, clip_lo, clip_hi)
    else:   # "actuel" - comportement historique
        if wb_strength > 0.0:
            gains   = _shades_of_grey_gains(out)
            partial = 1.0 + (gains - 1.0) * wb_strength
            out = np.clip(
                out.astype(np.float32) * partial[np.newaxis, np.newaxis, :], 0, 255,
            ).astype(np.uint8)
        if clip_lo > 0.0 or clip_hi > 0.0:
            out = _auto_levels(out, clip_lo, clip_hi)

    # 2. Saturation (espace HSV)
    if saturation != 1.0:
        hsv          = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out          = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Gamma
    if gamma != 1.0:
        lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255.0
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)

    # 4. Chaleur (decalage b* en espace LAB)
    if warmth != 0:
        lab          = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.int16)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + warmth, 0, 255)
        out          = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return out
