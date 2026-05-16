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
    name               = "6 · Auto niveaux & couleurs"
    short_name         = "AutoColor"
    slow               = False
    enabled_by_default = False   # désactivée par défaut

    param_defs = [
        {"key": "wb_strength", "label": "Balance blancs (force)", "type": "float",
         "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "clip_lo",    "label": "Seuil noir (%)",          "type": "float",
         "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
        {"key": "clip_hi",    "label": "Seuil blanc (%)",         "type": "float",
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

def _shades_of_grey_gains(img: np.ndarray, p: float = 6.0) -> np.ndarray:
    """Gains de balance des blancs par l'algorithme Shades of Grey.

    Finlayson & Trezzi (2004) : la p-norme de chaque canal estime la couleur
    de l'illuminant. Pour p=6, l'estimateur est nettement plus précis que
    le Grey World (p=1) ou Max RGB (p→∞) sur des images naturelles variées.

    Retourne un tableau [B_gain, G_gain, R_gain] tel que multiplier l'image
    par ces gains la ramène vers une scène sous illuminant blanc.
    """
    img_f = img.astype(np.float32)
    norm  = np.mean(img_f ** p, axis=(0, 1)) ** (1.0 / p)  # [B_norm, G_norm, R_norm]
    target = norm.mean()
    return target / (norm + 1e-6)


def _auto_levels(img: np.ndarray, clip_lo: float, clip_hi: float) -> np.ndarray:
    """Étirement histogramme par canal : clip aux percentiles clip_lo / (100−clip_hi)."""
    result = np.empty_like(img, dtype=np.float32)
    for i in range(3):
        p_lo = float(np.percentile(img[:, :, i], clip_lo))
        p_hi = float(np.percentile(img[:, :, i], 100.0 - clip_hi))
        if p_hi > p_lo:
            result[:, :, i] = (img[:, :, i].astype(np.float32) - p_lo) * 255.0 / (p_hi - p_lo)
        else:
            result[:, :, i] = img[:, :, i].astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def _auto_color(
    img:        np.ndarray,
    wb_strength: float = 0.8,
    clip_lo:    float  = 0.5,
    clip_hi:    float  = 0.5,
    saturation: float  = 1.0,
    gamma:      float  = 1.0,
    warmth:     int    = 0,
) -> np.ndarray:
    """Applique la correction automatique complète : WB → niveaux → sat → gamma → chaleur."""
    out = img.copy()

    # ── 1. Balance des blancs (Shades of Grey, p=6) ───────────────────────────
    if wb_strength > 0.0:
        gains   = _shades_of_grey_gains(out)
        partial = 1.0 + (gains - 1.0) * wb_strength   # interpolation selon force
        out     = np.clip(
            out.astype(np.float32) * partial[np.newaxis, np.newaxis, :],
            0, 255,
        ).astype(np.uint8)

    # ── 2. Niveaux automatiques par canal ─────────────────────────────────────
    if clip_lo > 0.0 or clip_hi > 0.0:
        out = _auto_levels(out, clip_lo, clip_hi)

    # ── 3. Saturation (espace HSV) ────────────────────────────────────────────
    if saturation != 1.0:
        hsv            = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1]   = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out            = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 4. Gamma ──────────────────────────────────────────────────────────────
    if gamma != 1.0:
        lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255.0
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)

    # ── 5. Chaleur (décalage b* en espace LAB) ────────────────────────────────
    if warmth != 0:
        lab            = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.int16)
        lab[:, :, 2]   = np.clip(lab[:, :, 2] + warmth, 0, 255)
        out            = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return out
