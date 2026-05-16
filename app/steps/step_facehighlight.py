"""steps/step_facehighlight.py — Etape 3 - Correction hautes lumieres.

Port de l'algorithme Shadows/Highlights de RawTherapee (ipshadowshighlights.cc,
Alberto Griggio, 2018) en mode Lab=false (espace RGB).

Principe :
  1. Construire un masque spatial base sur L* : les pixels brillants (L* > seuil
     hl_tonal_width) obtiennent mask=1, les pixels plus sombres une rampe en ^4.
  2. Lisser le masque par un guided filter (edge-aware) pour eviter les halos.
  3. Construire une LUT gamma : pour chaque valeur sRGB c -> applique
     pow(L*(c)/100, gamma) en espace L*, puis reconvertit en sRGB.
     gamma = 4^(highlights * 0.7 / 100) -- meme formule que RT.
  4. Pour chaque pixel : corrected_ch = mask * LUT[ch] + (1-mask) * ch
     (correction par canal RGB -> preserve la saturation comme RT Lab=false).

Parametres exposes (correspondent exactement aux cles du pp3 RawTherapee) :
  highlights      0-100   intensite de la correction des hautes lumieres
  hl_tonal_width  1-100   etendue tonale (L* en dessous duquel la correction s'estompe)
  shadows         0-100   intensite de l'eclaircissement des ombres
  sh_tonal_width  1-100   etendue tonale des ombres
  radius          1-100   rayon du guided filter (x10 pixels, comme dans RT)
  auto_detect     bool    calculer highlights et hl_tonal_width automatiquement
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from steps.base import StepBase

_MIN_REGION_AREA = 400


class FaceHighlightStep(StepBase):
    id = "facehighlight"
    name = "3 - Correction hautes lumieres"
    short_name = "HautesLum"
    slow = True
    enabled_by_default = False
    has_overlay = True

    param_defs = [
        {
            "key": "auto_detect",
            "label": "Detection automatique",
            "type": "bool",
            "default": True,
        },
        {
            "key": "highlights",
            "label": "Hautes lumieres (0-100)",
            "type": "int",
            "default": 100,
            "min": 0,
            "max": 100,
            "step": 1,
        },
        {
            "key": "hl_tonal_width",
            "label": "Etendue tonale HL (1-100)",
            "type": "int",
            "default": 58,
            "min": 1,
            "max": 100,
            "step": 1,
        },
        {
            "key": "shadows",
            "label": "Ombres (0-100)",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 100,
            "step": 1,
        },
        {
            "key": "sh_tonal_width",
            "label": "Etendue tonale ombres (1-100)",
            "type": "int",
            "default": 30,
            "min": 1,
            "max": 100,
            "step": 1,
        },
        {
            "key": "radius",
            "label": "Rayon filtre (1-100)",
            "type": "int",
            "default": 40,
            "min": 1,
            "max": 100,
            "step": 1,
        },
    ]

    def process(self, img: np.ndarray, params: dict, context: dict) -> tuple[np.ndarray, dict]:
        d = self.default_params()
        auto_detect  = bool(params.get("auto_detect",   d["auto_detect"]))
        highlights   = int(params.get("highlights",     d["highlights"]))
        hl_tonal     = int(params.get("hl_tonal_width", d["hl_tonal_width"]))
        shadows      = int(params.get("shadows",        d["shadows"]))
        sh_tonal     = int(params.get("sh_tonal_width", d["sh_tonal_width"]))
        radius       = int(params.get("radius",         d["radius"]))

        if auto_detect:
            highlights, hl_tonal = _auto_detect_params(img)

        if highlights == 0 and shadows == 0:
            return img.copy(), {"highlight_detections": []}

        result = _shadows_highlights(img, highlights, hl_tonal, shadows, sh_tonal, radius)
        detections = _build_region_detections(img, hl_tonal)
        return result, {"highlight_detections": detections}


# -- Detection automatique des parametres ------------------------------------

def _auto_detect_params(img: np.ndarray) -> tuple[int, int]:
    """Calcule highlights et hl_tonal_width selon la distribution tonale de l image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0 * 100.0

    pct_90 = float((L > 90).mean())
    pct_85 = float((L > 85).mean())
    pct_75 = float((L > 75).mean())

    # Severite -> intensite de correction
    severity = float(np.clip(pct_90 * 10.0 + pct_85 * 5.0 + pct_75 * 2.0, 0.0, 1.0))
    highlights = int(np.clip(severity * 110.0, 0, 100))

    # Etendue tonale : 40e percentile depuis le blanc donne la limite basse
    if highlights > 0:
        p60 = float(np.percentile(L, 60))   # 40 % des pixels sont au-dessus de cette valeur
        hl_tonal = int(np.clip(p60, 25, 80))
    else:
        hl_tonal = 58

    return highlights, hl_tonal


# -- Algorithme Shadows/Highlights -------------------------------------------

def _shadows_highlights(
    img: np.ndarray,
    highlights: int,
    hl_tonal: int,
    shadows: int,
    sh_tonal: int,
    radius: int,
) -> np.ndarray:
    """Port de ipshadowshighlights.cc -- RawTherapee, mode Lab=false."""
    # Travailler en float32 [0, 1]
    img_f = img.astype(np.float32) / 255.0

    # L* (0-100) pour le masque -- calcule une seule fois
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0 * 100.0

    if highlights > 0:
        img_f = _apply_sh(img_f, L, highlights, hl_tonal, radius, is_hl=True)
    if shadows > 0:
        img_f = _apply_sh(img_f, L, shadows, sh_tonal, radius, is_hl=False)

    return np.clip(img_f * 255.0, 0.0, 255.0).astype(np.uint8)


def _apply_sh(
    img_f: np.ndarray,
    L: np.ndarray,
    amount: int,
    tonal_width: int,
    radius: int,
    is_hl: bool,
) -> np.ndarray:
    """Une passe (hautes lumieres ou ombres) -- porte directe de apply() dans RT."""

    # -- Masque spatial -------------------------------------------------------
    # thresh correspond a tonal_width en L* 0-100
    thresh = float(tonal_width)
    scale  = 0.9 / thresh if thresh > 0 else 1.0

    if is_hl:
        # Pleinement corrige au-dessus du seuil ; rampe ^4 en dessous
        raw_mask = np.where(
            L > thresh,
            1.0,
            np.power(np.clip(L * scale, 0.0, 1.0), 4),
        )
        # Image de guidage : luminosite inversee
        guide = (1.0 - L / 100.0).astype(np.float32)
    else:
        raw_mask = np.where(
            L <= thresh,
            1.0,
            np.power(np.clip(scale / np.maximum(L, 1e-6), 0.0, 1.0), 4),
        )
        guide = (L / 100.0).astype(np.float32)

    raw_mask = raw_mask.astype(np.float32)

    # -- Guided filter (edge-aware) ------------------------------------------
    # RT : rayon en pixels = rad * 10 / scal ; scal=1 -> r_px = radius * 10
    r_px = max(1, radius * 10)
    mask = _guided_filter(guide, raw_mask, r_px, eps=0.075)
    mask = np.clip(mask, 0.0, 1.0)

    # -- LUT gamma -----------------------------------------------------------
    # RT : amount_eff = amount * 0.7 pour HL, * 0.6 pour ombres
    eff = amount * (0.7 if is_hl else 0.6)
    gamma = np.power(4.0, eff / 100.0)   # >1 assombrit (HL)
    if not is_hl:
        gamma = 1.0 / gamma              # ombres : eclaircissement

    lut = _build_lut(gamma)              # LUT float32 [256]

    # -- Application par canal RGB -------------------------------------------
    # Equivalent a : new_c = intp(blend, f[c], c) de RT
    result = img_f.copy()
    for ch in range(3):
        c   = img_f[:, :, ch]
        idx = np.clip((c * 255.0).astype(np.int32), 0, 255)
        f_c = lut[idx]
        result[:, :, ch] = mask * f_c + (1.0 - mask) * c

    return result


# -- LUT gamma dans l espace L* -----------------------------------------------

def _build_lut(gamma: float) -> np.ndarray:
    """Construit une LUT [256] float32 : valeur sRGB -> valeur sRGB corrigee.

    Identique au LUT RGB de RT (mode Lab=false) :
      sRGB c -> lineaire -> L* -> pow(L*/100, gamma) -> lineaire -> sRGB.
    """
    c = np.arange(256, dtype=np.float64) / 255.0   # sRGB [0, 1]

    # sRGB -> lineaire
    lin = np.where(
        c > 0.04045,
        np.power((c + 0.055) / 1.055, 2.4),
        c / 12.92,
    )

    # Lineaire Y -> L*  (gris neutre : Y = lin)
    L_star = np.where(
        lin > 0.008856,
        116.0 * np.cbrt(lin) - 16.0,
        903.3 * lin,
    )

    # Gamma en espace L*
    L_new = np.power(np.clip(L_star / 100.0, 0.0, 1.0), gamma) * 100.0

    # L* -> lineaire Y
    lin_new = np.where(
        L_new > 8.0,
        np.power((L_new + 16.0) / 116.0, 3.0),
        L_new / 903.3,
    )

    # Lineaire -> sRGB
    c_new = np.where(
        lin_new > 0.0031308,
        1.055 * np.power(np.maximum(lin_new, 0.0), 1.0 / 2.4) - 0.055,
        12.92 * lin_new,
    )

    return np.clip(c_new, 0.0, 1.0).astype(np.float32)


# -- Guided filter (implementation box-filter O(n)) --------------------------

def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Guided filter bord-preservant via box filters (He et al. 2010).

    guide, src : float32 2D [0, 1].
    radius     : rayon en pixels.
    eps        : regularisation (0.075 dans RT).
    """

    def box(x: np.ndarray) -> np.ndarray:
        return uniform_filter(x.astype(np.float64), size=2 * radius + 1, mode="reflect")

    I = guide.astype(np.float64)
    p = src.astype(np.float64)

    mean_I  = box(I)
    mean_p  = box(p)
    corr_I  = box(I * I)
    corr_Ip = box(I * p)

    var_I  = corr_I  - mean_I  * mean_I
    cov_Ip = corr_Ip - mean_I  * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a)
    mean_b = box(b)

    return (mean_a * I + mean_b).astype(np.float32)


# -- Detections pour l overlay UI -------------------------------------------

def _build_region_detections(img: np.ndarray, hl_tonal: int) -> list[dict]:
    """Retourne les regions surexposees pour l overlay rectangulaire."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0 * 100.0

    thresh_det = float(hl_tonal) * 0.85
    binary = (L > thresh_det).astype(np.uint8) * 255

    ks = max(3, min(img.shape[:2]) // 220)
    if ks % 2 == 0:
        ks += 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern)

    detections: list[dict] = []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < _MIN_REGION_AREA:
            continue
        x1 = int(stats[label, cv2.CC_STAT_LEFT])
        y1 = int(stats[label, cv2.CC_STAT_TOP])
        x2 = x1 + int(stats[label, cv2.CC_STAT_WIDTH])
        y2 = y1 + int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels == label
        detections.append({
            "type":      "region",
            "bbox":      (x1, y1, x2, y2),
            "area":      area,
            "overexp":   float((L[component] > 75).mean()),
            "corrected": True,
        })
    return detections

