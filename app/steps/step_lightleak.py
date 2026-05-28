"""steps/step_lightleak.py — Correction des fuites de lumière argentique.

Algorithme : Color Transfer LAB (Reinhard et al., 2001), appliqué localement
sur la zone affectée par la fuite de lumière.

Modèle physique
---------------
Une fuite de lumière argentique est une composante additive colorée (rose/rouge),
spatialement lisse et localisée, superposée à l'image originale :

    I_captée = I_scène + I_fuite

Principe de correction
----------------------
1. Détecter la zone contaminée dans le fond (teinte chaude HSV, saturation élevée).
2. Détecter la zone saine (fond hors zone contaminée, hors personnes).
3. Calculer les statistiques (moyenne + écart-type) LAB de chaque zone.
4. Normaliser la zone contaminée pour que sa distribution couleur corresponde
   à celle de la zone saine, canal par canal :

       pixel_corr = (pixel − μ_cont) / σ_cont × σ_clean + μ_clean

   → Centre-réduit le pixel par les stats contaminées, puis le redistribue
     selon les stats saines : supprime le décalage moyen (dominante rose)
     ET normalise l'étalement (contraste compressé par la surexposition).

5. Appliquer avec un masque gaussien doux (feathering) pour éviter tout bord.
6. Restituer les personnes à partir de l'image originale (masque U2-Net rembg).
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

from steps.base import StepBase


class LightLeakStep(StepBase):
    id                 = "lightleak"
    name               = "Lumière parasite"
    short_name         = "LightLeak"
    slow               = True            # rembg requis → lent
    enabled_by_default = False           # désactivée par défaut

    param_defs = [
        {"key": "strength",  "label": "Force",             "type": "float",
         "default": 1.0,  "min": 0.0,  "max": 1.0,  "step": 0.05},
        {"key": "hue_range", "label": "Amplitude teinte",  "type": "int",
         "default": 25,   "min": 5,    "max": 50,   "step": 1},
        {"key": "sat_min",   "label": "Sat. min.",          "type": "float",
         "default": 0.38, "min": 0.10, "max": 0.70, "step": 0.02},
        {"key": "dilation",  "label": "Dilatation",         "type": "int",
         "default": 30,   "min": 0,    "max": 80,   "step": 5},
        {"key": "feather",   "label": "Feathering σ",        "type": "int",
         "default": 30,   "min": 5,    "max": 80,   "step": 5},
        {"key": "bg_only",   "label": "Fond seulement",     "type": "bool",
         "default": True},
    ]

    def __init__(self):
        self._session = None

    def _get_session(self):
        if self._session is None:
            from rembg import new_session
            self._session = new_session("u2net")
        return self._session

    def process(self, img: np.ndarray, params: dict, context: dict):
        strength  = float(params.get("strength",  1.0))
        hue_range = int(  params.get("hue_range", 25))
        sat_min   = float(params.get("sat_min",   0.38))
        dilation  = int(  params.get("dilation",  30))
        feather   = int(  params.get("feather",   30))
        bg_only   = bool( params.get("bg_only",   True))

        session = self._get_session() if bg_only else None

        result = _correct_light_leak(
            img,
            strength    = strength,
            hue_range   = hue_range,
            sat_min     = sat_min,
            dilation_px = dilation,
            feather_sigma = feather,
            session     = session,
        )
        return result, {}


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions internes
# ─────────────────────────────────────────────────────────────────────────────

def _get_person_mask(img_bgr: np.ndarray, session) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (soft_alpha_f32, excl_mask_u8).

    soft_alpha  : canal alpha U2-Net normalisé 0.0–1.0, sans post-traitement.
                  Utilisé pour le mélange final : bords naturellement doux,
                  sans halo visible même sur des sujets fins (bras de bébé, etc.).
    excl_mask   : masque binaire avec dilation minimale (5 px), utilisé
                  uniquement pour exclure les personnes de la détection
                  des zones contaminées / saines. La petite dilation garantit
                  que les pixels de bord légèrement ambigués ne tombent pas
                  dans la zone « saine ».
    """
    from rembg import remove
    pil_in = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    alpha  = np.array(remove(pil_in, session=session))[:, :, 3]

    # Alpha doux pour le mélange final — U2-Net produit déjà des bords antialiasis
    soft_alpha = alpha.astype(np.float32) / 255.0

    # Masque d’exclusion légèrement dilaté pour la détection des zones seulement
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _, excl_mask = cv2.threshold(cv2.dilate(alpha, k), 127, 255, cv2.THRESH_BINARY)

    return soft_alpha, excl_mask


def _detect_zones(
    img_bgr:     np.ndarray,
    person_mask: np.ndarray | None,
    hue_range:   int,
    sat_min:     float,
    dilation_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Détecte les zones contaminée (rose/rouge) et saine dans le fond.

    Retourne (contaminated_mask, clean_mask) en uint8 (0 / 255).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[:, :, 0]          # 0–179 (OpenCV)
    S = hsv[:, :, 1] / 255.0
    V = hsv[:, :, 2] / 255.0

    # Teinte rouge/rose : basse (0–hue_range) OU haute miroir (179-hue_range–179)
    red_hue = (H <= hue_range) | (H >= (179 - hue_range))
    leak    = (red_hue & (S >= sat_min) & (V >= 0.38)).astype(np.uint8)

    if person_mask is not None:
        leak[person_mask > 127] = 0

    # Nettoyage morphologique : combler les trous dans les blobs
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    contaminated = cv2.morphologyEx(leak * 255, cv2.MORPH_CLOSE, k_close)

    # Dilatation : couvrir les bords dégradés du spot
    if dilation_px > 0:
        k_dilat = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px, dilation_px)
        )
        contaminated = cv2.dilate(contaminated, k_dilat)

    # Zone saine : fond hors zone contaminée et hors personnes
    clean = np.ones(img_bgr.shape[:2], np.uint8) * 255
    if person_mask is not None:
        clean[person_mask > 127] = 0
    clean[contaminated > 127] = 0

    return contaminated.astype(np.uint8), clean.astype(np.uint8)


def _color_transfer_lab(
    img_bgr:          np.ndarray,
    contaminated_mask: np.ndarray,
    clean_mask:       np.ndarray,
    strength:         float,
    feather_sigma:    float,
) -> np.ndarray:
    """Color Transfer LAB : normalise la distribution couleur de la zone contaminée
    vers celle de la zone saine.

    Pour chaque canal LAB :
        pixel_corr = (pixel − μ_cont) / σ_cont × σ_clean + μ_clean
    """
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    clean_px = img_lab[clean_mask > 127]
    cont_px  = img_lab[contaminated_mask > 127]

    # Garde-fous : pas assez de pixels → pas de correction
    if len(clean_px) < 100 or len(cont_px) < 100:
        return img_bgr

    clean_mean = clean_px.mean(axis=0)          # (3,)
    clean_std  = clean_px.std(axis=0)  + 1e-6
    cont_mean  = cont_px.mean(axis=0)
    cont_std   = cont_px.std(axis=0)   + 1e-6

    # Transformation centre-réduite + redistribution vers stats saines
    corrected_lab = (img_lab - cont_mean) / cont_std * clean_std + clean_mean
    corrected_lab = np.clip(corrected_lab, 0, 255)
    corrected_bgr = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Masque de mélange gaussien (feathering) × strength
    mask_f = contaminated_mask.astype(np.float32) / 255.0
    ksize  = int(6 * feather_sigma) | 1
    mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), float(feather_sigma))
    mask_f = np.clip(mask_f * strength, 0.0, 1.0)[:, :, np.newaxis]

    blended = (img_bgr.astype(np.float32) * (1.0 - mask_f) +
               corrected_bgr.astype(np.float32) * mask_f)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _correct_light_leak(
    img_bgr:      np.ndarray,
    strength:     float = 1.0,
    hue_range:    int   = 25,
    sat_min:      float = 0.38,
    dilation_px:  int   = 30,
    feather_sigma: float = 30.0,
    session              = None,
) -> np.ndarray:
    """Pipeline complet de correction lumière parasite."""
    soft_alpha: np.ndarray | None = None
    excl_mask:  np.ndarray | None = None
    if session is not None:
        soft_alpha, excl_mask = _get_person_mask(img_bgr, session)

    contaminated, clean = _detect_zones(
        img_bgr, excl_mask, hue_range, sat_min, dilation_px
    )

    # Pas de zone détectée → image inchangée
    n_cont  = int((contaminated > 127).sum())
    n_clean = int((clean > 127).sum())
    if n_cont < 500 or n_clean < 500:
        return img_bgr

    result = _color_transfer_lab(
        img_bgr, contaminated, clean, strength, feather_sigma
    )

    # Restituer les personnes depuis l’original avec l’alpha doux U2-Net
    # (bords anti-aliasés natifs — pas de halo visible)
    if soft_alpha is not None:
        alpha_f = soft_alpha[:, :, np.newaxis]
        result = (img_bgr.astype(np.float32) * alpha_f +
                  result.astype(np.float32) * (1.0 - alpha_f))
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result
