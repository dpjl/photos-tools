"""steps/step_rembg.py — Étape 4 : correction du cast argentique (Cast).

Utilise U2-Net (rembg) pour segmenter les personnes, puis corrige la dominante
rouge/jaune dans le fond via une zone de référence gauche et un champ de
diffusion spatiale gaussienne — algorithme identique à V1 « Cast argentique ».
"""

from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image

from steps.base import StepBase


class RembgStep(StepBase):
    id         = "rembg"
    name               = "Cast argentique"
    short_name         = "Cast"
    slow               = True
    enabled_by_default = False

    param_defs = [
        {"key": "strength_a", "label": "Force a* rouge", "type": "float",
         "default": 1.3, "min": 0.0, "max": 3.0, "step": 0.05},
        {"key": "strength_b", "label": "Force b* jaune", "type": "float",
         "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05},
        {"key": "sigma_fill", "label": "Diffusion σ",     "type": "int",
         "default": 120, "min": 20, "max": 300, "step": 10},
        {"key": "ref_x1",     "label": "Réf. début",     "type": "float",
         "default": 0.03, "min": 0.0, "max": 0.2, "step": 0.01},
        {"key": "ref_x2",     "label": "Réf. fin",        "type": "float",
         "default": 0.20, "min": 0.05, "max": 0.5, "step": 0.01},
    ]

    def __init__(self):
        self._session = None

    def _get_session(self):
        if self._session is None:
            from rembg import new_session
            self._session = new_session("u2net")
        return self._session

    def process(self, img: np.ndarray, params: dict, context: dict):
        result = _correct_color_rembg(
            img,
            session=self._get_session(),
            strength_a=float(params.get("strength_a", 1.3)),
            strength_b=float(params.get("strength_b", 0.4)),
            sigma_fill=int(params.get("sigma_fill", 120)),
            ref_x1=float(params.get("ref_x1", 0.03)),
            ref_x2=float(params.get("ref_x2", 0.20)),
        )
        return result, {}


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions de traitement internes
# ──────────────────────────────────────────────────────────────────────────────

_FG_BLUR_K     = 21  # kernel gaussien pour le lissage du masque premier plan
_FG_BLUR_SIGMA = 7   # sigma gaussien pour le lissage du masque premier plan


def _correct_color_rembg(
    img: np.ndarray,
    session,
    strength_a: float = 1.3,
    strength_b: float = 0.4,
    sigma_fill: int   = 120,
    ref_x1: float     = 0.03,
    ref_x2: float     = 0.20,
) -> np.ndarray:
    """Corrige la dominante rouge/jaune argentique (algorithme V1).

    1. Segmente les personnes (U2-Net, alpha matting) → masque fg
    2. Zone de référence gauche [ref_x1..ref_x2] : fond non dégradé
    3. a_ref = médiane a* dans la zone de référence (fond, lignes 10-90 %)
    4. b_ref_row[y] = médiane b* par ligne dans la zone → interpolé + lissé σ=20
    5. Champ de cast 2D : écart de chaque pixel de fond par rapport à la référence,
       diffusé sur toute l'image via GaussianBlur(sigma_fill)
    6. Correction unidirectionnelle : min(0, -cast)*strength — retire le cast
       sans sur-corriger (ne pousse pas dans l'autre sens)
    7. Fusion douce : restitue les personnes avec masque gaussien (k=21, σ=7)
    """
    from rembg import remove

    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # ── 1. Masque foreground (U2-Net) ─────────────────────────────────────────
    pil_in = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fg_pil = remove(
        pil_in, session=session, only_mask=True,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )
    fg_mask = np.array(fg_pil).astype(np.float32) / 255.0
    if fg_mask.ndim == 3:
        fg_mask = fg_mask[:, :, 0]
    bg_mask = fg_mask < 0.3   # True = fond / arrière-plan

    # ── 2. Référence : zone gauche (fond présumé non dégradé) ─────────────────
    rx1    = int(w * ref_x1)
    rx2    = max(int(w * ref_x2), rx1 + 4)
    row_lo = int(h * 0.10)
    row_hi = int(h * 0.90)

    ref_zone_bg = bg_mask[row_lo:row_hi, rx1:rx2]
    ref_a_vals  = a_ch[row_lo:row_hi, rx1:rx2]
    ref_b_vals  = b_ch[row_lo:row_hi, rx1:rx2]

    a_ref = float(np.median(ref_a_vals[ref_zone_bg])) if ref_zone_bg.any() else 128.0

    # b* : référence par ligne (médiane dans la zone de référence, fond seulement)
    b_ref_row = np.full(h, np.nan, dtype=np.float32)
    for y in range(h):
        left_bg = bg_mask[y, rx1:rx2]
        if left_bg.sum() >= 3:
            b_ref_row[y] = np.median(b_ch[y, rx1:rx2][left_bg])
    valid = ~np.isnan(b_ref_row)
    if valid.sum() > 2:
        b_ref_row = np.interp(np.arange(h), np.where(valid)[0], b_ref_row[valid])
        b_ref_row = gaussian_filter1d(b_ref_row, sigma=20)
    else:
        fallback  = float(np.median(ref_b_vals[ref_zone_bg])) if ref_zone_bg.any() else 128.0
        b_ref_row[:] = fallback

    # ── 3. Champ de cast 2D estimé par diffusion gaussienne ───────────────────
    bg_y, bg_x = np.where(bg_mask)
    a_dev = np.zeros((h, w), np.float32)
    b_dev = np.zeros((h, w), np.float32)
    va    = np.zeros((h, w), np.float32)
    vb    = np.zeros((h, w), np.float32)

    a_dev[bg_y, bg_x] = a_ch[bg_y, bg_x] - a_ref
    b_ref_2d = np.tile(b_ref_row[:, np.newaxis], (1, w))
    b_dev[bg_y, bg_x] = b_ch[bg_y, bg_x] - b_ref_2d[bg_y, bg_x]
    va[bg_y, bg_x] = 1.0
    vb[bg_y, bg_x] = 1.0

    eps    = 1e-6
    a_cast = (cv2.GaussianBlur(a_dev * va, (0, 0), sigma_fill) /
              (cv2.GaussianBlur(va, (0, 0), sigma_fill) + eps))
    b_cast = (cv2.GaussianBlur(b_dev * vb, (0, 0), sigma_fill) /
              (cv2.GaussianBlur(vb, (0, 0), sigma_fill) + eps))

    # Ne corriger que dans le sens de suppression du cast (pas de sur-correction)
    a_field = np.minimum(0.0, -a_cast) * strength_a
    b_field = np.minimum(0.0, -b_cast) * strength_b

    # ── 4. Application + fusion douce ─────────────────────────────────────────
    a_new = np.clip(a_ch + a_field, 0, 255).astype(np.uint8)
    b_new = np.clip(b_ch + b_field, 0, 255).astype(np.uint8)
    img_corrected = cv2.cvtColor(
        cv2.merge([np.clip(L_ch, 0, 255).astype(np.uint8), a_new, b_new]),
        cv2.COLOR_LAB2BGR,
    )

    # Masque gaussien pour une fusion douce aux contours des personnes
    fg_soft = cv2.GaussianBlur(fg_mask, (_FG_BLUR_K, _FG_BLUR_K), _FG_BLUR_SIGMA)
    fg_3d   = fg_soft[:, :, np.newaxis]
    result  = (img.astype(np.float32) * fg_3d +
               img_corrected.astype(np.float32) * (1.0 - fg_3d))
    return np.clip(result, 0, 255).astype(np.uint8)
