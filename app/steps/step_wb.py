"""steps/step_wb.py — Balance des blancs par pipette (style RawTherapee).

Algorithme (identique a RT pour les images non-RAW / deja traitees) :
  1. L'utilisateur selectionne un point de reference sur une zone neutre
     (blanc, gris) de l'image.
  2. Au traitement, on echantillonne un carre de cote (2*patch_radius+1) autour
     de ce point dans l'IMAGE COURANTE (sortie de l'etape precedente, pas l'image
     originale). La position en pixels est sauvegardee ; les couleurs sont lues
     a la volee.
  3. Conversion sRGB -> RGB lineaire (desgammaisation correcte).
  4. Moyennes R, G, B du patch en espace lineaire.
  5. Gris cible : target = (R_moy + G_moy + B_moy) / 3
     (conservation de la luminance, pas seulement le canal vert)
  6. Multiplicateurs : mul_R = target/R_moy, mul_G = target/G_moy, mul_B = target/B_moy
     Bornes : [0.10, 10.0] pour eviter les corrections absurdes.
  7. Application a toute l'image en espace lineaire ; re-gammaisation finale.

Si le patch est trop sombre (moy < 0.02 en lineaire), la correction est ignoree
(le pixel blanc n'est pas blanc, c'est probablement une erreur de selection).
"""

from __future__ import annotations

import cv2
import numpy as np

from steps.base import StepBase


# ── Conversion sRGB ↔ lineaire ─────────────────────────────────────────────────

def _srgb_to_linear(img_uint8: np.ndarray) -> np.ndarray:
    """uint8 HWC sRGB -> float32 HWC lineaire [0, 1]."""
    f = img_uint8.astype(np.float32) / 255.0
    return np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_uint8(linear: np.ndarray) -> np.ndarray:
    """float32 HWC lineaire [0, 1] -> uint8 HWC sRGB."""
    out = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


# ── Algorithme principal ────────────────────────────────────────────────────────

def apply_white_balance(
    img_bgr:      np.ndarray,
    pick_x:       int,
    pick_y:       int,
    patch_radius: int = 5,
) -> np.ndarray:
    """Corrige la balance des blancs d'apres le point de reference.

    img_bgr      : H x W x 3  uint8  BGR
    pick_x/y     : coordonnees du point de reference (zone neutre/blanche)
                   en pixels dans img_bgr
    patch_radius : demi-cote du carre d'echantillonnage (en pixels image)

    Retourne : H x W x 3  uint8  BGR
    """
    h, w = img_bgr.shape[:2]
    r  = int(patch_radius)
    px = int(np.clip(pick_x,  r, w - r - 1))
    py = int(np.clip(pick_y,  r, h - r - 1))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    linear  = _srgb_to_linear(img_rgb)

    # Echantillonnage du patch
    patch = linear[py - r : py + r + 1, px - r : px + r + 1, :]
    means = patch.mean(axis=(0, 1))          # [R_moy, G_moy, B_moy] lineaires

    # Securite : patch trop sombre → probablement mauvaise selection
    if means.min() < 0.02:
        return img_bgr.copy()

    # Gris cible = moyenne des 3 canaux (conserve la luminance percue)
    target = float(means.mean())
    muls   = np.clip(target / means, 0.10, 10.0).astype(np.float32)   # [mR, mG, mB]

    # Application a l'image entiere
    out_linear = np.clip(linear * muls[np.newaxis, np.newaxis, :], 0.0, 1.0)
    out_rgb    = _linear_to_srgb_uint8(out_linear)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def sample_patch_info(
    img_bgr:      np.ndarray,
    pick_x:       int,
    pick_y:       int,
    patch_radius: int = 5,
) -> dict:
    """Retourne les infos du patch echantillonne pour affichage dans l'UI.

    Retourne un dict avec :
      rgb_means  : (R, G, B) en sRGB [0-255], moyennes du patch
      muls       : (mul_R, mul_G, mul_B) multiplicateurs calcules
      too_dark   : True si le patch est trop sombre pour corriger
    """
    h, w = img_bgr.shape[:2]
    r  = int(patch_radius)
    px = int(np.clip(pick_x,  r, w - r - 1))
    py = int(np.clip(pick_y,  r, h - r - 1))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    patch_srgb = img_rgb[py - r : py + r + 1, px - r : px + r + 1, :]
    means_srgb = patch_srgb.mean(axis=(0, 1))

    linear     = _srgb_to_linear(img_rgb)
    patch_lin  = linear[py - r : py + r + 1, px - r : px + r + 1, :]
    means_lin  = patch_lin.mean(axis=(0, 1))

    too_dark = bool(means_lin.min() < 0.02)
    if too_dark or means_lin.min() < 1e-5:
        muls = (1.0, 1.0, 1.0)
    else:
        target = float(means_lin.mean())
        raw    = np.clip(target / means_lin, 0.10, 10.0)
        muls   = (float(raw[0]), float(raw[1]), float(raw[2]))

    return {
        "rgb_means": (float(means_srgb[0]), float(means_srgb[1]), float(means_srgb[2])),
        "muls":      muls,
        "too_dark":  too_dark,
    }


# ── Etape pipeline ─────────────────────────────────────────────────────────────

class WhiteBalanceStep(StepBase):
    id                = "wb"
    name              = "Balance des blancs"
    short_name        = "Blancs WB"
    slow              = False
    enabled_by_default = False
    has_overlay       = False
    has_color_picker  = True   # active le bouton « Sélectionner point blanc »
    previewable       = True   # aperçu instantané à chaque changement de param

    param_defs = [
        {
            "key":     "patch_radius",
            "label":   "Rayon d'éch. (px)",
            "type":    "int",
            "default": 5,
            "min":     1,
            "max":     30,
            "step":    1,
        },
    ]

    def __init__(self) -> None:
        super().__init__()
        self._pick_x: int | None = None
        self._pick_y: int | None = None

    # ── API depuis l'UI ──────────────────────────────────────────────────────

    def set_pick_point(self, x: int, y: int) -> None:
        self._pick_x = int(x)
        self._pick_y = int(y)

    def get_pick_point(self) -> tuple[int, int] | None:
        if self._pick_x is None or self._pick_y is None:
            return None
        return (self._pick_x, self._pick_y)

    def clear_pick_point(self) -> None:
        self._pick_x = None
        self._pick_y = None

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def process(
        self,
        img:     np.ndarray,
        params:  dict,
        context: dict,
    ) -> tuple[np.ndarray, dict]:
        if self._pick_x is None or self._pick_y is None:
            return img.copy(), {"wb_applied": False}

        d      = self.default_params()
        radius = int(params.get("patch_radius", d["patch_radius"]))

        # Echantillonner les infos pour le contexte
        info   = sample_patch_info(img, self._pick_x, self._pick_y, radius)
        result = apply_white_balance(img, self._pick_x, self._pick_y, radius)

        return result, {
            "wb_applied":    True,
            "wb_pick":       (self._pick_x, self._pick_y),
            "wb_patch_rgb":  info["rgb_means"],
            "wb_muls":       info["muls"],
            "wb_too_dark":   info["too_dark"],
        }
