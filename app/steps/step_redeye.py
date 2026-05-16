"""steps/step_redeye.py — Étape 2 : correction avancée des yeux rouges.

Architecture en deux phases :

1. **Localisation de l'iris** via RetinaFace (facexlib)
   Landmarks 5 points → centres des yeux → rayon anatomique :
   iris_r = distance_inter-pupillaire × IRIS_RATIO
   Le rayon est fixe pour un visage donné — il ne dépend PAS de la sensibilité.
   Fallback : détection de blobs rouges si aucun visage n'est détecté.

2. **Correction** dans le cercle iris précis
   a. Masque rouge : R > sensibilité × G  ET  R > sensibilité × B  ET  R > MIN_R_VALUE
   b. Masque gaussien centré sur l'iris (sigma = iris_r / 2) + cutoff dur à iris_r × expand
   c. R_corrigé = R × (1 − alpha) + (G + B)/2 × alpha

Avantage vs l'ancien blob-detection :
  · Le cercle de correction ne change PAS selon la valeur de sensibilité
  · Sensitivity contrôle UNIQUEMENT quels pixels DANS l'iris sont corrigés
  · Calibré contre MediaPipe FaceMesh : iris_r = 6.3 px (MediaPipe)
    vs 6.2 px (inter_eye × 0.105) — écart sub-pixel.
"""

from __future__ import annotations
import cv2
import numpy as np

from steps.base import StepBase

_IRIS_TO_IPD_RATIO = 0.105   # iris_r / distance inter-pupillaire (anatomique)
_SEARCH_TO_IRIS    = 3.0     # search_r = iris_r × 3 → zone large pour trouver le rouge
_MIN_IRIS_R        = 3.0
_MIN_RED_PIXELS    = 5
_MIN_R_VALUE       = 80
_MIN_BLOB_AREA     = 10
_MAX_BLOB_AREA     = 5000


class RedEyeStep(StepBase):
    id                 = "redeye"
    name               = "4 · Correction yeux rouges"
    short_name         = "YeuxRouges"
    slow               = True
    enabled_by_default = True
    has_overlay        = True

    param_defs = [
        {"key": "sensitivity", "label": "Sensibilité rouge (R/G et R/B)", "type": "float",
         "default": 1.5, "min": 1.1, "max": 4.0, "step": 0.1},
        {"key": "strength",    "label": "Force correction",               "type": "float",
         "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "expand",      "label": "Rayon correction (× rayon iris)", "type": "float",
         "default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1},
    ]

    def __init__(self):
        self._face_helper = None

    def _get_face_helper(self):
        if self._face_helper is None:
            import torch
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            self._face_helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model="retinaface_resnet50", save_ext="png",
                use_parse=False, device=torch.device("cpu"),
            )
        return self._face_helper

    def process(self, img, params, context):
        sensitivity = float(params.get("sensitivity", 1.5))
        strength    = float(params.get("strength",    1.0))
        expand      = float(params.get("expand",      1.5))

        iris_circles = _detect_iris_circles(self._get_face_helper(), img)
        if not iris_circles:
            iris_circles = _fallback_iris_circles(img, sensitivity)

        result     = img.copy()
        detections = []
        for (cx, cy, iris_r) in iris_circles:
            corrected, fix_cx, fix_cy = _correct_iris(
                result, cx, cy, iris_r, sensitivity, strength, expand)
            detections.append({"iris": (float(fix_cx), float(fix_cy), float(iris_r)),
                               "corrected": corrected})

        return result, {"redeye_detections": detections}


def _detect_iris_circles(face_helper, img):
    """RetinaFace landmarks + rayon anatomique iris_r = IPD * IRIS_TO_IPD_RATIO."""
    try:
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    except Exception:
        return []

    circles = []
    for lm5 in face_helper.all_landmarks_5:
        lm        = np.array(lm5)
        inter_eye = float(np.linalg.norm(lm[1] - lm[0]))
        if inter_eye < 5:
            continue
        iris_r = max(inter_eye * _IRIS_TO_IPD_RATIO, _MIN_IRIS_R)
        for eye_lm in (lm[0], lm[1]):
            circles.append((float(eye_lm[0]), float(eye_lm[1]), iris_r))
    return circles


def _fallback_iris_circles(img, sensitivity):
    """Fallback si aucun visage détecté : blobs rouges plausibles."""
    B, G, R = cv2.split(img)
    red_mask = (
        (R.astype(np.int32) > G.astype(np.int32) * sensitivity) &
        (R.astype(np.int32) > B.astype(np.int32) * sensitivity) &
        (R > _MIN_R_VALUE)
    ).astype(np.uint8) * 255
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(red_mask, 8)
    circles = []
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if _MIN_BLOB_AREA <= area <= _MAX_BLOB_AREA:
            cx, cy = centroids[lbl]
            circles.append((float(cx), float(cy),
                            max(float(np.sqrt(area / np.pi)), _MIN_IRIS_R)))
    return circles


def _correct_iris(result, cx, cy, iris_r, sensitivity, strength, expand):
    """Correction en deux passes.

    1. Zone de RECHERCHE large (search_r = iris_r × _SEARCH_TO_IRIS) autour du
       landmark RetinaFace pour trouver les pixels rouges, même si le centre
       détecté n'est pas exactement sur la pupille.
    2. Centroïde des pixels rouges trouvés → nouveau centre réel de correction.
    3. Masque gaussien (σ = iris_r) centré sur ce centroïde, cutoff dur à
       corr_r = iris_r × expand.

    Retourne (corrected: bool, final_cx: float, final_cy: float).
    """
    h, w     = result.shape[:2]
    search_r = iris_r * _SEARCH_TO_IRIS      # zone de recherche (large, dynamique)
    corr_r   = iris_r * expand               # zone de correction (anatomique)
    margin   = int(search_r) + 2
    x1 = max(0, int(cx) - margin);  y1 = max(0, int(cy) - margin)
    x2 = min(w, int(cx) + margin);  y2 = min(h, int(cy) + margin)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False, cx, cy

    crop          = result[y1:y2, x1:x2].copy()
    B_c, G_c, R_c = cv2.split(crop)
    lcx, lcy      = cx - x1, cy - y1
    ch, cw        = crop.shape[:2]
    yy, xx        = np.mgrid[:ch, :cw]
    dist_lm       = np.sqrt((xx - lcx) ** 2 + (yy - lcy) ** 2)

    # Masque rouge (calculé une seule fois, réutilisé pour centroïde et correction)
    red_m = (
        (R_c.astype(np.int32) > G_c.astype(np.int32) * sensitivity) &
        (R_c.astype(np.int32) > B_c.astype(np.int32) * sensitivity) &
        (R_c > _MIN_R_VALUE)
    ).astype(np.float32)

    # Passe 1 : y a-t-il assez de rouge dans la zone de recherche ?
    red_in_search = red_m * (dist_lm <= search_r).astype(np.float32)
    total_red     = float(red_in_search.sum())
    if total_red < _MIN_RED_PIXELS:
        return False, cx, cy

    # Passe 2 : centroïde du rouge → centre réel de correction
    rcx_local = float((red_in_search * xx).sum() / total_red)
    rcy_local = float((red_in_search * yy).sum() / total_red)
    dist_red  = np.sqrt((xx - rcx_local) ** 2 + (yy - rcy_local) ** 2)

    # Gaussienne σ = iris_r, fort au centre, fondu naturel vers corr_r
    sigma = max(iris_r, 1.0)
    gauss = np.exp(-0.5 * (dist_red / sigma) ** 2).astype(np.float32)
    gauss[dist_red > corr_r] = 0.0

    alpha = np.clip(gauss * red_m * strength, 0.0, 1.0)
    R_nat = (G_c.astype(np.float32) + B_c.astype(np.float32)) * 0.5
    R_new = R_c.astype(np.float32) * (1.0 - alpha) + R_nat * alpha

    crop[:, :, 2]         = np.clip(R_new, 0, 255).astype(np.uint8)
    result[y1:y2, x1:x2] = crop
    return True, rcx_local + x1, rcy_local + y1
