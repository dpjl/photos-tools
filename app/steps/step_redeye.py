"""steps/step_redeye.py — Étape 2 : correction avancée des yeux rouges.

Algorithme en 5 phases :

1. **Détection faciale** via RetinaFace (facexlib, déjà installé pour GFPGAN)
   → landmarks 5 points → centres des yeux → zones de recherche précises
   Fallback : si aucun visage, détection globale de blobs rouges.

2. **Détection des pixels rouges** dans chaque zone œil
   R > sensibilité × G  AND  R > sensibilité × B  AND  R > 80

3. **Nettoyage morphologique** (fermeture elliptique) + plus grande
   composante connexe → isole le blob pupille/iris.

4. **Cercle englobant minimum** sur le blob + expansion (paramètre `expand`)
   + masque gaussien doux (σ = rayon/2.5) pour une transition invisible.

5. **Correction couleur** : R_corrigé = (G + B) / 2
   Préserve la luminosité (G et B inchangés), retire le rouge.
   Intensité contrôlée par le paramètre `strength`.

Paramètre `show_detections` : quand activé, retourne l'IMAGE D'ENTRÉE avec
les cercles de détection tracés en vert — sans appliquer la correction.
Permet de vérifier visuellement la précision de la détection avant traitement.
"""

from __future__ import annotations
import cv2
import numpy as np

from steps.base import StepBase

# ── Seuils internes ───────────────────────────────────────────────────────────
_MIN_RED_PIXELS  = 20    # pixels rouges minimum pour valider un blob iris
_MIN_BLOB_AREA   = 10    # pixels² minimum d'une composante connexe valide
_MAX_BLOB_AREA   = 8000  # pixels² maximum (filtre les faux positifs larges)
_MIN_R_VALUE     = 80    # valeur minimale du canal R pour être considéré rouge


class RedEyeStep(StepBase):
    id                 = "redeye"
    name               = "2 · Correction yeux rouges"
    short_name         = "YeuxRouges"
    slow               = True
    enabled_by_default = True

    param_defs = [
        {"key": "sensitivity", "label": "Sensibilité (abaisser si photos fanées)", "type": "float",
         "default": 1.8, "min": 1.2, "max": 4.0, "step": 0.1},
        {"key": "strength",    "label": "Force correction",                  "type": "float",
         "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "expand",      "label": "Expansion rayon iris (×)",          "type": "float",
         "default": 1.4, "min": 1.0, "max": 3.0, "step": 0.1},
        {"key": "show_detections", "label": "Afficher détections (img. originale)",
         "type": "bool", "default": False},
    ]

    def __init__(self):
        self._face_helper = None

    def _get_face_helper(self):
        if self._face_helper is None:
            import torch
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            self._face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=False,
                device=torch.device("cpu"),
            )
        return self._face_helper

    def process(self, img: np.ndarray, params: dict, context: dict):
        sensitivity  = float(params.get("sensitivity",     2.0))
        strength     = float(params.get("strength",        1.0))
        expand       = float(params.get("expand",          1.4))
        show_det     = bool(params.get("show_detections",  False))

        # ── 1. Détecter les zones œil ────────────────────────────────────────
        eye_regions = _detect_eye_regions(self._get_face_helper(), img)
        if not eye_regions:
            eye_regions = _fallback_eye_regions(img, sensitivity)

        # ── 2. Appliquer la correction zone par zone ─────────────────────────
        result           = img.copy()
        detected_circles: list[tuple[float, float, float]] = []

        for (ex, ey, er) in eye_regions:
            circles = _correct_eye_region(result, ex, ey, er,
                                          sensitivity, strength, expand)
            detected_circles.extend(circles)

        # ── 3. Sortie ────────────────────────────────────────────────────────
        if show_det:
            # Retourne l'image d'ORIGINE annotée (pas la version corrigée)
            overlay = img.copy()
            for (cx, cy, r) in detected_circles:
                cv2.circle(overlay, (int(cx), int(cy)), max(int(r), 3),
                           (0, 210, 0), 2, cv2.LINE_AA)
                # Petit point central pour repérer le centre précis
                cv2.circle(overlay, (int(cx), int(cy)), 1,
                           (0, 255, 128), -1, cv2.LINE_AA)
            return overlay, {}

        return result, {}


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions de traitement internes
# ──────────────────────────────────────────────────────────────────────────────

def _detect_eye_regions(
    face_helper, img: np.ndarray
) -> list[tuple[int, int, int]]:
    """Retourne [(cx, cy, search_r), ...] pour chaque œil détecté via RetinaFace.

    Les landmarks RetinaFace [0]=œil gauche image, [1]=œil droit image.
    Le rayon de recherche est proportionnel à la distance inter-oculaire.
    """
    try:
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    except Exception:
        return []

    regions: list[tuple[int, int, int]] = []
    for lm5 in face_helper.all_landmarks_5:
        lm = np.array(lm5)
        inter_eye = float(np.linalg.norm(lm[1] - lm[0]))
        if inter_eye < 5:
            continue
        # Rayon de recherche ≈ largeur d'un œil (≈ inter-oculaire / 2.8)
        search_r = max(int(inter_eye / 2.8), 18)
        for eye_lm in (lm[0], lm[1]):   # [0]=gauche, [1]=droite
            regions.append((int(eye_lm[0]), int(eye_lm[1]), search_r))
    return regions


def _fallback_eye_regions(
    img: np.ndarray, sensitivity: float
) -> list[tuple[int, int, int]]:
    """Fallback (pas de visage détecté) : cherche les blobs rouges plausibles.

    Filtre par taille pour ne retenir que des zones compatibles avec des yeux.
    """
    B, G, R = cv2.split(img)
    red_mask = (
        (R.astype(np.int32) > G.astype(np.int32) * sensitivity) &
        (R.astype(np.int32) > B.astype(np.int32) * sensitivity) &
        (R > _MIN_R_VALUE)
    ).astype(np.uint8) * 255

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, 8)
    regions: list[tuple[int, int, int]] = []
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if _MIN_BLOB_AREA <= area <= _MAX_BLOB_AREA:
            cx, cy = centroids[lbl]
            # Rayon de recherche ≈ 3 × rayon équivalent du blob
            r = max(int(np.sqrt(area / np.pi) * 3), 10)
            regions.append((int(cx), int(cy), r))
    return regions


def _correct_eye_region(
    result:      np.ndarray,
    cx:          int,
    cy:          int,
    search_r:    int,
    sensitivity: float,
    strength:    float,
    expand:      float,
) -> list[tuple[float, float, float]]:
    """Corrige les yeux rouges dans la zone autour de (cx, cy).

    Modifie `result` in-place et retourne la liste des cercles effectivement
    corrigés sous forme [(iris_cx_global, iris_cy_global, iris_r), ...].
    """
    h, w = result.shape[:2]
    x1 = max(0, cx - search_r)
    y1 = max(0, cy - search_r)
    x2 = min(w, cx + search_r)
    y2 = min(h, cy + search_r)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return []

    crop = result[y1:y2, x1:x2].copy()
    B_c, G_c, R_c = cv2.split(crop)

    # ── a. Masque rouge brut ─────────────────────────────────────────────────
    red_m = (
        (R_c.astype(np.int32) > G_c.astype(np.int32) * sensitivity) &
        (R_c.astype(np.int32) > B_c.astype(np.int32) * sensitivity) &
        (R_c > _MIN_R_VALUE)
    ).astype(np.uint8) * 255

    # ── b. Nettoyage morphologique (fermeture) ───────────────────────────────
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_m = cv2.morphologyEx(red_m, cv2.MORPH_CLOSE, k, iterations=2)

    if int(red_m.sum()) // 255 < _MIN_RED_PIXELS:
        return []

    # ── c. Plus grande composante connexe ────────────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_m, 8)
    if n_labels < 2:
        return []
    largest   = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob_area = int(stats[largest, cv2.CC_STAT_AREA])
    if blob_area < _MIN_BLOB_AREA:
        return []

    blob = (labels == largest).astype(np.uint8)

    # ── d. Cercle englobant minimum + expansion ──────────────────────────────
    pts_xy        = np.column_stack(np.where(blob)[::-1]).astype(np.float32)  # (x,y)
    (bcx, bcy), base_r = cv2.minEnclosingCircle(pts_xy)
    iris_r        = max(base_r * expand, 3.0)

    # ── e. Masque gaussien doux ───────────────────────────────────────────────
    ch, cw = crop.shape[:2]
    yy, xx = np.mgrid[:ch, :cw]
    dist      = np.sqrt((xx - bcx) ** 2 + (yy - bcy) ** 2)
    sigma     = iris_r / 2.5
    soft_mask = np.exp(-0.5 * (dist / (sigma + 1e-6)) ** 2).astype(np.float32)

    # ── f. Correction : R_naturel = (G + B) / 2 ──────────────────────────────
    R_f   = R_c.astype(np.float32)
    R_nat = (G_c.astype(np.float32) + B_c.astype(np.float32)) * 0.5
    alpha = np.clip(soft_mask * strength, 0.0, 1.0)
    R_new = R_f * (1.0 - alpha) + R_nat * alpha

    crop[:, :, 2] = np.clip(R_new, 0, 255).astype(np.uint8)
    result[y1:y2, x1:x2] = crop

    return [(x1 + bcx, y1 + bcy, iris_r)]
