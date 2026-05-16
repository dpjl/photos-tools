"""steps/step_facehighlight.py — Étape 3 · Correction hautes lumières (visages).

Algorithme en 4 phases — conçu pour les portraits flash surexposés :

1. **Détection faciale** via RetinaFace (facexlib) → landmarks 5 points.
   Chaque visage est évalué : si la fraction de pixels avec L* > seuil
   dépasse _MIN_OVEREXP_FRACTION, il est corrigé.

2. **Courbe tonale en espace LAB** (shoulder Reinhard au-dessus du seuil L*).
   Comprime la plage [seuil, 100] vers [seuil, L_max] avec une smoothstep
   pour la transition — aucun artefact de bord, dérivée continue.

3. **Récupération de la couleur peau** — la zone surexposée perd sa teinte
   chaude (blanchiment flash). On ré-injecte les valeurs a*/b* échantillonnées
   dans la zone pré-seuil du même visage (ou une valeur de secours si la
   zone de référence est trop petite). La correction est pondérée par hmask.

4. **Microcontraste** (CLAHE) sur le canal L* dans la zone corrigée,
   pour restaurer la texture de peau que la compression tonale atténue.

Fusion finale : masque elliptique doux (landmarks → ellipse + blur gaussien)
pour intégrer la correction dans l'image sans bord visible.

Choix de l'approche LAB vs RGB direct :
  · L* est perceptuellement uniforme → le seuil est intuitif (0-100)
  · a*/b* séparent la couleur de la luminance → correction ciblée
  · Le shoulder Reinhard est un classique de la photographie numérique
    (Reinhard et al., "Photographic Tone Reproduction for Digital Images", 2002)
    adapté ici en « local face highlight » au lieu de HDR global.
"""

from __future__ import annotations
import cv2
import numpy as np

from steps.base import StepBase

_MIN_OVEREXP_FRACTION = 0.05   # ≥ 5 % du visage doit être surexposé


class FaceHighlightStep(StepBase):
    id                 = "facehighlight"
    name               = "3 · Hautes lumières visages"
    short_name         = "HautesLum"
    slow               = True
    enabled_by_default = False
    has_overlay        = True

    param_defs = [
        {"key": "threshold",
         "label": "Seuil hautes lumières (L*, 0-100)", "type": "int",
         "default": 85, "min": 70, "max": 98, "step": 1},
        {"key": "strength",
         "label": "Force correction luminosité", "type": "float",
         "default": 0.80, "min": 0.10, "max": 1.0, "step": 0.05},
        {"key": "recover_color",
         "label": "Récupération couleur peau", "type": "float",
         "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "texture",
         "label": "Restauration texture (CLAHE)", "type": "float",
         "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05},
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

    def process(self, img: np.ndarray, params: dict, context: dict):
        threshold     = int(  params.get("threshold",     85))
        strength      = float(params.get("strength",      0.80))
        recover_color = float(params.get("recover_color", 0.60))
        texture       = float(params.get("texture",       0.40))

        result, detections = _detect_and_correct(
            self._get_face_helper(), img, threshold, strength, recover_color, texture
        )
        return result, {"highlight_detections": detections}


# ── Détection et dispatch ──────────────────────────────────────────────────────

def _detect_and_correct(
    face_helper, img: np.ndarray,
    threshold: int, strength: float, recover_color: float, texture: float,
):
    try:
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    except Exception:
        return img, []

    if not face_helper.all_landmarks_5:
        return img, []

    result:     np.ndarray  = img.copy()
    detections: list[dict]  = []

    for lm5 in face_helper.all_landmarks_5:
        lm   = np.array(lm5)
        bbox = _bbox_from_landmarks(lm, img.shape)
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        face_crop = result[y1:y2, x1:x2].copy()

        # Score de surexposition (fraction de pixels > seuil L*)
        L_ch      = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)[:, :, 0]
        L_f       = L_ch.astype(np.float32) / 255.0 * 100.0
        overexp   = float((L_f > threshold).mean())

        if overexp < _MIN_OVEREXP_FRACTION:
            detections.append({
                "bbox":      (x1, y1, x2, y2),
                "corrected": False,
                "overexp":   overexp,
            })
            continue

        # Correction + masque de fusion
        corrected_crop = _correct_highlights(face_crop, threshold, strength,
                                             recover_color, texture)
        face_mask      = _face_blend_mask(face_crop.shape, lm, x1, y1)
        blended        = (
            face_crop.astype(np.float32) * (1.0 - face_mask[..., None])
            + corrected_crop.astype(np.float32) * face_mask[..., None]
        )
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        detections.append({
            "bbox":      (x1, y1, x2, y2),
            "corrected": True,
            "overexp":   overexp,
        })

    return result, detections


# ── Géométrie faciale ─────────────────────────────────────────────────────────

def _bbox_from_landmarks(lm: np.ndarray, img_shape: tuple) -> tuple[int, int, int, int]:
    """Bbox du visage estimée depuis les 5 landmarks RetinaFace.

    Les yeux (lm[0]=gauche, lm[1]=droite) se trouvent à ≈35 % depuis le haut.
    On étend d'1.8× la distance inter-yeux vers le haut (front) et 2.2× vers
    le bas (menton) pour inclure le visage entier.
    """
    h, w       = img_shape[:2]
    eye_dist   = float(np.linalg.norm(lm[1] - lm[0]))
    eye_cx     = float((lm[0][0] + lm[1][0]) / 2)
    eye_cy     = float((lm[0][1] + lm[1][1]) / 2)
    x1 = max(0, int(eye_cx - eye_dist * 1.45))
    y1 = max(0, int(eye_cy - eye_dist * 1.80))
    x2 = min(w, int(eye_cx + eye_dist * 1.45))
    y2 = min(h, int(eye_cy + eye_dist * 2.20))
    return x1, y1, x2, y2


def _face_blend_mask(
    shape: tuple, lm: np.ndarray, offset_x: int, offset_y: int
) -> np.ndarray:
    """Masque elliptique doux pour fusionner la correction dans l'image.

    Centré sur le barycentre des 5 landmarks, dimensions proportionnelles
    à la distance inter-yeux. Flou gaussien pour éviter un bord visible.
    """
    h, w     = shape[:2]
    mask     = np.zeros((h, w), dtype=np.float32)
    lm_local = lm - np.array([offset_x, offset_y])
    eye_dist = float(np.linalg.norm(lm_local[1] - lm_local[0]))
    center   = np.mean(lm_local, axis=0)
    cx, cy   = int(center[0]), int(center[1])
    rx       = max(1, int(eye_dist * 1.30))
    ry       = max(1, int(eye_dist * 1.75))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    sigma    = max(eye_dist * 0.25, 3.0)
    mask     = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    return np.clip(mask, 0, 1)


# ── Correction hautes lumières ─────────────────────────────────────────────────

def _correct_highlights(
    face_crop:     np.ndarray,
    threshold:     int,
    strength:      float,
    recover_color: float,
    texture_str:   float,
) -> np.ndarray:
    """Corrige les hautes lumières dans un crop de visage.

    L_max = threshold + (100-threshold)*(1 - strength*0.70)
    Shoulder Reinhard via smoothstep sur [threshold, 100] → [threshold, L_max].
    Transition progressive sur 8 unités L* pour éviter toute discontinuité.
    """
    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0].astype(np.float32) / 255.0 * 100.0   # [0, 100]
    A   = lab[:, :, 1].astype(np.float32) - 128.0            # [-128, 127]
    Bch = lab[:, :, 2].astype(np.float32) - 128.0

    # ── Masque de surexposition : transition smoothstep sur 8 L* units ──────
    t_mask = np.clip((L - threshold) / 8.0, 0.0, 1.0)
    hmask  = t_mask * t_mask * (3.0 - 2.0 * t_mask)         # smoothstep ∈ [0,1]

    # ── Couleur de peau de référence (zone pré-seuil : [threshold-12, threshold[) ─
    good_px = (L >= float(threshold - 12)) & (L < float(threshold))
    if good_px.sum() >= 25:
        a_ref = float(np.median(A[good_px]))
        b_ref = float(np.median(Bch[good_px]))
    else:
        # Secours : teint peau caucasien clair sous illuminant D65 (LAB)
        a_ref, b_ref = 7.0, 11.0

    # ── Courbe tonale : shoulder Reinhard (smoothstep au-dessus du seuil) ────
    # Mappe [threshold, 100] → [threshold, L_max] de façon continue.
    L_max  = float(threshold) + (100.0 - threshold) * (1.0 - strength * 0.70)
    t_tone = np.clip((L - threshold) / (100.0 - threshold + 1e-6), 0.0, 1.0)
    t_s    = t_tone * t_tone * (3.0 - 2.0 * t_tone)  # smoothstep non-linéaire
    L_red  = float(threshold) + t_s * (L_max - float(threshold))
    L_fin  = L * (1.0 - hmask) + L_red * hmask         # blend : orig → réduit

    # ── Récupération couleur peau ─────────────────────────────────────────────
    A_fin  = A   + (a_ref - A)   * hmask * recover_color
    B_fin  = Bch + (b_ref - Bch) * hmask * recover_color

    # ── Reconstruction LAB → BGR (uint8) ─────────────────────────────────────
    L_u8  = np.clip(L_fin / 100.0 * 255.0, 0, 255).astype(np.uint8)
    A_u8  = np.clip(A_fin  + 128.0,         0, 255).astype(np.uint8)
    B_u8  = np.clip(B_fin  + 128.0,         0, 255).astype(np.uint8)
    corrected = cv2.cvtColor(cv2.merge([L_u8, A_u8, B_u8]), cv2.COLOR_LAB2BGR)

    # ── Restauration texture via CLAHE (pondérée par hmask) ───────────────────
    if texture_str > 0.0:
        lab2     = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        tile     = max(4, min(16, min(face_crop.shape[:2]) // 12))
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile, tile))
        L_eq     = clahe.apply(lab2[:, :, 0]).astype(np.float32)
        L_old    = lab2[:, :, 0].astype(np.float32)
        lab2[:, :, 0] = np.clip(
            L_old + (L_eq - L_old) * hmask * texture_str, 0, 255
        ).astype(np.uint8)
        corrected = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    return corrected
