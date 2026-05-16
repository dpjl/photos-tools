"""steps/step_facehighlight.py — Étape 3 · Correction hautes lumières (global).

Algorithme — traitement global de TOUTES les zones surexposées de l'image
(visages, bras, objets proches) via séparation fréquentielle et inpainting :

1. **Masque global** en espace LAB (L* 0-100 perceptuellement uniforme).
   Transition smoothstep sur 10 L* unités pour éviter tout bord visible.

2. **Séparation fréquentielle** (filtre bilatéral edge-preserving) :
   · Couche base  = luminance basse fréquence (structures larges)
   · Couche détail = L − base (texture fine, contours)
   → Seule la couche base est comprimée ; la couche détail est préservée
   (voire amplifiée) pour éviter l'aspect « lisse et plastique ».

3. **Compression tonale shoulder Reinhard** sur la couche base :
   Mappe [seuil, 100] → [seuil, L_max] de façon continue (smoothstep).
   Avec strength=1 : tout au-dessus du seuil → seuil (compression max).
   Avec strength=0 : aucune compression.

4. **Récupération de couleur via inpainting Telea** (cv2.INPAINT_TELEA) :
   Les zones soufflées perdent leur couleur (a*,b* → 0 = blanc).
   L'inpainting propage la couleur depuis les bords non-surexposés
   vers l'intérieur. Fusion pondérée par masque × color_boost.

La détection faciale (RetinaFace) sert uniquement à enrichir l'overlay
(bbox de chaque visage + fraction de surexposition par visage).
"""

from __future__ import annotations
import cv2
import numpy as np

from steps.base import StepBase

_MIN_REGION_AREA = 300    # px² — ignorer les petites taches parasites


class FaceHighlightStep(StepBase):
    id                 = "facehighlight"
    name               = "3 · Correction hautes lumières"
    short_name         = "HautesLum"
    slow               = True
    enabled_by_default = False
    has_overlay        = True

    param_defs = [
        {"key": "threshold",
         "label": "Seuil luminosité (L*, 0-100)", "type": "int",
         "default": 82, "min": 70, "max": 98, "step": 1},
        {"key": "strength",
         "label": "Force de la compression", "type": "float",
         "default": 0.75, "min": 0.10, "max": 1.0, "step": 0.05},
        {"key": "texture",
         "label": "Restauration texture", "type": "float",
         "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05},
        {"key": "color_boost",
         "label": "Récupération couleur", "type": "float",
         "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05},
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
        threshold   = int(  params.get("threshold",   90))
        strength    = float(params.get("strength",    0.75))
        texture     = float(params.get("texture",     0.55))
        color_boost = float(params.get("color_boost", 0.60))

        result, detections = _correct_global(
            img, threshold, strength, texture, color_boost,
            self._get_face_helper(),
        )
        return result, {"highlight_detections": detections}


# ── Traitement global ──────────────────────────────────────────────────────────

def _correct_global(
    img:         np.ndarray,
    threshold:   int,
    strength:    float,
    texture:     float,
    color_boost: float,
    face_helper,
) -> tuple[np.ndarray, list[dict]]:
    """Applique la correction hautes lumières sur l'image entière."""

    h, w = img.shape[:2]

    # ── Conversion LAB ────────────────────────────────────────────────────────
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_u8 = lab[:, :, 0]                                # uint8 [0, 255]
    A_u8 = lab[:, :, 1]                                # uint8 offset +128
    B_u8 = lab[:, :, 2]
    L    = L_u8.astype(np.float32) / 255.0 * 100.0    # float [0, 100]
    A    = A_u8.astype(np.float32) - 128.0             # float [-128, 127]
    B    = B_u8.astype(np.float32) - 128.0

    # ── Masque de surexposition (smoothstep sur 10 L* unités) ─────────────────
    feather = 10.0
    t_mask  = np.clip((L - threshold) / feather, 0.0, 1.0)
    hmask   = t_mask * t_mask * (3.0 - 2.0 * t_mask)       # smoothstep ∈ [0,1]

    overexp_fraction = float((L > threshold).mean())
    if overexp_fraction < 0.001:
        # Aucune zone surexposée : retourner l'image inchangée avec les infos visages
        detections = _face_detections(face_helper, img, threshold)
        return img, detections

    # ── Compression tonale shoulder Reinhard (sur L original) ────────────────
    # Mappe [threshold, 100] → [threshold, L_max] de façon continue (smoothstep).
    # strength=1 → tout comprimé jusqu'au seuil.  strength=0 → aucun effet.
    L_range = max(100.0 - threshold, 1.0)
    L_max   = float(threshold) + L_range * (1.0 - strength)

    t_tone  = np.clip((L - threshold) / L_range, 0.0, 1.0)
    ts_tone = t_tone * t_tone * (3.0 - 2.0 * t_tone)       # smoothstep non-linéaire
    L_compressed = float(threshold) + ts_tone * (L_max - float(threshold))

    # Fusion : pixels non-surexposés inchangés, surexposés → L comprimé
    L_new = L * (1.0 - hmask) + L_compressed * hmask

    # ── Restauration de la texture (CLAHE sur le résultat comprimé) ──────────
    # CLAHE redistribue le contraste local dans les zones comprimées, révélant
    # les micro-détails subsistants sans amplifier le bruit hors-masque.
    if texture > 0.0:
        L_new_u8 = np.clip(L_new / 100.0 * 255.0, 0, 255).astype(np.uint8)
        tile     = max(4, min(16, min(h, w) // 12))
        clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tile, tile))
        L_clahe  = clahe.apply(L_new_u8).astype(np.float32) / 255.0 * 100.0
        # Appliqué uniquement dans les zones corrigées
        L_new = L_new * (1.0 - texture * hmask * 0.7) + L_clahe * (texture * hmask * 0.7)
        L_new = np.clip(L_new, 0.0, 100.0)

    # ── Récupération de couleur via inpainting ────────────────────────────────
    if color_boost > 0.0:
        blown_mask = (hmask > 0.3).astype(np.uint8) * 255
        if blown_mask.any():
            # Rayon d'inpainting : équilibre vitesse / propagation
            inpaint_r = max(8, min(25, min(h, w) // 45))
            A_inp = cv2.inpaint(A_u8, blown_mask, inpaint_r, cv2.INPAINT_TELEA)
            B_inp = cv2.inpaint(B_u8, blown_mask, inpaint_r, cv2.INPAINT_TELEA)
            A_inp_f = A_inp.astype(np.float32) - 128.0
            B_inp_f = B_inp.astype(np.float32) - 128.0
            # Fusion : zones soufflées reçoivent la couleur inpaintée
            A_new = A + (A_inp_f - A) * hmask * color_boost
            B_new = B + (B_inp_f - B) * hmask * color_boost
        else:
            A_new, B_new = A, B
    else:
        A_new, B_new = A, B

    # ── Reconstruction BGR ────────────────────────────────────────────────────
    L_out = np.clip(L_new / 100.0 * 255.0, 0, 255).astype(np.uint8)
    A_out = np.clip(A_new + 128.0,          0, 255).astype(np.uint8)
    B_out = np.clip(B_new + 128.0,          0, 255).astype(np.uint8)
    result = cv2.cvtColor(cv2.merge([L_out, A_out, B_out]), cv2.COLOR_LAB2BGR)

    # ── Détections pour overlay ───────────────────────────────────────────────
    detections = _build_detections(hmask, img, face_helper, threshold,
                                   overexp_fraction)
    return result, detections


# ── Overlay : régions lumineuses + visages ─────────────────────────────────────

def _build_detections(
    hmask: np.ndarray,
    img:   np.ndarray,
    face_helper,
    threshold: int,
    overexp_fraction: float,
) -> list[dict]:
    """Construit la liste de détections pour l'overlay."""
    detections: list[dict] = []

    # Composantes connexes des zones surexposées (bboxes pour overlay)
    blown_u8 = (hmask > 0.4).astype(np.uint8) * 255
    if blown_u8.any():
        n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(blown_u8)
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < _MIN_REGION_AREA:
                continue
            x1 = int(stats[i, cv2.CC_STAT_LEFT])
            y1 = int(stats[i, cv2.CC_STAT_TOP])
            x2 = x1 + int(stats[i, cv2.CC_STAT_WIDTH])
            y2 = y1 + int(stats[i, cv2.CC_STAT_HEIGHT])
            detections.append({
                "type":      "region",
                "bbox":      (x1, y1, x2, y2),
                "area":      area,
                "overexp":   overexp_fraction,
                "corrected": True,
            })

    # Visages : fraction de surexposition par visage (info complémentaire)
    for fd in _face_detections(face_helper, img, threshold):
        detections.append(fd)

    return detections


def _face_detections(face_helper, img: np.ndarray, threshold: int) -> list[dict]:
    """Détecte les visages et calcule leur taux de surexposition."""
    try:
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    except Exception:
        return []

    if not face_helper.all_landmarks_5:
        return []

    out: list[dict] = []
    for lm5 in face_helper.all_landmarks_5:
        lm   = np.array(lm5)
        bbox = _bbox_from_landmarks(lm, img.shape)
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        face_crop = img[y1:y2, x1:x2]
        L_crop    = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)[:, :, 0]
        overexp   = float((L_crop.astype(np.float32) / 255.0 * 100.0 > threshold).mean())
        out.append({
            "type":      "face",
            "bbox":      (x1, y1, x2, y2),
            "overexp":   overexp,
            "corrected": overexp >= 0.05,
        })
    return out


def _bbox_from_landmarks(lm: np.ndarray, img_shape: tuple) -> tuple[int, int, int, int]:
    h, w     = img_shape[:2]
    eye_dist = float(np.linalg.norm(lm[1] - lm[0]))
    eye_cx   = float((lm[0][0] + lm[1][0]) / 2)
    eye_cy   = float((lm[0][1] + lm[1][1]) / 2)
    x1 = max(0, int(eye_cx - eye_dist * 1.45))
    y1 = max(0, int(eye_cy - eye_dist * 1.80))
    x2 = min(w, int(eye_cx + eye_dist * 1.45))
    y2 = min(h, int(eye_cy + eye_dist * 2.20))
    return x1, y1, x2, y2
