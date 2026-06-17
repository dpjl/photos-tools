"""steps/step_redlips.py — Correction des lèvres trop rouges.

Architecture en deux phases, sur le modèle de step_redeye :

1. **Détection des visages** via RetinaFace (facexlib FaceRestoreHelper)
   → crop expansé de chaque visage.

2. **Localisation des lèvres et correction** sur le crop :
   a. MediaPipe FaceLandmarker : contours externe et interne de la bouche
      → masque « anneau » des lèvres (l'intérieur de la bouche — dents,
      langue — est exclu). Les landmarks sont détectés sur un crop
      upscalé ×3 (petits visages) mais la correction s'applique à la
      résolution native : pas de rééchantillonnage du visage.
   b. La peau autour de la bouche sert d'étalon : des lèvres naturelles
      sont un peu plus saturées que la peau (ratio chroma ≈ 1.4 en OKLab).
   c. Correction en OKLab, luminance intacte (texture préservée) : la
      chromaticité des pixels en excès est ramenée vers la cible
      « chroma peau × ratio naturel », avec un léger rappel de teinte vers
      celle de la peau. Seuls les pixels rouges au-dessus de la cible sont
      touchés → aucun effet sur des lèvres déjà naturelles.

Overlay : contour des bouches détectées (vert = corrigé, cyan = lèvres
déjà naturelles).

Mode manuel : chaque zone peinte dans l'onglet « Lèvres » est traitée comme
une bouche (peau étalonnée sur un anneau autour de la zone).

Fallback : aucun visage RetinaFace → MediaPipe sur l'image entière.
"""

from __future__ import annotations
import os
import cv2
import numpy as np

from steps.base import StepBase
from steps.step_redeye import _detect_faces_retina, _expand_box
from core.genref import _bgr_to_oklab, _oklab_to_bgr
from config import APP_DIR

# ── Constantes ───────────────────────────────────────────────────────────────
_MEDIAPIPE_MODEL = os.path.join(APP_DIR, "face_landmarker.task")
_UPSCALE         = 3        # upscale du crop pour la détection des landmarks

# Des lèvres naturelles sont un peu plus saturées que la peau (OKLab)
_LIP_CHROMA_RATIO = 1.40
# Plancher de chroma cible (peau quasi neutre → éviter des lèvres grises)
_MIN_TARGET_CHROMA = 0.035
# Marge au-delà de laquelle des lèvres sont jugées « trop rouges »
_EXCESS_GATE = 1.08
# Part de rappel de la teinte vers celle de la peau (le reste garde la
# teinte du pixel) — évite le virage magenta des tirages fanés
_HUE_PULL = 0.30

# Indices MediaPipe FaceLandmarker : contours de la bouche (boucles ordonnées)
_LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
               291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
_LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
               308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


class RedLipsStep(StepBase):
    id                 = "redlips"
    name               = "Correction lèvres trop rouges"
    short_name         = "Lèvres"
    slow               = True
    enabled_by_default = False
    has_overlay        = True

    param_defs = [
        {"key": "profil", "label": "Profil", "type": "choice",
         "as_buttons": True,
         "default": "auto",
         "choices": ["auto", "manuel"]},
        {"key": "strength", "label": "Force correction", "type": "float",
         "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
    ]

    def __init__(self):
        self._face_helper = None
        self._landmarker  = None
        self._redlips_mask: np.ndarray | None = None  # masque peint (mode manuel)

    # ── API masque lèvres (appelée depuis l'UI) ───────────────────────────────

    def set_redlips_mask(self, mask: np.ndarray | None) -> None:
        self._redlips_mask = mask.copy() if mask is not None else None

    def get_redlips_mask(self) -> np.ndarray | None:
        return self._redlips_mask

    def clear_redlips_mask(self) -> None:
        self._redlips_mask = None

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _get_face_helper(self):
        if self._face_helper is None:
            import torch
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._face_helper = FaceRestoreHelper(
                upscale_factor=1, face_size=512, crop_ratio=(1, 1),
                det_model="retinaface_resnet50", save_ext="png",
                use_parse=False, device=device,
            )
        return self._face_helper

    def _get_landmarker(self):
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            if not os.path.isfile(_MEDIAPIPE_MODEL):
                raise FileNotFoundError(
                    f"Modèle MediaPipe introuvable : {_MEDIAPIPE_MODEL}"
                )
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=_MEDIAPIPE_MODEL
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.20,
                min_face_presence_confidence=0.20,
                min_tracking_confidence=0.20,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        return self._landmarker

    # ── Process ──────────────────────────────────────────────────────────────

    def process(self, img: np.ndarray, params: dict, context: dict):
        strength = float(params.get("strength", 1.0))
        profil   = str(params.get("profil", "auto"))

        if profil == "manuel":
            return self._process_manual(img, strength)

        faces = _detect_faces_retina(img, self._get_face_helper())
        img_h, img_w = img.shape[:2]

        result     = img.copy()
        detections = []

        if faces:
            for (x1, y1, x2, y2, _score) in faces:
                ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, img_w, img_h)
                crop = result[ey1:ey2, ex1:ex2]
                corrected_crop, infos = _process_face_crop(
                    crop, self._get_landmarker(), _UPSCALE, strength,
                )
                if any(d["corrected"] for d in infos):
                    result[ey1:ey2, ex1:ex2] = corrected_crop
                for d in infos:
                    detections.append({
                        "polygon":   [(ex1 + px, ey1 + py)
                                      for (px, py) in d["polygon"]],
                        "corrected": d["corrected"],
                    })
        else:
            # Fallback : MediaPipe sur l'image entière
            corrected_full, infos = _process_face_crop(
                img, self._get_landmarker(), 1, strength,
            )
            if any(d["corrected"] for d in infos):
                result = corrected_full
            detections.extend(infos)

        face_bboxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _s) in faces] if faces else []
        return result, {"redlips_detections": detections, "face_bboxes": face_bboxes}

    def _process_manual(self, img: np.ndarray, strength: float):
        """Mode manuel : chaque zone peinte est corrigée comme une bouche."""
        if self._redlips_mask is None or not self._redlips_mask.any():
            return img.copy(), {"redlips_detections": []}

        h, w = img.shape[:2]
        mask = self._redlips_mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        result = img.copy()
        detections = []
        bin_mask = (mask > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
        for lbl in range(1, n):
            bx = int(stats[lbl, cv2.CC_STAT_LEFT])
            by = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            # Crop expansé : marge pour l'anneau de peau étalon
            pad = max(bw, bh) // 2 + 8
            x1, y1 = max(0, bx - pad), max(0, by - pad)
            x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = result[y1:y2, x1:x2]
            comp = (labels[y1:y2, x1:x2] == lbl).astype(np.uint8) * 255
            corrected, was_red = _correct_lips_zone(crop, comp, strength)
            if was_red:
                result[y1:y2, x1:x2] = corrected
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                detections.append({
                    "polygon":   [(x1 + int(p[0][0]), y1 + int(p[0][1]))
                                  for p in cnt],
                    "corrected": was_red,
                })
        return result, {"redlips_detections": detections}

    # ── Détection pour l'onglet (mode manuel) ────────────────────────────────

    def detect_lips_mask(self, img: np.ndarray, margin: float = 0.12) -> np.ndarray | None:
        """Détecte les bouches et retourne le masque des lèvres (anneau dilaté).

        margin : dilatation relative à la largeur de la bouche, pour que le
        masque peint couvre bien le contour des lèvres.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        faces = _detect_faces_retina(img, self._get_face_helper())
        crops: list[tuple[int, int, np.ndarray, int]] = []
        if faces:
            for (x1, y1, x2, y2, _s) in faces:
                ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, w, h)
                crops.append((ex1, ey1, img[ey1:ey2, ex1:ex2], _UPSCALE))
        else:
            crops.append((0, 0, img, 1))

        for ox, oy, crop, upscale in crops:
            polys = _detect_lip_polygons(crop, self._get_landmarker(), upscale)
            if polys is None:
                continue
            outer, inner = polys
            zone = _lips_ring_mask(crop.shape[:2], outer, inner)
            lip_w = max(outer[:, 0].max() - outer[:, 0].min(), 8)
            r = max(2, int(lip_w * margin))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (2 * r + 1, 2 * r + 1))
            zone = cv2.dilate(zone, kernel)
            ch, cw = zone.shape[:2]
            mask[oy:oy + ch, ox:ox + cw] = cv2.bitwise_or(
                mask[oy:oy + ch, ox:ox + cw], zone)

        return mask if cv2.countNonZero(mask) > 0 else None


# ── Localisation des lèvres (landmarks) ──────────────────────────────────────

def _detect_lip_polygons(crop_bgr, landmarker, upscale):
    """Retourne (contour_externe, contour_interne) en coordonnées NATIVES du
    crop (la détection se fait sur le crop upscalé), ou None si pas de visage."""
    import mediapipe as mp

    ch, cw = crop_bgr.shape[:2]
    up = (cv2.resize(crop_bgr, (cw * upscale, ch * upscale),
                     interpolation=cv2.INTER_CUBIC)
          if upscale > 1 else crop_bgr)
    rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    if len(landmarks) <= max(max(_LIPS_OUTER), max(_LIPS_INNER)):
        return None
    uh, uw = up.shape[:2]

    def pts(indices):
        return np.array(
            [(landmarks[i].x * uw / upscale, landmarks[i].y * uh / upscale)
             for i in indices],
            dtype=np.float32,
        )

    return pts(_LIPS_OUTER), pts(_LIPS_INNER)


def _lips_ring_mask(shape_hw, outer, inner) -> np.ndarray:
    """Masque binaire de l'anneau des lèvres : polygone externe − interne."""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(outer).astype(np.int32)], 255)
    hole = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(hole, [np.round(inner).astype(np.int32)], 255)
    # Bouche fermée : le contour interne est quasi plat → l'érosion par le
    # « trou » est négligeable, l'anneau couvre toute la bouche.
    mask[hole > 0] = 0
    return mask


def _process_face_crop(crop_bgr, landmarker, upscale, strength):
    """Détecte et corrige les lèvres trop rouges dans un crop visage.

    Retourne (crop corrigé, infos) où infos est une liste de
    {"polygon": [(x, y), ...] natifs au crop, "corrected": bool}.
    """
    polys = _detect_lip_polygons(crop_bgr, landmarker, upscale)
    if polys is None:
        return crop_bgr, []

    outer, inner = polys
    lip_mask = _lips_ring_mask(crop_bgr.shape[:2], outer, inner)
    if cv2.countNonZero(lip_mask) < 12:
        return crop_bgr, []

    corrected, was_red = _correct_lips_zone(crop_bgr, lip_mask, strength)
    polygon = [(float(x), float(y)) for (x, y) in outer]
    return corrected, [{"polygon": polygon, "corrected": was_red}]


# ── Correction OKLab (partagée auto / manuel) ────────────────────────────────

def _correct_lips_zone(img_bgr: np.ndarray, lip_mask: np.ndarray,
                       strength: float) -> tuple[np.ndarray, bool]:
    """Ramène la chromaticité des lèvres vers une cible naturelle.

    La peau d'un anneau autour de la zone sert d'étalon : cible de chroma
    = chroma peau × ratio naturel. Seuls les pixels rouges (a > 0) dont le
    chroma dépasse la cible sont corrigés, luminance intacte. Retourne
    (image corrigée, lèvres jugées trop rouges ?) — si False, l'image est
    retournée inchangée.
    """
    if strength <= 0.0 or cv2.countNonZero(lip_mask) < 12:
        return img_bgr, False

    h, w = img_bgr.shape[:2]
    area = float(cv2.countNonZero(lip_mask))
    radius = max(2, int(np.sqrt(area) * 0.35))

    # Anneau de peau étalon autour de la bouche
    k_out = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (4 * radius + 1, 4 * radius + 1))
    k_in = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius + 1, radius + 1))
    ring = cv2.dilate(lip_mask, k_out)
    ring[cv2.dilate(lip_mask, k_in) > 0] = 0

    lab = _bgr_to_oklab(img_bgr)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Étalon peau : médiane de chroma/teinte, hors ombres et hautes lumières
    ring_sel = (ring > 0) & (L > 0.25) & (L < 0.92)
    if ring_sel.sum() < 30:
        ring_sel = ring > 0
    if ring_sel.sum() < 30:
        return img_bgr, False
    a_skin = float(np.median(a[ring_sel]))
    b_skin = float(np.median(b[ring_sel]))
    c_skin = float(np.hypot(a_skin, b_skin))

    c_target = max(c_skin * _LIP_CHROMA_RATIO, _MIN_TARGET_CHROMA)

    # Lèvres trop rouges ? (médiane du chroma des pixels rouges de la zone)
    lip_sel = (lip_mask > 0) & (a > 0.0)
    if lip_sel.sum() < 12:
        return img_bgr, False
    c_lips = float(np.median(np.hypot(a[lip_sel], b[lip_sel])))
    if c_lips < c_target * _EXCESS_GATE:
        return img_bgr, False

    # Transition douce aux bords de la zone
    feather = cv2.GaussianBlur(
        (lip_mask > 0).astype(np.float32), (0, 0), max(1.5, radius * 0.6))

    chroma = np.hypot(a, b)
    safe_c = np.maximum(chroma, 1e-6)
    # Cible par pixel : teinte du pixel rappelée vers celle de la peau,
    # chroma plafonné à la cible
    if c_skin > 1e-6:
        dir_a = (a / safe_c) * (1.0 - _HUE_PULL) + (a_skin / c_skin) * _HUE_PULL
        dir_b = (b / safe_c) * (1.0 - _HUE_PULL) + (b_skin / c_skin) * _HUE_PULL
    else:
        dir_a, dir_b = a / safe_c, b / safe_c
    norm = np.maximum(np.hypot(dir_a, dir_b), 1e-6)
    tgt_a = dir_a / norm * c_target
    tgt_b = dir_b / norm * c_target

    # Pixels concernés : rouges (a > 0) et au-dessus de la cible
    red_w = np.clip(a / 0.03, 0.0, 1.0)
    excess = (chroma > c_target).astype(np.float32)
    alpha = np.clip(feather * red_w * excess * strength, 0.0, 1.0)

    lab_out = lab.copy()
    lab_out[:, :, 1] = a + alpha * (tgt_a - a)
    lab_out[:, :, 2] = b + alpha * (tgt_b - b)

    out = _oklab_to_bgr(lab_out)
    # N'écrire que les pixels réellement corrigés (pas de dérive de
    # conversion aller-retour sur le reste du crop)
    changed = alpha > 1e-3
    result = img_bgr.copy()
    result[changed] = out[changed]
    return result, True
