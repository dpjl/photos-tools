"""steps/step_redeye.py — Étape 4 : correction yeux rouges (algorithme v2).

Architecture en deux phases :

1. **Détection des visages** via RetinaFace (facexlib FaceRestoreHelper)
   Retourne des boîtes englobantes → crop expansé de chaque visage.

2. **Localisation de l'iris et correction** sur le crop (3× upscale) :
   a. MediaPipe FaceLandmarker (indices iris 469-472, 474-477)
      → ellipse précise avec les 4 points du contour de l'iris
   b. Masque rouge : filtre HSV teinte rouge + score redness + conditions R/G/B
      + filtrage composantes connexes (taille + position)
   c. Correction gaussienne centrée sur le centroïde réel du cluster rouge :
      alpha = gauss(dist_centroïde, σ=iris_r) × red_mask × strength
      R_new = R × (1-alpha) + (G+B)/2 × alpha  [float32, pas de troncature]

Overlay : cercles sur les iris détectés (vert = corrigé, cyan = non rouge)

Fallback :
  · Aucun visage RetinaFace → MediaPipe sur l'image entière
  · MediaPipe échoue aussi → détection de blobs rouges heuristique
"""

from __future__ import annotations
import os
import cv2
import numpy as np

from steps.base import StepBase
from config import APP_DIR

# ── Constantes ───────────────────────────────────────────────────────────────
_MEDIAPIPE_MODEL = os.path.join(APP_DIR, "face_landmarker.task")
_MIN_IRIS_R      = 3.0
_MIN_RED_PIXELS  = 8
_MIN_RED_RATIO   = 0.012
_MIN_BLOB_AREA   = 10
_MAX_BLOB_AREA   = 5000
_EXPAND_MARGIN   = 0.90     # facteur d'expansion du crop visage
_UPSCALE         = 3        # upscale fixe avant MediaPipe (optimal)

# Indices MediaPipe FaceLandmarker pour les contours d'iris
_RIGHT_IRIS = [469, 470, 471, 472]
_LEFT_IRIS  = [474, 475, 476, 477]

# Contours de l'oeil (paupières) pour le profil étendu
_RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173,
                      133, 155, 154, 153, 145, 144, 163, 7]
_LEFT_EYE_CONTOUR  = [263, 466, 388, 387, 386, 385, 384, 398,
                      362, 382, 381, 380, 374, 373, 390, 249]


class RedEyeStep(StepBase):
    id                 = "redeye"
    name               = "Correction yeux rouges"
    short_name         = "YeuxRouges"
    slow               = True
    enabled_by_default = True
    has_overlay        = True

    param_defs = [
        {"key": "profil", "label": "Profil", "type": "choice",
         "as_buttons": True,
         "default": "étendu",
         "choices": ["classique", "étendu", "manuel"]},
        {"key": "strength", "label": "Force correction", "type": "float",
         "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
    ]

    def __init__(self):
        self._face_helper  = None
        self._landmarker   = None
        self._redeye_mask: np.ndarray | None = None  # masque peint (mode manuel)

    # ── API masque yeux rouges (appelée depuis l'UI) ──────────────────────────

    def set_redeye_mask(self, mask: np.ndarray | None) -> None:
        self._redeye_mask = mask.copy() if mask is not None else None

    def get_redeye_mask(self) -> np.ndarray | None:
        return self._redeye_mask

    def clear_redeye_mask(self) -> None:
        self._redeye_mask = None

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
        profil   = str(params.get("profil", "étendu"))
        img_h, img_w = img.shape[:2]

        # ── Mode manuel : appliquer la correction sur le masque peint ─────────
        if profil == "manuel":
            return self._process_manual(img, strength)

        # ── Détection des visages via RetinaFace ──────────────────────────────
        faces = _detect_faces_retina(img, self._get_face_helper())

        result     = img.copy()
        detections = []

        if faces:
            # Traitement par crop de visage
            for (x1, y1, x2, y2, _score) in faces:
                ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, img_w, img_h)
                crop = result[ey1:ey2, ex1:ex2].copy()

                corrected_crop, eye_infos = _process_face_crop(
                    crop, self._get_landmarker(), _UPSCALE, strength,
                    profil,
                )

                if any(d["corrected"] for d in eye_infos):
                    result[ey1:ey2, ex1:ex2] = corrected_crop

                # Convertir coordonnées upscalées → espace image originale
                for d in eye_infos:
                    ux, uy, ur = d["iris_up"]
                    detections.append({
                        "iris":      (ex1 + ux / _UPSCALE,
                                      ey1 + uy / _UPSCALE,
                                      ur / _UPSCALE),
                        "corrected": d["corrected"],
                    })

        else:
            # Fallback 1 : MediaPipe sur l'image entière (upscale=1)
            corrected_full, eye_infos = _process_face_crop(
                img, self._get_landmarker(), 1, strength, profil,
            )
            if any(d["corrected"] for d in eye_infos):
                result = corrected_full
                for d in eye_infos:
                    ux, uy, ur = d["iris_up"]
                    detections.append({
                        "iris":      (float(ux), float(uy), float(ur)),
                        "corrected": d["corrected"],
                    })
            else:
                # Fallback 2 : blobs rouges heuristiques
                for (cx, cy, ir) in _fallback_blobs(img):
                    ok, fcx, fcy = _correct_blob(result, cx, cy, ir, strength)
                    detections.append({
                        "iris":      (float(fcx), float(fcy), float(ir)),
                        "corrected": ok,
                    })

        # Partager les bboxes dans le contexte pour les étapes suivantes (ex. SCUNet)
        face_bboxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _s) in faces] if faces else []
        return result, {"redeye_detections": detections, "face_bboxes": face_bboxes}

    def _process_manual(self, img: np.ndarray, strength: float):
        """Mode manuel : correction des pixels marqués dans le masque peint."""
        if self._redeye_mask is None or not self._redeye_mask.any():
            return img.copy(), {"redeye_detections": []}

        h, w = img.shape[:2]
        mask = self._redeye_mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        result = _correct_pixels_gaussian(
            img.copy(), mask,
            iris_center_xy=np.array([0.0, 0.0]),  # non utilisé (centroïde auto)
            iris_radius=max(h, w),                 # pas de cutoff
            strength=strength,
        )
        red_px = int((mask > 0).sum())
        return result, {"redeye_detections": [{
            "iris": (w / 2.0, h / 2.0, 0.0),
            "corrected": red_px > 0,
        }]}


# ── Détection visages ─────────────────────────────────────────────────────────

def _detect_faces_retina(img: np.ndarray, face_helper) -> list:
    """Retourne [(x1,y1,x2,y2,score), ...] via RetinaFace (facexlib)."""
    h, w = img.shape[:2]
    try:
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    except Exception:
        return []
    if not face_helper.det_faces:
        return []
    result = []
    for det in face_helper.det_faces:
        det = np.array(det).flatten()
        if len(det) < 5:
            continue
        x1, y1, x2, y2, score = det[0], det[1], det[2], det[3], det[4]
        x1 = max(0, int(round(float(x1))))
        y1 = max(0, int(round(float(y1))))
        x2 = min(w, int(round(float(x2))))
        y2 = min(h, int(round(float(y2))))
        if x2 > x1 and y2 > y1:
            result.append((x1, y1, x2, y2, float(score)))
    return result


def _expand_box(x1, y1, x2, y2, img_w, img_h, margin=_EXPAND_MARGIN):
    """Agrandit une boîte visage pour inclure le contexte autour des yeux."""
    bw   = x2 - x1
    bh   = y2 - y1
    cx   = x1 + bw / 2.0
    cy   = y1 + bh / 2.0
    size = max(bw, bh) * (1.0 + margin)
    nx1  = max(0,     int(round(cx - size / 2.0)))
    ny1  = max(0,     int(round(cy - size / 2.0)))
    nx2  = min(img_w, int(round(cx + size / 2.0)))
    ny2  = min(img_h, int(round(cy + size / 2.0)))
    return nx1, ny1, nx2, ny2


# ── Traitement d'un crop visage ───────────────────────────────────────────────

def _process_face_crop(crop_bgr, landmarker, upscale, strength, profil="étendu"):
    """Détecte et corrige les yeux rouges dans un crop visage.

    Retourne (corrected_crop, eye_infos) où eye_infos est une liste de :
        {"iris_up": (cx, cy, radius),   # dans l'espace du crop UPSCALÉ
         "corrected": bool}
    """
    import mediapipe as mp

    crop_h, crop_w = crop_bgr.shape[:2]

    if upscale > 1:
        up_bgr = cv2.resize(
            crop_bgr, (crop_w * upscale, crop_h * upscale),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        up_bgr = crop_bgr.copy()

    up_rgb    = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=up_rgb)
    lm_result = landmarker.detect(mp_image)

    if not lm_result.face_landmarks:
        return crop_bgr, []

    corrected_up = up_bgr.copy()
    eye_infos    = []

    use_extended = (profil == "étendu")

    for landmarks in lm_result.face_landmarks[:1]:
        eye_defs = (
            ("right", _RIGHT_IRIS, _RIGHT_EYE_CONTOUR),
            ("left",  _LEFT_IRIS,  _LEFT_EYE_CONTOUR),
        )
        for eye_name, iris_indices, eye_contour_indices in eye_defs:
            iris_mask, iris_points = _iris_mask_from_landmarks(
                corrected_up.shape, landmarks, iris_indices,
            )
            if iris_mask is None:
                continue

            # Géométrie de l'iris dans le crop upscalé
            pts_f    = iris_points.astype(np.float32)
            center   = pts_f.mean(axis=0)                              # [cx, cy]
            iris_w   = pts_f[:, 0].max() - pts_f[:, 0].min()
            iris_h   = pts_f[:, 1].max() - pts_f[:, 1].min()
            iris_r   = max(_MIN_IRIS_R * upscale, (iris_w + iris_h) / 4.0)

            if use_extended:
                corr_mask, stats = _red_eye_mask_extended(
                    corrected_up, landmarks, iris_indices, eye_contour_indices,
                    iris_points, upscale,
                )
            else:
                corr_mask, stats = _red_eye_mask(corrected_up, iris_mask, iris_points)

            is_red = (
                stats["red_pixels"] >= _MIN_RED_PIXELS
                and stats["red_ratio"] >= _MIN_RED_RATIO
                and stats["components"] > 0
            )

            corrected = False
            if is_red:
                if use_extended:
                    corrected_up = _correct_pixels_smooth(
                        corrected_up, corr_mask, iris_points,
                        eye_contour_indices, landmarks, iris_r, strength,
                    )
                else:
                    corrected_up = _correct_pixels_gaussian(
                        corrected_up, corr_mask, center, iris_r, strength,
                    )
                corrected = True

            eye_infos.append({
                "iris_up":   (float(center[0]), float(center[1]), float(iris_r)),
                "corrected": corrected,
            })

    if upscale > 1:
        corrected_crop = cv2.resize(
            corrected_up, (crop_w, crop_h), interpolation=cv2.INTER_AREA,
        )
    else:
        corrected_crop = corrected_up

    return corrected_crop, eye_infos


# ── Utilitaires landmarks ─────────────────────────────────────────────────────

def _iris_mask_from_landmarks(img_shape, landmarks, iris_indices):
    """Crée un masque ellipse de l'iris depuis les 4 points de contour MediaPipe."""
    h, w = img_shape[:2]
    if len(landmarks) <= max(iris_indices):
        return None, None

    points = np.array(
        [(int(round(landmarks[i].x * w)), int(round(landmarks[i].y * h)))
         for i in iris_indices],
        dtype=np.int32,
    )
    center   = points.mean(axis=0).astype(int)
    radius_x = max(int((points[:, 0].max() - points[:, 0].min()) / 2), 2)
    radius_y = max(int((points[:, 1].max() - points[:, 1].min()) / 2), 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, tuple(center), (radius_x, radius_y), 0, 0, 360, 255, -1)
    return mask, points


# ── Détection du rouge ────────────────────────────────────────────────────────

def _red_eye_mask(img_bgr, iris_mask, iris_points):
    """Masque rouge multi-condition : HSV + redness + composantes connexes."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img_bgr)

    pts_f       = np.asarray(iris_points, dtype=np.float32)
    center      = pts_f.mean(axis=0)
    iris_w      = pts_f[:, 0].max() - pts_f[:, 0].min()
    iris_h      = pts_f[:, 1].max() - pts_f[:, 1].min()
    iris_radius = max(2.0, (iris_w + iris_h) / 4.0)

    r_f = r.astype(np.float32)
    g_f = g.astype(np.float32)
    b_f = b.astype(np.float32)
    redness = r_f - 0.55 * g_f - 0.45 * b_f

    red_hue = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   45, 30]), np.array([12,  255, 255])),
        cv2.inRange(hsv, np.array([168, 45, 30]), np.array([180, 255, 255])),
    )

    yy, xx = np.indices(iris_mask.shape)
    dist   = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)

    candidate = (
        (red_hue > 0)
        & (redness > 25)
        & (r_f > 65)
        & (r_f > g_f * 1.25)
        & (r_f > b_f * 1.25)
        & (iris_mask > 0)
        & (dist <= iris_radius * 0.85)
    )

    mask   = candidate.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, comp_stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    final_mask      = np.zeros_like(mask)
    kept_components = 0

    for lbl in range(1, n):
        area = comp_stats[lbl, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        cx_c, cy_c  = centroids[lbl]
        blob_dist   = np.sqrt((cx_c - center[0]) ** 2 + (cy_c - center[1]) ** 2)
        if blob_dist > iris_radius * 0.55:
            continue
        if area > np.pi * (iris_radius ** 2) * 0.85:
            continue
        final_mask[labels == lbl] = 255
        kept_components += 1

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    iris_area  = max(1, cv2.countNonZero(iris_mask))
    final_area = cv2.countNonZero(final_mask)

    return final_mask, {
        "red_pixels": int(final_area),
        "iris_pixels": int(iris_area),
        "red_ratio": float(final_area / iris_area),
        "components": int(kept_components),
    }


# ── Correction gaussienne (algorithme v2) ────────────────────────────────────

def _correct_pixels_gaussian(img_bgr, mask, iris_center_xy, iris_radius, strength):
    """Correction avec poids gaussien centré sur le centroïde réel du rouge.

    alpha = gauss(dist_centroïde, σ=iris_radius) × red_mask × strength
    R_new = R × (1-alpha) + (G+B)/2 × alpha   [float32]
    """
    if cv2.countNonZero(mask) < 4:
        return img_bgr

    h, w = img_bgr.shape[:2]
    b_f, g_f, r_f = [c.astype(np.float32) for c in cv2.split(img_bgr)]
    yy, xx = np.mgrid[:h, :w]

    red_float = (mask > 0).astype(np.float32)
    total_red = float(red_float.sum())

    # Centroïde réel du cluster rouge (peut différer légèrement du centre iris)
    centroid_x = float((red_float * xx).sum() / total_red)
    centroid_y = float((red_float * yy).sum() / total_red)

    # Gaussienne σ = iris_radius, cutoff dur à iris_radius
    dist  = np.sqrt((xx - centroid_x) ** 2 + (yy - centroid_y) ** 2)
    sigma = max(float(iris_radius), 1.0)
    gauss = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
    gauss[dist > iris_radius] = 0.0

    # Double condition : rouge ET proche du centre
    alpha = np.clip(gauss * red_float * strength, 0.0, 1.0)

    R_nat = (g_f + b_f) * 0.5
    R_new = r_f * (1.0 - alpha) + R_nat * alpha

    return cv2.merge([
        b_f.astype(np.uint8),
        g_f.astype(np.uint8),
        np.clip(R_new, 0, 255).astype(np.uint8),
    ])


# ── Profil étendu : zone élargie + centralité ────────────────────────────────

_EXPAND_IRIS = 1.8   # facteur d'expansion de l'iris pour la zone de recherche

def _lm_points(landmarks, indices, w, h):
    """Extrait les coordonnées pixel d'un ensemble de landmarks."""
    return np.array(
        [(int(round(landmarks[i].x * w)), int(round(landmarks[i].y * h)))
         for i in indices if i < len(landmarks)],
        dtype=np.int32,
    )


def _red_eye_mask_extended(img_bgr, landmarks, iris_indices, eye_contour_indices,
                           iris_points, upscale):
    """Masque rouge étendu : zone élargie ∩ contour oeil + centralité.

    Différences par rapport à _red_eye_mask :
      · zone = iris × 1.8 ∩ polygone paupières (au lieu de iris seul + dist)
      · morpho close 5×5 pour connecter les fragments
      · seules les composantes connectées au tiers central de l'iris sont gardées
    """
    h, w = img_bgr.shape[:2]

    pts_f       = iris_points.astype(np.float32)
    center      = pts_f.mean(axis=0)
    iris_w      = pts_f[:, 0].max() - pts_f[:, 0].min()
    iris_h      = pts_f[:, 1].max() - pts_f[:, 1].min()
    rx          = max(int(iris_w / 2), 2)
    ry          = max(int(iris_h / 2), 2)
    iris_radius = max(2.0, (iris_w + iris_h) / 4.0)

    # Zone de recherche : iris élargi ∩ contour oeil
    erx, ery   = int(rx * _EXPAND_IRIS), int(ry * _EXPAND_IRIS)
    search_r   = iris_radius * _EXPAND_IRIS
    center_int = tuple(center.astype(int))

    search_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(search_mask, center_int, (erx, ery), 0, 0, 360, 255, -1)

    eye_pts = _lm_points(landmarks, eye_contour_indices, w, h)
    eye_mask = None
    if len(eye_pts) >= 6:
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(eye_mask, [eye_pts], 255)
        search_mask = cv2.bitwise_and(search_mask, eye_mask)

    # Filtres couleur identiques au profil classique
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    r_f = img_bgr[:, :, 2].astype(np.float32)
    g_f = img_bgr[:, :, 1].astype(np.float32)
    b_f = img_bgr[:, :, 0].astype(np.float32)
    redness = r_f - 0.55 * g_f - 0.45 * b_f

    red_hue = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   45, 30]), np.array([12,  255, 255])),
        cv2.inRange(hsv, np.array([168, 45, 30]), np.array([180, 255, 255])),
    )

    candidate = (
        (red_hue > 0)
        & (redness > 25)
        & (r_f > 65)
        & (r_f > g_f * 1.25)
        & (r_f > b_f * 1.25)
        & (search_mask > 0)
    )

    mask   = candidate.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask   = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    # Seed au tiers central de l'iris
    seed_rx = max(int(rx / 3), 2)
    seed_ry = max(int(ry / 3), 2)
    central_seed = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(central_seed, center_int, (seed_rx, seed_ry), 0, 0, 360, 255, -1)

    n, labels, comp_stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    final_mask      = np.zeros_like(mask)
    kept_components = 0

    for lbl in range(1, n):
        area = comp_stats[lbl, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        if area > np.pi * (search_r ** 2) * 0.85:
            continue
        component = (labels == lbl).astype(np.uint8) * 255
        if cv2.countNonZero(cv2.bitwise_and(component, central_seed)) > 0:
            final_mask[labels == lbl] = 255
            kept_components += 1

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    if eye_mask is not None:
        final_mask = cv2.bitwise_and(final_mask, eye_mask)

    search_area = max(1, cv2.countNonZero(search_mask))
    final_area  = cv2.countNonZero(final_mask)

    return final_mask, {
        "red_pixels":  int(final_area),
        "iris_pixels": int(search_area),
        "red_ratio":   float(final_area / search_area),
        "components":  int(kept_components),
    }


def _correct_pixels_smooth(img_bgr, mask, iris_points, eye_contour_indices,
                           landmarks, iris_radius, strength):
    """Correction avec masque lissé et pondération par redness.

    Le masque binaire est dilaté puis flouté (GaussianBlur) pour une transition
    douce aux bords.  L'intensité de correction est proportionnelle au score
    de redness du pixel (les pixels les plus rouges sont corrigés plus fort).
    """
    if cv2.countNonZero(mask) < 4:
        return img_bgr

    h, w = img_bgr.shape[:2]

    # Construire le masque du contour de l'oeil pour borner le lissage
    eye_pts  = _lm_points(landmarks, eye_contour_indices, w, h)
    eye_mask = None
    if len(eye_pts) >= 6:
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(eye_mask, [eye_pts], 255)

    # Dilater + Gaussian blur → transition douce
    smooth_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smooth_mask = cv2.dilate(mask, smooth_k, iterations=1)
    if eye_mask is not None:
        smooth_mask = cv2.bitwise_and(smooth_mask, eye_mask)
    blur_size = max(3, int(iris_radius * 0.3) | 1)
    alpha_map = cv2.GaussianBlur(
        smooth_mask.astype(np.float32) / 255.0,
        (blur_size, blur_size), 0,
    )
    if eye_mask is not None:
        alpha_map[eye_mask == 0] = 0.0

    total = float(alpha_map.sum())
    if total < 1.0:
        return img_bgr

    b_f, g_f, r_f = [c.astype(np.float32) for c in cv2.split(img_bgr)]
    yy, xx = np.mgrid[:h, :w]

    centroid_x = float((alpha_map * xx).sum() / total)
    centroid_y = float((alpha_map * yy).sum() / total)

    red_radius = np.sqrt(total / np.pi)
    sigma = max(float(red_radius), 1.0)

    dist  = np.sqrt((xx - centroid_x) ** 2 + (yy - centroid_y) ** 2)
    gauss = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)

    # Pondération par redness normalisée
    redness      = r_f - 0.55 * g_f - 0.45 * b_f
    redness_pos  = np.clip(redness, 0, None)
    red_max      = redness_pos[alpha_map > 0.1].max() if (alpha_map > 0.1).any() else 1.0
    redness_norm = np.clip(redness_pos / max(red_max, 1.0), 0, 1)
    redness_wt   = 0.6 + 0.4 * redness_norm       # ∈ [0.6, 1.0]

    alpha = np.clip(alpha_map * gauss * redness_wt * strength * 1.2, 0.0, 1.0)

    R_nat = (g_f + b_f) * 0.5
    R_new = r_f * (1.0 - alpha) + R_nat * alpha

    return cv2.merge([
        b_f.astype(np.uint8),
        g_f.astype(np.uint8),
        np.clip(R_new, 0, 255).astype(np.uint8),
    ])


# ── Fallbacks ─────────────────────────────────────────────────────────────────

def _fallback_blobs(img: np.ndarray) -> list:
    """Détecte les blobs rouges plausibles (fallback sans détection de visage)."""
    B, G, R = cv2.split(img)
    red_mask = (
        (R.astype(np.int32) > G.astype(np.int32) * 1.5) &
        (R.astype(np.int32) > B.astype(np.int32) * 1.5) &
        (R > 80)
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


def _correct_blob(result: np.ndarray, cx, cy, ir, strength) -> tuple:
    """Corrige un blob rouge heuristique avec le même algorithme gaussien."""
    h, w    = result.shape[:2]
    search  = ir * 2
    margin  = int(search) + 2
    x1 = max(0,  int(cx) - margin)
    y1 = max(0,  int(cy) - margin)
    x2 = min(w,  int(cx) + margin)
    y2 = min(h,  int(cy) + margin)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False, cx, cy

    crop          = result[y1:y2, x1:x2].copy()
    B_c, G_c, R_c = cv2.split(crop)
    lcx, lcy      = cx - x1, cy - y1
    ch, cw        = crop.shape[:2]
    yy, xx        = np.mgrid[:ch, :cw]
    dist_c        = np.sqrt((xx - lcx) ** 2 + (yy - lcy) ** 2)

    red_mask_c = (
        (R_c.astype(np.int32) > G_c.astype(np.int32) * 1.5) &
        (R_c.astype(np.int32) > B_c.astype(np.int32) * 1.5) &
        (R_c > 80) &
        (dist_c <= search)
    ).astype(np.uint8) * 255

    if cv2.countNonZero(red_mask_c) < 4:
        return False, cx, cy

    center_c    = np.array([lcx, lcy])
    fixed_crop  = _correct_pixels_gaussian(crop, red_mask_c, center_c, ir, strength)
    result[y1:y2, x1:x2] = fixed_crop
    return True, cx, cy
