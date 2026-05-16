"""steps/step_color.py — Étape 1 : correction couleur (histogramme + CLAHE + saturation)."""

from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from steps.base import StepBase


class ColorStep(StepBase):
    id         = "color"
    name       = "1 · Correction couleur"
    short_name = "Couleur"
    slow       = False

    param_defs = [
        {"key": "low_pct",    "label": "Pct. bas histogramme",  "type": "float",
         "default": 1.0,  "min": 0.0,  "max": 5.0,   "step": 0.1},
        {"key": "high_pct",   "label": "Pct. haut histogramme", "type": "float",
         "default": 99.0, "min": 94.0, "max": 100.0, "step": 0.1},
        {"key": "roi_inset",  "label": "Marge ROI (fraction)",  "type": "float",
         "default": 0.06, "min": 0.0,  "max": 0.2,   "step": 0.01},
        {"key": "clip_limit", "label": "CLAHE — contraste",     "type": "float",
         "default": 2.5,  "min": 0.5,  "max": 10.0,  "step": 0.1},
        {"key": "sat_factor", "label": "Saturation (×)",        "type": "float",
         "default": 1.35, "min": 0.5,  "max": 2.5,   "step": 0.05},
    ]

    def process(self, img: np.ndarray, params: dict, context: dict):
        result = _stretch_histogram(
            img,
            low_pct=params["low_pct"],
            high_pct=params["high_pct"],
            roi_inset=params["roi_inset"],
        )
        result = _apply_clahe(result, clip_limit=params["clip_limit"])
        result = _boost_saturation(result, factor=params["sat_factor"])
        return result, {}


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions de traitement internes
# ──────────────────────────────────────────────────────────────────────────────

def _stretch_histogram(img: np.ndarray, low_pct=1.0, high_pct=99.0, roi_inset=0.06) -> np.ndarray:
    """Étire l'histogramme sur la région centrale (évite les bords sombres)."""
    h, w = img.shape[:2]
    y0 = int(h * roi_inset)
    y1 = h - y0
    x0 = int(w * roi_inset)
    x1 = w - x0
    roi = img[y0:y1, x0:x1]

    result = img.copy().astype(np.float32)
    for c in range(3):
        lo = float(np.percentile(roi[:, :, c], low_pct))
        hi = float(np.percentile(roi[:, :, c], high_pct))
        if hi > lo:
            result[:, :, c] = np.clip((result[:, :, c] - lo) / (hi - lo) * 255, 0, 255)
    return result.astype(np.uint8)


def _apply_clahe(img: np.ndarray, clip_limit=2.5) -> np.ndarray:
    """CLAHE en espace LAB pour améliorer le contraste local."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _boost_saturation(img: np.ndarray, factor=1.35) -> np.ndarray:
    """Augmente la saturation en espace HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
