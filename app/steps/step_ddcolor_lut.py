"""Etape DDColor LUT: restauration couleur IA guidee par DDColor.

Cette step reprend la transformation experimentee dans
do-not-commit/bleu-violet-3/run10_ddcolor_modelscope_lut:
  1. DDColor modelscope propose une chrominance depuis la luminance.
  2. Une fusion chroma selective limite les hallucinations du modele.
  3. Une LUT polynomiale contrainte applique la palette de facon globale.

L'etape est marquee previewable et integree au groupe rapide. Le modele reste
charge paresseusement; les sorties DDColor sont cachees par image d'entree pour
eviter de relancer le reseau quand seuls les curseurs de finition changent.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from typing import Any

import cv2
import numpy as np

from steps.base import StepBase


class DDColorLUTStep(StepBase):
    id = "ddcolor_lut"
    name = "Couleur IA DDColor LUT"
    short_name = "DDColor"
    slow = False
    enabled_by_default = False
    previewable = True

    param_defs = [
        {"key": "taille_ia", "label": "Résolution IA", "type": "choice",
         "as_buttons": True, "default": "512", "choices": ["384", "512", "768"]},
        {"key": "force", "label": "Force", "type": "float",
         "default": 1.0, "min": 0.0, "max": 1.25, "step": 0.05},
        {"key": "influence_ia", "label": "Influence IA", "type": "float",
         "default": 1.0, "min": 0.4, "max": 1.4, "step": 0.05},
        {"key": "vibrance", "label": "Vibrance", "type": "float",
         "default": 0.16, "min": 0.0, "max": 0.35, "step": 0.02},
    ]

    def __init__(self) -> None:
        self._model = None
        self._pipeline = None
        self._device = None
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._lut_cache: OrderedDict[tuple[str, int, float, float], tuple[np.ndarray, dict, np.ndarray]] = OrderedDict()

    def process(self, img: np.ndarray, params: dict, context: dict):
        input_size = int(params.get("taille_ia", "512"))
        force = float(params.get("force", 1.0))
        influence = float(params.get("influence_ia", 1.0))
        vibrance = float(params.get("vibrance", 0.16))

        digest = _image_digest(img)
        ai, cached_ai = self._colorize_cached(img, input_size, digest)
        lut_key = (digest, input_size, round(influence, 4), round(vibrance, 4))
        cached_lut = self._lut_cache.get(lut_key)
        if cached_lut is None:
            restored, meta, weight = _ddcolor_learned_lut(
                img,
                ai,
                ai_weight_scale=influence,
                final_vibrance=vibrance,
            )
            self._lut_cache[lut_key] = (restored.copy(), dict(meta), weight.copy())
            while len(self._lut_cache) > 3:
                self._lut_cache.popitem(last=False)
            cached_lut_hit = False
        else:
            self._lut_cache.move_to_end(lut_key)
            restored, meta, weight = cached_lut
            restored = restored.copy()
            meta = dict(meta)
            weight = weight.copy()
            cached_lut_hit = True

        out = _blend_lab(img, restored, force)
        extras = {
            "ddcolor_lut": {
                "input_size": input_size,
                "force": force,
                "influence_ia": influence,
                "vibrance": vibrance,
                "weight_mean": float(weight.mean()),
                "weight_p95": float(np.percentile(weight, 95)),
                "skin_coverage": meta.get("skin_coverage"),
                "neutral_coverage": meta.get("neutral_coverage"),
                "cached_ai": cached_ai,
                "cached_lut": cached_lut_hit,
            }
        }
        return out, extras

    def _colorize_cached(self, img: np.ndarray, input_size: int, digest: str) -> tuple[np.ndarray, bool]:
        key = (digest, input_size)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached.copy(), True

        pipeline = self._get_pipeline(input_size)
        ai = pipeline.process(img)
        self._cache[key] = ai.copy()
        while len(self._cache) > 3:
            self._cache.popitem(last=False)
        return ai, False

    def _get_pipeline(self, input_size: int):
        if self._model is None:
            self._model, self._device = _load_ddcolor_model()

        from ddcolor_local import ColorizationPipeline

        if self._pipeline is None or self._pipeline.input_size != input_size:
            self._pipeline = ColorizationPipeline(
                self._model,
                input_size=input_size,
                device=self._device,
            )
        return self._pipeline


def _load_ddcolor_model():
    try:
        import torch
        from huggingface_hub import PyTorchModelHubMixin
        from ddcolor_local import DDColor
        import config
    except Exception as exc:
        raise RuntimeError(
            "Dependances DDColor manquantes. Installe torch et huggingface_hub."
        ) from exc

    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config: dict | None = None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)

    os.makedirs(config.DDCOLOR_CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = DDColorHF.from_pretrained(
            config.DDCOLOR_MODEL_REPO,
            cache_dir=config.DDCOLOR_CACHE_DIR,
        )
    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger le modele DDColor modelscope. "
            f"Depot: {config.DDCOLOR_MODEL_REPO}"
        ) from exc
    model = model.to(device)
    model.eval()
    return model, device


def _image_digest(img: np.ndarray) -> str:
    arr = np.ascontiguousarray(img)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def _lab_float(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def _lab_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = _lab_float(img)
    return lab[:, :, 0], lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0


def _chroma_from_lab(lab: np.ndarray) -> np.ndarray:
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    return np.sqrt(a * a + b * b)


def _neutral_candidate_mask(img: np.ndarray, inset: float = 0.04) -> np.ndarray:
    l, a, b = _lab_channels(img)
    chroma = np.sqrt(a * a + b * b)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    h, w = gray.shape
    border = np.ones_like(gray, dtype=bool)
    dy, dx = int(h * inset), int(w * inset)
    border[:dy, :] = False
    border[-dy:, :] = False
    border[:, :dx] = False
    border[:, -dx:] = False

    c_thr = max(18.0, float(np.percentile(chroma[border], 38)))
    s_thr = max(45.0, float(np.percentile(sat[border], 42)))
    g_thr = max(12.0, float(np.percentile(grad[border], 62)))
    mask = (
        border
        & (l > 35)
        & (l < 235)
        & (chroma <= c_thr)
        & (sat <= s_thr)
        & (grad <= g_thr)
    )
    if mask.mean() < 0.03:
        mask = border & (l > 30) & (l < 240) & (chroma <= np.percentile(chroma[border], 50))
    return mask


def _tone_cast_curve(img: np.ndarray, mask: np.ndarray, bins: int = 18):
    l, a, b = _lab_channels(img)
    centers = np.linspace(25, 235, bins).astype(np.float32)
    curve_a = np.full(bins, np.nan, np.float32)
    curve_b = np.full(bins, np.nan, np.float32)
    for i in range(bins):
        lo = 0 if i == 0 else (centers[i - 1] + centers[i]) / 2.0
        hi = 255 if i == bins - 1 else (centers[i] + centers[i + 1]) / 2.0
        m = mask & (l >= lo) & (l < hi)
        if m.sum() >= 250:
            curve_a[i] = float(np.median(a[m]))
            curve_b[i] = float(np.median(b[m]))
    valid = ~np.isnan(curve_a)
    if valid.sum() < 3:
        ca = float(np.median(a[mask])) if mask.any() else float(np.median(a))
        cb = float(np.median(b[mask])) if mask.any() else float(np.median(b))
        curve_a[:] = ca
        curve_b[:] = cb
    else:
        curve_a = np.interp(centers, centers[valid], curve_a[valid]).astype(np.float32)
        curve_b = np.interp(centers, centers[valid], curve_b[valid]).astype(np.float32)
        curve_a = cv2.GaussianBlur(curve_a.reshape(1, -1), (1, 5), 0).ravel()
        curve_b = cv2.GaussianBlur(curve_b.reshape(1, -1), (1, 5), 0).ravel()
    return centers, curve_a, curve_b


def _apply_tone_curve_cast(img: np.ndarray, strength: float = 0.86):
    mask = _neutral_candidate_mask(img)
    l, a, b = _lab_channels(img)
    chroma = np.sqrt(a * a + b * b)
    centers, ca, cb = _tone_cast_curve(img, mask)
    field_a = np.interp(l.ravel(), centers, ca).reshape(l.shape).astype(np.float32)
    field_b = np.interp(l.ravel(), centers, cb).reshape(l.shape).astype(np.float32)
    color_protect = 1.0 - 0.28 * np.clip((chroma - 34.0) / 50.0, 0.0, 1.0)
    highlight_protect = np.clip((245.0 - l) / 45.0, 0.25, 1.0)
    weight = strength * color_protect * highlight_protect

    lab = _lab_float(img)
    lab[:, :, 1] = np.clip(lab[:, :, 1] - field_a * weight, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] - field_b * weight, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    meta = {
        "neutral_coverage": float(mask.mean()),
        "curve_L": [float(x) for x in centers],
        "curve_a": [float(x) for x in ca],
        "curve_b": [float(x) for x in cb],
    }
    return out, meta, mask


def _luminance_stretch_lab(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    lab = _lab_float(img)
    l = lab[:, :, 0]
    p_lo = float(np.percentile(l, lo))
    p_hi = float(np.percentile(l, hi))
    if p_hi > p_lo:
        lab[:, :, 0] = np.clip((l - p_lo) * 255.0 / (p_hi - p_lo), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _contrast_curve_l_channel(img: np.ndarray) -> np.ndarray:
    lab = _lab_float(img)
    x = lab[:, :, 0] / 255.0
    y = x + 0.015 * (1.0 - x) * (1.0 - x) - 0.01 * x * x
    y = np.power(np.clip(y, 0, 1), 1.0 / 1.02)
    lab[:, :, 0] = np.clip(y * 255.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _clahe_l(img: np.ndarray, clip: float = 0.8) -> np.ndarray:
    if clip <= 0:
        return img.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _chroma_expand(img: np.ndarray, scale: float = 1.15) -> np.ndarray:
    lab = _lab_float(img)
    lab[:, :, 1] = np.clip(128.0 + (lab[:, :, 1] - 128.0) * scale, 0, 255)
    lab[:, :, 2] = np.clip(128.0 + (lab[:, :, 2] - 128.0) * scale, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _vibrance_lab(img: np.ndarray, strength: float = 0.28) -> np.ndarray:
    if strength <= 0:
        return img.copy()
    lab = _lab_float(img)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    c = np.sqrt(a * a + b * b)
    scale = 1.0 + strength * np.exp(-c / 42.0)
    lab[:, :, 1] = np.clip(128.0 + a * scale, 0, 255)
    lab[:, :, 2] = np.clip(128.0 + b * scale, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _midtone_warmth(img: np.ndarray, b_shift: float = 4.0, a_shift: float = -0.6) -> np.ndarray:
    lab = _lab_float(img)
    l = lab[:, :, 0]
    weight = np.clip((l - 28.0) / 70.0, 0.0, 1.0) * np.clip((220.0 - l) / 70.0, 0.0, 1.0)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift * weight, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift * weight, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _finishing(img: np.ndarray) -> np.ndarray:
    out = _luminance_stretch_lab(img, 1.0, 99.0)
    out = _contrast_curve_l_channel(out)
    out = _clahe_l(out, 0.55)
    out = _vibrance_lab(out, 0.16)
    out = _chroma_expand(out, 1.03)
    return out


def _detect_skin_mask(img: np.ndarray) -> np.ndarray:
    lab = _lab_float(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    hue_skin = (h <= 27) | (h >= 168)
    mask = hue_skin & (s >= 22) & (s <= 175) & (v >= 45) & (v <= 245) & (a > 1.5) & (b > 0.0)
    chroma = np.sqrt(a * a + b * b)
    mask &= chroma < 45.0
    if mask.mean() < 0.005:
        mask = hue_skin & (s >= 16) & (s <= 205) & (v >= 42) & (v <= 248) & (a > 0.0)
    m = mask.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m > 0


def _apply_skin_memory_soft(img: np.ndarray):
    prelim, tone_meta, _ = _apply_tone_curve_cast(img, strength=0.86)
    prelim = _finishing(prelim)
    skin = _detect_skin_mask(prelim)
    lab = _lab_float(prelim)
    l = lab[:, :, 0]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    meta: dict[str, Any] = {"skin_coverage": float(skin.mean()), **tone_meta}
    if skin.sum() >= 800:
        skin_a = float(np.median(a[skin]))
        skin_b = float(np.median(b[skin]))
        da = float(np.clip(14.0 - skin_a, -6.0, 6.0))
        db = float(np.clip(11.0 - skin_b, 0.0, 7.0))
        if da < 0:
            da *= 0.20
        tone_weight = np.clip((l - 35.0) / 70.0, 0.0, 1.0) * np.clip((220.0 - l) / 75.0, 0.0, 1.0)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + da * 0.58 * tone_weight, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + db * 0.68 * tone_weight, 0, 255)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        meta.update({"skin_a": skin_a, "skin_b": skin_b, "skin_da": da, "skin_db": db})
    else:
        out = prelim
        meta.update({"skin_a": None, "skin_b": None, "skin_da": 0.0, "skin_db": 0.0})

    out = _midtone_warmth(out, b_shift=2.6, a_shift=-0.25)
    out = _vibrance_lab(out, strength=0.28)
    out = _chroma_expand(out, scale=1.04)
    return out, meta, skin


def _suspect_cast_weight(original: np.ndarray, prelim: np.ndarray, ai: np.ndarray, scale: float) -> np.ndarray:
    l, a, b = _lab_channels(original)
    chroma = np.sqrt(a * a + b * b)
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    neutral = _neutral_candidate_mask(original)
    skin = _detect_skin_mask(prelim)
    cast_projection = a * 0.58 + (-b) * 0.82
    cast_weight = np.clip((cast_projection - 4.0) / 24.0, 0.0, 1.0)
    blue_violet_hue = ((h >= 95) & (h <= 155)).astype(np.float32)
    low_mid_sat = np.clip((150.0 - s) / 115.0, 0.0, 1.0)
    tone = np.clip((v - 35.0) / 55.0, 0.0, 1.0) * np.clip((238.0 - v) / 55.0, 0.0, 1.0)

    weight = 0.14 + 0.30 * cast_weight * low_mid_sat * tone
    weight += 0.28 * neutral.astype(np.float32)
    weight += 0.22 * blue_violet_hue * low_mid_sat * tone
    weight += 0.22 * skin.astype(np.float32)
    weight *= 1.0 - 0.45 * np.clip((chroma - 32.0) / 45.0, 0.0, 1.0)
    ai_c = _chroma_from_lab(_lab_float(ai))
    weight *= 1.0 - 0.30 * np.clip((ai_c - 48.0) / 35.0, 0.0, 1.0)
    weight = cv2.GaussianBlur((weight * scale).astype(np.float32), (0, 0), 5.0)
    return np.clip(weight, 0.05, 0.72)


def _ddcolor_chroma_fusion(original: np.ndarray, ai: np.ndarray, ai_weight_scale: float):
    prelim, meta, _skin = _apply_skin_memory_soft(original)
    w = _suspect_cast_weight(original, prelim, ai, ai_weight_scale)
    lab_pre = _lab_float(prelim)
    lab_ai = _lab_float(ai)
    lab_out = lab_pre.copy()
    lab_out[:, :, 1] = lab_pre[:, :, 1] * (1.0 - w) + lab_ai[:, :, 1] * w
    lab_out[:, :, 2] = lab_pre[:, :, 2] * (1.0 - w) + lab_ai[:, :, 2] * w
    out = cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = _vibrance_lab(out, strength=0.18)
    out = _chroma_expand(out, scale=1.03)
    meta.update({
        "ddcolor_weight_mean": float(w.mean()),
        "ddcolor_weight_p95": float(np.percentile(w, 95)),
    })
    return out, meta, w


def _poly_features(bgr: np.ndarray) -> np.ndarray:
    x = bgr.astype(np.float32) / 255.0
    b, g, r = x[:, 0], x[:, 1], x[:, 2]
    return np.column_stack([
        np.ones(len(x), np.float32),
        b, g, r,
        b * g, b * r, g * r,
        b * b, g * g, r * r,
    ])


def _identity_beta() -> np.ndarray:
    beta = np.zeros((10, 3), np.float32)
    beta[1, 0] = 1.0
    beta[2, 1] = 1.0
    beta[3, 2] = 1.0
    return beta


def _fit_weighted_poly_lut(original: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    h, w = original.shape[:2]
    scale = min(1.0, 520.0 / max(h, w))
    size = (int(w * scale), int(h * scale))
    src_small = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
    tgt_small = cv2.resize(target, size, interpolation=cv2.INTER_AREA)
    wei_small = cv2.resize(weight, size, interpolation=cv2.INTER_AREA).reshape(-1)

    src = src_small.reshape(-1, 3)
    tgt = tgt_small.reshape(-1, 3).astype(np.float32) / 255.0
    x = _poly_features(src)
    if len(src) > 35000:
        idx = np.linspace(0, len(src) - 1, 35000).astype(np.int64)
        x = x[idx]
        tgt = tgt[idx]
        wei_small = wei_small[idx]

    weights = np.clip(0.12 + wei_small * 1.35, 0.08, 1.0)
    grid = np.linspace(0, 255, 7, dtype=np.float32)
    bb, gg, rr = np.meshgrid(grid, grid, grid, indexing="ij")
    anchors = np.column_stack([bb.ravel(), gg.ravel(), rr.ravel()])
    x_anchor = _poly_features(anchors)
    y_anchor = anchors / 255.0
    w_anchor = np.full(len(anchors), 0.20, np.float32)

    x_all = np.vstack([x, x_anchor])
    y_all = np.vstack([tgt, y_anchor])
    w_all = np.concatenate([weights, w_anchor])
    sw = np.sqrt(w_all)[:, None]
    xw = x_all * sw
    yw = y_all * sw
    beta_id = _identity_beta()
    reg = np.diag([12.0, 18.0, 18.0, 18.0, 32.0, 32.0, 32.0, 36.0, 36.0, 36.0]).astype(np.float32)
    lhs = xw.T @ xw + reg
    rhs = xw.T @ yw + reg @ beta_id
    return np.linalg.solve(lhs, rhs).astype(np.float32)


def _apply_poly_lut(original: np.ndarray, beta: np.ndarray) -> np.ndarray:
    h, w = original.shape[:2]
    x = _poly_features(original.reshape(-1, 3))
    out = x @ beta
    return np.clip(out.reshape(h, w, 3) * 255.0, 0, 255).astype(np.uint8)


def _ddcolor_learned_lut(
    original: np.ndarray,
    ai: np.ndarray,
    ai_weight_scale: float,
    final_vibrance: float,
):
    fused, fusion_meta, weight = _ddcolor_chroma_fusion(original, ai, ai_weight_scale)
    beta = _fit_weighted_poly_lut(original, fused, weight)
    lut_out = _apply_poly_lut(original, beta)

    lab_fused = _lab_float(fused)
    lab_lut = _lab_float(lut_out)
    lab_lut[:, :, 0] = lab_fused[:, :, 0]
    out = cv2.cvtColor(np.clip(lab_lut, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    lab_out = _lab_float(out)
    lab_f = _lab_float(fused)
    mix = 0.35 + 0.35 * weight
    lab_out[:, :, 1] = lab_out[:, :, 1] * (1.0 - mix) + lab_f[:, :, 1] * mix
    lab_out[:, :, 2] = lab_out[:, :, 2] * (1.0 - mix) + lab_f[:, :, 2] * mix
    out = cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = _vibrance_lab(out, strength=final_vibrance)

    meta = dict(fusion_meta)
    meta["poly_beta"] = beta.round(6).tolist()
    return out, meta, weight


def _blend_lab(original: np.ndarray, restored: np.ndarray, strength: float) -> np.ndarray:
    if abs(strength - 1.0) < 1e-6:
        return restored
    lab_o = _lab_float(original)
    lab_r = _lab_float(restored)
    lab = lab_o + (lab_r - lab_o) * float(strength)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
