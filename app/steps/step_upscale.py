"""steps/step_upscale.py — Upscale IA via Real-ESRGAN.

Real-ESRGAN est utilise en lazy-loading : modele et dependances ne sont charges
qu'au premier passage effectif de l'etape.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from config import REALESRGAN_X2PLUS_PATH, REALESRGAN_X4PLUS_PATH
from steps.base import StepBase


_MODELS = {
    "Photo x2": {
        "path": REALESRGAN_X2PLUS_PATH,
        "scale": 2,
        "file": "RealESRGAN_x2plus.pth",
    },
    "Photo x4": {
        "path": REALESRGAN_X4PLUS_PATH,
        "scale": 4,
        "file": "RealESRGAN_x4plus.pth",
    },
}


class UpscaleStep(StepBase):
    id         = "upscale"
    name       = "Upscale IA Real-ESRGAN"
    short_name = "Upscale"
    slow       = True
    enabled_by_default = False

    param_defs = [
        {"key": "model", "label": "Modèle", "type": "choice",
         "as_buttons": True, "default": "Photo x2",
         "choices": ["Photo x2", "Photo x4"]},
        {"key": "scale", "label": "Échelle sortie", "type": "float",
         "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5},
        {"key": "tile", "label": "Tuile GPU/CPU", "type": "int",
         "default": 0, "min": 0, "max": 1024, "step": 64},
        {"key": "tile_pad", "label": "Marge tuile", "type": "int",
         "default": 10, "min": 0, "max": 64, "step": 2},
        {"key": "precision", "label": "Précision", "type": "choice",
         "default": "auto", "choices": ["auto", "fp32"]},
        {"key": "max_pixels", "label": "Limite mégapixels", "type": "int",
         "default": 80, "min": 8, "max": 300, "step": 4},
    ]

    def __init__(self) -> None:
        super().__init__()
        self._upsampler = None
        self._cache_key = None

    def process(self, img: np.ndarray, params: dict, context: dict):
        scale = float(params.get("scale", 2.0))
        if scale <= 1.0:
            return img.copy(), {"upscale_scale": 1.0}

        max_pixels = int(params.get("max_pixels", 80)) * 1_000_000
        out_pixels = int(round(img.shape[0] * scale)) * int(round(img.shape[1] * scale))
        if max_pixels > 0 and out_pixels > max_pixels:
            raise RuntimeError(
                "Upscale trop volumineux : "
                f"{out_pixels / 1_000_000:.1f} MP > limite {max_pixels / 1_000_000:.0f} MP"
            )

        model_key = str(params.get("model", "Photo x2"))
        tile = int(params.get("tile", 0))
        tile_pad = int(params.get("tile_pad", 10))
        precision = str(params.get("precision", "auto"))

        upsampler = self._load(model_key, tile, tile_pad, precision)
        out, _ = upsampler.enhance(img, outscale=scale)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.shape[2] == 4:
            out = out[:, :, :3]
        return out.astype(np.uint8, copy=False), {
            "upscale_scale": scale,
            "upscale_model": model_key,
        }

    def _load(self, model_key: str, tile: int, tile_pad: int, precision: str):
        if model_key not in _MODELS:
            model_key = "Photo x2"
        spec = _MODELS[model_key]
        model_path = spec["path"]
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Modèle Real-ESRGAN introuvable : {model_path}\n"
                "Lancez download_models.py pour télécharger les poids."
            )

        half = precision == "auto" and torch.cuda.is_available()
        cache_key = (model_key, tile, tile_pad, half)
        if self._upsampler is not None and self._cache_key == cache_key:
            return self._upsampler

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except Exception as exc:
            raise RuntimeError(
                "Dépendance Real-ESRGAN indisponible. Installez realesrgan "
                "puis relancez fix_packages.py si nécessaire."
            ) from exc

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=int(spec["scale"]),
        )
        self._upsampler = RealESRGANer(
            scale=int(spec["scale"]),
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=0,
            half=half,
            gpu_id=None,
        )
        self._cache_key = cache_key
        return self._upsampler
