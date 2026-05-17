"""core/lama_inpaint.py — Moteur d'inpainting LaMa.

Port minimal et autonome de lama-cleaner (ipshadowshighlights n'est pas requis).
Aucune dependance sur le package lama-cleaner / iopaint.

Le modele big-lama.pt est lu depuis le cache torch hub standard :
  ~/.cache/torch/hub/checkpoints/big-lama.pt

Il est telecharge automatiquement a la premiere utilisation si absent.
"""

from __future__ import annotations

import os
import numpy as np
import cv2
import torch

# URL et chemin local du modele
_LAMA_URL  = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
_LAMA_PATH = os.path.join(_CACHE_DIR, "big-lama.pt")

# Parametres de la strategie crop (identiques aux defauts de lama-cleaner)
_CROP_TRIGGER = 800   # si max(H,W) > cette valeur, utiliser le crop strategy
_CROP_MARGIN  = 128   # marge contextuelle en pixels autour de chaque region masquee
_PAD_MOD      = 8     # LaMa exige des dimensions multiples de 8


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _ceil_mod(x: int, mod: int) -> int:
    return x if x % mod == 0 else (x // mod + 1) * mod


def _pad_to_mod(arr: np.ndarray) -> np.ndarray:
    """Padde arr pour que H et W soient multiples de _PAD_MOD."""
    h, w = arr.shape[:2]
    ph = _ceil_mod(h, _PAD_MOD) - h
    pw = _ceil_mod(w, _PAD_MOD) - w
    if arr.ndim == 3:
        return np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
    return np.pad(arr, ((0, ph), (0, pw)), mode="symmetric")


def _norm(arr: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW float32 [0, 1]."""
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0


def _boxes_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Bounding boxes de chaque region masquee (255)."""
    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    result = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        box = np.array([x, y, x + bw, y + bh])
        box[::2]  = np.clip(box[::2],  0, w)
        box[1::2] = np.clip(box[1::2], 0, h)
        result.append(box)
    return result


# ── Inpainter ──────────────────────────────────────────────────────────────────

class LamaInpainter:
    """Inpainting LaMa — singleton avec chargement paresseux du modele."""

    _instance: LamaInpainter | None = None

    @classmethod
    def get(cls) -> LamaInpainter:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._model: torch.jit.ScriptModule | None = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not os.path.exists(_LAMA_PATH):
            from torch.hub import download_url_to_file
            os.makedirs(_CACHE_DIR, exist_ok=True)
            download_url_to_file(_LAMA_URL, _LAMA_PATH, progress=True)
        self._model = torch.jit.load(_LAMA_PATH, map_location="cpu").eval()

    @torch.no_grad()
    def _forward_once(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Passe directe sur un crop.

        image_rgb : H x W x 3  uint8  RGB
        mask      : H x W      uint8  (255 = zone a remplir)
        Retourne  : H x W x 3  uint8  BGR
        """
        h, w = image_rgb.shape[:2]
        t_img  = torch.from_numpy(_norm(_pad_to_mod(image_rgb))).unsqueeze(0)
        t_mask = torch.from_numpy(_norm(_pad_to_mod(mask))).unsqueeze(0)
        t_mask = (t_mask > 0).float()

        out = self._model(t_img, t_mask)
        res = out[0].permute(1, 2, 0).numpy()
        res = np.clip(res * 255, 0, 255).astype(np.uint8)[:h, :w]
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpainting complet avec strategie crop sur les grandes images.

        image_bgr : H x W x 3  uint8  BGR
        mask      : H x W      uint8  (255 = zone a remplir)
        Retourne  : H x W x 3  uint8  BGR

        Pour les images dont le cote le plus long depasse _CROP_TRIGGER pixels,
        chaque region masquee est traitee separement avec une marge contextuelle
        (identique a la strategie « crop » de lama-cleaner).
        """
        self._ensure_loaded()
        h, w   = image_bgr.shape[:2]
        rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = image_bgr.copy()

        if max(h, w) > _CROP_TRIGGER:
            # --- Strategie crop ---
            for box in _boxes_from_mask(mask):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cw = (x2 - x1) + _CROP_MARGIN * 2
                ch = (y2 - y1) + _CROP_MARGIN * 2
                cl = max(cx - cw // 2, 0)
                cr = min(cx + cw // 2, w)
                ct = max(cy - ch // 2, 0)
                cb = min(cy + ch // 2, h)
                crop_m = mask[ct:cb, cl:cr]
                if crop_m.max() == 0:
                    continue
                crop_bgr = self._forward_once(rgb[ct:cb, cl:cr], crop_m)
                m = crop_m[:, :, np.newaxis].astype(np.float32) / 255.0
                orig = image_bgr[ct:cb, cl:cr].astype(np.float32)
                result[ct:cb, cl:cr] = (crop_bgr * m + orig * (1.0 - m)).astype(np.uint8)
        else:
            # --- Image entiere ---
            res_bgr = self._forward_once(rgb, mask)
            m       = mask[:, :, np.newaxis].astype(np.float32) / 255.0
            result  = (res_bgr * m + image_bgr.astype(np.float32) * (1.0 - m)).astype(np.uint8)

        return result

    def unload(self) -> None:
        """Libere le modele de la memoire."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
