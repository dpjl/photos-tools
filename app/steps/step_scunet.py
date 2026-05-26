"""steps/step_scunet.py — Étape 3 : débruitage/embellissement SCUNet.

Inclut la protection des visages GFPGAN : les visages détectés à l'étape
précédente sont recomposés depuis l'image GFPGAN sur le résultat SCUNet.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import cv2
import torch

from steps.base import StepBase
from config import SCUNET_MODELS_DIR


class SCUNetStep(StepBase):
    id         = "scunet"
    name       = "Embellissement SCUNet"
    short_name = "SCUNet"
    slow       = True

    param_defs = [
        {"key": "mode", "label": "Mode SCUNet", "type": "choice",
         "default": "gan", "choices": ["gan", "psnr"]},
        {"key": "protect_faces", "label": "Protéger les visages GFPGAN", "type": "bool",
         "default": True},
        {"key": "expand", "label": "Expansion masque visage (%)", "type": "float",
         "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05},
    ]

    def __init__(self):
        self._model = None
        self._model_mode = None
        self._device: "torch.device | None" = None

    def _load(self, mode: str):
        if self._model is not None and self._model_mode == mode:
            return
        model_path = os.path.join(SCUNET_MODELS_DIR, f"scunet_color_real_{mode}.pth")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modèle SCUNet introuvable : {model_path}")

        # network_scunet.py se trouve dans le même dossier que les modèles
        if SCUNET_MODELS_DIR not in sys.path:
            sys.path.insert(0, SCUNET_MODELS_DIR)

        from network_scunet import SCUNet
        # config=[4,4,4,4,4,4,4] correspond aux poids officiels scunet_color_real_*.pth
        # (config=[2,2,2,2,2,2,2] est le défaut de SCUNet() mais ne correspond pas aux poids)
        model = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
        # Charger les poids en CPU d'abord pour éviter tout problème de device
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Compatibilité avec anciennes versions torch sans argument weights_only
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._device = _best_device()
        model.to(self._device)
        self._model = model
        self._model_mode = mode

    def process(self, img: np.ndarray, params: dict, context: dict):
        mode          = params.get("mode", "gan")
        protect_faces = bool(params.get("protect_faces", True))
        expand        = float(params.get("expand", 0.4))
        self._load(mode)

        result = _run_scunet(self._model, img, self._device)

        # Protection des visages : recomposer depuis GFPGAN si activé
        if protect_faces:
            face_bboxes = context.get("face_bboxes", [])
            if face_bboxes:
                result = _blend_faces(img, result, face_bboxes, expand=expand)

        return result, {}


# ──────────────────────────────────────────────────────────────────────────────
# Sélection du device
# ──────────────────────────────────────────────────────────────────────────────

def _best_device() -> "torch.device":
    """Retourne le meilleur device disponible pour SCUNet : CUDA > CPU.

    Note: DirectML (Intel/AMD iGPU via torch-directml) et OpenVINO GPU ne sont
    PAS utilisés ici. L'architecture Swin Transformer de SCUNet génère des tenseurs
    7D et des opérations ScatterND/Einsum que ces backends ne supportent pas
    (crash 0xC0000005 sur DML, erreur bfuwzyx sur OpenVINO GPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Fonctions de traitement internes
# ──────────────────────────────────────────────────────────────────────────────

def _run_scunet(model, img: np.ndarray, device: "torch.device") -> np.ndarray:
    """Exécute SCUNet sur l'image (uint8 BGR) et retourne uint8 BGR.

    Le modèle attend du RGB ; on convertit en entrée et en sortie.
    """
    # BGR → RGB (le modèle a été entraîné sur des images RGB)
    img_rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)  # 1×C×H×W
    with torch.no_grad():
        out = model(t)
    out_rgb = out.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    # RGB → BGR
    return (out_rgb[:, :, ::-1] * 255).round().astype(np.uint8)


def _blend_faces(
    src: np.ndarray,
    dst: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    expand: float = 0.4,
) -> np.ndarray:
    """Recompose les visages (src) sur dst avec un masque gaussien doux."""
    result = dst.copy().astype(np.float32)
    h, w = dst.shape[:2]
    src_f = src.astype(np.float32)

    for (x0, y0, x1, y1) in bboxes:
        fw = x1 - x0
        fh = y1 - y0
        pad_x = int(fw * expand)
        pad_y = int(fh * expand)
        rx0 = max(0, x0 - pad_x)
        ry0 = max(0, y0 - pad_y)
        rx1 = min(w, x1 + pad_x)
        ry1 = min(h, y1 + pad_y)

        rw = rx1 - rx0
        rh = ry1 - ry0
        if rw <= 0 or rh <= 0:
            continue

        # Masque gaussien
        mask = np.zeros((rh, rw), dtype=np.float32)
        cx, cy = rw // 2, rh // 2
        sx, sy = rw / 3, rh / 3
        yy, xx = np.mgrid[:rh, :rw]
        mask = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        mask = np.clip(mask, 0, 1)[:, :, np.newaxis]

        result[ry0:ry1, rx0:rx1] = (
            src_f[ry0:ry1, rx0:rx1] * mask
            + result[ry0:ry1, rx0:rx1] * (1 - mask)
        )

    return result.astype(np.uint8)
