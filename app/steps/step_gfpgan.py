"""steps/step_gfpgan.py — Étape 2 : restauration des visages avec GFPGAN v1.4."""

from __future__ import annotations
import os
import sys
import numpy as np

from steps.base import StepBase
from config import GFPGAN_MODEL_PATH


def _patch_basicsr_registry():
    """Rend BasicSR Registry._do_register idempotent (évite erreur re-import)."""
    try:
        from basicsr.utils.registry import Registry
        orig = Registry._do_register
        def _patched(self, name, obj, suffix=None):
            key = f"{name}_{suffix}" if suffix else name
            if key in self._obj_map:
                return
            orig(self, name, obj, suffix)
        Registry._do_register = _patched
    except Exception:
        pass


class GFPGANStep(StepBase):
    id         = "gfpgan"
    name       = "2 · Restauration visages (GFPGAN)"
    short_name = "GFPGAN"
    slow       = True

    param_defs = [
        {"key": "upscale", "label": "Upscale (×)", "type": "int",
         "default": 1, "min": 1, "max": 2, "step": 1},
        {"key": "weight",  "label": "Poids GFPGAN (0=source, 1=GFPGAN)", "type": "float",
         "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    ]

    def __init__(self):
        self._restorer = None   # chargé à la première utilisation

    def _load(self):
        if self._restorer is not None:
            return
        if not os.path.isfile(GFPGAN_MODEL_PATH):
            raise FileNotFoundError(f"Modèle GFPGAN introuvable : {GFPGAN_MODEL_PATH}")
        _patch_basicsr_registry()
        from gfpgan import GFPGANer
        self._restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

    def process(self, img: np.ndarray, params: dict, context: dict):
        self._load()
        upscale = int(params.get("upscale", 1))
        weight  = float(params.get("weight", 0.5))

        _, _, result = self._restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight,
        )
        if result is None:
            return img, {"face_bboxes": []}

        # Extraire les bboxes depuis le face_helper de GFPGAN (déjà calculées par enhance())
        face_bboxes = _extract_face_bboxes(self._restorer)
        return result, {"face_bboxes": face_bboxes}


def _extract_face_bboxes(restorer) -> list[tuple[int, int, int, int]]:
    """Extrait les boîtes des visages depuis le face_helper interne de GFPGAN.

    Plus fiable que de relancer un détecteur séparé : GFPGAN a déjà fait
    la détection lors de enhance() ; on réutilise ses résultats.
    """
    face_bboxes: list[tuple[int, int, int, int]] = []
    try:
        dets = getattr(restorer.face_helper, "face_det_results",
               getattr(restorer.face_helper, "det_faces", []))
        for det in dets:
            bbox = (det[0] if (isinstance(det, (list, tuple)) and len(det) == 2
                               and not isinstance(det[0], (int, float))) else det)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            face_bboxes.append((x1, y1, x2, y2))
    except Exception:
        pass
    return face_bboxes
