"""steps/step_gfpgan.py — Étape 2 : restauration des visages avec GFPGAN v1.4."""

from __future__ import annotations
import os
import sys
import cv2
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
    name       = "Restauration visages"
    short_name = "GFPGAN"
    slow       = True

    param_defs = [
        {"key": "weight", "label": "Poids GFPGAN (0=source, 1=plein effet)", "type": "float",
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
        weight = float(params.get("weight", 0.5))

        # Note : GFPGANv1Clean ignore silencieusement le paramètre `weight`
        # passé à enhance() (il est absorbé par **kwargs sans effet).
        # On passe donc weight=1.0 au modèle et on applique le mélange manuellement.
        _, _, result = self._restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=1.0,
        )
        if result is None:
            return img, {"face_bboxes": []}

        # Mélange manuel : weight=1 → sortie GFPGAN pure, weight=0 → image d'origine
        if weight < 1.0 and result.shape == img.shape:
            result = cv2.addWeighted(result, weight, img, 1.0 - weight, 0)

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
