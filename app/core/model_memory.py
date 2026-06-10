"""core/model_memory.py — Lecture de la VRAM et déchargement global des modèles.

Permet de « faire le ménage » avant une opération gourmande (ex. charger un VLM) :
décharge tous les modèles mis en cache par l'application (SCUNet, GFPGAN,
Real-ESRGAN, DDColor, LaMa, RedEye, détecteur de rayures, VLM…).  Chaque modèle
sera rechargé paresseusement à la prochaine utilisation.
"""

from __future__ import annotations

import gc
import subprocess


# Attributs d'instance qui contiennent un modèle chargé (sur les étapes du pipeline)
_MODEL_ATTRS = (
    "_model", "_restorer", "_upsampler", "_face_helper", "_landmarker",
    "_net", "_session",
)


def gpu_memory() -> "tuple[int, int] | None":
    """Mémoire GPU (utilisée, totale) en Mo via nvidia-smi, ou None si indisponible.

    Utilise nvidia-smi (et non torch) pour refléter l'usage de TOUTE la carte
    sans initialiser de contexte CUDA dans ce processus.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        used, total = (int(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return used, total
    except Exception:
        return None


def unload_all_models(steps=None) -> list[str]:
    """Décharge tous les modèles connus. Retourne la liste de ce qui a été libéré."""
    freed: list[str] = []

    # Étapes du pipeline (objets passés par l'appelant)
    for step in (steps or []):
        for attr in _MODEL_ATTRS:
            if getattr(step, attr, None) is not None:
                setattr(step, attr, None)
                freed.append(f"{type(step).__name__}{attr}")

    # Singletons cœur
    for import_path, cls_name, label in (
        ("core.lama_inpaint", "LamaInpainter", "LaMa"),
        ("core.artifact_detect", "ScratchDetector", "détecteur rayures"),
        ("core.vlm_refine", "VLMRefiner", "VLM"),
    ):
        try:
            mod = __import__(import_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            inst = getattr(cls, "_instance", None)
            if inst is not None and hasattr(inst, "unload"):
                inst.unload()
                freed.append(label)
        except Exception:
            pass

    # Session rembg partagée
    try:
        import core.rembg_session as rs
        if getattr(rs, "_shared_session", None) is not None:
            rs._shared_session = None
            freed.append("rembg")
    except Exception:
        pass

    gc.collect()
    try:
        import sys
        torch = sys.modules.get("torch")   # ne pas importer torch s'il ne l'est pas déjà
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return freed
