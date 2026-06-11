"""steps/step_genref.py — Couleur par référence IA (FLUX Kontext → LUT).

L'étape applique une LUT polynomiale (30 coefficients) apprise d'une
« référence » générée par FLUX.1 Kontext sur la même image (voir
core/genref.py). La génération (lourde, ~2-3 min GPU) se fait UNIQUEMENT
via le bouton « Générer la référence IA… » du panneau (dialogue dédié,
même logique que la pipette WB ou l'éditeur de masque) ; le pipeline et le
preview ne font qu'appliquer la LUT en cache — instantané.

Si aucune référence n'existe pour (image, style, graine), l'étape laisse
l'image inchangée et le signale dans les extras.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from steps.base import StepBase
from core import genref


class GenRefStep(StepBase):
    id                 = "genref"
    name               = "Couleur par référence IA (FLUX)"
    short_name         = "RefIA"
    slow               = False
    enabled_by_default = False
    previewable        = True     # l'application de la LUT est instantanée
    has_genref_dialog  = True     # bouton « Générer la référence IA… »

    param_defs = [
        {"key": "style",  "label": "Style de prompt", "type": "choice",
         "as_buttons": True, "default": "court",
         "choices": list(genref.PROMPT_STYLES)},
        {"key": "graine", "label": "Graine", "type": "int",
         "default": 42, "min": 0, "max": 9999, "step": 1},
        {"key": "force", "label": "Force", "type": "float",
         "default": 1.0, "min": 0.0, "max": 1.3, "step": 0.05},
        {"key": "saturation", "label": "Saturation", "type": "float",
         "default": 1.0, "min": 0.5, "max": 1.5, "step": 0.05},
    ]

    def __init__(self) -> None:
        # Petit cache mémoire des entrées chargées du disque (betas only)
        self._entries: OrderedDict[tuple[str, str, int], genref.GenRefEntry] = \
            OrderedDict()
        # Mode batch : référence injectée par la fenêtre batch (versions
        # sidecar). Quand présente, elle remplace le cache par digest ; la
        # LUT est ré-ajustée à la volée sur l'image d'entrée courante
        # (fit ~0,3 s, mis en cache par (digest, tag)).
        self._batch_ref: np.ndarray | None = None
        self._batch_tag: str = ""
        self._fit_cache: OrderedDict[tuple[str, str], tuple[np.ndarray, float]] = \
            OrderedDict()

    # ── API pour le dialogue (mode simple) ───────────────────────────────

    def invalidate_cache(self) -> None:
        """À appeler après une génération : force la relecture du disque."""
        self._entries.clear()

    # ── API pour la fenêtre batch (état d'instance, comme les masques) ──

    def set_batch_ref(self, ref_bgr: np.ndarray, tag: str) -> None:
        """Active une référence batch ; `tag` identifie la version (cache)."""
        self._batch_ref = ref_bgr
        self._batch_tag = tag

    def clear_batch_ref(self) -> None:
        self._batch_ref = None
        self._batch_tag = ""

    def _fit_for_batch(self, img: np.ndarray, digest: str) -> tuple[np.ndarray, float]:
        key = (digest, self._batch_tag)
        cached = self._fit_cache.get(key)
        if cached is not None:
            self._fit_cache.move_to_end(key)
            return cached
        beta, conf = genref.fit_lut(img, self._batch_ref)
        self._fit_cache[key] = (beta, conf)
        while len(self._fit_cache) > 6:
            self._fit_cache.popitem(last=False)
        return beta, conf

    def _get_entry(self, digest: str, style: str, seed: int):
        key = (digest, style, seed)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry
        entry = genref.load_entry(digest, style, seed, with_ref=False)
        if entry is not None and entry.beta is not None:
            self._entries[key] = entry
            while len(self._entries) > 8:
                self._entries.popitem(last=False)
        return entry

    # ── Pipeline ─────────────────────────────────────────────────────────

    def process(self, img: np.ndarray, params: dict, context: dict):
        d = self.default_params()
        style = str(params.get("style", d["style"]))
        seed = int(params.get("graine", d["graine"]))
        force = float(params.get("force", d["force"]))
        saturation = float(params.get("saturation", d["saturation"]))

        digest = genref.image_digest(img)

        # Mode batch : référence injectée → LUT ré-ajustée sur l'entrée
        # courante (réapplicable sur n'importe quelle base).
        if self._batch_ref is not None:
            beta, conf = self._fit_for_batch(img, digest)
            out = genref.apply_lut(img, beta, force=force,
                                   saturation=saturation)
            return out, {
                "genref": {
                    "status": "ok", "mode": "batch",
                    "version": self._batch_tag,
                    "force": force, "saturation": saturation,
                    "conf_flot": round(conf, 3),
                }
            }

        entry = self._get_entry(digest, style, seed)

        if entry is None or entry.beta is None:
            return img.copy(), {
                "genref": {
                    "status": "non générée",
                    "message": "Référence absente — utiliser « Générer la "
                               "référence IA… » dans le panneau de l'étape.",
                    "style": style, "graine": seed,
                }
            }

        out = genref.apply_lut(img, entry.beta, force=force,
                               saturation=saturation)
        return out, {
            "genref": {
                "status": "ok",
                "style": style, "graine": seed,
                "force": force, "saturation": saturation,
                "cast": entry.cast,
                "items": list(entry.items),
                "prompt": entry.prompt,
                "conf_flot": round(entry.conf_mean, 3),
            }
        }
