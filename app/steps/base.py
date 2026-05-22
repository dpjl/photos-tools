"""steps/base.py — Classe de base pour toutes les étapes du pipeline."""

from __future__ import annotations
import numpy as np


class StepBase:
    """Interface commune pour une étape de traitement d'image.

    Attributs de classe à surcharger :
        id         : identifiant unique (ex. "color", "gfpgan")
        name       : libellé affiché dans l'UI (ex. "1 · Correction couleur")
        short_name : libellé court pour les vignettes
        slow       : True si l'étape est longue (→ afficher un spinner)
        param_defs : liste de définitions de paramètres (voir format ci-dessous)

    Format d'un param_def :
        {
            "key":     str,           # clé dans le dict params
            "label":   str,           # libellé UI
            "type":    "float"|"int"|"choice",
            "default": any,
            "min":     number,        # float/int uniquement
            "max":     number,
            "step":    number,
            "choices": list[str],     # choice uniquement
        }
    """

    id:                str       = ""
    name:              str       = ""
    short_name:        str       = ""
    slow:              bool      = False
    param_defs:        list[dict] = []
    enabled_by_default: bool     = True   # False → étape décochée au démarrage
    has_overlay:       bool      = False  # True → checkbox overlay sans recalcul
    has_mask_editor:   bool      = False  # True → bouton « Peindre le masque »
    has_color_picker:  bool      = False  # True → bouton « Sélectionner point blanc »
    previewable:       bool      = False  # True → aperçu temps réel à chaque changement de param
    profile_param_key: str       = ""     # clé du param qui pilote les presets de profil
    param_presets:     dict      = {}     # {valeur: {param_key: val, ...}}

    def process(
        self,
        img:     np.ndarray,
        params:  dict,
        context: dict,
    ) -> tuple[np.ndarray, dict]:
        """Traite l'image et retourne (résultat, extras).

        extras : dict de valeurs ajoutées au contexte partagé (ex. face_bboxes).
        """
        raise NotImplementedError(f"{self.__class__.__name__}.process() non implémenté")

    def default_params(self) -> dict:
        """Retourne un dict {key: default} depuis param_defs."""
        return {p["key"]: p["default"] for p in self.param_defs}
