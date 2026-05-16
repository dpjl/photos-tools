"""steps/__init__.py — Liste canonique des étapes du pipeline."""

from steps.step_color          import ColorStep
from steps.step_autocolor      import AutoColorStep
from steps.step_facehighlight  import FaceHighlightStep
from steps.step_redeye         import RedEyeStep
from steps.step_gfpgan         import GFPGANStep
from steps.step_scunet         import SCUNetStep
from steps.step_rembg          import RembgStep

# Instances singleton — partagées dans toute l'application
ALL_STEPS = [
    ColorStep(),           # 1 · Correction couleur (manuelle)  — désactivée par défaut
    AutoColorStep(),       # 2 · Auto niveaux & couleurs         — activée par défaut
    FaceHighlightStep(),   # 3 · Hautes lumières visages         — désactivée par défaut
    RedEyeStep(),          # 4 · Correction yeux rouges
    GFPGANStep(),          # 5 · Restauration visages (GFPGAN)
    SCUNetStep(),          # 6 · Embellissement SCUNet
    RembgStep(),           # 7 · Cast argentique (rembg)
]

__all__ = [
    "ALL_STEPS",
    "ColorStep", "AutoColorStep", "FaceHighlightStep",
    "RedEyeStep", "GFPGANStep", "SCUNetStep", "RembgStep",
]
