"""steps/__init__.py — Liste canonique des étapes du pipeline."""

from steps.step_color     import ColorStep
from steps.step_redeye    import RedEyeStep
from steps.step_gfpgan    import GFPGANStep
from steps.step_scunet    import SCUNetStep
from steps.step_rembg     import RembgStep
from steps.step_autocolor import AutoColorStep

# Instances singleton — partagées dans toute l'application
ALL_STEPS = [
    ColorStep(),
    RedEyeStep(),      # 2 · Correction yeux rouges
    GFPGANStep(),      # 3 · Restauration visages (GFPGAN)
    SCUNetStep(),      # 4 · Embellissement SCUNet
    RembgStep(),       # 5 · Cast argentique (rembg)
    AutoColorStep(),   # 6 · Auto niveaux & couleurs (désactivée par défaut)
]

__all__ = [
    "ALL_STEPS",
    "ColorStep", "RedEyeStep", "GFPGANStep", "SCUNetStep", "RembgStep", "AutoColorStep",
]
