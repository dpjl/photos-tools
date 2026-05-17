"""steps/__init__.py — Liste canonique des étapes du pipeline."""

from steps.step_color          import ColorStep
from steps.step_wb             import WhiteBalanceStep
from steps.step_autocolor      import AutoColorStep
from steps.step_facehighlight  import FaceHighlightStep
from steps.step_inpaint        import InpaintStep
from steps.step_redeye         import RedEyeStep
from steps.step_gfpgan         import GFPGANStep
from steps.step_scunet         import SCUNetStep
from steps.step_rembg          import RembgStep

# Instances singleton — partagées dans toute l'application
ALL_STEPS = [
    ColorStep(),           # 1 · Correction couleur (manuelle)  — désactivée par défaut
    WhiteBalanceStep(),    # 2 · Balance des blancs (pipette)    — désactivée par défaut
    AutoColorStep(),       # 3 · Auto niveaux & couleurs         — activée par défaut
    FaceHighlightStep(),   # 4 · Correction hautes lumières      — désactivée par défaut
    InpaintStep(),         # 5 · Retouche inpainting (LaMa)      — désactivée par défaut
    RedEyeStep(),          # 6 · Correction yeux rouges
    GFPGANStep(),          # 7 · Restauration visages (GFPGAN)
    SCUNetStep(),          # 8 · Embellissement SCUNet
    RembgStep(),           # 8 · Cast argentique (rembg)
]

__all__ = [
    "ALL_STEPS",
    "ColorStep", "AutoColorStep", "FaceHighlightStep",
    "RedEyeStep", "GFPGANStep", "SCUNetStep", "RembgStep",
]
