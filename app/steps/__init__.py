"""steps/__init__.py — Liste canonique des étapes du pipeline."""

from steps.step_color          import ColorStep
from steps.step_wb             import WhiteBalanceStep
from steps.step_autocolor      import AutoColorStep
from steps.step_ddcolor_lut    import DDColorLUTStep
from steps.step_facehighlight  import FaceHighlightStep
from steps.step_inpaint        import InpaintStep
from steps.step_redeye         import RedEyeStep
from steps.step_gfpgan         import GFPGANStep
from steps.step_scunet         import SCUNetStep
from steps.step_rembg          import RembgStep
from steps.step_lightleak      import LightLeakStep

# Instances singleton — partagées dans toute l'application
ALL_STEPS = [
    ColorStep(),           # Correction couleur (manuelle)  — désactivée par défaut
    FaceHighlightStep(),   # Correction hautes lumières     — activée par défaut
    DDColorLUTStep(),      # Couleur IA DDColor LUT         — désactivée par défaut
    AutoColorStep(),       # Auto niveaux & couleurs        — activée par défaut
    WhiteBalanceStep(),    # Balance des blancs (pipette)   — désactivée par défaut
    InpaintStep(),         # Retouche inpainting (LaMa)     — désactivée par défaut
    RedEyeStep(),          # Correction yeux rouges         — activée par défaut
    GFPGANStep(),          # Restauration visages           — activée par défaut
    SCUNetStep(),          # Embellissement SCUNet          — activée par défaut
    LightLeakStep(),       # Lumière parasite               — désactivée par défaut
    RembgStep(),           # Cast argentique                — désactivée par défaut
]

__all__ = [
    "ALL_STEPS",
    "ColorStep", "AutoColorStep", "DDColorLUTStep", "FaceHighlightStep",
    "LightLeakStep", "RedEyeStep", "GFPGANStep", "SCUNetStep", "RembgStep",
]
