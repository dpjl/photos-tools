"""steps/__init__.py — Liste canonique des étapes du pipeline."""

from steps.step_color  import ColorStep
from steps.step_gfpgan import GFPGANStep
from steps.step_scunet import SCUNetStep
from steps.step_rembg  import RembgStep

# Instances singleton — partagées dans toute l'application
ALL_STEPS = [
    ColorStep(),
    GFPGANStep(),
    SCUNetStep(),
    RembgStep(),
]

__all__ = ["ALL_STEPS", "ColorStep", "GFPGANStep", "SCUNetStep", "RembgStep"]
