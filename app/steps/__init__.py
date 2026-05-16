"""steps/__init__.py — Liste canonique des étapes du pipeline."""

from steps.step_color     import ColorStep
from steps.step_gfpgan    import GFPGANStep
from steps.step_scunet    import SCUNetStep
from steps.step_rembg     import RembgStep
from steps.step_autocolor import AutoColorStep

# Instances singleton — partagées dans toute l'application
ALL_STEPS = [
    ColorStep(),
    GFPGANStep(),
    SCUNetStep(),
    RembgStep(),
    AutoColorStep(),   # désactivée par défaut (enabled_by_default=False)
]

__all__ = [
    "ALL_STEPS",
    "ColorStep", "GFPGANStep", "SCUNetStep", "RembgStep", "AutoColorStep",
]
