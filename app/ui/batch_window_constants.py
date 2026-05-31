"""ui/batch_window_constants.py — Constantes partagées entre BatchWindow et ses mixins."""

# Étapes dont les paramètres déclenchent un aperçu rapide
_FAST_PREVIEW_IDS = frozenset({"color", "facehighlight", "ddcolor_lut", "autocolor", "wb",
                               "lightleak", "rembg"})

# Indices d'onglets nommés
_TAB_PREVIEW  = 0
_TAB_MASK     = 1
_TAB_WB       = 2
_TAB_REDEYE   = 3
_TAB_ORIGIN   = 4
_TAB_RESULT   = 5

# Onglets où l'aperçu rapide est calculé à chaque changement de paramètre
_PREVIEW_TABS = frozenset({_TAB_PREVIEW, _TAB_MASK, _TAB_WB, _TAB_REDEYE})
