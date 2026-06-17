"""ui/batch_window_constants.py — Constantes partagées entre BatchWindow et ses mixins."""

# Étapes dont les paramètres déclenchent un aperçu rapide
_FAST_PREVIEW_IDS = frozenset({"color", "facehighlight", "ddcolor_lut", "genref",
                               "autocolor", "wb", "lightleak", "rembg", "redzone",
                               "crop"})

# Indices d'onglets nommés
_TAB_PREVIEW  = 0
_TAB_MASK     = 1
_TAB_REDZONE  = 2
_TAB_WB       = 3
_TAB_REDEYE   = 4
_TAB_REDLIPS  = 5
_TAB_CROP     = 6
_TAB_GENREF   = 7
_TAB_ORIGIN   = 8
_TAB_RESULT   = 9

# Onglets où l'aperçu rapide est calculé à chaque changement de paramètre
_PREVIEW_TABS = frozenset({_TAB_PREVIEW, _TAB_MASK, _TAB_REDZONE, _TAB_WB,
                           _TAB_REDEYE, _TAB_REDLIPS, _TAB_CROP})
