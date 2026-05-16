"""config.py — Chemins des modèles IA partagés entre tous les modules.

Les modèles restent dans le répertoire parent (d:\Plustek Photo\colorisation\)
et sont référencés par chemin relatif depuis app/.
"""
import os

# Répertoire de l'application (app/)
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
# Répertoire parent contenant les modèles
BASE_DIR = os.path.dirname(APP_DIR)

GFPGAN_MODEL_PATH = os.path.join(BASE_DIR, "GFPGANv1.4.pth")
SCUNET_MODELS_DIR = os.path.join(BASE_DIR, "models")
