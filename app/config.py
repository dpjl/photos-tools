"""config.py — Chemins des modèles IA partagés entre tous les modules.

Les modèles restent dans le répertoire parent (d:\Plustek Photo\colorisation\)
et sont référencés par chemin relatif depuis app/.
"""
import os

# Répertoire de l'application (app/)
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
# Répertoire parent contenant les modèles
BASE_DIR = os.path.dirname(APP_DIR)

# Cache HuggingFace (modèles VLM + token d'authentification) — placé sous models/
# pour rester sur le grand disque et co-localiser le token et les poids.
# Doit être défini AVANT tout import de transformers/huggingface_hub.
HF_HOME_DIR = os.path.join(BASE_DIR, "models", "hf_cache")
os.environ.setdefault("HF_HOME", HF_HOME_DIR)

GFPGAN_MODEL_PATH        = os.path.join(BASE_DIR, "GFPGANv1.4.pth")
SCUNET_MODELS_DIR        = os.path.join(BASE_DIR, "models")
MEDIAPIPE_LANDMARKER_PATH = os.path.join(APP_DIR,  "face_landmarker.task")
REALESRGAN_MODELS_DIR    = os.path.join(SCUNET_MODELS_DIR, "realesrgan")
REALESRGAN_X2PLUS_PATH   = os.path.join(REALESRGAN_MODELS_DIR, "RealESRGAN_x2plus.pth")
REALESRGAN_X4PLUS_PATH   = os.path.join(REALESRGAN_MODELS_DIR, "RealESRGAN_x4plus.pth")

# DDColor modelscope (couleur IA)
DDCOLOR_MODEL_REPO = "piddnad/ddcolor_modelscope"
DDCOLOR_CACHE_DIR  = os.path.join(SCUNET_MODELS_DIR, "ddcolor_modelscope")

# Détection automatique d'artefacts (rayures / plis) — Microsoft BOPBTL
BOPBTL_DETECTION_DIR     = os.path.join(SCUNET_MODELS_DIR, "bopbtl_detection")
BOPBTL_DETECTION_WEIGHTS = os.path.join(BOPBTL_DETECTION_DIR, "FT_Epoch_latest.pt")
BOPBTL_DETECTION_URL     = (
    "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/"
    "resolve/main/Global/checkpoints/detection/FT_Epoch_latest.pt"
)
