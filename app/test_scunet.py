#!/usr/bin/env python3
"""test_scunet.py — Vérifie que SCUNet se charge et s'exécute correctement.

Usage :
    .venv\Scripts\python.exe test_scunet.py [gan|psnr]

Teste :
  1. Présence des fichiers modèles (network_scunet.py + .pth)
  2. Import de einops (dépendance de network_scunet.py)
  3. Chargement du modèle avec la bonne configuration
  4. Inférence sur une image synthétique 256×256
  5. Validation forme / dtype / plage des valeurs de sortie
"""

import os
import sys
import traceback

# ── Chemins ─────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

mode = sys.argv[1] if len(sys.argv) > 1 else "gan"
if mode not in ("gan", "psnr"):
    print(f"Usage: python test_scunet.py [gan|psnr]  (reçu: {mode!r})")
    sys.exit(2)

NET_PATH   = os.path.join(MODELS_DIR, "network_scunet.py")
MODEL_PATH = os.path.join(MODELS_DIR, f"scunet_color_real_{mode}.pth")

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"

errors = 0

def check(label: str, fn):
    global errors
    try:
        result = fn()
        print(f"  {_PASS}  {label}")
        return result
    except Exception as exc:
        print(f"  {_FAIL}  {label}")
        print(f"         {exc}")
        traceback.print_exc()
        errors += 1
        return None


print(f"\n=== Test SCUNet mode={mode!r} ===\n")

# 1. Fichiers présents
check(
    f"network_scunet.py existe ({NET_PATH})",
    lambda: (
        None if os.path.isfile(NET_PATH)
        else (_ for _ in ()).throw(FileNotFoundError(NET_PATH))
    ),
)
check(
    f"scunet_color_real_{mode}.pth existe",
    lambda: (
        None if os.path.isfile(MODEL_PATH)
        else (_ for _ in ()).throw(FileNotFoundError(MODEL_PATH))
    ),
)

# 2. einops
check("import einops", lambda: __import__("einops"))

# 3. Import torch + numpy
import numpy as np
check("import torch", lambda: __import__("torch"))
import torch

# 4. Import network_scunet.SCUNet
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

SCUNet_cls = check(
    "from network_scunet import SCUNet",
    lambda: getattr(__import__("network_scunet"), "SCUNet"),
)

# 5. Instanciation avec la bonne config
model = check(
    "SCUNet(in_nc=3, config=[4,4,4,4,4,4,4], dim=64) — doit correspondre aux poids",
    lambda: SCUNet_cls(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64) if SCUNet_cls else None,
)

# 6. Chargement des poids
if model is not None:
    check(
        f"load_state_dict(strict=True) depuis {os.path.basename(MODEL_PATH)}",
        lambda: model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu"), strict=True
        ),
    )
    check("model.eval()", lambda: model.eval())

# 7. Inférence sur image synthétique 256×256
def _run_inference():
    if model is None:
        raise RuntimeError("Modèle non chargé (voir erreurs précédentes)")
    model.eval()
    # Image de test RGB float32 dans [0, 1]
    np.random.seed(42)
    img_bgr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    result = (out_np[:, :, ::-1] * 255).round().astype(np.uint8)
    assert result.shape == (256, 256, 3), f"Forme inattendue : {result.shape}"
    assert result.dtype == np.uint8,      f"dtype inattendu : {result.dtype}"
    assert result.min() >= 0,            f"valeur < 0 : {result.min()}"
    assert result.max() <= 255,          f"valeur > 255 : {result.max()}"
    return result

check("Inférence 256×256 + validation sortie", _run_inference)

# ── Résumé ───────────────────────────────────────────────────────────────────
print()
if errors == 0:
    print(f"=== {_PASS}  Tous les tests sont OK — SCUNet ({mode}) fonctionne correctement ===\n")
else:
    print(f"=== {_FAIL}  {errors} test(s) échoué(s) — voir détails ci-dessus ===\n")
    sys.exit(1)
