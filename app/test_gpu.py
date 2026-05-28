#!/usr/bin/env python3
"""test_gpu.py — Vérifie que les étapes IA utilisent bien le GPU CUDA.

Usage :
    .venv/bin/python test_gpu.py

Teste :
  1. Disponibilité CUDA
  2. SCUNet — chargement + inférence sur CUDA
  3. GFPGAN — chargement + device détecté sur CUDA
"""

import os
import sys
import traceback

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)

import numpy as np
import torch

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


print("\n=== Test GPU / CUDA ===\n")

# ── 1. CUDA disponible ─────────────────────────────────────────────────────
cuda_ok = check(
    f"CUDA disponible — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
    lambda: (
        None if torch.cuda.is_available()
        else (_ for _ in ()).throw(RuntimeError("torch.cuda.is_available() == False"))
    ),
)

# ── 2. SCUNet sur CUDA ─────────────────────────────────────────────────────
print()

MODELS_DIR = os.path.join(BASE_DIR, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

SCUNet_cls = check("from network_scunet import SCUNet", lambda: getattr(__import__("network_scunet"), "SCUNet"))

def _load_scunet():
    model_path = os.path.join(MODELS_DIR, "scunet_color_real_gan.pth")
    model = SCUNet_cls(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    return model, device

model_and_device = check("SCUNet chargé sur cuda", _load_scunet) if SCUNet_cls else None

def _scunet_inference_cuda():
    if model_and_device is None:
        raise RuntimeError("Modèle non chargé")
    model, device = model_and_device
    assert str(device) == "cuda", f"device inattendu : {device}"
    # Vérifier que les paramètres sont bien sur CUDA
    p = next(model.parameters())
    assert p.is_cuda, f"Paramètres du modèle pas sur CUDA : {p.device}"
    # Inférence
    np.random.seed(42)
    img_bgr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    assert t.is_cuda, "Tenseur d'entrée pas sur CUDA"
    with torch.no_grad():
        out = model(t)
    assert out.is_cuda, "Tenseur de sortie pas sur CUDA"
    result = out.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    result = (result[:, :, ::-1] * 255).round().astype(np.uint8)
    assert result.shape == (256, 256, 3)
    return result

check("SCUNet inférence 256×256 sur CUDA (tenseurs in/out vérifiés)", _scunet_inference_cuda)

# ── 3. GFPGAN détecte CUDA ─────────────────────────────────────────────────
print()

def _gfpgan_device():
    from config import GFPGAN_MODEL_PATH
    if not os.path.isfile(GFPGAN_MODEL_PATH):
        raise FileNotFoundError(f"GFPGAN model not found: {GFPGAN_MODEL_PATH}")
    # Patch basicsr registry
    try:
        from basicsr.utils.registry import Registry
        orig = Registry._do_register
        def _patched(self, name, obj, suffix=None):
            key = f"{name}_{suffix}" if suffix else name
            if key in self._obj_map:
                return
            orig(self, name, obj, suffix)
        Registry._do_register = _patched
    except Exception:
        pass
    from gfpgan import GFPGANer
    restorer = GFPGANer(
        model_path=GFPGAN_MODEL_PATH,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    assert str(restorer.device) == "cuda", f"GFPGAN device: {restorer.device} (attendu: cuda)"
    return restorer.device

check("GFPGAN device == cuda", _gfpgan_device)

def _gfpgan_inference():
    from config import GFPGAN_MODEL_PATH
    try:
        from basicsr.utils.registry import Registry
        orig = Registry._do_register
        def _patched(self, name, obj, suffix=None):
            key = f"{name}_{suffix}" if suffix else name
            if key in self._obj_map:
                return
            orig(self, name, obj, suffix)
        Registry._do_register = _patched
    except Exception:
        pass
    from gfpgan import GFPGANer
    restorer = GFPGANer(
        model_path=GFPGAN_MODEL_PATH,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    # Image de test avec un visage synthétique (simple bruit)
    np.random.seed(0)
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    _, _, result = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    # Sans vrai visage, result peut être None — c'est acceptable
    return "OK (aucun visage détecté)" if result is None else f"OK (sortie {result.shape})"

check("GFPGAN enhance() sur image synthétique", _gfpgan_inference)

# ── Résumé ──────────────────────────────────────────────────────────────────
print()
if errors == 0:
    print(f"=== {_PASS}  Tous les tests GPU sont OK ===\n")
else:
    print(f"=== {_FAIL}  {errors} test(s) échoué(s) ===\n")
    sys.exit(1)
