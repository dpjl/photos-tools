"""download_models.py — Télécharge tous les modèles nécessaires à l'application.

Usage :
    python download_models.py [--check]

Options :
    --check     Vérifie uniquement la présence des modèles, sans télécharger.

Les modèles volumineux téléchargés automatiquement à la première utilisation
(LaMa big-lama.pt, DDColor via HuggingFace) sont optionnellement pré-téléchargés
ici pour faciliter le déploiement hors-ligne.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import hashlib
import shutil

# ── Répertoires ───────────────────────────────────────────────────────────────

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

MODELS_DIR       = os.path.join(BASE_DIR, "models")
REALESRGAN_DIR   = os.path.join(MODELS_DIR, "realesrgan")
GFPGAN_WEIGHTS   = os.path.join(APP_DIR, "gfpgan", "weights")
LAMA_CACHE_DIR   = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")

# ── Catalogue des modèles ─────────────────────────────────────────────────────
# Chaque entrée : (dest_path, url, taille_approx_Mo)

MODELS = [
    # ── GFPGAN principal ──────────────────────────────────────────────────────
    (
        os.path.join(BASE_DIR, "GFPGANv1.4.pth"),
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        348,
    ),
    # ── Détection / parsing visages (pour GFPGANer + RedEye) ─────────────────
    (
        os.path.join(GFPGAN_WEIGHTS, "detection_Resnet50_Final.pth"),
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        109,
    ),
    (
        os.path.join(GFPGAN_WEIGHTS, "parsing_parsenet.pth"),
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        85,
    ),
    # ── SCUNet ────────────────────────────────────────────────────────────────
    (
        os.path.join(MODELS_DIR, "network_scunet.py"),
        "https://raw.githubusercontent.com/cszn/SCUNet/main/models/network_scunet.py",
        0,  # ~22 Ko
    ),
    (
        os.path.join(MODELS_DIR, "scunet_color_real_gan.pth"),
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth",
        69,
    ),
    (
        os.path.join(MODELS_DIR, "scunet_color_real_psnr.pth"),
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
        69,
    ),
    # ── Real-ESRGAN upscale ─────────────────────────────────────────────────
    (
        os.path.join(REALESRGAN_DIR, "RealESRGAN_x2plus.pth"),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        64,
    ),
    (
        os.path.join(REALESRGAN_DIR, "RealESRGAN_x4plus.pth"),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        64,
    ),
    # ── MediaPipe face landmarker ─────────────────────────────────────────────
    (
        os.path.join(APP_DIR, "face_landmarker.task"),
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        4,
    ),
    # ── LaMa inpainting ───────────────────────────────────────────────────────
    (
        os.path.join(LAMA_CACHE_DIR, "big-lama.pt"),
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        207,
    ),
]

# ── DDColor est géré séparément via HuggingFace hub ───────────────────────────
DDCOLOR_REPO = "piddnad/ddcolor_modelscope"
DDCOLOR_CACHE_DIR = os.path.join(MODELS_DIR, "ddcolor_modelscope")


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _mb(path: str) -> str:
    """Taille d'un fichier en Mo."""
    try:
        size = os.path.getsize(path)
        return f"{size / 1024 / 1024:.1f} Mo"
    except OSError:
        return "?"


def _download(dest: str, url: str, approx_mb: int) -> None:
    """Télécharge url vers dest avec une barre de progression simple."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    label = os.path.basename(dest)
    approx = f" (~{approx_mb} Mo)" if approx_mb > 0 else ""
    print(f"  Téléchargement : {label}{approx}")
    print(f"    depuis : {url}")

    try:
        def _reporthook(count: int, block_size: int, total_size: int) -> None:
            downloaded = count * block_size
            if total_size > 0:
                pct = min(downloaded * 100 // total_size, 100)
                bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                print(f"\r    [{bar}] {pct:3d}%  {downloaded / 1024 / 1024:.1f} Mo", end="", flush=True)
            else:
                print(f"\r    {downloaded / 1024 / 1024:.1f} Mo téléchargés…", end="", flush=True)

        urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
        print()  # saut de ligne après la barre
        shutil.move(tmp, dest)
        print(f"  ✔  {label}  ({_mb(dest)})\n")
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Échec du téléchargement de {label} : {exc}") from exc


def _download_ddcolor() -> None:
    """Télécharge DDColor via huggingface_hub.snapshot_download."""
    snapshot_path = os.path.join(DDCOLOR_CACHE_DIR, "models--piddnad--ddcolor_modelscope",
                                 "snapshots")
    # Chercher un snapshot existant avec pytorch_model.bin
    if os.path.isdir(snapshot_path):
        for rev in os.listdir(snapshot_path):
            model_file = os.path.join(snapshot_path, rev, "pytorch_model.bin")
            if os.path.isfile(model_file) and os.path.getsize(model_file) > 1_000_000:
                print(f"  ✔  DDColor (piddnad/ddcolor_modelscope) — déjà en cache\n")
                return

    print(f"  Téléchargement : DDColor (piddnad/ddcolor_modelscope, ~870 Mo)")
    print(f"    depuis : HuggingFace hub")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=DDCOLOR_REPO,
            cache_dir=DDCOLOR_CACHE_DIR,
            ignore_patterns=["*.msgpack", "flax_model.*", "tf_model.*", "*.h5"],
        )
        print(f"  ✔  DDColor téléchargé\n")
    except ImportError:
        print("  ⚠  huggingface_hub non disponible — DDColor sera téléchargé au 1er lancement\n")
    except Exception as exc:
        print(f"  ⚠  DDColor non téléchargé ({exc}) — sera téléchargé au 1er lancement\n")


# ── Fonctions principales ─────────────────────────────────────────────────────

def check_models() -> list[tuple[str, bool, str]]:
    """Retourne la liste (nom, présent, info) pour chaque modèle."""
    results = []
    for dest, url, _ in MODELS:
        name = os.path.basename(dest)
        present = os.path.isfile(dest)
        info = _mb(dest) if present else "MANQUANT"
        results.append((name, present, info))

    # DDColor
    snapshot_path = os.path.join(DDCOLOR_CACHE_DIR, "models--piddnad--ddcolor_modelscope",
                                 "snapshots")
    ddcolor_ok = False
    if os.path.isdir(snapshot_path):
        for rev in os.listdir(snapshot_path):
            model_file = os.path.join(snapshot_path, rev, "pytorch_model.bin")
            if os.path.isfile(model_file) and os.path.getsize(model_file) > 1_000_000:
                ddcolor_ok = True
                break
    results.append(("DDColor (pytorch_model.bin)", ddcolor_ok, "en cache" if ddcolor_ok else "MANQUANT"))

    # LaMa (déjà dans MODELS mais noter l'emplacement spécial)
    return results


def download_all(skip_existing: bool = True) -> None:
    """Télécharge tous les modèles manquants."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REALESRGAN_DIR, exist_ok=True)
    os.makedirs(GFPGAN_WEIGHTS, exist_ok=True)
    os.makedirs(LAMA_CACHE_DIR, exist_ok=True)

    errors: list[str] = []

    for dest, url, approx_mb in MODELS:
        name = os.path.basename(dest)
        if skip_existing and os.path.isfile(dest):
            print(f"  ✔  {name} — déjà présent ({_mb(dest)})")
            continue
        try:
            _download(dest, url, approx_mb)
        except RuntimeError as exc:
            print(f"  ✗  ERREUR : {exc}")
            errors.append(str(exc))

    _download_ddcolor()

    if errors:
        print("\n── Erreurs ──────────────────────────────────────────────")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("── Tous les modèles sont prêts ──────────────────────────")


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge les modèles IA.")
    parser.add_argument("--check", action="store_true",
                        help="Vérifie uniquement sans télécharger")
    parser.add_argument("--force", action="store_true",
                        help="Retélécharge même les fichiers déjà présents")
    args = parser.parse_args()

    print("═" * 60)
    print("  Restauration Photo — Vérification des modèles")
    print("═" * 60)
    print(f"  Répertoire racine  : {BASE_DIR}")
    print(f"  Répertoire modèles : {MODELS_DIR}")
    print()

    if args.check:
        results = check_models()
        all_ok = True
        for name, present, info in results:
            status = "✔" if present else "✗"
            print(f"  [{status}] {name:<45}  {info}")
            if not present:
                all_ok = False
        print()
        if all_ok:
            print("  Tous les modèles sont présents.")
        else:
            print("  Des modèles sont manquants. Lancez sans --check pour télécharger.")
            sys.exit(1)
    else:
        download_all(skip_existing=not args.force)
