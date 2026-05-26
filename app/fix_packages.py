"""
Corrections post-installation pour les paquets avec des problèmes de compatibilité connus.
À exécuter UNE FOIS après `pip install -r requirements.txt`.

Usage:
    .venv\Scripts\python.exe fix_packages.py
"""

import sys
import site
import os


def patch_basicsr_degradations():
    """basicsr 1.4.2 importe depuis torchvision.transforms.functional_tensor
    qui a été supprimé dans torchvision >= 0.16. Le remplacement correct est
    torchvision.transforms.functional (même fonction, nouveau chemin).
    """
    old_import = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    new_import = "from torchvision.transforms.functional import rgb_to_grayscale"

    for pkg_dir in site.getsitepackages():
        degradations_path = os.path.join(pkg_dir, "basicsr", "data", "degradations.py")
        if os.path.exists(degradations_path):
            with open(degradations_path, "r", encoding="utf-8") as f:
                content = f.read()
            if old_import in content:
                patched = content.replace(old_import, new_import)
                with open(degradations_path, "w", encoding="utf-8") as f:
                    f.write(patched)
                print(f"[OK] Patché : {degradations_path}")
            else:
                print(f"[OK] Déjà corrigé : {degradations_path}")
            return
    print("[WARN] basicsr non trouvé dans site-packages — patch non appliqué.")


if __name__ == "__main__":
    print("Application des corrections post-installation...")
    patch_basicsr_degradations()
    print("Terminé.")
