#!/usr/bin/env python3
"""
=============================================================================
Restauration des couleurs et des visages -- photo ancienne (1985)
=============================================================================
Photo : 1985-0043.jpg -- dominante orange typique du vieillissement des
couches cyan des films couleur Kodak annees 80.

PIPELINE :
  Etape 1 -- Correction couleur (commune a toutes les versions)
      a. Etirement d'histogramme par canal  -> supprime la dominante orange
      b. CLAHE                              -> contraste local adaptatif
      c. Boost saturation en espace LAB     -> ravive les couleurs recuperees

  Etape 2a -- Restauration des visages avec GFPGAN v1.4
      GFPGAN (Tencent, CVPR 2021) utilise les priors d'un GAN StyleGAN2
      pre-entraine pour reconstruire les details du visage.

  Etape 2b -- Restauration des visages avec CodeFormer
      CodeFormer (sczhou, NeurIPS 2022) utilise un VQGAN + Transformer,
      avec un parametre de fidelite (0=max restauration, 1=max fidelite).
      Plus recent que GFPGAN, souvent superieur sur les photos tres degradees.

      Applique AVANT le debruitage : le modele voit les details originaux,
      le denoise apres nettoie les eventuels artefacts du modele.

  Etape 3 -- Debruitage leger (apres restauration visage)
      Non-Local Means (NLM, h=4/5) -- grain reduit sans effacer les details.

SORTIES :
  01_couleur.jpg                       -> Correction couleur seule
  02_couleur_denoise.jpg               -> Correction couleur + debruitage leger
  03_couleur_visages.jpg               -> Correction couleur + GFPGAN (sans denoise)
  04_couleur_visages_denoise.jpg       -> Correction couleur + GFPGAN + denoise leger
  05_couleur_codeformer.jpg            -> Correction couleur + CodeFormer (sans denoise)
  06_couleur_codeformer_denoise.jpg    -> Correction couleur + CodeFormer + denoise leger
  comparaison.jpg                      -> Grille des 7 versions cote a cote

INSTALLATION :
  pip install opencv-python numpy
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install gfpgan basicsr facexlib
=============================================================================
"""

import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
INPUT_PATH        = r"d:\Plustek Photo\colorisation\1985-0043.jpg"
OUTPUT_DIR        = r"d:\Plustek Photo\colorisation"
GFPGAN_MODEL_PATH = os.path.join(OUTPUT_DIR, "GFPGANv1.4.pth")
BASICSR_SRC       = r"C:\Temp\BasicSR\BasicSR-master"


def _setup_ai_env():
    """
    Prepare l'environnement pour que GFPGAN et CodeFormer coexistent.
    Patche ARCH_REGISTRY pour etre idempotent (signature suffix=None).
    """
    try:
        from basicsr.utils.registry import Registry
        _orig = Registry._do_register

        def _idempotent(self, name, obj, suffix=None):
            if name not in self._obj_map:
                _orig(self, name, obj, suffix)

        Registry._do_register = _idempotent
    except Exception as e:
        print(f"  [init] Patch Registry echoue : {e}")


# ===========================================================================
# CORRECTION COULEUR TRADITIONNELLE
# ===========================================================================

def stretch_histogram_per_channel(img_bgr, low_pct=1, high_pct=99, roi_inset=0.06):
    """
    Etirement d'histogramme par canal -- equivalent 'Auto Levels'.
    Chaque canal R/G/B est etire vers [0-255] entre ses percentiles.
    Les percentiles sont calcules sur la zone centrale (roi_inset) pour eviter
    que les bords du film scanne ne biaisent le resultat.
    """
    img_f = img_bgr.astype(np.float32)
    h, w  = img_f.shape[:2]

    iy1, iy2 = int(h * roi_inset), int(h * (1 - roi_inset))
    ix1, ix2 = int(w * roi_inset), int(w * (1 - roi_inset))
    roi = img_f[iy1:iy2, ix1:ix2]

    result = np.zeros_like(img_f)
    for i in range(3):
        p_low  = np.percentile(roi[:, :, i], low_pct)
        p_high = np.percentile(roi[:, :, i], high_pct)
        if p_high > p_low:
            result[:, :, i] = np.clip(
                (img_f[:, :, i] - p_low) / (p_high - p_low) * 255.0, 0, 255
            )
        else:
            result[:, :, i] = img_f[:, :, i]
    return result.astype(np.uint8)


def apply_clahe(img_bgr, clip_limit=2.5, tile_size=(8, 8)):
    """CLAHE sur le canal L en espace LAB -- contraste local sans sur-saturation."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def boost_saturation_lab(img_bgr, factor=1.35):
    """
    Boost saturation en espace LAB.
    Amplifie les canaux a et b autour de 128 (valeur neutre LAB uint8 OpenCV).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    a_boost = np.clip(128 + (a - 128) * factor, 0, 255)
    b_boost = np.clip(128 + (b - 128) * factor, 0, 255)
    lab_boost = cv2.merge([l, a_boost, b_boost]).astype(np.uint8)
    return cv2.cvtColor(lab_boost, cv2.COLOR_LAB2BGR)


def color_correction(img_bgr):
    """
    Pipeline de correction couleur :
      1. Etirement histogramme (suppression dominante orange)
      2. CLAHE (contraste local)
      3. Boost saturation
    """
    corrected  = stretch_histogram_per_channel(img_bgr, low_pct=1, high_pct=99)
    contrasted = apply_clahe(corrected, clip_limit=2.5)
    enhanced   = boost_saturation_lab(contrasted, factor=1.35)
    return enhanced


# ===========================================================================
# DEBRUITAGE LEGER
# ===========================================================================

def denoise_light(img_bgr, h_lum=4, h_color=5):
    """
    Debruitage doux Non-Local Means (parametres reduits h=4/5).
    Conserve mieux les details fins que la version precedente (h=7/9).
    """
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None,
        h=h_lum, hColor=h_color,
        templateWindowSize=7,
        searchWindowSize=21
    )


# ===========================================================================
# RESTAURATION DES VISAGES -- GFPGAN v1.4
# ===========================================================================

def download_gfpgan_model(model_path):
    """Telecharge GFPGANv1.4.pth si absent."""
    if os.path.exists(model_path):
        return True
    url = ("https://github.com/TencentARC/GFPGAN/releases/download"
           "/v1.3.0/GFPGANv1.4.pth")
    print(f"  -> Telechargement du modele GFPGANv1.4 (~333 MB)...")
    try:
        import urllib.request
        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(count * block_size * 100 / total_size))
                print(f"\r     {pct}%", end="", flush=True)
        urllib.request.urlretrieve(url, model_path, reporthook=_progress)
        print()
        return True
    except Exception as e:
        print(f"\n  Telechargement echoue : {e}")
        return False


def apply_face_restoration(img_bgr, upscale=1):
    """
    Restauration des visages avec GFPGAN v1.4.

    GFPGAN detecte automatiquement tous les visages dans l'image complete,
    les restaure avec les priors StyleGAN2 puis les reintegre dans l'image
    (paste_back=True).

    upscale=1 : conserve la resolution originale.
    """
    try:
        from gfpgan import GFPGANer
    except ImportError:
        print("  GFPGAN non installe. pip install gfpgan basicsr facexlib")
        return None

    if not download_gfpgan_model(GFPGAN_MODEL_PATH):
        return None

    try:
        restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        print(f"  -> Detection et restauration des visages (upscale={upscale})...")
        _, _, restored_img = restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        return restored_img
    except Exception as e:
        print(f"  Erreur GFPGAN : {e}")
        import traceback
        traceback.print_exc()
        return None


# ===========================================================================
# RESTAURATION DES VISAGES -- CodeFormer (NeurIPS 2022)
# ===========================================================================

def apply_face_restoration_codeformer(img_bgr, fidelity_weight=0.5, upscale=1):
    """
    Restauration des visages avec CodeFormer.

    CodeFormer (sczhou, NeurIPS 2022) utilise un code-book VQGAN + Transformer
    avec un parametre de fidelite :
      fidelity_weight=0  -> restoration maximale (moins fidele au visage original)
      fidelity_weight=1  -> fidelite maximale (conserve mieux l'identite)
      fidelity_weight=0.5 -> equilibre (recommande)

    upscale=1 : conserve la resolution originale.
    """
    # Importer CodeFormer AVANT tout ajout BasicSR au sys.path
    # (codeformer bundle sa propre basicsr, pas de conflit)
    try:
        from codeformer import CodeFormer
    except ImportError:
        print("  CodeFormer non installe. pip install codeformer")
        return None

    try:
        print(f"  -> Initialisation CodeFormer "
              f"(fidelite={fidelity_weight}, upscale={upscale})...")
        restorer = CodeFormer(
            fidelity_weight=fidelity_weight,
            upscale=upscale,
            has_aligned=False,
            only_center_face=False,
            bg_enhance=False,
        )
        print("  -> Detection et restauration des visages...")
        # upscale_image attend du RGB (cv2.COLOR_RGB2BGR en interne)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = restorer.upscale_image(img_rgb)

        # upscale_image retourne bytes (JPEG encode) ou ndarray selon version
        if isinstance(result, (bytes, bytearray)):
            arr = np.frombuffer(result, dtype=np.uint8)
            result_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            result_img = result

        if result_img is None:
            print("  Erreur CodeFormer : image decodee est None")
            return None
        return result_img
    except Exception as e:
        print(f"  Erreur CodeFormer : {e}")
        import traceback
        traceback.print_exc()
        return None


# ===========================================================================
# GRILLE DE COMPARAISON
# ===========================================================================

def make_comparison_grid(images_dict, max_total_width=2500):
    """Grille de comparaison horizontale avec etiquettes."""
    n = len(images_dict)
    h0, w0 = next(iter(images_dict.values())).shape[:2]
    col_w = max_total_width // n
    col_h = int(h0 * col_w / w0)
    label_h = 44

    cols = []
    for label, img in images_dict.items():
        resized = cv2.resize(img, (col_w, col_h), interpolation=cv2.INTER_LANCZOS4)
        header  = np.full((label_h, col_w, 3), 40, dtype=np.uint8)
        cv2.putText(header, label, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cols.append(np.vstack([header, resized]))

    return np.hstack(cols)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  RESTAURATION COULEURS + VISAGES -- PHOTO 1985")
    print("=" * 62)
    print(f"\nImage : {INPUT_PATH}")

    # Prepare BasicSR + patch registry (avant tout import AI)
    _setup_ai_env()

    img_original = cv2.imread(INPUT_PATH)
    if img_original is None:
        print(f"Impossible de charger : {INPUT_PATH}")
        sys.exit(1)

    h, w = img_original.shape[:2]
    print(f"Dimensions : {w} x {h} px\n")

    results = {"Original": img_original}

    # ------------------------------------------------------------------
    # Correction couleur de base (commune)
    # ------------------------------------------------------------------
    print("[COULEUR] Etirement histogramme + CLAHE + saturation...")
    color_base = color_correction(img_original)
    print("  OK\n")

    # ------------------------------------------------------------------
    # 01 : couleur seule
    # ------------------------------------------------------------------
    p = os.path.join(OUTPUT_DIR, "1985-0043_01_couleur.jpg")
    cv2.imwrite(p, color_base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[01] {p}")
    results["01 Couleur"] = color_base

    # ------------------------------------------------------------------
    # 02 : couleur + debruitage leger
    # ------------------------------------------------------------------
    print("[02] Debruitage leger (NLM h=4/5)...")
    color_denoised = denoise_light(color_base)
    p = os.path.join(OUTPUT_DIR, "1985-0043_02_couleur_denoise.jpg")
    cv2.imwrite(p, color_denoised, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"     {p}\n")
    results["02 Couleur+Denoise"] = color_denoised

    # ------------------------------------------------------------------
    # Restauration des visages CodeFormer (avant GFPGAN pour eviter
    # tout conflit d'import avec notre BasicSR personnalise)
    # ------------------------------------------------------------------
    print("[CODEFORMER] Restauration des visages avec CodeFormer...")
    cf_faces = apply_face_restoration_codeformer(color_base, fidelity_weight=0.5)

    if cf_faces is not None:
        # 05 : couleur + CodeFormer (sans denoise)
        p = os.path.join(OUTPUT_DIR, "1985-0043_05_couleur_codeformer.jpg")
        cv2.imwrite(p, cf_faces, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[05] {p}")
        results["05 Codeformer"] = cf_faces

        # 06 : couleur + CodeFormer + denoise leger
        print("[06] Denoise leger apres CodeFormer...")
        cf_faces_dn = denoise_light(cf_faces)
        p = os.path.join(OUTPUT_DIR, "1985-0043_06_couleur_codeformer_denoise.jpg")
        cv2.imwrite(p, cf_faces_dn, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"     {p}\n")
        results["06 Codeformer+DN"] = cf_faces_dn
    else:
        print("  CodeFormer indisponible -- sorties 05 et 06 ignorees.\n")

    # ------------------------------------------------------------------
    # Restauration des visages GFPGAN
    # ------------------------------------------------------------------
    print("[GFPGAN] Restauration des visages avec GFPGAN v1.4...")
    color_faces = apply_face_restoration(color_base, upscale=1)

    if color_faces is not None:
        # 03 : couleur + visages (sans denoise)
        p = os.path.join(OUTPUT_DIR, "1985-0043_03_couleur_visages.jpg")
        cv2.imwrite(p, color_faces, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[03] {p}")
        results["03 Couleur+Visages"] = color_faces

        # 04 : couleur + visages + denoise leger
        print("[04] Denoise leger apres GFPGAN...")
        color_faces_dn = denoise_light(color_faces)
        p = os.path.join(OUTPUT_DIR, "1985-0043_04_couleur_visages_denoise.jpg")
        cv2.imwrite(p, color_faces_dn, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"     {p}\n")
        results["04 Visages+Denoise"] = color_faces_dn
    else:
        print("  GFPGAN indisponible -- sorties 03 et 04 ignorees.\n")

    # ------------------------------------------------------------------
    # Grille
    # ------------------------------------------------------------------
    print("[GRILLE] Generation...")
    grid = make_comparison_grid(results)
    p = os.path.join(OUTPUT_DIR, "1985-0043_comparaison.jpg")
    cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  {p}\n")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    print("=" * 62)
    print("  SORTIES")
    print("=" * 62)
    print("  01_couleur.jpg                   -- Couleur seule (reference)")
    print("  02_couleur_denoise.jpg           -- Couleur + denoise leger")
    print("  03_couleur_visages.jpg           -- Couleur + GFPGAN v1.4")
    print("  04_couleur_visages_denoise.jpg   -- Couleur + GFPGAN + denoise")
    print("  05_couleur_codeformer.jpg        -- Couleur + CodeFormer")
    print("  06_couleur_codeformer_denoise.jpg -- Couleur + CodeFormer + denoise")
    print("  comparaison.jpg                 -- Grille cote a cote")
    print()
    print("  ORDRE : couleur -> restauration visages -> denoise leger")
    if color_faces is None:
        print()
        print("  Pour activer GFPGAN :")
        print("    pip install gfpgan basicsr facexlib")
